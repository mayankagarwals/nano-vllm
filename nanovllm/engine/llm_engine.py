import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        '''
        Sequence initialized which has 

            status set to waiting
            seq_id initialized,
            num_prompt_tokens initialized (initial prompt)

            token_ids = initial prompt tokens. Updated in post process
            num_tokens initialized = len(self.token_ids). in postprocess, add to this. For the most part tracks len(token_ids). the separate counter is there for the distributed decode path, not because it tracks something conceptually different
            last_token initialized = token_ids[-1]. in postprocess, update  # last_token isn't tracking something different — it's a cached copy of token_ids[-1] that survives the intentional emptying of token_ids during decode serialization.

            num_cached_tokens,  = 0 # how many chanced in kv cache we are using
            num_scheduled_tokens, = 0 # how many scheduled

            is_prefill, = True , will we solve for prefill or decode currently
            block_table = [] , global kv cache indices for this request
        temperature, max_tokens, ignore_eos initialized from sampling params
        '''
        seq = Sequence(prompt, sampling_params)

        # add to waiting: deque[Sequence] 
        self.scheduler.add(seq)

    def step(self):
        '''
        is_prefill: one step will either be prefill or decode not both. 
        seqs: contains the sequences to be run prefill or decode on 

        First we evaluate from prefill sense 
        1. loop: as long as something in waiting + request budget exists 
        2. variables: first waiting request, remaining token budget
        3. If no remaining token budget move on
        4.1 calculate num_tokens as seq.num_tokens - cached ones (remember we are evaluating for prefill. if it was decode it would not be in waiting queue)
        4.2 if not enough free block ids, move on (some could be cached but not in free and in used. These won't be counted towards blocks needed )
        5. schedule these as partial or full prefill. If all tokens are covered, move it to running. 
        6. only one partial prefill allowed. if partial, break 

        If any prefill, return. we are prioritizing prefill over decode 

        Then we evaluate from decode lens 
        1. loop: as long as something in running + request budget exists 
        2. popleft from running. If we have block, great. if not, preempt last request in running . Update seq. Add back to running at start

        '''
        seqs, is_prefill = self.scheduler.schedule()
        # The negative sign is a sign convention to pack two pieces of information into one return value from step()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        '''
        # per seq once token 
        Decode: Each sequence feeds seq.last_token in, the model produces one logit row, the sampler picks one token, and that token is appended
        Prefill: sampled token — the model’s prediction for what comes after the last token in the current prefill chunk.
        '''
        token_ids = self.model_runner.call("run", seqs, is_prefill) 

        '''
        For each (token, sequence)
            update each block's hash between cached to scheduled, num_cached_tokens, reset num_scheduled_tokens
            if chunked prefill - return 
            add token to sequence 
            if reached max token/eos -> mark as finished + remove from running queue, deallocate blocks 
        '''

        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:

        # ignore for walk through
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        '''
        for example.py sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
        '''
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            '''
            Initialize a sequence with sensible defaults and add to scheduler waiting queue
            '''
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
