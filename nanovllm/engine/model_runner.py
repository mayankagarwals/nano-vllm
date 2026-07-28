import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype() 
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size # 8//1
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        '''
        28 layers, each will have attention. 
        '''
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        '''
        This runs after the model is loaded and warmup_model() has done a dummy forward pass — so it knows how much memory weights + peak activations actually cost.

        peak: Peak PyTorch allocation so far (model + max warmup activations)
        

        budget          = total * gpu_memory_utilization     # e.g. 8GB * 0.9 = 7.2GB
        reserve_peak    = peak - current                     # activation headroom warmup revealed
        available_for_kv = budget - used - reserve_peak # why not just budget - current - reserve_peak? used tracks total VRAM the CUDA driver considers occupied.
        num_kvcache_blocks = available_for_kv // block_bytes
        '''
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        '''
        For qwen 3-0.6B 
        28 layers, 256 block size, 8 heads (head//world size), 128 
        num_kvcache_blocks is computed above.
        '''
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            '''
            This allows for 
            All layers to share the same physical block numbering — block_table=[3, 7] means physical block 3 in layer 0's cache and layer 27's cache both hold the same token positions for that sequence.
            this way per sequence you arent taking 28 different block tables one per layer
            '''
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"): # defined in attention.py
                module.k_cache = self.kv_cache[0, layer_id]   # shape [num_blocks, 256, 8, 128]. Views not copies.   wires each Attention layer to its slice of one pre-allocated KV cache
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        '''
        input_ids -> tokens for prefill prompt (non-cached portion only)
        positions -> positions in the sequence (skipping already-cached tokens)
        cu_seqlens_q -> cumulative lengths of tokens we compute this pass; reconstructs seq boundaries in input_ids/positions
        cu_seqlens_k -> cumulative lengths including cached prefix tokens (full KV length per seq)
        slot_mapping -> flat list of global KV cache slot indices, one entry per token processed in this forward pass
                       slot = block_table[logical_block] * block_size + offset_in_block
                       contiguous only within one physical block; jumps at logical block boundaries
                       (physical blocks come from a free pool, not necessarily adjacent in memory)
        block_tables -> per-seq map from logical block index to physical KV cache block id (needed when any seq has cached prefix)
                       padded to max block_table len across batch with -1

        Example (block_size=4, batch of 3 — two prefix-cached + one fresh):
          # token ids are arbitrary vocab indices — can repeat and need not increase; positions are separate
          seq0 (cached): tokens [101,4521,892,33, 12004,7,991,4521], num_cached=4, block_table=[3,7]
          seq1 (cached): tokens [55,892,12004,33, 7,4521],          num_cached=4, block_table=[5,9]
          seq2 (fresh):  tokens [991,33,892],                         num_cached=0, block_table=[2]

          input_ids    = [12004, 7, 991, 4521,  7, 4521,  991, 33, 892]  # uncached tails; note repeats (4521, 7, 892)
          positions    = [4, 5, 6, 7,           4, 5,     0,  1,  2]     # where in seq; this is what increases/resets
          cu_seqlens_q = [0, 4, 6, 9]                                       # 4 + 2 + 3 tokens computed
          cu_seqlens_k = [0, 8, 14, 17]                                     # 8 + 6 + 3 total KV length (includes cached)



          slot_mapping = [28,29,30,31,  36,37,  8,9,10]                     # each seq's tokens fit in one logical block here
                       # seq0 phys block 7 -> 28-31, seq1 phys block 9 -> 36-37, seq2 phys block 2 -> 8-10
                       # gaps (31->36, 37->8) are normal: different seqs use different physical blocks
        Slot mapping is useful to just store the keys and values we will generate at the right place in the kv cache. 

          block_tables = [[3, 7], [5, 9], [2, -1]]                      # needed: k_len (17) > q_len (9); seq2 padded to max len 2

          # if uncached tokens span a block boundary, slots jump mid-sequence too:
          # e.g. 5-token fresh seq, block_table=[3,7] -> slot_mapping = [12,13,14,15, 28]  (15->28 gap)
        '''
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end]) # what input ids are non cached in this prefill 
            positions.extend(range(start, end)) # what positions
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                # block_table[i] = the physical block ID in GPU KV cache that holds tokens [i * block_size, (i+1) * block_size) for this sequence.
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # _CONTEXT is a module-level global singleton
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = [] # what token
        positions = [] # what position of last token
        slot_mapping = [] # what physical slot of last token
        context_lens = [] # length of sequence
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs) # all block tables for sequence
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # cuda graphs only for bs <= 512. probably should be along lines of use cuda graph for the largest and use eager for rest.
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        '''
        we set global context with some variables. 
        input_ids , positions -> 
        For prefill: tokens and positions for prefill prompt apart from cached ones
        For decode: last token and position
        '''
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None # simply list of temperatures per seq as gpu tensor
        logits = self.run_model(input_ids, positions, is_prefill) #  if 2 seqences, [2, vocab_size]   # one logit row per sequence. 
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None #  if 2 seqences, [2]
        reset_context() #  _CONTEXT = Context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size) # ParallelLMHead is awkward to graph. can read why not graph vocab size but also memory needed would be a lot
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        '''
        Optional memory pool shared across graphs

        cm = torch.cuda.graph(graph, self.graph_pool)
        cm.__enter__()          # start recording CUDA ops into `graph`
        outputs[:bs] = self.model(...)
        cm.__exit__(...)        # stop recording, finalize the graph

        '''
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            '''
            # A torch.cuda.CUDAGraph() object — the empty container being filled
            Everything executed on GPU inside the with block gets recorded into it. After capture, graph.replay() re-executes that exact kernel sequence.
            '''
            graph = torch.cuda.CUDAGraph() 
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            '''
            Equivalent to below with:
            cm = torch.cuda.graph(graph, self.graph_pool)
            cm.__enter__()          # start recording CUDA ops into `graph`
            outputs[:bs] = self.model(...)
            cm.__exit__(...)        # stop recording, finalize the graph
            '''
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize() # tells the CPU to wait until all GPU work on the current CUDA device has finished.
            reset_context()

        self.graph_vars = dict( 
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
