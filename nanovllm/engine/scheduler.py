from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        '''
        we want to do prefill in front of decode . Debatable and often a more nuanced implementation in production 
        '''


        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill - gather all prefill 
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0] # take first waiting request
            remaining = self.max_num_batched_tokens - num_batched_tokens # is there room to be prefilled.
            if remaining == 0:
                break
            # get num_tokens based on sequence at hand. Remember we are evaluating from prefilll lens right now 
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                # head of line blocking. If first request can't be allocated, break. this is fifo fairness. First request should be served at all cost.
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # already found something to prefill? let's not worry about something that isn't fitting
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            # just allocate blocks if new request.
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)  
            # we have all information on scheduling now. Partial or full prefill scheduled. If it's partial this 
            # will be the only partial one. at next partial we will break. Else will add as many full we can 
            # partial can only be first request. because if it is partial by definition there is not
            # budget left for a full or partial prefill
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            # update hash for all blocks between [num_cached_tokens and num_scheduled_tokens)
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # chunked prefill. no action, we dont add token 
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
