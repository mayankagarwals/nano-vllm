import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int, # vocab size
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))  # [Dout, Din]. Accessed as model.model.embed_tokens.weight
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        '''
        in load_modle the key would have been model.embed_tokens.weight
        '''
        param_data = param.data
        shard_size = param_data.size(0) # vocab 
        start_idx = self.tp_rank * shard_size
        '''
        Rank 0: narrow(0, 0, 2000) → rows 0–1999
        Rank 1: narrow(0, 2000, 2000) → rows 2000–3999
        Rank 2: narrow(0, 4000, 2000) → …
        Rank 3: narrow(0, 6000, 2000) → …

         for each rank:

Disk → CPU: f.get_tensor(...) loads the full weight into CPU RAM.
Slice on CPU: custom loaders like VocabParallelEmbedding.weight_loader do narrow(...).
CPU → HBM: param.data.copy_(...) copies only the shard into the rank’s GPU parameter.

        Note the difference in global vocab token and stored index. Index starts from 0 for even the chunk (2000, 3999). we correct that in forward

        '''
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size) # narrow takes view along one dimension (dimension, first row to keep in full checkpoitn, how many rows to take )
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor): # shape: [num_tokens], dtype: torch.int64

        '''
        Example: x = [500, 2500]
        
        Assume 
        vocab_size = 4000
        tp_size = 2

        Each rank stores half the embedding table:

        Rank 0: global tokens 0–1999
        Rank 1: global tokens 2000–3999


        Rank 0 
        vocab_start_idx = 0 
        vocab_end_idx = 2000
        mask = [True, False]
        x =  [1, 0] * ([500, 2500] - 0) = [500, 0]
        y = [rank0.weight[500], rank0.weight[0]]
        the 0 index can seem scary here but we will mask it out 

        Rank 1
        vocab_start_idx = 2000
        vocab_end_idx = 4000
        mask = [False, True]
        x =  [0, 1] * ([500, 2500] - 2000) = [0, 500]
        y = [rank0.weight[0], rank0.weight[500]]

        All_reduce -> elementwise sum across ranks. You get the right [num_tokens, hidden_dim]
        '''


        '''
        In decode, we have one last token per sequence
        In prefill we will have uncached prompt tokens across all seqs
        '''

        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # x - vocab_start_idx: map global token ID → local row in this rank’s weight matrix
            x = mask * (x - self.vocab_start_idx)
        '''
        roughly same as y = self.weight[x]
        Why use it instead of nn.Embedding?
        In this case mainly for communicating explicitly that this param is embedding table.
        But F.embedding has more embedding specific params

        '''
        y = F.embedding(x, self.weight)  # [num_tokens, hidden_dim]
        if self.tp_size > 1:
            '''
            mask has shape [num_tokens]. unsqueeze(1) makes it [num_tokens, 1], allowing it to broadcast across each embedding dimension.

            '''
            y = mask.unsqueeze(1) * y
            '''
            dist.all_reduce(y) propagates the result to every participating rank. Each rank’s y tensor is overwritten in place with the summed result.
            By contrast, dist.reduce(y, dst=0) stores the reduced result only on rank 0.
            '''
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor): # [num_tokens, hidden_dim]
        context = get_context()
        if context.is_prefill: # for prefill we will have [total_tokens, hidden_dim]. but we only want last token per seq. so we get to [num_sequences, hidden_dim].
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight) # per rank [num_tokens, local_vocab_size]
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None

            '''
            dist.gather(logits, all_logits, dst=0) collects each tensor-parallel rank’s local logits onto rank 0.
            '''
            dist.gather(logits, all_logits, 0) 
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None # [num_tokens, global_voab_size]
        return logits # None if any other rank. 0 rank has [num_tokens, global_voab_size]
