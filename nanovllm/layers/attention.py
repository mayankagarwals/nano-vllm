import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

'''
key, value   : [N, 8, 128]                      # this pass's newly computed K/V, one layer
k_cache      : [num_blocks, 256, 8, 128]        # view into the global cache for this layer
slot_mapping : [N]                              # int32, one destination slot per token. This is offset in the global kv cache. 
# slot=physical_block_id×block_size+offset_within_block

'''
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([]) # allocated as [num_blocks, block_size, num_kv_heads, head_dim] in model_runner.prepare_prefill

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        '''
        q: [num_tokens, num_heads, head_dim]
        k: [num_tokens, num_kv_heads, head_dim]
        v: [num_tokens, num_kv_heads, head_dim]
        '''
        context = get_context()
        '''
        k_cache / v_cache are views into one global paged KV cache:
            [num_blocks, block_size, num_kv_heads, head_dim]  e.g. [num_blocks, 256, 8, 128]
        Blocks are handed out from a free pool, so a sequence's KV is logically contiguous
        but physically scattered. context.block_tables is the page table between the two.
        '''
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            '''
            block_tables[i][j] = physical block id holding logical tokens
                [j*block_size, (j+1)*block_size) of sequence i; rows right-padded with -1.
            Shape [batch, max_blocks_per_seq]. Only built when some sequence has a cached
            prefix, i.e. cu_seqlens_k[-1] > cu_seqlens_q[-1] (see model_runner.prepare_prefill).
            '''
            if context.block_tables is not None:    # prefix cache
                # The cached prefix lives only in the paged cache -- the local k/v hold just
                # this pass's uncached tail (already written by store_kvcache above). Swap to
                # the full cache and let block_table do the addressing.
                k, v = k_cache, v_cache # [N, 8, 128] -> [num_blocks, 256, 8, 128]
            '''
            You can call flash_attn_varlen_func two ways: 

            without kv cache: k and v are 3d: [num_tokens, num_kv_heads, head_dim] 
            block_table is None 

            with kv cache: k and v are 4d: [num_blocks, block_size, num_kv_heads, head_dim]
            block_table[i][j] → block, then row within block

How it looks like when block table is passed. 

function attention(q, k_cache, v_cache,
                   cu_seqlens_q, cu_seqlens_k,
                   max_seqlen_q, max_seqlen_k,
                   block_table, scale):

    out = empty_like(q)                          # [total_q, H_q, D]
    num_seqs = len(cu_seqlens_q) - 1

    # max_seqlen_q / max_seqlen_k never enter the math -- a real kernel only needs them
    # on the host to size the launch grid without reading cu_seqlens off the GPU.

    for i in 0 .. num_seqs-1:

        # ---- 1. this sequence's queries ----
        qs, qe = cu_seqlens_q[i], cu_seqlens_q[i+1]
        Q      = q[qs:qe]                        # [q_len, H_q, D]
        q_len  = qe - qs                         # <= max_seqlen_q

        # ---- 2. this sequence's keys/values ----
        k_len = cu_seqlens_k[i+1] - cu_seqlens_k[i]   # <= max_seqlen_k

        if block_table is None:                  # dense: cu_seqlens_k indexes k/v directly
            K = k[cu_seqlens_k[i] : cu_seqlens_k[i+1]]
            V = v[cu_seqlens_k[i] : cu_seqlens_k[i+1]]
        else:                                    # paged: walk the page table per token
            K, V = empty(k_len, H_kv, D), empty(k_len, H_kv, D)
            for t in 0 .. k_len-1:
                phys = block_table[i][t / PAGE_SIZE]   # which physical block. PAGE_SIZE same as block_size
                slot = t % PAGE_SIZE                   # which row inside it
                K[t], V[t] = k_cache[phys][slot], v_cache[phys][slot]

        # ---- 3. GQA: each kv head serves H_q/H_kv query heads ----
        K = repeat_each_head(K, H_q / H_kv)      # [k_len, H_q, D]
        V = repeat_each_head(V, H_q / H_kv)

        for h in 0 .. H_q-1:
            # ---- 4. scores ----
            S = (Q[:, h, :] @ transpose(K[:, h, :])) * scale   # [q_len, k_len], scale = 1/sqrt(D)

            # ---- 5. causal mask, bottom-right aligned ----
            shift = k_len - q_len                # queries are the LAST q_len positions;
            for r in 0 .. q_len-1:               # shift == 0 when there is no prefix cache
                for c in 0 .. k_len-1:
                    if c > r + shift:
                        S[r][c] = -inf

            # ---- 6. softmax + weighted sum of V ----
            P = softmax(S, axis=-1)              # [q_len, k_len]
            out[qs:qe, h, :] = P @ V[:, h, :]    # [q_len, D]

    return out

            '''
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            '''
            Same math with q_len == 1: shift = k_len - 1, so the causal mask is all-ones
            and every query token attends over its whole history.

            q: [batch_size, 1, num_heads, head_dim] after the unsqueeze
            context_lens: [batch_size] -- full KV length per sequence (plays the role of
                          cu_seqlens_k; a length, not a cumulative sum, since q isn't packed)
            block_table is always required here: the history only exists in the paged cache.
            '''
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
