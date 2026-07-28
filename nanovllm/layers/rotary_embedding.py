from functools import lru_cache
import torch
from torch import nn


#The head dimension (128 in Qwen3) is split into 64 pairs, and each pair gets its own rotation frequency.
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))# shape (rotary_dim // 2,) . base^(-2k/d), k is each pair index 0,1,2,...
        t = torch.arange(max_position_embeddings, dtype=torch.float) # 0,1,...40960-1. shape  (max_position_embeddings,)
        freqs = torch.einsum("i,j -> ij", t, inv_freq) # equivalent to  t[:, None] * inv_freq[None, :]. freq[p,d] = p * inv_freq[d]. Row p is all angles at position p. shape (max_position_embeddings, rotary_dim // 2).
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1) # (40960, 1, 128)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions] # [num_tokens, 1, 128]
        cos, sin = cos_sin.chunk(2, dim=-1) # [num_tokens, 1, 64], [num_tokens, 1, 64]
        query = apply_rotary_emb(query, cos, sin) #  [num_tokens, 16, 128]
        key = apply_rotary_emb(key, cos, sin) #  [num_tokens, 8, 128]
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int, # 128
    rotary_dim: int, # 128
    max_position: int,  # 40960
    base: float, # 10000
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
