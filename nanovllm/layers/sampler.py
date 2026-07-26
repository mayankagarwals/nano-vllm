import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        '''
        probs is a [batch, vocab_size] tensor where each row sums to 1.
        torch.empty_like(probs) -> Allocate a same-shaped tensor (uninitialized).
        .exponential_(1) -> Fill in-place with Exp(1) noise: each entry ~ Exponential(rate=1).
        .clamp_min_(1e-10) -> Avoid dividing by ~0.
        probs.div_(...) -> In-place: probs[i] = probs[i] / noise[i].
        .argmax(dim=-1) -> Pick the vocab index with the largest ratio per batch row.

        This is gpu friendly way of doing 
        torch.multinomial(probs, num_samples=1)

        Why this works
        If E_i ~ Exp(1) independently, then:
        argmax_i ( p_i / E_i )  ~  Categorical(p) because of some math

        Not greedy decoding. Greedy would be probs.argmax(-1). This line adds randomness every step.
        Temperature still matters. It was applied earlier; lower temperature → sharper probs → sampling behaves more greedily.
        '''
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
