import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int, # 1024
        eps: float = 1e-6, # 1e-06
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x.shape == residual.shape == [num_tokens, hidden_size]

        '''

        Equivalent math 
        residual = x + residual 
        x = rms_norm(x)
        return x, residual 

        Adds the layer output to the existing residual stream in FP32 for numerical stability:
        Saves the unnormalized sum s for the next residual connection.

        '''
        orig_dtype = x.dtype
        x = x.float().add_(residual.float()) 
        residual = x.to(orig_dtype)

        var = x.pow(2).mean(dim=-1, keepdim=True)

        # x_norm = x * (1/ (var + eps)). eps is to make sure not deviding by 0 
        # var = (mean (square))
        # x_normi = gamma_i * x_normi . where gamma_is is per hidden feature learned weight
        # this is what rms norm is 
        x.mul_(torch.rsqrt(var + self.eps))
        '''
        Q: why are we multiplying sefl.weight after converting to originail dtype . why not do that also in fp32 for numerical stability
        Squaring and summing many hidden values can accumulate substantial FP16/BF16 error, so that part uses FP32.
        Multiplying by self.weight is only an element-wise operation—no reduction or accumulated error. Doing it in the model dtype is faster and produces the dtype expected by the next layer.

        You could write
        x.mul_(self.weight.float())
        x = x.to(orig_dtype)

        Marginally more accurate for marginally better efficiency
        '''
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x.shape == residual.shape == [num_tokens, hidden_size]
        '''
        num_tokens is:

        Prefill: total uncached scheduled tokens across sequences.
        Decode: batch size—one token per sequence.
        '''
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
