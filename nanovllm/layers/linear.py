import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int, # hidden_size: 1024
        output_size: int, # ((16+8*2)*128)//tp_size = 4096/4 = 1024
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim# 0
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x : [..., input_size]. ... is any leading dims. eg: [..., hidden_size]
        weight: [input_size, output_size]
        bias: [output_size] or None 
        '''
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int, 
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0) # essentially going to convert [num_tokens, input_dim]

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        '''
        sum(output_sizes) is not adding the outputs together mathematically. It computes the total output width of one merged weight matrix built by concatenating several linear layers along the output dimension.
        '''
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size # if tp is 4, each gpu would have stored [ gate_shard (768) | up_shard (768) ] 
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        '''
        originally in checkpoint we have gate_proj.weight, up_proj.weight each of [3072, hidden]. So when you multiply [N, hidden], you get [N, 3072]]
        we load it as gate_up_proj.weight with shard id 0 and 1 

        shard_offset: 
        output_sizes[:0] = [] → sum([]) = 0
        output_sizes[:1] = [3072] → sum([3072]) = 3072
        based on tp_size the right offsets can be 0,3072 or 0 + 3072/4 = 768, 3072 + 3072/4 = 3840, etc.
        shard_size: output_sizes[0] = 3072 → 3072/4 = 768
        that's how we rightly load the two [768, hidden] per rank. weight is (output_size, input_size)
        '''
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int, # 1024
        head_size: int, # 128
        total_num_heads: int, # 16
        total_num_kv_heads: int | None = None, # 8
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size #q,k,v
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        '''
        weight is [out_features, in_features]. in_features is always hidden_size (1024),
        but out_features is num_heads*head_size — NOT hidden_size. Qwen3 decouples head_size
        from hidden_size (head_size=128 is fixed), so 16*128 = 2048 != 1024 and q_proj is not square.
        (In classic MHA head_size = hidden/num_heads so q_proj is square — that intuition breaks here.)

        checkpoint tensors:
            q_proj.weight : [16*128, 1024] = [2048, 1024]
            k_proj.weight : [ 8*128, 1024] = [1024, 1024]   (GQA: only 8 kv heads)
            v_proj.weight : [ 8*128, 1024] = [1024, 1024]

        They are merged into one qkv_proj weight of [(16 + 2*8)*128, 1024] = [4096, 1024] globally,
        sharded on dim 0 (tp_dim=0) so each rank holds [4096/4, 1024] = [1024, 1024].

        per-rank shard sizes (tp_size=4): num_heads=4, num_kv_heads=2
            q -> 4*128 = 512  at offset 0
            k -> 2*128 = 256  at offset 512
            v -> 2*128 = 256  at offset 768      (512 + 256 + 256 = 1024 = local out_features)

        shard_offset/shard_size are already in LOCAL coordinates, so we narrow into the local
        param, then chunk the checkpoint tensor along dim 0 into tp_size pieces and take our rank's.
        '''
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size # because we store q,k,v as HXD
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size) # tensor.narrow(dim, start, length)

        '''
        chunk splits a tensor into equal pieces along one dimension; [self.tp_rank] picks this GPU's piece.
        tp_shard_size = loaded_weight.size(self.tp_dim) // self.tp_size
        start_idx = self.tp_rank * tp_shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, tp_shard_size)
        '''
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int, # 16*128
        output_size: int, # 1024
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        '''
        checkpoint tensor: [1024, 3072] = [hidden, intermediate]
        local param: divide(3072, 4) = 768 → [1024, 768], and tp_dim = 1

        '''
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim) # 768
        start_idx = self.tp_rank * shard_size  
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Multiply [N, intermediate_size/tp_size] with [intermediate_size/tp_size, hidden_size] to get [N, hidden_size]. Then all_reduce across tp_dim 0 to sum contributions from all ranks.
        '''
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
