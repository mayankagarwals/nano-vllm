import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    '''
        packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    '''
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v) # from model.layers.0.self_attn.q_proj.weight to model.layers.0.self_attn.qkv_proj.weight

                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else: # else runs only if the inner loop finishes without hitting break 
                    '''
                    model.get_parameter(weight_name) is a built-in PyTorch nn.Module method. It looks up a parameter by its fully qualified name (a dotted string path) and returns the nn.Parameter object.
                    model.get_parameter("model.embed_tokens.weight") -> equivalent to -> model.model.embed_tokens.weight
                    nn.Parameter is PyTorch's way of marking a tensor as a learnable model weight that belongs to an nn.Module.

                    eg: self.weight = nn.Parameter(torch.empty(8000, 4096))
                    This creates a tensor that is

                    Part of the model — registered on the module, shows up in model.parameters()
                    Trainable by default — included in optimizer updates (if you were training)
                    Saved/loaded with the model — appears in checkpoints as named weights

                    '''
                    param = model.get_parameter(weight_name)
                    # Some parameters have a custom loader (for tensor-parallel sharding):
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
