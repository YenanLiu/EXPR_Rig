import torch
import os

import torch.distributed as dist

def reduce_tensor(tensor, world_size=None):
    if world_size is None:
        world_size = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt