import torch
import numpy as np
import random
import os
from torch import optim as optim

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
    
def build_optimizer(config, model):
    """Build optimizer, set weight decay of normalization to 0 by default."""
    parameters = set_weight_decay(model, {}, {})

    opt_name = config.optimizer_name
    optimizer = None
    if opt_name == 'adamw':
        optimizer = optim.AdamW(
            parameters,
            eps=config.adamw_eps,
            betas=config.adamw_betas,
            lr=config.base_lr,
            weight_decay=config.weight_decay)
    else:
        raise ValueError(f'Unsupported optimizer: {opt_name}')
    return optimizer

def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias') or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)
    return [{'params': has_decay}, {'params': no_decay, 'weight_decay': 0.}]