
import logging
import torch
import os.path as osp

from mmcv.utils import get_logger as get_root_logger
from termcolor import colored
from omegaconf import OmegaConf

logger_name = None
_global_wandb = None

def initialize_wandb(cfg):
    global _global_wandb
    if _global_wandb is None and cfg.wandb:
        import wandb
        _global_wandb = wandb
        _global_wandb.init(
            project=cfg.project,
            name=cfg.wandb_name,
            dir=cfg.output,
        )
        print("Wandb initialized successfully")
    elif _global_wandb is None:
        print("Wandb initialization skipped because cfg.wandb is False")
    else:
        print("Wandb was already initialized")

def get_wandb():
    global _global_wandb
    if _global_wandb is None:
        return None
    return _global_wandb
 
def get_logger(cfg=None, log_level=logging.INFO):
    global logger_name
    if cfg is None:
        return get_root_logger(logger_name)

    # creating logger
    name = cfg.wandb_name 
    output = cfg.output
    logger_name = name

    logger = get_root_logger(name, osp.join(output, 'log.txt'), log_level=log_level, file_mode='a')

    fmt = '[%(asctime)s %(name)s]  %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') \
        + colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))

        if isinstance(handler, logging.FileHandler):
            handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))

    return logger

def setup_logging(filename, resume=False):
    root_logger = logging.getLogger()

    ch = logging.StreamHandler()
    fh = logging.FileHandler(filename=filename, mode='a' if resume else 'w')

    root_logger.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    root_logger.addHandler(ch)
    root_logger.addHandler(fh)

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item()**norm_type
    total_norm = total_norm**(1. / norm_type)
    return total_norm

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", fp=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.fp = fp

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        msg = ' '.join(entries)
        # print(msg, flush=True)
        if self.fp is not None:
            self.fp.write(msg+'\n')
        return msg

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

