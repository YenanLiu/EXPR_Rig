import os
import torch
import glob

from mmcv.runner import CheckpointLoader
from .custom_logging import get_logger
from apex import amp


def load_eval_checkpoint(config, model):
    logger = get_logger()
    logger.info(f'==============> Successfully Loading the Evaluation model from {config.eval_model_path}....................')
    checkpoint = CheckpointLoader.load_checkpoint(config.eval_model_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=True)
    logger.info(msg)

    logger.info(f"original save training epoch {checkpoint['epoch']}")
    logger.info(f"original mse {checkpoint['mse']}")


def save_checkpoint(config, epoch, model, optimizer, max_metrics, lr_scheduler, save_flag, suffix=''):
    save_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'config': config,
        'mse': max_metrics['min_mse'],
        'epoch': epoch}
        
    logger = get_logger()


    if save_flag:
        exists_pths = glob.glob(config.output + "/*_'min_mse_*.pth")
        if len(exists_pths) > 0:
            for p in exists_pths:
                os.remove(p)

        filename = f"ck_epoch_{epoch}_min_mse_{max_metrics['min_mse']:.3f}.pth"

        save_path = os.path.join(config.output, filename)
    
        logger.info(f'{save_path} saving......')
        torch.save(save_state, save_path)
        logger.info(f'{save_path} saved !!!')