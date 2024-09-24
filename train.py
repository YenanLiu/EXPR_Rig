import datetime
import time
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import defaultdict
from timm.utils import AverageMeter

from models import *
from dataLoad import * 
from utils import *

warnings.filterwarnings("ignore")

def train_worker(gpu, ngpus_per_node, cfg):
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    if cfg.use_ddp:
        rank = gpu
        dist.init_process_group(backend='nccl', init_method='env://', world_size=ngpus_per_node, rank=rank)
        print(f"Process group initialized. Rank: {rank}, World Size: {ngpus_per_node}")
    else:
        rank = 0

    if rank == 0:
        initialize_wandb(cfg)
        logger = get_logger()
        logger.info(f"Available device number is {torch.cuda.device_count()}")
    
    # init the dataloader
    train_dataset, train_loader = build_data_loader(cfg, is_train=True)
    val_dataset, val_loader = build_data_loader(cfg, is_train=False)

    fea_flag = cfg.fea_cos > 0 or cfg.fea_kl > 0 or cfg.fea_cst > 0
    rendered_img_flag = cfg.pixel_ls > 0 or cfg.percep_ls > 0

    model = rigModel(cfg.vgg_pretrain, cfg.render_pretrain, cfg.mae_pretrain, fea_flag, rendered_img_flag).to(device) 
 
    if rank == 0:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("==> Total params: %.2fM" % (n_parameters / 1e6))

    if cfg.use_ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)

    # optimizer & learning scheduler
    optimizer = build_optimizer(cfg, model) 
    lr_scheduler = build_scheduler(cfg, optimizer, len(train_loader))

    min_mse = float('inf')
    max_metrics = {'min_mse': min_mse}

    start_time = time.time()
    for epoch in range(0, cfg.epochs):
        if cfg.use_ddp:
            train_loader.sampler.set_epoch(epoch)
        if rank == 0:
            logger.info(f"****************     Starting training the {epoch} epoch       *********************")
            logger.info(f"The amount of training data: {len(train_dataset)}, validation data: {len(val_dataset)}")
        
        metric_dict = evaluation(cfg, model, val_loader, device, epoch=epoch)

        # Training 
        train_one_epoch(cfg, model, train_loader, optimizer, lr_scheduler, epoch, device)

        if cfg.use_ddp:
            dist.barrier()

        # Evaluation
        metric_dict = evaluation(cfg, model, val_loader, device, epoch=epoch)

        save_flag = metric_dict["mse"] < max_metrics["min_mse"]
        if rank == 0 and save_flag:
            max_metrics["min_mse"] = metric_dict["mse"]
            save_checkpoint(cfg, epoch, 
                model.module if cfg.use_ddp else model, optimizer, max_metrics, lr_scheduler, 
                save_flag)   

        if cfg.use_ddp:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if rank == 0:
        logger.info('Training and Eval time {}'.format(total_time_str))
    
def train_one_epoch(config, model, data_loader, optimizer, lr_scheduler, epoch, device):
    logger = get_logger()
    rank = dist.get_rank() if config.use_ddp else 0
    wandb = get_wandb()

    if rank == 0:
        logger.info(f'*******************   Training at epoch {epoch}   ***********************')

    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    log_vars_meters = defaultdict(AverageMeter) 

    start = time.time()
    end = time.time()
    
    for idx, batch_data in enumerate(data_loader):    
        ori_img_tensor, img_transform_tensor, rig_tensor, img_path = batch_data

        ori_img_tensor = ori_img_tensor.to(device, non_blocking=True)
        img_transform_tensor = img_transform_tensor.to(device, non_blocking=True)
        rig_tensor = rig_tensor.to(device, non_blocking=True)

        pred_rigs, pred_rendered_img, pred_img_embed, ori_img_embed = model(img_transform_tensor, ori_img_tensor, config.img_size)

        loss_dict = loss_calculation(pred_rigs, rig_tensor, ori_img_tensor, pred_rendered_img, ori_img_embed, pred_img_embed, \
                rig_mse=config.rig_mse, fea_cosine=config.fea_cos, fea_kl=config.fea_kl, \
                fea_cst=config.fea_cst, prceptual_ls=config.percep_ls, pixel_ls=config.pixel_ls, device=device)
        
        if config.use_ddp:
            for key, value in loss_dict.items():
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                loss_dict[key] = value / dist.get_world_size()

        total_loss = sum(loss_dict.values())
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
            
        if rank == 0:
            loss_meter.update(total_loss.item())
            for loss_name, value in loss_dict.items():
                log_vars_meters[loss_name].update(value.item())

        batch_time.update(time.time() - end)
        end = time.time()                
        if idx % config.print_freq == 0 and rank == 0:
            lr = optimizer.param_groups[0]['lr']
            log_vars_str = ' '.join(f'{n} {m.val:.4f} ({m.avg:.4f})' for n, m in log_vars_meters.items())
            logger.info(f'Train: [{epoch}/{config.epochs}][{idx}/{num_steps}] '
                        f'total_loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        f'{log_vars_str}')

            if wandb is not None and rank == 0:
                log_stat = {f'iter/train_{n}': m.avg for n, m in log_vars_meters.items()}
                log_stat['iter/train_total_loss'] = loss_meter.avg
                log_stat['iter/learning_rate'] = lr
                wandb.log(log_stat)

    epoch_time = time.time() - start
    if rank == 0:
        logger.info(f'EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}')
        logger.info(f"Avg Training Loss on epoch {epoch} is: {loss_meter.avg:.5f}")

def gather_global_mse_mae(local_mse, local_mae, num_samples):
    global_mse = torch.tensor([0.0], device=local_mse.device)
    global_mae = torch.tensor([0.0], device=local_mae.device)
    global_samples = torch.tensor([0.0], device=local_mse.device)

    dist.all_reduce(local_mse, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_mae, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)

    global_mse = local_mse / num_samples
    global_mae = local_mae / num_samples

    return global_mse, global_mae

@torch.no_grad()
def evaluation(config, model, data_loader, device, epoch):
    logger = get_logger()
    wandb = get_wandb()
    rank = dist.get_rank() if config.use_ddp else 0
     
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    log_vars_meters = defaultdict(AverageMeter)

    local_mse_sum = 0.0
    local_mae_sum = 0.0
    local_samples = 0

    vis_dict = {}
    end = time.time()
    for j, batch_data in enumerate(data_loader):
        ori_img_tensor, img_transform_tensor, rig_tensor, img_path = batch_data

        ori_img_tensor = ori_img_tensor.to(device, non_blocking=True)
        img_transform_tensor = img_transform_tensor.to(device, non_blocking=True)
        rig_tensor = rig_tensor.to(device, non_blocking=True)

        pred_rigs, pred_rendered_img, pred_img_embed, ori_img_embed = model(img_transform_tensor, ori_img_tensor, config.img_size)

        loss_dict = loss_calculation(pred_rigs, rig_tensor, ori_img_tensor, pred_rendered_img, ori_img_embed, pred_img_embed, \
                rig_mse=config.rig_mse, fea_cosine=config.fea_cos, fea_kl=config.fea_kl, \
                fea_cst=config.fea_cst, prceptual_ls=config.percep_ls, pixel_ls=config.pixel_ls, device=device)
        loss = sum(loss_dict.values())

        local_mse, local_mae = compute_metrics(pred_rigs, rig_tensor)
        local_mse_sum += local_mse.item()
        local_mae_sum += local_mae.item()
        local_samples += pred_rigs.size(0)

        if config.use_ddp:
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        loss_meter.update(reduced_loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        if j % config.print_freq == 0 and rank == 0:
            log_vars_str = ' '.join(f'{n} {m.val:.4f} ({m.avg:.4f})' for n, m in log_vars_meters.items())
 
            logger.info(f'Test: [{j}/{len(data_loader)}] '
                        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        f'{log_vars_str}')

            if wandb is not None and rank == 0:
                log_stat = {f'iter/val_{n}': m.avg for n, m in log_vars_meters.items()}
                log_stat['iter/val_total_loss'] = loss_meter.avg
                wandb.log(log_stat)

        # Visualization
        if j < int(config.vis_batch) and rank == 0:
            vis_dict = vis_imgs(ori_img_tensor, pred_rendered_img, img_path, vis_dict=vis_dict)

    #####################   Our metric   #####################

    local_mse_sum = torch.tensor(local_mse_sum, device=device)
    local_mae_sum = torch.tensor(local_mae_sum, device=device)
    local_samples = torch.tensor(local_samples, device=device)

    global_mse, global_mae = gather_global_mse_mae(local_mse_sum, local_mae_sum, local_samples)

    if config.use_ddp:
        dist.barrier()
    if rank == 0 and config.wandb:
        wandb.log(vis_dict)
    
    if rank == 0:
        logger.info(f"test rig_mse: {global_mse:.4f}, rig_mae: {global_mae:.4f}")

        if config.wandb:
            wandb.log({
                "epoch": epoch, 
                "mse": global_mse,
                "mae": global_mae,
                "total_loss": loss_meter.avg,
        })

    result_dict = dict(total_loss=loss_meter.avg)
    for n, m in log_vars_meters.items():
        result_dict[n] = m.avg

    if rank == 0:
        logger.info(f"Avg Validation Loss on epoch {epoch} is: {result_dict['total_loss']:.5f}")
    
    return {"mse": global_mse, "mae": global_mae}

