import argparse
import gc
import os
import time
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from mmcv.utils import collect_env
from mmcv.runner import set_random_seed

from utils import * 
from train import train_worker

warnings.filterwarnings("ignore")

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ddp", "-ddp", action="store_true", help="Use DDP for training") # 
    parser.add_argument("--wandb", "-wandb", action="store_true", help="Use Wandb for recording results")# 
    parser.add_argument("--wandb_name", "-wn", type=str, default="EXPRRig", help="helping distinguish the training change")
    parser.add_argument("--project",  type=str, default="EXPRRig", help="helping distinguish the training change")

    parser.add_argument("--opts", help="modify config options by adding 'KEY-VALUE' pairs", default=None, nargs='+')
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument("--seed", type=int, default=12345, help="")

    # dataload
    parser.add_argument("--rig_pkl_train_path", type=str, default="/project/zhangwei/xusheng/rig2img/rig2img_bernice_20240919_train_p2.pkl", help="")
    parser.add_argument("--rig_pkl_val_path", type=str, default="/project/zhangwei/xusheng/rig2img/rig2img_bernice_20240919_val_p2.pkl", help="")
    parser.add_argument("--data_dir", type=str, default="/project/zhangwei/xusheng/img2rig_data/", help="")
    parser.add_argument("--num_workers", type=int, default=8, help="")

    # pretrain model
    parser.add_argument("--render_pretrain", type=str, default="/project/_expr_train/rig/pre_pth/rig_pretrain.pth", help="")
    parser.add_argument("--vgg_pretrain", type=str, default="/project/_expr_train/rig/pre_pth/RepVGGplus_clean.pth", help="")
    parser.add_argument("--mae_pretrain", type=str, default="/project/ABAW6/MAE/MAE_expemb_acc_0.8792_clean.pth", help="")

    # optimizer
    parser.add_argument("--optimizer_name", type=str, default='adamw', help="")
    parser.add_argument("--adamw_eps", type=float, default=1e-8, help="")
    parser.add_argument("--adamw_betas", type=float, nargs=2, default=[0.9, 0.999], help="")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="")

    # scheduler
    parser.add_argument("--scheduler_name", type=str, default='cosine', choices=('cosine', 'setplr', 'multisetplr'), help="")
    parser.add_argument("--min_lr", type=float, default=4e-5, help="")
    #
    parser.add_argument("--step_size", type=int, default=40, help="")
    parser.add_argument("--step_gamma", type=float, default=0.1, help="")   

    parser.add_argument("--msteplr_gamma", type=float, default=0.1, help="")
    parser.add_argument("--msteplr_milestones", type=float, nargs=4, default=[1, 2, 3, 4], help="")

    # training
    parser.add_argument("--base_lr", "-blr", type=float, default=1e-4, help="")
    parser.add_argument("--epochs", type=int, default=150, help="")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="")
    parser.add_argument("--warmup_lr", type=float, default=4e-6, help="")
    parser.add_argument("--img_size", type=int, default=224, help="")

    parser.add_argument("--train_per_batch", "-train_b", default=2, type=int, help="resume from checkpoint")
    parser.add_argument("--val_per_batch", "-val_b", default=2, type=int, help="resume from checkpoint")
    parser.add_argument("--output", type=str, default="/project/_expr_train/rig/", help="root of output folder")

    # eval
    parser.add_argument("--eval_only", '-eval', action="store_true", help="")
    parser.add_argument("--eval_model_path", type=str, default="", help="")    
    parser.add_argument("--vis_batch", type=int, default=5, help="")    
    parser.add_argument("--print_freq", type=int, default=1,  help="")    

    # loss
    parser.add_argument("--rig_mse", '-rmse', type=float, default=1.0, help="")
    parser.add_argument("--fea_cos", '-fcos', type=float, default=0.0, help="")
    parser.add_argument("--fea_kl", '-fkl', type=float, default=0.0, help="")
    parser.add_argument("--fea_cst", '-fcst', type=float, default=0.0, help="")
    parser.add_argument("--pixel_ls", '-pixel', type=float, default=0.0, help="")
    parser.add_argument("--percep_ls", '-percept', type=float, default=0.0, help="")

    args = parser.parse_args()
    return args

def main():
    # This code implementation can only support one node with milti-GPUs training
    args = parser_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12366'

    set_random_seed(args.seed, use_rank_shift=True)
    cudnn.benchmark = True

    log_dir = os.path.join(args.output, args.wandb_name, '{}'.format(time.strftime('%Y%m%d-%H%M%S')))

    if os.path.exists(log_dir):
        log_dir = os.path.join(args.output, args.wandb_name, '{}_{}'.format(time.strftime('%Y%m%d-%H%M%S'), np.random.randint(1, 10)))
    
    args.output = log_dir
    os.makedirs(args.output, exist_ok=True)

    logger = get_logger(args)
    logger.info(f"Log dir is {args.output}")   

    world_size = torch.cuda.device_count() if args.use_ddp else 1

    path = os.path.join(args.output, 'config.json')
    logger.info(f'Full config saved to {path}')

    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

        
    if args.use_ddp :
        mp.spawn(train_worker, nprocs=world_size, args=(world_size, args))
    else:
        train_worker(gpu=0, ngpus_per_node=world_size, cfg=args)

 
if __name__ == "__main__":
    main()