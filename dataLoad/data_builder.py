import torch
import torch.distributed as dist

from .data_load import RigImgLoad
 
def build_data_loader(cfg, is_train):
    if not is_train:
        dataset = RigImgLoad(data_dir=cfg.data_dir, pkl_path=cfg.rig_pkl_val_path, input_size=cfg.img_size, is_train=False)
        if cfg.use_ddp:
            val_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.val_per_batch, shuffle=False, 
                                                num_workers=cfg.num_workers, sampler=val_sampler, pin_memory=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.val_per_batch, shuffle=True, 
                                                num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)

    else:
        dataset = RigImgLoad(data_dir=cfg.data_dir, pkl_path=cfg.rig_pkl_train_path, input_size=cfg.img_size, is_train=True)
        if cfg.use_ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.train_per_batch, shuffle=False, 
                                                   num_workers=cfg.num_workers, sampler=train_sampler, pin_memory=True, persistent_workers=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.train_per_batch, shuffle=False,
                                                num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)
    return dataset, loader