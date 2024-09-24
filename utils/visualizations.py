import torch
import numpy as np
import os

import torch.distributed as dist
from PIL import Image
from .custom_logging import get_wandb

def denormalize(tensor, mean, std):
    """
    反归一化，将图像恢复到原始的值
    :param tensor: 经过归一化的图像张量
    :param mean: 归一化时使用的均值
    :param std: 归一化时使用的标准差
    :return: 反归一化后的图像张量
    """
 
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    tensor = tensor * std + mean
    return tensor


@torch.no_grad()
def vis_imgs(ori_img, pred_img, frame_paths, vis_dict=None, vis_save_dir=None):
    """_summary_

    Args:
        ori_img (_type_): [B, 3, H, w]
        pred_img (_type_):  [B, 3, H, w]
    """
    wandb = get_wandb()

    # convert to [0, 1] from [-1, 1]
    pred_img = (pred_img + 1) / 2 

    B, _, H, W = ori_img.shape

    ori_img = ori_img.cpu().numpy()
    pred_img = pred_img.cpu().numpy()
    combined_imgs = []

    for ori_i, pred_i, f_path in zip(ori_img, pred_img, frame_paths):
        denormalized_ori_img = denormalize(torch.tensor(ori_i), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        denormalized_ori_img = denormalized_ori_img.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        denormalized_ori_img_Image = Image.fromarray((denormalized_ori_img * 255).astype(np.uint8))

        # Process predicted image
        pred_i = np.transpose(pred_i, (1, 2, 0))
        pred_i = (pred_i * 255).astype(np.uint8)
        pred_i_Image = Image.fromarray(pred_i)

        combined_img = Image.new('RGB', (W*2, H))
        combined_img.paste(denormalized_ori_img_Image, (0, 0))  
        combined_img.paste(pred_i_Image, (denormalized_ori_img_Image.width, 0)) 

        vname = f_path.rsplit("/", -2)[-1]

        if vis_save_dir is not None:
            os.makedirs(vis_save_dir, exist_ok=True)
            vis_save_path = os.path.join(vis_save_dir, vname + ".png")
            combined_img = combined_img.convert('RGB')
            combined_img.save(vis_save_path)
        
        if vis_dict is not None and wandb is not None:
            img_list = vis_dict.setdefault(vname, [])
            img_list.append(wandb.Image(combined_img))
    
    return vis_dict