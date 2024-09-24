import torch
import torch.nn as nn 
import timm.models.vision_transformer
from .models_vit import vit_base_patch16

from functools import partial

class MAE_embed(nn.Module):
    def __init__(self, img_size=224):
        super(MAE_embed, self).__init__()
        self.model = vit_base_patch16(
            global_pool=True,
            drop_path_rate=0.1,
            img_size=img_size,
        )
        self.model.head = nn.Identity() 
    
    def forward(self, x):
        _, fea = self.model(x, True)
        return fea
 
if __name__ == "__main__":
    B, H, W = 4, 224, 224
    inp = torch.randn((B, 3, H, W))
    embed_model = MAE_embed()

    fea = embed_model(inp)
    print(fea.shape) # [4, 768]

    state_dict = torch.load("/project/ABAW6/MAE/MAE_expemb_acc_0.8792_clean.pth")
    # new_state_dict = {}
    # for key, value in state_dict.items():
    #     new_key = key.replace("module.main", "model")
    #     if new_key in ['model.head.weight', 'model.head.bias']:
    #         continue

    #     new_state_dict[new_key] = value 
    
    # torch.save(new_state_dict, "/project/ABAW6/MAE/MAE_expemb_acc_0.8792_clean.pth")
    # embed_model.load_state_dict(new_state_dict, strict=True)