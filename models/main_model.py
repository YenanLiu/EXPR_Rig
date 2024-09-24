import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from .backbones import *
from .render import GlobalGeneratorForVal

class rigModel(nn.Module):
    def __init__(self, rig_state_dict_path, render_state_dict_path, mae_state_dict_path, fea_flag, rendered_img_flag):
        super(rigModel, self).__init__()
        self.backbone = create_RepVGGplus_by_name("RepVGGplus-L2pse", num_classes=139, deploy=False)
        
        self.__load_pretrain(self.backbone, rig_state_dict_path, False)
        self.act = nn.Sigmoid()
        self.rendered_img_flag = rendered_img_flag
        self.fea_flag = fea_flag

        # Render
        if rendered_img_flag or fea_flag:
            self.render = GlobalGeneratorForVal(139, 3, output_size=256, n_blocks=6, norm_layer=nn.BatchNorm2d)
            self.__load_pretrain(self.render, render_state_dict_path, True)
            self.__set_all_params_frozen(self.render)

        # MAE embedding
        if fea_flag:
            self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.mae_model = MAE_embed()
            self.__load_pretrain(self.mae_model, mae_state_dict_path, True)
            self.__set_all_params_frozen(self.mae_model)

    def __load_pretrain(self, model, state_dict_path, strict=False):
        state_dict = torch.load(state_dict_path)

        model.load_state_dict(state_dict, strict=strict)
        print(f"successfully load pretrain model {state_dict_path}")

    def __set_all_params_frozen(self, model):
        for _, param in model.named_parameters():
            param.requires_grad = False

    def forward(self, transform_img, ori_img, inp_size):
        """_summary_

        Args:
            x (torch.tensor): [B, 3, 224, 224]
            inp_size: 224
        """
        out = self.backbone(transform_img)
        rigs = self.act(out['main'])

        # rendered img gene
        if self.rendered_img_flag or self.fea_flag:
            rigs_resized = rigs.unsqueeze(-1).unsqueeze(-1)
            rendered_img = self.render(rigs_resized)
            resized_img = F.interpolate(rendered_img, size=(inp_size, inp_size), mode='bilinear', align_corners=False)
        else:
            resized_img = None
        # mae img embedding
        if self.fea_flag:
            tramsform_img = self.transform((resized_img + 1) / 2)

            rendered_img_embed = self.mae_model(tramsform_img)
            ori_img_embed = self.mae_model(ori_img)
        else:
            rendered_img_embed = None
            ori_img_embed = None
        return rigs, resized_img, rendered_img_embed, ori_img_embed

if __name__ == "__main__":
    B, W, H, C = 4, 224, 224, 3
    inp_img = torch.randn((B, C, W, H))
    ori_img = torch.randn((B, C, W, H))

    rig_state_dict_path = "/project/_expr_train/rig/pre_pth/RepVGGplus_clean.pth"
    render_state_dict_path = "/project/_expr_train/rig/pre_pth/rig_pretrain.pth"
    mae_state_dict_path = "/project/ABAW6/MAE/MAE_expemb_acc_0.8792_clean.pth"
    rigmodel = rigModel(rig_state_dict_path, render_state_dict_path, mae_state_dict_path)

    out, rendered_img, rendered_img_embed, ori_img_embed = rigmodel(inp_img, ori_img, 224)
    print(out.shape)  # [4, 139] 
    print(rendered_img.shape)  # [4, 139] 
    print(rendered_img_embed.shape)  # [4, 139] 
    print(ori_img_embed.shape)  # [4, 139] 


    # state_dict_path = "/project/_expr_train/rig/pre_pth/RepVGGplus-L2pse-train256-acc84.06.pth"
    # state_dict = torch.load(state_dict_path)

    # new_dict = {}
    # for key, value in state_dict.items():
    #     if "aux" in key or "linear" in key:
    #         continue
    #     else:
    #         new_dict[key] = value
    #         print(key)
    
    # torch.save(new_dict, "/project/_expr_train/rig/pre_pth/RepVGGplus_clean.pth")
    # print("ok")
