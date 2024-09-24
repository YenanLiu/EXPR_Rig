import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights

class PerceptualLoss(nn.Module):
    def __init__(self, device, feature_layers=[0, 5, 10, 19, 28], use_mse=True):
        """
        Initializes the perceptual loss module.
        
        Args:
            feature_layers (list): Indices of VGG layers to use for feature extraction.
            use_mse (bool): If True, use MSE loss; otherwise, use L1 loss.
        """
        super(PerceptualLoss, self).__init__()

        self.feature_layers = feature_layers
        
        # Load a pre-trained VGG16 network from torchvision
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        vgg.eval()
        vgg = vgg.to(device)

        # Extract the layers specified by feature_layers
        self.vgg_layers = nn.ModuleList([vgg[i] for i in range(max(feature_layers) + 1)])
        
        # Define the loss function (MSE or L1 loss)
        self.loss_fn = nn.MSELoss() if use_mse else nn.L1Loss()
        
        # Set VGG to evaluation mode and freeze its parameters
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
    
    def forward(self, input_img, target_img):

        perceptual_loss = 0.0
        with torch.no_grad():
            # Pass both input and target images through selected VGG layers
            for i, layer in enumerate(self.vgg_layers):
                input_img = layer(input_img)
                target_img = layer(target_img)
                # Calculate the loss between the feature maps
                if i in self.feature_layers:
                    # print(f"Layer {i} - Input shapes: pred {input_img.shape}, ori {target_img.shape}")
                    perceptual_loss += self.loss_fn(input_img, target_img)

        return perceptual_loss

def rig_mse_loss(predict, gt):
    loss = torch.mean((predict - gt) ** 2)
    return loss

def cosine_similarity_loss(fea1, fea2):
    normalized_fea1 = fea1 / fea1.norm(dim=1, keepdim=True)
    normalized_fea2 = fea2 / fea2.norm(dim=1, keepdim=True)
    return 1 - F.cosine_similarity(normalized_fea1, normalized_fea2).mean() 

def kl_divergence_loss(fea1, fea2):
    return F.kl_div(F.log_softmax(fea1, dim=-1), F.softmax(fea2, dim=-1), reduction='batchmean')

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.mse_loss = nn.MSELoss()

    def forward(self, fea1, fea2, label):
        # label = 1 if similar, 0 if dissimilar
        dist = torch.norm(fea1 - fea2, p=2, dim=1)  # Euclidean distance
        loss_similar = label * (dist ** 2)
        loss_dissimilar = (1 - label) * F.relu(self.margin - dist) ** 2
        return torch.mean(loss_similar + loss_dissimilar)
    
def loss_calculation(pred_rigs, gt_rigs, ori_img, pred_img, ori_embed=None, pred_embed=None, rig_mse=0, fea_cosine=0, fea_kl=0, fea_cst=0, prceptual_ls=0, pixel_ls=0, device=None):
    loss_dict = {}
    B = pred_rigs.shape[0]
    if rig_mse > 0:
        loss = rig_mse_loss(pred_rigs, gt_rigs)
        loss_dict["rig_mse"] = loss * rig_mse
    if fea_cosine > 0 and ori_embed is not None and pred_embed is not None:
        loss = cosine_similarity_loss(ori_embed, pred_embed)
        loss_dict["fea_cos"] = loss * fea_cosine
    if fea_kl > 0 and ori_embed is not None and pred_embed is not None:
        loss = kl_divergence_loss(ori_embed, pred_embed)
        loss_dict["fea_kl"] = loss
    if fea_cst > 0 and ori_embed is not None and pred_embed is not None:
        cst_func = ContrastiveLoss()
        dim = pred_embed.shape[-1]

        pred_embed = pred_embed / pred_embed.norm(dim=1, keepdim=True) # [B, dim]
        ori_embed = ori_embed / ori_embed.norm(dim=1, keepdim=True) # [B, dim]

        positive_pred = pred_embed
        positive_fea = ori_embed
        positive_labels = torch.ones(B)  # Positive labels (1 for all)

        pred_repeat = pred_embed.unsqueeze(1).repeat(1, B, 1)  
        fea_repeat = ori_embed.unsqueeze(0).repeat(B, 1, 1) 

        negative_mask = ~torch.eye(B, dtype=bool)  # Shape: [B, B], diagonal is False
        negative_pred = pred_repeat[negative_mask].view(-1, dim)
        negative_fea = fea_repeat[negative_mask].view(-1, dim)

        negative_pred = pred_repeat[negative_mask].view(B * (B - 1), dim)
        negative_fea = fea_repeat[negative_mask].view(B * (B - 1), dim)
        negative_labels = torch.zeros(B * (B - 1))  # Negative labels (0 for all)

        all_pred = torch.cat([positive_pred, negative_pred], dim=0)
        all_fea = torch.cat([positive_fea, negative_fea], dim=0)
        all_labels = torch.cat([positive_labels, negative_labels], dim=0)

        loss = cst_func(all_pred, all_fea, all_labels)
        loss_dict["fea_cst"] = loss * fea_cst
    
    if pixel_ls > 0:
        mse_loss = nn.MSELoss()

        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        pred_img = (pred_img + 1) / 2
        pred_img =  transform(pred_img)
        
        loss = mse_loss(pred_img, ori_img)
        loss_dict["pixel_mse_ls"] = loss * pixel_ls
        
    if prceptual_ls > 0:
        ls_func = PerceptualLoss(device)

        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        pred_img = (pred_img + 1) / 2
        pred_img =  transform(pred_img)

        loss = ls_func(pred_img, ori_img)
        loss_dict["percept_ls"] = loss * prceptual_ls

    return loss_dict

if __name__ == "__main__":
    B, rig_param, H, W, embed_dim = 4, 139, 256, 256, 384
    pred_rigs = torch.randn(B, rig_param)
    gt_rigs = torch.randn(B, rig_param)
    ori_img = torch.randn(B, 3, H, W)
    pred_img = torch.randn(B, 3, H, W)
    ori_embed = torch.randn(B, embed_dim)
    pred_embed = torch.randn(B, embed_dim)
    loss =  loss_calculation(pred_rigs, gt_rigs, ori_img, pred_img, ori_embed, pred_embed, rig_mse=1, fea_cosine=1, fea_kl=1, fea_cst=1, prceptual_ls=1, pixel_ls=1)
    print(loss)