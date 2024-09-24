
import torch

def compute_metrics(pred_rigs, GT_rigs):
    mse = torch.sum(torch.mean((pred_rigs - GT_rigs) ** 2, dim=1))
    mae = torch.sum(torch.mean(torch.abs(pred_rigs - GT_rigs), dim=1))
    return mse, mae 

if __name__ == "__main__":
    B, dim = 4, 139
    pred_rigs = torch.randn((B, dim))
    GT_rigs = torch.randn((B, dim))
    mse, mae = compute_metrics(pred_rigs, GT_rigs)
    print(mse, mse.shape)
    print(mae, mae.shape)
