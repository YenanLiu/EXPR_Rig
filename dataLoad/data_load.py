import pickle
import os
import torch
import json
import numpy as np
import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

class RigImgLoad(Dataset):
    def __init__(self, data_dir, pkl_path, input_size=256, is_train=False, **kwargs):
        super(RigImgLoad, self).__init__()

        with open(pkl_path, 'rb') as f:
           data = pickle.load(f)

        self.data_dir = data_dir
        self.is_train = is_train

        self.data = list(data.items())
        self.train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(size=(input_size, input_size), scale=(0.8, 1.2)), 
            transforms.Resize((input_size, input_size)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])             
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        name, rigs = self.data[index]
        img_path = os.path.join(self.data_dir, name)

        img = Image.open(img_path).convert("RGB")
        if self.is_train:
            img_transform = self.train_transform(img)
        else:
            img_transform = self.val_transform(img)
        
        ori_img_tensor = self.val_transform(img)

        rig_tensor = torch.tensor(rigs)
        
        return ori_img_tensor, img_transform, rig_tensor, img_path


if __name__ == "__main__":
    data_dir = "/project/zhangwei/xusheng/img2rig_data/"
    pkl_path = "/project/zhangwei/xusheng/rig2img/rig2img_bernice_20240919_train_p2.pkl"
    input_size = 256
    is_train = True

    dataset = RigImgLoad(data_dir, pkl_path, input_size, is_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=8, pin_memory=True)

    for i, (ori_img_tensor, img_transform, rig_tensor, img_path) in enumerate(loader):
        print(i)
        print(ori_img_tensor.shape)
        print(rig_tensor.shape)

