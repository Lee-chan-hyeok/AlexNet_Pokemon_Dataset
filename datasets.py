import os
import json

import torch
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset

class PokemonDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path
        self.transform = transform

        self.image_path_list = []
        self.label_list = []
        # self.class_to_idx = {}
        self.idx_to_class = {}

        extension = (".jpg", ".jpeg", "png")

        classes_name = sorted(os.listdir(data_path))
        for class_idx, class_name in enumerate(classes_name):
            # self.class_to_idx[class_name] = class_idx
            self.idx_to_class[class_idx] = class_name
            class_path = os.path.join(data_path, class_name)

            images = os.listdir(class_path)
            # 이미지 확장자를 가진 애들만 가져옴
            images = [image for image in images if image.lower().endswith(extension)]
            for image in images:                
                image_path = os.path.join(class_path, image)
                self.image_path_list.append(image_path)
                self.label_list.append(class_idx)


    def __len__(self):
        return len(self.image_path_list)
        
    def __getitem__(self, idx):
        # image_path = self.image_path_list[idx]
        # image = Image.open(image_path)
        # image = image.convert("RGB")  # P 모드나 투명도 문제 처리
        image = Image.open(self.image_path_list[idx])
        
        if image.mode == "P":
            image = image.convert("RGBA").convert("RGB")
        else:
            image = image.convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.label_list[idx], dtype=torch.long)
        return image, label
