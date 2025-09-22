# modified from origin
# batch_size: 128 => 64

import torch
from torchvision import transforms

# ============ Config related to model ============
num_classes = 149
# ====================================================


# ============ Config related to Dataset ============
train_data_path = "PokemonDataset_Sample/train"
valid_data_path = "PokemonDataset_Sample/valid"

# train No Aug
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# ====================================================


# ============ Config related to DataLoader ============
train_shuffle = True   # DataLoader.shuffle
valid_shuffle = False
num_workers = 4         # DataLoader.num_workers
# =======================================================


# ============ Config related to Train running ============
epochs = 100
batch_size = 64
learning_rate = 1e-3
# =========================================================


# ============ Config related to CrossEntropy Loss ============
loss_label_smoothing = 1e-1
# =============================================================

# ============ Config related to experiments name ============
exp_name = f"TransformOrigin_Epoch{epochs}_Batch{batch_size}_Lr{learning_rate}_Smoothing{loss_label_smoothing}/"
# =============================================================

# resume = "checkpoint/" + exp_name + "epoch_35.pt"
resume = None