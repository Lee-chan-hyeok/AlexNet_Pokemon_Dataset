# modified from origin
# Add transforms
#   - Resize(256)
#   - CenterCrop(256)
#   - RandomCrop(224)
#   - RandomHorizontalFlip()

import torch
from torchvision import transforms

# ============ Config related to model ============
num_classes = 149
# ====================================================


# ============ Config related to Dataset ============
train_data_path = "PokemonDataset_Split/train"
valid_data_path = "PokemonDataset_Split/valid"

train_transform = transforms.Compose([
    transforms.Resize(256),                 # short side resize 256
    transforms.CenterCrop(256),             # ramained side center crop 256
    transforms.RandomCrop(224),      # random crop 224 x 224
    transforms.RandomHorizontalFlip(p=0.5), # random horizontal flip
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
batch_size = 128
learning_rate = 1e-3
# early stop
patience = 0
early_stopping = 10
# =========================================================


# ============ Config related to CrossEntropy Loss ============
loss_label_smoothing = 1e-1
# =============================================================

# ============ Config related to experiments name ============
exp_name = f"HorizontalFlip_Epoch{epochs}_Batch{batch_size}_Lr{learning_rate}_Smoothing{loss_label_smoothing}/"
# =============================================================

# resume = "checkpoint/" + exp_name + "epoch_35.pt"
resume = None