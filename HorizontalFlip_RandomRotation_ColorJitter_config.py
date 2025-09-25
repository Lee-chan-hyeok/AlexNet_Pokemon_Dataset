# modified from origin
# Add transforms
#   - Resize(256)
#   - RandomCrop(224)
#   - RandomHorizontalFlip()
#   - RandomRotation()
#   - ColorJitter()

import torch
from torchvision import transforms

# ============ Config related to model ============
num_classes = 149
# ====================================================


# ============ Config related to Dataset ============
train_data_path = "PokemonDataset_Split/train"
valid_data_path = "PokemonDataset_Split/valid"

train_transform = transforms.Compose([
    transforms.Resize(256),
    # transforms.CenterCrop(256),   <- 256으로 Resize 후에 RandomCrop으로 다양성을 늘려보자!
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    # ↓↓↓ int 2를 주면 InterpolationMode.BILINEAR로 설정됨.
    # Default: NEAREST(빠르지만 이미지가 거칠어짐), BILINEAR: 일반적으로 많이 사용하고 이미지를 부드럽게 처리함
    transforms.RandomRotation(degrees=15, interpolation=2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05), # <- GPT 추천
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
valid_transform = transforms.Compose([
    transforms.Resize(256),         # <- 픽셀이 너무 깨지지 않게 256으로 resize
    transforms.CenterCrop(224),     # <- 중앙을 224로 잘라서 일관성을 맞춰주기
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
batch_size = 32
learning_rate = 1e-3
lr_scheduler = True

# early stop
patience = 0
early_stopping = 10
# =========================================================


# ============ Config related to CrossEntropy Loss ============
# 정답 라벨이 One-Hot Vector 형태로 나옴. ex) [1, 0, 0]
# 예측 결과가 One-Hot Vector과 같은 형태가 아닌 ex) [0.6, 0.3, 0.1] <- 이런식으로 나옴.
# 예측한 클래스가 정답과의 차이가 존재하고 이 값을 너무 [1, 0, 0] 이라고 하는 정답 값에 맞추려고 한다.
# 그러다보니 새로운 데이터가 들어왔을 때 Generalization이 잘 되지 않는 경우가 발생함.
# 그래서 다른 클래스들의 차이들도 조금은 고려하자는 차원에서 정답 라벨의 값 분포를 고르게 펼침 ex) [0.9, 0.05, 0.05]
# 클래스별 예측한 값과 정답 라벨간의 차이가 전체적으로 줄어들면서 오차가 덜 극단적으로 계산됨
loss_label_smoothing = 1e-1
# =============================================================

# ============ Config related to experiments name ============
exp_name = f"HorizontalFlip_RandomRotation_ColorJitter_Epoch{epochs}_Batch{batch_size}_Lr{learning_rate}_Scheduler{lr_scheduler}_Smoothing{loss_label_smoothing}/"
# =============================================================

# resume = "checkpoint/" + exp_name + "epoch_35.pt"
resume = None