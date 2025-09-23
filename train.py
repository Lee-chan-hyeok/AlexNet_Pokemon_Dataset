# from torchsummary import summary
# from torchinfo import summary
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torchvision

# Custom
from model import ChleeCNN
# import origin_config
# import origin_config_batch64 as config
# import origin_config_PaperTransform as config
import origin_config_RandomRotation as config
from datasets import PokemonDataset
from tqdm import tqdm

import matplotlib.pyplot as plt
import os

import math
from torch.optim.lr_scheduler import _LRScheduler
from datetime import datetime
import numpy as np

import argparse
import logging


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, total_epochs, warmup_epochs=5, eta_min=0, last_epoch=-1):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # cosine annealing
            cosine_epoch = self.last_epoch - self.warmup_epochs
            total_cosine_epochs = self.total_epochs - self.warmup_epochs
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * cosine_epoch / total_cosine_epochs)) / 2
                for base_lr in self.base_lrs
            ]


def run_epoch(
      model: ChleeCNN,
      data_loader: dict,
      optimizer: optim,
      loss_fn: nn.CrossEntropyLoss,
      epoch: int,
      device: str
    ):

    # train pipeline
    model.train()
    train_loader = data_loader['train_loader']

    train_loss = 0
    train_accuracy = 0
    train_total_samples = 0
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1} training...") as t_loader:
        for batch in t_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)             # 모델의 예측 값. shape: [batch_size, num_classes]
            loss = loss_fn(outputs, labels)     # 예측값과 정답값을 비교해서 손실값 계산
            
            optimizer.zero_grad()   # 파이토치에서 gradient가 누적되기 때문에 매 iter마다 0으로 초기화
            loss.backward()         # 각 parameter(weight, bias 등)의 gradient를 계산 -> "현재 파라미터가 얼마나 잘못되어있냐"를 계산함.
            optimizer.step()        # 계산된 gradient를 통해 parameter(weight, bias 등)를 업데이트

            # Train Loss 누적
            train_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            # Train Accuracy 누적
            train_accuracy += (preds == labels).sum().item()
            train_total_samples += labels.size(0)

    train_loss /= train_total_samples
    train_accuracy /= train_total_samples
    logging.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy*100:.2f}")


    # valid pipeline
    model.eval()
    valid_loader = data_loader['valid_loader']

    valid_loss = 0
    valid_accuracy = 0
    valid_total_samples = 0

    with torch.no_grad():
        with tqdm(valid_loader, desc=f"Epoch {epoch+1} evaluation...") as v_loader:
            for batch in v_loader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)

                # Validation Loss 누적
                valid_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                # Validation Accuracy 누적
                valid_accuracy += (preds == labels).sum().item()
                valid_total_samples += labels.size(0)
    valid_loss /= valid_total_samples
    valid_accuracy /= valid_total_samples
    logging.info(f"Epoch {epoch+1} - Validation Loss: {valid_loss:.4f} | Valid Accuracy: {valid_accuracy*100:.2f}")

    loss_dict = {
        "train_loss": train_loss,
        "valid_loss": valid_loss
    }
    accuracy_dict = {
        "train_accuracy": train_accuracy,
        "valid_accuracy": valid_accuracy
    }
    
    return loss_dict, accuracy_dict


def save_loss_accuracy_graph(train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list, model_save_path, cfg):
    epochs = range(1, len(train_loss_list) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, label="Train Loss")
    plt.plot(epochs, valid_loss_list, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy_list, label="Train Accuracy")
    plt.plot(epochs, valid_accuracy_list, label="Valid Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    save_dir = os.path.join(model_save_path, "result_graph")
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, "loss_accuracy_graph.png")
    plt.savefig(save_path)
    plt.close()

    logging.info(f"Loss and Accuracy Graph is saved at {save_path}!")


def save_lr_graph(lr_list: list, model_save_path, cfg):
    epochs = range(1, len(lr_list) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, lr_list, label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
    
    save_dir = os.path.join(model_save_path, "result_graph")
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, "lr_graph.png")
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Learning Rate graph saved at {save_path}")


def save_checkpoint(model, optimizer, epoch, save_path, filename, loss=None, scheduler=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if loss is not None:
        checkpoint['loss'] = loss
    
    torch.save(checkpoint, os.path.join(save_path, filename))


def main():
    cfg = config

    # save path
    model_save_path = os.path.join("checkpoint", cfg.exp_name)
    os.makedirs(model_save_path, exist_ok=True)

    # ================ 로그 파일 + 화면 출력 설정 ================
    logging.basicConfig(
        filename=os.path.join(model_save_path, "train_log.txt"),
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    # =========================================================
    
    # 'cuda', 'cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")

    # Init model
    model = ChleeCNN(cfg.num_classes)   # <- 149
    model.to(device)
    # summary(model, (1, 3, 224, 224))      # torchinfo

    # Dataset Method 1 : Create custom Dataset after inherit Dataset class
    train_dataset = PokemonDataset(data_path=cfg.train_data_path, transform=cfg.train_transform)
    valid_dataset = PokemonDataset(data_path=cfg.valid_data_path, transform=cfg.valid_transform)

    # Dataset Method 2 : Use ImageFolder
    # train_dataset = torchvision.datasets.ImageFolder(root=cfg.train_data_path, transform=cfg.train_transform)
    # valid_dataset = torchvision.datasets.ImageFolder(root=cfg.valid_data_path, transform=cfg.valid_transform)

    # Init DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.train_shuffle,
        num_workers=cfg.num_workers
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=cfg.valid_shuffle,
        num_workers=cfg.num_workers
    )
    
    loader_dict = {
        'train_loader' : train_loader,
        'valid_loader' : valid_loader
    }

    # Init optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.learning_rate)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 10 epoch마다 lr을 0.1배 감소
    # scheduler = WarmupCosineAnnealingLR(optimizer, total_epochs=cfg.epochs, warmup_epochs=5, eta_min=1e-6)

    # Init Loss Function -> CrossEntropyLoss
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.loss_label_smoothing)

    latest_loss = float(np.inf)
    patience = cfg.patience
    early_stopping = cfg.early_stopping

    train_loss_list, valid_loss_list = [], []
    train_accuracy_list, valid_accuracy_list = [], []
    lr_list = []

    # Train Loop
    for epoch in range(cfg.epochs):
        loss_dict, accuracy_dict = run_epoch(model, loader_dict, optimizer, loss_fn, epoch, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        lr_list.append(current_lr)
        
        # scheduler.step()

        train_loss_list.append(loss_dict['train_loss'])
        valid_loss_list.append(loss_dict['valid_loss'])
        train_accuracy_list.append(accuracy_dict['train_accuracy'])
        valid_accuracy_list.append(accuracy_dict['valid_accuracy'])

        # ===== early stopping =====
        if latest_loss > loss_dict['valid_loss']:
            latest_loss = loss_dict['valid_loss']
            patience = 0

            # checkpoint 저장 및 resume까지 고려하여 저장
            save_checkpoint(model, optimizer, epoch, model_save_path, f"epoch_{epoch+1}.pt", loss=loss_dict['train_loss'])
            # early stopping을 위한 best.pt만 고려하여 저장
            save_checkpoint(model, optimizer, epoch, model_save_path, "best.pt", loss=loss_dict['valid_loss'])
            logging.info(f"##### Saved best.pt file at {epoch+1}epoch #####")
        else:
            patience += 1
            if patience >= early_stopping:
                logging.info(f"Early stopping!")
                break

    save_loss_accuracy_graph(train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list, model_save_path, cfg)
    save_lr_graph(lr_list, model_save_path, cfg)

if __name__ == "__main__":
    main()

