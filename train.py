# from torchsummary import summary
from torchinfo import summary
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torchvision

# Custom
from model import ChleeCNN
import config
import datasets
from tqdm import tqdm

import matplotlib.pyplot as plt
import os

import math
from torch.optim.lr_scheduler import _LRScheduler
from datetime import datetime
import numpy as np

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


# run_epoch(model, loader_dict, optimizer, loss_fn, epoch)
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
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            optimizer.zero_grad()   # optimizer gradient를 0으로 초기화
            loss.backward()
            optimizer.step()

            print(f"loss: {loss.item():.4f}")

            # Train Loss 누적
            train_loss += loss.item() * labels.size(0)
            # Train Accuracy 누적
            preds = outputs.argmax(dim=1)
            train_accuracy += (preds == labels).sum().item()
            train_total_samples += labels.size(0)

    train_loss /= train_total_samples
    train_accuracy /= train_total_samples
    print(f"Train Loss: {train_loss:.2f} | Train Accuracy: {train_accuracy*100:.2f}")


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
                # Train Loss 누적
                preds = outputs.argmax(dim=1)
                valid_accuracy += (preds == labels).sum().item()
                valid_total_samples += labels.size(0)
    valid_loss /= valid_total_samples
    valid_accuracy /= valid_total_samples
    print(f"Validation Loss: {valid_loss:.2f} | Valid Accuracy: {valid_accuracy*100:.2f}")

    loss_dict = {
        "train_loss": train_loss,
        "valid_loss": valid_loss
    }
    accuracy_dict = {
        "train_accuracy": train_accuracy,
        "valid_accuracy": valid_accuracy
    }


    
    return loss_dict, accuracy_dict


def save_loss_accuracy_graph(train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list, cfg):
    epochs = range(1, cfg.epochs+1)

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

    save_dir = os.path.join("results", {cfg.exp_name})
    os.makedirs(exist_ok=True)
    
    save_path = os.path.join(save_dir, "loss_accuracy_graph.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Loss and Accuracy Graph is saved at {save_path}!")


def save_lr_graph(lr_list: list, cfg):
    epochs = range(1, cfg.epochs+1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, lr_list, label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
    
    save_dir = os.path.join("results", cfg.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, "lr_graph.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Learning Rate graph saved at {save_path}")


def save_checkpoint(model, optimizer, scheduler, epoch, save_dir, filename):
    """
    model, optimizer, scheduler 상태를 포함한 checkpoint 저장
    """
    save_path = os.path.join(save_dir, filename)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"):
    """
    checkpoint 불러오기.
    model은 필수, optimizer와 scheduler는 resume 학습용 선택적
    반환값: 이어서 학습할 start_epoch
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"{checkpoint_path} does not exist.")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 모델 상태 로드
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✅ Model weights loaded from {checkpoint_path}")

    start_epoch = checkpoint.get("epoch", 0) + 1  # 이어서 학습할 epoch

    # optimizer, scheduler 상태 로드 (resume 학습용)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("✅ Optimizer state loaded")

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("✅ Scheduler state loaded")

    return start_epoch


def main():
    cfg = config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Init model
    model = ChleeCNN(cfg.num_classes)
    model.to(device)
    summary(model, (1, 3, 224, 224))      # torchinfo

    # Dataset Method 1 : Create custom Dataset after inherit Dataset class
    train_dataset = datasets.PokemonDataset(data_path=cfg.train_data_path, transform=cfg.train_transform)
    valid_dataset = datasets.PokemonDataset(data_path=cfg.valid_data_path, transform=cfg.valid_transform)

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
    scheduler = WarmupCosineAnnealingLR(optimizer, total_epochs=cfg.epochs, warmup_epochs=5, eta_min=1e-6)

    # Init Loss Function -> CrossEntropyLoss
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.loss_label_smoothing)

    # save path
    model_save_path = os.path.join("checkpoint", cfg.exp_name)
    os.makedirs(model_save_path, exist_ok=True)

    latest_loss = np.inf
    patience = 0
    early_stopping = 10

    train_loss_list, valid_loss_list = [], []
    train_accuracy_list, valid_accuracy_list = [], []
    lr_list = []

    # Resume training
    start_epoch = 0
    if cfg.resume:
        start_epoch = load_checkpoint(cfg.resume, model, optimizer, scheduler, device)
    
    # Train Loop
    for epoch in range(start_epoch, cfg.epochs):     # 100 epoch
        loss_dict, accuracy_dict = run_epoch(model, loader_dict, optimizer, loss_fn, epoch, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        lr_list.append(current_lr)
        
        scheduler.step()

        train_loss_list.append(loss_dict['train_loss'])
        valid_loss_list.append(loss_dict['valid_loss'])
        train_accuracy_list.append(accuracy_dict['train_accuracy'])
        valid_accuracy_list.append(accuracy_dict['valid_accuracy'])

        # ↓ 지금 model의 파라미터만 저장중임
        # torch.save(model.state_dict(), os.path.join(model_save_path, f"epoch_{epoch+1}.pt"))
        save_checkpoint(model, optimizer, scheduler, epoch, model_save_path, f"epoch_{epoch+1}.pt")

        # ===== early stopping =====
        if latest_loss > loss_dict['valid_loss']:
            latest_loss = loss_dict['valid_loss']
            patience = 0
            # torch.save(model.state_dict(), os.path.join(model_save_path, f"best.pt"))
            save_checkpoint(model, optimizer, scheduler, epoch, model_save_path, "best.pt")
            print(f"Saved best.pt file at {epoch+1}epoch")
        else:
            patience += 1
            if patience >= early_stopping:
                print(f"Early stopping!")
                break


    save_loss_accuracy_graph(train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list, cfg)
    save_lr_graph(lr_list, cfg)

if __name__ == "__main__":
    main()

