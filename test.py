import Origin_config as config
import torch
from torch.utils.data import DataLoader

from model import ChleeCNN
from datasets import PokemonDataset

from tqdm import tqdm

import logging



def run_test():

    # ===== Logger 세팅 (파일 + 콘솔 동시 출력) =====
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("test_log.txt"),  # 파일에 기록
            logging.StreamHandler()               # 콘솔 출력
        ]
    )
    logger = logging.getLogger(__name__)

    # test할 pt 파일의 경로
    model_path = r"checkpoint\HorizontalFlip_RandomRotation_ColorJitter_Epoch100_Batch32_Lr0.001_SchedulerWarmupCosAnnealing_Smoothing0.1\best.pt"
    data_path = config.test_data_path
    batch_size = config.batch_size

    # ===== 모델 로딩 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChleeCNN(config.num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ===== 데이터셋 / 로더 =====
    test_dataset = PokemonDataset(data_path, transform=config.valid_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=config.valid_shuffle)

    idx_to_class = test_dataset.idx_to_class

    correct = 0

    with torch.no_grad():
        with tqdm(test_loader, desc=f"Test set evaluation...") as t_loader:
            for images, labels in t_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)  # [batch_size, num_classes]

                for i in range(len(images)):
                    pred_idx = outputs[i].argmax(dim=0).item()
                    confidence = outputs[i][pred_idx].item()
                    predicted = idx_to_class[pred_idx]
                    gt = idx_to_class[int(labels[i])]

                    if predicted == gt:
                        correct += 1

    accuracy = correct / len(test_dataset) * 100
    logger.info(f"Test Accuracy: {accuracy:.2f}%")
    logger.info(f"Evaluated on {len(test_dataset)} samples")


if __name__ == "__main__":
    run_test()
