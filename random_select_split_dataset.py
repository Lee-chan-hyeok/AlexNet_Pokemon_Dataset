import os
import shutil
import random

# 원본 데이터셋
DATASET_DIR = "PokemonDataset"
# 최종 출력 폴더 (train/valid)
DST_DIR = "PokemonDataset_Split"
TRAIN_RATIO = 0.8

train_dir = os.path.join(DST_DIR, "train")
valid_dir = os.path.join(DST_DIR, "valid")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# 클래스 리스트
classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])

for class_name in classes:
    class_path = os.path.join(DATASET_DIR, class_name)
    images = sorted(os.listdir(class_path))
    if len(images) == 0:
        continue

    # 클래스별 랜덤 100장 선택
    selected_images = random.sample(images, min(100, len(images)))

    # train/valid 비율로 나누기
    random.shuffle(selected_images)
    n_train = int(len(selected_images) * TRAIN_RATIO)
    train_imgs = selected_images[:n_train]
    valid_imgs = selected_images[n_train:]

    # 폴더 생성
    train_class_path = os.path.join(train_dir, class_name)
    valid_class_path = os.path.join(valid_dir, class_name)
    os.makedirs(train_class_path, exist_ok=True)
    os.makedirs(valid_class_path, exist_ok=True)

    # 이미지 복사
    for img_name in train_imgs:
        shutil.copy(os.path.join(class_path, img_name), os.path.join(train_class_path, img_name))
    for img_name in valid_imgs:
        shutil.copy(os.path.join(class_path, img_name), os.path.join(valid_class_path, img_name))

    print(f"[{class_name}] Total: {len(selected_images)}, Train: {len(train_imgs)}, Valid: {len(valid_imgs)}")

print("\n✅ 모든 클래스별로 랜덤 100장 선택 후 train/valid 분할 완료!")
