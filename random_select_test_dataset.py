# train, valid 데이터와 중복되지 않게 클래스당 20개 이미지 추출

import os
import shutil
import random

# 경로 설정
base_dir = r"C:\AlexNet_chlee\PokemonDataset"
split_dir = r"C:\AlexNet_chlee\PokemonDataset_Split"

train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")
test_dir = os.path.join(split_dir, "test")

os.makedirs(test_dir, exist_ok=True)

# train + valid 에 사용된 파일 모으기
used_files = set()

for split in ["train", "valid"]:
    split_path = os.path.join(split_dir, split)
    for cls in os.listdir(split_path):
        cls_path = os.path.join(split_path, cls)
        if not os.path.isdir(cls_path):
            continue
        for f in os.listdir(cls_path):
            used_files.add(f)   # 파일 이름만 기록 (클래스마다 고유하다고 가정)

# test set 추출
for cls in os.listdir(base_dir):
    cls_path = os.path.join(base_dir, cls)
    if not os.path.isdir(cls_path):
        continue

    # 클래스별 저장할 test 폴더 생성
    cls_test_path = os.path.join(test_dir, cls)
    os.makedirs(cls_test_path, exist_ok=True)

    # 이 클래스에 있는 전체 이미지 파일들
    all_imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # train/valid 에 이미 들어간 파일 제외
    remaining_imgs = [f for f in all_imgs if f not in used_files]

    # test용으로 20장 뽑기
    if len(remaining_imgs) < 20:
        print(f"[WARNING] {cls}: 남은 이미지 {len(remaining_imgs)}장밖에 없음. 모두 test로 복사함.")
        selected_imgs = remaining_imgs
    else:
        selected_imgs = random.sample(remaining_imgs, 20)

    # 선택한 이미지를 test 폴더로 복사
    for img in selected_imgs:
        src = os.path.join(cls_path, img)
        dst = os.path.join(cls_test_path, img)
        shutil.copy2(src, dst)

    print(f"{cls}: test set {len(selected_imgs)}장 추출 완료.")
