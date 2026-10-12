import os
import shutil
import glob
from tqdm import tqdm
import json

# 路径配置
train_dir = r"D:\Work\BraTS2024\unet\origin_data\MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth"
val_dir = r"D:\Work\BraTS2024\unet\origin_data\MICCAI2024-BraTS-GoAT-ValidationData\MICCAI2024-BraTS-GoAT-ValidationData"
target_dir = r"./data/nnUNet_raw/Dataset999_BraTS2024"

# 创建必要的文件夹
os.makedirs(os.path.join(target_dir, "imagesTr"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "labelsTr"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "imagesTs"), exist_ok=True)

# 获取一个患者文件夹样例并查看其内容
if os.path.exists(os.path.join(train_dir, "MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth")):
    # 如果有一个额外的嵌套层级
    train_dir = os.path.join(train_dir, "MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth")
    print(f"调整训练目录路径: {train_dir}")

# 检查验证目录结构
if os.path.exists(os.path.join(val_dir, "MICCAI2024-BraTS-GoAT-ValidationData")):
    # 如果有一个额外的嵌套层级
    val_dir = os.path.join(val_dir, "MICCAI2024-BraTS-GoAT-ValidationData")
    print(f"调整验证目录路径: {val_dir}")

# 列出患者文件夹
train_folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]
val_folders = [f for f in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, f))]

# 检查第一个患者文件夹的内容
if train_folders:
    sample_folder = train_folders[0]
    sample_path = os.path.join(train_dir, sample_folder)
    print(f"示例患者 {sample_folder} 的文件:")
    for f in os.listdir(sample_path):
        print(f"  {f}")

# 定义模态映射 - 根据BraTS命名约定
modality_map = {
    "t1n": "0000",  # T1 native
    "t1c": "0001",  # T1 contrast enhanced
    "t2w": "0002",  # T2-weighted
    "t2f": "0003"  # T2-FLAIR
}

# 处理训练数据
print("处理训练数据...")
num_processed = 0
for folder in tqdm(train_folders):
    case_id = folder  # 使用文件夹名作为case标识符
    folder_path = os.path.join(train_dir, folder)

    # 检查每个可能的模态文件
    for modality, mod_id in modality_map.items():
        source_file = os.path.join(folder_path, f"{folder}-{modality}.nii.gz")
        if os.path.exists(source_file):
            target_file = os.path.join(target_dir, "imagesTr", f"{case_id}_{mod_id}.nii.gz")
            shutil.copy(source_file, target_file)
            num_processed += 1

    # 检查分割文件
    seg_file = os.path.join(folder_path, f"{folder}-seg.nii.gz")
    if os.path.exists(seg_file):
        target_seg_file = os.path.join(target_dir, "labelsTr", f"{case_id}.nii.gz")
        shutil.copy(seg_file, target_seg_file)
        num_processed += 1

print(f"训练数据处理完成，共复制了 {num_processed} 个文件")

# 处理验证数据
print("处理验证数据...")
num_processed = 0
for folder in tqdm(val_folders):
    case_id = folder  # 使用文件夹名作为case标识符
    folder_path = os.path.join(val_dir, folder)

    # 检查每个可能的模态文件
    for modality, mod_id in modality_map.items():
        source_file = os.path.join(folder_path, f"{folder}-{modality}.nii.gz")
        if os.path.exists(source_file):
            target_file = os.path.join(target_dir, "imagesTs", f"{case_id}_{mod_id}.nii.gz")
            shutil.copy(source_file, target_file)
            num_processed += 1

print(f"验证数据处理完成，共复制了 {num_processed} 个文件")

# 创建dataset.json文件
print("创建dataset.json文件...")
dataset_json = {
    "channel_names": {
        "0": "t1n",
        "1": "t1c",
        "2": "t2w",
        "3": "t2f"
    },
    "labels": {
        "background": 0,
        "necrotic_tumor_core": 1,
        "edematous_tissue": 2,
        "enhancing_tumor": 3
    },
    "numTraining": len(train_folders),
    "file_ending": ".nii.gz"
}

with open(os.path.join(target_dir, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=4)

print("数据转换完成！")
