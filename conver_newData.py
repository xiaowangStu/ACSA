import os
import shutil
import glob
from tqdm import tqdm

# 源数据目录和目标目录
source_dir = r"/root/autodl-tmp/U-Mamba/ASNR-MICCAI-BraTS2023-MET-Challenge-TrainingData"  # 修改为您下载的数据集路径
target_dir = r"/root/autodl-tmp/U-Mamba/data/nnUNet_raw/Dataset999_BraTS2024"

# 确保目标目录存在
os.makedirs(os.path.join(target_dir, "imagesTs"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "labelsTs"), exist_ok=True)  # 如果需要存放分割标签

# 模态映射
modality_map = {
    "t1n": "0000",  # T1 native
    "t1c": "0001",  # T1 contrast enhanced
    "t2w": "0002",  # T2-weighted
    "t2f": "0003"  # T2-FLAIR
}

def process_patient_data(patient_folders):
    for patient_folder in tqdm(patient_folders):
        if not os.path.isdir(patient_folder):
            continue
            
        # 查找该文件夹下的所有nii文件
        nii_files = glob.glob(os.path.join(patient_folder, "*.nii*"))
        
        # 处理每个文件
        for nii_file in nii_files:
            filename = os.path.basename(nii_file)
            
            # 从文件名提取病人ID，例如从 BraTS-MET-00002-000-t1n.nii 提取 00002
            if "BraTS-MET-" in filename:
                parts = filename.split("-")
                if len(parts) >= 4:
                    # 提取病人编号并删除前导零，例如 00002 → 2
                    patient_num = str(int(parts[2]))
                    patient_id = f"BRATS_{patient_num}"
                    
                    # 处理分割文件
                    if "seg" in filename:
                        target_file = os.path.join(target_dir, "labelsTs", f"{patient_id}.nii.gz")
                        shutil.copy(nii_file, target_file)
                        print(f"Copied segmentation: {filename} -> {target_file}")
                    
                    # 处理模态文件
                    else:
                        for modality, suffix in modality_map.items():
                            if modality in filename:
                                target_file = os.path.join(target_dir, "imagesTs", f"{patient_id}_{suffix}.nii.gz")
                                shutil.copy(nii_file, target_file)
                                print(f"Copied {modality}: {filename} -> {target_file}")
                                break

# 获取所有病人文件夹
patient_folders = glob.glob(os.path.join(source_dir, "*"))
process_patient_data(patient_folders)

print("转换完成！")