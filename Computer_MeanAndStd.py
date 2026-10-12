import nibabel as nib
import numpy as np
import glob
import random

def intensity_stats(folder, sample_size=None, files_to_sample=None):
    means, stds, mins, maxs = [], [], [], []

    # 遍历文件夹中的文件（只取 *0000.nii.gz 文件）
    files = sorted(glob.glob(f"{folder}/*0000.nii.gz"))
    if files_to_sample:
        files = random.sample(files, files_to_sample)  # 随机选择部分文件

    for f in files:
        img = nib.load(f).get_fdata()
        img = img[np.isfinite(img)]  # 去除NaN和Inf

        mean_val = np.mean(img)
        std_val = np.std(img)
        min_val = np.min(img)
        max_val = np.max(img)

        means.append(mean_val)
        stds.append(std_val)
        mins.append(min_val)
        maxs.append(max_val)

        print(f"{f}: mean={mean_val:.2f}, std={std_val:.2f}, min={min_val:.2f}, max={max_val:.2f}")

    return means, stds, mins, maxs

# 数据路径（只保留新数据集）
new_dataset_path = r"/root/autodl-tmp/U-Mamba/data/nnUNet_raw/Dataset999_BraTS2024/imagesTs"

# 统计新数据集
new_means, new_stds, new_mins, new_maxs = intensity_stats(new_dataset_path)

# 计算整体均值和标准差
new_mean = np.mean(new_means)
new_std = np.mean(new_stds)

print("\n===== 数据集整体统计 =====")
print(f"均值范围: {min(new_means):.2f} ~ {max(new_means):.2f}, 平均值={new_mean:.2f}")
print(f"标准差范围: {min(new_stds):.2f} ~ {max(new_stds):.2f}, 平均标准差={new_std:.2f}")
print(f"强度最小值范围: {min(new_mins):.2f} ~ {max(new_mins):.2f}")
print(f"强度最大值范围: {min(new_maxs):.2f} ~ {max(new_maxs):.2f}")
