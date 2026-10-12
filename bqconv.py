import os
import nibabel as nib
import numpy as np

# 设置路径：原始数据集路径和保存重映射结果的路径
source_dir = "/root/autodl-tmp/U-Mamba/data/nnUNet_raw/Dataset999_BraTS2024/labelsTs"  # 这里改成你的原始目录
output_dir = "/root/autodl-tmp/U-Mamba/data/nnUNet_raw/Dataset999_BraTS2024/labelsTs_"   # 这里改成你想保存处理后文件的目录

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 标签重映射函数
def remap_segmentation_labels(seg_path, output_path):
    # 加载 segmentation 文件
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()

    # 重映射标签
    seg_data[seg_data == 1] = 1   # 坏死 -> Necrosis
    seg_data[seg_data == 2] = 2   # 肿瘤周围区域 -> Peritumoral Region
    seg_data[seg_data == 4] = 3   # 增强肿瘤 -> Enhancing Tumor
    seg_data[seg_data == 0] = 0   # 背景标签（统一为 0）

    # 保存修改后的 segmentation 文件
    new_seg_img = nib.Nifti1Image(seg_data, seg_img.affine)
    nib.save(new_seg_img, output_path)

    print(f"Processed: {seg_path} -> {output_path}")

# 批量处理目录中的所有 segmentation.nii.gz 文件
for filename in os.listdir(source_dir):
    if filename.lower().endswith(".nii.gz"):  # 处理所有以 .nii.gz 结尾的文件
        seg_path = os.path.join(source_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 执行标签重映射
        remap_segmentation_labels(seg_path, output_path)

print("All segmentation files have been processed.")
