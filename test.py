import nibabel as nib
import numpy as np
import glob

def intensity_stats(folder):
    for f in sorted(glob.glob(f"{folder}/*0000.nii.gz")):
        img = nib.load(f).get_fdata()
        print(f"{f}: mean={np.mean(img):.2f}, std={np.std(img):.2f}, min={np.min(img):.2f}, max={np.max(img):.2f}")

# 旧数据集
intensity_stats(r"/root/autodl-tmp/U-Mamba/data/nnUNet_raw/Dataset999_BraTS2024/imagesTs_24")
# 新数据集
intensity_stats(r"/root/autodl-tmp/U-Mamba/data/nnUNet_raw/Dataset999_BraTS2024/imagesTs")