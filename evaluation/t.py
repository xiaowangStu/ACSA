import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# 加载.nii.gz文件
file_path = "/root/autodl-tmp/U-Mamba/data/nnUNet_results/Dataset999_BraTS2024/nnUNetTrainerUMambaEnc__nnUNetPlans__3d_fullres/output_predict/manual_ensemble/BraTS-GoAT-02483.nii.gz"
img = nib.load(file_path)

# 获取数据
data = img.get_fdata()
print(f"数据形状: {data.shape}")
print(f"数据类型: {data.dtype}")

# 假设是3D体积数据，可以查看不同切片
# 选择中间切片
if len(data.shape) >= 3:
    # 沿着不同维度显示中间切片
    mid_x = data.shape[0] // 2
    mid_y = data.shape[1] // 2
    mid_z = data.shape[2] // 2
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(data[mid_x, :, :], cmap='gray')
    plt.title(f'Sagittal Slice (x={mid_x})')
    plt.colorbar()
    
    plt.subplot(132)
    plt.imshow(data[:, mid_y, :], cmap='gray')
    plt.title(f'Coronal Slice (y={mid_y})')
    plt.colorbar()
    
    plt.subplot(133)
    plt.imshow(data[:, :, mid_z], cmap='gray')
    plt.title(f'Axial Slice (z={mid_z})')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
else:
    # 如果是2D数据
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='gray')
    plt.colorbar()
    plt.title('2D Image')
    plt.show()
