import os
import numpy as np
import SimpleITK as sitk
from multiprocessing import Pool
from tqdm import tqdm

def load_segmentation(file_path):
    """加载分割文件"""
    return sitk.ReadImage(file_path), sitk.GetArrayFromImage(sitk.ReadImage(file_path))

def process_case(case_id):
    """处理单个案例，对所有折的预测结果进行集成"""
    pred_arrays = []
    reference_image = None
    
    # 从每个折加载该案例的预测
    for fold in range(5):  # 假设有5个折
        pred_folder = f"/root/autodl-tmp/U-Mamba/data/nnUNet_results/Dataset999_BraTS2024/nnUNetTrainerUMambaEnc__nnUNetPlans__3d_fullres/output_predict_RHUH/{fold}"
        #pred_folder = f"/root/autodl-tmp/U-Mamba/data/nnUNet_results/Dataset999_BraTS2024/nnUNetTrainerUMambaEnc__nnUNetPlans__3d_fullres/output_predict_MET/{fold}"
        pred_file = os.path.join(pred_folder, case_id)
        
        if os.path.exists(pred_file):
            img, arr = load_segmentation(pred_file)
            if reference_image is None:
                reference_image = img  # 使用第一个图像作为参考
            pred_arrays.append(arr)
    
    if not pred_arrays:
        print(f"Warning: No predictions found for {case_id}")
        return None
    
    # 进行多数投票集成
    pred_arrays = np.array(pred_arrays)
    ensemble_result = np.zeros_like(pred_arrays[0])
    
    # 对每个体素进行投票
    for i in range(1, 4):  # 假设有三个类别：1, 2, 3
        votes = np.sum(pred_arrays == i, axis=0)
        ensemble_result[votes > len(pred_arrays) / 2] = i  # 多数投票
    
    # 创建输出图像
    result_image = sitk.GetImageFromArray(ensemble_result)
    result_image.CopyInformation(reference_image)  # 复制元数据
    
    # 保存结果
    output_folder = "/root/autodl-tmp/U-Mamba/data/nnUNet_results/Dataset999_BraTS2024/nnUNetTrainerUMambaEnc__nnUNetPlans__3d_fullres/output_predict_RHUH/manual_ensemble"
    #output_folder = "/root/autodl-tmp/U-Mamba/data/nnUNet_results/Dataset999_BraTS2024/nnUNetTrainerUMambaEnc__nnUNetPlans__3d_fullres/output_predict_MET/manual_ensemble"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, case_id)
    sitk.WriteImage(result_image, output_file)
    
    return case_id

def main():
    # 获取所有案例ID
    fold0_folder = "/root/autodl-tmp/U-Mamba/data/nnUNet_results/Dataset999_BraTS2024/nnUNetTrainerUMambaEnc__nnUNetPlans__3d_fullres/output_predict_RHUH/0"
    #fold0_folder = "/root/autodl-tmp/U-Mamba/data/nnUNet_results/Dataset999_BraTS2024/nnUNetTrainerUMambaEnc__nnUNetPlans__3d_fullres/output_predict_MET/0"
    case_ids = [f for f in os.listdir(fold0_folder) if f.endswith('.nii.gz')]
    
    print(f"Found {len(case_ids)} cases to process")
    
    # 并行处理所有案例
    with Pool(processes=8) as pool:
        list(tqdm(pool.imap(process_case, case_ids), total=len(case_ids)))
    
    print("Ensemble complete!")

if __name__ == "__main__":
    main()
