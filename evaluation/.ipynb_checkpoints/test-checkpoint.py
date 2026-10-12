import os
import json
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from medpy.metric import binary

def compute_dice(gt, pred):
    """计算Dice系数"""
    intersection = np.sum(gt * pred)
    return (2. * intersection) / (np.sum(gt) + np.sum(pred) + 1e-8)

def compute_hd95(gt, pred, voxel_spacing):
    """计算95%Hausdorff距离"""
    if np.sum(gt) == 0 or np.sum(pred) == 0:
        return float('nan')  # 如果任一掩码为空，返回NaN
    
    try:
        return binary.hd95(pred, gt, voxel_spacing)
    except Exception as e:
        print(f"Error computing HD95: {e}")
        return float('nan')

def main():
    gt_folder = "/root/autodl-tmp/U-Mamba/data/nnUNet_raw/Dataset999_BraTS2024/labelsTs"
    #pred_folder = "/root/autodl-tmp/U-Mamba/data/nnUNet_results/Dataset999_BraTS2024/nnUNetTrainerUMambaEnc__nnUNetPlans__3d_fullres/output_predict_RHUH/manual_ensemble"
    pred_folder = "/root/autodl-tmp/U-Mamba/data/nnUNet_results/Dataset999_BraTS2024/nnUNetTrainerUMambaEnc__nnUNetPlans__3d_fullres/output_predict_MET/advanced_domain_adaptation"
    
    # 获取所有预测文件
    pred_files = [f for f in os.listdir(pred_folder) if f.endswith('.nii.gz')]
    
    # 初始化结果
    results = {
        'ET_dice': [], 'TC_dice': [], 'WT_dice': [],
        'ET_hd95': [], 'TC_hd95': [], 'WT_hd95': []
    }
    
    # 处理每个文件
    for file in tqdm(pred_files):
        gt_path = os.path.join(gt_folder, file)
        pred_path = os.path.join(pred_folder, file)
        
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth file not found for {file}")
            continue
        
        # 加载真实标签和预测
        gt_img = sitk.ReadImage(gt_path)
        pred_img = sitk.ReadImage(pred_path)
        
        # 获取体素间距
        spacing = gt_img.GetSpacing()
        voxel_spacing = spacing[::-1]  # SimpleITK和NumPy的轴顺序相反
        
        gt = sitk.GetArrayFromImage(gt_img)
        pred = sitk.GetArrayFromImage(pred_img)
        
        # 计算复合区域的二值掩码
        # ET (Enhancing Tumor): 只包含标签3
        et_gt = (gt == 3).astype(np.uint8)
        et_pred = (pred == 3).astype(np.uint8)
        
        # TC (Tumor Core): 标签1 + 标签3
        tc_gt = ((gt == 1) | (gt == 3)).astype(np.uint8)
        tc_pred = ((pred == 1) | (pred == 3)).astype(np.uint8)
        
        # WT (Whole Tumor): 标签1 + 标签2 + 标签3
        wt_gt = ((gt == 1) | (gt == 2) | (gt == 3)).astype(np.uint8)
        wt_pred = ((pred == 1) | (pred == 2) | (pred == 3)).astype(np.uint8)
        
        # 计算Dice系数
        results['ET_dice'].append(compute_dice(et_gt, et_pred))
        results['TC_dice'].append(compute_dice(tc_gt, tc_pred))
        results['WT_dice'].append(compute_dice(wt_gt, wt_pred))
        
        # 计算HD95
        results['ET_hd95'].append(compute_hd95(et_gt, et_pred, voxel_spacing))
        results['TC_hd95'].append(compute_hd95(tc_gt, tc_pred, voxel_spacing))
        results['WT_hd95'].append(compute_hd95(wt_gt, wt_pred, voxel_spacing))
    
    # 计算平均值（忽略NaN值）
    mean_results = {}
    for metric in results:
        values = np.array(results[metric])
        mean_results[metric] = float(np.nanmean(values))
    
    # 打印结果
    print("\n===== BraTS Evaluation Results =====")
    print(f"Number of cases: {len(pred_files)}")
    
    print("\nMean Dice scores for composite regions:")
    print(f"ET (Enhancing Tumor): {mean_results['ET_dice']:.4f}")
    print(f"TC (Tumor Core): {mean_results['TC_dice']:.4f}")
    print(f"WT (Whole Tumor): {mean_results['WT_dice']:.4f}")
    print(f"Average Dice: {np.mean([mean_results['ET_dice'], mean_results['TC_dice'], mean_results['WT_dice']]):.4f}")
    
    print("\nMean HD95 for composite regions (mm):")
    print(f"ET (Enhancing Tumor): {mean_results['ET_hd95']:.4f}")
    print(f"TC (Tumor Core): {mean_results['TC_hd95']:.4f}")
    print(f"WT (Whole Tumor): {mean_results['WT_hd95']:.4f}")
    print(f"Average HD95: {np.mean([mean_results['ET_hd95'], mean_results['TC_hd95'], mean_results['WT_hd95']]):.4f}")
    
    # 保存结果
    output_file = os.path.join(pred_folder, "brats_metrics_with.json")
    with open(output_file, 'w') as f:
        json.dump({
            'case_results': {k: [float(x) if not np.isnan(x) else None for x in v] for k, v in results.items()},
            'mean': mean_results
        }, f, indent=4)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()