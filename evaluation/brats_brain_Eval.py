# brats_evaluation.py
import numpy as np
import SimpleITK as sitk
import os
import json
from multiprocessing import Pool
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance

def load_segmentation(filename):
    return sitk.ReadImage(filename)

def compute_dice(gt, pred, label):
    """计算特定标签的Dice系数"""
    gt_binary = (gt == label).astype(np.float32)
    pred_binary = (pred == label).astype(np.float32)
    
    intersection = np.sum(gt_binary * pred_binary)
    return (2. * intersection) / (np.sum(gt_binary) + np.sum(pred_binary) + 1e-8)

def compute_hd95(gt, pred, label, spacing):
    """计算特定标签的95%Hausdorff距离"""
    gt_binary = (gt == label).astype(np.bool)
    pred_binary = (pred == label).astype(np.bool)
    
    surface_distances = compute_surface_distances(gt_binary, pred_binary, spacing)
    hd95 = np.percentile(surface_distances['distances_gt_to_pred'], 95)
    
    return hd95

def compute_brats_metrics(gt_npy, pred_npy, spacing):
    """计算BraTS的复合区域指标"""
    # 计算各个标签的原始指标
    metrics = {}
    
    # 标签1：坏死肿瘤核心
    # 标签2：水肿组织
    # 标签3：增强肿瘤
    for label in [1, 2, 3]:
        metrics[f'label_{label}_dice'] = compute_dice(gt_npy, pred_npy, label)
        metrics[f'label_{label}_hd95'] = compute_hd95(gt_npy, pred_npy, label, spacing)
    
    # 计算复合区域
    # ET (Enhancing Tumor): 只包含标签3
    et_gt = (gt_npy == 3).astype(np.float32)
    et_pred = (pred_npy == 3).astype(np.float32)
    
    # TC (Tumor Core): 标签1 + 标签3
    tc_gt = ((gt_npy == 1) | (gt_npy == 3)).astype(np.float32)
    tc_pred = ((pred_npy == 1) | (pred_npy == 3)).astype(np.float32)
    
    # WT (Whole Tumor): 标签1 + 标签2 + 标签3
    wt_gt = ((gt_npy == 1) | (gt_npy == 2) | (gt_npy == 3)).astype(np.float32)
    wt_pred = ((pred_npy == 1) | (pred_npy == 2) | (pred_npy == 3)).astype(np.float32)
    
    # 计算复合区域的Dice和HD95
    metrics['ET_dice'] = (2. * np.sum(et_gt * et_pred)) / (np.sum(et_gt) + np.sum(et_pred) + 1e-8)
    metrics['TC_dice'] = (2. * np.sum(tc_gt * tc_pred)) / (np.sum(tc_gt) + np.sum(tc_pred) + 1e-8)
    metrics['WT_dice'] = (2. * np.sum(wt_gt * wt_pred)) / (np.sum(wt_gt) + np.sum(wt_pred) + 1e-8)
    
    # 计算复合区域的HD95
    # 这里需要额外实现或使用库来计算HD95
    
    return metrics

def evaluate_case(args):
    gt_file, pred_file = args
    print(f"Processing {os.path.basename(gt_file)}")
    
    gt_img = load_segmentation(gt_file)
    pred_img = load_segmentation(pred_file)
    
    spacing = gt_img.GetSpacing()
    
    gt_npy = sitk.GetArrayFromImage(gt_img)
    pred_npy = sitk.GetArrayFromImage(pred_img)
    
    return compute_brats_metrics(gt_npy, pred_npy, spacing)

def main():
    gt_folder = "/root/autodl-tmp/U-Mamba/data/nnUNet_raw/Dataset999_BraTS2024/labelsTs"
    pred_folder = "/root/autodl-tmp/U-Mamba/data/nnUNet_results/Dataset999_BraTS2024/nnUNetTrainerUMambaEnc__nnUNetPlans__3d_fullres/output_predict/manual_ensemble"
    output_file = os.path.join(pred_folder, "brats_metrics.json")
    
    # 获取所有预测文件和对应的真实标签文件
    pred_files = [os.path.join(pred_folder, f) for f in os.listdir(pred_folder) if f.endswith('.nii.gz')]
    gt_files = []
    
    for pred_file in pred_files:
        base_name = os.path.basename(pred_file)
        gt_file = os.path.join(gt_folder, base_name)
        if os.path.exists(gt_file):
            gt_files.append(gt_file)
        else:
            print(f"Warning: Ground truth file not found for {base_name}")
    
    # 确保文件数量匹配
    assert len(gt_files) == len(pred_files), "Number of ground truth and prediction files don't match"
    
    # 并行处理所有案例
    with Pool(processes=8) as pool:
        results = pool.map(evaluate_case, zip(gt_files, pred_files))
    
    # 计算平均指标
    avg_metrics = {}
    for metric in results[0].keys():
        avg_metrics[metric] = np.mean([r[metric] for r in results])
    
    # 打印结果
    print("===== BraTS Evaluation Results =====")
    print(f"Number of cases: {len(results)}")
    print("\nIndividual Regions:")
    for label in [1, 2, 3]:
        print(f"  Label {label} - Dice: {avg_metrics[f'label_{label}_dice']:.4f}, HD95: {avg_metrics[f'label_{label}_hd95']:.4f} mm")
    
    print("\nComposite Regions:")
    print(f"  ET - Dice: {avg_metrics['ET_dice']:.4f}")
    print(f"  TC - Dice: {avg_metrics['TC_dice']:.4f}")
    print(f"  WT - Dice: {avg_metrics['WT_dice']:.4f}")
    
    # 保存结果
    with open(output_file, 'w') as f:
        json.dump({"case_results": results, "average": avg_metrics}, f, indent=4)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
