"""
多尺度特征融合与注意力重校准策略
针对跨域泛化的深度学习优化方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
import os
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

class MultiScaleFeatureFusion:
    """多尺度特征融合处理器"""
    
    def __init__(self, scales=[0.8, 1.0, 1.2, 1.4], safe_mode=True):
        self.scales = scales
        self.safe_mode = safe_mode  # 安全模式：如果多尺度失败就退化到单一尺度
    
    def multi_scale_inference(self, predictor, input_data, properties):
        """多尺度推理并融合结果"""
        predictions = []
        
        for i, scale in enumerate(self.scales):
            print(f"  Processing scale {i+1}/{len(self.scales)}: {scale}x")
            
            try:
                # 缩放输入数据
                if scale != 1.0:
                    scaled_data = self._scale_data(input_data, scale)
                else:
                    scaled_data = input_data
                
                # 预测（加强的错误处理）
                try:
                    # 确保数据类型正确
                    if scaled_data.dtype != np.float32:
                        scaled_data = scaled_data.astype(np.float32)
                    
                    pred = predictor.predict_single_npy_array(
                        scaled_data, properties, None, None, False
                    )
                except RuntimeError as e:
                    if "Input type" in str(e) and "weight type" in str(e):
                        print(f"    Data type error, trying to fix: {e}")
                        # 尝试强制转换数据类型
                        scaled_data = scaled_data.astype(np.float32)
                        try:
                            pred = predictor.predict_single_npy_array(
                                scaled_data, properties, None, None, False
                            )
                        except Exception as e2:
                            print(f"    Still failed after type conversion: {e2}")
                            raise e2
                    else:
                        raise e
                
                print(f"    Prediction shape after inference: {pred.shape}")
                
                # 如果进行了缩放，需要将预测结果缩放回原尺寸
                if scale != 1.0:
                    try:
                        pred = self._scale_prediction_back(pred, input_data.shape[1:])
                        print(f"    Prediction shape after scaling back: {pred.shape}")
                    except Exception as e:
                        print(f"    WARNING: Scaling back failed for scale {scale}: {e}")
                        print(f"    Prediction shape: {pred.shape}, Target shape: {input_data.shape[1:]}")
                        # 如果缩放失败，跳过这个尺度
                        continue
                
                predictions.append(pred)
                
                # 清理内存
                del pred, scaled_data
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"    ERROR: Scale {scale} failed completely: {e}")
                continue
        
        if not predictions:
            if self.safe_mode:
                print("    WARNING: 多尺度推理失败，退化到单一尺度")
                # 退化到单一尺度（原尺寸）
                try:
                    # 确保数据类型正确
                    if input_data.dtype != np.float32:
                        input_data = input_data.astype(np.float32)
                    
                    pred = predictor.predict_single_npy_array(
                        input_data, properties, None, None, False
                    )
                    return pred
                except Exception as e:
                    print(f"    ERROR: 单一尺度也失败了: {e}")
                    raise e
            else:
                raise RuntimeError("所有尺度的推理都失败了")
        
        # 融合多尺度预测结果
        fused_prediction = self._fuse_predictions(predictions)
        return fused_prediction
    
    def _scale_data(self, data, scale):
        """缩放输入数据"""
        # 确保数据类型是float32
        if data.dtype != np.float32:
            print(f"    Converting data type from {data.dtype} to float32")
            data = data.astype(np.float32)
        
        # data shape: [C, D, H, W]
        c, d, h, w = data.shape
        new_d, new_h, new_w = int(d * scale), int(h * scale), int(w * scale)
        
        print(f"DEBUG: _scale_data - original shape: {data.shape}, scale: {scale}, new shape: ({new_d}, {new_h}, {new_w})")
        
        # 使用PyTorch进行插值
        data_tensor = torch.from_numpy(data).unsqueeze(0).float()  # [1, C, D, H, W]
        
        # 检查输入张量的维度
        if data_tensor.dim() != 5:
            raise ValueError(f"Data tensor must be 5D [B,C,D,H,W], got {data_tensor.dim()}D: {data_tensor.shape}")
        
        scaled_tensor = F.interpolate(
            data_tensor, 
            size=(new_d, new_h, new_w), 
            mode='trilinear', 
            align_corners=False
        )
        
        result = scaled_tensor.squeeze(0).numpy().astype(np.float32)
        print(f"    Scaled data shape: {result.shape}, dtype: {result.dtype}")
        return result
    
    def _scale_prediction_back(self, prediction, target_shape):
        """将预测结果缩放回原尺寸"""
        print(f"DEBUG: _scale_prediction_back - input prediction shape: {prediction.shape}")
        
        # 检查预测结果的维度
        if len(prediction.shape) == 3:
            # 如果是3D，说明缺少通道维度，需要添加
            print(f"    WARNING: Prediction is 3D, missing channel dimension. Adding channel dimension...")
            # 假设是4类别（背景 + 3个肿瘤类别）
            # 使用argmax逆推出概率
            num_classes = 4
            one_hot = np.zeros((num_classes,) + prediction.shape)
            for c in range(num_classes):
                one_hot[c] = (prediction == c).astype(np.float32)
            prediction = one_hot
            print(f"    Converted to 4D shape: {prediction.shape}")
        elif len(prediction.shape) != 4:
            raise ValueError(f"Prediction must be 3D or 4D, got {len(prediction.shape)}D: {prediction.shape}")
        
        # prediction shape: [num_classes, D, H, W]
        pred_tensor = torch.from_numpy(prediction).unsqueeze(0).float()  # [1, C, D, H, W]
        
        # 确保target_shape是3D
        if len(target_shape) != 3:
            raise ValueError(f"Target shape must be 3D, got {len(target_shape)}D: {target_shape}")
        
        # 检查输入张量的维度
        if pred_tensor.dim() != 5:  # 应该是 [batch, channels, depth, height, width]
            raise ValueError(f"Prediction tensor must be 5D [B,C,D,H,W], got {pred_tensor.dim()}D: {pred_tensor.shape}")
        
        print(f"DEBUG: _scale_prediction_back - pred_tensor shape: {pred_tensor.shape}, target_shape: {target_shape}")
        
        scaled_back = F.interpolate(
            pred_tensor,
            size=target_shape,
            mode='trilinear',
            align_corners=False
        )
        
        return scaled_back.squeeze(0).numpy()
    
    def _fuse_predictions(self, predictions):
        """融合多个预测结果"""
        if not predictions:
            raise ValueError("没有预测结果可以融合")
        
        if len(predictions) == 1:
            return predictions[0]
        
        # 使用加权平均，给不同尺度不同权重
        # 根据实际的predictions数量动态调整权重
        num_preds = len(predictions)
        if num_preds == 4:
            weights = [0.2, 0.4, 0.3, 0.1]  # 原尺寸权重最高
        elif num_preds == 3:
            weights = [0.25, 0.5, 0.25]
        elif num_preds == 2:
            weights = [0.4, 0.6]
        else:
            # 如果数量不匹配，使用平均权重
            weights = [1.0/num_preds] * num_preds
        
        print(f"    Fusing {num_preds} predictions with weights: {weights}")
        
        fused = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            fused += weights[i] * pred
        
        return fused

class AttentionRecalibration:
    """注意力重校准处理器"""
    
    def __init__(self, temperature=1.0):
        self.temperature = temperature
    
    def spatial_attention_recalibration(self, prediction_probs, original_images):
        """空间注意力重校准"""
        # 基于T1CE图像计算空间注意力
        t1ce = original_images[1]  # T1CE通道
        
        # 计算空间注意力权重
        attention_weights = self._compute_spatial_attention(t1ce)
        
        # 应用注意力权重到预测概率
        recalibrated_probs = prediction_probs.copy()
        
        # 对每个类别应用空间注意力
        for class_idx in range(prediction_probs.shape[0]):
            if class_idx == 0:  # 背景类别，反向应用注意力
                recalibrated_probs[class_idx] *= (1.0 - attention_weights)
            else:  # 肿瘤类别，正向应用注意力
                recalibrated_probs[class_idx] *= (1.0 + attention_weights)
        
        # 重新归一化
        prob_sum = np.sum(recalibrated_probs, axis=0, keepdims=True)
        prob_sum[prob_sum == 0] = 1  # 避免除零
        recalibrated_probs = recalibrated_probs / prob_sum
        
        return recalibrated_probs
    
    def _compute_spatial_attention(self, t1ce_image):
        """计算基于T1CE的空间注意力"""
        # 归一化
        t1ce_norm = (t1ce_image - t1ce_image.min()) / (t1ce_image.max() - t1ce_image.min() + 1e-8)
        
        # 计算局部方差作为注意力权重
        from scipy import ndimage
        
        # 使用3x3x3窗口计算局部方差
        kernel = np.ones((3, 3, 3)) / 27
        local_mean = ndimage.convolve(t1ce_norm, kernel, mode='reflect')
        local_var = ndimage.convolve((t1ce_norm - local_mean)**2, kernel, mode='reflect')
        
        # 归一化方差到0-1
        attention = (local_var - local_var.min()) / (local_var.max() - local_var.min() + 1e-8)
        
        return attention
    
    def channel_attention_recalibration(self, prediction_probs):
        """通道注意力重校准"""
        # 计算每个通道的全局统计信息
        channel_stats = []
        for c in range(prediction_probs.shape[0]):
            channel_prob = prediction_probs[c]
            mean_val = np.mean(channel_prob)
            std_val = np.std(channel_prob)
            max_val = np.max(channel_prob)
            
            channel_stats.append({
                'mean': mean_val,
                'std': std_val,
                'max': max_val,
                'entropy': self._compute_entropy(channel_prob)
            })
        
        # 基于统计信息计算通道权重
        channel_weights = self._compute_channel_weights(channel_stats)
        
        # 应用通道权重
        recalibrated_probs = prediction_probs.copy()
        for c in range(prediction_probs.shape[0]):
            recalibrated_probs[c] *= channel_weights[c]
        
        # 重新归一化
        prob_sum = np.sum(recalibrated_probs, axis=0, keepdims=True)
        prob_sum[prob_sum == 0] = 1
        recalibrated_probs = recalibrated_probs / prob_sum
        
        return recalibrated_probs
    
    def _compute_entropy(self, prob_map):
        """计算概率图的熵"""
        # 避免log(0)
        prob_map_safe = np.clip(prob_map, 1e-8, 1.0)
        entropy = -np.sum(prob_map_safe * np.log(prob_map_safe))
        return entropy
    
    def _compute_channel_weights(self, channel_stats):
        """基于通道统计信息计算权重"""
        weights = []
        
        for i, stats in enumerate(channel_stats):
            if i == 0:  # 背景通道
                # 背景通道权重基于熵，熵越高权重越低
                weight = 1.0 / (1.0 + stats['entropy'] / 100)
            else:  # 肿瘤通道
                # 肿瘤通道权重基于最大值和标准差
                weight = stats['max'] * (1.0 + stats['std'])
            
            weights.append(weight)
        
        # 归一化权重
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum * len(weights) for w in weights]
        else:
            weights = [1.0] * len(weights)
        
        return weights

class UncertaintyAwareEnsemble:
    """不确定性感知集成"""
    
    def __init__(self, num_monte_carlo=10):
        self.num_monte_carlo = num_monte_carlo
    
    def monte_carlo_inference(self, predictor, input_data, properties):
        """蒙特卡罗推理获取不确定性估计"""
        predictions = []
        
        for i in range(self.num_monte_carlo):
            try:
                # 确保输入数据类型正确
                if input_data.dtype != np.float32:
                    input_data = input_data.astype(np.float32)
                
                # 添加轻微的随机扰动
                noise_scale = 0.01
                noise = np.random.normal(0, noise_scale, input_data.shape).astype(np.float32)
                noisy_input = (input_data + noise).astype(np.float32)
                
                pred = predictor.predict_single_npy_array(
                    noisy_input, properties, None, None, False
                )
                predictions.append(pred)
                
                # 清理内存
                del noisy_input, noise
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "Input type" in str(e) and "weight type" in str(e):
                    print(f"    Monte Carlo iteration {i+1} failed due to data type issue: {e}")
                    # 跳过这次迭代
                    continue
                else:
                    print(f"    Monte Carlo iteration {i+1} failed: {e}")
                    continue
            except Exception as e:
                print(f"    Monte Carlo iteration {i+1} failed: {e}")
                continue
        
        if not predictions:
            print("    WARNING: 所有Monte Carlo迭代都失败了，返回零值")
            # 返回零值作为默认值
            dummy_shape = (4,) + input_data.shape[1:]  # 4个类别
            return np.zeros(dummy_shape, dtype=np.float32), np.ones(dummy_shape, dtype=np.float32)
        
        # 计算平均预测和不确定性
        mean_prediction = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_prediction, uncertainty
    
    def uncertainty_weighted_ensemble(self, predictions, uncertainties):
        """基于不确定性的加权集成"""
        # 不确定性越低，权重越高
        weights = 1.0 / (uncertainties + 1e-8)
        
        # 归一化权重
        weight_sum = np.sum(weights, axis=0, keepdims=True)
        weight_sum[weight_sum == 0] = 1
        normalized_weights = weights / weight_sum
        
        # 加权平均
        weighted_prediction = np.sum(predictions * normalized_weights, axis=0)
        
        return weighted_prediction

class AdvancedDomainAdaptationPredictor:
    """高级域适应预测器"""
    
    def __init__(self, model_folder, device='cuda'):
        self.device = device
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device(device),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        self.predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=(0, 1, 2, 3, 4),
            checkpoint_name='checkpoint_final.pth'
        )
        
        self.multi_scale_fusion = MultiScaleFeatureFusion(scales=[1.0], safe_mode=True)  # 首先只用单一尺度
        self.attention_recalibration = AttentionRecalibration()
        self.uncertainty_ensemble = UncertaintyAwareEnsemble()
    
    def advanced_predict(self, input_folder, output_folder, use_uncertainty=True, enable_multiscale=False):
        """使用高级域适应策略进行预测"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 动态设置多尺度
        if enable_multiscale:
            print("  启用多尺度融合")
            self.multi_scale_fusion.scales = [0.9, 1.0, 1.1]  # 使用保守的尺度范围
        else:
            print("  使用单一尺度 + 注意力机制")
            self.multi_scale_fusion.scales = [1.0]
        
        # 获取输入文件
        input_files = [f for f in os.listdir(input_folder) if f.endswith('.nii.gz')]
        
        # 提取case_ids - 适应不同的命名格式
        case_ids = set()
        for f in input_files:
            # 移除模态后缀，提取case ID
            if '_0000.nii.gz' in f:
                case_id = f.replace('_0000.nii.gz', '')
            elif '_0001.nii.gz' in f:
                case_id = f.replace('_0001.nii.gz', '')
            elif '_0002.nii.gz' in f:
                case_id = f.replace('_0002.nii.gz', '')
            elif '_0003.nii.gz' in f:
                case_id = f.replace('_0003.nii.gz', '')
            else:
                # 如果不符合标准格式，尝试其他模式
                base_name = f.replace('.nii.gz', '')
                if base_name.endswith(('_t1', '_t1ce', '_t2', '_flair')):
                    case_id = '_'.join(base_name.split('_')[:-1])
                else:
                    continue
            case_ids.add(case_id)
        
        case_ids = list(case_ids)
        print(f"发现 {len(case_ids)} 个案例: {case_ids[:5]}{'...' if len(case_ids) > 5 else ''}")
        
        for case_id in case_ids:
            print(f"Processing {case_id} with advanced domain adaptation...")
            
            # 尝试不同的文件命名模式
            input_files_case = []
            
            # 模式1: standard nnUNet format
            for modality in ['0000', '0001', '0002', '0003']:
                file_path = os.path.join(input_folder, f"{case_id}_{modality}.nii.gz")
                if os.path.exists(file_path):
                    input_files_case.append(file_path)
            
            # 如果模式1不成功，尝试模式2: BraTS format
            if len(input_files_case) != 4:
                input_files_case = []
                for modality in ['t1', 't1ce', 't2', 'flair']:
                    file_path = os.path.join(input_folder, f"{case_id}_{modality}.nii.gz")
                    if os.path.exists(file_path):
                        input_files_case.append(file_path)
            
            # 如果还是不成功，尝试模式3: 直接搜索包含case_id的文件
            if len(input_files_case) != 4:
                input_files_case = []
                matching_files = [f for f in input_files if case_id in f]
                matching_files.sort()  # 确保顺序一致
                for file_name in matching_files:
                    file_path = os.path.join(input_folder, file_name)
                    input_files_case.append(file_path)
            
            if len(input_files_case) != 4:
                print(f"Warning: 找到 {len(input_files_case)} 个模态文件，期望4个模态 for {case_id}")
                print(f"找到的文件: {[os.path.basename(f) for f in input_files_case]}")
                continue
            
            # 加载原始图像
            original_images = []
            for file_path in input_files_case:
                img = sitk.ReadImage(file_path)
                img_array = sitk.GetArrayFromImage(img)
                # 确保数据类型是float32
                img_array = img_array.astype(np.float32)
                original_images.append(img_array)
            original_images = np.stack(original_images, axis=0)
            
            print(f"  Input shape: {original_images.shape}, dtype: {original_images.dtype}")
            
            # 1. 多尺度特征融合预测
            properties = {"spacing": [1.0, 1.0, 1.0]}
            multi_scale_pred = self.multi_scale_fusion.multi_scale_inference(
                self.predictor, original_images, properties
            )
            
            # 2. 空间注意力重校准
            spatial_recalibrated = self.attention_recalibration.spatial_attention_recalibration(
                multi_scale_pred, original_images
            )
            
            # 3. 通道注意力重校准
            channel_recalibrated = self.attention_recalibration.channel_attention_recalibration(
                spatial_recalibrated
            )
            
            # 4. 不确定性感知推理（可选）
            if use_uncertainty:
                try:
                    print("  Applying uncertainty-aware inference...")
                    mc_pred, uncertainty = self.uncertainty_ensemble.monte_carlo_inference(
                        self.predictor, original_images, properties
                    )
                    
                    # 融合确定性预测和不确定性感知预测
                    final_pred = 0.7 * channel_recalibrated + 0.3 * mc_pred
                    print("  Uncertainty-aware inference completed.")
                except Exception as e:
                    print(f"  WARNING: Uncertainty-aware inference failed: {e}")
                    print("  Falling back to deterministic prediction.")
                    final_pred = channel_recalibrated
            else:
                final_pred = channel_recalibrated
            
            # 转换为标签
            prediction = np.argmax(final_pred, axis=0).astype(np.uint8)
            
            # 保存结果
            output_path = os.path.join(output_folder, f"{case_id}.nii.gz")
            
            # 读取原始图像以获取正确的空间信息
            reference_img = sitk.ReadImage(input_files_case[0])
            result_img = sitk.GetImageFromArray(prediction)
            result_img.CopyInformation(reference_img)
            sitk.WriteImage(result_img, output_path)
            
            print(f"Saved advanced prediction to {output_path}")

def main():
    """主函数"""
    model_folder = "/root/autodl-tmp/U-Mamba/data/nnUNet_results/Dataset999_BraTS2024/nnUNetTrainerUMambaEnc__nnUNetPlans__3d_fullres"
    input_folder = "/root/autodl-tmp/U-Mamba/data/nnUNet_raw/Dataset999_BraTS2024/imagesTs"
    output_folder = "/root/autodl-tmp/U-Mamba/data/nnUNet_results/Dataset999_BraTS2024/nnUNetTrainerUMambaEnc__nnUNetPlans__3d_fullres/output_predict_MET/advanced_domain_adaptation"
    
    predictor = AdvancedDomainAdaptationPredictor(model_folder)
    predictor.advanced_predict(input_folder, output_folder, use_uncertainty=False, enable_multiscale=True)

if __name__ == "__main__":
    main()