from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from numpy import number


class ImageNormalization(ABC):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = None

    def __init__(self, use_mask_for_norm: bool = None, intensityproperties: dict = None,
                 target_dtype: Type[number] = np.float32):
        assert use_mask_for_norm is None or isinstance(use_mask_for_norm, bool)
        self.use_mask_for_norm = use_mask_for_norm
        assert isinstance(intensityproperties, dict)
        self.intensityproperties = intensityproperties
        self.target_dtype = target_dtype

    @abstractmethod
    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        Image and seg must have the same shape. Seg is not always used
        """
        pass


class ZScoreNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True

    def __init__(self, use_mask_for_norm: bool = None, intensityproperties: dict = None,
                 target_dtype: Type[number] = np.float32, skip_if_normalized: bool = True):
        super().__init__(use_mask_for_norm, intensityproperties, target_dtype)
        self.skip_if_normalized = skip_if_normalized

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        import os
        
        image = image.astype(self.target_dtype)
        
        # 检查环境变量是否强制跳过标准化
        force_skip = os.environ.get('NNUNET_SKIP_NORMALIZATION', '').lower() in ('true', '1', 't', 'yes')
        
        if force_skip:
            print(f"[nnUNet] 强制跳过标准化 - 环境变量 NNUNET_SKIP_NORMALIZATION=true")
            return image
            
        # 智能检测是否已经标准化
        if self.skip_if_normalized and self._is_already_normalized(image, seg):
            if self.use_mask_for_norm is not None and self.use_mask_for_norm and seg is not None:
                mask = seg >= 0
                if np.sum(mask) > 0:
                    mean_val = float(image[mask].mean())
                    std_val = float(image[mask].std())
                else:
                    mean_val = float(image.mean())
                    std_val = float(image.std())
            else:
                mean_val = float(image.mean())
                std_val = float(image.std())
            print(f"[nnUNet] 跳过标准化 - 数据似乎已标准化: mean={mean_val:.3f}, std={std_val:.3f}")
            return image
            
        # 执行标准化
        if self.use_mask_for_norm is not None and self.use_mask_for_norm:
            # negative values in the segmentation encode the 'outside' region (think zero values around the brain as
            # in BraTS). We want to run the normalization only in the brain region, so we need to mask the image.
            # The default nnU-net sets use_mask_for_norm to True if cropping to the nonzero region substantially
            # reduced the image size.
            mask = seg >= 0
            mean = image[mask].mean()
            std = image[mask].std()
            image[mask] = (image[mask] - mean) / (max(std, 1e-8))
            print(f"[nnUNet] 标准化完成(带掩码): 原始 mean={mean:.3f}, std={std:.3f}")
        else:
            mean = image.mean()
            std = image.std()
            image = (image - mean) / (max(std, 1e-8))
            print(f"[nnUNet] 标准化完成: 原始 mean={mean:.3f}, std={std:.3f}")
        return image
        
    def _is_already_normalized(self, image: np.ndarray, seg: np.ndarray = None) -> bool:
        """
        智能检测数据是否已经标准化
        基于用户数据集的实际统计特征：
        - 均值范围: -10.811 ~ -2.987
        - 标准差范围: 1.359 ~ 3.397
        """
        if self.use_mask_for_norm is not None and self.use_mask_for_norm and seg is not None:
            mask = seg >= 0
            if np.sum(mask) == 0:
                return False
            mean_val = float(image[mask].mean())
            std_val = float(image[mask].std())
            data_min = float(image[mask].min())
            data_max = float(image[mask].max())
        else:
            mean_val = float(image.mean())
            std_val = float(image.std())
            data_min = float(image.min())
            data_max = float(image.max())
        
        # 基于实际数据特征的检测条件
        # 1. 均值在官方标准化后的范围内（扩展一些容差）
        mean_in_normalized_range = -15 < mean_val < 0  # 比实际范围稍微放宽
        
        # 2. 标准差在合理范围内（标准化后的典型范围）
        std_in_reasonable_range = 1.0 < std_val < 5.0  # 比实际范围稍微放宽
        
        # 3. 数据不是原始医学图像的大范围（原始CT通常是-1024到几千）
        not_raw_medical_range = (data_max - data_min) < 50  # 标准化后范围相对较小
        
        # 4. 排除明显的原始数据特征
        # 原始医学图像通常有很大的动态范围和特定的数值特征
        likely_raw_data = (
            (data_max - data_min) > 1000 or  # 范围太大
            abs(mean_val) > 100 or           # 均值太大
            data_max > 1000 or               # 最大值太大（原始CT等）
            data_min < -1000                 # 最小值太小（原始CT等）
        )
        
        already_normalized = (
            mean_in_normalized_range and 
            std_in_reasonable_range and 
            not_raw_medical_range and 
            not likely_raw_data
        )
        
        # 调试信息（可以取消注释来查看检测过程）
        # print(f"[调试] mean={mean_val:.3f}, std={std_val:.3f}, range=[{data_min:.1f}, {data_max:.1f}]")
        # print(f"[调试] mean_ok={mean_in_normalized_range}, std_ok={std_in_reasonable_range}, range_ok={not_raw_medical_range}, not_raw={not likely_raw_data}")
        
        return already_normalized


class CTNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"
        image = image.astype(self.target_dtype)
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['std']
        lower_bound = self.intensityproperties['percentile_00_5']
        upper_bound = self.intensityproperties['percentile_99_5']
        image = np.clip(image, lower_bound, upper_bound)
        image = (image - mean_intensity) / max(std_intensity, 1e-8)
        return image


class NoNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        return image.astype(self.target_dtype)


class RescaleTo01Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype)
        image = image - image.min()
        image = image / np.clip(image.max(), a_min=1e-8, a_max=None)
        return image


class RGBTo01Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert image.min() >= 0, "RGB images are uint 8, for whatever reason I found pixel values smaller than 0. " \
                                 "Your images do not seem to be RGB images"
        assert image.max() <= 255, "RGB images are uint 8, for whatever reason I found pixel values greater than 255" \
                                   ". Your images do not seem to be RGB images"
        image = image.astype(self.target_dtype)
        image = image / 255.
        return image

