训练U-Mamba模型：
五折交叉验证训练：nnUNetv2_train 999 3d_fullres 0 -tr nnUNetTrainerUMambaEnc && \
nnUNetv2_train 999 3d_fullres 1 -tr nnUNetTrainerUMambaEnc && \
nnUNetv2_train 999 3d_fullres 2 -tr nnUNetTrainerUMambaEnc && \
nnUNetv2_train 999 3d_fullres 3 -tr nnUNetTrainerUMambaEnc && \
nnUNetv2_train 999 3d_fullres 4 -tr nnUNetTrainerUMambaEnc（输入一次就可运行五个折，前一折运行完之后才会进行下一折）

训练完模型之后：对测试集进行评估：
    首先运行nnUNetv2_predict -i /root/autodl-tmp/U-Mamba/data/nnUNet_raw/Dataset999_BraTS2024/imagesTs -o /root/autodl-tmp/U-Mamba/data/nnUNet_results/Dataset999_BraTS2024/nnUNetTrainerUMambaEnc__nnUNetPlans__3d_fullres/output_predict/0 -d 999 -c 3d_fullres -f 0 -tr nnUNetTrainerUMambaEnc --disable_tta（0的地方修改：表示第几折）
或者直接运行for fold in 0 1 2 3 4; do nnUNetv2_predict -i /root/autodl-tmp/U-Mamba/data/nnUNet_raw/Dataset999_BraTS2024/imagesTs -o /root/autodl-tmp/U-Mamba/data/nnUNet_results/Dataset999_BraTS2024/nnUNetTrainerUMambaEnc__nnUNetPlans__3d_fullres/output_predict_MET/$fold -d 999 -c 3d_fullres -f $fold -tr nnUNetTrainerUMambaEnc --disable_tta; done

然后运行U-Mamba目录下的manual_ensemble.py文件
后运行evaluation目录下的test.py文件进行评估
Batch_size在nnUNet_preprocessed/Dataset999_BraTS2024下的nnUNetPlans.json下

