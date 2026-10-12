#!/usr/bin/env python3
"""
创建U-Mamba增强网络架构图
生成类似于原始U-Mamba-network.png的可视化图像
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Arrow
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_enhanced_umamba_architecture():
    """创建增强版U-Mamba完整网络架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 定义颜色
    colors = {
        'input': '#E3F2FD',
        'encoder': '#BBDEFB',
        'lsa_enhanced': '#2196F3',
        'decoder': '#C8E6C9',
        'output': '#81C784',
        'skip': '#FFF3E0',
        'text': '#333333'
    }
    
    # 标题
    ax.text(8, 11.5, 'Enhanced U-Mamba Architecture\nLSA + Semantic Consistency + Channel Attention', 
            ha='center', va='center', fontsize=16, fontweight='bold', color=colors['text'])
    
    # 输入层
    input_box = FancyBboxPatch((1, 10), 2, 0.8, boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 10.4, 'Input\n4 Modalities\n[B,4,D,H,W]', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 编码器路径
    encoder_stages = [
        {'pos': (1, 8.5), 'text': 'Stem Conv\n[B,32,D,H,W]'},
        {'pos': (1, 7), 'text': 'LSA-Enhanced\nMambaBlock 1\n[B,32,D,H,W]'},
        {'pos': (1, 5.5), 'text': 'LSA-Enhanced\nMambaBlock 2\n[B,64,D/2,H/2,W/2]'},
        {'pos': (1, 4), 'text': 'LSA-Enhanced\nMambaBlock 3\n[B,128,D/4,H/4,W/4]'},
        {'pos': (1, 2.5), 'text': 'LSA-Enhanced\nMambaBlock 4\n[B,256,D/8,H/8,W/8]'},
        {'pos': (1, 1), 'text': 'Bottleneck\nLSA-Enhanced\n[B,320,D/16,H/16,W/16]'}
    ]
    
    for i, stage in enumerate(encoder_stages):
        if i == 0:
            color = colors['encoder']
        else:
            color = colors['lsa_enhanced']
        
        box = FancyBboxPatch(stage['pos'], 2.5, 1.2, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(stage['pos'][0] + 1.25, stage['pos'][1] + 0.6, stage['text'], 
                ha='center', va='center', fontsize=9, color='white' if i > 0 else 'black', fontweight='bold')
    
    # 解码器路径
    decoder_stages = [
        {'pos': (12.5, 2.5), 'text': 'Decoder Block 1\n+ Skip Connection\n[B,256,D/8,H/8,W/8]'},
        {'pos': (12.5, 4), 'text': 'Decoder Block 2\n+ Skip Connection\n[B,128,D/4,H/4,W/4]'},
        {'pos': (12.5, 5.5), 'text': 'Decoder Block 3\n+ Skip Connection\n[B,64,D/2,H/2,W/2]'},
        {'pos': (12.5, 7), 'text': 'Decoder Block 4\n+ Skip Connection\n[B,32,D,H,W]'},
        {'pos': (12.5, 8.5), 'text': 'Output Conv\n[B,num_classes,D,H,W]'}
    ]
    
    for stage in decoder_stages:
        color = colors['decoder'] if 'Decoder' in stage['text'] else colors['output']
        box = FancyBboxPatch(stage['pos'], 2.5, 1.2, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(stage['pos'][0] + 1.25, stage['pos'][1] + 0.6, stage['text'], 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 跳跃连接
    skip_connections = [
        ((3.5, 7.6), (12.5, 7.6)),
        ((3.5, 6.1), (12.5, 6.1)),
        ((3.5, 4.6), (12.5, 4.6)),
        ((3.5, 3.1), (12.5, 3.1))
    ]
    
    for start, end in skip_connections:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='orange', connectionstyle="arc3,rad=0.3"))
    
    # 主要数据流箭头
    main_flow = [
        ((2, 10), (2, 9.7)),
        ((2, 8.5), (2, 8.2)),
        ((2, 7), (2, 6.7)),
        ((2, 5.5), (2, 5.2)),
        ((2, 4), (2, 3.7)),
        ((2, 2.5), (2, 2.2))
    ]
    
    for start, end in main_flow:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
    
    # 底部到解码器的连接
    ax.annotate('', xy=(12.5, 3.1), xytext=(3.5, 1.6),
               arrowprops=dict(arrowstyle='->', lw=3, color='blue', connectionstyle="arc3,rad=-0.3"))
    
    # 解码器内部连接
    decoder_flow = [
        ((14, 2.5), (14, 3.7)),
        ((14, 4), (14, 5.2)),
        ((14, 5.5), (14, 6.7)),
        ((14, 7), (14, 8.2))
    ]
    
    for start, end in decoder_flow:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=3, color='green'))
    
    # 添加LSA模块说明
    lsa_box = FancyBboxPatch((5, 0.5), 6, 1.5, boxstyle="round,pad=0.1",
                            facecolor='#F3E5F5', edgecolor='purple', linewidth=2)
    ax.add_patch(lsa_box)
    ax.text(8, 1.25, 'RegionAwareLSA Module\nStyle Enhancement + Channel Attention\nDomain Adaptation for Cross-Domain Generalization', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='purple')
    
    plt.tight_layout()
    plt.savefig('d:/work/UMamba/enhanced_umamba_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_lsa_module_detail():
    """创建LSA模块详细结构图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(7, 9.5, 'RegionAwareLSA Module Architecture', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # 定义颜色
    colors = {
        'input': '#E3F2FD',
        'norm': '#BBDEFB', 
        'lsa': '#F3E5F5',
        'attention': '#E0F2F1',
        'mamba': '#FFF3E0',
        'output': '#E8F5E8'
    }
    
    # 模块位置
    modules = [
        {'pos': (1, 8), 'size': (2, 0.8), 'color': colors['input'], 'text': 'Input Features\n[B, C, D, H, W]'},
        {'pos': (1, 6.5), 'size': (2, 0.8), 'color': colors['norm'], 'text': 'Reshape\n[B, L, C]'},
        {'pos': (1, 5), 'size': (2, 0.8), 'color': colors['norm'], 'text': 'Layer Norm'},
        
        # LSA模块
        {'pos': (5, 7), 'size': (3, 2), 'color': colors['lsa'], 'text': 'RegionAwareLSA\n• Style Bank Sampling\n• AdaIN Transform\n• Domain Adaptation'},
        
        # 通道注意力
        {'pos': (9, 7), 'size': (3, 2), 'color': colors['attention'], 'text': 'Channel Attention\n• Global Avg Pool\n• SE Module\n• Channel Reweighting'},
        
        {'pos': (6, 4), 'size': (2, 0.8), 'color': colors['mamba'], 'text': 'Mamba Block'},
        {'pos': (6, 2.5), 'size': (2, 0.8), 'color': colors['norm'], 'text': 'Reshape\n[B, C, D, H, W]'},
        {'pos': (6, 1), 'size': (2, 0.8), 'color': colors['output'], 'text': 'Output Features\n[B, C, D, H, W]'}
    ]
    
    for module in modules:
        box = FancyBboxPatch(module['pos'], module['size'][0], module['size'][1], 
                            boxstyle="round,pad=0.1", facecolor=module['color'], 
                            edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(module['pos'][0] + module['size'][0]/2, module['pos'][1] + module['size'][1]/2, 
                module['text'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 连接箭头
    connections = [
        ((2, 8), (2, 7.3)),
        ((2, 6.5), (2, 5.8)),
        ((3, 5.4), (5, 7.5)),
        ((8, 8), (9, 8)),
        ((10.5, 7), (7.5, 4.8)),
        ((7, 4), (7, 3.3)),
        ((7, 2.5), (7, 1.8))
    ]
    
    for start, end in connections:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # 残差连接
    ax.annotate('', xy=(8.5, 2.9), xytext=(1.5, 8.4),
               arrowprops=dict(arrowstyle='->', lw=2, color='red', connectionstyle="arc3,rad=0.5"))
    ax.text(5, 0.3, 'Residual Connection', ha='center', va='center', fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('d:/work/UMamba/lsa_module_detail.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_semantic_consistency_training():
    """创建语义一致性训练架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(8, 9.5, 'Semantic Consistency Training Architecture', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # 输入
    input_box = FancyBboxPatch((1, 8), 2, 1, boxstyle="round,pad=0.1",
                              facecolor='#E3F2FD', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 8.5, 'Input Batch\n[B, 4, D, H, W]', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 双路径网络
    # Path 1 (with gradients)
    path1_box = FancyBboxPatch((4, 7), 3, 1.5, boxstyle="round,pad=0.1",
                              facecolor='#BBDEFB', edgecolor='blue', linewidth=2)
    ax.add_patch(path1_box)
    ax.text(5.5, 7.75, 'U-Mamba Network\n(with gradients)\nPath 1', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Path 2 (no gradients)
    path2_box = FancyBboxPatch((4, 5), 3, 1.5, boxstyle="round,pad=0.1",
                              facecolor='#F3E5F5', edgecolor='purple', linewidth=2)
    ax.add_patch(path2_box)
    ax.text(5.5, 5.75, 'U-Mamba Network\n(torch.no_grad())\nPath 2', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 输出
    output1_box = FancyBboxPatch((9, 7), 2, 1, boxstyle="round,pad=0.1",
                                facecolor='#C8E6C9', edgecolor='green', linewidth=2)
    ax.add_patch(output1_box)
    ax.text(10, 7.5, 'Output 1\nSoftmax', ha='center', va='center', fontsize=10, fontweight='bold')
    
    output2_box = FancyBboxPatch((9, 5.5), 2, 1, boxstyle="round,pad=0.1",
                                facecolor='#FFCDD2', edgecolor='red', linewidth=2)
    ax.add_patch(output2_box)
    ax.text(10, 6, 'Output 2\nSoftmax', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 损失计算
    seg_loss_box = FancyBboxPatch((12.5, 6.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#FFF3E0', edgecolor='orange', linewidth=2)
    ax.add_patch(seg_loss_box)
    ax.text(13.75, 7.25, 'Segmentation\nLoss\nDiceCE + GT', ha='center', va='center', fontsize=10, fontweight='bold')
    
    consistency_box = FancyBboxPatch((12.5, 4.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                                   facecolor='#FFEB3B', edgecolor='#F57F17', linewidth=3)
    ax.add_patch(consistency_box)
    ax.text(13.75, 5.25, 'Consistency Loss\nMSE(Prob1, Prob2)\nλ = 0.05', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 总损失
    total_loss_box = FancyBboxPatch((6, 2.5), 4, 1.2, boxstyle="round,pad=0.1",
                                   facecolor='#FFCDD2', edgecolor='red', linewidth=3)
    ax.add_patch(total_loss_box)
    ax.text(8, 3.1, 'Total Loss = 0.5×(L1+L2) + 0.05×Lc\nBackpropagation', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 连接箭头
    arrows = [
        ((3, 8.5), (4, 7.75)),
        ((3, 8.5), (4, 5.75)),
        ((7, 7.75), (9, 7.5)),
        ((7, 5.75), (9, 6)),
        ((11, 7.5), (12.5, 7.25)),
        ((11, 6), (12.5, 5.25)),
        ((13.75, 6.5), (8.5, 3.7)),
        ((13.75, 4.5), (8.5, 3.7))
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    plt.tight_layout()
    plt.savefig('d:/work/UMamba/semantic_consistency_training.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_intelligent_preprocessing():
    """创建智能推理预处理架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 标题
    ax.text(7, 11.5, 'Intelligent Inference Preprocessing Pipeline', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # 输入
    input_box = FancyBboxPatch((1, 10), 3, 1, boxstyle="round,pad=0.1",
                              facecolor='#E3F2FD', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2.5, 10.5, 'External Test Data\nMulti-Domain', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 检测模块
    detection_box = FancyBboxPatch((6, 9.5), 4, 2, boxstyle="round,pad=0.1",
                                  facecolor='#F3E5F5', edgecolor='purple', linewidth=2)
    ax.add_patch(detection_box)
    ax.text(8, 10.5, 'Smart Detection\n• Environment Variables\n• Statistical Analysis\n• Normalization Status', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 处理策略
    strategies = [
        {'pos': (1, 7.5), 'text': 'Skip\nNormalization', 'color': '#C8E6C9'},
        {'pos': (5, 7.5), 'text': 'Apply\nZ-score Norm', 'color': '#FFCDD2'},
        {'pos': (9, 7.5), 'text': 'Frame\nRecovery', 'color': '#FFF3E0'}
    ]
    
    for strategy in strategies:
        box = FancyBboxPatch(strategy['pos'], 2.5, 1.5, boxstyle="round,pad=0.1",
                            facecolor=strategy['color'], edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(strategy['pos'][0] + 1.25, strategy['pos'][1] + 0.75, strategy['text'], 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # LSA激活推理
    lsa_inference_box = FancyBboxPatch((4, 5), 6, 2, boxstyle="round,pad=0.1",
                                      facecolor='#E1F5FE', edgecolor='blue', linewidth=3)
    ax.add_patch(lsa_inference_box)
    ax.text(7, 6, 'LSA-Activated Inference\n• 5-Fold Model Ensemble\n• Domain Adaptation (p=0.8)\n• Cross-Domain Generalization', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 输出
    output_box = FancyBboxPatch((5.5, 2.5), 3, 1, boxstyle="round,pad=0.1",
                               facecolor='#E8F5E8', edgecolor='green', linewidth=2)
    ax.add_patch(output_box)
    ax.text(7, 3, 'Final Segmentation\nResults', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 连接箭头
    arrows = [
        ((4, 10.5), (6, 10.5)),
        ((8, 9.5), (2.25, 9)),
        ((8, 9.5), (6.25, 9)),
        ((8, 9.5), (10.25, 9)),
        ((2.25, 7.5), (5.5, 7)),
        ((6.25, 7.5), (6.5, 7)),
        ((10.25, 7.5), (8.5, 7)),
        ((7, 5), (7, 3.5))
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    plt.tight_layout()
    plt.savefig('d:/work/UMamba/intelligent_preprocessing.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_diagram():
    """创建技术对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 原始U-Mamba
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Original U-Mamba', fontsize=14, fontweight='bold', pad=20)
    
    original_components = [
        {'pos': (2, 8), 'text': 'Mamba Block', 'color': '#BBDEFB'},
        {'pos': (2, 6.5), 'text': 'Single Forward', 'color': '#BBDEFB'},
        {'pos': (2, 5), 'text': 'Standard Inference', 'color': '#BBDEFB'},
        {'pos': (2, 3.5), 'text': 'Fixed Preprocessing', 'color': '#BBDEFB'}
    ]
    
    for comp in original_components:
        box = FancyBboxPatch(comp['pos'], 6, 1, boxstyle="round,pad=0.1",
                            facecolor=comp['color'], edgecolor='black', linewidth=1)
        ax1.add_patch(box)
        ax1.text(comp['pos'][0] + 3, comp['pos'][1] + 0.5, comp['text'], 
                ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 增强U-Mamba
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Enhanced U-Mamba', fontsize=14, fontweight='bold', pad=20)
    
    enhanced_components = [
        {'pos': (2, 8), 'text': 'Mamba + LSA + Attention', 'color': '#2196F3'},
        {'pos': (2, 6.5), 'text': 'Dual Forward + Consistency', 'color': '#7B1FA2'},
        {'pos': (2, 5), 'text': 'LSA-Activated Inference', 'color': '#388E3C'},
        {'pos': (2, 3.5), 'text': 'Smart Preprocessing', 'color': '#F57C00'}
    ]
    
    for comp in enhanced_components:
        box = FancyBboxPatch(comp['pos'], 6, 1, boxstyle="round,pad=0.1",
                            facecolor=comp['color'], edgecolor='black', linewidth=2)
        ax2.add_patch(box)
        ax2.text(comp['pos'][0] + 3, comp['pos'][1] + 0.5, comp['text'], 
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # 改进点标注
    improvements = [
        '+ Domain Adaptation',
        '+ Generalization Stability', 
        '+ Cross-Domain Inference',
        '+ Adaptive Processing'
    ]
    
    for i, improvement in enumerate(improvements):
        ax2.text(9.5, 8.5 - i*1.5, improvement, ha='center', va='center', 
                fontsize=10, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig('d:/work/UMamba/technical_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("🎨 开始创建U-Mamba增强架构图...")
    
    print("📊 创建完整网络架构图...")
    create_enhanced_umamba_architecture()
    
    print("🔍 创建LSA模块详细图...")
    create_lsa_module_detail()
    
    print("🔄 创建语义一致性训练图...")
    create_semantic_consistency_training()
    
    print("⚙️ 创建智能预处理图...")
    create_intelligent_preprocessing()
    
    print("📈 创建技术对比图...")
    create_comparison_diagram()
    
    print("✅ 所有架构图已生成完成！")
    print("📁 图片保存在: d:/work/UMamba/")
    print("🖼️ 生成的图片文件:")
    print("   - enhanced_umamba_architecture.png")
    print("   - lsa_module_detail.png")
    print("   - semantic_consistency_training.png")
    print("   - intelligent_preprocessing.png")
    print("   - technical_comparison.png")