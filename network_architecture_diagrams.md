# U-Mamba增强网络架构图：LSA+语义一致性+通道注意力

## 1. 增强版U-Mamba完整网络架构

```mermaid
graph TB
    subgraph "输入层 Input Layer"
        A["医学图像输入<br/>4 Modalities<br/>[B, 4, D, H, W]"]
    end
    
    subgraph "编码器 Encoder Path"
        B["Stem Conv<br/>Conv3d + GroupNorm + SiLU<br/>[B, 32, D, H, W]"]
        
        subgraph "Stage 1"
            C1["ResidualMambaBlock 1<br/>LSA Enhanced<br/>[B, 32, D, H, W]"]
        end
        
        D1["Downsample 1<br/>Stride Conv<br/>[B, 64, D/2, H/2, W/2]"]
        
        subgraph "Stage 2"
            C2["ResidualMambaBlock 2<br/>LSA Enhanced<br/>[B, 64, D/2, H/2, W/2]"]
        end
        
        D2["Downsample 2<br/>Stride Conv<br/>[B, 128, D/4, H/4, W/4]"]
        
        subgraph "Stage 3"
            C3["ResidualMambaBlock 3<br/>LSA Enhanced<br/>[B, 128, D/4, H/4, W/4]"]
        end
        
        D3["Downsample 3<br/>Stride Conv<br/>[B, 256, D/8, H/8, W/8]"]
        
        subgraph "Stage 4"
            C4["ResidualMambaBlock 4<br/>LSA Enhanced<br/>[B, 256, D/8, H/8, W/8]"]
        end
        
        D4["Downsample 4<br/>Stride Conv<br/>[B, 320, D/16, H/16, W/16]"]
        
        subgraph "Bottleneck"
            C5["ResidualMambaBlock 5<br/>LSA Enhanced<br/>[B, 320, D/16, H/16, W/16]"]
        end
    end
    
    subgraph "解码器 Decoder Path"
        U1["Upsample 1<br/>TransposeConv<br/>[B, 256, D/8, H/8, W/8]"]
        
        subgraph "Decoder Stage 1"
            R1["ResidualBlock 1<br/>Skip Connection<br/>[B, 256, D/8, H/8, W/8]"]
        end
        
        U2["Upsample 2<br/>TransposeConv<br/>[B, 128, D/4, H/4, W/4]"]
        
        subgraph "Decoder Stage 2"
            R2["ResidualBlock 2<br/>Skip Connection<br/>[B, 128, D/4, H/4, W/4]"]
        end
        
        U3["Upsample 3<br/>TransposeConv<br/>[B, 64, D/2, H/2, W/2]"]
        
        subgraph "Decoder Stage 3"
            R3["ResidualBlock 3<br/>Skip Connection<br/>[B, 64, D/2, H/2, W/2]"]
        end
        
        U4["Upsample 4<br/>TransposeConv<br/>[B, 32, D, H, W]"]
        
        subgraph "Decoder Stage 4"
            R4["ResidualBlock 4<br/>Skip Connection<br/>[B, 32, D, H, W]"]
        end
    end
    
    subgraph "输出层 Output Layer"
        O["Output Conv<br/>1x1x1 Conv<br/>[B, num_classes, D, H, W]"]
    end
    
    subgraph "跳跃连接 Skip Connections"
        S1["Skip 1"] 
        S2["Skip 2"]
        S3["Skip 3"] 
        S4["Skip 4"]
    end
    
    A --> B --> C1 --> D1 --> C2 --> D2 --> C3 --> D3 --> C4 --> D4 --> C5
    C5 --> U1 --> R1 --> U2 --> R2 --> U3 --> R3 --> U4 --> R4 --> O
    
    C1 --> S1 --> R4
    C2 --> S2 --> R3  
    C3 --> S3 --> R2
    C4 --> S4 --> R1
    
    style C1 fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style C2 fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style C3 fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style C4 fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style C5 fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
```

## 2. 增强版MambaLayer详细结构

```mermaid
graph TB
    subgraph "Enhanced MambaLayer Architecture"
        A["输入特征<br/>[B, C, D, H, W]"]
        
        subgraph "预处理 Preprocessing"
            B["Reshape<br/>[B, C, D×H×W] → [B, D×H×W, C]"]
            C["Layer Normalization<br/>LayerNorm(C)"]
        end
        
        subgraph "RegionAwareLSA Module"
            D["特征统计计算<br/>Mean & Std per Sample"]
            
            subgraph "样式增强 Style Enhancement"
                E["样式库采样<br/>Style Bank Sampling"]
                F["AdaIN变换<br/>Adaptive Instance Norm"]
            end
            
            subgraph "通道注意力 Channel Attention"
                G["全局平均池化<br/>Global Average Pooling"]
                H["SE模块<br/>Linear→ReLU→Linear→Sigmoid"]
                I["通道权重<br/>Channel Weights [B, C]"]
            end
            
            J["重要性掩码<br/>Importance Masking"]
            K["风格融合<br/>Style Fusion with γ"]
        end
        
        subgraph "Mamba处理 Mamba Processing"
            L["Mamba Block<br/>State Space Model"]
            M["选择性扫描<br/>Selective Scan"]
            N["输出投影<br/>Output Projection"]
        end
        
        subgraph "后处理 Postprocessing"
            O["Reshape<br/>[B, D×H×W, C] → [B, C, D, H, W]"]
            P["残差连接<br/>Residual Connection"]
        end
        
        Q["输出特征<br/>[B, C, D, H, W]"]
    end
    
    A --> B --> C --> D
    D --> E --> F
    D --> G --> H --> I --> J
    F --> K
    I --> K
    C --> L
    K --> L --> M --> N --> O
    A --> P
    O --> P --> Q
    
    style D fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style H fill:#e0f2f1,stroke:#388e3c,stroke-width:2px
    style K fill:#fff8e1,stroke:#f57c00,stroke-width:2px
    style L fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
```

## 3. 语义一致性训练架构

```mermaid
graph TB
    subgraph "Semantic Consistency Training Architecture"
        A["输入批次<br/>Input Batch<br/>[B, 4, D, H, W]"]
        
        subgraph "双路径前向传播 Dual Forward Paths"
            subgraph "路径1 Path 1 (with gradients)"
                B1["U-Mamba Network"]
                C1["输出1<br/>Output 1<br/>[B, C, D, H, W]"]
                D1["Softmax<br/>Probability 1"]
            end
            
            subgraph "路径2 Path 2 (no gradients)"
                B2["U-Mamba Network<br/>(torch.no_grad())"]
                C2["输出2<br/>Output 2<br/>[B, C, D, H, W]"]
                D2["Softmax<br/>Probability 2"]
            end
        end
        
        subgraph "损失计算 Loss Computation"
            E["分割损失1<br/>Segmentation Loss 1<br/>DiceCE(Output1, GT)"]
            F["分割损失2<br/>Segmentation Loss 2<br/>DiceCE(Output2, GT)"]
            G["一致性损失<br/>Consistency Loss<br/>MSE(Prob1, Prob2)"]
            
            H["总损失<br/>Total Loss<br/>L = 0.5×(L1+L2) + 0.05×Lc"]
        end
        
        subgraph "优化器 Optimizer"
            I["反向传播<br/>Backpropagation"]
            J["梯度裁剪<br/>Gradient Clipping"]
            K["参数更新<br/>Parameter Update"]
        end
    end
    
    A --> B1 --> C1 --> D1
    A --> B2 --> C2 --> D2
    
    C1 --> E
    C2 --> F
    D1 --> G
    D2 --> G
    
    E --> H
    F --> H  
    G --> H
    
    H --> I --> J --> K
    
    style B1 fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style B2 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style G fill:#fff3e0,stroke:#f57c00,stroke-width:3px
    style H fill:#ffebee,stroke:#d32f2f,stroke-width:2px
```

## 4. 智能推理预处理架构

```mermaid
graph TB
    subgraph "Intelligent Inference Preprocessing Architecture"
        A["外部测试数据<br/>External Test Data<br/>Various Domains"]
        
        subgraph "数据状态检测 Data Status Detection"
            B["环境变量检查<br/>NNUNET_SKIP_NORMALIZATION"]
            C["统计特征分析<br/>Statistical Analysis"]
            D["数据范围检测<br/>Value Range Detection"]
            E["标准化状态判断<br/>Normalization Status"]
        end
        
        subgraph "自适应处理策略 Adaptive Processing"
            F["跳过标准化<br/>Skip Normalization"]
            G["执行标准化<br/>Apply Z-score Norm"]
            H["失帧修复<br/>Missing Frame Recovery"]
        end
        
        subgraph "失帧修复方法 Recovery Methods"
            I["线性插值<br/>Linear Interpolation"]
            J["三次样条<br/>Cubic Spline"]
            K["邻近复制<br/>Adjacent Copy"]
        end
        
        subgraph "LSA激活推理 LSA-Activated Inference"
            L["5折模型加载<br/>5-Fold Model Loading"]
            M["LSA模块激活<br/>LSA Module Activation<br/>p=0.8, training=True"]
            N["域适应推理<br/>Domain Adaptive Inference"]
        end
        
        O["预处理完成数据<br/>Preprocessed Data"]
        P["推理结果<br/>Inference Results"]
    end
    
    A --> B --> C --> D --> E
    E --> F
    E --> G  
    E --> H
    
    H --> I
    H --> J
    H --> K
    
    F --> O
    G --> O
    I --> O
    J --> O  
    K --> O
    
    O --> L --> M --> N --> P
    
    style C fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    style E fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style H fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style M fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
```

## 5. 端到端系统架构

```mermaid
graph TB
    subgraph "End-to-End Enhanced U-Mamba System"
        subgraph "训练阶段 Training Phase"
            A1["BraTS训练数据<br/>Training Data"]
            B1["数据增强<br/>Data Augmentation"]
            C1["Enhanced U-Mamba<br/>LSA + Consistency"]
            D1["双路径损失<br/>Dual-Path Loss"]
            E1["模型保存<br/>Model Checkpoint"]
        end
        
        subgraph "推理阶段 Inference Phase"  
            A2["外部测试数据<br/>External Test Data"]
            B2["智能预处理<br/>Smart Preprocessing"]
            C2["LSA激活推理<br/>LSA-Activated Inference"]
            D2["5折集成<br/>5-Fold Ensemble"]
            E2["后处理优化<br/>Post-processing"]
        end
        
        subgraph "核心创新模块 Core Innovations"
            F1["RegionAwareLSA<br/>局部风格增强"]
            F2["Semantic Consistency<br/>语义一致性"]
            F3["Channel Attention<br/>通道注意力"]
        end
        
        subgraph "质量保障 Quality Assurance"
            G1["训练监控<br/>Training Monitor"]
            G2["推理质控<br/>Inference QC"]
            G3["异常处理<br/>Exception Handler"]
        end
    end
    
    A1 --> B1 --> C1 --> D1 --> E1
    A2 --> B2 --> C2 --> D2 --> E2
    
    C1 --> F1
    C1 --> F2  
    C1 --> F3
    C2 --> F1
    C2 --> F2
    C2 --> F3
    
    C1 --> G1
    C2 --> G2
    G1 --> G3
    G2 --> G3
    
    style F1 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px
    style F2 fill:#fff3e0,stroke:#f57c00,stroke-width:3px
    style F3 fill:#e0f2f1,stroke:#388e3c,stroke-width:3px
    style G3 fill:#ffebee,stroke:#d32f2f,stroke-width:2px
```

## 6. 关键技术对比图

```mermaid
graph LR
    subgraph "原始U-Mamba vs 增强U-Mamba"
        subgraph "Original U-Mamba"
            A1["Mamba Block"]
            A2["Single Forward"]
            A3["Standard Inference"]
            A4["Fixed Preprocessing"]
        end
        
        subgraph "Enhanced U-Mamba"
            B1["Mamba + LSA + Attention"]
            B2["Dual Forward + Consistency"]
            B3["LSA-Activated Inference"]
            B4["Smart Preprocessing"]
        end
        
        subgraph "Improvements"
            C1["+ 域适应能力<br/>Domain Adaptation"]
            C2["+ 泛化稳定性<br/>Generalization Stability"] 
            C3["+ 跨域推理<br/>Cross-Domain Inference"]
            C4["+ 自适应处理<br/>Adaptive Processing"]
        end
    end
    
    A1 --> B1 --> C1
    A2 --> B2 --> C2
    A3 --> B3 --> C3  
    A4 --> B4 --> C4
    
    style B1 fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style B2 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style B3 fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style B4 fill:#fff8e1,stroke:#f57c00,stroke-width:2px
    style C1 fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    style C2 fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    style C3 fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    style C4 fill:#ffebee,stroke:#d32f2f,stroke-width:2px
```

这些架构图采用了类似于原始U-Mamba-network.png的风格，展现了：

1. **完整网络架构** - 编码器-解码器结构与跳跃连接
2. **增强MambaLayer** - LSA+通道注意力的详细结构
3. **语义一致性训练** - 双路径前向传播机制
4. **智能推理预处理** - 数据检测与自适应处理
5. **端到端系统** - 训练与推理的完整流程
6. **技术对比** - 原始vs增强版的改进点

每个图都突出了您的三大创新点：LSA、语义一致性训练和通道注意力机制，并且采用了清晰的模块化设计和专业的配色方案。