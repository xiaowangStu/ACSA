# 智能推理预处理框架：数据标准化检测与处理

## 1. 推理时数据预处理完整流程

```mermaid
graph TB
    subgraph "外部测试数据输入"
        A[新的外部数据集<br/>Different Domain Data]
        B[数据加载<br/>Data Loading]
    end
    
    subgraph "智能预处理检测"
        C[环境变量检查<br/>NNUNET_SKIP_NORMALIZATION]
        D{强制跳过标准化?}
        E[数据质量检测<br/>Data Quality Assessment]
        
        subgraph "多维度检测机制"
            F[统计特征分析<br/>Statistical Analysis]
            G[数据范围检测<br/>Value Range Detection]
            H[失帧检测<br/>Missing Frame Detection]
            I[标准化状态判断<br/>Normalization Status]
        end
        
        J{数据状态判断}
        K[标准化处理<br/>Apply Normalization]
        L[跳过标准化<br/>Skip Normalization]
        M[失帧修复<br/>Frame Recovery]
    end
    
    subgraph "自适应处理策略"
        N[掩码标准化<br/>Masked Normalization]
        O[全局标准化<br/>Global Normalization]
        P[插值修复<br/>Interpolation Recovery]
        Q[邻近帧填充<br/>Adjacent Frame Filling]
    end
    
    subgraph "推理准备"
        R[预处理完成数据<br/>Preprocessed Data]
        S[U-Mamba推理<br/>Model Inference]
        T[后处理<br/>Post-processing]
        U[最终输出<br/>Final Segmentation]
    end
    
    A --> B --> C --> D
    D -->|是| L
    D -->|否| E
    
    E --> F
    E --> G
    E --> H
    F --> I
    G --> I
    H --> I
    I --> J
    
    J -->|已标准化| L
    J -->|未标准化| K
    J -->|有失帧| M
    
    K --> N
    K --> O
    M --> P
    M --> Q
    
    L --> R
    N --> R
    O --> R
    P --> R
    Q --> R
    
    R --> S --> T --> U
    
    style F fill:#e1f5fe
    style I fill:#f3e5f5
    style J fill:#fff3e0
    style S fill:#e8f5e8
```

## 2. 数据状态智能检测算法

```mermaid
graph TB
    subgraph "数据标准化状态检测"
        A[输入医学图像<br/>Input Medical Image]
        B[计算全局统计量<br/>Global Statistics]
        C[计算掩码统计量<br/>Masked Statistics]
        
        subgraph "多重检测条件"
            D[均值范围检测<br/>-15 < mean < 0]
            E[标准差范围检测<br/>1.0 < std < 5.0]
            F[数据范围检测<br/>(max-min) < 50]
            G[原始数据排除<br/>Non-raw Medical Range]
        end
        
        H[综合判断<br/>Combined Decision]
        I{是否已标准化?}
        J[返回: 已标准化<br/>Skip Normalization]
        K[返回: 需标准化<br/>Apply Normalization]
    end
    
    A --> B
    A --> C
    B --> D
    B --> E
    B --> F
    C --> D
    C --> E
    C --> F
    B --> G
    C --> G
    
    D --> H
    E --> H
    F --> H
    G --> H
    
    H --> I
    I -->|是| J
    I -->|否| K
    
    style D fill:#e8eaf6
    style E fill:#e8eaf6
    style F fill:#e8eaf6
    style G fill:#e8eaf6
    style H fill:#fff8e1
```

## 3. 失帧检测与修复机制

```mermaid
graph TB
    subgraph "失帧检测与处理"
        A[3D医学图像<br/>Shape: [D, H, W]]
        B[逐层分析<br/>Slice-wise Analysis]
        C[异常检测<br/>Anomaly Detection]
        
        subgraph "失帧类型判断"
            D[全零切片检测<br/>Zero Slice Detection]
            E[异常值切片检测<br/>Outlier Slice Detection]
            F[不连续切片检测<br/>Discontinuity Detection]
        end
        
        subgraph "修复策略选择"
            G[线性插值<br/>Linear Interpolation]
            H[三次样条插值<br/>Cubic Spline]
            I[邻近切片复制<br/>Adjacent Copy]
            J[深度学习修复<br/>DL-based Inpainting]
        end
        
        K[质量验证<br/>Quality Validation]
        L[修复完成<br/>Recovery Complete]
    end
    
    A --> B --> C
    C --> D
    C --> E
    C --> F
    
    D --> G
    E --> H
    F --> I
    D --> J
    E --> J
    F --> J
    
    G --> K
    H --> K
    I --> K
    J --> K
    K --> L
    
    style C fill:#ffebee
    style G fill:#e8f5e8
    style H fill:#e8f5e8
    style I fill:#e8f5e8
    style J fill:#e8f5e8
```

## 4. 推理流程集成架构

```mermaid
graph TB
    subgraph "完整推理流程"
        A[外部测试数据<br/>External Test Data]
        
        subgraph "预处理模块"
            B[数据加载器<br/>Data Loader]
            C[智能检测器<br/>Smart Detector]
            D[自适应处理器<br/>Adaptive Processor]
        end
        
        subgraph "U-Mamba推理引擎"
            E[编码器<br/>Encoder with LSA]
            F[解码器<br/>Decoder]
            G[特征融合<br/>Feature Fusion]
        end
        
        subgraph "后处理模块"
            H[概率转换<br/>Probability Conversion]
            I[形态学处理<br/>Morphological Ops]
            J[连通域分析<br/>Connected Component]
        end
        
        K[最终分割结果<br/>Final Segmentation]
        
        subgraph "质量保障"
            L[推理质量监控<br/>Inference Quality Monitor]
            M[异常结果检测<br/>Anomaly Result Detection]
            N[自动重试机制<br/>Auto Retry Mechanism]
        end
    end
    
    A --> B --> C --> D
    D --> E --> F --> G
    G --> H --> I --> J --> K
    
    E --> L
    F --> L
    G --> L
    L --> M
    M --> N
    N --> E
    
    style C fill:#e1f5fe
    style D fill:#f3e5f5
    style L fill:#fff3e0
    style M fill:#ffebee
```

## 5. 环境配置与控制接口

```mermaid
graph TB
    subgraph "推理控制接口"
        A[环境变量配置<br/>Environment Variables]
        B[NNUNET_SKIP_NORMALIZATION<br/>强制跳过标准化]
        C[NNUNET_DETECTION_MODE<br/>检测模式设置]
        D[NNUNET_RECOVERY_METHOD<br/>修复方法选择]
        
        subgraph "动态参数调整"
            E[检测阈值设置<br/>Detection Thresholds]
            F[修复策略选择<br/>Recovery Strategy]
            G[质量控制参数<br/>Quality Control Params]
        end
        
        subgraph "推理监控"
            H[实时状态监控<br/>Real-time Monitoring]
            I[处理日志记录<br/>Processing Logs]
            J[异常警报系统<br/>Alert System]
        end
        
        K[推理执行引擎<br/>Inference Engine]
    end
    
    A --> B
    A --> C
    A --> D
    
    B --> E
    C --> F
    D --> G
    
    E --> K
    F --> K
    G --> K
    
    K --> H
    K --> I
    K --> J
    
    style A fill:#e8eaf6
    style K fill:#e8f5e8
    style H fill:#fff8e1
```