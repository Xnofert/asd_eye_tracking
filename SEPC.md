

---

# 需求规格说明书 (Spec.md): ASD 眼动筛查多模态模型

## 1. 项目概述
本项目的目标是开发一个基于深度学习的辅助诊断模型，通过分析受试者在查看特定视觉刺激时的**原始图像 (RGB)** 和生成的**眼动热力图 (Heatmap)**，实现对自闭症谱系障碍 (ASD) 的自动筛查。

## 2. 数据规格 (Data Specifications)

### 2.1 输入数据定义
模型接收双路输入：
* **模态 A (Input_RGB):** 原始视觉刺激图片（自然场景、社交互动、面部特征等）。
    * 格式：`224 x 224 x 3` (RGB)
    * 归一化：均值/标准差对齐 ImageNet 预训练标准。
* **模态 B (Input_Heatmap):** 对应时段的聚合眼动热力图。
    * 格式：`224 x 224 x 1` (灰度) 
    * 对齐要求：热力图的空间坐标必须与原始图像像素严格点对点匹配。

### 2.2 数据增强策略
* **联动变换 (Linked Transformation):** 所有的几何变换（旋转、翻转、缩放）必须同时作用于一对 RGB 和 Heatmap 样本，确保语义与行为的一致性。
* **禁忌操作：** 禁止对热力图进行单独的色彩扰动，以免破坏密度分布信息。

---

## 3. 模型架构 (Model Architecture)

### 3.1 双路特征提取器 (Two-Stream Backbone)
* **Branch_1 (Semantic):** * 模型：`ResNet-50` (去掉最后分类层)。
    * 初始化：加载 `ImageNet` 预训练权重。
    * 目标：提取场景中的社交语义（人脸、关键物体）。
* **Branch_2 (Gaze):** * 模型：轻量级 5 层卷积神经网络。
    * 初始化：He 正态分布。
    * 目标：提取空间注意力分布模式。

### 3.2 融合层 (Fusion Layer)
* **方法：** 后期特征拼接 (Concatenation)。
* **结构：**
    1.  `GlobalAveragePooling2D()` 处理两路分支输出。
    2.  `Concat([Feature_RGB, Feature_Heatmap])`。
    3.  `Dense(512, activation='relu')` + `Dropout(0.5)`。
    4.  `Dense(1, activation='sigmoid')` 用于二分类。

---

## 4. 训练规格 (Training Specs)

| 参数项 | 设定值 |
| :--- | :--- |
| **损失函数** | Binary Cross-Entropy (带 Class Weight，比例依据样本分布) |
| **优化器** | AdamW (Learning Rate: $1 \times 10^{-4}$) |
| **批大小 (Batch Size)** | 32 (视显存情况而定) |
| **冻结策略** | 前 10 个 Epoch 冻结 Branch_1 的权重，之后微调高层 |

---

## 5. 功能需求与验收标准

### 5.1 核心功能
* **特征解耦：** 模型能够分别识别出场景物体和眼动焦点。
* **判别推断：** 输入一对图片，输出 ASD 风险概率分数。

### 5.2 性能指标 (KPI)
1.  **AUC-ROC**
2.  **Sensitivity (召回率)**
3.  **Specificity (特异性)**

---

## 6. 潜在风险与约束
* **过拟合风险：** 由于 ASD 数据集通常较小，必须执行验证集交叉验证。

---

## 7. 目录结构
```
asd_eye_tracking/
├── data/          ← 数据集
├── output/       ← 训练结果
├── scripts/       ← 脚本
└─  src/             ← 源码
```

```
asd_eye_tracking/src/
├── dataset/     ← 数据集加载器
├── model/       ← 模型
└─  train/          ← 训练器
```