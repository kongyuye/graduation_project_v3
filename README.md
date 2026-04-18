# 项目架构 (Project Architecture)

本项目是一个面向工业设备故障预测与健康管理（PHM, Prognostics and Health Management）的端到端深度学习框架。核心架构采用 **时空图神经网络（ST-GNN）**、**Transformer** 与 **域自适应（Domain Adaptation）** 的混合对抗模型，旨在同时解决跨工况条件下的设备退化阶段分类和剩余使用寿命（RUL）的回归预测问题。

## 1. 核心模型架构 (Core Model Architecture)

模型形成从基础特征提取、时空拓扑聚合到全局时序演化，再到多任务与域自适应输出的完整流水线：

### 1.1 基础特征工程 (`feature_extractor.py`)
用于从原始振动信号中提取关键物理特征：
- **时域特征**：提取均方根 (RMS)、峭度 (Kurtosis)、偏度 (Skewness) 等能够敏感反映设备早期冲击和退化趋势的统计学特征，为下游模型提供高质量的先验输入。

### 1.2 时空特征主干网络 (`st_backbone.py` & `dynamic_graph.py`)
负责提取设备传感器数据在时间和空间（特征通道/传感器节点）维度上的局部特征：
- **动态自适应图拓扑 (`dynamic_graph.py`)**：摒弃了传统的静态先验物理图，引入可学习的高维嵌入向量 $E$，通过计算 $A = \text{Softmax}(\text{ReLU}(E \cdot E^T))$ 动态生成节点间的注意力权重邻接矩阵。能够自动挖掘隐含的物理耦合关系，并通过 ReLU 稀疏化网络去除负相关的噪声连接。
- **时间维度提取 (`st_backbone.py`)**：使用一维卷积网络 (`TemporalCNN`) 沿着时间轴提取局部平滑特征，捕捉短期的退化趋势。
- **空间维度聚合 (`st_backbone.py` / `model_architecture.py`)**：基于动态生成的邻接矩阵，使用图卷积网络（GCN）或图注意力网络（DenseGATConv）进行节点间的空间消息传递与特征聚合。

### 1.3 全局时序演化层 (`model_architecture.py`)
在提取时空局部特征后，利用 Transformer 架构捕捉长程的时序退化演化规律：
- 将空间聚合后的特征展平并进行线性投影以匹配隐藏维度 (`d_model`)。
- 通过多层 Transformer Encoder（多头自注意力机制）对整个时间窗内的全局特征进行深度建模，学习设备性能退化的长期依赖关系。

### 1.4 多任务与域自适应对抗头 (`model_architecture.py` & `mt_head.py`)
接收 Transformer 输出的全局时序特征，执行联合预测及跨域特征对齐：
- **全局时序池化与降维**：通过全局平均池化 (GAP) 将时间窗特征浓缩为宏观健康摘要，并经过 Shared MLP 提炼高阶语义。
- **核心主任务双分支**：
  - **分类头 (Classification Branch)**：预测设备当前的退化阶段（如：正常、轻微退化、严重退化）。
  - **回归头 (Regression Branch)**：预测设备的剩余使用寿命百分比（RUL）。
- **域自适应对抗分支 (Domain Adversarial Branch)**：
  - 引入 **梯度反转层 (GRL, Gradient Reversal Layer)** 与 **域分类器 (Domain Classifier)**。判别器试图正确分类数据所属的工况（域），而特征提取器通过 GRL 接收反向传播的负梯度，被优化为“欺骗”判别器，从而提取出**工况无关 (Domain-invariant)** 的共有特征。
  - 结合 **多核最大均值差异 (Multi-kernel MMD Loss)**，将源域与目标域特征映射到 RKHS 空间中并最小化分布距离，有效防止跨工况预测时的负迁移 (Negative Transfer) 现象。
