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

## 2. 训练与评估机制 (Training & Evaluation)

### 2.1 极小化-极大化对抗博弈训练 (Min-Max Game)
针对跨域场景，训练过程包含：
- **极小化 (Minimization)**：最小化源域 RUL 预测误差 (MSE) 与源域/目标域间的 MMD 距离，确保主任务精度与特征分布对齐。
- **极大化 (Maximization)**：通过 GRL 隐式实现，最大化域判别器的分类误差，抹除特征中的工况特异性痕迹。

### 2.2 工业级非对称惩罚评估 (Asymmetric Scoring)
针对工业 PHM 领域的特殊需求，支持非对称指数惩罚损失机制：
- **高估寿命惩罚**：模型预测寿命大于实际寿命会导致意外停机等灾难性后果，因此施加更严厉的指数惩罚。
- **低估寿命惩罚**：预测寿命小于实际寿命仅会导致提前维护（过度维护成本），惩罚相对较轻。

## 3. 边缘部署与量化 (Edge Deployment & Quantization)

为满足工业现场边缘计算设备（如工业 IPC、ARM 芯片）的低延迟部署需求，项目内置了**量化感知训练（QAT, Quantization-Aware Training）**管道：
- **INT8 精度转换**：通过插入伪量化节点（FakeQuantize）进行微调，将模型从 FP32 平滑过渡到 INT8 格式。
- **精度崩塌保护**：针对 Transformer 和 GNN 在量化时极易崩溃的痛点，对 Softmax、LayerNorm 等对精度极度敏感的算子实施了保护策略（强制回退为 Float32），确保模型性能。
- **算子融合 (Fusion)**：支持 Conv+ReLU 等算子合并，大幅降低边缘设备的内存访问延迟，实现推理加速和模型体积压缩。
