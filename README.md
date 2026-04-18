# 项目架构 (Project Architecture)

本项目是一个面向工业设备故障预测与健康管理（PHM, Prognostics and Health Management）的端到端深度学习框架。核心架构采用**时空图神经网络（ST-GNN）**与**Transformer**的混合模型，旨在同时解决设备退化阶段的分类问题和剩余使用寿命（RUL）的回归预测问题。

## 1. 核心模型架构 (Core Model Architecture)

模型主要分为四个核心模块，形成从局部特征提取到全局时序演化，再到多任务输出的完整流水线：

dynamic_graph.py
### 1.1 动态自适应图构建层 (Dynamic Adaptive Graph Construction)
摒弃了传统的静态先验物理图（如皮尔逊相关系数矩阵），创新性地引入了数据驱动的**动态自适应图拓扑结构**。
- 为每个特征通道（节点）引入可学习的高维嵌入向量 $E$。
- 通过计算 $A = \text{Softmax}(\text{ReLU}(E \cdot E^T))$ 动态生成节点间的注意力权重邻接矩阵。
- 能够自动挖掘隐含的物理耦合关系，并通过 ReLU 稀疏化网络去除负相关的噪声连接。

feature_extractor.py
### 1.2 时空特征主干网络 (Spatio-Temporal Backbone)
负责提取设备传感器数据在时间和空间（传感器节点）维度上的局部特征：
- **时间维度 (Temporal 1D-CNN)**：使用一维卷积网络沿着时间轴提取局部平滑特征，捕捉短期的退化趋势。
- **空间维度 (Spatial Dense GAT / GCN)**：基于动态生成的邻接矩阵，使用图注意力网络（DenseGATConv）或图卷积网络（GCN）进行节点间的空间消息传递与特征聚合。

model_architecture.py
### 1.3 全局时序演化层 (Global Transformer Encoder)
在提取时空局部特征后，利用 Transformer 架构捕捉长程的时序退化演化规律：
- 将空间聚合后的特征展平并进行线性投影。
- 注入正弦位置编码（Positional Encoding）以保留时间顺序信息。
- 通过多层 Transformer Encoder（多头自注意力机制）对整个时间窗内的全局特征进行深度建模。

mt_head.py
### 1.4 多任务预测头 (Multi-Task Prediction Head)
接收 Transformer 输出的全局时序特征，执行联合预测：
- **全局时序池化 (GAP)**：沿着时间轴取平均值，将时间窗内的特征浓缩为宏观的健康摘要。
- **共享特征降维 (Shared MLP)**：提炼高阶语义并降维。
- **双分支输出**：
  - **分支 A (分类头)**：预测设备当前的退化阶段（如：正常、轻微退化、严重退化）。
  - **分支 B (回归头)**：预测设备的剩余使用寿命百分比（RUL）。通过 Sigmoid 激活函数在架构层面强制保证预测值的物理合法性（严格限制在 $[0, 1]$ 之间）。

## 2. 训练与评估机制 (Training & Evaluation)

### 2.1 工业级非对称惩罚评估 (Asymmetric Scoring)
针对工业 PHM 领域的特殊需求，设计了非对称指数惩罚损失机制：
- **高估寿命惩罚**：模型预测寿命大于实际寿命会导致意外停机等灾难性后果，因此施加更严厉的指数惩罚。
- **低估寿命惩罚**：预测寿命小于实际寿命仅会导致提前维护（过度维护成本），惩罚相对较轻。

## 3. 边缘部署与量化 (Edge Deployment & Quantization)

为满足工业现场边缘计算设备（如工业 IPC、ARM 芯片）的低延迟部署需求，项目内置了**量化感知训练（QAT, Quantization-Aware Training）**管道：
- **INT8 精度转换**：通过插入伪量化节点（FakeQuantize）进行微调，将模型从 FP32 平滑过渡到 INT8 格式。
- **精度崩塌保护**：针对 Transformer 和 GNN 在量化时极易崩溃的痛点，对 Softmax、LayerNorm 等对精度极度敏感的算子实施了保护策略（强制回退为 Float32），确保模型性能。
- **算子融合 (Fusion)**：支持 Conv+ReLU 等算子合并，大幅降低边缘设备的内存访问延迟，实现推理加速和模型体积压缩。
