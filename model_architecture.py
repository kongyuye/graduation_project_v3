import torch
import torch.nn as pd
import torch.nn as nn
import torch.nn.functional as F

class DynamicGraphConstruction(nn.Module):
    """
    动态自适应图构建层 (Dynamic Adaptive Graph)
    根据 PDF 指南:
    通过为每一个节点引入一个可学习的高维嵌入向量 E，
    通过计算 A = Softmax(ReLU(E * E^T)) 动态生成节点间的注意力权重矩阵。
    """
    def __init__(self, num_nodes, embedding_dim=64):
        super(DynamicGraphConstruction, self).__init__()
        # E in R^{N x d}
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        nn.init.xavier_uniform_(self.node_embeddings)

    def forward(self):
        # E * E^T
        e_e_t = torch.matmul(self.node_embeddings, self.node_embeddings.transpose(0, 1))
        # A = Softmax(ReLU(E * E^T))
        adj = F.softmax(F.relu(e_e_t), dim=-1)
        return adj

class GraphConvolution(nn.Module):
    """
    基于谱图理论或空间消息传递的基础图卷积层
    这里实现一个基于动态邻接矩阵的标准 GCN 层
    """
    def __init__(self, in_channels, out_channels):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x shape: (Batch, Nodes, in_channels)
        # adj shape: (Nodes, Nodes)
        support = torch.matmul(x, self.weight) # (Batch, Nodes, out_channels)
        # 节点间消息传递
        output = torch.einsum('ij, bjc -> bic', adj, support) + self.bias
        return F.relu(output)

class STGNN_Block(nn.Module):
    """
    时空图卷积模块
    PDF 指南中提到：局部特征提取层(1D-CNN) + 空间拓扑聚合层(GCN)
    """
    def __init__(self, in_channels, spatial_channels, num_nodes):
        super(STGNN_Block, self).__init__()
        # 局部特征提取层 (沿时间轴 1D-CNN) - PDF 中推荐多尺度，这里先用基础 1D CNN 示例
        self.temporal_conv = nn.Conv2d(in_channels, spatial_channels, kernel_size=(1, 3), padding=(0, 1))
        # 图空间聚合层
        self.dynamic_graph = DynamicGraphConstruction(num_nodes)
        self.gcn = GraphConvolution(spatial_channels, spatial_channels)
        
    def forward(self, x):
        # x shape: (Batch, in_channels, Nodes, Time_Steps)
        
        # 1. 局部时序特征提取
        x = self.temporal_conv(x) # (B, spatial_channels, N, T)
        
        B, C, N, T = x.shape
        # 调整形状以适应 GCN: 把时间步视为 Batch 的一部分 -> (B*T, N, C)
        x_gcn = x.permute(0, 3, 2, 1).contiguous().view(B * T, N, C)
        
        # 2. 动态生成邻接矩阵
        adj = self.dynamic_graph() # (N, N)
        
        # 3. 空间特征聚合
        x_gcn = self.gcn(x_gcn, adj) # (B*T, N, C)
        
        # 恢复形状: (B, C, N, T)
        out = x_gcn.view(B, T, N, C).permute(0, 3, 2, 1)
        return out

class TransformerTemporalModule(nn.Module):
    """
    全局时序演化层 (Transformer Encoder)
    PDF 指南要求：引入正弦位置编码；堆叠 3 层 Encoder Block；隐藏层维度 256；8头注意力。
    但为了防止小数据集过拟合，我们适度降低了层数和维度。
    """
    def __init__(self, feature_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.3):
        super(TransformerTemporalModule, self).__init__()
        # 将图卷积输出的特征维度映射到 d_model
        self.input_projection = nn.Linear(feature_dim, d_model)
        
        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True # PyTorch 1.9+ 支持 batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.d_model = d_model

    def forward(self, x):
        # x shape 预期为: (Batch, Time_Steps, Feature_Dim)
        x = self.input_projection(x) # (B, T, d_model)
        
        # TODO: 可以加入正弦位置编码 (Positional Encoding)
        # 这里简化处理，直接传入 Transformer
        out = self.transformer_encoder(x) # (B, T, d_model)
        return out

# =====================================================================
# 新增：域适应模块 (Domain Adaptation) 与梯度反转层 (GRL)
# =====================================================================
class GradientReversalFunction(torch.autograd.Function):
    """
    梯度反转函数 (Gradient Reversal Function)
    前向传播时，特征原样通过；
    反向传播时，梯度乘以一个负标量 (-alpha)，实现特征提取器与域分类器的对抗训练。
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    """
    梯度反转层 (Gradient Reversal Layer, GRL)
    包装 GradientReversalFunction 以便作为普通的 nn.Module 插入网络中。
    """
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class PredictionHead(nn.Module):
    """
    多任务预测输出层 + 域自适应对抗头
    包含：
    1. 共享的特征降维 MLP
    2. RUL 回归分支 / 状态分类分支 (可选)
    3. [新增] 域对抗分类分支 (Domain Classifier)，前面接入 GRL
    """
    # 增加 Dropout 防止过拟合，并缩小全连接层的神经元数量
    def __init__(self, d_model, num_classes=3, num_domains=3, dropout_rate=0.3):
        super(PredictionHead, self).__init__()
        # 将 d_model (当前为64) 降维到 32
        self.fc1 = nn.Linear(d_model, 32)
        self.fc2 = nn.Linear(32, 16)
        
        # 增加 Dropout 层以防止过拟合
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # === 核心任务分支 (Main Task Branches) ===
        # 分支 A: 状态分类头 (例如: 正常、轻微退化、严重退化)
        self.classifier = nn.Linear(16, num_classes)
        # 分支 B: 回归头 (预测 RUL 百分比或绝对值)
        self.regressor = nn.Linear(16, 1)
        
        # === 域适应对抗分支 (Domain Adversarial Branch) ===
        # 梯度反转层，alpha 参数可以在训练时根据 epoch 动态调整，默认 1.0
        self.grl = GradientReversalLayer(alpha=1.0)
        # 域分类器：通过多层感知机判断当前特征属于哪种工况/域
        self.domain_classifier = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(8, num_domains) # 输出每种工况的 logits
        )
        
    def forward(self, x, alpha=1.0):
        """
        :param x: 形状 (Batch, Time_Steps, d_model)
        :param alpha: 用于 GRL 层的动态反转权重
        """
        # 1. 全局平均池化 (GAP) - 沿着时间轴平均，提取全局健康摘要
        x_gap = torch.mean(x, dim=1) # (Batch, d_model)
        
        # 2. 共享特征降维 (Shared MLP)
        feat = F.relu(self.fc1(x_gap))
        feat = self.dropout(feat)
        feat = F.relu(self.fc2(feat))
        feat = self.dropout(feat) # shape: (Batch, 64)
        
        # 3. 核心任务预测 (无梯度反转，正常更新特征提取器)
        logits = self.classifier(feat)
        rul_pred = self.regressor(feat)
        
        # 4. 域对抗预测 (经过 GRL，反向传播时将域分类器的梯度取反传给前面网络)
        # 更新 GRL 层的 alpha 值 (支持训练时动态调节退火)
        self.grl.alpha = alpha
        domain_feat = self.grl(feat)
        domain_logits = self.domain_classifier(domain_feat)
        
        return logits, rul_pred, domain_logits, feat

class STGNN_Transformer_Model(nn.Module):
    """
    端到端混合架构模型 (End-to-End Model)
    整合: ST-GNN (提取时空局部特征) -> Flatten -> Transformer (全局时序) -> Prediction Head (带域对抗)
    """
    # 调低默认 d_model=64 以降低参数量
    def __init__(self, in_channels=1, num_nodes=14, gcn_out_channels=32, 
                 d_model=64, nhead=4, num_layers=2, num_classes=3, num_domains=3):
        super(STGNN_Transformer_Model, self).__init__()
        
        # 1. 空间拓扑聚合 (含局部特征提取)
        self.stgnn = STGNN_Block(in_channels, gcn_out_channels, num_nodes)
        
        # 将 GCN 输出的多节点特征展平，准备输入 Transformer
        transformer_input_dim = num_nodes * gcn_out_channels
        
        # 2. 全局时序演化
        self.transformer = TransformerTemporalModule(
            feature_dim=transformer_input_dim, 
            d_model=d_model, 
            nhead=nhead, 
            num_layers=num_layers
        )
        
        # 3. 多任务预测头 (带域适应对抗)
        self.prediction_head = PredictionHead(d_model, num_classes, num_domains)

    def forward(self, x, alpha=1.0):
        # x shape: (Batch, Channels=1, Nodes=14, Time_Steps=30)
        
        # --- ST-GNN 阶段 ---
        x = self.stgnn(x) # (B, gcn_out_channels, N, T)
        
        # --- 维度重塑阶段 ---
        B, C, N, T = x.shape
        x = x.permute(0, 3, 2, 1).contiguous() # (B, T, N, C)
        x = x.view(B, T, N * C) # (B, T, N*C)
        
        # --- Transformer 阶段 ---
        x = self.transformer(x) # (B, T, d_model)
        
        # --- 预测阶段 ---
        # 提取出了 domain_logits 和深层特征 feat (用于可能的 MMD loss 计算)
        logits, rul, domain_logits, feat = self.prediction_head(x, alpha=alpha)
        
        return logits, rul, domain_logits, feat

# 测试模型维度的简单代码
if __name__ == "__main__":
    # 模拟我们预处理好的数据形状: (Batch, C, V, T) -> (32, 1, 14, 30)
    batch_size = 32
    channels = 1
    nodes = 14
    time_steps = 30
    
    dummy_input = torch.randn(batch_size, channels, nodes, time_steps)
    
    print(f"输入数据形状: {dummy_input.shape}")
    
    # 初始化模型
    model = STGNN_Transformer_Model(in_channels=channels, num_nodes=nodes)
    
    # 前向传播 (alpha 控制域对抗强度)
    logits, rul_pred, domain_logits, feat = model(dummy_input, alpha=1.0)
    
    print(f"分类输出形状: {logits.shape}")
    print(f"回归输出形状: {rul_pred.shape}")
    print(f"域分类输出形状: {domain_logits.shape}")
    print(f"全局特征形状: {feat.shape}")

# =====================================================================
# 辅助代码：损失函数计算逻辑示例 (Loss Function Example)
# 包含 RUL 回归、分类、域对抗以及 MMD 的组合应用
# =====================================================================
def compute_mmd_loss(source_features, target_features, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算源域和目标域特征的最大均值差异 (MMD Loss)。
    通过多核 (Multi-kernel) 映射到 RKHS 空间拉近两者分布。
    """
    n_s = source_features.shape[0]
    n_t = target_features.shape[0]
    n_samples = n_s + n_t
    
    # 将源域和目标域特征拼接
    total = torch.cat([source_features, target_features], dim=0)
    
    # 计算 L2 距离矩阵
    total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.shape[1])
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.shape[1])
    L2_distance = ((total0 - total1) ** 2).sum(2)
    
    # 计算核宽度带宽 (Bandwidth)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    
    # 多核计算
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    kernels = sum(kernel_val)
    
    # 根据 MMD 公式拆分矩阵
    XX = kernels[:n_s, :n_s]
    YY = kernels[n_s:, n_s:]
    XY = kernels[:n_s, n_s:]
    YX = kernels[n_s:, :n_s]
    
    # MMD 损失计算
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
    return loss

def example_train_step(model, source_data, source_rul_labels, source_domain_labels, target_data, target_domain_labels, alpha=1.0):
    """
    极小化-极大化 (Min-Max) 对抗博弈训练 伪代码示例。
    同时实现了 GRL 域对抗和 MMD 特征对齐。
    """
    model.train()
    
    # 1. 前向传播：源域数据
    src_logits, src_rul, src_domain_logits, src_feat = model(source_data, alpha=alpha)
    # 2. 前向传播：目标域数据 (通常无 RUL 标签)
    tgt_logits, tgt_rul, tgt_domain_logits, tgt_feat = model(target_data, alpha=alpha)
    
    # ================= 极小化 (Minimization) 部分 =================
    # 主任务 Loss：预测源域的 RUL (均方误差 MSE)
    loss_rul = nn.MSELoss()(src_rul.squeeze(), source_rul_labels)
    
    # (可选) 局部特征对齐：MMD Loss 防治负迁移
    # 为了实现子域适应 (Subdomain Adaptation)，可以在这里根据 src_logits/tgt_logits 的预测伪标签，
    # 将属于同一退化阶段的特征传入 compute_mmd_loss()。这里展示全局 MMD 示例：
    loss_mmd = compute_mmd_loss(src_feat, tgt_feat)
    
    # ================= 极大化 (Maximization) 部分 (通过 GRL 隐式实现) =================
    # 域对抗 Loss：判别器试图正确分类源域和目标域 (交叉熵)
    # 但由于 GRL 的存在，反向传播时主干网络收到的梯度是相反的，
    # 从而主干网络被优化为“欺骗”判别器，使得特征不再包含工况特异性信息。
    domain_logits_concat = torch.cat([src_domain_logits, tgt_domain_logits], dim=0)
    domain_labels_concat = torch.cat([source_domain_labels, target_domain_labels], dim=0)
    loss_domain = nn.CrossEntropyLoss()(domain_logits_concat, domain_labels_concat)
    
    # 总 Loss 组合
    # 在这个组合中：
    # 1. 优化器更新 domain_classifier 时，loss_domain 使其具备判别工况的能力。
    # 2. 优化器更新 Feature Extractor 时，由于 GRL，loss_domain 的梯度反转，抹除工况痕迹；
    #    同时 loss_rul 和 loss_mmd 引导提取出有利于寿命预测且对齐的共有特征。
    lambda_domain = 1.0  # 域对抗损失权重
    lambda_mmd = 0.5     # MMD 损失权重
    total_loss = loss_rul + lambda_domain * loss_domain + lambda_mmd * loss_mmd
    
    # 正常进行反向传播即可
    # total_loss.backward()
    # optimizer.step()
    
    return total_loss
