import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from torch_geometric.nn import DenseGATConv

# ==========================================
# 子模块 1: 局部特征提取层 (Temporal 1D-CNN)
# ==========================================
class TemporalCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        """
        沿着时间轴提取局部平滑特征
        """
        super(TemporalCNN, self).__init__()
        # 使用 1D 卷积，保持时间步长 T 不变 (padding = kernel_size // 2)
        self.conv1d = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2
        )
        # 增加 BatchNorm1d 层以加速收敛并防止过拟合
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 初始输入特征, Shape: [B, T, N, C_in]
        :return: CNN 提取后的特征, Shape: [B, T, N, C_cnn]
        """
        # --- Shape Transition ---
        # 1. 初始输入: [B, T, N, C_in]
        # 2. 融合 B 和 N 以适应 1D-CNN (要求输入为 [Batch, Channels, Length])
        # 使用 einops.rearrange 优雅且安全地重构维度
        # Shape 变为: [B*N, C_in, T]
        x_cnn = rearrange(x, 'b t n c -> (b n) c t')
        
        # 3. 通过 1D 卷积层、BatchNorm 和激活层
        # Shape 变为: [B*N, C_cnn, T]
        x_cnn = self.conv1d(x_cnn)
        x_cnn = self.bn(x_cnn)  # 应用 BatchNorm1d
        x_cnn = self.activation(x_cnn)
        
        # 4. 恢复为四维张量
        # 获取原始的 B 和 N 维度信息用于重构
        B, T, N, _ = x.shape
        # Shape 变为: [B, T, N, C_cnn]
        out = rearrange(x_cnn, '(b n) c t -> b t n c', b=B, n=N)
        
        return out


# ==========================================
# 子模块 2: 空间拓扑聚合层 (Spatial Dense GAT)
# ==========================================
class SpatialDenseGAT(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4):
        """
        基于 DenseGATConv 的空间图注意力聚合
        """
        super(SpatialDenseGAT, self).__init__()
        # 使用 PyG 提供的 DenseGATConv 处理稠密邻接矩阵
        # concat=False 意味着多头注意力结果会求平均而不是拼接，从而保持输出通道数为 out_channels
        self.gat = DenseGATConv(
            in_channels=in_channels, 
            out_channels=out_channels, 
            heads=heads, 
            concat=False
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        :param x: 节点特征, Shape: [B, T, N, C_cnn]
        :param adj: 动态邻接矩阵, Shape: [B, N, N] 或 [N, N]
        :return: 空间聚合后的特征, Shape: [B, T, N, d_gcn]
        """
        B, T, N, C = x.shape
        
        # --- Shape Transition ---
        # 1. DenseGATConv 期望的特征输入 Shape 为 [Batch, Max_Nodes, Features]
        # 由于我们有时间维度 T，我们需要将 B 和 T 融合作为逻辑上的 "Batch"
        # Shape 变为: [B*T, N, C_cnn]
        x_flat = rearrange(x, 'b t n c -> (b t) n c')
        
        # 2. 邻接矩阵广播 (Broadcasting)
        # 如果 adj 是全局共享的 [N, N]，需要扩展为 [B*T, N, N]
        if adj.dim() == 2:
            # [N, N] -> [1, N, N] -> [B*T, N, N]
            adj_batched = adj.unsqueeze(0).expand(B * T, N, N)
        # 如果 adj 是针对每个 Batch 动态生成的 [B, N, N]
        elif adj.dim() == 3:
            # 沿着时间轴 T 复制邻接矩阵
            # [B, N, N] -> [B, 1, N, N] -> [B, T, N, N] -> [B*T, N, N]
            adj_batched = adj.unsqueeze(1).expand(B, T, N, N)
            adj_batched = rearrange(adj_batched, 'b t n1 n2 -> (b t) n1 n2')
        else:
            raise ValueError(f"邻接矩阵维度异常，期望 2 或 3，实际为 {adj.dim()}")

        # 3. 通过 Dense GAT 层
        # mask=None 表示所有节点都是真实的（没有 padding 节点）
        # Shape 变为: [B*T, N, d_gcn]
        out_flat = self.gat(x_flat, adj_batched, mask=None)
        out_flat = self.activation(out_flat)
        
        # 4. 恢复为四维张量
        # Shape 变为: [B, T, N, d_gcn]
        out = rearrange(out_flat, '(b t) n d -> b t n d', b=B, t=T)
        
        return out


# ==========================================
# 子模块 3: 全局时序演化层 (Transformer Encoder)
# ==========================================
class PositionalEncoding(nn.Module):
    """
    简易正弦位置编码
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Shape: [1, max_len, d_model] 方便与 [B, T, d_model] 广播相加
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Shape [B, T, d_model]
        """
        # 取出当前序列长度 T 对应的位置编码并相加
        return x + self.pe[:, :x.size(1), :]


class GlobalTransformer(nn.Module):
    def __init__(self, in_features: int, d_model: int = 256, nhead: int = 8, num_layers: int = 3):
        """
        捕捉长程时序退化演化规律
        """
        super(GlobalTransformer, self).__init__()
        self.d_model = d_model
        
        # 将展平后的空间特征映射到 Transformer 隐藏维度
        self.input_projection = nn.Linear(in_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        
        # Transformer Encoder
        # batch_first=True 表示输入张量的第一个维度是 Batch size
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 空间聚合后的特征, Shape: [B, T, N, d_gcn]
        :return: 时序演化后的全局特征, Shape: [B, T, d_model]
        """
        B, T, N, d_gcn = x.shape
        
        # --- Shape Transition ---
        # 1. 展平空间维度 (将所有节点的特征拼接到一起)
        # Shape 变为: [B, T, N * d_gcn]
        x_flat = rearrange(x, 'b t n d -> b t (n d)')
        
        # 2. 线性投影到 d_model
        # Shape 变为: [B, T, d_model]
        x_proj = self.input_projection(x_flat)
        
        # 3. 注入位置编码
        # Shape 保持: [B, T, d_model]
        x_pe = self.pos_encoder(x_proj)
        
        # 4. 送入 Transformer Encoder
        # Shape 保持: [B, T, d_model]
        out = self.transformer_encoder(x_pe)
        
        return out


# ==========================================
# 混合架构拼装: SpatioTemporalBackbone
# ==========================================
class SpatioTemporalBackbone(nn.Module):
    """
    ST-GNN + Transformer 主干网络
    """
    def __init__(self, 
                 num_nodes: int = 8, 
                 c_in: int = 1, 
                 c_cnn: int = 32, 
                 d_gcn: int = 64, 
                 d_model: int = 128, 
                 tf_heads: int = 4, 
                 tf_layers: int = 2):
        super(SpatioTemporalBackbone, self).__init__()
        
        # 1. 局部特征提取 (1D-CNN)
        self.temporal_cnn = TemporalCNN(in_channels=c_in, out_channels=c_cnn)
        
        # 2. 空间拓扑聚合 (Dense GAT)
        self.spatial_gat = SpatialDenseGAT(in_channels=c_cnn, out_channels=d_gcn, heads=4)
        
        # 3. 全局时序演化 (Transformer Encoder)
        # 展平后的特征维度为 N * d_gcn
        self.global_transformer = GlobalTransformer(
            in_features=num_nodes * d_gcn, 
            d_model=d_model, 
            nhead=tf_heads, 
            num_layers=tf_layers
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播流水线
        :param x: 原始特征张量, Shape: [B, T, N, C_in]
        :param adj: 动态生成的邻接矩阵, Shape: [B, N, N] 或 [N, N]
        :return: 最终的时空混合特征表示, Shape: [B, T, d_model]
        """
        # =======================================================
        # Step 1: Temporal 1D-CNN (局部特征提取)
        # Input Shape:  [B, T, N, C_in]
        # Output Shape: [B, T, N, C_cnn]
        # =======================================================
        h_cnn = self.temporal_cnn(x)
        
        # =======================================================
        # Step 2: Spatial Dense GAT (空间图聚合)
        # Input Shape:  [B, T, N, C_cnn]  & adj: [B, N, N]
        # Output Shape: [B, T, N, d_gcn]
        # =======================================================
        h_gcn = self.spatial_gat(h_cnn, adj)
        
        # =======================================================
        # Step 3: Global Transformer (全局长程时序演化)
        # Input Shape:  [B, T, N, d_gcn]
        # Output Shape: [B, T, d_model]
        # =======================================================
        out = self.global_transformer(h_gcn)
        
        return out


# ==========================================
# 验证与测试代码
# ==========================================
def test_spatiotemporal_backbone():
    print("=== 开始测试 SpatioTemporalBackbone ===")
    
    # 模拟超参数
    B = 16        # 批次大小
    T = 50        # 时间步长 (例如 50 个连续时刻的特征)
    N = 8         # 节点数 (例如 8 个 WPT 频带)
    C_in = 1      # 初始特征数 (通常单个统计值为 1)
    
    # 模拟输入张量
    x_mock = torch.randn(B, T, N, C_in)
    print(f"1. 模拟输入张量 X 形状: {x_mock.shape} -> [B, T, N, C_in]")
    
    # 模拟由阶段二生成的稠密邻接矩阵 (这里用随机 Softmax 模拟)
    adj_mock = F.softmax(torch.randn(N, N), dim=-1)
    print(f"2. 模拟动态邻接矩阵 A 形状: {adj_mock.shape} -> [N, N]")
    
    # 实例化主干网络
    model = SpatioTemporalBackbone(
        num_nodes=N,
        c_in=C_in,
        c_cnn=32,
        d_gcn=64,
        d_model=256,
        tf_heads=8,
        tf_layers=2
    )
    
    # 前向传播测试
    out = model(x_mock, adj_mock)
    
    print(f"3. 最终输出张量形状: {out.shape} -> [B, T, d_model]")
    assert out.shape == (B, T, 256), "输出维度与预期不符！"
    print("=== 测试通过！流水线维度流转严密准确 ===")

if __name__ == "__main__":
    test_spatiotemporal_backbone()
