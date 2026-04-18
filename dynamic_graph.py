import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from typing import Tuple
import config

class DynamicGraphConstructor(nn.Module):
    """
    动态自适应图拓扑构建器
    在 ST-GNN（时空图神经网络）中，通过数据驱动的方式自动学习各特征通道（节点）间的隐含物理耦合关系。
    放弃了静态的先验物理图（如皮尔逊相关系数矩阵），转而采用可学习的高维节点嵌入。
    """
    def __init__(self, num_nodes: int = 8, embed_dim: int = 10, alpha: float = 3.0):
        """
        初始化自适应图生成器
        :param num_nodes: 图的节点数 N，即特征通道数（例如 WPT 小波包分解的 8 个频带，默认 8）
        :param embed_dim: 节点的可学习嵌入维度 d（默认 10）
        :param alpha: 控制 Softmax 温度或 ReLU 后缩放的超参数，防止注意力过度平滑（可选）
        """
        super(DynamicGraphConstructor, self).__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.alpha = alpha
        
        # 1. 可学习的节点嵌入 (Learnable Node Embeddings)
        # 初始化一个形状为 [N, d] 的随机矩阵 E，表示 N 个节点在 d 维隐藏空间中的特征表达。
        # 使用 nn.Parameter 包装，使其成为网络的可训练权重，能够被优化器（如 Adam）追踪和更新，
        # 并自动继承模型所在设备（CPU/CUDA）的属性。
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))
        
        # 使用 Xavier 初始化或者正态分布初始化有助于训练初期的稳定性
        nn.init.xavier_uniform_(self.node_embeddings)

    def forward(self) -> torch.Tensor:
        """
        前向传播：动态生成邻接矩阵
        :return: 密集型注意力邻接矩阵 A, 形状为 [N, N]
        """
        # E: [N, d]
        E = self.node_embeddings
        
        # 1. 计算内积相似度：E 乘以 E的转置 -> [N, d] @ [d, N] = [N, N]
        # 结果矩阵的元素 (i, j) 代表了节点 i 和节点 j 在隐藏空间中的关联强度。
        similarity = torch.matmul(E, E.t())
        
        # 2. ReLU 稀疏化网络
        # 去除负相关的噪声连接，强制网络只关注有正向协同作用的节点关系。
        # 形状保持不变: [N, N]
        sparse_sim = F.relu(similarity)
        
        # 3. Softmax 概率归一化
        # 在行维度 (dim=-1 或 dim=1) 上应用 Softmax，确保对于任何节点 i，
        # 它分配给所有节点 j 的注意力权重之和等于 1。
        # 这是典型的自注意力（Self-Attention）归一化方式，便于后续的图卷积聚合信息。
        adj_matrix = F.softmax(sparse_sim * self.alpha, dim=-1)
        
        return adj_matrix

    def get_pyg_graph(self, threshold: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        辅助函数：将密集稠密矩阵转换为 PyTorch Geometric (PyG) 所需的稀疏图格式。
        :param threshold: 边权重阈值，低于该值的微弱连接将被硬截断丢弃，以进一步提高图的稀疏性。
        :return: 
            edge_index: 边索引张量，形状为 [2, E] (E为边数)，数据类型为 torch.long
            edge_weight: 边权重张量，形状为 [E]
        """
        # 1. 首先获取当前网络计算出的稠密邻接矩阵 A -> [N, N]
        A = self.forward()
        
        # 2. 如果不需要梯度反向传播给 PyG 构建步骤，或者只是做前向预测，可以考虑 detach，
        #    但由于我们需要通过 GNN 的损失反向传播来更新节点嵌入 E，
        #    所以此处的 A 必须保留梯度（不能 detach！）。
        
        # 3. 根据阈值生成布尔掩码矩阵（哪些连接被保留）
        # mask 形状: [N, N]
        mask = A > threshold
        
        # 4. 获取满足条件（权重非零）的非零元素坐标
        # nonzero() 返回形状为 [E, 2] 的张量，其中每一行是一个 (row, col) 坐标对。
        # as_tuple=False 兼容旧版本，t() 转置后变成 [2, E] 以满足 PyG edge_index 规范。
        # 此处 edge_index 的数据类型默认为 torch.long
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()
        
        # 5. 根据行、列坐标索引提取对应的边权重
        # 形状: [E]
        # 注意：此处切片操作能够保留张量的梯度追踪（requires_grad=True）
        edge_weight = A[edge_index[0], edge_index[1]]
        
        return edge_index, edge_weight


def test_dynamic_graph_with_data():
    """
    测试驱动函数：
    模拟从 /Users/wanglixiao/Desktop/大学/大四上/毕设/newproduct6/processed_data/train 
    中加载特征数据，并验证 DynamicGraphConstructor 模块的正确性。
    """
    print("=== 开始测试动态自适应图构建器 ===")
    
    # 1. 模拟设备选择
    device = config.DEVICE
    print(f"使用计算设备: {device}")
    
    # 2. 假设我们提取了 WPT 的 8 个频带作为特征通道 (N=8)
    # 这些通道将作为图神经网络中的 8 个物理空间节点
    num_nodes = 8
    # 假设每个节点的嵌入维度为 10
    # 这里可以根据实际情况调整，例如 128, 256 等
    # 这个维度将作为图神经网络中的隐藏层维度，影响模型的表达能力
    embed_dim = 10
    
    # 3. 实例化图构建器并放置到指定设备上
    graph_constructor = DynamicGraphConstructor(num_nodes=num_nodes, embed_dim=embed_dim).to(device)
    
    # 验证 Parameter 所在的设备
    print(f"节点嵌入矩阵 E 所在的设备: {graph_constructor.node_embeddings.device}")
    
    # 4. 执行前向传播，获取稠密邻接矩阵
    # 自动执行DynamicGraphConstructor类中的前向传播方法
    adj_matrix = graph_constructor()
    print(f"\n稠密邻接矩阵 A 的形状: {adj_matrix.shape}")
    print(f"邻接矩阵第一行的权重之和 (应该等于 1.0): {adj_matrix[0].sum().item():.4f}")
    
    # 5. 测试 PyG 接口转换
    # 设定一个截断阈值，例如 1/(2*N)，过滤掉远低于平均水平的噪声连接
    # 可以理解为邻接表
    threshold = 1.0 / (2 * num_nodes)
    edge_index, edge_weight = graph_constructor.get_pyg_graph(threshold=threshold)
    
    print(f"\nPyG 稀疏图格式测试:")
    print(f"使用的阈值: {threshold:.4f}")
    print(f"边索引 (edge_index) 形状: {edge_index.shape} (保留了 {edge_index.shape[1]} 条边)")
    print(f"边权重 (edge_weight) 形状: {edge_weight.shape}")
    print(f"边索引数据类型: {edge_index.dtype}")
    
    # ==========================================
    # 新增：可视化生成的动态图结构
    # ==========================================
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        print("\n=== 正在生成图结构可视化 ===")
        
        # 1. 创建有向图
        G = nx.DiGraph()
        
        # 2. 添加节点
        for i in range(num_nodes):
            G.add_node(i, label=f"Node {i}")
            
        # 3. 添加边和权重
        edges = edge_index.t().cpu().numpy()
        weights = edge_weight.detach().cpu().numpy()
        
        for i in range(len(edges)):
            src, dst = edges[i]
            w = weights[i]
            G.add_edge(src, dst, weight=w)
            
        # 4. 绘制图形
        plt.figure(figsize=(10, 8))
        
        # 使用圆形布局让节点均匀分布
        pos = nx.circular_layout(G)
        
        # 获取边权重以决定线条粗细
        edge_widths = [G[u][v]['weight'] * 10 for u, v in G.edges()]
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=1500, edgecolors='darkblue')
        
        # 绘制边 (根据权重调整粗细和透明度)
        nx.draw_networkx_edges(G, pos, width=edge_widths, 
                             alpha=0.6, edge_color='gray', 
                             arrowsize=20, connectionstyle='arc3,rad=0.1')
        
        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # 绘制边权重标签 (保留两位小数)
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
        
        plt.title(f"Dynamic Adaptive Graph (Threshold: {threshold:.4f})", fontsize=16)
        plt.axis('off')
        
        # 保存图片而不是阻塞终端
        save_path = "graph_visualization.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图结构已成功可视化并保存至: {os.path.abspath(save_path)}")
        plt.close()
        
    except ImportError:
        print("\n[提示] 如果想看到图的可视化，请安装 networkx 和 matplotlib:")
        print("pip install networkx matplotlib")
    
    # 6. 读取真实数据验证维度逻辑
    data_dir = "/Users/wanglixiao/Desktop/大学/大四上/毕设/newproduct6/processed_data/train"
    if os.path.exists(data_dir):
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        if csv_files:
            sample_file = csv_files[0]
            print(f"\n尝试读取真实训练数据: {os.path.basename(sample_file)}")
            df = pd.read_csv(sample_file)
            
            # 例如：提取水平振动的 WPT 特征（8个频带）
            wpt_cols = [f'h_wpt_energy_ratio_{i}' for i in range(8)]
            if all(c in df.columns for c in wpt_cols):
                # 获取一个批次的数据 [Batch_size, Num_nodes]
                batch_data = df[wpt_cols].values
                batch_tensor = torch.tensor(batch_data, dtype=torch.float32).to(device)
                
                print(f"成功提取节点特征张量 X, 形状为 [Batch, N]: {batch_tensor.shape}")
                print(f"这些特征 X 可以与图结构 (edge_index, edge_weight) 一起送入 PyG 的 GCNConv 中计算了！")
            else:
                print("未在数据中找到预期的 WPT 特征列。")
    
    print("\n=== 测试通过！模块运行正常 ===")

if __name__ == "__main__":
    test_dynamic_graph_with_data()
