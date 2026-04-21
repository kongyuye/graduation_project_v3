import torch
import torch.nn as nn
from typing import Dict
import config

class MultiTaskHead(nn.Module):
    """
    电子设备故障预测（PHM）多任务预测头
    接收 ST-GNN + Transformer 主干网络输出的时序特征，执行：
    1. 全局时序池化 (GAP) 与 共享特征降维 (Shared MLP)
    2. 双分支联合预测：退化阶段分类 (Classification) 与 剩余寿命预测 (RUL Regression)
    """
    def __init__(self, 
                 d_model: int = 256, 
                 hidden_dim: int = 128, 
                 num_classes: int = 3, 
                 dropout_prob: float = config.DROPOUT):
        """
        初始化多任务预测头
        :param d_model: 主干网络输出的特征维度 (例如 256)
        :param hidden_dim: 共享 MLP 降维后的隐藏层维度 (例如 128)
        :param num_classes: 退化阶段的分类类别数 (默认 3: 正常, 轻微退化, 严重退化)
        :param dropout_prob: 防止过拟合的 Dropout 概率
        """
        super(MultiTaskHead, self).__init__()
        
        # ==========================================
        # 1. 共享特征压缩层 (Shared MLP)
        # 负责将时序池化后的宏观摘要进一步提炼并降维
        # ==========================================
        self.shared_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),  # 相比 ReLU，GELU 在 Transformer 架构后通常有更平滑的梯度表现
            nn.Dropout(p=dropout_prob)
        )
        
        # ==========================================
        # 2. 分支 A：退化阶段分类头 (Classification Head)
        # ==========================================
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # ==========================================
        # 3. 分支 B：剩余寿命回归头 (Regression Head)
        # ==========================================
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            # 选择 Sigmoid 的理由：
            # RUL（剩余寿命百分比）的物理定义域严格限制在 [0.0, 1.0] 之间（0% ~ 100%）。
            # 相比于 ReLU（仅限制非负但无上界，可能输出 > 1 的无意义值），
            # Sigmoid 能在架构层面强制保证预测值的物理合法性，稳定回归损失（如 MSELoss 或 L1Loss）。
            # 注意：如果实际训练中由于 Sigmoid 两端梯度消失导致 RUL 难以收敛到 0 或 1，
            #      后续可以考虑改为无限制输出配合 Huber Loss，或扩大 Sigmoid 范围。
            nn.Sigmoid() 
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        :param x: 主干网络传来的特征张量, Shape: [B, T, d_model]
        :return: 包含 'class_logits' 和 'rul_pred' 的字典
        """
        # --- 输入 ---
        # x Shape: [B, T, d_model]
        
        # ==========================================
        # Step 1: 时序全局平均池化 (Global Average Pooling, GAP)
        # ==========================================
        # 沿着时间轴 T (dim=1) 取平均值，将时间窗内的特征浓缩为一个宏观健康摘要
        # x Shape: [B, T, d_model] -> [B, d_model]
        pooled_feat = torch.mean(x, dim=1)
        
        # ==========================================
        # Step 2: 共享多层感知机降维 (Shared MLP)
        # ==========================================
        # 提炼高阶语义并降维
        # pooled_feat Shape: [B, d_model] -> [B, hidden_dim]
        shared_feat = self.shared_mlp(pooled_feat)
        
        # ==========================================
        # Step 3: 双分支联合输出 (Dual-Branch Output)
        # ==========================================
        
        # 分支 A: 分类头
        # 直接输出 Logits，不加 Softmax，以适配 nn.CrossEntropyLoss
        # class_logits Shape: [B, hidden_dim] -> [B, num_classes]
        class_logits = self.classifier(shared_feat)
        
        # 分支 B: 回归头
        # 预测剩余寿命百分比，由 Sigmoid 约束在 [0, 1] 范围内
        # rul_pred Shape: [B, hidden_dim] -> [B, 1]
        rul_pred = self.regressor(shared_feat)
        
        # 返回字典以保证语义清晰，调用者可以通过键名精确获取对应预测
        return {
            'class_logits': class_logits,
            'rul_pred': rul_pred
        }


# ==========================================
# 测试与验证代码
# ==========================================
def test_multi_task_head():
    print("=== 开始测试多任务预测头 MultiTaskHead ===")
    
    # 模拟主干网络输出的超参数
    B = 32            # 批次大小
    T = 64            # 时间步长
    d_model = 256     # Transformer 隐藏层维度
    hidden_dim = 128  # 共享层降维后维度
    num_classes = 3   # 类别数
    
    # 1. 模拟主干网络传来的特征张量
    mock_input = torch.randn(B, T, d_model)
    print(f"1. 模拟输入张量 X 形状: {mock_input.shape} -> [B, T, d_model]")
    
    # 2. 实例化多任务预测头
    mt_head = MultiTaskHead(
        d_model=d_model, 
        hidden_dim=hidden_dim, 
        num_classes=num_classes,
        dropout_prob=0.3
    )
    
    # 3. 前向传播
    outputs = mt_head(mock_input)
    
    # 4. 验证分类分支
    logits = outputs['class_logits']
    print(f"2. 分类分支输出形状 (class_logits): {logits.shape} -> [B, num_classes]")
    assert logits.shape == (B, num_classes), "分类输出维度错误！"
    
    # 5. 验证回归分支
    rul = outputs['rul_pred']
    print(f"3. 回归分支输出形状 (rul_pred): {rul.shape} -> [B, 1]")
    assert rul.shape == (B, 1), "回归输出维度错误！"
    
    # 验证回归值范围是否被约束在 [0, 1] 之间 (因为加了 Sigmoid)
    min_val, max_val = rul.min().item(), rul.max().item()
    print(f"   RUL 预测值范围: [{min_val:.4f}, {max_val:.4f}] (预期在 0.0 到 1.0 之间)")
    assert 0.0 <= min_val <= 1.0 and 0.0 <= max_val <= 1.0, "回归值超出了 [0, 1] 范围！"
    
    print("=== 测试通过！张量 Shape 流转与物理约束严密准确 ===")

if __name__ == "__main__":
    test_multi_task_head()
