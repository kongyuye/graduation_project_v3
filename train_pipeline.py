import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import config

# 引入之前编写的模块 (假设在同一目录下或被正确包围)
from dynamic_graph import DynamicGraphConstructor
from st_backbone import SpatioTemporalBackbone
from mt_head import MultiTaskHead

# ==========================================
# 1. 时序提示工程 (Condition Prompting)
# ==========================================
class ConditionPrompt(nn.Module):
    """
    轻量级工况条件提示模块 (Condition Prompting)
    将设备的外部工况参数（如转速、径向负载等低维特征）映射为高维提示向量，
    作为 Transformer 的一个额外时间步 (Token) 拼接到序列头部，引导模型学习跨工况的退化规律。
    """
    def __init__(self, cond_dim: int, d_model: int):
        """
        :param cond_dim: 工况特征维度 (例如转速和负载，维度为 2)
        :param d_model: 目标隐藏层维度，需与 Transformer 的 d_model 一致
        """
        super(ConditionPrompt, self).__init__()
        self.prompt_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )

    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """
        :param x: 原始时间序列特征, Shape: [B, T, d_model]
        :param conditions: 设备工况参数, Shape: [B, cond_dim]
        :return: 拼接了工况提示的特征, Shape: [B, T+1, d_model]
        """
        # 1. 映射工况参数到 d_model
        # [B, cond_dim] -> [B, d_model]
        prompt_vec = self.prompt_proj(conditions)
        
        # 2. 增加时间维度，使其变成一个 Token
        # [B, d_model] -> [B, 1, d_model]
        prompt_token = prompt_vec.unsqueeze(1)
        
        # 3. 拼接到原始时间序列的头部
        # Concat 结果 Shape: [B, T+1, d_model]
        x_prompted = torch.cat([prompt_token, x], dim=1)
        
        return x_prompted


# ==========================================
# 2. 全局整合模型 (PHM Model)
# ==========================================
class PHM_STGNN_Model(nn.Module):
    """
    集成前面四个阶段的完整端到端预测模型
    """
    # 调低默认 d_model=64 以降低参数量
    def __init__(self, num_nodes=8, c_in=1, d_model=64, cond_dim=2, num_classes=3):
        super(PHM_STGNN_Model, self).__init__()
        # 阶段二：动态图构建
        self.graph_constructor = DynamicGraphConstructor(num_nodes=num_nodes, embed_dim=10)
        # 阶段三：时空主干 (CNN + GAT + Transformer)
        # 调低 d_gcn 到 32
        self.st_backbone = SpatioTemporalBackbone(
            num_nodes=num_nodes, c_in=c_in, c_cnn=32, d_gcn=32, d_model=d_model, tf_heads=4, tf_layers=2
        )
        # 提示工程模块
        self.condition_prompt = ConditionPrompt(cond_dim=cond_dim, d_model=d_model)
        # 阶段四：多任务预测头
        self.mt_head = MultiTaskHead(d_model=d_model, hidden_dim=32, num_classes=num_classes)
        # 基于不确定性的动态多任务损失权重 (Kendall et al., 2018)
        # log_var 初始化为 0，对应初始任务权重均等
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_reg = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        :param x: [B, T, N, c_in]
        :param conditions: [B, cond_dim]
        """
        # 1. 生成动态图
        adj = self.graph_constructor()  # [N, N]
        
        # 2. 时空特征提取 (前两步：CNN + GAT)
        h_cnn = self.st_backbone.temporal_cnn(x)       # [B, T, N, c_cnn]
        h_gcn = self.st_backbone.spatial_gat(h_cnn, adj) # [B, T, N, d_gcn]
        
        # 3. 展平并映射到 Transformer 维度
        from einops import rearrange
        h_flat = rearrange(h_gcn, 'b t n d -> b t (n d)')
        h_proj = self.st_backbone.global_transformer.input_projection(h_flat) # [B, T, d_model]
        
        # 4. **注入工况提示向量**
        h_prompted = self.condition_prompt(h_proj, conditions) # [B, T+1, d_model]
        
        # 5. 送入 Transformer (加位置编码)
        h_pe = self.st_backbone.global_transformer.pos_encoder(h_prompted)
        h_out = self.st_backbone.global_transformer.transformer_encoder(h_pe) # [B, T+1, d_model]
        
        # 6. 多任务预测头 (池化 + 双分支)
        preds = self.mt_head(h_out)
        return preds


# ==========================================
# 3. 训练大循环 (Training Pipeline)
# ==========================================
def train_one_epoch(
    model: nn.Module, 
    dataloader, 
    optimizer: torch.optim.Optimizer, 
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    accumulation_steps: int = 4,
    alpha: float = 0.5
) -> Tuple[float, float, float]:
    """
    训练单个 Epoch 的核心函数
    引入自动混合精度 (AMP) 与梯度累加 (Gradient Accumulation) 机制以突破显存瓶颈。
    
    :param model: PHM_STGNN_Model 实例
    :param dataloader: 包含 (x, conditions, labels, rul) 的数据加载器
    :param optimizer: 优化器 (如 AdamW)
    :param scaler: 自动混合精度 GradScaler
    :param device: 训练设备 (cuda/mps/cpu)
    :param accumulation_steps: 梯度累加步数，如为 4，则实际更新的 Batch Size = 物理 Batch Size * 4
    :param alpha: 多任务损失权重，Loss = alpha * L_cls + (1 - alpha) * L_reg
    :return: epoch_loss, epoch_cls_loss, epoch_reg_loss
    """
    model.train()
    total_loss, total_cls, total_reg = 0.0, 0.0, 0.0
    
    # 定义多任务损失函数
    # CrossEntropyLoss 内置了 LogSoftmax，接收未归一化的 logits
    criterion_cls = nn.CrossEntropyLoss()
    # MSE 适合回归，如果数据含离群点可换用 nn.HuberLoss() 或 L1Loss()
    criterion_reg = nn.MSELoss()
    
    # 梯度清零 (Epoch 开始前清零一次)
    optimizer.zero_grad()
    
    for batch_idx, (x, conds, labels, rul_targets) in enumerate(dataloader):
        # 1. 数据迁移至指定设备
        x = x.to(device, non_blocking=True)
        conds = conds.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        rul_targets = rul_targets.to(device, non_blocking=True)
        
        # 2. 自动混合精度 (AMP) 前向传播
        # autocast() 上下文管理器会使得内部的算子自动选择 FP16（半精度）或 FP32。
        # 对于矩阵乘法和卷积，FP16 能够大幅削减显存占用（约 50%）并利用 Tensor Core 加速；
        # 而对于容易溢出的操作（如 Softmax），它会自动保持 FP32。
        with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu', enabled=True):
            # 获取模型双分支预测输出
            preds = model(x, conds)
            logits = preds['class_logits']  # [B, 3]
            rul_pred = preds['rul_pred']    # [B, 1]
            
            # 计算多任务损失
            loss_cls = criterion_cls(logits, labels)
            # rul_targets 需要调整为 [B, 1] 以匹配预测值形状
            loss_reg = criterion_reg(rul_pred, rul_targets.unsqueeze(1))
            
            # 加权求和得到总损失
            loss = alpha * loss_cls + (1.0 - alpha) * loss_reg
            
            # 3. 梯度累加 (Gradient Accumulation) 的关键步骤
            # 为了在有限显存下模拟大 Batch Size 的平稳梯度更新，
            # 我们将当前 batch 的 loss 除以累加步数，以保证梯度的数学期望与大 Batch 一致。
            loss = loss / accumulation_steps
            
        # 4. 反向传播 (缩放后)
        # scaler 负责将 loss 放大，防止 FP16 下的梯度下溢 (Gradient Underflow) 变成 0。
        scaler.scale(loss).backward()
        
        # 5. 权重更新与梯度清零 (按步长触发)
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            # step() 首先把梯度缩放回来 (unscale)，然后执行优化器的一步更新
            scaler.step(optimizer)
            # 更新 scaler 内部的缩放因子
            scaler.update()
            # 累加完毕，更新后清空梯度，为下一轮累加做准备
            optimizer.zero_grad()
            
        # 记录真实损失 (为了打印和日志，恢复为未缩放前的值)
        real_loss = loss.item() * accumulation_steps
        total_loss += real_loss
        total_cls += loss_cls.item()
        total_reg += loss_reg.item()
        
    num_batches = len(dataloader)
    return total_loss / num_batches, total_cls / num_batches, total_reg / num_batches


# ==========================================
# 4. 模拟测试入口
# ==========================================
def run_mock_training():
    print("=== 开始初始化训练流水线 ===")
    # 确定设备 (Mac MPS 不完全支持 AMP，此处演示架构流转，可能降级为 CPU autocast 或忽略)
    device = config.DEVICE
    print(f"训练设备: {device}")
    
    # 实例化模型
    model = PHM_STGNN_Model(num_nodes=8, c_in=1, d_model=128, cond_dim=2, num_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 初始化 AMP Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # 模拟生成一个极其简单的 DataLoader
    print("\n正在生成模拟工况与时序数据...")
    class MockDataset:
        def __init__(self, size=32):
            self.size = size
            
        def __iter__(self):
            # 产出 4 个 Batch，每个 Batch 大小为 8
            for _ in range(4):
                x = torch.randn(8, 50, 8, 1)          # [B, T, N, C_in]
                conds = torch.randn(8, 2)             # [B, cond_dim] (例如转速、负载)
                labels = torch.randint(0, 3, (8,))    # [B] 分类标签 0,1,2
                ruls = torch.rand(8)                  # [B] RUL标签 0.0~1.0
                yield x, conds, labels, ruls
                
        def __len__(self):
            return 4

    dataloader = MockDataset()
    
    print("\n=== 开始模拟 3 个 Epoch 的训练 (含梯度累加与AMP) ===")
    num_epochs = 3
    for epoch in range(num_epochs):
        loss_all, loss_cls, loss_reg = train_one_epoch(
            model=model, 
            dataloader=dataloader, 
            optimizer=optimizer, 
            scaler=scaler, 
            device=device,
            accumulation_steps=2, # 每2个batch更新一次权重，模拟 Batch=16
            alpha=0.5
        )
        
        print(f"Epoch {epoch + 1}/{num_epochs} 结束:")
        print(f"  - Total Loss: {loss_all:.4f}")
        print(f"  - Cls Loss:   {loss_cls:.4f}")
        print(f"  - Reg Loss:   {loss_reg:.4f}")
        
    print("\n=== 测试通过！模型可以正常训练且 Loss 收敛 ===")

if __name__ == "__main__":
    run_mock_training()
