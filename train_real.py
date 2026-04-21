import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt

# 引入全局路径配置
import config
from config import DEVICE

# 引入已有的核心模型模块
from train_pipeline import PHM_STGNN_Model

class PHMRealDataset(Dataset):
    """
    针对 IEEE PHM 2012 预处理数据的 PyTorch Dataset。
    核心机制：从特征时间序列中执行「滑动窗口」截取，生成 [T, N, C_in] 的三维张量。
    """
    def __init__(self, data_dir: str, window_size: int = 50, stride: int = 20):
        self.window_size = window_size
        self.windows_x = []
        self.windows_cond = []
        self.windows_y_cls = []
        self.windows_y_rul = []
        
        # 匹配之前处理好的 train_features.csv 文件
        csv_files = glob.glob(os.path.join(data_dir, "*_train_features.csv"))
        print(f"找到 {len(csv_files)} 个训练集轴承特征文件。")
        
        # 引入数据增强机制（路线 A：宏观截取 + 微观滑动窗口）
        num_augments = 10  # 每个轴承文件增强/截取 10 次
        
        for file in csv_files:
            df = pd.read_csv(file)
            total_len = len(df)
            if total_len < window_size:
                continue
            
            # 对同一个轴承数据进行多次随机截取（Data Augmentation）
            for aug in range(num_augments):
                # --- 路线 A：宏观随机截取 ---
                # 1. 确定截取的起止点 (起点 20~50%, 终点 75~100%)
                # 如果是第一次循环 (aug==0)，我们保留完整的全生命周期，保证基线数据完整
                if aug == 0:
                    start_idx = 0
                    end_idx = total_len
                else:
                    start_ratio = np.random.uniform(0.2, 0.5)
                    end_ratio = np.random.uniform(0.75, 1.0)
                    start_idx = int(start_ratio * total_len)
                    end_idx = int(end_ratio * total_len)
                    
                    # 容错处理：确保截取长度大于滑动窗口大小
                    if end_idx - start_idx < window_size:
                        start_idx = 0
                        end_idx = total_len

                # 2. 截取对应片段的数据
                truncated_df = df.iloc[start_idx:end_idx].copy()
                truncated_len = len(truncated_df)
                
                # --- 构建目标标签 (修改为分段线性 RUL) --- 
                absolute_steps = np.arange(start_idx, end_idx) 
                # 原始的绝对线性 RUL 
                linear_rul = (total_len - absolute_steps - 1) / total_len 
                
                # 此处不设定健康阈值
                healthy_threshold = 1.0
                
                # 将 RUL 封顶 
                piecewise_rul = np.clip(linear_rul, a_min=0.0, a_max=healthy_threshold) 
                # 重新归一化到 0~1 之间 (可选，方便模型输出层的 Sigmoid 或直接拟合) 
                rul_array = piecewise_rul / healthy_threshold 
                
                # 分类标签 (0: 正常 >0.8, 1: 轻微退化 0.3~0.8, 2: 严重退化 <=0.3)
                cls_array = np.zeros(truncated_len, dtype=np.int64)
                cls_array[(rul_array <= 0.8) & (rul_array > 0.3)] = 1
                cls_array[rul_array <= 0.3] = 2
                
                # --- 提取输入特征 ---
                # 时域特征 (4*2 = 8)
                h_time_cols = ['h_skewness', 'h_kurtosis', 'h_p2p', 'h_shape_factor']
                v_time_cols = ['v_skewness', 'v_kurtosis', 'v_p2p', 'v_shape_factor']
                # 频域特征 (9*2 = 18)
                h_freq_cols = ['h_spectral_centroid'] + [f'h_fft_band_energy_ratio_{i}' for i in range(8)]
                v_freq_cols = ['v_spectral_centroid'] + [f'v_fft_band_energy_ratio_{i}' for i in range(8)]
                # 时频域特征 (8*2 = 16)
                h_wpt_cols = [f'h_wpt_energy_ratio_{i}' for i in range(8)]
                v_wpt_cols = [f'v_wpt_energy_ratio_{i}' for i in range(8)]
                
                node_cols = h_time_cols + v_time_cols + h_freq_cols + v_freq_cols + h_wpt_cols + v_wpt_cols
                
                x_data = truncated_df[node_cols].values # 形状: (L, 42)
                x_data = np.expand_dims(x_data, axis=-1) # 增加通道维度 -> (L, 42, 1)
                
                # 【核心修改点】将 health_indicator 加入到 cond_cols 中！
                cond_cols = ['h_rms', 'temperature', 'health_indicator'] 
                cond_data = truncated_df[cond_cols].values # 形状变为: (L, 3)
                
                # --- 滑动窗口切片 ---
                for i in range(0, truncated_len - window_size + 1, stride):
                    self.windows_x.append(x_data[i : i + window_size])
                    self.windows_cond.append(cond_data[i + window_size - 1])
                    self.windows_y_cls.append(cls_array[i + window_size - 1])
                    self.windows_y_rul.append(rul_array[i + window_size - 1])
                
        # 转换为内存连续的 Numpy 数组，提升 PyTorch 加载效率
        # - 第 1 个维度：样本索引（第几个滑动窗口）。
        # - 第 2 个维度：时间步长（50 个时刻）。
        # - 第 3 个维度：空间节点（42 个特征）。
        # - 第 4 个维度：通道数（1）。
        self.windows_x = np.array(self.windows_x, dtype=np.float32)
        self.windows_cond = np.array(self.windows_cond, dtype=np.float32)
        self.windows_y_cls = np.array(self.windows_y_cls, dtype=np.int64)
        self.windows_y_rul = np.array(self.windows_y_rul, dtype=np.float32)
        
        print(f"成功生成 {len(self.windows_x)} 个时间窗样本 (窗口大小: {window_size}, 步长: {stride})")

    def __len__(self):
        return len(self.windows_x)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.windows_x[idx]),
            torch.tensor(self.windows_cond[idx]),
            torch.tensor(self.windows_y_cls[idx]),
            torch.tensor(self.windows_y_rul[idx])
        )

# 1. 定义非对称 MSE 损失函数 
class AsymmetricMSELoss(nn.Module): 
    def __init__(self, penalty_ratio=5.0): 
        """ 
        penalty_ratio: 惩罚系数。默认5.0，表示“预测寿命偏高”的惩罚是“预测寿命偏低”的 5 倍。 
        """ 
        super(AsymmetricMSELoss, self).__init__() 
        self.penalty_ratio = penalty_ratio 

    def forward(self, y_pred, y_true): 
        error = y_pred - y_true 
        # 如果 error > 0 (预测寿命 > 真实寿命)，说明是危险的过预测，给予 penalty_ratio 倍的惩罚 
        # 如果 error <= 0 (预测寿命 <= 真实寿命)，说明是安全的保守预测，正常惩罚 
        loss = torch.where(error > 0, self.penalty_ratio * (error ** 2), error ** 2) 
        return torch.mean(loss) 

def main():
    print("=== 初始化 ST-GNN 真实数据训练流程 ===")
    
    # 1. 准备数据 (使用全局配置路径)
    data_dir = config.TRAIN_DATA_DIR
    dataset = PHMRealDataset(data_dir, window_size=50, stride=20)
    # 使用 DataLoader 批量加载数据，启用 shuffle 打乱时序防止过拟合
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # 2. 设备与模型初始化
    device = config.DEVICE
    print(f"训练加速设备: {device}")
    
    # 实例化我们的多任务模型
    # 【降低参数量】d_model 从 128 降至 64
    model = PHM_STGNN_Model(num_nodes=42, c_in=1, d_model=64, cond_dim=3, num_classes=3).to(device)
    
    # === 优化点 1: 引入更强的权重衰减 (L2 正则化) ===
    # 防止过拟合的另一个关键：加大权重衰减惩罚
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    # 定义损失函数
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = AsymmetricMSELoss(penalty_ratio=5.0)
    
    num_epochs = 150
    # alpha 已由模型内置的不确定性动态权重取代，无需手动设置
    
    # === 添加学习率调度器：Warmup (前20%) + 余弦退火 (Cosine Annealing) ===
    warmup_epochs = int(num_epochs * 0.2)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性 Warmup：从 0.1 倍逐渐增加到 1.0
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        else:
            # 余弦退火：从 1.0 衰减到 0.001
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.001 + 0.5 * (1 - 0.001) * (1 + np.cos(np.pi * progress))
            
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # 用于记录 Loss 曲线的数据
    history_total_loss = []
    history_cls_loss = []
    history_reg_loss = []
    history_lr = []
    
    print("\n=== 开始模型训练 ===")
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_cls_loss, total_reg_loss = 0.0, 0.0, 0.0
        
        for batch_idx, (x, cond, y_cls, y_rul) in enumerate(dataloader):
            # 将张量挂载到 GPU/MPS
            x, cond = x.to(device), cond.to(device)
            y_cls = y_cls.to(device)
            y_rul = y_rul.to(device).unsqueeze(-1) # 形状对齐为 (B, 1)
            
            # === 优化点 2: 引入更强的数据增强 (Noise Injection & Dropout/Masking) ===
            # 在训练阶段，不仅向输入特征 x 加入微小的高斯噪声，
            # 还可以随机 mask 掉一部分传感器节点，强迫模型从其他节点学习 (Node Dropout)
            if model.training:
                # 1. 噪声注入
                noise_std = 0.2  # 噪声标准差适当调大 (原为 0.01)
                noise = torch.randn_like(x) * noise_std
                x = x + noise
                
                # 2. 节点/特征遮蔽 (Feature Masking) - 20% 的概率将某个时间步的某节点特征置零
                mask = (torch.rand_like(x) > 0.2).float()
                x = x * mask
            
            optimizer.zero_grad()
            
            # 前向传播
            preds = model(x, cond)
            
            # 多任务损失计算 —— 基于不确定性的动态权重 (Kendall et al., 2018)
            # loss = L_cls * exp(-log_var_cls) + log_var_cls
            #      + L_reg * exp(-log_var_reg) + log_var_reg
            # exp(-log_var) 随任务损失自动增大而缩小，log_var 作正则项防止权重坍缩到0
            loss_cls = criterion_cls(preds['class_logits'], y_cls)
            loss_reg = criterion_reg(preds['rul_pred'], y_rul)
            loss = (loss_cls * torch.exp(-model.log_var_cls) + model.log_var_cls) + \
                   (loss_reg * torch.exp(-model.log_var_reg) + model.log_var_reg)
            
            # 反向传播与参数更新
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_reg_loss += loss_reg.item()
            
            # 打印日志 (每 20 个 Batch)
            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{batch_idx+1:3d}/{len(dataloader)}] | "
                      f"Total Loss: {loss.item():.4f} (Cls: {loss_cls.item():.4f}, Reg: {loss_reg.item():.4f})")
                
        # 打印 Epoch 级统计 (这能真实反映模型是否在收敛)
        avg_loss = total_loss / len(dataloader)
        avg_cls_loss = total_cls_loss / len(dataloader)
        avg_reg_loss = total_reg_loss / len(dataloader)
        
        # 记录并更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        history_lr.append(current_lr)
        scheduler.step()
        
        # 记录到历史列表中
        history_total_loss.append(avg_loss)
        history_cls_loss.append(avg_cls_loss)
        history_reg_loss.append(avg_reg_loss)
        
        print(f"==> Epoch {epoch+1} 结束 | LR: {current_lr:.6f} | 平均 Loss: {avg_loss:.4f} (Cls: {avg_cls_loss:.4f}, Reg: {avg_reg_loss:.4f}) | "
              f"log_var_cls: {model.log_var_cls.item():.4f}, log_var_reg: {model.log_var_reg.item():.4f}\n")
        
    # 3. 固化模型并保存权重 (使用全局配置路径)
    save_path = config.MODEL_WEIGHTS_PATH
    torch.save(model.state_dict(), save_path)
    print(f"🎉 训练大循环圆满完成！模型已固化并成功保存至:\n {save_path}")
    
    # 4. 绘制并保存 Loss 曲线
    print("\n=== 绘制 Loss 曲线 ===")
    # 修改为使用 subplots，生成 3 个独立的子图
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    epochs_range = range(1, num_epochs + 1)
    
    # 子图 1: Total Loss
    axs[0].plot(epochs_range, history_total_loss, label='Total Loss', color='black', linewidth=2)
    axs[0].set_title('Total Training Loss')
    axs[0].set_ylabel('Loss Value')
    axs[0].legend()
    axs[0].grid(True, linestyle=':', alpha=0.6)
    
    # 子图 2: Classification Loss
    axs[1].plot(epochs_range, history_cls_loss, label='Classification Loss (CrossEntropy)', color='blue', linewidth=2)
    axs[1].set_title('Classification Loss')
    axs[1].set_ylabel('Loss Value')
    axs[1].legend()
    axs[1].grid(True, linestyle=':', alpha=0.6)
    
    # 子图 3: Regression Loss
    axs[2].plot(epochs_range, history_reg_loss, label='Regression Loss (MSE)', color='red', linewidth=2)
    axs[2].set_title('Regression Loss')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Loss Value')
    axs[2].legend()
    axs[2].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    # 保存图片 (使用全局配置路径)
    plot_save_path = config.LOSS_CURVE_PATH
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    print(f"📈 Loss 曲线已保存至: {plot_save_path}")
    
    # 尝试直接显示图片（如果在支持 GUI 的环境下）
    try:
        plt.show()
    except Exception as e:
        print("当前环境无法直接显示弹窗，请直接查看保存的图片文件。")

if __name__ == "__main__":
    main()