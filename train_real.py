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
    
    :param condition_id: 工况编号 (1/2/3)。指定后仅加载该工况的轴承数据；
                         为 None 时加载全部轴承（兼容旧行为）。
    """
    def __init__(self, data_dir: str, window_size: int = config.WINDOW_SIZE, stride: int = config.STRIDE,
                 condition_id: int = None):
        self.window_size = window_size
        self.condition_id = condition_id
        self.windows_x = []
        self.windows_cond = []
        self.windows_y_cls = []
        self.windows_y_rul = []
        
        # 匹配之前处理好的 train_features.csv 文件
        csv_files = glob.glob(os.path.join(data_dir, "*_train_features.csv"))
        
        # 按工况筛选：仅保留指定工况前缀的文件
        if condition_id is not None:
            prefix = config.CONDITION_PREFIXES[condition_id]
            csv_files = [f for f in csv_files if os.path.basename(f).startswith(prefix)]
            print(f"[工况 {condition_id}] 找到 {len(csv_files)} 个训练集轴承特征文件。")
        else:
            print(f"找到 {len(csv_files)} 个训练集轴承特征文件（全部工况）。")
        
        # 引入数据增强机制（路线 A：宏观截取 + 微观滑动窗口）
        num_augments = config.NUM_AUGMENTS  # 每个轴承文件增强/截取次数
        
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
                    start_ratio = np.random.uniform(config.AUG_START_RATIO_MIN, config.AUG_START_RATIO_MAX)
                    end_ratio   = np.random.uniform(config.AUG_END_RATIO_MIN,   config.AUG_END_RATIO_MAX)
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
                healthy_threshold = config.HEALTHY_THRESHOLD
                
                # 将 RUL 封顶 
                piecewise_rul = np.clip(linear_rul, a_min=0.0, a_max=healthy_threshold) 
                # 重新归一化到 0~1 之间 (可选，方便模型输出层的 Sigmoid 或直接拟合) 
                rul_array = piecewise_rul / healthy_threshold 
                
                # 分类标签 (0: 正常 >0.8, 1: 轻微退化 0.3~0.8, 2: 严重退化 <=0.3)
                cls_array = np.zeros(truncated_len, dtype=np.int64)
                cls_array[(rul_array <= config.RUL_NORMAL_THRESHOLD) & (rul_array > config.RUL_DEGRADED_THRESHOLD)] = 1
                cls_array[rul_array <= config.RUL_DEGRADED_THRESHOLD] = 2
                
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
    def __init__(self, penalty_ratio=config.PENALTY_RATIO): 
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

def train_single_condition(cond_id: int):
    """
    针对单个工况训练一个独立的 ST-GNN 模型。
    
    :param cond_id: 工况编号 (1, 2 或 3)
    """
    print(f"\n{'='*60}")
    print(f"=== 开始训练工况 {cond_id} ({config.CONDITION_PREFIXES[cond_id]}*) 的独立模型 ===")
    print(f"{'='*60}")
    
    # 1. 准备数据 (仅加载该工况的轴承特征文件)
    data_dir = config.TRAIN_DATA_DIR
    dataset = PHMRealDataset(data_dir, window_size=config.WINDOW_SIZE, stride=config.STRIDE,
                             condition_id=cond_id)
    
    if len(dataset) == 0:
        print(f"[!] 工况 {cond_id} 无可用训练样本，跳过。")
        return
    
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 2. 设备与模型初始化 (每个工况独立实例化)
    device = config.DEVICE
    print(f"训练加速设备: {device}")
    
    model = PHM_STGNN_Model().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # 定义损失函数
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = AsymmetricMSELoss(penalty_ratio=config.PENALTY_RATIO)
    
    num_epochs = config.NUM_EPOCHS
    alpha = config.ALPHA
    
    # 学习率调度器：Warmup + 余弦退火
    warmup_epochs = int(num_epochs * config.WARMUP_RATIO)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.001 + 0.5 * (1 - 0.001) * (1 + np.cos(np.pi * progress))
            
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Loss 曲线记录
    history_total_loss = []
    history_cls_loss = []
    history_reg_loss = []
    history_lr = []
    
    print(f"\n=== 工况 {cond_id}: 开始模型训练 ({num_epochs} Epochs) ===")
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_cls_loss, total_reg_loss = 0.0, 0.0, 0.0
        
        for batch_idx, (x, cond, y_cls, y_rul) in enumerate(dataloader):
            x, cond = x.to(device), cond.to(device)
            y_cls = y_cls.to(device)
            y_rul = y_rul.to(device).unsqueeze(-1)
            
            # 数据增强 (Noise Injection & Feature Masking)
            if model.training:
                noise = torch.randn_like(x) * config.NOISE_STD
                x = x + noise
                mask = (torch.rand_like(x) > config.FEATURE_MASK_PROB).float()
                x = x * mask
            
            optimizer.zero_grad()
            preds = model(x, cond)
            
            loss_cls = criterion_cls(preds['class_logits'], y_cls)
            loss_reg = criterion_reg(preds['rul_pred'], y_rul)
            loss = alpha * loss_cls + (1 - alpha) * loss_reg * config.REG_LOSS_SCALE
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_reg_loss += loss_reg.item()
            
            if (batch_idx + 1) % 20 == 0:
                print(f"[工况{cond_id}] Epoch [{epoch+1}/{num_epochs}] | Batch [{batch_idx+1:3d}/{len(dataloader)}] | "
                      f"Total Loss: {loss.item():.4f} (Cls: {loss_cls.item():.4f}, Reg: {loss_reg.item():.4f})")
                
        avg_loss = total_loss / len(dataloader)
        avg_cls_loss = total_cls_loss / len(dataloader)
        avg_reg_loss = total_reg_loss / len(dataloader)
        
        current_lr = optimizer.param_groups[0]['lr']
        history_lr.append(current_lr)
        scheduler.step()
        
        history_total_loss.append(avg_loss)
        history_cls_loss.append(avg_cls_loss)
        history_reg_loss.append(avg_reg_loss)
        
        print(f"[工况{cond_id}] Epoch {epoch+1} 结束 | LR: {current_lr:.6f} | 平均 Loss: {avg_loss:.4f} (Cls: {avg_cls_loss:.4f}, Reg: {avg_reg_loss:.4f})")
        
    # 3. 保存模型权重到工况专属路径
    save_path = config.get_model_weights_path(cond_id)
    torch.save(model.state_dict(), save_path)
    print(f"\n[工况{cond_id}] 训练完成！模型已保存至: {save_path}")
    
    # 4. 绘制并保存该工况的 Loss 曲线
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig.suptitle(f'Training Loss - Condition {cond_id} ({config.CONDITION_PREFIXES[cond_id]}*)', fontsize=14)
    epochs_range = range(1, num_epochs + 1)
    
    axs[0].plot(epochs_range, history_total_loss, label='Total Loss', color='black', linewidth=2)
    axs[0].set_title('Total Training Loss')
    axs[0].set_ylabel('Loss Value')
    axs[0].legend()
    axs[0].grid(True, linestyle=':', alpha=0.6)
    
    axs[1].plot(epochs_range, history_cls_loss, label='Classification Loss (CrossEntropy)', color='blue', linewidth=2)
    axs[1].set_title('Classification Loss')
    axs[1].set_ylabel('Loss Value')
    axs[1].legend()
    axs[1].grid(True, linestyle=':', alpha=0.6)
    
    axs[2].plot(epochs_range, history_reg_loss, label='Regression Loss (MSE)', color='red', linewidth=2)
    axs[2].set_title('Regression Loss')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Loss Value')
    axs[2].legend()
    axs[2].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    plot_save_path = config.get_loss_curve_path(cond_id)
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[工况{cond_id}] Loss 曲线已保存至: {plot_save_path}")


def main():
    """
    多工况分别训练入口：依次为 3 个工况各训练一个独立模型。
    """
    print("=== 初始化 ST-GNN 多工况独立训练流程 ===")
    print(f"将为 {len(config.CONDITIONS)} 个工况分别训练独立模型")
    print(f"训练加速设备: {config.DEVICE}")
    
    for cond_id in config.CONDITIONS:
        train_single_condition(cond_id)
    
    print(f"\n{'='*60}")
    print("=== 全部工况训练完成！===")
    for cond_id in config.CONDITIONS:
        path = config.get_model_weights_path(cond_id)
        status = "已保存" if os.path.exists(path) else "未生成"
        print(f"  工况 {cond_id}: {path} [{status}]")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()