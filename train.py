import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import config

# 导入我们刚刚编写的模型
from model_architecture import STGNN_Transformer_Model

class BearingDataset(Dataset):
    """
    加载之前生成的 ST-GCN .npy 数据集
    """
    def __init__(self, data_path, is_train=True):
        prefix = 'train' if is_train else 'test'
        self.X = np.load(os.path.join(data_path, f'X_{prefix}.npy')) # (Batch, 1, 14, 30)
        self.Y_rul = np.load(os.path.join(data_path, f'Y_{prefix}.npy')) # (Batch,)
        
        # 将数据转换为 torch.Tensor
        self.X = torch.FloatTensor(self.X)
        self.Y_rul = torch.FloatTensor(self.Y_rul)
        
        # 生成分类标签 (伪标签逻辑，仅供跑通流程)
        # 根据 RUL 剩余寿命的百分比划分为 3 个状态: 
        # 0: 正常 (剩余 > 60%), 1: 轻微退化 (30% - 60%), 2: 严重退化 (< 30%)
        # 为了简单起见，这里假设每个轴承的最大寿命约为 2800，我们用绝对值划分
        self.Y_cls = torch.zeros_like(self.Y_rul, dtype=torch.long)
        self.Y_cls[self.Y_rul < 1500] = 1 # 轻微
        self.Y_cls[self.Y_rul < 500] = 2  # 严重
        
        print(f"[{'Train' if is_train else 'Test'}] 数据集加载完成: X: {self.X.shape}, RUL: {self.Y_rul.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y_cls[idx], self.Y_rul[idx]


def train_model():
    # 1. 基础配置
    data_dir = "/Users/wanglixiao/Desktop/大学/大四上/毕设/newproduct5/stgcn_dataset_v2"
    batch_size = 64
    num_epochs = 5 # 演示目的只跑5个Epoch
    learning_rate = 1e-4
    device = config.DEVICE
    print(f"使用设备: {device}")

    # 2. 构建 DataLoader
    train_dataset = BearingDataset(data_dir, is_train=True)
    test_dataset = BearingDataset(data_dir, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3. 初始化模型
    model = STGNN_Transformer_Model(in_channels=1, num_nodes=14, num_classes=3).to(device)
    
    # 4. 定义损失函数与优化器
    criterion_cls = nn.CrossEntropyLoss() # 分类损失
    criterion_reg = nn.MSELoss()          # 回归损失 (MSE / MAE)
    
    # 根据 PDF，分类提供状态约束，平滑回归任务。这里设置一个权重 lambda
    lambda_reg = 0.01 # 因为 RUL 数值可能很大，MSE loss 会非常大，需要缩放
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # 5. 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_cls_loss = 0
        total_reg_loss = 0
        
        # 使用 tqdm 显示进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_X, batch_Y_cls, batch_Y_rul in pbar:
            batch_X = batch_X.to(device)
            batch_Y_cls = batch_Y_cls.to(device)
            batch_Y_rul = batch_Y_rul.unsqueeze(1).to(device) # (Batch, 1)
            
            optimizer.zero_grad()
            
            # 前向传播
            logits, rul_pred = model(batch_X)
            
            # 计算多任务损失
            loss_cls = criterion_cls(logits, batch_Y_cls)
            loss_reg = criterion_reg(rul_pred, batch_Y_rul)
            
            loss = loss_cls + lambda_reg * loss_reg
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_reg_loss += loss_reg.item()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Cls': f'{loss_cls.item():.4f}', 'Reg': f'{loss_reg.item():.2f}'})
            
        avg_train_loss = total_loss / len(train_loader)
        
        # 6. 验证循环
        model.eval()
        val_loss = 0
        val_mae = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_Y_cls, batch_Y_rul in test_loader:
                batch_X = batch_X.to(device)
                batch_Y_cls = batch_Y_cls.to(device)
                batch_Y_rul = batch_Y_rul.unsqueeze(1).to(device)
                
                logits, rul_pred = model(batch_X)
                
                loss_cls = criterion_cls(logits, batch_Y_cls)
                loss_reg = criterion_reg(rul_pred, batch_Y_rul)
                val_loss += (loss_cls + lambda_reg * loss_reg).item()
                
                # 计算 MAE
                val_mae += torch.abs(rul_pred - batch_Y_rul).sum().item()
                
                # 计算分类准确率
                _, predicted = torch.max(logits.data, 1)
                total += batch_Y_cls.size(0)
                correct += (predicted == batch_Y_cls).sum().item()
                
        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = 100 * correct / total
        avg_mae = val_mae / total
        
        print(f"Epoch {epoch+1} 总结: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | Val RUL MAE: {avg_mae:.2f}")
        print("-" * 60)

    # 保存模型权重
    torch.save(model.state_dict(), "/Users/wanglixiao/Desktop/大学/大四上/毕设/newproduct5/stgnn_transformer_weights.pth")
    print("训练完成！模型权重已保存。")

if __name__ == "__main__":
    train_model()
