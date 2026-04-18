import torch
from train_pipeline import PHM_STGNN_Model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 实例化模型
model = PHM_STGNN_Model(num_nodes=8, c_in=1, d_model=128, cond_dim=2, num_classes=3)
total_params = count_parameters(model)
print(f"Model Total Trainable Parameters: {total_params:,}")

# 估算张量尺寸 (以滑动窗口 T=50, Batch=32 为例)
# B=32, T=50, N=8, C_in=1
dummy_x = torch.randn(32, 50, 8, 1)
dummy_cond = torch.randn(32, 2)

import time
import config
device = config.DEVICE
print(f"Using device: {device}")

model = model.to(device)
dummy_x = dummy_x.to(device)
dummy_cond = dummy_cond.to(device)

# 预热 (Warm-up)
for _ in range(5):
    _ = model(dummy_x, dummy_cond)

# 测速
start_time = time.time()
num_iterations = 50
for _ in range(num_iterations):
    _ = model(dummy_x, dummy_cond)
end_time = time.time()

avg_time_per_batch = (end_time - start_time) / num_iterations
print(f"Average Forward Pass Time per Batch (B=32, T=50): {avg_time_per_batch*1000:.2f} ms")
