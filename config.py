import os
import torch

# =========================================================================
# 项目全局路径配置文件 (Global Path Configuration)
# 在此集中管理所有数据、模型、结果和依赖代码的路径。
# 当在不同环境（本地 macOS / 远程 Linux 虚拟机等）运行时，只需修改此文件的 BASE_DIR 即可。
# =========================================================================

# 1. 定义项目根目录 (根据您当前所在的环境修改这个路径)
BASE_DIR = "/root/autodl-tmp/graduation_project_v2/newproduct7"

# 2. 核心代码模块目录 (用于 sys.path.append，确保能找到模型结构)
PROJECT_DIR = os.path.join(BASE_DIR, "graduation-project")

# 3. 预处理好的训练集与测试集特征目录
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "processed_data/train")
TEST_DATA_DIR = os.path.join(BASE_DIR, "processed_data/test")

# 4. 模型权重保存与加载路径
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "phm_stgnn_model_weights.pth")

# 5. 训练生成的 Loss 曲线保存路径
LOSS_CURVE_PATH = os.path.join(PROJECT_DIR, "loss_curve.png")

# 6. 预测结果 CSV 文件保存路径
PREDICT_RESULTS_PATH = os.path.join(BASE_DIR, "predict_results.csv")

# 7. 设备配置选项 (cuda, mps, cpu)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
