import os
import torch

# =========================================================================
# 项目全局路径配置文件 (Global Path Configuration)
# 在此集中管理所有数据、模型、结果和依赖代码的路径。
# 当在不同环境（本地 macOS / 远程 Linux 虚拟机等）运行时，只需修改此文件的 BASE_DIR 即可。
# =========================================================================

# 1. 定义项目根目录 (自动定位当前脚本所在目录，跨平台通用)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 核心代码模块目录 (用于 sys.path.append，确保能找到模型结构)
PROJECT_DIR = os.path.join(BASE_DIR, "graduation-project")

# 3. 预处理好的训练集与测试集特征目录
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
TEST_DATA_DIR = os.path.join(BASE_DIR, "processed_data")

# 4. 模型权重保存与加载路径
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "phm_stgnn_model_weights.pth")

# 5. 训练生成的 Loss 曲线保存路径
LOSS_CURVE_PATH = os.path.join(BASE_DIR, "loss_curve.png")

# 6. 预测结果 CSV 文件保存路径
PREDICT_RESULTS_PATH = os.path.join(BASE_DIR, "predict_results.csv")

# 8. 原始 PHM 2012 数据集根目录 (Windows 本地路径)
RAW_DATA_DIR = r"c:\Users\Administrator\Desktop\wlx\ieee-phm-2012-data-challenge-dataset-master"

# 7. 设备配置选项 (cuda, mps, cpu)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# =========================================================================
# 模型架构超参数 (Model Architecture Hyperparameters)
# =========================================================================
NUM_NODES      = 42    # 图节点数 = 特征数（8时域 + 18频域 + 16时频 = 42）
C_IN           = 1     # 输入通道数
D_MODEL        = 16    # Transformer 隐藏层维度
COND_DIM       = 3     # 工况条件特征维度 (h_rms, temperature, health_indicator)
NUM_CLASSES    = 3     # 分类任务类别数 (正常/轻微/严重)
EMBED_DIM      = 8    # DynamicGraphConstructor 节点嵌入维度
C_CNN          = 16    # 时域 CNN 输出通道数
D_GCN          = 16    # GAT 图卷积输出维度
TF_HEADS       = 4     # Transformer 多头注意力头数
TF_LAYERS      = 2     # Transformer 编码器层数
HIDDEN_DIM     = 16    # MultiTaskHead 隐藏层维度
DROPOUT        = 0.1   # 全局 Dropout 概率（用于 Transformer / MultiTaskHead）


# =========================================================================
# 数据集与预处理超参数 (Dataset & Preprocessing Hyperparameters)
# =========================================================================
WINDOW_SIZE           = 50    # 滑动窗口大小（时间步数）
STRIDE                = 20    # 滑动窗口步长
NUM_AUGMENTS          = 15    # 每个轴承文件的数据增强次数
AUG_START_RATIO_MIN   = 0   # 宏观截取：起点比例下限
AUG_START_RATIO_MAX   = 0.2   # 宏观截取：起点比例上限
AUG_END_RATIO_MIN     = 0.7  # 宏观截取：终点比例下限
AUG_END_RATIO_MAX     = 0.9   # 宏观截取：终点比例上限
HEALTHY_THRESHOLD     = 1.0   # 分段线性 RUL 封顶阈值
RUL_NORMAL_THRESHOLD  = 0.85   # RUL > 该值 → 类别 0（正常）
RUL_DEGRADED_THRESHOLD= 0.35   # RUL ≤ 该值 → 类别 2（严重退化）


# =========================================================================
# 训练超参数 (Training Hyperparameters)
# =========================================================================
BATCH_SIZE         = 32      # 每个 mini-batch 的样本数
NUM_EPOCHS         = 150     # 总训练轮数
LEARNING_RATE      = 3e-4    # 初始学习率
WEIGHT_DECAY       = 1e-4    # L2 权重衰减（正则化）
ALPHA              = 0.5     # 多任务损失权重: Loss = α·L_cls + (1-α)·L_reg·REG_SCALE
REG_LOSS_SCALE     = 10      # 回归损失的缩放系数（平衡量纲差异）
WARMUP_RATIO       = 0.2     # Warmup 阶段占总 Epoch 的比例
NOISE_STD          = 0.05    # 输入噪声注入的标准差
FEATURE_MASK_PROB  = 0.05    # 特征遮蔽（Feature Masking）概率
PENALTY_RATIO      = 5.0     # 非对称 MSE 损失的过预测惩罚系数


# =========================================================================
# 多工况配置 (Multi-Condition Configuration)
# PHM 2012 数据集包含 3 种工况，按轴承编号前缀区分：
#   工况 1: 1800 rpm, 4000N (Bearing1_x)
#   工况 2: 1650 rpm, 4200N (Bearing2_x)
#   工况 3: 1500 rpm, 5000N (Bearing3_x)
# =========================================================================
CONDITIONS = [1, 2, 3]
CONDITION_PREFIXES = {1: "Bearing1_", 2: "Bearing2_", 3: "Bearing3_"}

def get_model_weights_path(condition_id):
    """返回指定工况的模型权重保存/加载路径"""
    return os.path.join(BASE_DIR, f"phm_stgnn_model_weights_cond{condition_id}.pth")

def get_loss_curve_path(condition_id):
    """返回指定工况的 Loss 曲线保存路径"""
    return os.path.join(BASE_DIR, f"loss_curve_cond{condition_id}.png")
