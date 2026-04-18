import torch
import torch.nn as nn
import torch.quantization
from typing import Dict, Tuple

# 引入已有的完整模型结构（假设其存在）
from train_pipeline import PHM_STGNN_Model

# ==========================================
# 1. 工业级非对称惩罚评估函数 (Asymmetric Scoring)
# ==========================================
def asymmetric_score(y_true: torch.Tensor, y_pred: torch.Tensor, a1: float = 10.0, a2: float = 13.0) -> torch.Tensor:
    """
    计算 PHM 领域的非对称指数惩罚得分 (Asymmetric Score)。
    物理意义：
    - 高估寿命 (E_i > 0)：模型认为设备还能撑很久，但实际很快就坏了。这会导致灾难性的意外停机，因此施加更严厉的惩罚 (a1 较小)。
    - 低估寿命 (E_i < 0)：模型认为设备快坏了，但实际还能用。这仅会导致提前维护（过度维护成本），惩罚相对较轻 (a2 较大)。
    
    :param y_true: 真实剩余寿命百分比 (0~100) 或真实寿命步数
    :param y_pred: 预测的剩余寿命百分比 (0~100) 或预测寿命步数
    :param a1: 高估寿命时的惩罚因子 (默认 10.0，值越小惩罚越剧烈)
    :param a2: 低估寿命时的惩罚因子 (默认 13.0，相对 a1 惩罚较轻)
    :return: 该批次样本的平均非对称得分 (数值越小模型越好)
    """
    # 确保张量形状一致
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    
    # 误差 E_i = 预测值 - 真实值
    # E_i > 0 表示高估 (Over-estimation)
    # E_i < 0 表示低估 (Under-estimation)
    error = y_pred - y_true
    
    # 初始化得分张量
    scores = torch.zeros_like(error)
    
    # 掩码：找到高估和低估的索引
    mask_over = error > 0
    mask_under = error <= 0
    
    # 计算高估的指数惩罚: exp(E_i / a1) - 1
    if mask_over.any():
        scores[mask_over] = torch.exp(error[mask_over] / a1) - 1.0
        
    # 计算低估的指数惩罚: exp(-E_i / a2) - 1
    if mask_under.any():
        scores[mask_under] = torch.exp(-error[mask_under] / a2) - 1.0
        
    # 返回该 Batch 的平均得分
    return scores.mean()


# ==========================================
# 2. 量化感知训练 (Quantization-Aware Training, QAT)
# ==========================================

# ---------------------------------------------------------
# [警告/注意事项: Transformer 与 GNN 量化精度崩塌陷阱]
# 在将 ST-GNN + Transformer 转换为 INT8 时，请极其小心以下层：
# 1. Softmax (在 GAT 和 Transformer 的 Attention 中大量存在)：
#    INT8 的动态范围非常窄 (-128~127)，而 Softmax 的输入是内积，容易出现极端极化分布。
#    直接量化 Attention 权重极易导致精度断崖式下跌。
#    建议：将 Softmax 操作及其前后的矩阵乘法保留为 FP32/FP16，或使用对数量化 (LogQuant)。
# 2. LayerNorm / BatchNorm：
#    归一化层计算方差时对精度极其敏感，通常不建议量化，强制保持在 Float32 运行。
# 3. GELU / Sigmoid 等非线性激活：
#    查表法 (Lookup Table) 模拟的 INT8 GELU 误差较大，如果发现模型崩塌，可将这些激活层排除在量化之外。
# ---------------------------------------------------------

class QuantizedPHMModel(nn.Module):
    """
    为了支持 PyTorch 原生 QAT，我们需要对原始模型进行包装：
    1. 插入 QuantStub 和 DeQuantStub 节点
    2. 使用融合操作 (Fusion) 将 Conv+ReLU 等算子合并，减少内存访问延迟 (Memory Bound)
    """
    def __init__(self, model_fp32: nn.Module):
        super(QuantizedPHMModel, self).__init__()
        # 量化输入桩 (将 Float32 转换为 INT8)
        self.quant = torch.quantization.QuantStub()
        
        # 原始的 FP32 浮点模型
        self.model = model_fp32
        
        # 反量化输出桩 (将 INT8 恢复为 Float32 以计算 Loss 或直接输出)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. 输入数据从 FP32 量化为 INT8
        x_q = self.quant(x)
        conds_q = self.quant(conditions)
        
        # 2. 在 INT8 域（模拟）下执行前向传播
        preds_q = self.model(x_q, conds_q)
        
        # 3. 将输出反量化回 FP32 供评估或损失计算
        class_logits = self.dequant(preds_q['class_logits'])
        rul_pred = self.dequant(preds_q['rul_pred'])
        
        return {
            'class_logits': class_logits,
            'rul_pred': rul_pred
        }


def run_qat_finetuning_pipeline():
    """
    量化感知训练 (QAT) 微调骨架代码
    """
    print("=== 开始量化感知训练 (QAT) 适配流程 ===")
    
    # QAT 通常只能在 CPU 上进行模拟训练 (或者需要特殊的 GPU Backend 支持)
    # 边缘部署设备（如工业 IPC、ARM 芯片）通常使用 CPU 进行 INT8 推理
    device = torch.device('cpu')
    print(f"QAT 模拟训练设备强制切换至: {device}")
    
    # 1. 实例化原始 FP32 预训练模型 (这里用新初始化的代替)
    model_fp32 = PHM_STGNN_Model().to(device)
    
    # [最佳实践]: 在这一步应该加载您之前正常训练好的 FP32 最优模型权重
    # model_fp32.load_state_dict(torch.load('best_model_fp32.pth'))
    
    # 2. 包装为 QAT 兼容模型
    qat_model = QuantizedPHMModel(model_fp32)
    
    # 3. 指定量化配置 (QConfig)
    # 使用官方推荐的 qnnpack 后端（适合 ARM 架构的边缘计算设备）
    torch.backends.quantized.engine = 'qnnpack'
    qat_model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    
    # 【核心保护策略】：避免 Transformer 中某些不支持动态量化的层报错
    # 禁用对 nn.modules.linear.NonDynamicallyQuantizableLinear (常在 Transformer 中使用) 的量化
    for name, module in qat_model.named_modules():
        if isinstance(module, torch.nn.modules.linear.NonDynamicallyQuantizableLinear):
            module.qconfig = None
        # 同样，提前禁用 LayerNorm 等对量化极度敏感的层
        elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.TransformerEncoderLayer):
            module.qconfig = None
    
    # 4. 融合 (Fusion) 模块 (可选但极度推荐)
    # 将 Conv1d 和 ReLU 融合为一个底层算子，大幅降低边缘设备的推理延迟
    # (注意：需要手动指定模型内部 Conv 层的精确路径，这里提供伪代码结构)
    # torch.quantization.fuse_modules(qat_model.model.st_backbone.temporal_cnn, 
    #                                 [['conv1d', 'activation']], inplace=True)
    
    # 5. 插入模拟量化的 FakeQuantize 节点
    # 这一步会在模型的权重和激活张量处插入伪量化节点，用于在反向传播时模拟 INT8 的截断误差
    torch.quantization.prepare_qat(qat_model, inplace=True)
    print("已成功插入 FakeQuantize 模拟量化节点。")
    
    # 6. QAT 微调循环 (Fine-Tuning)
    # 通常只需要 1~3 个 Epoch，学习率设为原本的 1/10 到 1/100
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=1e-5)
    criterion_rul = nn.MSELoss()
    
    qat_model.train()
    print("\n开始 QAT 模拟微调 (Fake Quantization Fine-tuning)...")
    
    # 模拟几个 Batch 的微调
    for step in range(3):
        # 模拟输入 (B=8)
        x = torch.randn(8, 50, 8, 1)
        conds = torch.randn(8, 2)
        y_true = torch.rand(8) * 100  # 假设 RUL 百分比 0~100
        
        optimizer.zero_grad()
        
        # 前向传播 (此时数据流经 FakeQuantize 节点，模拟了 INT8 精度损失)
        preds = qat_model(x, conds)
        y_pred = preds['rul_pred'].squeeze() * 100 # 恢复到 0~100 比例
        
        # 1. 使用 MSE 计算梯度反向传播
        loss = criterion_rul(y_pred, y_true)
        loss.backward()
        optimizer.step()
        
        # 2. 同时调用我们自定义的工业非对称评估函数，监控惩罚得分
        # detach 避免评估函数影响计算图
        score = asymmetric_score(y_true.detach(), y_pred.detach())
        
        print(f"Step {step+1} | MSE Loss: {loss.item():.4f} | 工业非对称惩罚得分 (Score): {score.item():.4f}")
    
    # 7. 将 QAT 模型转换为真正的 INT8 模型 (Convert)
    print("\n微调完成，准备转换为纯 INT8 边缘推理模型...")
    qat_model.eval() # 必须在 eval 模式下转换
    
    # 禁用 Transformer 注意力层的量化 (避免精度崩塌的核心保护策略)
    # 将那些对 INT8 极度敏感的层强制回退为 Float32
    # (具体层路径根据实际网络结构调整，此处提供保护接口示例)
    for name, module in qat_model.named_modules():
        if isinstance(module, nn.TransformerEncoderLayer) or isinstance(module, nn.LayerNorm):
            module.qconfig = None 
            
    # 执行最终转换，FakeQuantize 节点将被替换为真正的 INT8 算子 (如 nn.quantized.Conv1d)
    int8_model = torch.quantization.convert(qat_model, inplace=True)
    
    print("模型已成功转换为 INT8 格式！")
    print("边缘部署就绪：推理延迟预计将下降 2~4 倍，模型体积减小 75%。")
    
    # 8. (可选) 保存 JIT Script 以供 C++ (LibTorch) 在边缘设备上调用
    # scripted_model = torch.jit.script(int8_model)
    # scripted_model.save("phm_edge_model_int8.pt")


if __name__ == "__main__":
    run_qat_finetuning_pipeline()
