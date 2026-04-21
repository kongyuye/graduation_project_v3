import os
import glob
import pandas as pd
import numpy as np
import torch
import sys
from datetime import datetime
import config

# 将包含模型的目录加入环境变量
sys.path.append(config.PROJECT_DIR)
from train_pipeline import PHM_STGNN_Model

# =========================================================
# IEEE PHM 2012 Challenge 官方测试集真实 RUL (单位: 秒)
# =========================================================
actual_ruls_sec = {
    "Bearing1_3": 5730,
    "Bearing1_4": 339,
    "Bearing1_5": 1610,
    "Bearing1_6": 1460,
    "Bearing1_7": 7570,
    "Bearing2_3": 7530,
    "Bearing2_4": 1390,
    "Bearing2_5": 3090,
    "Bearing2_6": 1290,
    "Bearing2_7": 580,
    "Bearing3_3": 820
}

def get_condition_id(bearing_name: str) -> int:
    """根据轴承名称推断其所属工况编号"""
    for cond_id, prefix in config.CONDITION_PREFIXES.items():
        if bearing_name.startswith(prefix.rstrip("_")):
            return cond_id
    return None

def main():
    print("=== 初始化 ST-GNN 多工况模型预测流程 ===")
    
    # 确定推理设备 (推荐 CPU 或 MPS)
    device = config.DEVICE
    print(f"[*] 推理设备: {device}")
    
    # ---------------------------------------------------------
    # 1. 预加载所有工况的独立模型
    # ---------------------------------------------------------
    models = {}
    for cond_id in config.CONDITIONS:
        model_path = config.get_model_weights_path(cond_id)
        if not os.path.exists(model_path):
            print(f"[!] 工况 {cond_id} 的模型权重文件不存在: {model_path}，跳过。")
            continue
        m = PHM_STGNN_Model().to(device)
        m.load_state_dict(torch.load(model_path, map_location=device))
        m.eval()
        models[cond_id] = m
        print(f"[*] 工况 {cond_id} 模型加载成功: {model_path}")
    
    if not models:
        print("[!] 没有任何工况模型可用。请先运行 train_real.py 完成训练。")
        return
    
    print(f"[*] 已成功加载 {len(models)} 个工况模型。")

    # ---------------------------------------------------------
    # 2. 读取并预处理测试集特征数据
    # ---------------------------------------------------------
    test_data_dir = config.TEST_DATA_DIR
    csv_files = glob.glob(os.path.join(test_data_dir, "*_test_features.csv"))
    print(f"[*] 找到 {len(csv_files)} 个测试集特征文件。")
    
    window_size = config.WINDOW_SIZE
    results = []
    
    print("\n=== 开始预测 ===")
    print("Bearing      | 工况 | 数据行数       | 预测RUL(p)     | 预测RUL(s)     | 实际RUL(s)     | 误差(%)")
    print("-" * 110)
    
    # ---------------------------------------------------------
    # 3. 遍历测试集进行推理
    # ---------------------------------------------------------
    for file in sorted(csv_files):
        filename = os.path.basename(file)
        bearing_name = filename.replace("_test_features.csv", "")
        if bearing_name not in actual_ruls_sec:
            continue
        
        # 根据轴承名称路由到对应工况的模型
        cond_id = get_condition_id(bearing_name)
        if cond_id is None or cond_id not in models:
            print(f"[!] {bearing_name} 无法匹配工况模型，跳过。")
            continue
        model = models[cond_id]
            
        df = pd.read_csv(file)
        truncated_len = len(df)
        
        if truncated_len < window_size:
            print(f"[!] {bearing_name} 数据长度 ({truncated_len}) 不足一个窗口 ({window_size})，跳过。")
            continue
            
        # a. 提取 42 个空间节点特征 (必须与训练集匹配)
        h_time_cols = ['h_skewness', 'h_kurtosis', 'h_p2p', 'h_shape_factor']
        v_time_cols = ['v_skewness', 'v_kurtosis', 'v_p2p', 'v_shape_factor']
        h_freq_cols = ['h_spectral_centroid'] + [f'h_fft_band_energy_ratio_{i}' for i in range(8)]
        v_freq_cols = ['v_spectral_centroid'] + [f'v_fft_band_energy_ratio_{i}' for i in range(8)]
        h_wpt_cols = [f'h_wpt_energy_ratio_{i}' for i in range(8)]
        v_wpt_cols = [f'v_wpt_energy_ratio_{i}' for i in range(8)]
        
        node_cols = h_time_cols + v_time_cols + h_freq_cols + v_freq_cols + h_wpt_cols + v_wpt_cols
        
        x_data = df[node_cols].values
        x_data = np.expand_dims(x_data, axis=-1) # 增加通道维度 (L, 42, 1)
        
        # b. 提取 3 个全局工况特征 (RMS, 温度, 以及新加入的单调健康因子 HI)
        cond_cols = ['h_rms', 'temperature', 'health_indicator']
        cond_data = df[cond_cols].values # (L, 3)
        
        # c. 截取时间序列最后一个窗口，代表设备的"当前最新状态"
        last_x = x_data[-window_size:] # (50, 42, 1)
        last_cond = cond_data[-1]      # 取最后一个时刻的工况 (3,)
        
        # d. 转换为 Tensor 并增加 Batch 维度 (Batch Size = 1)
        x_tensor = torch.tensor(last_x, dtype=torch.float32).unsqueeze(0).to(device)
        cond_tensor = torch.tensor(last_cond, dtype=torch.float32).unsqueeze(0).to(device)
        
        # e. 使用对应工况模型进行前向推理
        with torch.no_grad():
            preds = model(x_tensor, cond_tensor)  # model 已按工况路由选取
            p = preds['rul_pred'].item() # 预测的 RUL 百分比 p ∈ [0.0, 1.0]，由 Sigmoid 层强制约束
            pred_class = torch.argmax(preds['class_logits'], dim=1).item() # 预测的分类标签 (0,1,2)
            
        # ---------------------------------------------------------
        # 4. 反归一化：自参照公式推断绝对物理时间（无需固定常数）
        # ---------------------------------------------------------
        # 推导：p = remaining_rows / total_len，(1-p) ≈ truncated_len / total_len
        # 因此：total_len ≈ truncated_len / (1-p)
        #       predicted_rul_rows = p * total_len = p * truncated_len / (1-p)
        p_safe = max(0.0001, min(0.9999, p))   # 防止除零
        predicted_rul_rows = p_safe * truncated_len / (1 - p_safe)
        predicted_rul_sec = predicted_rul_rows * 10   # PHM2012: 每行 = 10 秒
        
        actual_rul_sec = actual_ruls_sec[bearing_name]
        error_percent = (predicted_rul_sec - actual_rul_sec) / actual_rul_sec * 100
        
        # 打印日志
        print(f"{bearing_name:<12} | {cond_id:<4} | {truncated_len:<10} | {p:.4f}      | {predicted_rul_sec:<12.1f} | {actual_rul_sec:<12} | {error_percent:>+.2f}%")
        
        # 保存到结果列表
        results.append({
            "Bearing": bearing_name,
            "工况": cond_id,
            "数据行数(行)": truncated_len,
            "预测RUL百分比(p)": round(p, 4),
            "预测健康状态(0-2)": pred_class,
            "预测RUL(秒)": round(predicted_rul_sec, 1),
            "实际RUL(秒)": actual_rul_sec,
            "误差Error(%)": f"{error_percent:+.2f}%",
        })
        
    # ---------------------------------------------------------
    # 5. 结果保存
    # ---------------------------------------------------------
    if results:
        res_df = pd.DataFrame(results)
        
        mean_p = res_df["预测RUL百分比(p)"].mean()
        actual_ruls = res_df["实际RUL(秒)"].values
        predicted_ruls = res_df["预测RUL(秒)"].values
        mae_value = np.mean(np.abs(actual_ruls - predicted_ruls))
        
        print(f"\n[评估统计] 平均预测RUL百分比: {mean_p:.4f} ({mean_p*100:.2f}%) | MAE: {mae_value:.2f} 秒")

        out_path = config.PREDICT_RESULTS_PATH
        res_df.to_csv(out_path, index=False)
        print("-" * 100)
        print(f"[*] 所有测试集预测结果已成功保存至: {out_path}")

        # ---------------------------------------------------------
        # 6. 额外保存带时间戳的历史记录（用于跨次对比）
        # ---------------------------------------------------------
        history_dir = os.path.join(config.BASE_DIR, "predict_history")
        os.makedirs(history_dir, exist_ok=True)

        run_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        res_df_hist = res_df.copy()
        res_df_hist.insert(0, "运行时间", run_time)
        res_df_hist["平均RUL百分比"] = round(mean_p, 4)
        res_df_hist["MAE(秒)"] = round(mae_value, 2)

        hist_path = os.path.join(history_dir, f"predict_{run_time}.csv")
        res_df_hist.to_csv(hist_path, index=False)
        print(f"[*] 历史记录已归档至: {hist_path}")

if __name__ == "__main__":
    main()
