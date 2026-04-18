import os
import glob
import pandas as pd
import numpy as np
import torch
import sys
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

def main():
    print("=== 初始化 ST-GNN 模型预测流程 ===")
    
    # 确定推理设备 (推荐 CPU 或 MPS)
    device = config.DEVICE
    print(f"[*] 推理设备: {device}")
    
    # ---------------------------------------------------------
    # 1. 实例化模型并加载权重
    # ---------------------------------------------------------
    # 参数需与 train_real.py 保持完全一致：42个节点，d_model=64 (降低参数量防过拟合)，工况维度3
    model = PHM_STGNN_Model(num_nodes=42, c_in=1, d_model=64, cond_dim=3, num_classes=3).to(device)
    
    model_path = config.MODEL_WEIGHTS_PATH
    if not os.path.exists(model_path):
        print(f"[!] 找不到模型权重文件: {model_path}\n请先运行 train_real.py 完成训练。")
        return
        
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 切换至评估模式，关闭 Dropout 等
    print("[*] 模型权重加载成功。")

    # ---------------------------------------------------------
    # 2. 读取并预处理测试集特征数据
    # ---------------------------------------------------------
    test_data_dir = config.TEST_DATA_DIR
    csv_files = glob.glob(os.path.join(test_data_dir, "*_test_features.csv"))
    print(f"[*] 找到 {len(csv_files)} 个测试集特征文件。")
    
    window_size = 50
    results = []
    
    print("\n=== 开始预测 ===")
    print(f"{'Bearing':<12} | {'截断行数':<10} | {'预测寿命比例(p)':<15} | {'预测状态':<10} | {'实际RUL(s)':<12} | {'预测RUL(s)':<12} | {'误差(%)'}")
    print("-" * 100)
    
    # ---------------------------------------------------------
    # 3. 遍历测试集进行推理
    # ---------------------------------------------------------
    for file in sorted(csv_files):
        filename = os.path.basename(file)
        bearing_name = filename.replace("_test_features.csv", "")
        if bearing_name not in actual_ruls_sec:
            continue
            
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
        
        # e. 模型前向推理
        with torch.no_grad():
            preds = model(x_tensor, cond_tensor)
            p = preds['rul_pred'].item() # 预测的寿命百分比 p (0.0 ~ 1.0 之间)
            pred_class = torch.argmax(preds['class_logits'], dim=1).item() # 预测的分类标签 (0,1,2)
            
        # ---------------------------------------------------------
        # 4. 反归一化计算：由百分比转换为具体物理时间
        # ---------------------------------------------------------
        # 限制 p 防止极端值导致除以 0 或结果无限大
        p_clipped = max(0.01, min(0.99, p))
        
        # 根据模型训练时的归一化公式: p = RUL_rows / (Truncated_len + RUL_rows)
        # 推导得出: RUL_rows = p * Truncated_len / (1 - p)
        predicted_rul_rows = p_clipped * truncated_len / (1 - p_clipped)
        
        # PHM2012 数据集采样规则：每 10 秒记录一次信号。所以每行特征代表 10 秒钟
        predicted_rul_sec = predicted_rul_rows * 10
        
        actual_rul_sec = actual_ruls_sec[bearing_name]
        error_percent = (predicted_rul_sec - actual_rul_sec) / actual_rul_sec * 100
        
        # 打印日志
        print(f"{bearing_name:<12} | {truncated_len:<10} | {p:>.4f}          | {pred_class:<10} | {actual_rul_sec:<12} | {predicted_rul_sec:<12.1f} | {error_percent:>6.2f}%")
        
        # 保存到结果列表
        results.append({
            "Bearing": bearing_name,
            "截断特征长度(行)": truncated_len,
            "预测剩余寿命比例(p)": round(p, 4),
            "预测健康状态(0-2)": pred_class,
            "实际RUL(秒)": actual_rul_sec,
            "预测RUL(秒)": round(predicted_rul_sec, 1),
            "误差Error(%)": f"{error_percent:.2f}%"
        })
        
    # ---------------------------------------------------------
    # 5. 结果保存
    # ---------------------------------------------------------
    if results:
        res_df = pd.DataFrame(results)
        
        # === 计算总体 MAE (Mean Absolute Error) ===
        # 提取真实值和预测值列
        actual_ruls = res_df["实际RUL(秒)"].values
        predicted_ruls = res_df["预测RUL(秒)"].values
        
        # 计算绝对误差
        absolute_errors = np.abs(actual_ruls - predicted_ruls)
        # 求平均
        mae_value = np.mean(absolute_errors)
        
        print(f"\n[评估指标] 测试集整体 MAE: {mae_value:.2f} 秒")
        # ==========================================

        out_path = config.PREDICT_RESULTS_PATH
        res_df.to_csv(out_path, index=False)
        print("-" * 100)
        print(f"[*] 所有测试集预测结果已成功保存至: {out_path}")
    else:
        print("[!] 没有产生任何预测结果，请检查数据。")

if __name__ == "__main__":
    main()
