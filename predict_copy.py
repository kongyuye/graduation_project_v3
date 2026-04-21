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
# (此处为了使用训练集，我们不再依赖此字典)
# =========================================================
# actual_ruls_sec = { ... }

def main():
    print("=== 初始化 ST-GNN 模型预测流程 ===")
    
    # 确定推理设备 (推荐 CPU 或 MPS)
    device = config.DEVICE
    print(f"[*] 推理设备: {device}")
    
    # ---------------------------------------------------------
    # 1. 实例化模型并加载权重
    # ---------------------------------------------------------
    # 所有架构参数均从 config 读取，与 train_real.py 保持完全一致
    model = PHM_STGNN_Model().to(device)
    
    model_path = config.MODEL_WEIGHTS_PATH
    if not os.path.exists(model_path):
        print(f"[!] 找不到模型权重文件: {model_path}\n请先运行 train_real.py 完成训练。")
        return
        
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 切换至评估模式，关闭 Dropout 等
    print("[*] 模型权重加载成功。")

    # ---------------------------------------------------------
    # 2. 读取并预处理训练集特征数据
    # ---------------------------------------------------------
    train_data_dir = config.TRAIN_DATA_DIR
    csv_files = glob.glob(os.path.join(train_data_dir, "*_train_features.csv"))
    print(f"[*] 找到 {len(csv_files)} 个训练集特征文件。")
    
    window_size = config.WINDOW_SIZE
    results = []
    
    print("\n=== 开始预测 ===")
    print("轴承名称       | 截断行数       | 预测RUL(p)     | 预测RUL(s)     | 实际RUL(s)     | 误差(%)")
    print("-" * 100)
    
    # ---------------------------------------------------------
    # 3. 遍历训练集进行模拟推理 (随机截断)
    # ---------------------------------------------------------
    for file in sorted(csv_files):
        filename = os.path.basename(file)
        bearing_name = filename.replace("_train_features.csv", "")
            
        df = pd.read_csv(file)
        total_len = len(df)
        
        # 模拟测试集：我们在 [20%, 80%] 之间随机截断这个训练集文件
        # 来看看模型能不能预测出剩下的时间
        truncate_ratio = np.random.uniform(0.2, 0.8)
        truncated_len = int(total_len * truncate_ratio)
        
        if truncated_len < window_size:
            continue
            
        # 实际剩余行数 = 总行数 - 截断行数
        actual_rul_rows = total_len - truncated_len
        actual_rul_sec = actual_rul_rows * 10   # PHM2012: 每行 = 10 秒
        
        # 截断 DataFrame，模拟这是我们当前“已知”的所有数据
        df_truncated = df.iloc[:truncated_len]
            
        # a. 提取 42 个空间节点特征 (必须与训练集匹配)
        h_time_cols = ['h_skewness', 'h_kurtosis', 'h_p2p', 'h_shape_factor']
        v_time_cols = ['v_skewness', 'v_kurtosis', 'v_p2p', 'v_shape_factor']
        h_freq_cols = ['h_spectral_centroid'] + [f'h_fft_band_energy_ratio_{i}' for i in range(8)]
        v_freq_cols = ['v_spectral_centroid'] + [f'v_fft_band_energy_ratio_{i}' for i in range(8)]
        h_wpt_cols = [f'h_wpt_energy_ratio_{i}' for i in range(8)]
        v_wpt_cols = [f'v_wpt_energy_ratio_{i}' for i in range(8)]
        
        node_cols = h_time_cols + v_time_cols + h_freq_cols + v_freq_cols + h_wpt_cols + v_wpt_cols
        
        x_data = df_truncated[node_cols].values
        x_data = np.expand_dims(x_data, axis=-1) # 增加通道维度 (L, 42, 1)
        
        # b. 提取 3 个全局工况特征 (RMS, 温度, 以及新加入的单调健康因子 HI)
        cond_cols = ['h_rms', 'temperature', 'health_indicator']
        cond_data = df_truncated[cond_cols].values # (L, 3)
        
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
        # 4. 反归一化：total_len 已知，直接计算预测 RUL
        # ---------------------------------------------------------
        # 训练标签公式: p = remaining_rows / total_len
        # 因此: predicted_rul_rows = p * total_len
        actual_p = actual_rul_rows / total_len         # 实际剩余寿命百分比
        predicted_rul_sec = p * total_len * 10         # 预测 RUL 秒数
        error_percent = (predicted_rul_sec - actual_rul_sec) / actual_rul_sec * 100
        
        # 打印日志
        print(f"{bearing_name:<12} | {truncated_len:<10} | {p:.4f}      | {predicted_rul_sec:<12.1f} | {actual_rul_sec:<12} | {error_percent:>+.2f}%")
        
        # 保存到结果列表
        results.append({
            "Bearing": bearing_name,
            "截断特征长度(行)": truncated_len,
            "预测RUL百分比(p)": round(p, 4),
            "实际RUL百分比": round(actual_p, 4),
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
        actual_ruls = res_df["实际RUL(秒)"].values
        predicted_ruls = res_df["预测RUL(秒)"].values
        mae_value = np.mean(np.abs(actual_ruls - predicted_ruls))
        
        out_path = config.PREDICT_RESULTS_PATH
        res_df.to_csv(out_path, index=False)
        print("-" * 100)
        print(f"\n[评估统计] 训练集模拟推理 MAE: {mae_value:.2f} 秒")
        print(f"[*] 所有训练集模拟推理结果已成功保存至: {out_path}")
    else:
        print("[!] 没有产生任何预测结果，请检查数据。")

if __name__ == "__main__":
    main()
