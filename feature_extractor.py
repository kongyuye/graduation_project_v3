import os
import glob
import re
import numpy as np
import pandas as pd
import scipy.stats as stats
import pywt
from typing import Dict, List, Tuple

class FeatureExtractor:
    """
    轴承数据特征提取器
    用于从时域、频域和时频域提取关键物理特征。
    """
    def __init__(self, fs: int = 25600):
        self.fs = fs

    def extract_time_domain(self, signal: np.ndarray) -> Dict[str, float]:
        """
        时域特征提取
        提取能够反映设备退化的统计学特征
        :param signal: 1D numpy array, 振动信号片段 (例如 2560 个点)
        :return: 包含各项时域特征的字典
        """
        # 异常值处理：如果全为NaN或者长度为0，返回0
        if len(signal) == 0 or np.all(np.isnan(signal)):
            return {
                'rms': 0.0, 'kurtosis': 0.0, 'skewness': 0.0, 
                'p2p': 0.0, 'shape_factor': 0.0
            }

        # 清除NaN
        sig = signal[~np.isnan(signal)]
        
        # 均方根 (RMS) - 反映信号的整体能量
        rms = np.sqrt(np.mean(sig ** 2))
        
        # 峭度 (Kurtosis) - 反映信号中冲击成分的敏感性（轴承早期故障敏感）
        kurtosis = stats.kurtosis(sig, fisher=False)
        
        # 偏度 (Skewness) - 反映信号概率密度分布的不对称性
        skewness = stats.skew(sig)
        
        # 峰峰值 (Peak-to-Peak) - 信号波动的最大范围
        p2p = np.max(sig) - np.min(sig)
        
        # 绝对均值 (Mean Absolute Value)
        mav = np.mean(np.abs(sig))
        
        # 波形因子 (Shape Factor) - RMS与绝对均值的比值
        shape_factor = rms / mav if mav != 0 else 0.0
        
        # Python 的 字典 (Dictionary) 返回语句 。
        return {
            'rms': rms,
            'kurtosis': kurtosis,
            'skewness': skewness,
            'p2p': p2p,
            'shape_factor': shape_factor
        }

    def extract_freq_domain(self, signal: np.ndarray, n_bins: int = 8) -> Dict[str, float]:
        """
        频域特征提取 (FFT)
        提取频谱的主频能量分布以及频率统计特征
        :param signal: 1D numpy array, 振动信号片段
        :param n_bins: 将频谱划分为几个频带计算能量分布
        :return: 包含各项频域特征的字典
        """
        if len(signal) == 0 or np.all(np.isnan(signal)):
            return {'spectral_centroid': 0.0}

        sig = signal[~np.isnan(signal)]
        n = len(sig)
        
        # 计算 FFT
        fft_values = np.fft.rfft(sig)
        # 频率轴
        freqs = np.fft.rfftfreq(n, d=1/self.fs)
        # 幅值谱
        amplitudes = np.abs(fft_values) / n
        
        # 为了避免除零错误
        sum_amp = np.sum(amplitudes)
        if sum_amp == 0:
            sum_amp = 1e-10
            
        # 1. 频谱质心 (Spectral Centroid) - 频率的主体位置
        spectral_centroid = np.sum(freqs * amplitudes) / sum_amp
        
        features = {'spectral_centroid': spectral_centroid}
        
        # 2. 频带能量分布 (Frequency Band Energy Distribution)
        # 将整个Nyquist频率范围均分为 n_bins 个频带，计算每个频带的能量比例
        band_edges = np.linspace(0, self.fs / 2, n_bins + 1)
        total_energy = np.sum(amplitudes ** 2)
        if total_energy == 0:
            total_energy = 1e-10
            
        for i in range(n_bins):
            # 找到在当前频带内的频率索引
            idx = np.where((freqs >= band_edges[i]) & (freqs < band_edges[i+1]))[0]
            band_energy = np.sum(amplitudes[idx] ** 2)
            features[f'fft_band_energy_ratio_{i}'] = band_energy / total_energy
            
        return features

    def extract_time_freq_domain(self, signal: np.ndarray, wavelet: str = 'db4', level: int = 3) -> Dict[str, float]:
        """
        时频域特征提取 (WPT)
        使用小波包分解提取各频带能量比例
        :param signal: 1D numpy array, 振动信号片段
        :param wavelet: 小波基名称，默认为 'db4'
        :param level: 分解层数，默认为 3
        :return: 包含各项时频域特征的字典
        """
        if len(signal) == 0 or np.all(np.isnan(signal)):
            # 3层分解会产生 2^3 = 8 个节点
            return {f'wpt_energy_ratio_{i}': 0.0 for i in range(2**level)}

        sig = signal[~np.isnan(signal)]
        
        # 小波包分解
        wp = pywt.WaveletPacket(data=sig, wavelet=wavelet, mode='symmetric', maxlevel=level)
        
        # 获取第 level 层的所有节点
        nodes = wp.get_level(level, 'freq') # 按频率排序
        
        energies = []
        for node in nodes:
            # 计算每个节点的能量
            energy = np.sum(node.data ** 2)
            energies.append(energy)
            
        total_energy = sum(energies)
        if total_energy == 0:
            total_energy = 1e-10
            
        # 计算能量比例
        features = {}
        for i, energy in enumerate(energies):
            features[f'wpt_energy_ratio_{i}'] = energy / total_energy
            
        return features

    def extract_all(self, signal: np.ndarray, prefix: str = '') -> Dict[str, float]:
        """
        提取所有维度的特征
        :param signal: 1D numpy array, 振动信号片段
        :param prefix: 特征名称前缀 (如 'horiz_' 或 'vert_')
        :return: 包含所有特征的字典
        """
        features = {}
        
        # 时域特征
        t_feat = self.extract_time_domain(signal)
        # 频域特征
        f_feat = self.extract_freq_domain(signal)
        # 时频域特征
        tf_feat = self.extract_time_freq_domain(signal)
        
        for k, v in t_feat.items(): features[prefix + k] = v
        for k, v in f_feat.items(): features[prefix + k] = v
        for k, v in tf_feat.items(): features[prefix + k] = v
        
        return features

from typing import List
import pandas as pd

class BaselineNormalizer:
    """
    基于正常状态的基线归一化策略 (Baseline Normalization)
    用于解决 PHM 2012 截断数据集导致的全局统计尺度失真问题。
    """
    def __init__(self, method: str = 'z-score', baseline_ratio: float = 0.05, min_k_points: int = 10):
        """
        :param method: 归一化方法，推荐使用 'z-score' 以反映物理偏离度
        :param baseline_ratio: 选取时间序列前百分之几的数据作为确定无疑的健康基线
        :param min_k_points: 兜底策略，确保至少有 N 个点用于计算统计量
        """
        self.method = method
        self.baseline_ratio = baseline_ratio
        self.min_k_points = min_k_points
        self.params = {} # 存储健康基线的参数

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        在轴承序列上提取【健康基线参数】并转换整个生命周期/截断序列
        """
        df_norm = df.copy()
        n_samples = len(df)
        
        # 1. 自适应划定基线窗口 K 的大小 (对应公式中的 X_base 集合)
        k_points = max(int(n_samples * self.baseline_ratio), self.min_k_points)
        
        # 2. 截取前 K 个点作为绝对健康基线 (Healthy Baseline)
        df_baseline = df[columns].iloc[:k_points]
        
        for col in columns:
            if self.method == 'z-score':
                # 仅利用健康期数据计算基线均值 (mu_base) 和基线标准差 (sigma_base) 
                col_mean = df_baseline[col].mean()
                col_std = df_baseline[col].std()
                
                self.params[col] = {'mean': col_mean, 'std': col_std}
                
                # 使用固定的健康基准标尺转换当前序列的后续所有时间步 
                if col_std == 0:
                    df_norm[col] = 0
                else:
                    df_norm[col] = (df[col] - col_mean) / col_std
                    
            elif self.method == 'min-max':
                col_min = df_baseline[col].min()
                col_max = df_baseline[col].max()
                
                self.params[col] = {'min': col_min, 'max': col_max}
                
                if col_max - col_min == 0:
                    df_norm[col] = 0
                else:
                    df_norm[col] = (df[col] - col_min) / (col_max - col_min)
                    
        return df_norm

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        使用已提取的健康基线参数转换目标测试集 
        (仅在需要将某一轴承的健康标准强加于另一轴承时使用，常规独立提取直接用 fit_transform)
        """
        df_norm = df.copy()
        for col in columns:
            if col not in self.params:
                continue
                
            if self.method == 'z-score':
                col_mean = self.params[col]['mean']
                col_std = self.params[col]['std']
                if col_std == 0:
                    df_norm[col] = 0
                else:
                    df_norm[col] = (df[col] - col_mean) / col_std
        return df_norm

class HealthIndicatorConstructor:
    """
    融合多域信息的单调健康因子 (HI) 无监督构建
    基于 PHM 2012 理论：利用马氏距离 (MD) 构建平滑且单调的综合健康因子
    """
    def __init__(self, baseline_ratio: float = 0.05, min_k_points: int = 10, smooth_method: str = 'ewma', ewma_alpha: float = 0.05):
        """
        :param baseline_ratio: 前百分之几的数据作为健康基线
        :param min_k_points: 最少点数兜底
        :param smooth_method: 平滑方法，'ewma' 或 'sg' (Savitzky-Golay)
        :param ewma_alpha: 指数加权移动平均的衰减率
        """
        self.baseline_ratio = baseline_ratio
        self.min_k_points = min_k_points
        self.smooth_method = smooth_method
        self.ewma_alpha = ewma_alpha
        self.params = {}

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        提取特征并生成单调健康因子 (HI) 列
        """
        df_hi = df.copy()
        n_samples = len(df)
        
        # 自适应划定基线窗口 K 的大小
        k_points = max(int(n_samples * self.baseline_ratio), self.min_k_points)
        
        # 截取前 K 个点作为绝对健康基线
        baseline_data = df[columns].iloc[:k_points].values
        all_data = df[columns].values
        
        # === 步骤 1：构建多元健康基线分布模型 ===
        # 计算基线均值 mu_base
        mu_base = np.mean(baseline_data, axis=0)
        
        # 计算基线协方差矩阵 Sigma_base
        sigma_base = np.cov(baseline_data, rowvar=False)
        
        # 【鲁棒性约束】添加极小正则化项，防止协方差矩阵为奇异矩阵 (Singular Matrix)
        reg_term = np.eye(sigma_base.shape[0]) * 1e-6
        sigma_base += reg_term
        
        # 【鲁棒性约束】使用伪逆 (Pseudo-inverse) 替代普通逆矩阵，增强数值稳定性
        sigma_base_inv = np.linalg.pinv(sigma_base)
        
        # === 步骤 2：马氏距离 (MD) 映射计算 ===
        # 计算所有时间步的 MD^2 = (x_t - mu_base)^T * Sigma_base^-1 * (x_t - mu_base)
        diff = all_data - mu_base # shape: (N, D)
        # 向量化计算马氏距离平方：沿特征维度求内积
        md_squared = np.sum(np.dot(diff, sigma_base_inv) * diff, axis=1)
        
        # === 步骤 3：单调性平滑处理 ===
        # 克服局部物理扰动（如轴承自愈效应），确保 HI 具有平滑趋势
        if self.smooth_method == 'ewma':
            # 叠加指数加权移动平均 (EWMA) 滤波器
            smoothed_hi = pd.Series(md_squared).ewm(alpha=self.ewma_alpha, adjust=False).mean().values
        elif self.smooth_method == 'sg':
            # 使用 Savitzky-Golay 滤波器
            import scipy.signal
            window_length = min(31, n_samples)
            if window_length % 2 == 0: window_length -= 1
            if window_length < 3: window_length = 3
            smoothed_hi = scipy.signal.savgol_filter(md_squared, window_length, 2)
        else:
            smoothed_hi = md_squared
            
        # 施加绝对单调递增约束 (累计最大值) 以保证 HI 单调性
        monotonic_hi = np.maximum.accumulate(smoothed_hi)
        
        # (可选) 将 HI 归一化到 [0, 1] 表示退化度
        hi_min, hi_max = np.min(monotonic_hi), np.max(monotonic_hi)
        if hi_max > hi_min:
            monotonic_hi = (monotonic_hi - hi_min) / (hi_max - hi_min)
        else:
            monotonic_hi = np.zeros_like(monotonic_hi)
            
        # === 步骤 4：数据输出更新 ===
        # 将平滑后的健康因子作为一个新的特征列拼接到 DataFrame 中
        df_hi['health_indicator'] = monotonic_hi
        
        return df_hi

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        兼容接口：使用已有的分布参数对新数据进行变换 (测试集适用，若独立基线则直接调用fit_transform)
        """
        # 如果需要将某个轴承的 HI 映射关系强加给另一个轴承，可以在此实现。
        # 当前基于 PDF 理论，各个轴承独立计算基线，所以直接调用 fit_transform 即可。
        return self.fit_transform(df, columns)

def process_dataset(raw_dir: str, output_dir: str, is_train: bool = True):
    """
    数据处理流水线 (已修复版)
    """
    extractor = FeatureExtractor()
    
    # 获取所有的轴承目录
    bearing_dirs = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
    
    for bearing_name in bearing_dirs:
        print(f"Processing {bearing_name}...")
        bearing_path = os.path.join(raw_dir, bearing_name)
        all_features = []
        
        # 修复 1：安全的时序排序逻辑，强制根据文件名中的数字索引进行排序
        def extract_file_idx(filename):
            match = re.search(r'acc_(\d+)\.csv', os.path.basename(filename))
            return int(match.group(1)) if match else -1
            
        csv_files = glob.glob(os.path.join(bearing_path, "acc_*.csv"))
        # 过滤掉无法匹配正则的异常文件并按真实数字时序排序
        csv_files = sorted([f for f in csv_files if extract_file_idx(f) != -1], key=extract_file_idx)
        
        temp_cache = {}
        
        for file_path in csv_files:
            try:
                # 工业数据常常因为末尾逗号导致列错位，使用 on_bad_lines='skip' 提升鲁棒性
                df = pd.read_csv(file_path, header=None, on_bad_lines='skip')
                
                if df.shape[1] < 6:
                    print(f"Warning: Unexpected data shape {df.shape} in {file_path}")
                    continue

                # 修复 3：强制转换为数值类型，将无法解析的字符转为 NaN，避免后续 numpy 计算崩溃
                horiz_acc = pd.to_numeric(df.iloc[:, 4], errors='coerce').values
                vert_acc = pd.to_numeric(df.iloc[:, 5], errors='coerce').values
                
                # 提取特征
                h_feat = extractor.extract_all(horiz_acc, prefix='h_')
                v_feat = extractor.extract_all(vert_acc, prefix='v_')
                
                # 温度对齐逻辑
                acc_filename = os.path.basename(file_path)
                acc_idx = extract_file_idx(acc_filename)
                
                # 修复 2：初始化为 NaN 而非 0.0，防止零值破坏马氏距离基线
                temp_mean = np.nan 
                
                temp_idx = (acc_idx - 1) // 6 + 1
                temp_file = os.path.join(bearing_path, f"temp_{temp_idx:05d}.csv")
                
                if temp_file in temp_cache:
                    temp_mean = temp_cache[temp_file]
                elif os.path.exists(temp_file):
                    df_temp = pd.read_csv(temp_file, header=None, on_bad_lines='skip')
                    if df_temp.shape[1] >= 5:
                        # 同样强制数值转换
                        temp_series = pd.to_numeric(df_temp.iloc[:, 4], errors='coerce')
                        temp_mean = temp_series.mean()
                        # 仅当温度有效时才存入缓存
                        if not np.isnan(temp_mean):
                            temp_cache[temp_file] = temp_mean
                
                # 合并特征
                row_feat = {'bearing': bearing_name, 'file': acc_filename}
                row_feat.update(h_feat)
                row_feat.update(v_feat)
                row_feat['temperature'] = temp_mean
                
                all_features.append(row_feat)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
        # 转换为 DataFrame
        features_df = pd.DataFrame(all_features)
        
        # 修复 4：空级联拦截，防止协方差矩阵和归一化崩溃
        if features_df.empty:
            print(f"Skipping {bearing_name}: No valid features extracted.")
            continue
            
        # 修复 2（延续）：对缺失的温度值进行前向填充和后向填充（维持物理状态平稳），而不是用0干扰模型
        if 'temperature' in features_df.columns:
            features_df['temperature'] = features_df['temperature'].ffill().bfill()
            # 如果整个序列全是 NaN（比如温度传感器坏了全没采到），填充该列均值或 0 避免后续报错
            features_df['temperature'] = features_df['temperature'].fillna(0)
        
        # 归一化与健康因子构建
        feat_cols = [c for c in features_df.columns if c not in ['bearing', 'file']]
        
        normalizer = BaselineNormalizer(method='z-score', baseline_ratio=0.05)
        features_df = normalizer.fit_transform(features_df, feat_cols)
        
        hi_constructor = HealthIndicatorConstructor(baseline_ratio=0.05, smooth_method='ewma', ewma_alpha=0.1)
        features_df = hi_constructor.fit_transform(features_df, feat_cols)
                
        # 保存结果
        prefix_name = 'train' if is_train else 'test'
        output_file = os.path.join(output_dir, f"{bearing_name}_{prefix_name}_features.csv")
        features_df.to_csv(output_file, index=False)
        print(f"Saved {output_file} (Shape: {features_df.shape})")

if __name__ == "__main__":
    import config
    
    # 原始数据路径从 config.RAW_DATA_DIR 读取，输出到项目下的 processed_data 目录
    train_dir = os.path.join(config.RAW_DATA_DIR, "Learning_set")
    test_dir  = os.path.join(config.RAW_DATA_DIR, "Test_set")
    output_dir = os.path.join(config.BASE_DIR, "processed_data")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== 开始处理训练集 ===")
    process_dataset(train_dir, output_dir, is_train=True)
    
    print("\n=== 开始处理测试集 ===")
    process_dataset(test_dir, output_dir, is_train=False)
    
    print("\n数据处理全部完成！")
