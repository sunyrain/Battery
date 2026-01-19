"""
数据预处理模块

提供超声信号的加载、预处理和标准化功能。
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Optional, Union, Dict, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_signal_from_csv(
    csv_path: str,
    data_root: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 CSV 文件加载超声信号
    
    CSV 格式: 两列，第一列时间，第二列电压
    
    Args:
        csv_path: CSV 文件路径
        data_root: 数据根目录（用于拼接相对路径）
    
    Returns:
        time: 时间数组
        voltage: 电压数组
    """
    if data_root is not None:
        csv_path = os.path.join(data_root, csv_path)
    
    try:
        df = pd.read_csv(csv_path, header=None)
        time = df.iloc[:, 0].values
        voltage = df.iloc[:, 1].values
        return time.astype(np.float32), voltage.astype(np.float32)
    except Exception as e:
        logger.error(f"加载信号失败: {csv_path}, 错误: {e}")
        raise


def normalize_signal(
    signal: np.ndarray,
    method: str = "zscore",
    stats: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    信号标准化
    
    Args:
        signal: 输入信号
        method: 标准化方法 ("zscore", "minmax", "robust")
        stats: 预计算的统计量（用于推理时保持一致）
    
    Returns:
        normalized: 标准化后的信号
        stats: 统计量字典
    """
    if method == "zscore":
        if stats is None:
            mean = signal.mean()
            std = signal.std()
            stats = {"mean": float(mean), "std": float(std)}
        else:
            mean = stats["mean"]
            std = stats["std"]
        
        normalized = (signal - mean) / (std + 1e-8)
    
    elif method == "minmax":
        if stats is None:
            min_val = signal.min()
            max_val = signal.max()
            stats = {"min": float(min_val), "max": float(max_val)}
        else:
            min_val = stats["min"]
            max_val = stats["max"]
        
        normalized = (signal - min_val) / (max_val - min_val + 1e-8)
    
    elif method == "robust":
        # 基于中位数和 IQR 的鲁棒标准化
        if stats is None:
            median = np.median(signal)
            q75, q25 = np.percentile(signal, [75, 25])
            iqr = q75 - q25
            stats = {"median": float(median), "iqr": float(iqr)}
        else:
            median = stats["median"]
            iqr = stats["iqr"]
        
        normalized = (signal - median) / (iqr + 1e-8)
    
    else:
        raise ValueError(f"未知的标准化方法: {method}")
    
    return normalized.astype(np.float32), stats


def pad_or_truncate(
    signal: np.ndarray,
    target_length: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    将信号填充或截断到目标长度
    
    Args:
        signal: 输入信号
        target_length: 目标长度
        pad_value: 填充值
    
    Returns:
        处理后的信号
    """
    current_length = len(signal)
    
    if current_length == target_length:
        return signal
    elif current_length > target_length:
        # 截断（从中间取）
        start = (current_length - target_length) // 2
        return signal[start:start + target_length]
    else:
        # 填充（两端对称填充）
        pad_total = target_length - current_length
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(signal, (pad_left, pad_right), 'constant', constant_values=pad_value)


class SignalProcessor:
    """
    信号处理器
    
    提供统一的信号预处理流程。
    
    Args:
        signal_length: 目标信号长度
        normalize_method: 标准化方法
        global_stats: 全局统计量（用于保持标准化一致性）
    """
    
    def __init__(
        self,
        signal_length: int = 3000,
        normalize_method: str = "zscore",
        global_stats: Optional[Dict[str, float]] = None,
    ):
        self.signal_length = signal_length
        self.normalize_method = normalize_method
        self.global_stats = global_stats
    
    def process(
        self,
        csv_path: str,
        data_root: Optional[str] = None,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        处理单个信号文件
        
        Args:
            csv_path: CSV 文件路径
            data_root: 数据根目录
            return_stats: 是否返回统计量
        
        Returns:
            tensor: [1, signal_length] 处理后的信号张量
            stats: (可选) 统计量字典
        """
        # 加载信号
        _, voltage = load_signal_from_csv(csv_path, data_root)
        
        # 标准化
        normalized, stats = normalize_signal(
            voltage, 
            method=self.normalize_method,
            stats=self.global_stats,
        )
        
        # 长度调整
        processed = pad_or_truncate(normalized, self.signal_length)
        
        # 转换为张量 [1, length]
        tensor = torch.from_numpy(processed).unsqueeze(0)
        
        if return_stats:
            return tensor, stats
        return tensor
    
    def process_pair(
        self,
        before_path: str,
        after_path: str,
        data_root: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理 before/after 信号对
        
        Args:
            before_path: before 信号路径
            after_path: after 信号路径
            data_root: 数据根目录
        
        Returns:
            before_tensor: [1, signal_length]
            after_tensor: [1, signal_length]
        """
        before_tensor = self.process(before_path, data_root)
        after_tensor = self.process(after_path, data_root)
        return before_tensor, after_tensor
    
    def compute_global_stats(
        self,
        csv_paths: List[str],
        data_root: Optional[str] = None,
        sample_ratio: float = 0.1,
    ) -> Dict[str, float]:
        """
        从数据集计算全局统计量
        
        Args:
            csv_paths: 所有 CSV 文件路径
            data_root: 数据根目录
            sample_ratio: 采样比例（加速计算）
        
        Returns:
            全局统计量
        """
        import random
        
        # 采样
        num_samples = max(1, int(len(csv_paths) * sample_ratio))
        sampled_paths = random.sample(csv_paths, num_samples)
        
        # 收集所有值
        all_values = []
        for path in sampled_paths:
            try:
                _, voltage = load_signal_from_csv(path, data_root)
                all_values.extend(voltage.tolist())
            except Exception as e:
                logger.warning(f"跳过文件 {path}: {e}")
        
        all_values = np.array(all_values)
        
        if self.normalize_method == "zscore":
            stats = {
                "mean": float(all_values.mean()),
                "std": float(all_values.std()),
            }
        elif self.normalize_method == "minmax":
            stats = {
                "min": float(all_values.min()),
                "max": float(all_values.max()),
            }
        elif self.normalize_method == "robust":
            stats = {
                "median": float(np.median(all_values)),
                "iqr": float(np.percentile(all_values, 75) - np.percentile(all_values, 25)),
            }
        else:
            raise ValueError(f"未知的标准化方法: {self.normalize_method}")
        
        self.global_stats = stats
        logger.info(f"计算全局统计量完成: {stats}")
        
        return stats


def create_signal_augmentation(
    time_jitter: float = 0.01,
    noise_scale: float = 0.001,
    amplitude_scale: Tuple[float, float] = (0.95, 1.05),
) -> callable:
    """
    创建信号增强函数
    
    Args:
        time_jitter: 时间抖动幅度
        noise_scale: 高斯噪声标准差
        amplitude_scale: 幅度缩放范围
    
    Returns:
        augment_fn: 增强函数
    """
    def augment(signal: torch.Tensor) -> torch.Tensor:
        """
        对信号进行随机增强
        
        Args:
            signal: [C, L] 输入信号
        
        Returns:
            augmented: 增强后的信号
        """
        augmented = signal.clone()
        
        # 高斯噪声
        if noise_scale > 0:
            noise = torch.randn_like(augmented) * noise_scale
            augmented = augmented + noise
        
        # 幅度缩放
        if amplitude_scale != (1.0, 1.0):
            scale = torch.empty(1).uniform_(amplitude_scale[0], amplitude_scale[1])
            augmented = augmented * scale
        
        # 时间抖动 (简单的随机位移)
        if time_jitter > 0:
            shift = int(signal.shape[-1] * time_jitter * torch.rand(1).item())
            if torch.rand(1) > 0.5:
                augmented = torch.roll(augmented, shifts=shift, dims=-1)
            else:
                augmented = torch.roll(augmented, shifts=-shift, dims=-1)
        
        return augmented
    
    return augment
