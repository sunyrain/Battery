"""
Flow Matching 数据集

提供用于 Flow Matching 训练的数据集类，
支持信号级别和潜空间级别的配对数据。
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict, Callable, Any
from pathlib import Path
import logging

from .preprocessing import SignalProcessor, create_signal_augmentation

logger = logging.getLogger(__name__)


class BatteryFlowDataset(Dataset):
    """
    电池 Flow Matching 数据集 (信号级别)
    
    从 CSV 索引文件加载数据，返回配对的 (early, late) 信号用于训练。
    
    数据格式:
    - 索引 CSV 包含: filename, before_path, after_path, Cycle, state, ...
    - 每个 cycle 的信号存储在单独的 CSV 文件中
    
    Args:
        csv_path: 索引 CSV 文件路径
        data_root: 数据根目录
        signal_processor: 信号处理器
        min_cycle_gap: 配对的最小 cycle 差
        max_cycle_gap: 配对的最大 cycle 差
        augmentation: 数据增强函数
        max_pairs_per_anchor: 每个锚点的最大配对数
    """
    
    def __init__(
        self,
        csv_path: str,
        data_root: str,
        signal_processor: Optional[SignalProcessor] = None,
        min_cycle_gap: int = 10,
        max_cycle_gap: int = 100,
        augmentation: Optional[Callable] = None,
        max_pairs_per_anchor: int = 5,
    ):
        self.csv_path = csv_path
        self.data_root = data_root
        self.signal_processor = signal_processor or SignalProcessor()
        self.min_cycle_gap = min_cycle_gap
        self.max_cycle_gap = max_cycle_gap
        self.augmentation = augmentation
        self.max_pairs_per_anchor = max_pairs_per_anchor
        
        # 加载索引
        self.df = pd.read_csv(csv_path)
        logger.info(f"加载数据索引: {len(self.df)} 条记录")
        
        # 构建配对
        self.pairs = self._build_pairs()
        logger.info(f"构建配对: {len(self.pairs)} 对")
    
    def _build_pairs(self) -> List[Tuple[int, int]]:
        """
        构建 (early, late) 配对
        
        Returns:
            pairs: [(idx_early, idx_late), ...]
        """
        pairs = []
        
        # 按 cycle 分组
        cycles = self.df['Cycle'].unique()
        cycle_to_indices = {
            c: self.df[self.df['Cycle'] == c].index.tolist()
            for c in cycles
        }
        
        for cycle_early in cycles:
            early_indices = cycle_to_indices[cycle_early]
            
            # 找到满足 gap 条件的 late cycles
            valid_late_cycles = [
                c for c in cycles
                if self.min_cycle_gap <= (c - cycle_early) <= self.max_cycle_gap
            ]
            
            for cycle_late in valid_late_cycles:
                late_indices = cycle_to_indices[cycle_late]
                
                # 为每个 early 样本配对
                for idx_early in early_indices:
                    # 随机选择最多 max_pairs_per_anchor 个 late 样本
                    selected_late = random.sample(
                        late_indices, 
                        min(self.max_pairs_per_anchor, len(late_indices))
                    )
                    for idx_late in selected_late:
                        pairs.append((idx_early, idx_late))
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def _load_signal_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """加载单个样本的 before/after 信号对"""
        row = self.df.iloc[idx]
        before_path = row['before_path']
        after_path = row['after_path']
        
        before_tensor = self.signal_processor.process(before_path, self.data_root)
        after_tensor = self.signal_processor.process(after_path, self.data_root)
        
        return before_tensor, after_tensor
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个配对样本
        
        Returns:
            字典包含:
            - signal_after_0: early 的 after 信号
            - signal_before_0: early 的 before 信号
            - signal_after_1: late 的 after 信号
            - signal_before_1: late 的 before 信号
            - cycle_0: early 的 cycle 数
            - cycle_1: late 的 cycle 数
        """
        idx_early, idx_late = self.pairs[idx]
        
        # 加载 early 信号
        before_0, after_0 = self._load_signal_pair(idx_early)
        cycle_0 = self.df.iloc[idx_early]['Cycle']
        
        # 加载 late 信号
        before_1, after_1 = self._load_signal_pair(idx_late)
        cycle_1 = self.df.iloc[idx_late]['Cycle']
        
        # 数据增强
        if self.augmentation is not None:
            before_0 = self.augmentation(before_0)
            after_0 = self.augmentation(after_0)
            before_1 = self.augmentation(before_1)
            after_1 = self.augmentation(after_1)
        
        return {
            'signal_after_0': after_0,
            'signal_before_0': before_0,
            'signal_after_1': after_1,
            'signal_before_1': before_1,
            'cycle_0': torch.tensor(cycle_0, dtype=torch.long),
            'cycle_1': torch.tensor(cycle_1, dtype=torch.long),
        }


class LatentPairDataset(Dataset):
    """
    潜空间配对数据集
    
    使用预计算的潜空间向量，避免重复编码。
    适合大规模训练。
    
    Args:
        latent_vectors: [N, latent_dim] 所有样本的潜空间向量
        cycles: [N] 每个样本对应的 cycle 数
        min_cycle_gap: 最小 cycle 差
        max_cycle_gap: 最大 cycle 差
    """
    
    def __init__(
        self,
        latent_vectors: torch.Tensor,
        cycles: torch.Tensor,
        min_cycle_gap: int = 10,
        max_cycle_gap: int = 100,
    ):
        self.latent_vectors = latent_vectors
        self.cycles = cycles
        self.min_cycle_gap = min_cycle_gap
        self.max_cycle_gap = max_cycle_gap
        
        # 构建配对
        self.pairs = self._build_pairs()
        logger.info(f"潜空间配对数据集: {len(self.pairs)} 对")
    
    def _build_pairs(self) -> List[Tuple[int, int]]:
        """构建配对索引"""
        pairs = []
        n = len(self.cycles)
        
        for i in range(n):
            cycle_i = self.cycles[i].item()
            
            for j in range(n):
                if i == j:
                    continue
                
                cycle_j = self.cycles[j].item()
                gap = cycle_j - cycle_i
                
                if self.min_cycle_gap <= gap <= self.max_cycle_gap:
                    pairs.append((i, j))
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个配对
        
        Returns:
            z_0: early 的潜空间向量
            z_1: late 的潜空间向量
            cycle_0: early 的 cycle
            cycle_1: late 的 cycle
        """
        idx_0, idx_1 = self.pairs[idx]
        
        return {
            'z_0': self.latent_vectors[idx_0],
            'z_1': self.latent_vectors[idx_1],
            'cycle_0': self.cycles[idx_0],
            'cycle_1': self.cycles[idx_1],
        }


class OnlineFlowDataset(Dataset):
    """
    在线 Flow 数据集
    
    在 __getitem__ 中实时编码，适合小规模数据或验证。
    
    Args:
        csv_path: 索引 CSV 路径
        data_root: 数据根目录
        encoder: SmartWave Encoder
        signal_processor: 信号处理器
        device: 编码设备
    """
    
    def __init__(
        self,
        csv_path: str,
        data_root: str,
        encoder: torch.nn.Module,
        signal_processor: Optional[SignalProcessor] = None,
        device: torch.device = torch.device('cpu'),
        min_cycle_gap: int = 10,
        max_cycle_gap: int = 100,
    ):
        self.csv_path = csv_path
        self.data_root = data_root
        self.encoder = encoder
        self.encoder.eval()
        self.signal_processor = signal_processor or SignalProcessor()
        self.device = device
        self.min_cycle_gap = min_cycle_gap
        self.max_cycle_gap = max_cycle_gap
        
        # 加载索引
        self.df = pd.read_csv(csv_path)
        
        # 构建配对
        self.pairs = self._build_pairs()
    
    def _build_pairs(self) -> List[Tuple[int, int]]:
        """构建配对"""
        pairs = []
        cycles = self.df['Cycle'].unique()
        
        for i, row_i in self.df.iterrows():
            cycle_i = row_i['Cycle']
            
            for j, row_j in self.df.iterrows():
                if i == j:
                    continue
                
                cycle_j = row_j['Cycle']
                gap = cycle_j - cycle_i
                
                if self.min_cycle_gap <= gap <= self.max_cycle_gap:
                    pairs.append((i, j))
        
        # 限制配对数量
        if len(pairs) > 10000:
            pairs = random.sample(pairs, 10000)
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def _encode(self, idx: int) -> torch.Tensor:
        """编码单个样本"""
        row = self.df.iloc[idx]
        before_path = row['before_path']
        after_path = row['after_path']
        
        before_tensor = self.signal_processor.process(before_path, self.data_root)
        after_tensor = self.signal_processor.process(after_path, self.data_root)
        
        # 添加 batch 维度
        before_tensor = before_tensor.unsqueeze(0).to(self.device)
        after_tensor = after_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, _, _, z = self.encoder(after_tensor, before_tensor)
        
        return z.squeeze(0).cpu()
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        idx_0, idx_1 = self.pairs[idx]
        
        z_0 = self._encode(idx_0)
        z_1 = self._encode(idx_1)
        
        return {
            'z_0': z_0,
            'z_1': z_1,
            'cycle_0': torch.tensor(self.df.iloc[idx_0]['Cycle'], dtype=torch.long),
            'cycle_1': torch.tensor(self.df.iloc[idx_1]['Cycle'], dtype=torch.long),
        }


def create_flow_dataloader(
    csv_path: str,
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    signal_processor: Optional[SignalProcessor] = None,
    min_cycle_gap: int = 10,
    max_cycle_gap: int = 100,
    augmentation: bool = True,
    shuffle: bool = True,
    **kwargs,
) -> DataLoader:
    """
    创建 Flow Matching DataLoader 的便捷函数
    
    Args:
        csv_path: 索引 CSV 路径
        data_root: 数据根目录
        batch_size: 批大小
        num_workers: 工作进程数
        signal_processor: 信号处理器
        min_cycle_gap: 最小 cycle 差
        max_cycle_gap: 最大 cycle 差
        augmentation: 是否使用数据增强
        shuffle: 是否打乱
        **kwargs: 传递给 Dataset 的其他参数
    
    Returns:
        DataLoader 实例
    """
    augment_fn = create_signal_augmentation() if augmentation else None
    
    dataset = BatteryFlowDataset(
        csv_path=csv_path,
        data_root=data_root,
        signal_processor=signal_processor,
        min_cycle_gap=min_cycle_gap,
        max_cycle_gap=max_cycle_gap,
        augmentation=augment_fn,
        **kwargs,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader
