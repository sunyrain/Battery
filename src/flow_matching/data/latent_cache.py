"""
潜空间缓存模块

预计算并缓存所有样本的潜空间向量，加速 Flow Matching 训练。
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Tuple, List
from pathlib import Path
from tqdm import tqdm
import logging
import json

from .preprocessing import SignalProcessor

logger = logging.getLogger(__name__)


class LatentCache:
    """
    潜空间缓存
    
    预计算所有样本的潜空间向量并保存到磁盘，
    训练时直接加载，避免重复编码。
    
    Args:
        cache_dir: 缓存目录
        encoder: SmartWave Encoder
        signal_processor: 信号处理器
    """
    
    def __init__(
        self,
        cache_dir: str,
        encoder: Optional[torch.nn.Module] = None,
        signal_processor: Optional[SignalProcessor] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.encoder = encoder
        self.signal_processor = signal_processor or SignalProcessor()
        
        self.latent_file = self.cache_dir / "latent_vectors.pt"
        self.metadata_file = self.cache_dir / "metadata.json"
        
        self._latent_vectors = None
        self._metadata = None
    
    @property
    def is_cached(self) -> bool:
        """检查缓存是否存在"""
        return self.latent_file.exists() and self.metadata_file.exists()
    
    def compute_and_save(
        self,
        csv_path: str,
        data_root: str,
        device: torch.device = torch.device('cpu'),
        batch_size: int = 32,
    ):
        """
        计算所有样本的潜空间向量并保存
        
        Args:
            csv_path: 索引 CSV 路径
            data_root: 数据根目录
            device: 计算设备
            batch_size: 批大小
        """
        if self.encoder is None:
            raise RuntimeError("Encoder 未设置")
        
        self.encoder.to(device)
        self.encoder.eval()
        
        # 加载索引
        df = pd.read_csv(csv_path)
        logger.info(f"开始计算潜空间缓存: {len(df)} 个样本")
        
        # 临时 Dataset
        class TempDataset(Dataset):
            def __init__(self, df, data_root, processor):
                self.df = df
                self.data_root = data_root
                self.processor = processor
            
            def __len__(self):
                return len(self.df)
            
            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                before = self.processor.process(row['before_path'], self.data_root)
                after = self.processor.process(row['after_path'], self.data_root)
                return {
                    'before': before,
                    'after': after,
                    'cycle': row['Cycle'],
                    'idx': idx,
                }
        
        dataset = TempDataset(df, data_root, self.signal_processor)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0  # Windows 兼容
        )
        
        all_latents = []
        all_cycles = []
        all_indices = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing latent vectors"):
                before = batch['before'].to(device)
                after = batch['after'].to(device)
                
                # 编码
                _, _, _, z = self.encoder(after, before)
                
                all_latents.append(z.cpu())
                all_cycles.extend(batch['cycle'].tolist())
                all_indices.extend(batch['idx'].tolist())
        
        # 合并
        latent_vectors = torch.cat(all_latents, dim=0)
        cycles = torch.tensor(all_cycles, dtype=torch.long)
        indices = torch.tensor(all_indices, dtype=torch.long)
        
        # 保存
        torch.save({
            'latent_vectors': latent_vectors,
            'cycles': cycles,
            'indices': indices,
        }, self.latent_file)
        
        # 保存元数据
        metadata = {
            'csv_path': csv_path,
            'data_root': data_root,
            'num_samples': len(df),
            'latent_dim': latent_vectors.shape[1],
            'min_cycle': int(cycles.min()),
            'max_cycle': int(cycles.max()),
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"潜空间缓存已保存: {self.latent_file}")
        logger.info(f"  - 样本数: {len(df)}")
        logger.info(f"  - 潜空间维度: {latent_vectors.shape[1]}")
        logger.info(f"  - Cycle 范围: [{metadata['min_cycle']}, {metadata['max_cycle']}]")
        
        self._latent_vectors = latent_vectors
        self._metadata = metadata
    
    def load(self) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        加载缓存的潜空间向量
        
        Returns:
            latent_vectors: [N, latent_dim]
            cycles: [N]
            metadata: 元数据字典
        """
        if not self.is_cached:
            raise RuntimeError("缓存不存在，请先调用 compute_and_save()")
        
        data = torch.load(self.latent_file)
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self._latent_vectors = data['latent_vectors']
        self._metadata = metadata
        
        logger.info(f"加载潜空间缓存: {len(self._latent_vectors)} 个样本")
        
        return data['latent_vectors'], data['cycles'], metadata
    
    def get_latent_by_cycle(
        self,
        cycles: torch.Tensor,
        target_cycle: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定 cycle 的所有潜空间向量
        
        Args:
            cycles: 所有样本的 cycle
            target_cycle: 目标 cycle
        
        Returns:
            latents: 该 cycle 的潜空间向量
            indices: 对应的索引
        """
        if self._latent_vectors is None:
            self.load()
        
        mask = cycles == target_cycle
        indices = torch.where(mask)[0]
        latents = self._latent_vectors[mask]
        
        return latents, indices
    
    def get_latent_by_cycle_range(
        self,
        cycles: torch.Tensor,
        min_cycle: int,
        max_cycle: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取指定 cycle 范围内的潜空间向量
        
        Returns:
            latents: 潜空间向量
            corresponding_cycles: 对应的 cycle
            indices: 原始索引
        """
        if self._latent_vectors is None:
            self.load()
        
        mask = (cycles >= min_cycle) & (cycles <= max_cycle)
        indices = torch.where(mask)[0]
        latents = self._latent_vectors[mask]
        corresponding_cycles = cycles[mask]
        
        return latents, corresponding_cycles, indices
    
    def sample_pair(
        self,
        cycles: torch.Tensor,
        min_gap: int = 10,
        max_gap: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        随机采样一对 (early, late) 潜空间向量
        
        Returns:
            z_0: early 潜空间
            z_1: late 潜空间
            cycle_0: early cycle
            cycle_1: late cycle
        """
        if self._latent_vectors is None:
            self.load()
        
        n = len(cycles)
        
        while True:
            idx_0 = torch.randint(n, (1,)).item()
            cycle_0 = cycles[idx_0].item()
            
            # 找到满足条件的 idx_1
            valid_mask = (cycles - cycle_0 >= min_gap) & (cycles - cycle_0 <= max_gap)
            valid_indices = torch.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                idx_1 = valid_indices[torch.randint(len(valid_indices), (1,))].item()
                cycle_1 = cycles[idx_1].item()
                
                return (
                    self._latent_vectors[idx_0],
                    self._latent_vectors[idx_1],
                    cycle_0,
                    cycle_1,
                )
    
    def clear(self):
        """清除缓存"""
        if self.latent_file.exists():
            os.remove(self.latent_file)
        if self.metadata_file.exists():
            os.remove(self.metadata_file)
        
        self._latent_vectors = None
        self._metadata = None
        
        logger.info("缓存已清除")
