"""
基于 Score 的潜空间缓存模块

使用 SelfLearning 打分器预计算 score，作为 Flow Matching 的时间标签。

核心改进:
- score ∈ [0, 1] 直接作为时间 t，不再依赖 max_cycle
- 虚拟 cycle = score × 99 + 1 ∈ [1, 100]
- 支持跨电池泛化
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


class ScoreBasedLatentCache:
    """
    基于 Score 的潜空间缓存
    
    同时计算:
    1. 潜空间向量 z (来自 SmartWave Encoder)
    2. 退化分数 score (来自 SelfLearning 打分器)
    
    score 直接作为 Flow Matching 的时间条件 t ∈ [0, 1]
    
    Args:
        cache_dir: 缓存目录
        encoder: SmartWave Encoder (用于计算 z)
        scorer: SelfLearning 打分器 (用于计算 score)
        signal_processor: 信号处理器
    """
    
    def __init__(
        self,
        cache_dir: str,
        encoder: Optional[torch.nn.Module] = None,
        scorer: Optional[torch.nn.Module] = None,
        signal_processor: Optional[SignalProcessor] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.encoder = encoder
        self.scorer = scorer
        self.signal_processor = signal_processor or SignalProcessor()
        
        self.latent_file = self.cache_dir / "latent_vectors_with_scores.pt"
        self.metadata_file = self.cache_dir / "metadata_with_scores.json"
        
        self._latent_vectors = None
        self._scores = None
        self._cycles = None  # 保留原始 cycle 用于参考
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
        计算所有样本的潜空间向量和 score 并保存
        
        Args:
            csv_path: 索引 CSV 路径
            data_root: 数据根目录
            device: 计算设备
            batch_size: 批大小
        """
        if self.encoder is None:
            raise RuntimeError("Encoder 未设置")
        if self.scorer is None:
            raise RuntimeError("Scorer 未设置")
        
        self.encoder.to(device)
        self.encoder.eval()
        self.scorer.to(device)
        self.scorer.eval()
        
        # 加载索引
        df = pd.read_csv(csv_path)
        logger.info(f"开始计算 Score-based 潜空间缓存: {len(df)} 个样本")
        
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
        all_scores = []
        all_cycles = []
        all_indices = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing latents and scores"):
                before = batch['before'].to(device)
                after = batch['after'].to(device)
                
                # 1. 编码到潜空间 (SmartWave)
                _, _, _, z = self.encoder(after, before)
                
                # 2. 计算 score (SelfLearning)
                score = self.scorer(after, before)
                
                all_latents.append(z.cpu())
                all_scores.append(score.cpu())
                all_cycles.extend(batch['cycle'].tolist())
                all_indices.extend(batch['idx'].tolist())
        
        # 合并
        latent_vectors = torch.cat(all_latents, dim=0)
        scores = torch.cat(all_scores, dim=0)
        cycles = torch.tensor(all_cycles, dtype=torch.long)
        indices = torch.tensor(all_indices, dtype=torch.long)
        
        # 保存
        torch.save({
            'latent_vectors': latent_vectors,
            'scores': scores,
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
            'min_score': float(scores.min()),
            'max_score': float(scores.max()),
            'mean_score': float(scores.mean()),
            'std_score': float(scores.std()),
            'score_based': True,  # 标记这是 score-based 缓存
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Score-based 潜空间缓存已保存: {self.latent_file}")
        logger.info(f"  - 样本数: {len(df)}")
        logger.info(f"  - 潜空间维度: {latent_vectors.shape[1]}")
        logger.info(f"  - Cycle 范围: [{metadata['min_cycle']}, {metadata['max_cycle']}]")
        logger.info(f"  - Score 范围: [{metadata['min_score']:.4f}, {metadata['max_score']:.4f}]")
        logger.info(f"  - Score 均值: {metadata['mean_score']:.4f} ± {metadata['std_score']:.4f}")
        
        self._latent_vectors = latent_vectors
        self._scores = scores
        self._cycles = cycles
        self._metadata = metadata
    
    def load(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        加载缓存的潜空间向量和 scores
        
        Returns:
            latent_vectors: [N, latent_dim]
            scores: [N] 退化分数 ∈ [0, 1]
            cycles: [N] 原始 cycle (用于参考)
            metadata: 元数据字典
        """
        if not self.is_cached:
            raise RuntimeError("缓存不存在，请先调用 compute_and_save()")
        
        data = torch.load(self.latent_file)
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self._latent_vectors = data['latent_vectors']
        self._scores = data['scores']
        self._cycles = data['cycles']
        self._metadata = metadata
        
        logger.info(f"加载 Score-based 潜空间缓存: {len(self._latent_vectors)} 个样本")
        logger.info(f"  - Score 范围: [{metadata['min_score']:.4f}, {metadata['max_score']:.4f}]")
        
        return data['latent_vectors'], data['scores'], data['cycles'], metadata
    
    def get_latent_by_score_range(
        self,
        min_score: float,
        max_score: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取指定 score 范围内的潜空间向量
        
        Returns:
            latents: 潜空间向量
            corresponding_scores: 对应的 scores
            indices: 原始索引
        """
        if self._latent_vectors is None:
            self.load()
        
        mask = (self._scores >= min_score) & (self._scores <= max_score)
        indices = torch.where(mask)[0]
        latents = self._latent_vectors[mask]
        corresponding_scores = self._scores[mask]
        
        return latents, corresponding_scores, indices
    
    def sample_pair_by_score(
        self,
        min_gap: float = 0.1,
        max_gap: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        基于 score 差异采样配对
        
        Returns:
            z_0: 低 score (健康) 的潜空间
            z_1: 高 score (退化) 的潜空间
            score_0: z_0 的 score
            score_1: z_1 的 score
        """
        if self._latent_vectors is None:
            self.load()
        
        n = len(self._scores)
        
        while True:
            idx_0 = torch.randint(n, (1,)).item()
            score_0 = self._scores[idx_0].item()
            
            # 找到满足条件的 idx_1 (score_1 > score_0)
            score_diff = self._scores - score_0
            valid_mask = (score_diff >= min_gap) & (score_diff <= max_gap)
            valid_indices = torch.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                idx_1 = valid_indices[torch.randint(len(valid_indices), (1,))].item()
                score_1 = self._scores[idx_1].item()
                
                return (
                    self._latent_vectors[idx_0],
                    self._latent_vectors[idx_1],
                    score_0,
                    score_1,
                )
    
    def clear(self):
        """清除缓存"""
        if self.latent_file.exists():
            os.remove(self.latent_file)
        if self.metadata_file.exists():
            os.remove(self.metadata_file)
        
        self._latent_vectors = None
        self._scores = None
        self._cycles = None
        self._metadata = None
        
        logger.info("Score-based 缓存已清除")


class ScoreBasedFlowDataset(Dataset):
    """
    基于 Score 的 Flow Matching 数据集
    
    使用 score 作为时间条件 t，而不是 cycle/max_cycle。
    
    配对策略:
    - 采样 (z_0, score_0) 和 (z_1, score_1)
    - 要求 score_1 > score_0 (即更退化的样本)
    - t = score 直接作为时间条件
    
    Args:
        latent_vectors: [N, latent_dim] 潜空间向量
        scores: [N] 退化分数 ∈ [0, 1]
        min_score_gap: 最小 score 差
        max_score_gap: 最大 score 差
    """
    
    def __init__(
        self,
        latent_vectors: torch.Tensor,
        scores: torch.Tensor,
        min_score_gap: float = 0.1,
        max_score_gap: float = 0.8,
        cycles: torch.Tensor = None,  # 可选，用于记录
    ):
        self.latent_vectors = latent_vectors
        self.scores = scores
        self.cycles = cycles
        self.min_score_gap = min_score_gap
        self.max_score_gap = max_score_gap
        
        # 构建配对
        self.pairs = self._build_pairs()
        logger.info(f"Score-based 配对数据集: {len(self.pairs)} 对")
        logger.info(f"  - Score gap 范围: [{min_score_gap}, {max_score_gap}]")
    
    def _build_pairs(self) -> List[Tuple[int, int]]:
        """基于 score 差异构建配对"""
        pairs = []
        n = len(self.scores)
        
        # 按 score 排序，便于快速查找
        sorted_indices = torch.argsort(self.scores)
        sorted_scores = self.scores[sorted_indices]
        
        for i in range(n):
            score_i = self.scores[i].item()
            
            # 找到 score 差在 [min_gap, max_gap] 范围内的样本
            for j in range(n):
                if i == j:
                    continue
                
                score_j = self.scores[j].item()
                gap = score_j - score_i  # score_j > score_i
                
                if self.min_score_gap <= gap <= self.max_score_gap:
                    pairs.append((i, j))
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个配对
        
        Returns:
            z_0: 低 score 的潜空间向量 (更健康)
            z_1: 高 score 的潜空间向量 (更退化)
            t_0: z_0 的 score (作为时间 t)
            t_1: z_1 的 score (作为时间 t)
            cycle_0, cycle_1: 原始 cycle (可选，用于记录)
        """
        idx_0, idx_1 = self.pairs[idx]
        
        result = {
            'z_0': self.latent_vectors[idx_0],
            'z_1': self.latent_vectors[idx_1],
            't_0': self.scores[idx_0],  # score 直接作为 t
            't_1': self.scores[idx_1],
        }
        
        # 添加原始 cycle 信息 (用于记录/调试)
        if self.cycles is not None:
            result['cycle_0'] = self.cycles[idx_0]
            result['cycle_1'] = self.cycles[idx_1]
        
        return result


def score_to_virtual_cycle(score: torch.Tensor, max_virtual_cycle: int = 100) -> torch.Tensor:
    """
    将 score 转换为虚拟 cycle
    
    虚拟 cycle = score × (max_virtual_cycle - 1) + 1
    
    score=0 → cycle=1 (完全健康)
    score=1 → cycle=100 (完全退化)
    
    Args:
        score: ∈ [0, 1]
        max_virtual_cycle: 最大虚拟 cycle (默认 100)
    
    Returns:
        virtual_cycle: ∈ [1, max_virtual_cycle]
    """
    return score * (max_virtual_cycle - 1) + 1


def virtual_cycle_to_score(cycle: torch.Tensor, max_virtual_cycle: int = 100) -> torch.Tensor:
    """
    将虚拟 cycle 转换为 score
    
    score = (cycle - 1) / (max_virtual_cycle - 1)
    """
    return (cycle - 1).float() / (max_virtual_cycle - 1)
