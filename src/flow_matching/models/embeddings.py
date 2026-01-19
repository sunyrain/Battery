"""
嵌入模块 - 时间嵌入与条件嵌入

包含:
- SinusoidalTimeEmbedding: 正弦时间嵌入 (类似 Transformer/Diffusion)
- LearnedTimeEmbedding: 可学习时间嵌入
- ConditionEmbedding: 条件嵌入 (电池类型、倍率等)
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class SinusoidalTimeEmbedding(nn.Module):
    """
    正弦时间嵌入
    
    使用正弦/余弦函数将标量时间 t ∈ [0, 1] 映射到高维空间，
    类似于 Transformer 的位置编码和 Diffusion Models 的时间嵌入。
    
    数学形式:
        PE(t, 2i) = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))
    
    Args:
        embed_dim: 嵌入维度
        max_period: 最大周期 (默认 10000)
        learnable_scale: 是否学习缩放因子
    """
    
    def __init__(
        self,
        embed_dim: int,
        max_period: float = 10000.0,
        learnable_scale: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        
        # 预计算频率
        half_dim = embed_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("freqs", freqs)
        
        # 可选的可学习缩放
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("scale", torch.ones(1))
        
        # 可选的后处理 MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 时间张量, shape [batch] 或 [batch, 1] 或标量, 值域 [0, 1]
        
        Returns:
            时间嵌入, shape [batch, embed_dim]
        """
        # 确保 t 至少是 1 维的
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        # 如果是 [batch, 1]，squeeze 成 [batch]
        if t.dim() == 2 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        
        # 现在 t 是 [batch]，扩展维度用于广播: [batch, 1] * [half_dim] -> [batch, half_dim]
        t_scaled = t.unsqueeze(-1) * self.freqs * self.scale
        
        # 正弦/余弦嵌入
        embeddings = torch.cat([
            torch.sin(t_scaled),
            torch.cos(t_scaled),
        ], dim=-1)
        
        # 如果 embed_dim 是奇数，需要填充
        if self.embed_dim % 2 == 1:
            embeddings = torch.cat([
                embeddings,
                torch.zeros_like(embeddings[..., :1])
            ], dim=-1)
        
        # MLP 后处理
        embeddings = self.mlp(embeddings)
        
        return embeddings


class LearnedTimeEmbedding(nn.Module):
    """
    可学习时间嵌入
    
    通过离散化时间步然后查表实现，适合时间步数有限的场景。
    同时结合连续嵌入以处理任意时间值。
    
    Args:
        embed_dim: 嵌入维度
        num_steps: 离散时间步数 (用于查表)
        interpolate: 是否对离散嵌入进行插值
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_steps: int = 1000,
        interpolate: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_steps = num_steps
        self.interpolate = interpolate
        
        # 离散嵌入表
        self.embedding_table = nn.Embedding(num_steps + 1, embed_dim)
        
        # 连续嵌入 (正弦)
        self.sinusoidal = SinusoidalTimeEmbedding(embed_dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embedding_table.weight, std=0.02)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 时间张量, 值域 [0, 1]
        
        Returns:
            时间嵌入, shape [..., embed_dim]
        """
        # 连续嵌入
        continuous_emb = self.sinusoidal(t)
        
        # 离散嵌入 (带插值)
        t_scaled = t * self.num_steps
        
        if self.interpolate:
            t_floor = t_scaled.floor().long().clamp(0, self.num_steps - 1)
            t_ceil = (t_floor + 1).clamp(0, self.num_steps)
            t_frac = (t_scaled - t_floor.float()).unsqueeze(-1)
            
            emb_floor = self.embedding_table(t_floor)
            emb_ceil = self.embedding_table(t_ceil)
            discrete_emb = emb_floor * (1 - t_frac) + emb_ceil * t_frac
        else:
            t_idx = t_scaled.round().long().clamp(0, self.num_steps)
            discrete_emb = self.embedding_table(t_idx)
        
        # 融合
        combined = torch.cat([continuous_emb, discrete_emb], dim=-1)
        return self.fusion(combined)


class ConditionEmbedding(nn.Module):
    """
    条件嵌入模块
    
    将电池的外部条件（类型、倍率、温度等）嵌入到统一的条件向量中。
    支持离散条件（类别）和连续条件（数值）。
    
    Args:
        embed_dim: 嵌入维度
        num_battery_types: 电池类型数量 (离散)
        num_c_rates: 充放电倍率数量 (离散)
        continuous_dim: 连续条件维度 (温度、SOC等)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_battery_types: int = 10,
        num_c_rates: int = 20,
        continuous_dim: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 离散条件嵌入
        self.battery_type_emb = nn.Embedding(num_battery_types, embed_dim // 2)
        self.c_rate_emb = nn.Embedding(num_c_rates, embed_dim // 2)
        
        # 连续条件投影
        self.continuous_proj = nn.Sequential(
            nn.Linear(continuous_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.battery_type_emb, self.c_rate_emb]:
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        battery_type: Optional[torch.Tensor] = None,
        c_rate: Optional[torch.Tensor] = None,
        continuous_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            battery_type: 电池类型索引 [batch]
            c_rate: 充放电倍率索引 [batch]
            continuous_cond: 连续条件 [batch, continuous_dim]
        
        Returns:
            条件嵌入 [batch, embed_dim]
        """
        embeddings = []
        
        # 处理离散条件
        if battery_type is not None:
            embeddings.append(self.battery_type_emb(battery_type))
        
        if c_rate is not None:
            embeddings.append(self.c_rate_emb(c_rate))
        
        # 拼接离散嵌入
        if embeddings:
            discrete_emb = torch.cat(embeddings, dim=-1)
            # 如果维度不够，padding
            if discrete_emb.shape[-1] < self.embed_dim:
                padding = torch.zeros(
                    *discrete_emb.shape[:-1], 
                    self.embed_dim - discrete_emb.shape[-1],
                    device=discrete_emb.device
                )
                discrete_emb = torch.cat([discrete_emb, padding], dim=-1)
        else:
            # 无离散条件时使用零向量
            batch_size = continuous_cond.shape[0] if continuous_cond is not None else 1
            discrete_emb = torch.zeros(batch_size, self.embed_dim, device=self._get_device())
        
        # 处理连续条件
        if continuous_cond is not None:
            continuous_emb = self.continuous_proj(continuous_cond)
        else:
            continuous_emb = torch.zeros_like(discrete_emb)
        
        # 融合
        combined = torch.cat([discrete_emb, continuous_emb], dim=-1)
        return self.fusion(combined)
    
    def _get_device(self):
        return next(self.parameters()).device


class FourierFeatures(nn.Module):
    """
    随机傅里叶特征嵌入
    
    用于将低维输入（如时间 t）映射到高维空间，
    提供更好的高频信息捕捉能力。
    
    基于论文: "Fourier Features Let Networks Learn High Frequency Functions"
    
    Args:
        input_dim: 输入维度
        embed_dim: 输出嵌入维度
        scale: 频率缩放因子
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        embed_dim: int = 256,
        scale: float = 16.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # 随机频率矩阵 (固定)
        B = torch.randn(input_dim, embed_dim // 2) * scale
        self.register_buffer("B", B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [..., input_dim]
        
        Returns:
            傅里叶特征 [..., embed_dim]
        """
        # x @ B: [..., embed_dim // 2]
        x_proj = 2 * math.pi * x @ self.B
        
        # 拼接 sin 和 cos
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
