"""
速度场网络 - Flow Matching 的核心组件

速度场网络 v_θ(z_t, t, c) 学习从源分布到目标分布的向量场，
使得 ODE dz/dt = v_θ(z, t) 能够将初始分布传输到目标分布。

架构特点:
1. 时间自适应: 通过 AdaLN 或 FiLM 将时间信息注入网络
2. 条件注入: 支持外部条件（电池类型等）
3. 残差连接: 保证梯度流通和训练稳定性
4. 跳跃连接: U-Net 风格的特征融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .embeddings import SinusoidalTimeEmbedding, ConditionEmbedding


class AdaptiveLayerNorm(nn.Module):
    """
    自适应层归一化 (AdaLN)
    
    通过时间/条件嵌入调制 LayerNorm 的 scale 和 shift 参数，
    实现时间自适应的特征调制。
    
    y = (x - μ) / σ * (1 + γ(t)) + β(t)
    
    Args:
        normalized_shape: 归一化的形状
        cond_dim: 条件嵌入维度
    """
    
    def __init__(self, normalized_shape: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        
        # 从条件嵌入预测 scale 和 shift
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, normalized_shape * 2),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # 初始化为恒等变换
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [batch, ..., normalized_shape]
            cond: 条件嵌入 [batch, cond_dim]
        
        Returns:
            调制后的特征
        """
        # 预测 scale 和 shift
        params = self.proj(cond)
        scale, shift = params.chunk(2, dim=-1)
        
        # 扩展维度以匹配 x
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        
        # 应用 AdaLN
        x_norm = self.norm(x)
        return x_norm * (1 + scale) + shift


class FiLMLayer(nn.Module):
    """
    特征线性调制层 (FiLM)
    
    通过条件嵌入对特征进行仿射变换:
    y = γ(c) * x + β(c)
    
    Args:
        feature_dim: 特征维度
        cond_dim: 条件嵌入维度
    """
    
    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, feature_dim * 2)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        # 初始化 scale 为 1
        self.proj.bias.data[:self.proj.out_features // 2] = 1.0
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        params = self.proj(cond)
        gamma, beta = params.chunk(2, dim=-1)
        
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        
        return gamma * x + beta


class ResidualBlock(nn.Module):
    """
    残差块 with 时间/条件调制
    
    结构:
    x -> [AdaLN -> Linear -> Activation -> Linear] + x
    
    Args:
        dim: 特征维度
        cond_dim: 条件维度
        dropout: Dropout 概率
        use_adaln: 是否使用 AdaLN (否则使用 FiLM)
    """
    
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        dropout: float = 0.1,
        use_adaln: bool = True,
    ):
        super().__init__()
        
        # 第一层
        if use_adaln:
            self.norm1 = AdaptiveLayerNorm(dim, cond_dim)
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.film1 = FiLMLayer(dim, cond_dim)
        
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # 第二层
        if use_adaln:
            self.norm2 = AdaptiveLayerNorm(dim * 4, cond_dim)
        else:
            self.norm2 = nn.LayerNorm(dim * 4)
            self.film2 = FiLMLayer(dim * 4, cond_dim)
        
        self.linear2 = nn.Linear(dim * 4, dim)
        
        self.use_adaln = use_adaln
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # 第一层
        if self.use_adaln:
            x = self.norm1(x, cond)
        else:
            x = self.film1(self.norm1(x), cond)
        
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        
        # 第二层
        if self.use_adaln:
            x = self.norm2(x, cond)
        else:
            x = self.film2(self.norm2(x), cond)
        
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x + residual


class VelocityNetwork(nn.Module):
    """
    速度场神经网络
    
    核心任务: 学习 v_θ(z_t, t, c)，即在时间 t、条件 c 下，
    潜空间点 z_t 应该移动的方向和速度。
    
    架构: 类 U-Net 的 MLP，带残差连接和跳跃连接
    
    Args:
        latent_dim: 潜空间维度 (与 SmartWave Encoder 输出一致)
        hidden_dim: 隐藏层维度
        time_embed_dim: 时间嵌入维度
        cond_embed_dim: 条件嵌入维度
        num_layers: 残差块数量
        dropout: Dropout 概率
        use_adaln: 是否使用 AdaLN
        use_skip_connections: 是否使用跳跃连接
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        time_embed_dim: int = 128,
        cond_embed_dim: int = 128,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_adaln: bool = True,
        use_skip_connections: bool = True,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_skip_connections = use_skip_connections
        
        # ============ 嵌入层 ============
        # 时间嵌入
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        
        # 条件嵌入的投影
        self.cond_proj = nn.Linear(cond_embed_dim, hidden_dim) if cond_embed_dim > 0 else None
        
        # 组合嵌入维度 = time_embed_dim + hidden_dim (projected cond)
        combined_cond_dim = time_embed_dim + (hidden_dim if cond_embed_dim > 0 else 0)
        
        # ============ 输入投影 ============
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # ============ Encoder (下采样路径) ============
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downs = nn.ModuleList()
        
        current_dim = hidden_dim
        encoder_dims = [current_dim]
        
        for i in range(num_layers // 2):
            self.encoder_blocks.append(
                ResidualBlock(current_dim, combined_cond_dim, dropout, use_adaln)
            )
            # 维度变换 (可选)
            next_dim = min(current_dim * 2, hidden_dim * 4)
            self.encoder_downs.append(nn.Linear(current_dim, next_dim))
            current_dim = next_dim
            encoder_dims.append(current_dim)
        
        # ============ 中间层 (Bottleneck) ============
        self.mid_block1 = ResidualBlock(current_dim, combined_cond_dim, dropout, use_adaln)
        self.mid_block2 = ResidualBlock(current_dim, combined_cond_dim, dropout, use_adaln)
        
        # ============ Decoder (上采样路径) ============
        self.decoder_blocks = nn.ModuleList()
        self.decoder_ups = nn.ModuleList()
        
        for i in range(num_layers // 2):
            # 跳跃连接的维度
            skip_dim = encoder_dims[-(i + 2)] if use_skip_connections else 0
            
            # 上采样
            prev_dim = encoder_dims[-(i + 2)]
            self.decoder_ups.append(nn.Linear(current_dim, prev_dim))
            
            # 如果有跳跃连接，需要融合
            if use_skip_connections:
                self.decoder_blocks.append(
                    nn.ModuleList([
                        nn.Linear(prev_dim * 2, prev_dim),  # 融合跳跃连接
                        ResidualBlock(prev_dim, combined_cond_dim, dropout, use_adaln),
                    ])
                )
            else:
                self.decoder_blocks.append(
                    ResidualBlock(prev_dim, combined_cond_dim, dropout, use_adaln)
                )
            
            current_dim = prev_dim
        
        # ============ 输出投影 ============
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # 保存 encoder 维度用于跳跃连接
        self.encoder_dims = encoder_dims
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 输出层初始化为零，使初始预测为零向量
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)
        
        # 其他层使用 Xavier 初始化
        for module in self.modules():
            if isinstance(module, nn.Linear) and module not in self.output_proj:
                if hasattr(module, 'weight'):
                    nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        预测速度场
        
        Args:
            z_t: 当前潜空间点 [batch, latent_dim]
            t: 当前时间 [batch] 或 [batch, 1]，值域 [0, 1]
            condition: 条件嵌入 [batch, cond_embed_dim]，可选
        
        Returns:
            velocity: 预测的速度向量 [batch, latent_dim]
        """
        batch_size = z_t.shape[0]
        device = z_t.device
        
        # ============ 构建条件嵌入 ============
        # 时间嵌入
        t_emb = self.time_embed(t)  # [batch, time_embed_dim]
        
        # 组合时间和条件
        if self.cond_proj is not None:
            if condition is not None:
                cond_proj = self.cond_proj(condition)  # [batch, hidden_dim]
            else:
                # 当 condition 为 None 时，使用零向量以保持维度一致
                cond_proj = torch.zeros(batch_size, self.hidden_dim, device=device)
            combined_cond = torch.cat([t_emb, cond_proj], dim=-1)
        else:
            combined_cond = t_emb
        
        # ============ 输入投影 ============
        h = self.input_proj(z_t)  # [batch, hidden_dim]
        
        # ============ Encoder 路径 ============
        skip_connections = [h]
        
        for i, (block, down) in enumerate(zip(self.encoder_blocks, self.encoder_downs)):
            h = block(h, combined_cond)
            h = down(h)
            h = F.gelu(h)
            skip_connections.append(h)
        
        # ============ 中间层 ============
        h = self.mid_block1(h, combined_cond)
        h = self.mid_block2(h, combined_cond)
        
        # ============ Decoder 路径 ============
        for i, (up, block) in enumerate(zip(self.decoder_ups, self.decoder_blocks)):
            h = up(h)
            h = F.gelu(h)
            
            if self.use_skip_connections:
                # 获取对应的跳跃连接
                skip = skip_connections[-(i + 2)]
                # 融合
                h = torch.cat([h, skip], dim=-1)
                fusion, res_block = block
                h = fusion(h)
                h = res_block(h, combined_cond)
            else:
                h = block(h, combined_cond)
        
        # ============ 输出投影 ============
        velocity = self.output_proj(h)
        
        return velocity
    
    def get_num_params(self) -> int:
        """返回参数数量"""
        return sum(p.numel() for p in self.parameters())


class LightweightVelocityNetwork(nn.Module):
    """
    轻量级速度场网络
    
    更简单的 MLP 结构，适合快速实验和小规模数据。
    
    Args:
        latent_dim: 潜空间维度
        hidden_dim: 隐藏层维度
        time_embed_dim: 时间嵌入维度
        num_layers: 隐藏层数量
        dropout: Dropout 概率
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        time_embed_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # 时间嵌入
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        
        # 输入: z_t + time_emb
        input_dim = latent_dim + time_embed_dim
        
        # MLP 层
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.net = nn.Sequential(*layers)
        
        # 零初始化输出层
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        预测速度场
        
        Args:
            z_t: [batch, latent_dim]
            t: [batch] 或 [batch, 1] 或标量
            condition: 未使用，保持接口一致
        
        Returns:
            velocity: [batch, latent_dim]
        """
        # 确保 t 是正确的形状 [batch]
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(z_t.shape[0])
        elif t.dim() == 2:
            t = t.squeeze(-1)
        
        t_emb = self.time_embed(t)
        x = torch.cat([z_t, t_emb], dim=-1)
        return self.net(x)

    def get_num_params(self) -> int:
        """返回参数总数"""
        return sum(p.numel() for p in self.parameters())
