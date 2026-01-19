import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE (Rotary Position Embedding) 实现 - 改进版本，增强数值稳定性
    """
    def __init__(self, dim, max_seq_len=10000, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # 创建旋转角度，使用更稳定的计算方式
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算位置编码缓存
        self._build_cache(max_seq_len)
    
    def _build_cache(self, max_seq_len):
        """构建位置编码缓存"""
        # 创建位置索引
        seq_idx = torch.arange(max_seq_len, dtype=torch.float)
        
        # 计算频率矩阵
        inv_freq_tensor = torch.as_tensor(self.inv_freq)
        freqs = torch.outer(seq_idx, inv_freq_tensor)
        
        # 创建复数表示（更数值稳定）
        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()
        
        # 缓存cos和sin值 [seq_len, dim//2]
        self.register_buffer('freqs_cos', freqs_cos.clone().detach())
        self.register_buffer('freqs_sin', freqs_sin.clone().detach())
    
    def rotate_half(self, x):
        """旋转操作：将特征向量的后半部分移到前面并取负"""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            rotated_x: [batch_size, seq_len, d_model]
        """
        seq_len = x.shape[-2]
        
        # 如果序列长度超过缓存，重新构建缓存
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        
        # 获取对应长度的cos和sin值
        cos = torch.as_tensor(self.freqs_cos)[:seq_len]  # [seq_len, dim//2]
        sin = torch.as_tensor(self.freqs_sin)[:seq_len]  # [seq_len, dim//2]
        
        # 扩展到完整维度 [seq_len, d_model]
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        
        # 广播到batch维度 [1, seq_len, d_model]
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        
        # 应用旋转
        return (x * cos) + (self.rotate_half(x) * sin)

class _BaseTransformerBlock(nn.Module):
    """
    Pre-LN Transformer基础模块，封装公共的注意力和FFN结构。
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead,
            dropout=dropout,
            batch_first=True,
            add_bias_kv=False,
            add_zero_attn=False
        )
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def _self_attention(
        self,
        src,
        attn_mask=None,
        need_weights: bool = False,
        average_attn_weights: bool = True,
    ):
        src_norm = self.norm1(src)
        attn_output, attn_weights = self.self_attn(
            src_norm,
            src_norm,
            src_norm,
            attn_mask=attn_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )
        src = src + self.dropout1(attn_output)
        if need_weights:
            return src, attn_weights
        return src
    
    def _feed_forward(self, src):
        src_norm = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        return src + self.dropout2(src2)


class MaskedTransformerBlock(_BaseTransformerBlock):
    """
    默认使用上三角因果掩码的Transformer层。
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__(d_model, nhead, dim_feedforward, dropout)
    
    @staticmethod
    def _build_causal_mask(seq_len, device):
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        return torch.triu(mask, diagonal=1)
    
    def forward(
        self,
        src,
        attn_mask=None,
        return_attn_weights: bool = False,
        average_attn_weights: bool = True,
    ):
        if attn_mask is None:
            attn_mask = self._build_causal_mask(src.size(1), src.device)
        
        if return_attn_weights:
            src, attn_weights = self._self_attention(
                src,
                attn_mask=attn_mask,
                need_weights=True,
                average_attn_weights=average_attn_weights,
            )
        else:
            src = self._self_attention(
                src,
                attn_mask=attn_mask,
                need_weights=False,
            )
        src = self._feed_forward(src)
        if return_attn_weights:
            return src, attn_weights
        return src


class TransformerBlock(_BaseTransformerBlock):
    """
    不使用任何注意力掩码的标准Transformer层。
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__(d_model, nhead, dim_feedforward, dropout)
    
    def forward(
        self,
        src,
        return_attn_weights: bool = False,
        average_attn_weights: bool = True,
    ):
        if return_attn_weights:
            src, attn_weights = self._self_attention(
                src,
                attn_mask=None,
                need_weights=True,
                average_attn_weights=average_attn_weights,
            )
        else:
            src = self._self_attention(
                src,
                attn_mask=None,
                need_weights=False,
            )
        src = self._feed_forward(src)
        if return_attn_weights:
            return src, attn_weights
        return src


class RoPE(nn.Module):
    """
    标准RoPE (Rotary Position Embedding) 实现
    """
    def __init__(self, head_dim: int, ROPE_max_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        self.ROPE_max_len = ROPE_max_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)

        # [ROPE_max_len, head_dim/2]
        t = torch.arange(ROPE_max_len).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=True)
        self.register_buffer("sin", freqs.sin(), persistent=True)

    def _rotate_every_two(self, x):
        # x: [..., head_dim]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        # 交错配对的 2D 旋转
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def apply_rotary(self, x, seq_len: int):
        # x: [B, H, T, head_dim]
        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0)  # [1,1,T,head_dim/2]
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)

        # 扩到偶奇交错维度
        cos = torch.repeat_interleave(cos, 2, dim=-1)       # [1,1,T,head_dim]
        sin = torch.repeat_interleave(sin, 2, dim=-1)

        return (x * cos) + (self._rotate_every_two(x) * sin)

    def forward(self, q, k):
        # q,k: [B, H, T, head_dim]
        T = q.size(-2)
        q = self.apply_rotary(q, T)
        k = self.apply_rotary(k, T)
        return q, k


class WaveletScaleEmbedding(nn.Module):
    """
    小波尺度位置编码：为小波分解的不同尺度(level)和类型(band)学习可训练的嵌入
    
    用于频域分支，将小波系数的尺度信息和类型信息编码为位置特征
    - E_level: 编码小波分解的层级 (1, 2, 3, ..., max_level)
    - E_band: 编码系数类型 ('A'=近似系数, 'D'=细节系数)
    
    使用方式: token = projected_feature + E_level + E_band
    """
    def __init__(self, d_model: int, max_level: int = 10):
        """
        Args:
            d_model: 嵌入维度
            max_level: 最大小波分解层级
        """
        super().__init__()
        self.d_model = d_model
        self.max_level = max_level
        
        # 尺度嵌入: 为每个分解层级学习一个嵌入向量
        # 层级从1开始，所以需要 max_level+1 个嵌入（索引0不使用）
        self.level_embeddings = nn.Embedding(max_level + 1, d_model)
        
        # 类型嵌入: 2种类型 (0=近似系数A, 1=细节系数D)
        self.band_embeddings = nn.Embedding(2, d_model)
        
        # 初始化为小值，避免位置编码主导特征
        nn.init.normal_(self.level_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.band_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, x, level: int, band_type: str):
        """
        为输入特征添加小波尺度和类型的位置编码
        
        Args:
            x: [batch, seq_len, d_model] 或 [batch, d_model] 的特征
            level: 小波分解层级 (1, 2, 3, ..., max_level)
            band_type: 系数类型 ('A' 或 'D')
        
        Returns:
            x + E_level + E_band: 添加位置编码后的特征
        """
        assert 1 <= level <= self.max_level, f"level必须在[1, {self.max_level}]范围内"
        assert band_type in ['A', 'D'], "band_type必须是'A'或'D'"
        
        device = x.device
        
        # 获取尺度嵌入
        level_idx = torch.tensor([level], dtype=torch.long, device=device)
        E_level = self.level_embeddings(level_idx)  # [1, d_model]
        
        # 获取类型嵌入 (A=0, D=1)
        band_idx = torch.tensor([0 if band_type == 'A' else 1], dtype=torch.long, device=device)
        E_band = self.band_embeddings(band_idx)  # [1, d_model]
        
        # 广播到batch维度
        if x.dim() == 3:  # [B, T, d_model]
            E_level = E_level.unsqueeze(1)  # [1, 1, d_model]
            E_band = E_band.unsqueeze(1)    # [1, 1, d_model]
        elif x.dim() == 2:  # [B, d_model]
            pass  # E_level, E_band 已经是 [1, d_model]
        else:
            raise ValueError(f"输入x的维度必须是2或3，当前为{x.dim()}")
        
        # 加性位置编码
        return x + E_level + E_band
    
    def get_embedding(self, level: int, band_type: str):
        """
        直接获取指定level和band_type的组合嵌入向量（不加到特征上）
        
        Args:
            level: 小波分解层级
            band_type: 系数类型 ('A' 或 'D')
        
        Returns:
            E_level + E_band: [d_model] 的嵌入向量
        """
        assert 1 <= level <= self.max_level, f"level必须在[1, {self.max_level}]范围内"
        assert band_type in ['A', 'D'], "band_type必须是'A'或'D'"
        
        # 使用CPU构建索引
        level_idx = torch.tensor([level], dtype=torch.long)
        band_idx = torch.tensor([0 if band_type == 'A' else 1], dtype=torch.long)
        
        E_level = self.level_embeddings(level_idx).squeeze(0)  # [d_model]
        E_band = self.band_embeddings(band_idx).squeeze(0)     # [d_model]
        
        return E_level + E_band


class ROPETransformerBlock(nn.Module):
    """
    集成RoPE位置编码的Pre-LN Transformer块
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, ROPE_max_len=2048, theta=10000.0):
        super().__init__()
        assert d_model % nhead == 0, "d_model必须能被nhead整除"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # RoPE位置编码
        self.rope = RoPE(self.head_dim, ROPE_max_len=ROPE_max_len, theta=theta)
        
        # Q、K、V投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.GELU()
        
        # LayerNorm和Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)
    
    def _reshape_to_heads(self, x):
        """[B, T, d_model] -> [B, H, T, head_dim]"""
        B, T, _ = x.shape
        return x.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
    
    def _reshape_from_heads(self, x):
        """[B, H, T, head_dim] -> [B, T, d_model]"""
        B, H, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)
    
    def _attention(self, src, attn_mask=None):
        """带RoPE的自注意力"""
        B, T, _ = src.shape
        
        # 投影Q、K、V
        q = self.q_proj(src)  # [B, T, d_model]
        k = self.k_proj(src)
        v = self.v_proj(src)
        
        # 重塑为多头格式
        q = self._reshape_to_heads(q)  # [B, H, T, head_dim]
        k = self._reshape_to_heads(k)
        v = self._reshape_to_heads(v)
        
        # 应用RoPE
        q, k = self.rope(q, k)
        
        # 计算注意力分数
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, T, T]
        
        # 应用注意力掩码
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_attn(attn_weights)
        
        # 注意力输出
        attn_output = torch.matmul(attn_weights, v)  # [B, H, T, head_dim]
        attn_output = self._reshape_from_heads(attn_output)  # [B, T, d_model]
        attn_output = self.out_proj(attn_output)
        
        return attn_output
    
    def forward(self, src, attn_mask=None):
        """
        Args:
            src: [B, T, d_model]
            attn_mask: [T, T] 布尔掩码，True表示被掩盖的位置
        Returns:
            output: [B, T, d_model]
        """
        # Pre-LN自注意力
        src_norm = self.norm1(src)
        attn_output = self._attention(src_norm, attn_mask)
        src = src + self.dropout1(attn_output)
        
        # Pre-LN前馈网络
        src_norm = self.norm2(src)
        ffn_output = self.linear2(self.dropout_ffn(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(ffn_output)
        
        return src