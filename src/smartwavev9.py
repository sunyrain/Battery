# 时频域部分先差分再编码#

import torch
import torch.nn as nn

from typing import Dict, Any, Optional

from src.wavelet import wavelet_coeff_raw
from src.transformer import (
    TransformerBlock,
    ROPETransformerBlock,
    WaveletScaleEmbedding,
)


class TimeDomainBranch(nn.Module):
    """
    时域分支：对原始信号做嵌入并用Transformer提取特征。
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        ROPE_max_len: int,
        dropout: float = 0.1,
        patch_size: int = 50,
        stride: int = 50,
    ) -> None:
        super().__init__()
        # 使用 Conv1d 进行 patch embedding
        # input: [B, C, T] -> output: [B, D, T']
        self.patch_embed = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=d_model, 
            kernel_size=patch_size, 
            stride=stride
        )
        # self.pos_embedding = RotaryPositionalEmbedding(d_model, ROPE_max_len)
        self.transformer_layers = nn.ModuleList(
            [
                ROPETransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    ROPE_max_len=ROPE_max_len,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, seq_len] 的输入信号
        """

        # 使用 Conv1d 进行 patch embedding
        # [B, C, T] -> [B, D, T']
        x = self.patch_embed(x)
        # [B, D, T'] -> [B, T', D]
        x = x.transpose(1, 2)
        # x = self.pos_embedding(x)

        for layer in self.transformer_layers:
            x = layer(x)

        pooled = x.mean(dim=1)

        return self.output_projection(pooled)


class DeltaFrequencyDomainBranch(nn.Module):
    """
    差分频域分支：先对 after 和 before 信号做小波分解，
    然后直接在系数层面做差分（after - before），
    再通过 Transformer 编码差分后的系数。
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float = 0.1,
        max_level: int = 6,
        wavelet: str = 'sym4',
        print_coeffs: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_level = max_level
        self.wavelet = wavelet
        self.print_coeffs = print_coeffs  # 是否打印系数信息
        self.channel_projections = nn.ModuleDict()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 使用小波尺度位置编码替代传统RoPE
        self.wavelet_pos_embedding = WaveletScaleEmbedding(d_model, max_level=max_level)
        
        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_projection = nn.Linear(d_model, d_model)

        # # Init CLS token
        # nn.init.normal_(self.cls_token, std=0.02)
        
        # # 用于控制打印频率（避免过多输出）
        # self.forward_count = 0
        # self.print_interval = 100  # 每100次forward打印一次
    
    # def _normalize_coeffs(self, freq_coeffs: dict, device) -> dict:
    #     """
    #     对每一层小波系数进行 z-score 归一化
        
    #     Args:
    #         freq_coeffs: 小波系数字典 {name: tensor}
    #         device: 设备
            
    #     Returns:
    #         归一化后的小波系数字典
    #     """
    #     normalized_coeffs = {}
        
    #     for name, coeff_tensor in freq_coeffs.items():
    #         coeff = coeff_tensor.to(device=device, dtype=torch.float32).flatten()
            
    #         # 计算均值和标准差
    #         mean = coeff.mean()
    #         std = coeff.std()
            
    #         # z-score 归一化: (x - mean) / std
    #         # 添加小的epsilon避免除零
    #         epsilon = 1e-8
    #         if std > epsilon:
    #             normalized_coeff = (coeff - mean) / (std + epsilon)
    #         else:
    #             # 如果标准差接近0，说明系数基本恒定，直接减去均值
    #             normalized_coeff = coeff - mean
            
    #         # 重新reshape回原来的形状
    #         normalized_coeffs[name] = normalized_coeff.reshape(coeff_tensor.shape)
        
    #     return normalized_coeffs

    def forward(
        self, 
        x_after: torch.Tensor,
        x_before: torch.Tensor,
        # return_attn: bool = False,
        # print_coeffs: bool = None
    ) -> torch.Tensor:
        """
        Args:
            x_after: [batch, channels, seq_len] 运行后信号
            x_before: [batch, channels, seq_len] 运行前信号
            # return_attn: 是否返回注意力权重
            # print_coeffs: 是否打印小波系数统计信息 (None表示使用默认设置)
        """
        device = x_after.device
        batch_size = x_after.size(0)
        flattened_samples = []
        
        for i in range(batch_size):
            # 处理 after 信号
            signal_after = x_after[i:i+1]
            if signal_after.dim() == 2:
                signal_after = signal_after.unsqueeze(1)
            if signal_after.size(1) > 1:
                signal_after = signal_after[:, -1:, :]

            # 处理 before 信号
            signal_before = x_before[i:i+1]
            if signal_before.dim() == 2:
                signal_before = signal_before.unsqueeze(1)
            if signal_before.size(1) > 1:
                signal_before = signal_before[:, -1:, :]

            # 小波分解 after 和 before
            freq_coeffs_after = wavelet_coeff_raw(
                signal_after,
                level=self.max_level,
                wavelet=self.wavelet,
                device=device,
            )
            
            freq_coeffs_before = wavelet_coeff_raw(
                signal_before,
                level=self.max_level,
                wavelet=self.wavelet,
                device=device,
            )

            sample_tokens = []
            
            # 对每个系数做差分：after - before
            for name in freq_coeffs_after.keys():
                coeff_after = freq_coeffs_after[name].to(device=device, dtype=torch.float32).flatten()
                coeff_before = freq_coeffs_before[name].to(device=device, dtype=torch.float32).flatten()
                
                # 差分系数
                delta_coeff = coeff_after - coeff_before

                # 动态创建投影层
                if name not in self.channel_projections:
                    self.channel_projections[name] = nn.Linear(
                        delta_coeff.numel(), self.d_model
                    ).to(device)

                # 投影差分系数
                projected = self.channel_projections[name](delta_coeff)  # [d_model]
                
                # 解析小波系数的尺度和类型信息
                # 名称格式: 'cA6', 'cD6', 'cD5', ..., 'cD1'
                if name.startswith('cA'):
                    band_type = 'A'  # 近似系数
                    level = int(name[2:])
                elif name.startswith('cD'):
                    band_type = 'D'  # 细节系数
                    level = int(name[2:])
                else:
                    raise ValueError(f"未知的小波系数名称格式: {name}")
                
                # 添加小波尺度位置编码: token = projected + E_level + E_band
                projected = projected.unsqueeze(0)  # [1, d_model]
                projected_with_pos = self.wavelet_pos_embedding(projected, level, band_type)
                projected = projected_with_pos.squeeze(0)  # [d_model]
                
                sample_tokens.append(projected)
            

            flattened_samples.append(torch.stack(sample_tokens, dim=0))

        x = torch.stack(flattened_samples, dim=0)  # [B, Seq_len, D]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, Seq_len + 1, D]

        # 注意：小波尺度位置编码已经在投影时添加到每个token上了
        # 这里不再需要额外的位置编码

        for layer in self.transformer_layers:
            x = layer(x)

        # Pool from CLS token (index 0)
        pooled = x[:, 0]
        output = self.output_projection(pooled)
        
        return output

    # def print_wavelet_stats_detailed(
    #     self,
    #     x_after: torch.Tensor,
    #     x_before: torch.Tensor,
    # ):
    #     """
    #     打印详细的小波系数统计信息（包括范围、能量、百分位数等）
    #     """
    #     import numpy as np
        
    #     device = x_after.device if hasattr(x_after, 'device') else next(self.parameters()).device
    #     batch_size = x_after.size(0)
        
    #     for i in range(batch_size):
    #         # 处理 after 信号
    #         signal_after = x_after[i:i+1]
    #         if signal_after.dim() == 2:
    #             signal_after = signal_after.unsqueeze(1)
    #         if signal_after.size(1) > 1:
    #             signal_after = signal_after[:, -1:, :]

    #         # 处理 before 信号
    #         signal_before = x_before[i:i+1]
    #         if signal_before.dim() == 2:
    #             signal_before = signal_before.unsqueeze(1)
    #         if signal_before.size(1) > 1:
    #             signal_before = signal_before[:, -1:, :]

    #         # 小波分解
    #         freq_coeffs_after = wavelet_coeff_raw(
    #             signal_after,
    #             level=self.max_level,
    #             wavelet=self.wavelet,
    #             device=device,
    #         )
            
    #         freq_coeffs_before = wavelet_coeff_raw(
    #             signal_before,
    #             level=self.max_level,
    #             wavelet=self.wavelet,
    #             device=device,
    #         )

    #         print(f"\n{'='*120}")
    #         print(f"Batch {i+1}/{batch_size} - 详细小波系数统计")
    #         print(f"{'='*120}")
            
    #         # 打印After信号的系数
    #         print(f"\n【After信号小波系数】")
    #         print(f"{'-'*120}")
    #         print(f"{'系数':<8} {'长度':<8} {'均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12} "
    #               f"{'绝对值均值':<12} {'能量':<12}")
    #         print(f"{'-'*120}")
            
    #         for name, coeff_tensor in freq_coeffs_after.items():
    #             coeff = coeff_tensor.cpu().numpy().flatten()
    #             print(f"{name:<8} {len(coeff):<8} {np.mean(coeff):<12.6f} {np.std(coeff):<12.6f} "
    #                   f"{np.min(coeff):<12.6f} {np.max(coeff):<12.6f} {np.mean(np.abs(coeff)):<12.6f} "
    #                   f"{np.sum(coeff**2):<12.4e}")
            
    #         # 打印Before信号的系数
    #         print(f"\n【Before信号小波系数】")
    #         print(f"{'-'*120}")
    #         print(f"{'系数':<8} {'长度':<8} {'均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12} "
    #               f"{'绝对值均值':<12} {'能量':<12}")
    #         print(f"{'-'*120}")
            
    #         for name, coeff_tensor in freq_coeffs_before.items():
    #             coeff = coeff_tensor.cpu().numpy().flatten()
    #             print(f"{name:<8} {len(coeff):<8} {np.mean(coeff):<12.6f} {np.std(coeff):<12.6f} "
    #                   f"{np.min(coeff):<12.6f} {np.max(coeff):<12.6f} {np.mean(np.abs(coeff)):<12.6f} "
    #                   f"{np.sum(coeff**2):<12.4e}")
            
    #         # 打印差分系数
    #         print(f"\n【差分系数 (After - Before)】")
    #         print(f"{'-'*120}")
    #         print(f"{'系数':<8} {'长度':<8} {'均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12} "
    #               f"{'绝对值均值':<12} {'能量':<12}")
    #         print(f"{'-'*120}")
            
    #         for name in freq_coeffs_after.keys():
    #             coeff_after = freq_coeffs_after[name].cpu().numpy().flatten()
    #             coeff_before = freq_coeffs_before[name].cpu().numpy().flatten()
    #             delta_coeff = coeff_after - coeff_before
                
    #             print(f"{name:<8} {len(delta_coeff):<8} {np.mean(delta_coeff):<12.6f} {np.std(delta_coeff):<12.6f} "
    #                   f"{np.min(delta_coeff):<12.6f} {np.max(delta_coeff):<12.6f} "
    #                   f"{np.mean(np.abs(delta_coeff)):<12.6f} {np.sum(delta_coeff**2):<12.4e}")
            
    #         print(f"{'='*120}\n")

    # def visualize_attention(
    #     self, 
    #     x_after: torch.Tensor,
    #     x_before: torch.Tensor,
    #     save_path: str = "attention_delta.png"
    # ):
    #     """
    #     可视化 [CLS] token 对各个差分小波系数的注意力。
    #     """
    #     import matplotlib.pyplot as plt
    #     import numpy as np
    #     import pandas as pd
    #     import logging

    #     self.eval()
    #     with torch.no_grad():
    #         # Ensure x is on the correct device
    #         device = next(self.parameters()).device
    #         x_after = x_after.to(device)
    #         x_before = x_before.to(device)
            
    #         # 获取输出和注意力权重
    #         _, attn_weights, token_names = self.forward(x_after, x_before, return_attn=True)
            
    #         # attn_weights: [Batch, Seq_len+1, Seq_len+1] (averaged over heads)
    #         # 我们关注 [CLS] token (index 0) 对其他 tokens (index 1:) 的注意力
    #         # 取 batch 的平均值
    #         cls_attn = attn_weights[:, 0, 1:].mean(dim=0).cpu().numpy()
            
    #         # 绘图
    #         plt.figure(figsize=(12, 6))
    #         bars = plt.bar(token_names, cls_attn)
            
    #         plt.title("Average Attention of [CLS] Token to Delta Wavelet Coefficients")
    #         plt.xlabel("Delta Wavelet Coefficients (After - Before)")
    #         plt.ylabel("Attention Weight")
    #         plt.xticks(rotation=45)
            
    #         # 在柱状图上标数值
    #         for bar in bars:
    #             yval = bar.get_height()
    #             plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), ha='center', va='bottom')
            
    #         plt.tight_layout()
    #         plt.savefig(save_path)
    #         plt.close()
    #         logging.info(f"差分注意力图已保存到: {save_path}")
            
    #         # 保存为 CSV 文件
    #         csv_path = save_path.replace('.png', '.csv')
    #         df = pd.DataFrame({
    #             'token_name': token_names,
    #             'attention_weight': cls_attn
    #         })
    #         df.to_csv(csv_path, index=False)
    #         logging.info(f"差分注意力数据已保存到: {csv_path}")
            
    #         # 找出注意力最高的系数
    #         max_idx = np.argmax(cls_attn)
    #         logging.info(f"最高注意力系数: {token_names[max_idx]} (权重: {cls_attn[max_idx]:.4f})")


class DeltaBatteryModel(nn.Module):
    """
    差分电池模型 V9：
    1. 时域：分别对 after 和 before 进行 transformer 编码后再做差分
    2. 频域：在小波变换后直接做差分，然后再进行 transformer 编码
    3. 通过可学习的 alpha 加权混合时频差分特征
    4. 通过 MLP head 输出分类结果
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        ROPE_max_len: int = 10000,
        num_classes: int = 1,
        alpha: float = 0.0,
        dropout: float = 0.1,
        task_type: str = "classification",
        max_level: int = 6,
        wavelet: str = 'sym4',
        patch_size: int = 50,
        stride: int = 50,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.d_model = d_model
        
        # 可学习的时频融合权重
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        
        # 时域分支：使用因果mask的Transformer
        self.time_branch = TimeDomainBranch(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ROPE_max_len=ROPE_max_len,
            dropout=dropout,
            patch_size=patch_size,
            stride=stride,
        )
        
        # 频域分支：先差分再编码
        self.freq_branch = DeltaFrequencyDomainBranch(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            max_level=max_level,
            wavelet=wavelet,
            print_coeffs=True,  # 默认打印系数信息
        )

        # MLP head for classification/regression
        hidden_dim = max(d_model // 2, 1)
        
        if task_type == "classification":
            inter_dim = max(hidden_dim // 2, 1)
            self.output_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, inter_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(inter_dim, num_classes),
            )
        elif task_type == "regression":
            self.output_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")

        self._init_weights()

    def _init_weights(self) -> None:
        """初始化权重"""
        for module in self.output_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        signals_after: torch.Tensor,
        signals_before: torch.Tensor,
    ):
        """
        Args:
            signals_after: [batch, channels, seq_len] 运行后信号
            signals_before: [batch, channels, seq_len] 运行前信号
        
        Returns:
            outputs: 模型输出
            time_delta: 时域差分特征
            freq_delta: 频域差分特征
            fused_delta: 融合后的差分特征
        """
        # 时域分支：先编码再差分
        time_feat_after = self.time_branch(signals_after) 
        time_feat_before = self.time_branch(signals_before)
        time_delta = time_feat_after - time_feat_before
        
        # 频域分支：先差分再编码
        freq_delta = self.freq_branch(signals_after, signals_before)
        
        # Alpha加权融合
        alpha = torch.sigmoid(self.alpha)
        fused_delta = alpha * time_delta + (1 - alpha) * freq_delta
        
        # 通过输出头
        outputs = self.output_head(fused_delta)
        
        return outputs, time_delta, freq_delta, fused_delta

    @torch.no_grad()
    def get_delta_features(
        self,
        signals_after: torch.Tensor,
        signals_before: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        获取各层特征用于分析
        """
        outputs, time_delta, freq_delta, fused_delta = self.forward(
            signals_after, signals_before
        )
        
        alpha = torch.sigmoid(self.alpha)
        
        result = {
            "time_delta_features": time_delta,
            "freq_delta_features": freq_delta,
            "fused_delta_features": fused_delta,
            "alpha": alpha.item(),
            "outputs": outputs,
        }

        if self.task_type == "classification":
            if outputs.shape[1] == 1:
                result["probs"] = torch.sigmoid(outputs)
            else:
                result["probs"] = torch.softmax(outputs, dim=1)
        else:
            result["predictions"] = outputs
            
        return result

    # def visualize_frequency_attention(
    #     self,
    #     signals_after: torch.Tensor,
    #     signals_before: torch.Tensor,
    #     save_path: str = "freq_attention_delta.png"
    # ):
    #     """
    #     可视化频域分支的注意力
    #     """
    #     self.freq_branch.visualize_attention(signals_after, signals_before, save_path)


def create_battery_model_from_config(
    config: Optional[Dict[str, Any]] = None, **kwargs
) -> DeltaBatteryModel:
    """
    根据配置创建差分电池模型。
    """
    config = config or {}
    model_cfg = config.get("model", config)
    delta_cfg = config.get("delta_model", {})

    params = {
        "input_dim": model_cfg.get("input_dim", 1),
        "d_model": model_cfg.get("d_model", 64),
        "nhead": model_cfg.get("nhead", 4),
        "num_layers": model_cfg.get("num_layers", 2),
        "ROPE_max_len": model_cfg.get("ROPE_max_len", 5000),
        "num_classes": model_cfg.get("num_classes", 1),
        "alpha": model_cfg.get("alpha", 0.0),
        "dropout": model_cfg.get("dropout", 0.1),
        "task_type": model_cfg.get("task_type", "classification"),
        "max_level": model_cfg.get("max_level", 6),
        "wavelet": model_cfg.get("wavelet", "sym4"),
        "patch_size": model_cfg.get("patch_size", 50),
        "stride": model_cfg.get("stride", 50),
    }
    params.update(kwargs)
    
    # 打印模型配置
    print(f"\n{'='*80}")
    print("创建 DeltaBatteryModel")
    print(f"{'='*80}")
    print(f"模型参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print(f"{'='*80}\n")

    if params["d_model"] % params["nhead"] != 0:
        raise ValueError(
            f"d_model ({params['d_model']}) 必须能被 nhead ({params['nhead']}) 整除"
        )

    return DeltaBatteryModel(**params)


__all__ = [
    "TimeDomainBranch",
    "DeltaFrequencyDomainBranch",
    "DeltaBatteryModel",
    "create_battery_model_from_config",
]

