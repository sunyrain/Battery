"""
Battery Flow Model - 电池潜空间流匹配主模型

将 SmartWave Encoder 与 Flow Matching 结合，
实现电池全生命周期潜空间演化预测。

核心功能:
1. 编码: 将超声信号编码到潜空间
2. 流传输: 学习从健康到退化的潜空间演化
3. 预测: 给定初始状态，预测任意时刻的潜空间表示
4. 解码: (可选) 从潜空间还原信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any, Union, List
from dataclasses import dataclass
import logging

from .velocity_net import VelocityNetwork, LightweightVelocityNetwork
from .embeddings import ConditionEmbedding
from ..core.ode_solver import ODESolver, ODEFunction, create_solver
from ..core.flow_matching_loss import FlowMatchingLoss, OptimalTransportPath

logger = logging.getLogger(__name__)


@dataclass
class BatteryFlowConfig:
    """Battery Flow Model 配置"""
    
    # 潜空间维度 (需与 SmartWave Encoder 输出一致)
    latent_dim: int = 128
    
    # 速度场网络配置
    hidden_dim: int = 512
    time_embed_dim: int = 128
    cond_embed_dim: int = 128
    num_layers: int = 6
    dropout: float = 0.1
    use_adaln: bool = True
    use_skip_connections: bool = True
    
    # 条件配置
    num_battery_types: int = 10
    num_c_rates: int = 20
    continuous_cond_dim: int = 2
    
    # ODE 求解器配置
    solver_type: str = "dopri5"
    solver_rtol: float = 1e-5
    solver_atol: float = 1e-5
    
    # 训练配置
    path_type: str = "linear"  # "linear", "vp", "spherical"
    
    # 最大 cycle 数 (用于时间归一化)
    max_cycle: int = 200
    
    # 是否使用轻量级网络
    lightweight: bool = False


class BatteryFlowModel(nn.Module):
    """
    电池流匹配模型
    
    结合预训练的 SmartWave Encoder 和 Flow Matching，
    学习电池从健康到退化的潜空间演化轨迹。
    
    工作流程:
    1. Encoder 将超声信号 (after, before) 编码到潜空间 z
    2. Flow Network 学习 z 随 cycle 演化的速度场
    3. 推理时，通过 ODE 积分预测任意 cycle 的潜空间状态
    
    Args:
        config: 模型配置
        encoder: 预训练的 SmartWave Encoder (可选，可后续设置)
    """
    
    def __init__(
        self,
        config: BatteryFlowConfig,
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.config = config
        self.latent_dim = config.latent_dim
        self.max_cycle = config.max_cycle
        
        # ============ Encoder (来自 SmartWave) ============
        self.encoder = encoder
        self._encoder_frozen = False
        
        # ============ 条件嵌入 ============
        if config.cond_embed_dim > 0:
            self.condition_embedding = ConditionEmbedding(
                embed_dim=config.cond_embed_dim,
                num_battery_types=config.num_battery_types,
                num_c_rates=config.num_c_rates,
                continuous_dim=config.continuous_cond_dim,
            )
        else:
            self.condition_embedding = None
        
        # ============ 速度场网络 ============
        if config.lightweight:
            self.velocity_net = LightweightVelocityNetwork(
                latent_dim=config.latent_dim,
                hidden_dim=config.hidden_dim,
                time_embed_dim=config.time_embed_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
            )
        else:
            self.velocity_net = VelocityNetwork(
                latent_dim=config.latent_dim,
                hidden_dim=config.hidden_dim,
                time_embed_dim=config.time_embed_dim,
                cond_embed_dim=config.cond_embed_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
                use_adaln=config.use_adaln,
                use_skip_connections=config.use_skip_connections,
            )
        
        # ============ ODE 求解器 ============
        self.solver = create_solver(
            config.solver_type,
            rtol=config.solver_rtol,
            atol=config.solver_atol,
        )
        
        # ============ 损失函数 ============
        self.flow_loss = FlowMatchingLoss(path_type=config.path_type)
        
        # ============ 健康评分头 (可选) ============
        self.health_head = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        logger.info(f"BatteryFlowModel 初始化完成")
        logger.info(f"  - 潜空间维度: {config.latent_dim}")
        logger.info(f"  - 速度场网络参数量: {self.velocity_net.get_num_params():,}")
        logger.info(f"  - ODE 求解器: {config.solver_type}")
    
    def set_encoder(self, encoder: nn.Module, freeze: bool = True):
        """
        设置 Encoder 并可选冻结
        
        Args:
            encoder: SmartWave Encoder
            freeze: 是否冻结 Encoder 参数
        """
        self.encoder = encoder
        
        if freeze:
            self.freeze_encoder()
        
        logger.info(f"Encoder 已设置, 冻结状态: {freeze}")
    
    def freeze_encoder(self):
        """冻结 Encoder 参数"""
        if self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            self._encoder_frozen = True
    
    def unfreeze_encoder(self):
        """解冻 Encoder 参数"""
        if self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = True
            self._encoder_frozen = False
    
    def cycle_to_time(self, cycle: torch.Tensor) -> torch.Tensor:
        """
        将 cycle 数转换为归一化时间 t ∈ [0, 1]
        
        t = (cycle - 1) / (max_cycle - 1)
        """
        return (cycle - 1).float() / (self.max_cycle - 1)
    
    def time_to_cycle(self, t: torch.Tensor) -> torch.Tensor:
        """
        将归一化时间转换回 cycle 数
        """
        return t * (self.max_cycle - 1) + 1
    
    def encode(
        self,
        signal_after: torch.Tensor,
        signal_before: torch.Tensor,
    ) -> torch.Tensor:
        """
        将超声信号编码到潜空间
        
        Args:
            signal_after: [batch, channels, seq_len] 运行后信号
            signal_before: [batch, channels, seq_len] 运行前信号
        
        Returns:
            z: [batch, latent_dim] 潜空间表示
        """
        if self.encoder is None:
            raise RuntimeError("Encoder 未设置，请先调用 set_encoder()")
        
        # 如果 encoder 已冻结，不计算梯度
        if self._encoder_frozen:
            with torch.no_grad():
                # SmartWave Encoder 返回 (outputs, time_delta, freq_delta, fused_delta)
                # 我们使用 fused_delta 作为潜空间表示
                _, _, _, z = self.encoder(signal_after, signal_before)
        else:
            _, _, _, z = self.encoder(signal_after, signal_before)
        
        return z
    
    def get_condition_embedding(
        self,
        battery_type: Optional[torch.Tensor] = None,
        c_rate: Optional[torch.Tensor] = None,
        continuous_cond: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        获取条件嵌入
        
        Args:
            battery_type: 电池类型索引 [batch]
            c_rate: 充放电倍率索引 [batch]
            continuous_cond: 连续条件 [batch, dim]
        
        Returns:
            condition: [batch, cond_embed_dim] 或 None
        """
        if self.condition_embedding is None:
            return None
        
        return self.condition_embedding(battery_type, c_rate, continuous_cond)
    
    def velocity(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        预测速度场
        
        Args:
            z_t: [batch, latent_dim] 当前潜空间点
            t: [batch] 当前时间
            condition: [batch, cond_embed_dim] 条件嵌入
        
        Returns:
            v: [batch, latent_dim] 速度向量
        """
        return self.velocity_net(z_t, t, condition)
    
    def compute_loss(
        self,
        z_0: torch.Tensor,
        z_1: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算 Flow Matching 损失
        
        Args:
            z_0: [batch, latent_dim] 早期 cycle 的潜空间表示
            z_1: [batch, latent_dim] 晚期 cycle 的潜空间表示
            condition: [batch, cond_embed_dim] 条件嵌入
        
        Returns:
            损失字典
        """
        return self.flow_loss(self.velocity_net, z_0, z_1, condition)
    
    def predict_trajectory(
        self,
        z_0: torch.Tensor,
        t_span: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        预测潜空间轨迹
        
        从初始状态 z_0 出发，通过 ODE 积分预测完整的演化轨迹。
        
        Args:
            z_0: [batch, latent_dim] 初始潜空间状态
            t_span: [num_steps] 时间点序列 (归一化到 [0, 1])
            condition: [batch, cond_embed_dim] 条件嵌入
        
        Returns:
            trajectory: [num_steps, batch, latent_dim] 潜空间轨迹
        """
        # 创建 ODE 函数
        ode_func = ODEFunction(
            velocity_fn=self.velocity_net,
            condition=condition,
        )
        
        # 求解 ODE
        with torch.no_grad():
            trajectory = self.solver.solve(
                func=ode_func,
                z_0=z_0,
                t_span=t_span,
                return_trajectory=True,
            )
        
        return trajectory
    
    def predict_at_cycle(
        self,
        z_0: torch.Tensor,
        cycle_0: Union[int, torch.Tensor],
        target_cycles: Union[List[int], torch.Tensor],
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        预测指定 cycle 的潜空间状态
        
        Args:
            z_0: [batch, latent_dim] 初始状态
            cycle_0: 初始 cycle 数
            target_cycles: 目标 cycle 列表
            condition: 条件嵌入
        
        Returns:
            z_targets: [num_targets, batch, latent_dim] 目标 cycle 的潜空间状态
        """
        # 转换为时间
        if isinstance(cycle_0, int):
            t_0 = torch.tensor(self.cycle_to_time(torch.tensor(cycle_0)), device=z_0.device)
        else:
            t_0 = self.cycle_to_time(cycle_0)
        
        if isinstance(target_cycles, list):
            target_cycles = torch.tensor(target_cycles, device=z_0.device)
        t_targets = self.cycle_to_time(target_cycles)
        
        # 构建时间序列
        t_span = torch.cat([t_0.unsqueeze(0), t_targets])
        t_span = torch.sort(t_span)[0]
        
        # 预测轨迹
        trajectory = self.predict_trajectory(z_0, t_span, condition)
        
        # 提取目标时间点
        target_indices = []
        for t_target in t_targets:
            idx = (t_span - t_target).abs().argmin()
            target_indices.append(idx)
        
        return trajectory[target_indices]
    
    def predict_full_lifecycle(
        self,
        signal_after: torch.Tensor,
        signal_before: torch.Tensor,
        num_steps: int = 100,
        condition: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        预测完整生命周期
        
        给定初始状态的超声信号，预测从 cycle 1 到 max_cycle 的完整轨迹。
        
        Args:
            signal_after: [batch, channels, seq_len] 初始信号 (after)
            signal_before: [batch, channels, seq_len] 初始信号 (before)
            num_steps: 轨迹采样点数
            condition: 条件嵌入
        
        Returns:
            字典包含:
            - trajectory: [num_steps, batch, latent_dim] 潜空间轨迹
            - health_scores: [num_steps, batch] 健康评分轨迹
            - cycles: [num_steps] 对应的 cycle 数
        """
        # 编码初始状态
        z_0 = self.encode(signal_after, signal_before)
        
        # 生成时间序列
        t_span = torch.linspace(0, 1, num_steps, device=z_0.device)
        
        # 预测轨迹
        trajectory = self.predict_trajectory(z_0, t_span, condition)
        
        # 计算健康评分
        health_scores = []
        for z_t in trajectory:
            score = self.health_head(z_t).squeeze(-1)
            health_scores.append(score)
        health_scores = torch.stack(health_scores, dim=0)
        
        # 转换回 cycle
        cycles = self.time_to_cycle(t_span)
        
        return {
            "trajectory": trajectory,
            "health_scores": health_scores,
            "cycles": cycles,
        }
    
    def forward(
        self,
        signal_after_0: torch.Tensor,
        signal_before_0: torch.Tensor,
        signal_after_1: torch.Tensor,
        signal_before_1: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        训练时的前向传播
        
        Args:
            signal_after_0, signal_before_0: 早期 cycle 的信号对
            signal_after_1, signal_before_1: 晚期 cycle 的信号对
            condition: 条件嵌入
        
        Returns:
            损失字典
        """
        # 编码
        z_0 = self.encode(signal_after_0, signal_before_0)
        z_1 = self.encode(signal_after_1, signal_before_1)
        
        # 计算 Flow Matching 损失
        loss_dict = self.compute_loss(z_0, z_1, condition)
        
        return loss_dict
    
    def get_num_trainable_params(self) -> int:
        """返回可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self) -> int:
        """返回总参数数量"""
        return sum(p.numel() for p in self.parameters())


def create_battery_flow_model(
    latent_dim: int = 128,
    encoder: Optional[nn.Module] = None,
    freeze_encoder: bool = True,
    **kwargs,
) -> BatteryFlowModel:
    """
    创建 Battery Flow Model 的便捷函数
    
    Args:
        latent_dim: 潜空间维度
        encoder: SmartWave Encoder
        freeze_encoder: 是否冻结 Encoder
        **kwargs: 其他配置参数
    
    Returns:
        BatteryFlowModel 实例
    """
    config = BatteryFlowConfig(latent_dim=latent_dim, **kwargs)
    model = BatteryFlowModel(config, encoder)
    
    if encoder is not None and freeze_encoder:
        model.freeze_encoder()
    
    return model
