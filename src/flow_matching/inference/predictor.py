"""
生命周期预测器

提供基于 Flow Matching 的电池生命周期预测功能。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import logging

from ..models.flow_model import BatteryFlowModel
from ..core.ode_solver import ODESolver, ODEFunction, create_solver

logger = logging.getLogger(__name__)


class LifecyclePredictor:
    """
    电池生命周期预测器
    
    基于 Flow Matching 模型，从初始状态预测电池的完整生命周期演化。
    
    功能:
    1. 编码初始超声信号到潜空间
    2. 通过 ODE 积分预测潜空间轨迹
    3. 从潜空间计算健康评分
    4. 预测剩余使用寿命 (RUL)
    
    Args:
        model: 训练好的 BatteryFlowModel
        device: 计算设备
        solver_type: ODE 求解器类型
    """
    
    def __init__(
        self,
        model: BatteryFlowModel,
        device: Optional[torch.device] = None,
        solver_type: str = "dopri5",
    ):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        self.solver = create_solver(solver_type)
        
        logger.info(f"LifecyclePredictor 初始化完成，设备: {self.device}")
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        encoder: nn.Module,
        device: Optional[torch.device] = None,
        **model_kwargs,
    ) -> 'LifecyclePredictor':
        """
        从检查点加载预测器
        
        Args:
            checkpoint_path: 检查点路径
            encoder: SmartWave Encoder
            device: 设备
            **model_kwargs: 模型配置参数
        
        Returns:
            LifecyclePredictor 实例
        """
        from ..models.flow_model import BatteryFlowModel, BatteryFlowConfig
        
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 创建模型
        config = BatteryFlowConfig(**model_kwargs)
        model = BatteryFlowModel(config, encoder)
        
        # 加载权重
        if 'ema_state_dict' in checkpoint:
            # 优先使用 EMA 权重
            for name, param in model.named_parameters():
                if name in checkpoint['ema_state_dict']:
                    param.data = checkpoint['ema_state_dict'][name]
            logger.info("使用 EMA 权重")
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        return cls(model, device)
    
    @torch.no_grad()
    def encode(
        self,
        signal_after: torch.Tensor,
        signal_before: torch.Tensor,
    ) -> torch.Tensor:
        """
        编码超声信号到潜空间
        
        Args:
            signal_after: [batch, 1, seq_len] 运行后信号
            signal_before: [batch, 1, seq_len] 运行前信号
        
        Returns:
            z: [batch, latent_dim] 潜空间表示
        """
        signal_after = signal_after.to(self.device)
        signal_before = signal_before.to(self.device)
        
        return self.model.encode(signal_after, signal_before)
    
    @torch.no_grad()
    def predict_trajectory(
        self,
        z_0: torch.Tensor,
        num_steps: int = 100,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测潜空间轨迹
        
        Args:
            z_0: [batch, latent_dim] 初始潜空间状态
            num_steps: 时间步数
            t_start: 起始时间
            t_end: 结束时间
        
        Returns:
            trajectory: [num_steps, batch, latent_dim] 轨迹
            t_span: [num_steps] 时间点
        """
        z_0 = z_0.to(self.device)
        t_span = torch.linspace(t_start, t_end, num_steps, device=self.device)
        
        trajectory = self.model.predict_trajectory(z_0, t_span)
        
        return trajectory, t_span
    
    @torch.no_grad()
    def predict_health_scores(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """
        从轨迹计算健康评分
        
        Args:
            trajectory: [num_steps, batch, latent_dim]
        
        Returns:
            scores: [num_steps, batch] 健康评分 (0-1, 0=健康, 1=退化)
        """
        scores = []
        for z_t in trajectory:
            score = self.model.health_head(z_t).squeeze(-1)
            scores.append(score)
        
        return torch.stack(scores, dim=0)
    
    @torch.no_grad()
    def predict_full_lifecycle(
        self,
        signal_after: torch.Tensor,
        signal_before: torch.Tensor,
        num_steps: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """
        预测完整生命周期
        
        Args:
            signal_after: [batch, 1, seq_len]
            signal_before: [batch, 1, seq_len]
            num_steps: 时间步数
        
        Returns:
            字典包含:
            - z_0: 初始潜空间
            - trajectory: 潜空间轨迹
            - health_scores: 健康评分轨迹
            - cycles: 对应的 cycle 数
            - t_span: 归一化时间
        """
        # 编码
        z_0 = self.encode(signal_after, signal_before)
        
        # 预测轨迹
        trajectory, t_span = self.predict_trajectory(z_0, num_steps)
        
        # 计算健康评分
        health_scores = self.predict_health_scores(trajectory)
        
        # 转换为 cycle
        cycles = self.model.time_to_cycle(t_span)
        
        return {
            'z_0': z_0.cpu(),
            'trajectory': trajectory.cpu(),
            'health_scores': health_scores.cpu(),
            'cycles': cycles.cpu(),
            't_span': t_span.cpu(),
        }
    
    @torch.no_grad()
    def predict_rul(
        self,
        signal_after: torch.Tensor,
        signal_before: torch.Tensor,
        current_cycle: int,
        failure_threshold: float = 0.8,
        num_steps: int = 200,
    ) -> Dict[str, Union[int, float, torch.Tensor]]:
        """
        预测剩余使用寿命 (RUL)
        
        Args:
            signal_after: 当前超声信号 (after)
            signal_before: 当前超声信号 (before)
            current_cycle: 当前 cycle 数
            failure_threshold: 失效阈值 (健康评分达到此值视为失效)
            num_steps: 预测步数
        
        Returns:
            字典包含:
            - rul: 预测的剩余 cycle 数
            - failure_cycle: 预测的失效 cycle
            - confidence: 预测置信度
            - trajectory: 预测轨迹
        """
        # 计算当前时间
        t_current = self.model.cycle_to_time(torch.tensor(current_cycle))
        
        # 编码
        z_0 = self.encode(signal_after, signal_before)
        
        # 预测从当前到结束的轨迹
        trajectory, t_span = self.predict_trajectory(
            z_0, num_steps, 
            t_start=t_current.item(), 
            t_end=1.0
        )
        
        # 计算健康评分
        health_scores = self.predict_health_scores(trajectory)
        cycles = self.model.time_to_cycle(t_span)
        
        # 找到首次超过阈值的点
        scores_np = health_scores.squeeze().cpu().numpy()
        cycles_np = cycles.cpu().numpy()
        
        failure_indices = np.where(scores_np >= failure_threshold)[0]
        
        if len(failure_indices) > 0:
            failure_idx = failure_indices[0]
            failure_cycle = int(cycles_np[failure_idx])
            rul = failure_cycle - current_cycle
            
            # 计算置信度（基于评分变化的平滑度）
            score_diff = np.diff(scores_np)
            confidence = 1.0 - np.std(score_diff) / (np.mean(np.abs(score_diff)) + 1e-6)
            confidence = float(np.clip(confidence, 0, 1))
        else:
            # 未到达阈值
            failure_cycle = int(self.model.max_cycle)
            rul = failure_cycle - current_cycle
            confidence = 0.5  # 不确定
        
        return {
            'rul': rul,
            'failure_cycle': failure_cycle,
            'current_cycle': current_cycle,
            'confidence': confidence,
            'health_scores': health_scores.cpu(),
            'cycles': cycles.cpu(),
            'threshold': failure_threshold,
        }
    
    @torch.no_grad()
    def compare_trajectories(
        self,
        z_0_list: List[torch.Tensor],
        labels: Optional[List[str]] = None,
        num_steps: int = 100,
    ) -> Dict[str, List]:
        """
        比较多个初始状态的轨迹
        
        Args:
            z_0_list: 初始潜空间列表
            labels: 标签列表
            num_steps: 时间步数
        
        Returns:
            比较结果字典
        """
        if labels is None:
            labels = [f"Sample {i}" for i in range(len(z_0_list))]
        
        results = {
            'labels': labels,
            'trajectories': [],
            'health_scores': [],
        }
        
        t_span = torch.linspace(0, 1, num_steps, device=self.device)
        
        for z_0 in z_0_list:
            z_0 = z_0.to(self.device)
            if z_0.dim() == 1:
                z_0 = z_0.unsqueeze(0)
            
            trajectory = self.model.predict_trajectory(z_0, t_span)
            health_scores = self.predict_health_scores(trajectory)
            
            results['trajectories'].append(trajectory.cpu())
            results['health_scores'].append(health_scores.squeeze().cpu())
        
        results['t_span'] = t_span.cpu()
        results['cycles'] = self.model.time_to_cycle(t_span).cpu()
        
        return results
    
    @torch.no_grad()
    def interpolate_states(
        self,
        z_0: torch.Tensor,
        z_1: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        在两个潜空间状态之间进行插值
        
        使用 Flow Matching 学到的路径进行插值，
        而不是简单的线性插值。
        
        Args:
            z_0: 起始状态
            z_1: 结束状态
            num_steps: 插值步数
        
        Returns:
            interpolations: [num_steps, latent_dim]
        """
        z_0 = z_0.to(self.device)
        z_1 = z_1.to(self.device)
        
        if z_0.dim() == 1:
            z_0 = z_0.unsqueeze(0)
        if z_1.dim() == 1:
            z_1 = z_1.unsqueeze(0)
        
        # 使用 OT 路径插值
        t_values = torch.linspace(0, 1, num_steps, device=self.device)
        
        interpolations = []
        for t in t_values:
            z_t = (1 - t) * z_0 + t * z_1
            interpolations.append(z_t)
        
        return torch.cat(interpolations, dim=0)
