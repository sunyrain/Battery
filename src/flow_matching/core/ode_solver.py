"""
ODE 求解器封装

提供用于 Flow Matching 推理的 ODE 求解功能，
支持多种数值积分方法和自适应步长控制。

支持的求解器:
- Euler: 最简单，一阶精度
- Midpoint: 二阶精度
- RK4: 经典四阶 Runge-Kutta
- Dopri5: 自适应步长 Dormand-Prince (推荐)
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple, List, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SolverType(Enum):
    """ODE 求解器类型"""
    EULER = "euler"
    MIDPOINT = "midpoint"
    RK4 = "rk4"
    DOPRI5 = "dopri5"
    ADAPTIVE_HEUN = "adaptive_heun"


@dataclass
class ODESolverConfig:
    """ODE 求解器配置"""
    solver_type: SolverType = SolverType.DOPRI5
    rtol: float = 1e-5  # 相对误差容限
    atol: float = 1e-5  # 绝对误差容限
    min_step: float = 1e-5  # 最小步长
    max_step: float = 0.5  # 最大步长
    max_num_steps: int = 1000  # 最大步数


class ODEFunction(nn.Module):
    """
    ODE 函数封装
    
    将速度场网络封装为 ODE 函数形式:
    dz/dt = v_θ(z, t, c)
    
    Args:
        velocity_fn: 速度场函数
        condition: 条件嵌入（固定）
    """
    
    def __init__(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        condition: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.velocity_fn = velocity_fn
        self.condition = condition
        self.nfe = 0  # Number of function evaluations
    
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 当前时间 (标量)
            z: 当前状态 [batch, latent_dim]
        
        Returns:
            dz/dt: 速度 [batch, latent_dim]
        """
        self.nfe += 1
        
        # 扩展 t 到 batch 维度
        batch_size = z.shape[0]
        t_batch = t.expand(batch_size)
        
        return self.velocity_fn(z, t_batch, self.condition)
    
    def reset_nfe(self):
        self.nfe = 0


class ODESolver:
    """
    ODE 求解器
    
    用于将初始状态 z_0 通过 ODE 积分演化到任意时刻。
    
    Args:
        config: 求解器配置
    """
    
    def __init__(self, config: Optional[ODESolverConfig] = None):
        self.config = config or ODESolverConfig()
    
    def solve(
        self,
        func: ODEFunction,
        z_0: torch.Tensor,
        t_span: torch.Tensor,
        return_trajectory: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        求解 ODE: dz/dt = func(t, z), z(t_0) = z_0
        
        Args:
            func: ODE 函数
            z_0: 初始状态 [batch, latent_dim]
            t_span: 时间点 [num_steps]，必须单调递增
            return_trajectory: 是否返回完整轨迹
        
        Returns:
            如果 return_trajectory=True:
                trajectory: [num_steps, batch, latent_dim]
            否则:
                z_T: [batch, latent_dim] 最终状态
        """
        func.reset_nfe()
        
        solver_type = self.config.solver_type
        
        if solver_type == SolverType.EULER:
            result = self._euler_solve(func, z_0, t_span)
        elif solver_type == SolverType.MIDPOINT:
            result = self._midpoint_solve(func, z_0, t_span)
        elif solver_type == SolverType.RK4:
            result = self._rk4_solve(func, z_0, t_span)
        elif solver_type == SolverType.DOPRI5:
            result = self._dopri5_solve(func, z_0, t_span)
        elif solver_type == SolverType.ADAPTIVE_HEUN:
            result = self._adaptive_heun_solve(func, z_0, t_span)
        else:
            raise ValueError(f"未知的求解器类型: {solver_type}")
        
        logger.debug(f"ODE 求解完成，函数评估次数: {func.nfe}")
        
        if return_trajectory:
            return result
        else:
            return result[-1]
    
    def _euler_solve(
        self,
        func: ODEFunction,
        z_0: torch.Tensor,
        t_span: torch.Tensor,
    ) -> torch.Tensor:
        """
        欧拉法 (一阶)
        
        z_{n+1} = z_n + h * f(t_n, z_n)
        """
        trajectory = [z_0]
        z = z_0
        
        for i in range(len(t_span) - 1):
            t = t_span[i]
            h = t_span[i + 1] - t_span[i]
            
            dz = func(t, z)
            z = z + h * dz
            
            trajectory.append(z)
        
        return torch.stack(trajectory, dim=0)
    
    def _midpoint_solve(
        self,
        func: ODEFunction,
        z_0: torch.Tensor,
        t_span: torch.Tensor,
    ) -> torch.Tensor:
        """
        中点法 (二阶)
        
        k1 = f(t_n, z_n)
        k2 = f(t_n + h/2, z_n + h/2 * k1)
        z_{n+1} = z_n + h * k2
        """
        trajectory = [z_0]
        z = z_0
        
        for i in range(len(t_span) - 1):
            t = t_span[i]
            h = t_span[i + 1] - t_span[i]
            
            k1 = func(t, z)
            k2 = func(t + h / 2, z + h / 2 * k1)
            z = z + h * k2
            
            trajectory.append(z)
        
        return torch.stack(trajectory, dim=0)
    
    def _rk4_solve(
        self,
        func: ODEFunction,
        z_0: torch.Tensor,
        t_span: torch.Tensor,
    ) -> torch.Tensor:
        """
        经典四阶 Runge-Kutta
        
        k1 = f(t_n, z_n)
        k2 = f(t_n + h/2, z_n + h/2 * k1)
        k3 = f(t_n + h/2, z_n + h/2 * k2)
        k4 = f(t_n + h, z_n + h * k3)
        z_{n+1} = z_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        """
        trajectory = [z_0]
        z = z_0
        
        for i in range(len(t_span) - 1):
            t = t_span[i]
            h = t_span[i + 1] - t_span[i]
            
            k1 = func(t, z)
            k2 = func(t + h / 2, z + h / 2 * k1)
            k3 = func(t + h / 2, z + h / 2 * k2)
            k4 = func(t + h, z + h * k3)
            
            z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            
            trajectory.append(z)
        
        return torch.stack(trajectory, dim=0)
    
    def _dopri5_solve(
        self,
        func: ODEFunction,
        z_0: torch.Tensor,
        t_span: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dormand-Prince 5(4) 自适应步长方法
        
        使用 5 阶公式推进，4 阶公式估计误差。
        """
        # Butcher tableau for Dopri5
        c = torch.tensor([0, 1/5, 3/10, 4/5, 8/9, 1, 1], device=z_0.device)
        a = [
            [],
            [1/5],
            [3/40, 9/40],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84],
        ]
        b = torch.tensor([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0], device=z_0.device)
        b_star = torch.tensor([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40], device=z_0.device)
        
        trajectory = [z_0]
        z = z_0
        
        t_current = t_span[0]
        t_end = t_span[-1]
        t_eval_idx = 1
        
        h = (t_span[1] - t_span[0]) if len(t_span) > 1 else (t_end - t_current) / 10
        h = min(h, self.config.max_step)
        
        step_count = 0
        
        while t_current < t_end and step_count < self.config.max_num_steps:
            # 确保不超过终点
            h = min(h, t_end - t_current)
            h = max(h, self.config.min_step)
            
            # 计算 k 值
            k = [func(t_current, z)]
            
            for i in range(1, 7):
                t_i = t_current + c[i] * h
                z_i = z.clone()
                for j, a_ij in enumerate(a[i]):
                    z_i = z_i + h * a_ij * k[j]
                k.append(func(t_i, z_i))
            
            # 5 阶解
            z_new = z.clone()
            for i, b_i in enumerate(b):
                z_new = z_new + h * b_i * k[i]
            
            # 4 阶解（用于误差估计）
            z_star = z.clone()
            for i, b_i in enumerate(b_star):
                z_star = z_star + h * b_i * k[i]
            
            # 误差估计
            error = torch.abs(z_new - z_star).max().item()
            tolerance = self.config.atol + self.config.rtol * torch.abs(z_new).max().item()
            
            if error <= tolerance or h <= self.config.min_step:
                # 接受步长
                z = z_new
                t_current = t_current + h
                step_count += 1
                
                # 插值到评估点
                while t_eval_idx < len(t_span) and t_span[t_eval_idx] <= t_current:
                    # 简单线性插值
                    alpha = (t_span[t_eval_idx] - (t_current - h)) / h
                    z_interp = (1 - alpha) * trajectory[-1] + alpha * z
                    trajectory.append(z_interp)
                    t_eval_idx += 1
                
                # 调整步长（增大）
                if error > 0:
                    h = h * min(2.0, max(0.5, 0.9 * (tolerance / error) ** 0.2))
            else:
                # 拒绝步长，减小
                h = h * max(0.1, 0.9 * (tolerance / error) ** 0.25)
            
            h = min(h, self.config.max_step)
        
        # 确保最后一个点
        if len(trajectory) < len(t_span):
            trajectory.append(z)
        
        return torch.stack(trajectory[:len(t_span)], dim=0)
    
    def _adaptive_heun_solve(
        self,
        func: ODEFunction,
        z_0: torch.Tensor,
        t_span: torch.Tensor,
    ) -> torch.Tensor:
        """
        自适应 Heun 方法 (二阶自适应)
        
        比 Dopri5 简单，但仍支持自适应步长。
        """
        trajectory = [z_0]
        z = z_0
        
        t_current = t_span[0]
        t_end = t_span[-1]
        t_eval_idx = 1
        
        h = (t_span[1] - t_span[0]) if len(t_span) > 1 else (t_end - t_current) / 10
        
        step_count = 0
        
        while t_current < t_end and step_count < self.config.max_num_steps:
            h = min(h, t_end - t_current, self.config.max_step)
            h = max(h, self.config.min_step)
            
            # Heun 方法
            k1 = func(t_current, z)
            z_euler = z + h * k1
            k2 = func(t_current + h, z_euler)
            z_heun = z + h * 0.5 * (k1 + k2)
            
            # 误差估计
            error = torch.abs(z_heun - z_euler).max().item() / 2
            tolerance = self.config.atol + self.config.rtol * torch.abs(z_heun).max().item()
            
            if error <= tolerance or h <= self.config.min_step:
                z = z_heun
                t_current = t_current + h
                step_count += 1
                
                # 插值到评估点
                while t_eval_idx < len(t_span) and t_span[t_eval_idx] <= t_current:
                    alpha = (t_span[t_eval_idx] - (t_current - h)) / h
                    z_interp = (1 - alpha) * trajectory[-1] + alpha * z
                    trajectory.append(z_interp)
                    t_eval_idx += 1
                
                # 调整步长
                if error > 0:
                    h = h * min(2.0, 0.9 * (tolerance / error) ** 0.5)
            else:
                h = h * max(0.1, 0.9 * (tolerance / error) ** 0.5)
        
        if len(trajectory) < len(t_span):
            trajectory.append(z)
        
        return torch.stack(trajectory[:len(t_span)], dim=0)


def create_solver(solver_type: str = "dopri5", **kwargs) -> ODESolver:
    """
    创建 ODE 求解器的便捷函数
    
    Args:
        solver_type: 求解器类型 ("euler", "midpoint", "rk4", "dopri5", "adaptive_heun")
        **kwargs: 其他配置参数
    
    Returns:
        ODESolver 实例
    """
    solver_map = {
        "euler": SolverType.EULER,
        "midpoint": SolverType.MIDPOINT,
        "rk4": SolverType.RK4,
        "dopri5": SolverType.DOPRI5,
        "adaptive_heun": SolverType.ADAPTIVE_HEUN,
    }
    
    if solver_type not in solver_map:
        raise ValueError(f"未知的求解器类型: {solver_type}, 可选: {list(solver_map.keys())}")
    
    config = ODESolverConfig(solver_type=solver_map[solver_type], **kwargs)
    return ODESolver(config)
