"""
Flow Matching 损失函数

实现条件流匹配 (Conditional Flow Matching) 的训练目标。

核心思想:
给定配对数据 (z_0, z_1)，沿着插值路径 z_t 训练速度场网络
预测正确的传输方向。

支持的路径类型:
1. Linear (OT): z_t = (1-t)*z_0 + t*z_1, 目标速度 = z_1 - z_0
2. VP (Variance Preserving): 保方差路径
3. Cosine: 余弦调度路径
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Callable, Literal
from dataclasses import dataclass
from enum import Enum
import math


class PathType(Enum):
    """插值路径类型"""
    LINEAR = "linear"  # 线性/最优传输路径
    VP = "vp"  # Variance Preserving
    COSINE = "cosine"  # 余弦调度
    SPHERICAL = "spherical"  # 球面插值 (SLERP)


@dataclass
class PathConfig:
    """路径配置"""
    path_type: PathType = PathType.LINEAR
    sigma_min: float = 1e-4  # VP/Cosine 的最小标准差
    sigma_max: float = 1.0  # VP/Cosine 的最大标准差


class OptimalTransportPath:
    """
    最优传输线性路径
    
    定义从 z_0 到 z_1 的线性插值:
    z_t = (1 - t) * z_0 + t * z_1
    
    对应的条件向量场:
    u_t(z | z_0, z_1) = z_1 - z_0
    
    这是最简单且理论上最优的路径选择（在欧几里得距离意义下）。
    """
    
    @staticmethod
    def interpolate(
        z_0: torch.Tensor,
        z_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算插值点 z_t
        
        Args:
            z_0: 起点 [batch, dim]
            z_1: 终点 [batch, dim]
            t: 时间 [batch] 或 [batch, 1]
        
        Returns:
            z_t: 插值点 [batch, dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return (1 - t) * z_0 + t * z_1
    
    @staticmethod
    def target_velocity(
        z_0: torch.Tensor,
        z_1: torch.Tensor,
        t: torch.Tensor,  # 未使用，但保持接口一致
    ) -> torch.Tensor:
        """
        计算目标速度
        
        对于线性路径，目标速度是常数:
        v = z_1 - z_0
        
        Args:
            z_0: 起点 [batch, dim]
            z_1: 终点 [batch, dim]
            t: 时间（未使用）
        
        Returns:
            velocity: 目标速度 [batch, dim]
        """
        return z_1 - z_0
    
    @staticmethod
    def sample_t(
        batch_size: int,
        device: torch.device,
        stratified: bool = False,
    ) -> torch.Tensor:
        """
        采样时间 t
        
        Args:
            batch_size: 批大小
            device: 设备
            stratified: 是否使用分层采样（减少方差）
        
        Returns:
            t: [batch_size]
        """
        if stratified:
            # 分层采样：将 [0,1] 分成 batch_size 个区间
            t = (torch.arange(batch_size, device=device) + torch.rand(batch_size, device=device)) / batch_size
        else:
            t = torch.rand(batch_size, device=device)
        
        return t


class VariancePreservingPath:
    """
    保方差路径 (VP Path)
    
    类似于扩散模型的前向过程:
    z_t = α_t * z_1 + σ_t * z_0
    
    其中 α_t² + σ_t² = 1（保持方差）
    
    常用调度: α_t = cos(π*t/2), σ_t = sin(π*t/2)
    """
    
    def __init__(self, sigma_min: float = 1e-4):
        self.sigma_min = sigma_min
    
    def get_alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 α_t 和 σ_t"""
        alpha_t = torch.cos(math.pi * t / 2)
        sigma_t = torch.sin(math.pi * t / 2)
        sigma_t = sigma_t.clamp(min=self.sigma_min)
        return alpha_t, sigma_t
    
    def interpolate(
        self,
        z_0: torch.Tensor,
        z_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """VP 插值"""
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        alpha_t, sigma_t = self.get_alpha_sigma(t)
        return alpha_t * z_1 + sigma_t * z_0
    
    def target_velocity(
        self,
        z_0: torch.Tensor,
        z_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """VP 目标速度"""
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # dz_t/dt = dα_t/dt * z_1 + dσ_t/dt * z_0
        d_alpha_dt = -math.pi / 2 * torch.sin(math.pi * t / 2)
        d_sigma_dt = math.pi / 2 * torch.cos(math.pi * t / 2)
        
        return d_alpha_dt * z_1 + d_sigma_dt * z_0


class SphericalPath:
    """
    球面线性插值 (SLERP)
    
    在单位球面上的测地线插值，适合归一化的潜空间。
    
    SLERP(z_0, z_1, t) = sin((1-t)*Ω)/sin(Ω) * z_0 + sin(t*Ω)/sin(Ω) * z_1
    其中 Ω = arccos(z_0 · z_1)
    """
    
    @staticmethod
    def interpolate(
        z_0: torch.Tensor,
        z_1: torch.Tensor,
        t: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """球面插值"""
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # 归一化
        z_0_norm = F.normalize(z_0, dim=-1)
        z_1_norm = F.normalize(z_1, dim=-1)
        
        # 计算夹角
        dot = (z_0_norm * z_1_norm).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps)
        omega = torch.acos(dot)
        
        # SLERP
        sin_omega = torch.sin(omega).clamp(min=eps)
        w_0 = torch.sin((1 - t) * omega) / sin_omega
        w_1 = torch.sin(t * omega) / sin_omega
        
        # 保持原始范数
        norm_0 = z_0.norm(dim=-1, keepdim=True)
        norm_1 = z_1.norm(dim=-1, keepdim=True)
        norm_t = (1 - t) * norm_0 + t * norm_1
        
        z_t = w_0 * z_0_norm + w_1 * z_1_norm
        return z_t * norm_t


class FlowMatchingLoss(nn.Module):
    """
    Flow Matching 训练损失
    
    实现条件流匹配的损失函数:
    L_FM = E_{t, z_0, z_1} [ || v_θ(z_t, t) - u_t(z_t | z_0, z_1) ||² ]
    
    Args:
        path_type: 插值路径类型
        sigma_min: 最小标准差
        reduction: 损失归约方式
        velocity_weighting: 速度加权方式
    """
    
    def __init__(
        self,
        path_type: str = "linear",
        sigma_min: float = 1e-4,
        reduction: Literal["mean", "sum", "none"] = "mean",
        velocity_weighting: Optional[str] = None,
    ):
        super().__init__()
        
        self.path_type = PathType(path_type) if isinstance(path_type, str) else path_type
        self.sigma_min = sigma_min
        self.reduction = reduction
        self.velocity_weighting = velocity_weighting
        
        # 初始化路径
        if self.path_type == PathType.LINEAR:
            self.path = OptimalTransportPath()
        elif self.path_type == PathType.VP:
            self.path = VariancePreservingPath(sigma_min)
        elif self.path_type == PathType.SPHERICAL:
            self.path = SphericalPath()
        else:
            self.path = OptimalTransportPath()
    
    def forward(
        self,
        velocity_net: nn.Module,
        z_0: torch.Tensor,
        z_1: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算 Flow Matching 损失
        
        Args:
            velocity_net: 速度场网络 v_θ
            z_0: 源分布样本 [batch, dim]
            z_1: 目标分布样本 [batch, dim]
            condition: 条件嵌入 [batch, cond_dim]，可选
            t: 时间采样 [batch]，可选（默认随机采样）
        
        Returns:
            损失字典，包含:
            - loss: 总损失
            - mse_loss: MSE 损失
            - velocity_norm: 速度范数（用于监控）
        """
        batch_size = z_0.shape[0]
        device = z_0.device
        
        # 采样时间 t
        if t is None:
            t = torch.rand(batch_size, device=device)
        
        # 计算插值点 z_t
        if self.path_type == PathType.LINEAR:
            z_t = self.path.interpolate(z_0, z_1, t)
            target_v = self.path.target_velocity(z_0, z_1, t)
        elif self.path_type == PathType.VP:
            z_t = self.path.interpolate(z_0, z_1, t)
            target_v = self.path.target_velocity(z_0, z_1, t)
        elif self.path_type == PathType.SPHERICAL:
            z_t = self.path.interpolate(z_0, z_1, t)
            # 球面路径的目标速度需要数值计算
            eps = 1e-4
            z_t_plus = self.path.interpolate(z_0, z_1, t + eps)
            target_v = (z_t_plus - z_t) / eps
        else:
            z_t = OptimalTransportPath.interpolate(z_0, z_1, t)
            target_v = OptimalTransportPath.target_velocity(z_0, z_1, t)
        
        # 预测速度
        pred_v = velocity_net(z_t, t, condition)
        
        # 计算损失
        mse_loss = F.mse_loss(pred_v, target_v, reduction="none")
        
        # 可选的速度加权
        if self.velocity_weighting == "target_norm":
            # 按目标速度范数加权（给高速区域更多权重）
            weight = target_v.norm(dim=-1, keepdim=True) + 1.0
            mse_loss = mse_loss * weight
        elif self.velocity_weighting == "time":
            # 时间加权（可以给某些时间段更多权重）
            weight = 1 + torch.sin(math.pi * t).unsqueeze(-1)  # 中间时刻权重更大
            mse_loss = mse_loss * weight
        
        # 归约
        if self.reduction == "mean":
            loss = mse_loss.mean()
        elif self.reduction == "sum":
            loss = mse_loss.sum()
        else:
            loss = mse_loss
        
        return {
            "loss": loss,
            "mse_loss": mse_loss.mean(),
            "pred_velocity_norm": pred_v.norm(dim=-1).mean(),
            "target_velocity_norm": target_v.norm(dim=-1).mean(),
        }
    
    def compute_with_noise(
        self,
        velocity_net: nn.Module,
        z_0: torch.Tensor,
        z_1: torch.Tensor,
        noise_scale: float = 0.01,
        condition: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        带噪声的 Flow Matching 损失（用于正则化）
        
        在 z_t 上添加小噪声，提高训练稳定性。
        """
        batch_size = z_0.shape[0]
        device = z_0.device
        
        t = torch.rand(batch_size, device=device)
        z_t = OptimalTransportPath.interpolate(z_0, z_1, t)
        
        # 添加噪声
        noise = torch.randn_like(z_t) * noise_scale
        z_t_noisy = z_t + noise
        
        target_v = OptimalTransportPath.target_velocity(z_0, z_1, t)
        pred_v = velocity_net(z_t_noisy, t, condition)
        
        loss = F.mse_loss(pred_v, target_v)
        
        return {"loss": loss, "mse_loss": loss}


class MiniBatchOTLoss(nn.Module):
    """
    Mini-batch 最优传输 Flow Matching 损失
    
    在 mini-batch 内部使用最优传输匹配 (z_0, z_1) 对，
    而不是随机配对。这可以减少方差，加速训练。
    
    适用场景：z_0 和 z_1 分别从两个分布独立采样（非配对数据）
    """
    
    def __init__(
        self,
        reduction: str = "mean",
        sinkhorn_iterations: int = 100,
        sinkhorn_epsilon: float = 0.01,
    ):
        super().__init__()
        self.reduction = reduction
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_epsilon = sinkhorn_epsilon
    
    def sinkhorn(
        self,
        cost_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sinkhorn 算法求解软最优传输
        
        Returns:
            传输计划矩阵 [batch, batch]
        """
        n = cost_matrix.shape[0]
        
        # 初始化
        K = torch.exp(-cost_matrix / self.sinkhorn_epsilon)
        u = torch.ones(n, device=cost_matrix.device)
        v = torch.ones(n, device=cost_matrix.device)
        
        # Sinkhorn 迭代
        for _ in range(self.sinkhorn_iterations):
            u = 1.0 / (K @ v + 1e-8)
            v = 1.0 / (K.T @ u + 1e-8)
        
        # 传输计划
        plan = torch.diag(u) @ K @ torch.diag(v)
        return plan
    
    def forward(
        self,
        velocity_net: nn.Module,
        z_0: torch.Tensor,
        z_1: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Mini-batch OT Flow Matching 损失
        """
        batch_size = z_0.shape[0]
        device = z_0.device
        
        # 计算成本矩阵（欧几里得距离）
        cost_matrix = torch.cdist(z_0, z_1, p=2)
        
        # 求解 OT
        plan = self.sinkhorn(cost_matrix)
        
        # 根据传输计划采样配对
        # 使用 Gumbel-Softmax 进行可微采样
        indices = torch.multinomial(plan, 1).squeeze(-1)
        z_1_matched = z_1[indices]
        
        # 使用标准 Flow Matching 损失
        t = torch.rand(batch_size, device=device)
        z_t = OptimalTransportPath.interpolate(z_0, z_1_matched, t)
        target_v = OptimalTransportPath.target_velocity(z_0, z_1_matched, t)
        pred_v = velocity_net(z_t, t, condition)
        
        loss = F.mse_loss(pred_v, target_v, reduction=self.reduction)
        
        return {
            "loss": loss,
            "mse_loss": loss,
            "ot_cost": (cost_matrix * plan).sum(),
        }
