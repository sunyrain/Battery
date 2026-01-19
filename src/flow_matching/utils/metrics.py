"""
评估指标模块
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple


def compute_trajectory_mse(
    pred_trajectory: torch.Tensor,
    true_trajectory: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    计算轨迹 MSE
    
    Args:
        pred_trajectory: [T, B, D] 预测轨迹
        true_trajectory: [T, B, D] 真实轨迹
        reduction: 归约方式
    
    Returns:
        MSE 损失
    """
    mse = (pred_trajectory - true_trajectory) ** 2
    
    if reduction == "mean":
        return mse.mean()
    elif reduction == "sum":
        return mse.sum()
    elif reduction == "none":
        return mse
    else:
        raise ValueError(f"未知的 reduction: {reduction}")


def compute_wasserstein_distance(
    source: torch.Tensor,
    target: torch.Tensor,
    p: int = 2,
) -> torch.Tensor:
    """
    计算 Wasserstein 距离 (使用 Sinkhorn 近似)
    
    Args:
        source: [N, D] 源分布样本
        target: [M, D] 目标分布样本
        p: 距离阶数
    
    Returns:
        W_p 距离
    """
    from ..core.optimal_transport import wasserstein_distance
    return wasserstein_distance(source, target, p)


def compute_health_score_accuracy(
    pred_scores: torch.Tensor,
    true_scores: torch.Tensor,
    threshold: float = 0.1,
) -> Dict[str, float]:
    """
    计算健康评分预测的准确性指标
    
    Args:
        pred_scores: 预测的健康评分
        true_scores: 真实的健康评分
        threshold: 误差阈值
    
    Returns:
        指标字典
    """
    if isinstance(pred_scores, torch.Tensor):
        pred_scores = pred_scores.cpu().numpy()
    if isinstance(true_scores, torch.Tensor):
        true_scores = true_scores.cpu().numpy()
    
    # MAE
    mae = np.abs(pred_scores - true_scores).mean()
    
    # RMSE
    rmse = np.sqrt(((pred_scores - true_scores) ** 2).mean())
    
    # 在阈值内的比例
    within_threshold = (np.abs(pred_scores - true_scores) < threshold).mean()
    
    # 相关系数
    if len(pred_scores) > 1:
        correlation = np.corrcoef(pred_scores.flatten(), true_scores.flatten())[0, 1]
    else:
        correlation = 0.0
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'within_threshold': float(within_threshold),
        'correlation': float(correlation),
    }


def compute_rul_metrics(
    pred_rul: np.ndarray,
    true_rul: np.ndarray,
) -> Dict[str, float]:
    """
    计算 RUL 预测指标
    
    Args:
        pred_rul: 预测的 RUL
        true_rul: 真实的 RUL
    
    Returns:
        指标字典
    """
    # MAE
    mae = np.abs(pred_rul - true_rul).mean()
    
    # RMSE
    rmse = np.sqrt(((pred_rul - true_rul) ** 2).mean())
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.abs((pred_rul - true_rul) / (true_rul + 1e-8)).mean() * 100
    
    # 准时预测率 (Early/Late/On-time)
    early = (pred_rul < true_rul * 0.9).mean()
    late = (pred_rul > true_rul * 1.1).mean()
    on_time = 1 - early - late
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'early_rate': float(early),
        'late_rate': float(late),
        'on_time_rate': float(on_time),
    }


def compute_flow_matching_metrics(
    velocity_net: torch.nn.Module,
    z_0: torch.Tensor,
    z_1: torch.Tensor,
    num_samples: int = 100,
) -> Dict[str, float]:
    """
    计算 Flow Matching 特定指标
    
    Args:
        velocity_net: 速度场网络
        z_0: 源分布样本
        z_1: 目标分布样本
        num_samples: 时间采样数
    
    Returns:
        指标字典
    """
    device = z_0.device
    batch_size = z_0.shape[0]
    
    # 采样多个时间点的速度预测误差
    velocity_errors = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            t = torch.rand(batch_size, device=device)
            t_expanded = t.unsqueeze(-1)
            
            # 插值
            z_t = (1 - t_expanded) * z_0 + t_expanded * z_1
            
            # 目标速度
            target_v = z_1 - z_0
            
            # 预测速度
            pred_v = velocity_net(z_t, t)
            
            # 误差
            error = (pred_v - target_v).pow(2).sum(dim=-1).sqrt().mean()
            velocity_errors.append(error.item())
    
    return {
        'mean_velocity_error': float(np.mean(velocity_errors)),
        'std_velocity_error': float(np.std(velocity_errors)),
        'max_velocity_error': float(np.max(velocity_errors)),
    }
