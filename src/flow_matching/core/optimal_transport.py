"""
最优传输工具函数

提供最优传输相关的计算功能，包括:
1. OT 计划计算 (Sinkhorn, Hungarian)
2. Wasserstein 距离计算
3. OT 配对采样
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import math


def compute_cost_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    metric: str = "euclidean",
) -> torch.Tensor:
    """
    计算成本矩阵
    
    Args:
        x: [n, d] 源点集
        y: [m, d] 目标点集
        metric: 距离度量 ("euclidean", "cosine", "squared_euclidean")
    
    Returns:
        cost: [n, m] 成本矩阵
    """
    if metric == "euclidean":
        return torch.cdist(x, y, p=2)
    elif metric == "squared_euclidean":
        return torch.cdist(x, y, p=2) ** 2
    elif metric == "cosine":
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)
        return 1 - x_norm @ y_norm.T
    else:
        raise ValueError(f"未知的距离度量: {metric}")


def sinkhorn_algorithm(
    cost_matrix: torch.Tensor,
    epsilon: float = 0.1,
    num_iterations: int = 100,
    convergence_threshold: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sinkhorn 算法求解熵正则化最优传输
    
    求解: min_P <C, P> + ε * H(P)
    subject to: P @ 1 = a, P.T @ 1 = b
    
    其中 H(P) = -sum(P * log(P)) 是熵正则化项。
    
    Args:
        cost_matrix: [n, m] 成本矩阵
        epsilon: 熵正则化系数（越小越接近精确 OT，但数值越不稳定）
        num_iterations: 最大迭代次数
        convergence_threshold: 收敛阈值
    
    Returns:
        plan: [n, m] 传输计划
        dual_potentials: (u, v) 对偶变量
    """
    n, m = cost_matrix.shape
    device = cost_matrix.device
    dtype = cost_matrix.dtype
    
    # 均匀边际分布
    a = torch.ones(n, device=device, dtype=dtype) / n
    b = torch.ones(m, device=device, dtype=dtype) / m
    
    # 数值稳定化: 使用 log-domain Sinkhorn
    # log K = -C / epsilon
    log_K = -cost_matrix / epsilon
    
    # 初始化 log 缩放因子
    log_u = torch.zeros(n, device=device, dtype=dtype)
    log_v = torch.zeros(m, device=device, dtype=dtype)
    
    # Sinkhorn 迭代 (log domain)
    for iteration in range(num_iterations):
        log_u_old = log_u.clone()
        
        # 更新 log_u 和 log_v
        log_u = torch.log(a + 1e-10) - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = torch.log(b + 1e-10) - torch.logsumexp(log_K.T + log_u.unsqueeze(0), dim=1)
        
        # 检查收敛
        if (log_u - log_u_old).abs().max() < convergence_threshold:
            break
    
    # 计算传输计划 (log domain to linear)
    log_plan = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    plan = torch.exp(log_plan)
    
    return plan, (torch.exp(log_u), torch.exp(log_v))


def compute_ot_plan(
    source: torch.Tensor,
    target: torch.Tensor,
    metric: str = "squared_euclidean",
    epsilon: float = 0.01,
    num_iterations: int = 100,
) -> torch.Tensor:
    """
    计算源和目标之间的最优传输计划
    
    Args:
        source: [n, d] 源点集
        target: [m, d] 目标点集
        metric: 距离度量
        epsilon: Sinkhorn 正则化系数
        num_iterations: Sinkhorn 迭代次数
    
    Returns:
        plan: [n, m] 传输计划，plan[i,j] 表示从 source[i] 传输到 target[j] 的质量
    """
    cost_matrix = compute_cost_matrix(source, target, metric)
    plan, _ = sinkhorn_algorithm(cost_matrix, epsilon, num_iterations)
    return plan


def sample_ot_pairs(
    source: torch.Tensor,
    target: torch.Tensor,
    num_samples: int,
    epsilon: float = 0.01,
    replacement: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据 OT 计划采样配对
    
    Args:
        source: [n, d] 源点集
        target: [m, d] 目标点集
        num_samples: 采样数量
        epsilon: Sinkhorn 正则化系数
        replacement: 是否有放回采样
    
    Returns:
        source_samples: [num_samples, d] 采样的源点
        target_samples: [num_samples, d] 对应的目标点
    """
    plan = compute_ot_plan(source, target, epsilon=epsilon)
    
    # 将计划展平为一维分布
    flat_plan = plan.flatten()
    
    # 采样索引
    indices = torch.multinomial(flat_plan, num_samples, replacement=replacement)
    
    # 转换为 (i, j) 索引
    n, m = plan.shape
    source_indices = indices // m
    target_indices = indices % m
    
    return source[source_indices], target[target_indices]


def wasserstein_distance(
    source: torch.Tensor,
    target: torch.Tensor,
    p: int = 2,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """
    计算 Wasserstein-p 距离的近似值
    
    W_p(μ, ν) = (min_P sum_{i,j} c_{ij}^p * P_{ij})^{1/p}
    
    使用 Sinkhorn 近似。
    
    Args:
        source: [n, d] 源分布样本
        target: [m, d] 目标分布样本
        p: 距离的阶数
        epsilon: Sinkhorn 正则化系数
    
    Returns:
        W_p 距离
    """
    cost_matrix = compute_cost_matrix(source, target, "euclidean") ** p
    plan, _ = sinkhorn_algorithm(cost_matrix, epsilon)
    
    # 计算传输成本
    transport_cost = (cost_matrix * plan).sum()
    
    return transport_cost ** (1 / p)


def batch_wasserstein_distance(
    source_batch: torch.Tensor,
    target_batch: torch.Tensor,
    p: int = 2,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """
    批量计算 Wasserstein 距离
    
    Args:
        source_batch: [batch, n, d] 源分布批次
        target_batch: [batch, m, d] 目标分布批次
        p: 距离阶数
        epsilon: Sinkhorn 正则化系数
    
    Returns:
        distances: [batch] 每对分布的 W_p 距离
    """
    batch_size = source_batch.shape[0]
    distances = []
    
    for i in range(batch_size):
        dist = wasserstein_distance(source_batch[i], target_batch[i], p, epsilon)
        distances.append(dist)
    
    return torch.stack(distances)


def ot_barycenter(
    distributions: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    epsilon: float = 0.01,
    num_iterations: int = 100,
) -> torch.Tensor:
    """
    计算多个分布的 Wasserstein 重心
    
    用于找到多个潜空间分布的"平均"。
    
    Args:
        distributions: [k, n, d] k 个分布，每个有 n 个样本
        weights: [k] 各分布的权重，默认均匀
        epsilon: Sinkhorn 正则化系数
        num_iterations: 迭代次数
    
    Returns:
        barycenter: [n, d] 重心分布的样本
    """
    k, n, d = distributions.shape
    device = distributions.device
    dtype = distributions.dtype
    
    if weights is None:
        weights = torch.ones(k, device=device, dtype=dtype) / k
    else:
        weights = weights / weights.sum()
    
    # 初始化重心为第一个分布（或加权平均）
    barycenter = (distributions * weights.view(-1, 1, 1)).sum(dim=0)
    
    for iteration in range(num_iterations):
        # 计算每个分布到重心的传输计划
        transported = torch.zeros_like(barycenter)
        
        for i in range(k):
            plan = compute_ot_plan(barycenter, distributions[i], epsilon=epsilon)
            # 传输：barycenter -> distributions[i]
            transported += weights[i] * (plan @ distributions[i])
        
        # 更新重心
        barycenter = transported
    
    return barycenter


def interpolate_distributions(
    source: torch.Tensor,
    target: torch.Tensor,
    num_steps: int = 10,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """
    在两个分布之间进行 Wasserstein 插值
    
    生成从 source 到 target 的分布插值序列（位移插值）。
    
    Args:
        source: [n, d] 源分布
        target: [n, d] 目标分布（需要相同数量的样本）
        num_steps: 插值步数
        epsilon: Sinkhorn 正则化系数
    
    Returns:
        interpolations: [num_steps, n, d] 插值序列
    """
    plan = compute_ot_plan(source, target, epsilon=epsilon)
    
    # 获取最优匹配
    # 使用 argmax 获取确定性匹配（近似）
    matching = plan.argmax(dim=1)
    target_matched = target[matching]
    
    # 线性插值
    t_values = torch.linspace(0, 1, num_steps, device=source.device)
    interpolations = []
    
    for t in t_values:
        interp = (1 - t) * source + t * target_matched
        interpolations.append(interp)
    
    return torch.stack(interpolations, dim=0)


class OTSampler:
    """
    OT 采样器：用于训练 Flow Matching 时的高效配对采样
    
    预计算 OT 计划，加速训练时的配对采样。
    """
    
    def __init__(
        self,
        epsilon: float = 0.01,
        cache_size: int = 1000,
    ):
        self.epsilon = epsilon
        self.cache_size = cache_size
        self._cache = {}
    
    def get_pairs(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取 OT 配对的 mini-batch
        
        Args:
            source: [n, d] 源数据集
            target: [m, d] 目标数据集
            batch_size: 批大小
        
        Returns:
            source_batch: [batch_size, d]
            target_batch: [batch_size, d]
        """
        return sample_ot_pairs(
            source, target, batch_size, 
            epsilon=self.epsilon, replacement=True
        )
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
