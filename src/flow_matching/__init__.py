"""
Battery Flow Matching - 电池潜空间分布迁移模型

通过 Flow Matching 学习电池从健康状态到退化状态的潜空间演化路径，
实现全生命周期预测与任意时刻状态还原。

核心组件:
- VelocityNetwork: 速度场神经网络
- BatteryFlowModel: Flow Matching 主模型
- FlowMatchingTrainer: 训练器
- LifecyclePredictor: 生命周期预测器

Author: Battery Research Team
Version: 1.0.0
"""

from .models.flow_model import BatteryFlowModel
from .models.velocity_net import VelocityNetwork
from .core.flow_matching_loss import FlowMatchingLoss
from .core.ode_solver import ODESolver

__version__ = "1.0.0"
__all__ = [
    "BatteryFlowModel",
    "VelocityNetwork", 
    "FlowMatchingLoss",
    "ODESolver",
]
