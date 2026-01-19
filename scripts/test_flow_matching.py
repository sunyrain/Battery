#!/usr/bin/env python
"""
Flow Matching 模块完整性测试

验证所有模块能正确导入和基本功能正常
"""

import os
import sys

# 添加项目根目录到 path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import numpy as np

def test_imports():
    """测试所有模块导入"""
    print("=" * 60)
    print("测试模块导入...")
    print("=" * 60)
    
    try:
        # Core 模块
        from src.flow_matching.core import (
            ODESolver, create_solver,
            FlowMatchingLoss, OptimalTransportPath,
            compute_ot_plan, wasserstein_distance, OTSampler,
        )
        print("✓ Core 模块导入成功")
        
        # Models 模块
        from src.flow_matching.models import (
            SinusoidalTimeEmbedding, ConditionEmbedding,
            VelocityNetwork, LightweightVelocityNetwork,
            BatteryFlowModel, BatteryFlowConfig,
        )
        print("✓ Models 模块导入成功")
        
        # Data 模块
        from src.flow_matching.data import (
            BatteryFlowDataset, LatentPairDataset,
            SignalProcessor, LatentCache,
        )
        print("✓ Data 模块导入成功")
        
        # Training 模块
        from src.flow_matching.training import (
            FlowMatchingTrainer, TrainerConfig,
            CheckpointCallback, LoggingCallback, EarlyStoppingCallback,
        )
        print("✓ Training 模块导入成功")
        
        # Inference 模块
        from src.flow_matching.inference import (
            LifecyclePredictor, TrajectoryVisualizer,
        )
        print("✓ Inference 模块导入成功")
        
        # Utils 模块
        from src.flow_matching.utils import (
            load_config, FlowMatchingConfig,
            compute_trajectory_mse, compute_health_score_accuracy,
        )
        print("✓ Utils 模块导入成功")
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        return False


def test_velocity_network():
    """测试 VelocityNetwork"""
    print("\n" + "=" * 60)
    print("测试 VelocityNetwork...")
    print("=" * 60)
    
    from src.flow_matching.models import VelocityNetwork
    
    latent_dim = 128
    hidden_dim = 256
    cond_embed_dim = 64
    batch_size = 4
    
    net = VelocityNetwork(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        time_embed_dim=64,
        cond_embed_dim=cond_embed_dim,
        num_layers=4,
    )
    
    z = torch.randn(batch_size, latent_dim)
    t = torch.rand(batch_size)
    # 条件需要是 [batch, cond_embed_dim] 维度的向量
    c = torch.randn(batch_size, cond_embed_dim)
    
    # 测试带条件
    v = net(z, t, c)
    assert v.shape == z.shape, f"输出形状不匹配: {v.shape} vs {z.shape}"
    print(f"✓ 带条件输入: z={z.shape}, t={t.shape}, c={c.shape}")
    print(f"✓ 输出: v={v.shape}")
    
    # 测试不带条件
    v2 = net(z, t, None)
    print(f"✓ 无条件输出: {v2.shape}")
    print(f"✓ 参数量: {sum(p.numel() for p in net.parameters()):,}")


def test_ode_solver():
    """测试 ODE Solver"""
    print("\n" + "=" * 60)
    print("测试 ODE Solver...")
    print("=" * 60)
    
    from src.flow_matching.core.ode_solver import ODESolver, ODEFunction, ODESolverConfig, SolverType
    
    latent_dim = 64
    batch_size = 2
    
    # 定义简单的速度函数用于测试
    def simple_velocity_fn(z, t, c=None):
        # 简单的线性速度场: v = -z (收缩到原点)
        if isinstance(t, float):
            t = torch.tensor(t, device=z.device)
        return -z * 0.1
    
    # 创建 ODE 函数包装器
    ode_func = ODEFunction(velocity_fn=simple_velocity_fn)
    
    # 创建求解器配置
    config = ODESolverConfig(solver_type=SolverType.EULER)
    solver = ODESolver(config)
    
    z_0 = torch.randn(batch_size, latent_dim)
    t_span = torch.linspace(0, 1, 11)
    
    trajectory = solver.solve(ode_func, z_0, t_span, return_trajectory=True)
    
    print(f"✓ 初始状态: {z_0.shape}")
    print(f"✓ 轨迹: {trajectory.shape}")
    print(f"✓ 时间点数: {trajectory.shape[0]}")


def test_flow_matching_loss():
    """测试 FlowMatchingLoss"""
    print("\n" + "=" * 60)
    print("测试 FlowMatchingLoss...")
    print("=" * 60)
    
    from src.flow_matching.core import FlowMatchingLoss
    from src.flow_matching.models import VelocityNetwork
    
    latent_dim = 64
    batch_size = 8
    
    velocity_net = VelocityNetwork(
        latent_dim=latent_dim,
        hidden_dim=128,
        num_layers=2,
    )
    
    # FlowMatchingLoss 只接受 path_type，velocity_net 在 forward 时传入
    loss_fn = FlowMatchingLoss(path_type='linear')
    
    z_0 = torch.randn(batch_size, latent_dim)
    z_1 = torch.randn(batch_size, latent_dim)
    
    # forward 返回字典
    result = loss_fn(velocity_net, z_0, z_1)
    loss = result['loss']
    
    print(f"✓ 源分布: {z_0.shape}")
    print(f"✓ 目标分布: {z_1.shape}")
    print(f"✓ 损失值: {loss.item():.6f}")
    
    # 测试反向传播
    loss.backward()
    print("✓ 反向传播成功")


def test_optimal_transport():
    """测试最优传输"""
    print("\n" + "=" * 60)
    print("测试最优传输...")
    print("=" * 60)
    
    from src.flow_matching.core import compute_ot_plan, wasserstein_distance
    
    n, m = 32, 32
    dim = 64
    
    source = torch.randn(n, dim)
    target = torch.randn(m, dim)
    
    # compute_ot_plan 返回 plan tensor
    plan = compute_ot_plan(source, target)
    
    print(f"✓ 源分布: {source.shape}")
    print(f"✓ 目标分布: {target.shape}")
    print(f"✓ 传输计划: {plan.shape}")
    print(f"✓ 计划和: {plan.sum().item():.6f} (应接近 1.0)")
    
    # Wasserstein 距离
    w_dist = wasserstein_distance(source, target)
    print(f"✓ Wasserstein 距离: {w_dist.item():.6f}")


def test_flow_model():
    """测试 BatteryFlowModel (不含 Encoder)"""
    print("\n" + "=" * 60)
    print("测试 BatteryFlowModel...")
    print("=" * 60)
    
    from src.flow_matching.models import BatteryFlowModel, BatteryFlowConfig
    
    config = BatteryFlowConfig(
        latent_dim=64,
        hidden_dim=128,
        num_layers=3,
        solver_type='euler',
    )
    
    model = BatteryFlowModel(config, encoder=None)
    
    batch_size = 4
    z_0 = torch.randn(batch_size, config.latent_dim)
    
    # 测试轨迹预测 - 需要传入 t_span
    t_span = torch.linspace(0, 1, 20)
    trajectory = model.predict_trajectory(z_0, t_span)
    
    print(f"✓ 初始潜向量: {z_0.shape}")
    print(f"✓ 轨迹: {trajectory.shape}")
    print(f"✓ 可训练参数: {model.get_num_trainable_params():,}")


def test_config():
    """测试配置加载"""
    print("\n" + "=" * 60)
    print("测试配置管理...")
    print("=" * 60)
    
    from src.flow_matching.utils import FlowMatchingConfig
    
    config = FlowMatchingConfig(
        latent_dim=128,
        hidden_dim=512,
        num_epochs=100,
    )
    
    print(f"✓ 创建配置: latent_dim={config.latent_dim}, hidden_dim={config.hidden_dim}")
    
    # 转换为字典
    config_dict = config.to_dict()
    print(f"✓ 转换为字典: {len(config_dict)} 个字段")
    
    # 从字典恢复
    config2 = FlowMatchingConfig.from_dict(config_dict)
    print(f"✓ 从字典恢复: latent_dim={config2.latent_dim}")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("   Flow Matching 模块完整性测试   ")
    print("=" * 60 + "\n")
    
    tests = [
        ("模块导入", test_imports),
        ("VelocityNetwork", test_velocity_network),
        ("ODE Solver", test_ode_solver),
        ("Flow Matching Loss", test_flow_matching_loss),
        ("最优传输", test_optimal_transport),
        ("Flow Model", test_flow_model),
        ("配置管理", test_config),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
