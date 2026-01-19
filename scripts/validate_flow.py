#!/usr/bin/env python
"""
Flow Matching 模型验证脚本

验证内容:
1. 轨迹重建质量 - 给定 (z_0, z_1)，预测的轨迹 z_T 是否接近 z_1
2. 健康评分曲线 - 预测的退化曲线是否符合物理意义
3. 潜空间可视化 - 降维后查看轨迹分布
4. RUL 预测精度 - 对已知生命周期的样本评估 RUL 预测

使用方法:
    python scripts/validate_flow.py --checkpoint experiments/flow_matching/checkpoints/best_model.pt
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.flow_matching.models.flow_model import BatteryFlowModel, BatteryFlowConfig
from src.flow_matching.core.ode_solver import create_solver
from src.flow_matching.data.latent_cache import LatentCache
from src.flow_matching.utils.config import load_config
from src.smartwavev9 import DeltaBatteryModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_encoder(checkpoint_path: str, device: torch.device):
    """加载预训练 Encoder"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 获取配置
    if 'config' in checkpoint and 'model' in checkpoint['config']:
        model_cfg = checkpoint['config']['model']['model_config']
    else:
        model_cfg = {'d_model': 256, 'nhead': 4, 'num_layers': 1}
    
    encoder = DeltaBatteryModel(
        input_dim=model_cfg.get('input_dim', 1),
        d_model=model_cfg.get('d_model', 256),
        nhead=model_cfg.get('nhead', 4),
        num_layers=model_cfg.get('num_layers', 1),
        ROPE_max_len=model_cfg.get('ROPE_max_len', 5000),
        num_classes=model_cfg.get('num_classes', 4),
        task_type=model_cfg.get('task_type', 'classification'),
        max_level=model_cfg.get('max_level', 6),
        wavelet=model_cfg.get('wavelet', 'sym4'),
        patch_size=model_cfg.get('patch_size', 10),
        stride=model_cfg.get('stride', 10),
        dropout=model_cfg.get('dropout', 0.2),
    )
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith('model.')}
    
    # 预创建投影层
    for key in state_dict.keys():
        if key.startswith('freq_branch.channel_projections.') and key.endswith('.weight'):
            name = key.split('.')[2]
            weight = state_dict[key]
            encoder.freq_branch.channel_projections[name] = torch.nn.Linear(weight.shape[1], weight.shape[0])
    
    # 过滤旧参数
    keys_to_ignore = {'beta1', 'beta2_offset'}
    state_dict = {k: v for k, v in state_dict.items() if k not in keys_to_ignore}
    
    encoder.load_state_dict(state_dict, strict=False)
    encoder.to(device)
    encoder.eval()
    
    return encoder, model_cfg.get('d_model', 256)


def load_flow_model(checkpoint_path: str, encoder: torch.nn.Module, latent_dim: int, device: torch.device):
    """加载 Flow Matching 模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 从检查点获取配置
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        config = BatteryFlowConfig(
            latent_dim=cfg.get('latent_dim', latent_dim),
            hidden_dim=cfg.get('hidden_dim', 512),
            num_layers=cfg.get('num_layers', 6),
            max_cycle=cfg.get('max_cycle', 100),
        )
    else:
        config = BatteryFlowConfig(latent_dim=latent_dim)
    
    model = BatteryFlowModel(config, encoder)
    
    # 优先加载 EMA 权重
    if 'ema_state_dict' in checkpoint:
        model.velocity_net.load_state_dict(checkpoint['ema_state_dict'])
        logger.info("使用 EMA 权重")
    elif 'model_state_dict' in checkpoint:
        model.velocity_net.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    return model, config


@torch.no_grad()
def validate_trajectory_reconstruction(model, latent_vectors, cycles, num_samples=100, device='cuda'):
    """
    验证 1: 轨迹重建质量
    
    对于随机采样的 (z_0, z_1) 对，从 z_0 积分到 t=1，检查是否接近 z_1
    """
    logger.info("=" * 60)
    logger.info("验证 1: 轨迹重建质量")
    logger.info("=" * 60)
    
    n = len(cycles)
    errors = []
    relative_errors = []
    
    for _ in range(num_samples):
        # 随机选择一对
        idx_0 = np.random.randint(n)
        cycle_0 = cycles[idx_0].item()
        
        # 找一个较晚的 cycle
        valid_mask = (cycles - cycle_0 >= 10) & (cycles - cycle_0 <= 50)
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            continue
        
        idx_1 = valid_indices[np.random.randint(len(valid_indices))].item()
        cycle_1 = cycles[idx_1].item()
        
        z_0 = latent_vectors[idx_0:idx_0+1].to(device)
        z_1 = latent_vectors[idx_1:idx_1+1].to(device)
        
        # 积分预测
        t_span = torch.linspace(0, 1, 50, device=device)
        trajectory = model.predict_trajectory(z_0, t_span)
        z_pred = trajectory[-1]  # 最终预测
        
        # 计算误差
        error = torch.norm(z_pred - z_1).item()
        z_1_norm = torch.norm(z_1).item()
        rel_error = error / (z_1_norm + 1e-8)
        
        errors.append(error)
        relative_errors.append(rel_error)
    
    if errors:
        logger.info(f"  采样数: {len(errors)}")
        logger.info(f"  绝对误差: {np.mean(errors):.4f} ± {np.std(errors):.4f}")
        logger.info(f"  相对误差: {np.mean(relative_errors):.2%} ± {np.std(relative_errors):.2%}")
        
        # 评判标准
        if np.mean(relative_errors) < 0.1:
            logger.info("  ✓ 重建质量: 优秀 (相对误差 < 10%)")
        elif np.mean(relative_errors) < 0.2:
            logger.info("  ✓ 重建质量: 良好 (相对误差 < 20%)")
        else:
            logger.info("  ✗ 重建质量: 需改进 (相对误差 >= 20%)")
    
    return {'mean_error': np.mean(errors), 'mean_rel_error': np.mean(relative_errors)}


@torch.no_grad()
def validate_health_score_curve(model, latent_vectors, cycles, num_samples=20, device='cuda'):
    """
    验证 2: 健康评分曲线
    
    检查预测的健康曲线是否满足:
    1. 单调递增 (退化应该越来越严重)
    2. 范围在 [0, 1] 内
    3. 早期 cycle 评分低，晚期 cycle 评分高
    """
    logger.info("=" * 60)
    logger.info("验证 2: 健康评分曲线 (单调性 & 物理意义)")
    logger.info("=" * 60)
    
    monotonic_count = 0
    valid_range_count = 0
    early_late_correct = 0
    
    # 获取早期和晚期的 cycle
    min_cycle = cycles.min().item()
    max_cycle = cycles.max().item()
    early_threshold = min_cycle + (max_cycle - min_cycle) * 0.2
    late_threshold = min_cycle + (max_cycle - min_cycle) * 0.8
    
    for _ in range(num_samples):
        # 随机选择一个早期样本
        early_mask = cycles <= early_threshold
        early_indices = torch.where(early_mask)[0]
        if len(early_indices) == 0:
            continue
        
        idx = early_indices[np.random.randint(len(early_indices))].item()
        z_0 = latent_vectors[idx:idx+1].to(device)
        
        # 预测轨迹
        t_span = torch.linspace(0, 1, 100, device=device)
        trajectory = model.predict_trajectory(z_0, t_span)
        
        # 计算健康评分
        health_scores = []
        for z_t in trajectory:
            score = model.health_head(z_t).squeeze().item()
            health_scores.append(score)
        
        health_scores = np.array(health_scores)
        
        # 检查单调性 (允许一些波动)
        diffs = np.diff(health_scores)
        monotonic_ratio = np.mean(diffs >= -0.01)  # 允许微小下降
        if monotonic_ratio > 0.9:
            monotonic_count += 1
        
        # 检查范围
        if health_scores.min() >= -0.1 and health_scores.max() <= 1.1:
            valid_range_count += 1
        
        # 检查早期低、晚期高
        early_score = health_scores[:20].mean()
        late_score = health_scores[-20:].mean()
        if late_score > early_score:
            early_late_correct += 1
    
    logger.info(f"  采样数: {num_samples}")
    logger.info(f"  单调递增比例: {monotonic_count}/{num_samples} ({monotonic_count/num_samples:.1%})")
    logger.info(f"  范围有效比例: {valid_range_count}/{num_samples} ({valid_range_count/num_samples:.1%})")
    logger.info(f"  早低晚高比例: {early_late_correct}/{num_samples} ({early_late_correct/num_samples:.1%})")
    
    if monotonic_count/num_samples > 0.8 and early_late_correct/num_samples > 0.9:
        logger.info("  ✓ 健康曲线符合物理意义")
    else:
        logger.info("  ✗ 健康曲线需要检查")
    
    return {
        'monotonic_ratio': monotonic_count/num_samples,
        'valid_range_ratio': valid_range_count/num_samples,
        'early_late_ratio': early_late_correct/num_samples,
    }


@torch.no_grad()
def validate_latent_space_structure(model, latent_vectors, cycles, device='cuda'):
    """
    验证 3: 潜空间结构
    
    检查:
    1. 不同 cycle 的潜向量是否有序分布
    2. 轨迹是否平滑连续
    """
    logger.info("=" * 60)
    logger.info("验证 3: 潜空间结构")
    logger.info("=" * 60)
    
    try:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("需要 sklearn 和 matplotlib 进行可视化")
        return {}
    
    # PCA 降维
    latent_np = latent_vectors.cpu().numpy()
    cycles_np = cycles.cpu().numpy()
    
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_np)
    
    # 计算解释方差
    explained_var = pca.explained_variance_ratio_.sum()
    logger.info(f"  PCA 2D 解释方差: {explained_var:.1%}")
    
    # 计算 cycle 与 PC1 的相关性
    from scipy.stats import spearmanr
    corr, p_value = spearmanr(cycles_np, latent_2d[:, 0])
    logger.info(f"  Cycle vs PC1 相关性: {corr:.3f} (p={p_value:.2e})")
    
    if abs(corr) > 0.5:
        logger.info("  ✓ 潜空间结构: cycle 与 PC1 显著相关，说明退化方向已被学习")
    else:
        logger.info("  ⚠ 潜空间结构: cycle 与 PC1 相关性较弱")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图: 所有样本的潜空间分布
    scatter = axes[0].scatter(latent_2d[:, 0], latent_2d[:, 1], c=cycles_np, cmap='viridis', alpha=0.6)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('Latent Space Distribution (colored by Cycle)')
    plt.colorbar(scatter, ax=axes[0], label='Cycle')
    
    # 右图: 预测轨迹
    # 选择一个早期样本，预测其轨迹
    early_idx = cycles_np.argmin()
    z_0 = latent_vectors[early_idx:early_idx+1].to(device)
    t_span = torch.linspace(0, 1, 100, device=device)
    trajectory = model.predict_trajectory(z_0, t_span).cpu().numpy()[:, 0, :]
    trajectory_2d = pca.transform(trajectory)
    
    axes[1].scatter(latent_2d[:, 0], latent_2d[:, 1], c=cycles_np, cmap='viridis', alpha=0.3)
    axes[1].plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 'r-', linewidth=2, label='Predicted Trajectory')
    axes[1].scatter(trajectory_2d[0, 0], trajectory_2d[0, 1], c='green', s=100, marker='o', label='Start (t=0)')
    axes[1].scatter(trajectory_2d[-1, 0], trajectory_2d[-1, 1], c='red', s=100, marker='x', label='End (t=1)')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title('Predicted Trajectory in Latent Space')
    axes[1].legend()
    
    plt.tight_layout()
    
    # 保存
    save_path = Path('experiments/flow_matching/validation')
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / 'latent_space_validation.png', dpi=150, bbox_inches='tight')
    logger.info(f"  可视化已保存: {save_path / 'latent_space_validation.png'}")
    plt.close()
    
    return {'explained_variance': explained_var, 'cycle_pc1_correlation': corr}


@torch.no_grad()
def validate_rul_prediction(model, latent_vectors, cycles, num_samples=50, device='cuda'):
    """
    验证 4: RUL 预测精度
    
    对于已知 cycle 的样本，预测其到达某个 "失效" 点的 cycle 数
    """
    logger.info("=" * 60)
    logger.info("验证 4: RUL (剩余寿命) 预测能力")
    logger.info("=" * 60)
    
    max_cycle = cycles.max().item()
    min_cycle = cycles.min().item()
    
    # 选择中间阶段的样本作为测试
    mid_start = min_cycle + (max_cycle - min_cycle) * 0.3
    mid_end = min_cycle + (max_cycle - min_cycle) * 0.6
    
    test_mask = (cycles >= mid_start) & (cycles <= mid_end)
    test_indices = torch.where(test_mask)[0]
    
    if len(test_indices) == 0:
        logger.warning("  没有足够的中期样本进行 RUL 验证")
        return {}
    
    # 随机采样
    sample_indices = test_indices[torch.randperm(len(test_indices))[:min(num_samples, len(test_indices))]]
    
    rul_errors = []
    
    for idx in sample_indices:
        current_cycle = cycles[idx].item()
        z_0 = latent_vectors[idx:idx+1].to(device)
        
        # 真实 RUL (到最大 cycle)
        true_rul = max_cycle - current_cycle
        
        # 预测 RUL
        t_current = current_cycle / max_cycle
        t_span = torch.linspace(t_current, 1.0, 50, device=device)
        trajectory = model.predict_trajectory(z_0, t_span)
        
        # 计算健康评分
        health_scores = []
        for z_t in trajectory:
            score = model.health_head(z_t).squeeze().item()
            health_scores.append(score)
        
        health_scores = np.array(health_scores)
        
        # 找到超过阈值的点 (假设 0.8 为失效阈值)
        threshold = 0.8
        failure_indices = np.where(health_scores >= threshold)[0]
        
        if len(failure_indices) > 0:
            failure_t = t_span[failure_indices[0]].item()
            predicted_failure_cycle = failure_t * max_cycle
            predicted_rul = predicted_failure_cycle - current_cycle
        else:
            predicted_rul = max_cycle - current_cycle  # 未到阈值
        
        rul_error = abs(predicted_rul - true_rul)
        rul_errors.append(rul_error)
    
    if rul_errors:
        logger.info(f"  测试样本数: {len(rul_errors)}")
        logger.info(f"  RUL 误差 (cycles): {np.mean(rul_errors):.1f} ± {np.std(rul_errors):.1f}")
        logger.info(f"  RUL 相对误差: {np.mean(rul_errors)/max_cycle:.1%}")
        
        if np.mean(rul_errors)/max_cycle < 0.15:
            logger.info("  ✓ RUL 预测: 精度较好 (< 15% 误差)")
        else:
            logger.info("  ⚠ RUL 预测: 误差较大，可能需要更多训练")
    
    return {'mean_rul_error': np.mean(rul_errors) if rul_errors else None}


def generate_demo_prediction(model, latent_vectors, cycles, device='cuda'):
    """
    生成演示预测图
    
    展示从不同初始 cycle 出发的预测轨迹
    """
    logger.info("=" * 60)
    logger.info("生成演示预测图")
    logger.info("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("需要 matplotlib")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    max_cycle = cycles.max().item()
    
    # 选择不同阶段的起点
    percentiles = [0.1, 0.3, 0.5, 0.7]
    colors = ['blue', 'green', 'orange', 'red']
    
    for ax_idx, (pct, color) in enumerate(zip(percentiles, colors)):
        ax = axes[ax_idx // 2, ax_idx % 2]
        
        target_cycle = cycles.min().item() + (max_cycle - cycles.min().item()) * pct
        idx = (torch.abs(cycles - target_cycle)).argmin().item()
        start_cycle = cycles[idx].item()
        
        z_0 = latent_vectors[idx:idx+1].to(device)
        
        # 预测轨迹
        t_start = start_cycle / max_cycle
        t_span = torch.linspace(t_start, 1.0, 100, device=device)
        trajectory = model.predict_trajectory(z_0, t_span)
        
        # 计算健康评分
        health_scores = []
        for z_t in trajectory:
            score = model.health_head(z_t).squeeze().item()
            health_scores.append(score)
        
        predicted_cycles = t_span.cpu().numpy() * max_cycle
        
        ax.plot(predicted_cycles, health_scores, color=color, linewidth=2, 
                label=f'From Cycle {int(start_cycle)}')
        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Failure Threshold')
        ax.axvline(x=start_cycle, color=color, linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Health Score (Degradation)')
        ax.set_title(f'Starting from Cycle {int(start_cycle)} ({pct:.0%} of lifecycle)')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Flow Matching Lifecycle Predictions', fontsize=14)
    plt.tight_layout()
    
    save_path = Path('experiments/flow_matching/validation')
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / 'lifecycle_predictions.png', dpi=150, bbox_inches='tight')
    logger.info(f"  预测图已保存: {save_path / 'lifecycle_predictions.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Flow Matching Model Validation")
    parser.add_argument('--checkpoint', type=str, required=True, help='Flow Matching 模型检查点')
    parser.add_argument('--encoder_checkpoint', type=str, default='latest.pth', help='Encoder 检查点')
    parser.add_argument('--cache_dir', type=str, default='experiments/flow_matching/cache', 
                        help='潜空间缓存目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 1. 加载 Encoder
    logger.info("加载 Encoder...")
    encoder, latent_dim = load_encoder(args.encoder_checkpoint, device)
    
    # 2. 加载 Flow Matching 模型
    logger.info("加载 Flow Matching 模型...")
    model, config = load_flow_model(args.checkpoint, encoder, latent_dim, device)
    
    # 3. 加载潜空间缓存
    logger.info("加载潜空间缓存...")
    cache_dir = Path(args.cache_dir)
    if not (cache_dir / 'latent_vectors.pt').exists():
        logger.error(f"缓存不存在: {cache_dir}")
        logger.info("请先运行训练脚本生成缓存: python scripts/train_flow.py --compute_cache")
        return
    
    cache_data = torch.load(cache_dir / 'latent_vectors.pt')
    latent_vectors = cache_data['latent_vectors']
    cycles = cache_data['cycles']
    
    logger.info(f"加载 {len(latent_vectors)} 个潜向量")
    logger.info(f"Cycle 范围: {cycles.min().item()} - {cycles.max().item()}")
    
    # 4. 运行验证
    logger.info("\n" + "=" * 60)
    logger.info("开始 Flow Matching 模型验证")
    logger.info("=" * 60 + "\n")
    
    results = {}
    
    # 验证 1: 轨迹重建
    results['reconstruction'] = validate_trajectory_reconstruction(
        model, latent_vectors, cycles, num_samples=100, device=device
    )
    
    # 验证 2: 健康曲线
    results['health_curve'] = validate_health_score_curve(
        model, latent_vectors, cycles, num_samples=20, device=device
    )
    
    # 验证 3: 潜空间结构
    results['latent_structure'] = validate_latent_space_structure(
        model, latent_vectors, cycles, device=device
    )
    
    # 验证 4: RUL 预测
    results['rul'] = validate_rul_prediction(
        model, latent_vectors, cycles, num_samples=50, device=device
    )
    
    # 生成演示图
    generate_demo_prediction(model, latent_vectors, cycles, device=device)
    
    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("验证总结")
    logger.info("=" * 60)
    
    all_passed = True
    
    if results['reconstruction'].get('mean_rel_error', 1) < 0.2:
        logger.info("✓ 轨迹重建: 通过")
    else:
        logger.info("✗ 轨迹重建: 需改进")
        all_passed = False
    
    if results['health_curve'].get('monotonic_ratio', 0) > 0.8:
        logger.info("✓ 健康曲线: 通过")
    else:
        logger.info("✗ 健康曲线: 需改进")
        all_passed = False
    
    if abs(results['latent_structure'].get('cycle_pc1_correlation', 0)) > 0.5:
        logger.info("✓ 潜空间结构: 通过")
    else:
        logger.info("⚠ 潜空间结构: 相关性较弱")
    
    if all_passed:
        logger.info("\n🎉 模型验证通过！可以进行推理部署。")
    else:
        logger.info("\n⚠ 部分验证未通过，建议继续训练或调整超参数。")


if __name__ == "__main__":
    main()
