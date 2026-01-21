#!/usr/bin/env python
"""
新电池退化预测脚本

功能：
    给定一个新电池的超声信号（after, before），预测其退化轨迹和 RUL

使用方法:
    # 方式1: 提供 embedding 文件（如果已经编码好）
    python scripts/predict_new_battery.py --embedding path/to/embedding.pt
    
    # 方式2: 提供原始信号文件
    python scripts/predict_new_battery.py --after_signal path/to/after.csv --before_signal path/to/before.csv
    
    # 方式3: 提供 after 和 before 目录（自动找最新文件）
    python scripts/predict_new_battery.py --after_dir Data/NewBattery/after --before_dir Data/NewBattery/before
    
    # 指定当前 cycle（默认为 1）
    python scripts/predict_new_battery.py --after_signal ... --current_cycle 5
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.finetuneV6_fourcls import DeltaBatteryModel
from src.flow_matching.models.flow_model import BatteryFlowModel, BatteryFlowConfig


# ============================================================
# 模型加载
# ============================================================

def load_encoder(encoder_path, device):
    """加载 Encoder"""
    checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)
    
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
    
    for key in state_dict.keys():
        if key.startswith('freq_branch.channel_projections.') and key.endswith('.weight'):
            name = key.split('.')[2]
            weight = state_dict[key]
            encoder.freq_branch.channel_projections[name] = torch.nn.Linear(weight.shape[1], weight.shape[0])
    
    encoder.load_state_dict(state_dict, strict=False)
    encoder.to(device).eval()
    
    # 获取归一化参数
    normalize_mean = 0.0014818326526033054
    normalize_std = 0.15278572162321524
    
    return encoder, normalize_mean, normalize_std


def load_flow_model(flow_path, encoder, device):
    """加载 Flow Model"""
    flow_ckpt = torch.load(flow_path, map_location=device, weights_only=False)
    
    config = BatteryFlowConfig(
        latent_dim=256,
        hidden_dim=256,
        cond_embed_dim=64,
        time_embed_dim=64,
        num_layers=4,
        max_cycle=100,
        lightweight=True,
    )
    
    model = BatteryFlowModel(config, encoder)
    
    if 'ema_state_dict' in flow_ckpt:
        model.load_state_dict(flow_ckpt['ema_state_dict'], strict=False)
    else:
        model.load_state_dict(flow_ckpt['model_state_dict'], strict=False)
    
    model.to(device).eval()
    
    return model


# ============================================================
# 信号处理
# ============================================================

def load_signal_from_csv(csv_path, signal_length=3000):
    """从 CSV 文件加载信号"""
    df = pd.read_csv(csv_path)
    
    # 假设信号在第一列或名为 'signal', 'value', 'amplitude' 的列
    if 'signal' in df.columns:
        signal = df['signal'].values
    elif 'value' in df.columns:
        signal = df['value'].values
    elif 'amplitude' in df.columns:
        signal = df['amplitude'].values
    else:
        # 使用第一列（跳过可能的时间列）
        signal = df.iloc[:, -1].values if df.shape[1] > 1 else df.iloc[:, 0].values
    
    # 截断或填充到指定长度
    if len(signal) > signal_length:
        signal = signal[:signal_length]
    elif len(signal) < signal_length:
        signal = np.pad(signal, (0, signal_length - len(signal)), mode='constant')
    
    return signal.astype(np.float32)


def encode_signal(encoder, after_signal, before_signal, normalize_mean, normalize_std, device):
    """
    将原始信号编码为潜向量
    
    Args:
        encoder: DeltaBatteryModel
        after_signal: after 信号 numpy array [signal_length]
        before_signal: before 信号 numpy array [signal_length]
        normalize_mean: 归一化均值
        normalize_std: 归一化标准差
        device: 设备
    
    Returns:
        z: 潜向量 [1, latent_dim]
    """
    # 归一化 (对原始信号分别归一化)
    after_norm = (after_signal - normalize_mean) / normalize_std
    before_norm = (before_signal - normalize_mean) / normalize_std
    
    # 转换为 tensor [1, 1, signal_length] (batch, channels, seq_len)
    x_after = torch.tensor(after_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    x_before = torch.tensor(before_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # 使用 Encoder 获取潜向量 (fused_delta)
    with torch.no_grad():
        # 时域：分别编码后差分
        time_feat_after = encoder.time_branch(x_after)
        time_feat_before = encoder.time_branch(x_before)
        time_delta = time_feat_after - time_feat_before
        
        # 频域：内部先差分后编码
        freq_delta = encoder.freq_branch(x_after, x_before)
        
        # Alpha 加权融合
        alpha = torch.sigmoid(encoder.alpha)
        z = alpha * time_delta + (1 - alpha) * freq_delta  # [1, latent_dim]
    
    return z


# ============================================================
# 评分器
# ============================================================

class PretrainedScorer:
    """使用预训练 output_head 计算退化评分"""
    
    def __init__(self, encoder, device):
        self.output_head = encoder.output_head
        self.device = device
        self.weights = torch.tensor([0, 0.33, 0.67, 1.0], device=device)
    
    @torch.no_grad()
    def score(self, z):
        """计算退化评分 (0=健康, 1=失效)"""
        z = z.to(self.device)
        logits = self.output_head(z)
        probs = F.softmax(logits, dim=-1)
        score = (probs * self.weights).sum(dim=-1)
        return score.item() if score.numel() == 1 else score.cpu().numpy()
    
    @torch.no_grad()
    def score_trajectory(self, trajectory):
        """计算轨迹上所有点的评分"""
        scores = []
        for z in trajectory:
            scores.append(self.score(z))
        return np.array(scores)


# ============================================================
# 预测
# ============================================================

def predict_lifecycle(model, z_0, current_cycle, max_cycle=100, end_cycle=100, device='cuda'):
    """
    预测从当前 cycle 到 end_cycle 的退化轨迹
    
    Args:
        model: Flow Model
        z_0: 初始潜向量 [1, latent_dim]
        current_cycle: 当前 cycle 数
        max_cycle: 归一化用的最大 cycle
        end_cycle: 预测终点 cycle
        device: 设备
    
    Returns:
        trajectory: 轨迹 [num_steps, 1, latent_dim]
        cycles: cycle 数组
    """
    # 归一化时间
    t_start = (current_cycle - 1) / (max_cycle - 1)
    t_end = (end_cycle - 1) / (max_cycle - 1)
    
    num_steps = end_cycle - current_cycle + 1
    t_span = torch.linspace(t_start, t_end, num_steps, device=device)
    
    with torch.no_grad():
        trajectory = model.predict_trajectory(z_0, t_span)
    
    cycles = np.linspace(current_cycle, end_cycle, num_steps)
    
    return trajectory, cycles


def find_failure_point(scores, cycles, threshold=1):
    """找到第一个达到失效阈值的点"""
    for i, s in enumerate(scores):
        if s >= threshold:
            return int(cycles[i]), i
    return None, None


# ============================================================
# 可视化
# ============================================================

def plot_prediction(cycles, scores, current_cycle, failure_cycle, rul, 
                    failure_threshold=1, true_end_of_life=54, 
                    title_prefix="新电池", save_path=None, show=True):
    """
    绘制预测结果
    
    左图: 退化轨迹
    右图: RUL 指示器
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ==================== 左图: 退化轨迹 ====================
    ax1 = axes[0]
    
    # 绘制退化曲线
    ax1.plot(cycles, scores, 'b-', linewidth=2.5, label='预测退化曲线')
    
    # 当前位置
    current_score = scores[0]
    ax1.scatter([current_cycle], [current_score], c='green', s=150, 
                zorder=5, label=f'当前（Cycle {current_cycle}）')
    
    # 失效阈值线
    ax1.axhline(y=failure_threshold, color='gray', linestyle=':', 
                linewidth=1.5, label=f'失效阈值（{failure_threshold}）')
    
    # 预测失效点
    if failure_cycle:
        ax1.axvline(x=failure_cycle, color='red', linestyle='--', 
                    linewidth=2, label=f'预测失效（Cycle {failure_cycle}）')
        
        # 填充危险区域
        ax1.fill_between(cycles, failure_threshold, 1.0, 
                         alpha=0.2, color='orange')
        
        # RUL 标注
        mid_cycle = (current_cycle + failure_cycle) / 2
        mid_score = np.interp(mid_cycle, cycles, scores)
        ax1.annotate(f'RUL={rul}', xy=(mid_cycle, mid_score), 
                     fontsize=14, fontweight='bold',
                     ha='center', va='bottom')
    
    ax1.set_xlabel('Cycle', fontsize=12)
    ax1.set_ylabel('退化评分（0=健康，1=失效）', fontsize=12)
    ax1.set_title(f'{title_prefix} - 退化轨迹预测', fontsize=14, fontweight='bold')
    ax1.set_xlim(current_cycle - 1, max(cycles) + 1)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ==================== 右图: RUL 指示器 ====================
    ax2 = axes[1]
    
    # 定义区域（基于预期寿命）
    expected_life = true_end_of_life
    danger_zone = expected_life * 0.3    # 危险: 0-30%
    warning_zone = expected_life * 0.7   # 警告: 30%-70%
    
    # 绘制背景色条
    ax2.axvspan(0, danger_zone, alpha=0.6, color='#FF6B6B', label='危险')
    ax2.axvspan(danger_zone, warning_zone, alpha=0.6, color='#FFE66D', label='警告')
    ax2.axvspan(warning_zone, expected_life, alpha=0.6, color='#88D498', label='健康')
    
    # 绘制 RUL 指示线
    if rul is not None:
        rul_display = min(rul, expected_life)  # 限制显示范围
        ax2.axvline(x=rul_display, color='blue', linewidth=4, label=f'RUL = {rul}')
    
    # 标注
    ax2.text(danger_zone / 2, 0.5, '危险', fontsize=14, ha='center', va='center',
             color='darkred', fontweight='bold', transform=ax2.get_xaxis_transform())
    ax2.text((danger_zone + warning_zone) / 2, 0.5, '警告', fontsize=14, ha='center', va='center',
             color='#996600', fontweight='bold', transform=ax2.get_xaxis_transform())
    ax2.text((warning_zone + expected_life) / 2, 0.5, '健康', fontsize=14, ha='center', va='center',
             color='darkgreen', fontweight='bold', transform=ax2.get_xaxis_transform())
    
    ax2.set_xlabel('剩余使用寿命（Cycles）', fontsize=12)
    ax2.set_xlim(0, expected_life)
    ax2.set_ylim(0, 1)
    ax2.set_title(f'RUL = {rul if rul else "N/A"} cycles', fontsize=14, fontweight='bold')
    ax2.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="新电池退化预测")
    
    # 输入方式 1: 已有 embedding
    parser.add_argument('--embedding', type=str, help='embedding 文件路径 (.pt)')
    
    # 输入方式 2: 原始信号文件
    parser.add_argument('--after_signal', type=str, help='after 信号 CSV 文件')
    parser.add_argument('--before_signal', type=str, help='before 信号 CSV 文件')
    
    # 输入方式 3: 信号目录（自动找最新）
    parser.add_argument('--after_dir', type=str, help='after 信号目录')
    parser.add_argument('--before_dir', type=str, help='before 信号目录')
    
    # 通用参数
    parser.add_argument('--current_cycle', type=int, default=1, help='当前 cycle（默认=1，表示新电池）')
    parser.add_argument('--encoder_path', type=str, default='latest.pth', help='Encoder 路径')
    parser.add_argument('--flow_path', type=str, 
                        default='experiments/flow_matching/checkpoints/best_model.pt',
                        help='Flow Model 路径')
    parser.add_argument('--save_dir', type=str, default='experiments/flow_matching/predictions',
                        help='保存目录')
    parser.add_argument('--max_cycle', type=int, default=100, help='归一化用最大 cycle')
    parser.add_argument('--expected_life', type=int, default=54, help='预期寿命（用于 RUL 指示器）')
    parser.add_argument('--failure_threshold', type=float, default=1, help='失效阈值')
    parser.add_argument('--signal_length', type=int, default=3000, help='信号长度')
    parser.add_argument('--no_show', action='store_true', help='不显示图片')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载模型...")
    encoder, normalize_mean, normalize_std = load_encoder(args.encoder_path, device)
    model = load_flow_model(args.flow_path, encoder, device)
    scorer = PretrainedScorer(encoder, device)
    
    # 获取潜向量
    z_0 = None
    
    if args.embedding:
        # 方式 1: 加载已有 embedding
        print(f"加载 embedding: {args.embedding}")
        z_0 = torch.load(args.embedding, map_location=device, weights_only=False)
        if z_0.dim() == 1:
            z_0 = z_0.unsqueeze(0)
    
    elif args.after_signal and args.before_signal:
        # 方式 2: 从信号文件编码
        print(f"加载信号文件...")
        print(f"  After: {args.after_signal}")
        print(f"  Before: {args.before_signal}")
        
        after_signal = load_signal_from_csv(args.after_signal, args.signal_length)
        before_signal = load_signal_from_csv(args.before_signal, args.signal_length)
        
        print("编码信号...")
        z_0 = encode_signal(encoder, after_signal, before_signal, 
                           normalize_mean, normalize_std, device)
    
    elif args.after_dir and args.before_dir:
        # 方式 3: 从目录自动找最新文件
        after_dir = Path(args.after_dir)
        before_dir = Path(args.before_dir)
        
        # 找最新的 CSV 文件
        after_files = sorted(after_dir.glob('*.csv'))
        before_files = sorted(before_dir.glob('*.csv'))
        
        if not after_files or not before_files:
            print("错误: 目录中没有找到 CSV 文件")
            return
        
        after_path = after_files[-1]  # 最新的
        before_path = before_files[-1]
        
        print(f"找到信号文件:")
        print(f"  After: {after_path}")
        print(f"  Before: {before_path}")
        
        after_signal = load_signal_from_csv(after_path, args.signal_length)
        before_signal = load_signal_from_csv(before_path, args.signal_length)
        
        print("编码信号...")
        z_0 = encode_signal(encoder, after_signal, before_signal,
                           normalize_mean, normalize_std, device)
    
    else:
        print("错误: 请提供以下输入之一:")
        print("  --embedding <path>")
        print("  --after_signal <path> --before_signal <path>")
        print("  --after_dir <path> --before_dir <path>")
        return
    
    # 当前状态评估
    current_score = scorer.score(z_0)
    print(f"\n当前状态:")
    print(f"  Cycle: {args.current_cycle}")
    print(f"  退化评分: {current_score:.4f}")
    
    if current_score < 0.33:
        status = "健康"
    elif current_score < 0.67:
        status = "轻度退化"
    elif current_score < 1:
        status = "中度退化"
    else:
        status = "严重退化/接近失效"
    print(f"  状态评估: {status}")
    
    # 预测未来轨迹
    print("\n预测未来轨迹...")
    trajectory, cycles = predict_lifecycle(
        model, z_0,
        current_cycle=args.current_cycle,
        max_cycle=args.max_cycle,
        end_cycle=args.expected_life,
        device=device
    )
    
    # 计算健康评分
    scores = scorer.score_trajectory(trajectory)
    
    # 找到失效点
    failure_cycle, failure_idx = find_failure_point(
        scores, cycles, threshold=args.failure_threshold
    )
    
    if failure_cycle:
        rul = failure_cycle - args.current_cycle
        print(f"\n预测结果:")
        print(f"  预测失效 Cycle: {failure_cycle}")
        print(f"  剩余使用寿命 (RUL): {rul} cycles")
    else:
        rul = args.expected_life - args.current_cycle
        print(f"\n预测结果:")
        print(f"  在预期寿命内未达到失效阈值")
        print(f"  最低剩余寿命: {rul} cycles")
    
    # 绘图
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'new_battery_cycle_{args.current_cycle}.png'
    
    plot_prediction(
        cycles, scores,
        current_cycle=args.current_cycle,
        failure_cycle=failure_cycle,
        rul=rul,
        failure_threshold=args.failure_threshold,
        true_end_of_life=args.expected_life,
        title_prefix=f"新电池 (Cycle {args.current_cycle})",
        save_path=save_path,
        show=not args.no_show
    )
    
    # 返回结果（方便 API 调用）
    return {
        'current_cycle': args.current_cycle,
        'current_score': current_score,
        'failure_cycle': failure_cycle,
        'rul': rul,
        'cycles': cycles,
        'scores': scores,
        'embedding': z_0.cpu()
    }


if __name__ == '__main__':
    main()
