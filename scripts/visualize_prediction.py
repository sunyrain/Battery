#!/usr/bin/env python
"""
电池退化预测可视化脚本

生成图表：
1. 电池退化轨迹预测（左图）
2. RUL 剩余寿命指示器（右图）

使用方法:
    # 从指定 cycle 开始预测
    python scripts/visualize_prediction.py --start_cycle 20
    
    # 从缓存中的样本预测
    python scripts/visualize_prediction.py --start_cycle 10 --save_dir outputs
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.finetuneV6_fourcls import DeltaBatteryModel
from src.flow_matching.models.flow_model import BatteryFlowModel, BatteryFlowConfig


def load_models(encoder_path, flow_path, device):
    """加载 Encoder 和 Flow Model"""
    
    # 1. 加载 Encoder
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
    
    # 2. 加载 Flow Model
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
    
    return encoder, model


class PretrainedScorer:
    """使用预训练 output_head 计算退化评分"""
    
    def __init__(self, encoder, device):
        self.output_head = encoder.output_head
        self.device = device
        self.weights = torch.tensor([0, 0.33, 0.67, 1.0], device=device)
    
    @torch.no_grad()
    def score(self, z):
        """计算单个潜向量的退化评分"""
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


def predict_trajectory(model, z_0, start_cycle, end_cycle, max_cycle=100, device='cuda'):
    """预测从 start_cycle 到 end_cycle 的轨迹"""
    
    # 归一化时间
    t_start = (start_cycle - 1) / (max_cycle - 1)
    t_end = (end_cycle - 1) / (max_cycle - 1)
    
    num_steps = end_cycle - start_cycle + 1
    t_span = torch.linspace(t_start, t_end, num_steps, device=device)
    
    with torch.no_grad():
        trajectory = model.predict_trajectory(z_0, t_span)
    
    cycles = np.linspace(start_cycle, end_cycle, num_steps)
    
    return trajectory, cycles


def plot_prediction(cycles, scores, current_cycle, failure_cycle, rul, 
                    failure_threshold=1, true_end_of_life=54, save_path=None):
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
    ax1.axvline(x=failure_cycle, color='red', linestyle='--', 
                linewidth=2, label=f'预测失效（Cycle {failure_cycle}）')
    
    # 填充危险区域 (score >= threshold)
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
    ax1.set_title('电池退化轨迹预测', fontsize=14, fontweight='bold')
    ax1.set_xlim(current_cycle - 1, max(cycles) + 1)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ==================== 右图: RUL 指示器 ====================
    ax2 = axes[1]
    
    # 定义区域
    danger_zone = true_end_of_life * 0.3   # 危险: 0-30%
    warning_zone = true_end_of_life * 0.7  # 警告: 30%-70%
    # 健康: 70%-100%
    
    # 绘制背景色条
    ax2.axvspan(0, danger_zone, alpha=0.6, color='#FF6B6B', label='危险')
    ax2.axvspan(danger_zone, warning_zone, alpha=0.6, color='#FFE66D', label='警告')
    ax2.axvspan(warning_zone, true_end_of_life, alpha=0.6, color='#88D498', label='健康')
    
    # 绘制 RUL 指示线
    ax2.axvline(x=rul, color='blue', linewidth=4, label=f'RUL = {rul}')
    
    # 标注
    ax2.text(danger_zone / 2, 0.5, '危险', fontsize=14, ha='center', va='center',
             color='darkred', fontweight='bold', transform=ax2.get_xaxis_transform())
    ax2.text((danger_zone + warning_zone) / 2, 0.5, '警告', fontsize=14, ha='center', va='center',
             color='#996600', fontweight='bold', transform=ax2.get_xaxis_transform())
    ax2.text((warning_zone + true_end_of_life) / 2, 0.5, '健康', fontsize=14, ha='center', va='center',
             color='darkgreen', fontweight='bold', transform=ax2.get_xaxis_transform())
    
    ax2.set_xlabel('剩余使用寿命（Cycles）', fontsize=12)
    ax2.set_xlim(0, true_end_of_life)
    ax2.set_ylim(0, 1)
    ax2.set_title(f'RUL = {rul} cycles', fontsize=14, fontweight='bold')
    ax2.set_yticks([])  # 隐藏 Y 轴
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="电池退化预测可视化")
    parser.add_argument('--start_cycle', type=int, default=20, help='起始 cycle')
    parser.add_argument('--encoder_path', type=str, default='latest.pth', help='Encoder 路径')
    parser.add_argument('--flow_path', type=str, 
                        default='experiments/flow_matching/checkpoints/best_model.pt',
                        help='Flow Model 路径')
    parser.add_argument('--cache_path', type=str,
                        default='experiments/flow_matching/latent_cache_correct/latent_vectors.pt',
                        help='潜空间缓存路径')
    parser.add_argument('--save_dir', type=str, default='experiments/flow_matching/predictions',
                        help='保存目录')
    parser.add_argument('--max_cycle', type=int, default=100, help='最大 cycle（归一化用）')
    parser.add_argument('--true_end_of_life', type=int, default=54, help='电池真实寿命')
    parser.add_argument('--failure_threshold', type=float, default=1, help='失效阈值')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载模型...")
    encoder, model = load_models(args.encoder_path, args.flow_path, device)
    scorer = PretrainedScorer(encoder, device)
    
    # 加载缓存
    print("加载潜空间缓存...")
    cache_data = torch.load(args.cache_path, weights_only=False)
    latent_vectors = cache_data['latent_vectors']
    cycles = cache_data['cycles']
    
    # 找到起始 cycle 对应的潜向量
    start_cycle = args.start_cycle
    idx = (cycles == start_cycle).nonzero()
    if len(idx) == 0:
        # 找最近的
        idx = (torch.abs(cycles - start_cycle)).argmin().item()
        start_cycle = cycles[idx].item()
        print(f"未找到 cycle {args.start_cycle}，使用最近的 cycle {start_cycle}")
    else:
        idx = idx[0].item()
    
    z_0 = latent_vectors[idx:idx+1].to(device)
    print(f"起始 cycle: {start_cycle}")
    
    # 预测轨迹
    print("预测轨迹...")
    trajectory, pred_cycles = predict_trajectory(
        model, z_0, 
        start_cycle=start_cycle, 
        end_cycle=args.true_end_of_life,
        max_cycle=args.max_cycle,
        device=device
    )
    
    # 计算健康评分
    scores = scorer.score_trajectory(trajectory)
    
    # 找到失效点
    failure_idx = None
    for i, s in enumerate(scores):
        if s >= args.failure_threshold:
            failure_idx = i
            break
    
    if failure_idx is not None:
        failure_cycle = int(pred_cycles[failure_idx])
    else:
        failure_cycle = args.true_end_of_life
    
    rul = failure_cycle - start_cycle
    print(f"预测失效 cycle: {failure_cycle}")
    print(f"RUL: {rul} cycles")
    
    # 绘图
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'prediction_cycle_{start_cycle}.png'
    
    plot_prediction(
        pred_cycles, scores, 
        current_cycle=start_cycle,
        failure_cycle=failure_cycle,
        rul=rul,
        failure_threshold=args.failure_threshold,
        true_end_of_life=args.true_end_of_life,
        save_path=save_path
    )


if __name__ == '__main__':
    main()
