#!/usr/bin/env python
"""
Battery Flow Matching 推理脚本

使用方法:
    # 预测完整生命周期
    python scripts/inference_flow.py --mode lifecycle --checkpoint checkpoints/best_model.pt --signal_after data/after.csv --signal_before data/before.csv
    
    # 预测 RUL
    python scripts/inference_flow.py --mode rul --checkpoint checkpoints/best_model.pt --current_cycle 50
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.flow_matching.inference.predictor import LifecyclePredictor
from src.flow_matching.inference.visualizer import TrajectoryVisualizer
from src.flow_matching.data.preprocessing import SignalProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Battery Flow Matching Inference")
    parser.add_argument('--mode', type=str, choices=['lifecycle', 'rul', 'compare'],
                        default='lifecycle', help='推理模式')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--encoder_checkpoint', type=str, default='latest.pth',
                        help='Encoder 检查点路径')
    parser.add_argument('--signal_after', type=str, help='After 信号 CSV 路径')
    parser.add_argument('--signal_before', type=str, help='Before 信号 CSV 路径')
    parser.add_argument('--current_cycle', type=int, default=1, help='当前 cycle 数')
    parser.add_argument('--num_steps', type=int, default=100, help='预测步数')
    parser.add_argument('--output_dir', type=str, default='experiments/flow_matching/inference',
                        help='输出目录')
    parser.add_argument('--visualize', action='store_true', help='生成可视化')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载 Encoder
    from src.smartwavev9 import DeltaBatteryModel
    
    encoder = DeltaBatteryModel(
        input_dim=1,
        d_model=128,
        nhead=4,
        num_layers=2,
        num_classes=4,
        task_type="classification",
    )
    
    checkpoint = torch.load(args.encoder_checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
    encoder.load_state_dict(state_dict, strict=False)
    
    # 创建预测器
    predictor = LifecyclePredictor.from_checkpoint(
        args.checkpoint,
        encoder=encoder,
        device=device,
        latent_dim=128,
    )
    
    # 创建可视化器
    visualizer = TrajectoryVisualizer(save_dir=args.output_dir) if args.visualize else None
    
    # 加载信号
    if args.signal_after and args.signal_before:
        processor = SignalProcessor()
        signal_after = processor.process(args.signal_after).unsqueeze(0)
        signal_before = processor.process(args.signal_before).unsqueeze(0)
    else:
        # 使用示例数据
        logger.warning("未提供信号文件，使用随机数据作为演示")
        signal_after = torch.randn(1, 1, 3000)
        signal_before = torch.randn(1, 1, 3000)
    
    # 执行推理
    if args.mode == 'lifecycle':
        logger.info("预测完整生命周期...")
        result = predictor.predict_full_lifecycle(
            signal_after, signal_before, 
            num_steps=args.num_steps
        )
        
        logger.info(f"轨迹形状: {result['trajectory'].shape}")
        logger.info(f"健康评分范围: [{result['health_scores'].min():.3f}, {result['health_scores'].max():.3f}]")
        
        if visualizer:
            visualizer.plot_health_score_trajectory(
                result['cycles'],
                result['health_scores'].squeeze(),
                save_name='lifecycle_prediction.png',
                show=False,
            )
            visualizer.plot_latent_space_2d(
                result['trajectory'],
                save_name='latent_trajectory.png',
                show=False,
            )
            logger.info(f"可视化已保存到: {args.output_dir}")
    
    elif args.mode == 'rul':
        logger.info(f"预测 RUL (当前 cycle: {args.current_cycle})...")
        result = predictor.predict_rul(
            signal_after, signal_before,
            current_cycle=args.current_cycle,
        )
        
        logger.info(f"预测 RUL: {result['rul']} cycles")
        logger.info(f"预测失效 cycle: {result['failure_cycle']}")
        logger.info(f"置信度: {result['confidence']:.2%}")
        
        if visualizer:
            visualizer.plot_rul_prediction(
                result,
                save_name='rul_prediction.png',
                show=False,
            )
    
    logger.info("推理完成！")


if __name__ == "__main__":
    main()
