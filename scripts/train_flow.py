#!/usr/bin/env python
"""
Battery Flow Matching 训练脚本

使用方法:
    python scripts/train_flow.py --config configs/flow_matching_config.yaml
    python scripts/train_flow.py --config configs/flow_matching_config.yaml --resume checkpoints/last.pt
"""

import os
import sys
import argparse
import logging
import random
import numpy as np
import torch

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.flow_matching.models.flow_model import BatteryFlowModel, BatteryFlowConfig
from src.flow_matching.data.dataset import BatteryFlowDataset, create_flow_dataloader
from src.flow_matching.data.preprocessing import SignalProcessor
from src.flow_matching.data.latent_cache import LatentCache
from src.flow_matching.training.trainer import FlowMatchingTrainer, TrainerConfig
from src.flow_matching.training.callbacks import (
    CheckpointCallback,
    LoggingCallback,
    EarlyStoppingCallback,
    VisualizationCallback,
)
from src.flow_matching.utils.config import load_config, FlowMatchingConfig

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_encoder(checkpoint_path: str, device: torch.device):
    """加载 SmartWave Encoder"""
    from src.smartwavev9 import DeltaBatteryModel
    
    logger.info(f"加载 Encoder: {checkpoint_path}")
    
    # 加载检查点以获取模型配置
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 从检查点获取模型配置
    if 'config' in checkpoint and 'model' in checkpoint['config']:
        model_cfg = checkpoint['config']['model']['model_config']
        logger.info(f"从检查点加载模型配置: d_model={model_cfg['d_model']}, patch_size={model_cfg['patch_size']}")
    else:
        # 默认配置
        model_cfg = {
            'input_dim': 1,
            'd_model': 256,
            'nhead': 4,
            'num_layers': 1,
            'ROPE_max_len': 5000,
            'num_classes': 4,
            'dropout': 0.2,
            'task_type': 'classification',
            'max_level': 6,
            'wavelet': 'sym4',
            'patch_size': 10,
            'stride': 10,
        }
    
    # 创建模型
    encoder = DeltaBatteryModel(
        input_dim=model_cfg.get('input_dim', 1),
        d_model=model_cfg.get('d_model', 256),
        nhead=model_cfg.get('nhead', 4),
        num_layers=model_cfg.get('num_layers', 1),
        ROPE_max_len=model_cfg.get('ROPE_max_len', 5000),  # 添加 ROPE_max_len
        num_classes=model_cfg.get('num_classes', 4),
        task_type=model_cfg.get('task_type', 'classification'),
        max_level=model_cfg.get('max_level', 6),
        wavelet=model_cfg.get('wavelet', 'sym4'),
        patch_size=model_cfg.get('patch_size', 10),
        stride=model_cfg.get('stride', 10),
        dropout=model_cfg.get('dropout', 0.2),
    )
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 处理可能的 "model." 前缀
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
    
    # 预先创建 freq_branch.channel_projections 层（这些是动态创建的，需要从检查点恢复）
    for key in state_dict.keys():
        if key.startswith("freq_branch.channel_projections.") and key.endswith(".weight"):
            # 解析名称: freq_branch.channel_projections.cA6.weight -> cA6
            name = key.split(".")[2]
            weight = state_dict[key]
            in_features = weight.shape[1]
            out_features = weight.shape[0]
            # 创建对应的 Linear 层
            encoder.freq_branch.channel_projections[name] = torch.nn.Linear(in_features, out_features)
            logger.info(f"预创建投影层: {name} ({in_features} -> {out_features})")
    
    # 过滤掉不需要的键（旧版本模型的参数）
    keys_to_ignore = {'beta1', 'beta2_offset'}
    filtered_state_dict = {k: v for k, v in state_dict.items() if k not in keys_to_ignore}
    if len(state_dict) != len(filtered_state_dict):
        logger.info(f"过滤掉 {len(state_dict) - len(filtered_state_dict)} 个旧版本参数: {keys_to_ignore & set(state_dict.keys())}")
    
    # 使用 strict=False 允许缺失/额外的键
    missing, unexpected = encoder.load_state_dict(filtered_state_dict, strict=False)
    if missing:
        logger.warning(f"缺失的键: {len(missing)} 个")
        for k in missing[:5]:  # 只显示前5个
            logger.warning(f"  - {k}")
    if unexpected:
        logger.warning(f"意外的键: {len(unexpected)} 个")
        for k in unexpected[:5]:  # 只显示前5个
            logger.warning(f"  - {k}")
    
    encoder.to(device)
    encoder.eval()
    
    logger.info("Encoder 加载完成")
    return encoder, model_cfg.get('d_model', 256)


def main():
    parser = argparse.ArgumentParser(description="Battery Flow Matching Training")
    parser.add_argument('--config', type=str, default='configs/flow_matching_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--use_cache', action='store_true',
                        help='使用潜空间缓存')
    parser.add_argument('--compute_cache', action='store_true',
                        help='计算潜空间缓存')
    args = parser.parse_args()
    
    # 加载配置
    logger.info(f"加载配置: {args.config}")
    config = load_config(args.config)
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载 Encoder
    encoder, encoder_dim = load_encoder(config.encoder_checkpoint, device)
    
    # 使用 encoder 的实际输出维度作为 latent_dim
    latent_dim = encoder_dim
    logger.info(f"Encoder 输出维度 (latent_dim): {latent_dim}")
    
    # 创建 Flow Model
    flow_config = BatteryFlowConfig(
        latent_dim=latent_dim,  # 使用 encoder 的实际输出维度
        hidden_dim=config.hidden_dim,
        time_embed_dim=config.time_embed_dim,
        cond_embed_dim=config.cond_embed_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        use_adaln=config.use_adaln,
        use_skip_connections=config.use_skip_connections,
        solver_type=config.solver_type,
        path_type=config.path_type,
        max_cycle=config.max_cycle,
        lightweight=config.lightweight,
    )
    
    model = BatteryFlowModel(flow_config, encoder)
    if config.freeze_encoder:
        model.freeze_encoder()
    
    logger.info(f"模型参数量: {model.get_num_trainable_params():,} (可训练), {model.get_num_total_params():,} (总计)")
    
    # 创建数据加载器
    signal_processor = SignalProcessor(signal_length=config.signal_length)
    
    if args.use_cache or args.compute_cache:
        # 使用潜空间缓存
        cache_dir = "experiments/flow_matching/latent_cache"
        cache = LatentCache(cache_dir, encoder, signal_processor)
        
        if args.compute_cache or not cache.is_cached:
            logger.info("计算潜空间缓存...")
            cache.compute_and_save(config.csv_path, config.data_root, device)
        
        latent_vectors, cycles, metadata = cache.load()
        
        from src.flow_matching.data.dataset import LatentPairDataset
        from torch.utils.data import DataLoader
        
        train_dataset = LatentPairDataset(
            latent_vectors, cycles,
            min_cycle_gap=config.min_cycle_gap,
            max_cycle_gap=config.max_cycle_gap,
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = None  # 可以创建验证集
    else:
        # 直接使用信号数据集
        train_loader = create_flow_dataloader(
            csv_path=config.csv_path,
            data_root=config.data_root,
            batch_size=config.batch_size,
            signal_processor=signal_processor,
            min_cycle_gap=config.min_cycle_gap,
            max_cycle_gap=config.max_cycle_gap,
            augmentation=True,
        )
        val_loader = None
    
    logger.info(f"训练数据: {len(train_loader.dataset)} 个样本")
    
    # 创建训练器配置
    trainer_config = TrainerConfig(
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_epochs=config.warmup_epochs,
        mixed_precision=config.mixed_precision,
        device=config.device,
    )
    
    # 创建回调
    callbacks = [
        CheckpointCallback(
            save_dir="experiments/flow_matching/checkpoints",
            save_every=10,
            keep_last=5,
        ),
        LoggingCallback(
            log_dir="experiments/flow_matching/logs",
            use_tensorboard=True,
        ),
        VisualizationCallback(
            save_dir="experiments/flow_matching/plots",
            plot_every=10,
        ),
    ]
    
    # 创建训练器
    trainer = FlowMatchingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        callbacks=callbacks,
    )
    
    # 恢复训练
    if args.resume:
        logger.info(f"恢复训练: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()
    
    logger.info("训练完成！")


if __name__ == "__main__":
    main()
