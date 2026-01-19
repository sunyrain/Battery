"""
配置管理模块
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class FlowMatchingConfig:
    """Flow Matching 完整配置"""
    
    # 模型配置
    latent_dim: int = 128
    hidden_dim: int = 512
    time_embed_dim: int = 128
    cond_embed_dim: int = 128
    num_layers: int = 6
    dropout: float = 0.1
    use_adaln: bool = True
    use_skip_connections: bool = True
    path_type: str = "linear"
    max_cycle: int = 200
    lightweight: bool = False
    
    # ODE 求解器
    solver_type: str = "dopri5"
    solver_rtol: float = 1e-5
    solver_atol: float = 1e-5
    
    # 数据配置
    data_root: str = "Data"
    csv_path: str = "Data/LiCu_10C-1/LiCu_10C_1.csv"
    signal_length: int = 3000
    min_cycle_gap: int = 10
    max_cycle_gap: int = 100
    
    # 训练配置
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    
    # 编码器配置
    encoder_checkpoint: str = "latest.pth"
    freeze_encoder: bool = True
    
    # 其他
    seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FlowMatchingConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def load_config(config_path: str) -> FlowMatchingConfig:
    """
    从 YAML 文件加载配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        FlowMatchingConfig 实例
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 展平嵌套结构
    flat_config = {}
    
    def flatten(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, dict):
                flatten(v, prefix + k + '_')
            else:
                # 尝试匹配配置字段
                key = k if prefix == '' else prefix.rstrip('_') + '_' + k
                flat_key = k  # 直接使用原始键名
                flat_config[flat_key] = v
    
    # 特殊处理嵌套配置
    if 'model' in config_dict:
        model_cfg = config_dict['model']
        flat_config.update({
            'latent_dim': model_cfg.get('latent_dim', 128),
            'path_type': model_cfg.get('path_type', 'linear'),
            'max_cycle': model_cfg.get('max_cycle', 200),
            'lightweight': model_cfg.get('lightweight', False),
        })
        if 'velocity_net' in model_cfg:
            vn = model_cfg['velocity_net']
            flat_config.update({
                'hidden_dim': vn.get('hidden_dim', 512),
                'time_embed_dim': vn.get('time_embed_dim', 128),
                'cond_embed_dim': vn.get('cond_embed_dim', 128),
                'num_layers': vn.get('num_layers', 6),
                'dropout': vn.get('dropout', 0.1),
                'use_adaln': vn.get('use_adaln', True),
                'use_skip_connections': vn.get('use_skip_connections', True),
            })
        if 'solver' in model_cfg:
            solver = model_cfg['solver']
            flat_config.update({
                'solver_type': solver.get('type', 'dopri5'),
                'solver_rtol': solver.get('rtol', 1e-5),
                'solver_atol': solver.get('atol', 1e-5),
            })
    
    if 'data' in config_dict:
        data_cfg = config_dict['data']
        flat_config.update({
            'data_root': data_cfg.get('data_root', 'Data'),
        })
        if 'dataset' in data_cfg:
            ds = data_cfg['dataset']
            flat_config.update({
                'csv_path': ds.get('csv_path', ''),
                'signal_length': ds.get('signal_length', 3000),
            })
            if 'pairing' in ds:
                pairing = ds['pairing']
                flat_config.update({
                    'min_cycle_gap': pairing.get('min_cycle_gap', 10),
                    'max_cycle_gap': pairing.get('max_cycle_gap', 100),
                })
    
    if 'training' in config_dict:
        train_cfg = config_dict['training']
        flat_config.update({
            'batch_size': train_cfg.get('batch_size', 32),
            'num_epochs': train_cfg.get('num_epochs', 100),
        })
        if 'optimizer' in train_cfg:
            opt = train_cfg['optimizer']
            flat_config.update({
                'learning_rate': opt.get('lr', 1e-4),
                'weight_decay': opt.get('weight_decay', 0.01),
            })
        if 'scheduler' in train_cfg:
            sched = train_cfg['scheduler']
            flat_config.update({
                'warmup_epochs': sched.get('warmup_epochs', 5),
            })
    
    if 'encoder' in config_dict:
        enc_cfg = config_dict['encoder']
        flat_config.update({
            'encoder_checkpoint': enc_cfg.get('checkpoint_path', 'latest.pth'),
            'freeze_encoder': enc_cfg.get('freeze', True),
        })
    
    if 'experiment' in config_dict:
        exp_cfg = config_dict['experiment']
        flat_config.update({
            'seed': exp_cfg.get('seed', 42),
            'device': exp_cfg.get('device', 'cuda'),
        })
        if 'mixed_precision' in exp_cfg:
            flat_config['mixed_precision'] = exp_cfg['mixed_precision'].get('enabled', True)
    
    return FlowMatchingConfig.from_dict(flat_config)


def save_config(config: FlowMatchingConfig, save_path: str):
    """
    保存配置到 YAML 文件
    
    Args:
        config: 配置对象
        save_path: 保存路径
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)
