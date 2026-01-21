"""
Flow Matching 训练脚本

核心设计:
- t ∈ [0, 1] 对应 cycle ∈ [1, max_cycle]
- max_cycle 从配置文件读取（默认 54，与训练数据一致）
- Flow Matching 学习 z(t) 的演化
- 用 SelfLearning 的 output_head + beta 判断失效

使用方法:
    # 重新计算缓存并训练
    python scripts/train_flow.py --config configs/flow_matching_config.yaml --compute_cache --epochs 100
    
    # 使用已有缓存训练
    python scripts/train_flow.py --config configs/flow_matching_config.yaml --epochs 100
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import logging
import yaml
from tqdm import tqdm
from typing import List, Tuple, Dict
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.smartwavev9 import DeltaBatteryModel
from src.flow_matching.models.flow_model import BatteryFlowModel, BatteryFlowConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# 数据集
# ============================================================
class CyclePairDataset(Dataset):
    """Cycle 配对数据集"""
    
    def __init__(
        self,
        latent_vectors: torch.Tensor,
        cycles: torch.Tensor,
        max_cycle: int,
        min_cycle_gap: int = 5,
        max_cycle_gap: int = None,
    ):
        self.latent_vectors = latent_vectors
        self.cycles = cycles
        self.max_cycle = max_cycle
        self.min_cycle_gap = min_cycle_gap
        self.max_cycle_gap = max_cycle_gap or max_cycle
        
        self.pairs = self._build_pairs()
        logger.info(f"数据集: {len(self.pairs)} 对")
        logger.info(f"  - Cycle 范围: [{cycles.min().item()}, {cycles.max().item()}]")
        logger.info(f"  - Gap 范围: [{min_cycle_gap}, {self.max_cycle_gap}]")
    
    def _build_pairs(self) -> List[Tuple[int, int]]:
        pairs = []
        n = len(self.cycles)
        
        for i in range(n):
            cycle_i = self.cycles[i].item()
            for j in range(n):
                if i == j:
                    continue
                cycle_j = self.cycles[j].item()
                gap = cycle_j - cycle_i
                if self.min_cycle_gap <= gap <= self.max_cycle_gap:
                    pairs.append((i, j))
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        idx_0, idx_1 = self.pairs[idx]
        return {
            'z_0': self.latent_vectors[idx_0],
            'z_1': self.latent_vectors[idx_1],
            'cycle_0': self.cycles[idx_0],
            'cycle_1': self.cycles[idx_1],
        }


# ============================================================
# 辅助函数
# ============================================================
def cycle_to_t(cycle: torch.Tensor, max_cycle: int) -> torch.Tensor:
    """cycle → t: t = (cycle - 1) / (max_cycle - 1)"""
    return (cycle.float() - 1) / (max_cycle - 1)


def load_encoder(checkpoint_path: str, device: torch.device):
    """加载 Encoder 和相关参数"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']
    config = ckpt.get('config', {})
    
    # 创建 Encoder
    encoder = DeltaBatteryModel(
        input_dim=1, d_model=256, nhead=4, num_layers=1, ROPE_max_len=5000,
        num_classes=4, task_type='classification', max_level=6, wavelet='sym4',
        patch_size=10, stride=10, dropout=0.1
    )
    
    # 预创建频域投影层
    for key in state_dict.keys():
        if key.startswith('freq_branch.channel_projections.') and key.endswith('.weight'):
            name = key.split('.')[2]
            weight = state_dict[key]
            encoder.freq_branch.channel_projections[name] = nn.Linear(weight.shape[1], weight.shape[0])
    
    encoder.load_state_dict(state_dict, strict=False)
    encoder.to(device)
    encoder.eval()
    
    # 提取归一化参数
    normalize_params = {
        'mean': config.get('data', {}).get('li_cu_mean', 0.0),
        'std': config.get('data', {}).get('li_cu_std', 1.0),
    }
    
    return encoder, state_dict, normalize_params


def compute_cache(
    encoder: nn.Module,
    csv_path: str,
    normalize_params: dict,
    cache_dir: str,
    device: torch.device,
):
    """计算潜空间缓存"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    logger.info(f"计算缓存: {len(df)} 个样本")
    
    mean, std = normalize_params['mean'], normalize_params['std']
    
    def load_signal(path, length=3000):
        data = pd.read_csv(path, header=None)
        signal = data.iloc[:, 1].values
        if len(signal) < length:
            signal = np.pad(signal, (0, length - len(signal)), mode='constant')
        else:
            signal = signal[:length]
        signal = (signal - mean) / (std + 1e-8)
        return torch.tensor(signal, dtype=torch.float32)
    
    all_z = []
    all_cycles = []
    
    with torch.no_grad():
        for i in tqdm(range(len(df)), desc="Computing cache"):
            row = df.iloc[i]
            before = load_signal(row['before_path']).unsqueeze(0).unsqueeze(0).to(device)
            after = load_signal(row['after_path']).unsqueeze(0).unsqueeze(0).to(device)
            
            _, _, _, z = encoder(after, before)
            all_z.append(z.cpu())
            all_cycles.append(row['Cycle'])
    
    z_tensor = torch.cat(all_z, dim=0)
    cycles_tensor = torch.tensor(all_cycles)
    
    torch.save({
        'latent_vectors': z_tensor,
        'cycles': cycles_tensor,
        'indices': torch.arange(len(df)),
    }, cache_dir / 'latent_vectors.pt')
    
    with open(cache_dir / 'metadata.json', 'w') as f:
        json.dump({
            'csv_path': csv_path,
            'num_samples': len(df),
            'latent_dim': z_tensor.shape[1],
            'min_cycle': int(cycles_tensor.min()),
            'max_cycle': int(cycles_tensor.max()),
            'normalize_mean': mean,
            'normalize_std': std,
        }, f, indent=2)
    
    logger.info(f"缓存已保存: {cache_dir}")
    return z_tensor, cycles_tensor


def load_cache(cache_dir: str):
    """加载缓存"""
    cache_dir = Path(cache_dir)
    data = torch.load(cache_dir / 'latent_vectors.pt', weights_only=False)
    with open(cache_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    return data['latent_vectors'], data['cycles'], metadata


# ============================================================
# 训练
# ============================================================
def train_epoch(
    flow_model: BatteryFlowModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_cycle: int,
    gradient_clip: float = 1.0,
):
    flow_model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        z_0 = batch['z_0'].to(device)
        z_1 = batch['z_1'].to(device)
        cycle_0 = batch['cycle_0'].to(device)
        cycle_1 = batch['cycle_1'].to(device)
        
        # cycle → t
        t_0 = cycle_to_t(cycle_0, max_cycle)
        t_1 = cycle_to_t(cycle_1, max_cycle)
        
        # 随机插值
        alpha = torch.rand(z_0.size(0), 1, device=device)
        t_interp = t_0.unsqueeze(1) + alpha * (t_1.unsqueeze(1) - t_0.unsqueeze(1))
        t_interp = t_interp.squeeze(1)
        
        z_t = (1 - alpha) * z_0 + alpha * z_1
        
        # 真实速度
        dt = (t_1 - t_0).unsqueeze(1) + 1e-8
        v_true = (z_1 - z_0) / dt
        
        # 预测速度
        v_pred = flow_model.velocity_net(z_t, t_interp)
        
        # 损失
        loss = F.mse_loss(v_pred, v_true)
        
        optimizer.zero_grad()
        loss.backward()
        
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), gradient_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f"{loss.item():.6f}", 'avg': f"{total_loss/num_batches:.6f}"})
    
    return total_loss / num_batches


def validate(flow_model, dataloader, device, max_cycle):
    flow_model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            z_0 = batch['z_0'].to(device)
            z_1 = batch['z_1'].to(device)
            cycle_0 = batch['cycle_0'].to(device)
            cycle_1 = batch['cycle_1'].to(device)
            
            t_0 = cycle_to_t(cycle_0, max_cycle)
            t_1 = cycle_to_t(cycle_1, max_cycle)
            
            alpha = torch.rand(z_0.size(0), 1, device=device)
            t_interp = t_0.unsqueeze(1) + alpha * (t_1.unsqueeze(1) - t_0.unsqueeze(1))
            t_interp = t_interp.squeeze(1)
            
            z_t = (1 - alpha) * z_0 + alpha * z_1
            dt = (t_1 - t_0).unsqueeze(1) + 1e-8
            v_true = (z_1 - z_0) / dt
            v_pred = flow_model.velocity_net(z_t, t_interp)
            
            loss = F.mse_loss(v_pred, v_true)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Flow Matching 训练")
    parser.add_argument('--config', type=str, default='configs/flow_matching_config.yaml')
    parser.add_argument('--compute_cache', action='store_true', help='重新计算缓存')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"设备: {device}")
    
    # 配置参数
    max_cycle = config['model']['max_cycle']
    encoder_path = config['encoder']['checkpoint_path']
    csv_path = config['data']['dataset']['csv_path']
    cache_dir = config['data']['cache']['cache_dir']
    
    batch_size = args.batch_size or config['training']['batch_size']
    num_epochs = args.epochs or config['training']['num_epochs']
    lr = args.lr or config['training']['optimizer']['lr']
    
    logger.info(f"max_cycle = {max_cycle}")
    
    # 加载 Encoder
    logger.info(f"加载 Encoder: {encoder_path}")
    encoder, state_dict, normalize_params = load_encoder(encoder_path, device)
    
    # 缓存
    cache_path = Path(cache_dir) / 'latent_vectors.pt'
    if args.compute_cache or not cache_path.exists():
        logger.info("计算潜空间缓存...")
        latent_vectors, cycles = compute_cache(
            encoder, csv_path, normalize_params, cache_dir, device
        )
    else:
        logger.info(f"加载缓存: {cache_dir}")
        latent_vectors, cycles, metadata = load_cache(cache_dir)
        logger.info(f"  样本数: {len(latent_vectors)}, cycle 范围: [{metadata['min_cycle']}, {metadata['max_cycle']}]")
    
    # Flow Model
    logger.info("创建 Flow Model")
    flow_config = BatteryFlowConfig(
        latent_dim=256,
        hidden_dim=config['model']['velocity_net']['hidden_dim'],
        time_embed_dim=config['model']['velocity_net']['time_embed_dim'],
        cond_embed_dim=config['model']['velocity_net']['cond_embed_dim'],
        num_layers=config['model']['velocity_net']['num_layers'],
        dropout=config['model']['velocity_net']['dropout'],
        solver_type=config['model']['solver']['type'],
        lightweight=config['model']['lightweight'],
        max_cycle=max_cycle,
    )
    flow_model = BatteryFlowModel(config=flow_config)
    flow_model.set_encoder(encoder)
    flow_model = flow_model.to(device)
    
    # 数据集
    dataset = CyclePairDataset(
        latent_vectors=latent_vectors,
        cycles=cycles,
        max_cycle=max_cycle,
        min_cycle_gap=config['data']['dataset']['pairing']['min_cycle_gap'],
        max_cycle_gap=config['data']['dataset']['pairing']['max_cycle_gap'],
    )
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"训练: {len(train_dataset)}, 验证: {len(val_dataset)}")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        flow_model.velocity_net.parameters(),
        lr=lr,
        weight_decay=config['training']['optimizer']['weight_decay'],
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=config['training']['scheduler']['min_lr']
    )
    
    # 检查点目录
    checkpoint_dir = Path(config['training']['checkpoint']['save_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练
    best_val_loss = float('inf')
    
    logger.info("=" * 60)
    logger.info("开始训练")
    logger.info(f"  t ∈ [0, 1] ↔ cycle ∈ [1, {max_cycle}]")
    logger.info(f"  Epochs: {num_epochs}, Batch size: {batch_size}, LR: {lr}")
    logger.info("=" * 60)
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(flow_model, train_loader, optimizer, device, epoch, max_cycle)
        val_loss = validate(flow_model, val_loader, device, max_cycle)
        scheduler.step()
        
        logger.info(f"Epoch {epoch}/{num_epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': flow_model.state_dict(),
                'ema_state_dict': flow_model.state_dict(),  # 兼容性
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'latent_dim': 256,
                    'hidden_dim': config['model']['velocity_net']['hidden_dim'],
                    'max_cycle': max_cycle,
                },
            }, checkpoint_dir / 'best_model.pt')
            logger.info(f"  ✓ 保存最佳模型 (val_loss={val_loss:.6f})")
        
        if epoch % config['training']['checkpoint']['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': flow_model.state_dict(),
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
    
    logger.info("=" * 60)
    logger.info(f"训练完成! 最佳验证损失: {best_val_loss:.6f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
