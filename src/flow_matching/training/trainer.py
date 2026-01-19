"""
Flow Matching 训练器

提供完整的训练流程，包括:
- 训练循环
- 验证评估
- 检查点管理
- 日志记录
- EMA (指数移动平均)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
import logging
import time
import copy

from ..models.flow_model import BatteryFlowModel
from ..core.flow_matching_loss import FlowMatchingLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """训练器配置"""
    
    # 基本设置
    num_epochs: int = 100
    
    # 优化器
    optimizer_type: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    
    # 学习率调度
    scheduler_type: str = "cosine"  # cosine, linear, constant, warmup_cosine
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # 梯度裁剪
    gradient_clip_enabled: bool = True
    gradient_clip_max_norm: float = 1.0
    
    # EMA
    ema_enabled: bool = True
    ema_decay: float = 0.9999
    
    # 检查点
    checkpoint_dir: str = "experiments/flow_matching/checkpoints"
    save_every: int = 10
    keep_last: int = 5
    
    # 日志
    log_dir: str = "experiments/flow_matching/logs"
    log_every: int = 100
    
    # 设备
    device: str = "cuda"
    mixed_precision: bool = True
    
    # 评估
    eval_every: int = 5
    
    # 随机种子
    seed: int = 42


class EMAModel:
    """
    指数移动平均模型
    
    维护模型参数的 EMA 版本，用于推理时获得更稳定的结果。
    
    Args:
        model: 原始模型
        decay: 衰减率
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 复制参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        """更新 EMA 参数"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self, model: nn.Module):
        """将 EMA 参数应用到模型（备份原参数）"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        """恢复原始参数"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow.copy()
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.shadow = state_dict.copy()


class FlowMatchingTrainer:
    """
    Flow Matching 训练器
    
    Args:
        model: BatteryFlowModel 实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（可选）
        config: 训练配置
        callbacks: 回调函数列表
    """
    
    def __init__(
        self,
        model: BatteryFlowModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainerConfig] = None,
        callbacks: Optional[List[Callable]] = None,
    ):
        self.config = config or TrainerConfig()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks or []
        
        # 设备
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # EMA
        self.ema = None
        if self.config.ema_enabled:
            self.ema = EMAModel(self.model, self.config.ema_decay)
        
        # 混合精度
        self.scaler = None
        if self.config.mixed_precision and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler()
        
        # 状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # 创建目录
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("FlowMatchingTrainer 初始化完成")
        logger.info(f"  - 设备: {self.device}")
        logger.info(f"  - 训练样本: {len(train_loader.dataset)}")
        if val_loader:
            logger.info(f"  - 验证样本: {len(val_loader.dataset)}")
        logger.info(f"  - 混合精度: {self.config.mixed_precision}")
        logger.info(f"  - EMA: {self.config.ema_enabled}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        # 只优化需要梯度的参数
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.config.optimizer_type == "adamw":
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
            )
        elif self.config.optimizer_type == "adam":
            return optim.Adam(
                params,
                lr=self.config.learning_rate,
                betas=self.config.betas,
            )
        elif self.config.optimizer_type == "sgd":
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"未知的优化器类型: {self.config.optimizer_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        total_steps = len(self.train_loader) * self.config.num_epochs
        warmup_steps = len(self.train_loader) * self.config.warmup_epochs
        
        if self.config.scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.config.min_lr,
            )
        elif self.config.scheduler_type == "linear":
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.min_lr / self.config.learning_rate,
                total_iters=total_steps,
            )
        elif self.config.scheduler_type == "warmup_cosine":
            # 组合 warmup + cosine
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    return self.config.min_lr / self.config.learning_rate + \
                           0.5 * (1 - self.config.min_lr / self.config.learning_rate) * \
                           (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif self.config.scheduler_type == "constant":
            return None
        else:
            raise ValueError(f"未知的调度器类型: {self.config.scheduler_type}")
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        
        # 移动数据到设备
        if 'z_0' in batch:
            # 潜空间数据集
            z_0 = batch['z_0'].to(self.device)
            z_1 = batch['z_1'].to(self.device)
            
            # 计算损失
            with torch.amp.autocast(device_type='cuda', enabled=self.scaler is not None):
                loss_dict = self.model.compute_loss(z_0, z_1)
                loss = loss_dict['loss']
        else:
            # 信号数据集
            signal_after_0 = batch['signal_after_0'].to(self.device)
            signal_before_0 = batch['signal_before_0'].to(self.device)
            signal_after_1 = batch['signal_after_1'].to(self.device)
            signal_before_1 = batch['signal_before_1'].to(self.device)
            
            with torch.amp.autocast(device_type='cuda', enabled=self.scaler is not None):
                loss_dict = self.model(
                    signal_after_0, signal_before_0,
                    signal_after_1, signal_before_1,
                )
                loss = loss_dict['loss']
        
        # 反向传播
        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            
            if self.config.gradient_clip_enabled:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_max_norm,
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            if self.config.gradient_clip_enabled:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_max_norm,
                )
            
            self.optimizer.step()
        
        # 更新学习率
        if self.scheduler is not None:
            self.scheduler.step()
        
        # 更新 EMA
        if self.ema is not None:
            self.ema.update(self.model)
        
        # 返回损失
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        # 如果有 EMA，使用 EMA 参数
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
        
        total_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            if 'z_0' in batch:
                z_0 = batch['z_0'].to(self.device)
                z_1 = batch['z_1'].to(self.device)
                loss_dict = self.model.compute_loss(z_0, z_1)
            else:
                signal_after_0 = batch['signal_after_0'].to(self.device)
                signal_before_0 = batch['signal_before_0'].to(self.device)
                signal_after_1 = batch['signal_after_1'].to(self.device)
                signal_before_1 = batch['signal_before_1'].to(self.device)
                loss_dict = self.model(
                    signal_after_0, signal_before_0,
                    signal_after_1, signal_before_1,
                )
            
            total_loss += loss_dict['loss'].item()
            num_batches += 1
        
        # 恢复原始参数
        if self.ema is not None:
            self.ema.restore(self.model)
        
        return {'val_loss': total_loss / max(num_batches, 1)}
    
    def _save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = Path(self.config.checkpoint_dir) / filename
        torch.save(checkpoint, path)
        logger.info(f"检查点已保存: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'ema_state_dict' in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"检查点已加载: {path}, epoch={self.current_epoch}")
    
    def train(self):
        """完整训练流程"""
        logger.info(f"开始训练，共 {self.config.num_epochs} 个 epoch")
        
        # 调用回调
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin(self)
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 调用回调
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_begin'):
                    callback.on_epoch_begin(self, epoch)
            
            # 训练一个 epoch
            epoch_losses = []
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                loss_dict = self._train_step(batch)
                epoch_losses.append(loss_dict['loss'])
                self.global_step += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss_dict['loss']:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                })
                
                # 日志
                if self.global_step % self.config.log_every == 0:
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_step'):
                            callback.on_step(self, self.global_step, loss_dict)
            
            # Epoch 结束
            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            epoch_time = time.time() - epoch_start_time
            
            # 验证
            val_metrics = {}
            if (epoch + 1) % self.config.eval_every == 0 and self.val_loader is not None:
                val_metrics = self._validate()
                
                if val_metrics.get('val_loss', float('inf')) < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self._save_checkpoint('best_model.pt')
            
            # 日志
            val_loss = val_metrics.get('val_loss')
            val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else "N/A"
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Loss: {epoch_loss:.4f} - "
                f"Val Loss: {val_loss_str} - "
                f"Time: {epoch_time:.1f}s"
            )
            
            # 保存检查点
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
            
            # 调用回调
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(self, epoch, {'loss': epoch_loss, **val_metrics})
        
        # 训练结束
        self._save_checkpoint('final_model.pt')
        
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(self)
        
        logger.info("训练完成！")
    
    def get_ema_model(self) -> Optional[BatteryFlowModel]:
        """获取 EMA 模型"""
        if self.ema is None:
            return None
        
        # 创建模型副本
        ema_model = copy.deepcopy(self.model)
        
        # 应用 EMA 参数
        for name, param in ema_model.named_parameters():
            if name in self.ema.shadow:
                param.data = self.ema.shadow[name].clone()
        
        return ema_model
