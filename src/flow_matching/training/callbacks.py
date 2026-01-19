"""
训练回调模块

提供可插拔的训练回调，用于:
- 检查点保存
- 日志记录
- 早停
- 可视化
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from abc import ABC, abstractmethod
import logging
import json

logger = logging.getLogger(__name__)


class Callback(ABC):
    """回调基类"""
    
    def on_train_begin(self, trainer: Any):
        """训练开始"""
        pass
    
    def on_train_end(self, trainer: Any):
        """训练结束"""
        pass
    
    def on_epoch_begin(self, trainer: Any, epoch: int):
        """Epoch 开始"""
        pass
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]):
        """Epoch 结束"""
        pass
    
    def on_step(self, trainer: Any, step: int, logs: Dict[str, float]):
        """训练步结束"""
        pass


class CheckpointCallback(Callback):
    """
    检查点保存回调
    
    Args:
        save_dir: 保存目录
        save_every: 每 N 个 epoch 保存
        keep_last: 保留最近 N 个检查点
        save_best_only: 是否只保存最佳模型
    """
    
    def __init__(
        self,
        save_dir: str,
        save_every: int = 10,
        keep_last: int = 5,
        save_best_only: bool = False,
        monitor: str = 'val_loss',
        mode: str = 'min',
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.keep_last = keep_last
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.saved_checkpoints = []
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]):
        current_value = logs.get(self.monitor)
        
        # 检查是否是最佳
        is_best = False
        if current_value is not None:
            if self.mode == 'min' and current_value < self.best_value:
                self.best_value = current_value
                is_best = True
            elif self.mode == 'max' and current_value > self.best_value:
                self.best_value = current_value
                is_best = True
        
        # 保存最佳模型
        if is_best:
            self._save(trainer, 'best_model.pt')
        
        # 定期保存
        if not self.save_best_only and (epoch + 1) % self.save_every == 0:
            filename = f'checkpoint_epoch_{epoch+1}.pt'
            self._save(trainer, filename)
            self.saved_checkpoints.append(filename)
            
            # 清理旧检查点
            while len(self.saved_checkpoints) > self.keep_last:
                old_file = self.saved_checkpoints.pop(0)
                old_path = self.save_dir / old_file
                if old_path.exists():
                    os.remove(old_path)
    
    def _save(self, trainer: Any, filename: str):
        checkpoint = {
            'epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'best_value': self.best_value,
        }
        
        if hasattr(trainer, 'ema') and trainer.ema is not None:
            checkpoint['ema_state_dict'] = trainer.ema.state_dict()
        
        path = self.save_dir / filename
        torch.save(checkpoint, path)


class LoggingCallback(Callback):
    """
    日志记录回调
    
    Args:
        log_dir: 日志目录
        use_tensorboard: 是否使用 TensorBoard
        use_wandb: 是否使用 W&B
    """
    
    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        
        self.writer = None
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir)
            except ImportError:
                logger.warning("TensorBoard 未安装，跳过")
        
        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project or "battery-flow-matching")
            except ImportError:
                logger.warning("W&B 未安装，跳过")
    
    def on_step(self, trainer: Any, step: int, logs: Dict[str, float]):
        if self.writer is not None:
            for key, value in logs.items():
                self.writer.add_scalar(f'train/{key}', value, step)
            self.writer.add_scalar('train/lr', trainer.optimizer.param_groups[0]['lr'], step)
        
        if self.use_wandb:
            try:
                import wandb
                wandb.log({f'train/{k}': v for k, v in logs.items()}, step=step)
                wandb.log({'train/lr': trainer.optimizer.param_groups[0]['lr']}, step=step)
            except:
                pass
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]):
        # 记录历史
        self.history['train_loss'].append(logs.get('loss', 0))
        self.history['val_loss'].append(logs.get('val_loss', 0))
        self.history['lr'].append(trainer.optimizer.param_groups[0]['lr'])
        
        if self.writer is not None:
            for key, value in logs.items():
                self.writer.add_scalar(f'epoch/{key}', value, epoch)
        
        if self.use_wandb:
            try:
                import wandb
                wandb.log({f'epoch/{k}': v for k, v in logs.items()}, step=epoch)
            except:
                pass
        
        # 保存历史到文件
        with open(self.log_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def on_train_end(self, trainer: Any):
        if self.writer is not None:
            self.writer.close()


class EarlyStoppingCallback(Callback):
    """
    早停回调
    
    Args:
        monitor: 监控的指标
        patience: 容忍的 epoch 数
        min_delta: 最小改善量
        mode: 'min' 或 'max'
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.should_stop = False
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]):
        current_value = logs.get(self.monitor)
        
        if current_value is None:
            return
        
        improved = False
        if self.mode == 'min':
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                logger.info(f"早停触发: {self.monitor} 在 {self.patience} 个 epoch 内没有改善")
                self.should_stop = True


class VisualizationCallback(Callback):
    """
    可视化回调
    
    定期生成训练过程的可视化图表。
    
    Args:
        save_dir: 保存目录
        plot_every: 每 N 个 epoch 绘图
    """
    
    def __init__(
        self,
        save_dir: str,
        plot_every: int = 5,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.plot_every = plot_every
        
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]):
        self.epochs.append(epoch)
        self.train_losses.append(logs.get('loss', 0))
        self.val_losses.append(logs.get('val_loss', 0))
        
        if (epoch + 1) % self.plot_every == 0:
            self._plot_losses(epoch)
    
    def _plot_losses(self, epoch: int):
        """绘制损失曲线"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(self.epochs, self.train_losses, label='Train Loss', marker='o')
            if any(v > 0 for v in self.val_losses):
                ax.plot(self.epochs, self.val_losses, label='Val Loss', marker='s')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.save_dir / f'loss_curve_epoch_{epoch+1}.png', dpi=150)
            plt.close()
            
        except ImportError:
            logger.warning("Matplotlib 未安装，跳过可视化")
    
    def on_train_end(self, trainer: Any):
        """训练结束时绘制最终图表"""
        if len(self.epochs) > 0:
            self._plot_losses(self.epochs[-1])


class GradientMonitorCallback(Callback):
    """
    梯度监控回调
    
    监控训练过程中的梯度统计，用于调试和诊断。
    """
    
    def __init__(self, log_every: int = 100):
        self.log_every = log_every
    
    def on_step(self, trainer: Any, step: int, logs: Dict[str, float]):
        if step % self.log_every != 0:
            return
        
        total_norm = 0.0
        max_norm = 0.0
        
        for name, param in trainer.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_norm = max(max_norm, param_norm)
        
        total_norm = total_norm ** 0.5
        
        logger.debug(
            f"Step {step} - Gradient norm: {total_norm:.4f}, "
            f"Max param grad: {max_norm:.4f}"
        )
