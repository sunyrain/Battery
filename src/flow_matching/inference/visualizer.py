"""
轨迹可视化模块

提供潜空间轨迹和健康评分的可视化功能。
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TrajectoryVisualizer:
    """
    轨迹可视化器
    
    提供多种可视化方法来展示电池生命周期预测结果。
    
    Args:
        save_dir: 保存目录
        dpi: 图像 DPI
        figsize: 图像尺寸
    """
    
    def __init__(
        self,
        save_dir: Optional[str] = None,
        dpi: int = 150,
        figsize: Tuple[int, int] = (10, 6),
    ):
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.dpi = dpi
        self.figsize = figsize
    
    def plot_health_score_trajectory(
        self,
        cycles: Union[torch.Tensor, np.ndarray],
        health_scores: Union[torch.Tensor, np.ndarray],
        true_scores: Optional[Union[torch.Tensor, np.ndarray]] = None,
        title: str = "Battery Health Score Trajectory",
        save_name: Optional[str] = None,
        show: bool = True,
    ):
        """
        绘制健康评分轨迹
        
        Args:
            cycles: cycle 数组
            health_scores: 预测的健康评分
            true_scores: 真实健康评分（可选）
            title: 图标题
            save_name: 保存文件名
            show: 是否显示图像
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib 未安装，跳过可视化")
            return
        
        if isinstance(cycles, torch.Tensor):
            cycles = cycles.cpu().numpy()
        if isinstance(health_scores, torch.Tensor):
            health_scores = health_scores.cpu().numpy()
        if true_scores is not None and isinstance(true_scores, torch.Tensor):
            true_scores = true_scores.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(cycles, health_scores, 'b-', linewidth=2, label='Predicted')
        
        if true_scores is not None:
            ax.plot(cycles, true_scores, 'r--', linewidth=2, label='True')
        
        ax.axhline(y=0.8, color='orange', linestyle=':', label='Failure Threshold (0.8)')
        
        ax.set_xlabel('Cycle', fontsize=12)
        ax.set_ylabel('Health Score (Degradation)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_latent_space_2d(
        self,
        trajectory: Union[torch.Tensor, np.ndarray],
        method: str = "pca",
        color_by_time: bool = True,
        title: str = "Latent Space Trajectory",
        save_name: Optional[str] = None,
        show: bool = True,
    ):
        """
        2D 潜空间可视化
        
        Args:
            trajectory: [num_steps, latent_dim] 轨迹
            method: 降维方法 ("pca", "tsne", "umap")
            color_by_time: 是否按时间着色
            title: 图标题
            save_name: 保存文件名
            show: 是否显示
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
        except ImportError:
            logger.warning("Required packages not installed")
            return
        
        if isinstance(trajectory, torch.Tensor):
            trajectory = trajectory.cpu().numpy()
        
        # 如果是 3D (num_steps, batch, dim)，取第一个样本
        if trajectory.ndim == 3:
            trajectory = trajectory[:, 0, :]
        
        # 降维
        if method == "pca":
            reducer = PCA(n_components=2)
            coords_2d = reducer.fit_transform(trajectory)
        elif method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, perplexity=min(30, len(trajectory) - 1))
            coords_2d = reducer.fit_transform(trajectory)
        elif method == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_components=2)
                coords_2d = reducer.fit_transform(trajectory)
            except ImportError:
                logger.warning("UMAP 未安装，使用 PCA")
                reducer = PCA(n_components=2)
                coords_2d = reducer.fit_transform(trajectory)
        else:
            raise ValueError(f"未知的降维方法: {method}")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if color_by_time:
            colors = np.arange(len(coords_2d))
            scatter = ax.scatter(
                coords_2d[:, 0], coords_2d[:, 1],
                c=colors, cmap='viridis', s=50, alpha=0.7
            )
            plt.colorbar(scatter, ax=ax, label='Time Step')
        else:
            ax.scatter(coords_2d[:, 0], coords_2d[:, 1], s=50, alpha=0.7)
        
        # 绘制轨迹线
        ax.plot(coords_2d[:, 0], coords_2d[:, 1], 'k-', alpha=0.3, linewidth=1)
        
        # 标记起点和终点
        ax.scatter([coords_2d[0, 0]], [coords_2d[0, 1]], 
                   c='green', s=200, marker='*', label='Start', zorder=5)
        ax.scatter([coords_2d[-1, 0]], [coords_2d[-1, 1]], 
                   c='red', s=200, marker='X', label='End', zorder=5)
        
        ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_rul_prediction(
        self,
        rul_result: Dict,
        title: str = "Remaining Useful Life Prediction",
        save_name: Optional[str] = None,
        show: bool = True,
    ):
        """
        RUL 预测可视化
        
        Args:
            rul_result: 来自 LifecyclePredictor.predict_rul() 的结果
            title: 图标题
            save_name: 保存文件名
            show: 是否显示
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib 未安装")
            return
        
        cycles = rul_result['cycles'].numpy()
        health_scores = rul_result['health_scores'].numpy()
        if health_scores.ndim > 1:
            health_scores = health_scores.squeeze()
        
        current_cycle = rul_result['current_cycle']
        failure_cycle = rul_result['failure_cycle']
        threshold = rul_result['threshold']
        rul = rul_result['rul']
        confidence = rul_result['confidence']
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 绘制健康评分曲线
        ax.plot(cycles, health_scores, 'b-', linewidth=2, label='Predicted Health Score')
        
        # 绘制阈值线
        ax.axhline(y=threshold, color='red', linestyle='--', 
                   label=f'Failure Threshold ({threshold})')
        
        # 标记当前位置
        ax.axvline(x=current_cycle, color='green', linestyle=':', 
                   label=f'Current Cycle ({current_cycle})')
        
        # 标记预测失效点
        ax.axvline(x=failure_cycle, color='red', linestyle=':', 
                   label=f'Predicted Failure ({failure_cycle})')
        
        # 填充 RUL 区域
        ax.axvspan(current_cycle, failure_cycle, alpha=0.2, color='yellow', 
                   label=f'RUL = {rul} cycles')
        
        ax.set_xlabel('Cycle', fontsize=12)
        ax.set_ylabel('Health Score (Degradation)', fontsize=12)
        ax.set_title(f'{title}\nRUL: {rul} cycles, Confidence: {confidence:.2%}', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_comparison(
        self,
        comparison_result: Dict,
        title: str = "Trajectory Comparison",
        save_name: Optional[str] = None,
        show: bool = True,
    ):
        """
        多轨迹比较可视化
        
        Args:
            comparison_result: 来自 LifecyclePredictor.compare_trajectories() 的结果
            title: 图标题
            save_name: 保存文件名
            show: 是否显示
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib 未安装")
            return
        
        labels = comparison_result['labels']
        health_scores_list = comparison_result['health_scores']
        cycles = comparison_result['cycles'].numpy()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
        
        for label, scores, color in zip(labels, health_scores_list, colors):
            scores_np = scores.numpy() if isinstance(scores, torch.Tensor) else scores
            ax.plot(cycles, scores_np, linewidth=2, label=label, color=color)
        
        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Threshold')
        
        ax.set_xlabel('Cycle', fontsize=12)
        ax.set_ylabel('Health Score', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_animation(
        self,
        trajectory: Union[torch.Tensor, np.ndarray],
        health_scores: Union[torch.Tensor, np.ndarray],
        cycles: Union[torch.Tensor, np.ndarray],
        save_name: str = "lifecycle_animation.gif",
        fps: int = 10,
    ):
        """
        创建生命周期演化动画
        
        Args:
            trajectory: 轨迹
            health_scores: 健康评分
            cycles: cycle 数
            save_name: 保存文件名
            fps: 帧率
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            from sklearn.decomposition import PCA
        except ImportError:
            logger.warning("Required packages not installed for animation")
            return
        
        if isinstance(trajectory, torch.Tensor):
            trajectory = trajectory.cpu().numpy()
        if isinstance(health_scores, torch.Tensor):
            health_scores = health_scores.cpu().numpy()
        if isinstance(cycles, torch.Tensor):
            cycles = cycles.cpu().numpy()
        
        if trajectory.ndim == 3:
            trajectory = trajectory[:, 0, :]
        if health_scores.ndim > 1:
            health_scores = health_scores.squeeze()
        
        # PCA 降维
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(trajectory)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        def update(frame):
            for ax in axes:
                ax.clear()
            
            # 左图：潜空间轨迹
            axes[0].scatter(coords_2d[:frame+1, 0], coords_2d[:frame+1, 1],
                           c=np.arange(frame+1), cmap='viridis', s=50)
            axes[0].plot(coords_2d[:frame+1, 0], coords_2d[:frame+1, 1], 'k-', alpha=0.3)
            axes[0].scatter([coords_2d[frame, 0]], [coords_2d[frame, 1]], 
                           c='red', s=200, marker='o')
            axes[0].set_xlim(coords_2d[:, 0].min() - 0.5, coords_2d[:, 0].max() + 0.5)
            axes[0].set_ylim(coords_2d[:, 1].min() - 0.5, coords_2d[:, 1].max() + 0.5)
            axes[0].set_xlabel('PC1')
            axes[0].set_ylabel('PC2')
            axes[0].set_title(f'Latent Space (Cycle {int(cycles[frame])})')
            axes[0].grid(True, alpha=0.3)
            
            # 右图：健康评分
            axes[1].plot(cycles[:frame+1], health_scores[:frame+1], 'b-', linewidth=2)
            axes[1].scatter([cycles[frame]], [health_scores[frame]], c='red', s=100)
            axes[1].axhline(y=0.8, color='orange', linestyle='--', alpha=0.5)
            axes[1].set_xlim(cycles[0], cycles[-1])
            axes[1].set_ylim(-0.05, 1.05)
            axes[1].set_xlabel('Cycle')
            axes[1].set_ylabel('Health Score')
            axes[1].set_title(f'Health Score: {health_scores[frame]:.3f}')
            axes[1].grid(True, alpha=0.3)
            
            return axes
        
        anim = animation.FuncAnimation(
            fig, update, frames=len(trajectory),
            interval=1000 // fps, blit=False
        )
        
        if self.save_dir:
            anim.save(self.save_dir / save_name, writer='pillow', fps=fps)
            logger.info(f"动画已保存: {self.save_dir / save_name}")
        
        plt.close()
