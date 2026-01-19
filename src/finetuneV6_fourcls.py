"""
基于已训练的 smartwavev12 分类模型进行排序约束的自监督微调（V6版本 - 四分类）
使用时序关系（late > early）来训练 SEI 严重程度打分模型

V6 核心改进（四分类版本）：
- 使用固定的分数映射: v0=0(固定), v1=sigmoid(beta1)(可学习), v2=v1+sigmoid(beta2)*(1-v1)(可学习，约束v2>v1), v3=1(固定)
- cycle=1 时，约束 score ≈ 0 - 完全健康
- cycle 在 late 区时，约束 score ≈ late_cycle_target_score - 完全退化
- 四分类：Class 0=健康, Class 1=轻度退化, Class 2=中度退化, Class 3=严重退化

核心思想：
- 冻结特征提取器（time_branch + freq_branch），保留已学到的特征表示能力（V12 时/频域分支内部完成差分）
- 解冻分类头（output_head），让其重新学习如何"打分"
- 输出 SEI 严重程度分数（0-1），而不是分类标签

训练策略：
使用排序损失 MarginRankingLoss，数学表达式：
    L_Rank = max(0, -(score_late - score_early) + margin)
这等同于要求 score_late > score_early + margin

使用 src/train.py 中的训练、验证和测试模块进行实际训练。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import logging
import numpy as np
import pandas as pd

# 按 smartwave V9 版本
from src.smartwavev9 import DeltaBatteryModel
from src.load_pretrained import load_pretrained_model


class SelfLearning(nn.Module):
    """
    基于已训练分类模型的自监督打分模型（四分类版本）
    
    核心思想：
    1. 冻结特征提取器（time_branch + freq_branch），保留已学到的特征表示能力
    2. 解冻分类头（output_head），让其重新学习如何"打分"
    3. 输出严重程度分数（0-1），而不是分类标签
    4. 四分类映射到四个分数值，v1 < v2 确保单调性
    """
    
    def __init__(self, original_model: DeltaBatteryModel):
        """
        Args:
            original_model: 已训练好的 DeltaBatteryModel 模型
        """
        super().__init__()
        
        # 1. 提取原始模型的各个组件（smartwave V9）
        self.time_branch = original_model.time_branch
        self.freq_branch = original_model.freq_branch
        self.alpha = original_model.alpha
        self.d_model = original_model.d_model
        self.output_head = original_model.output_head
        
        # 2. 冻结特征提取器
        for param in self.time_branch.parameters():
            param.requires_grad = False
        for param in self.freq_branch.parameters():
            param.requires_grad = False
        # alpha 也冻结，保持原有的时频融合权重
        self.alpha.requires_grad = False
        
        # 3. 解冻分类头（让它重新学习如何打分）
        for param in self.output_head.parameters():
            param.requires_grad = True

        # 冻结分支后切换到 eval，避免 Dropout/LayerNorm 统计干扰微调
        self.time_branch.eval()
        self.freq_branch.eval()
        
        # 4. 强制将最后一层分类器的偏置设为0并冻结
        # output_head 是一个 Sequential，最后一层是 Linear
        last_layer = None
        for module in self.output_head.modules():
            if isinstance(module, nn.Linear):
                last_layer = module
        
        if last_layer is not None and last_layer.bias is not None:
            # 将偏置设为0
            nn.init.constant_(last_layer.bias, 0.0)
            # 冻结偏置参数，不再更新
            last_layer.bias.requires_grad = False
            logging.info("[SelfLearning V6] 最后一层分类器的偏置已设为0并冻结")
        
        # 5. 可训练的类别分数参数（四分类版本）
        # v0=0 (固定): 完全健康
        # v3=1 (固定): 完全失效（严重死锂/短路）
        # v1=sigmoid(beta1) (可学习): 轻度退化，自动在(0,1)之间
        # v2=sigmoid(beta2) (可学习): 中度退化，通过参数化确保 beta1 < beta2，从而保证 v1 < v2
        self.beta1 = nn.Parameter(torch.tensor(-1.0))  # 初始化为-1，sigmoid(-1)≈0.27
        self.beta2_offset = nn.Parameter(torch.tensor(0.0))  # 偏移量，通过 softplus 确保为正
        
        logging.info("[SelfLearning V6] 模型初始化完成（四分类版本）")
        logging.info(f"  - 冻结参数: time_branch, freq_branch（含 channel_projections 与 cls_token）, alpha, output_head最后一层bias")
        logging.info(f"  - 解冻参数: output_head (除最后一层bias外), beta1, beta2_offset")
        logging.info(f"  - V6特性: 四分类，分数映射为 v0=0(固定), v1=sigmoid(beta1)(可学习), v2=sigmoid(beta2)(可学习), v3=1(固定)")
        logging.info(f"  - 约束机制: beta2 = beta1 + softplus(beta2_offset)，确保 beta1 < beta2，从而保证 v1 < v2")
        logging.info(f"  - 分支处于 eval 模式，保持 V12 特征一致性，微调时仅更新分类头与分数映射参数")
    
    def forward(self, signals_after: torch.Tensor, signals_before: torch.Tensor, return_probs: bool = False):
        """
        前向传播，输出 SEI 严重程度分数
        
        Args:
            signals_after: [batch, channels, seq_len] 运行后信号
            signals_before: [batch, channels, seq_len] 运行前信号
            return_probs: 是否返回概率值
        
        Returns:
            如果 return_probs=False:
                score: [batch] SEI 严重程度分数，范围 0-1
            如果 return_probs=True:
                (score, probs): score [batch], probs [batch, num_classes]
        """
        # 使用冻结的特征提取器（smartwave V9 API）
        with torch.no_grad():
            # 时域：分别编码后差分
            time_feat_after = self.time_branch(signals_after)
            time_feat_before = self.time_branch(signals_before)
            time_delta = time_feat_after - time_feat_before
            
            # 频域：内部先差分后编码
            freq_delta = self.freq_branch(signals_after, signals_before)
            
            # Alpha 加权融合
            alpha = torch.sigmoid(self.alpha)
            fused_delta = alpha * time_delta + (1 - alpha) * freq_delta
        
        # 通过解冻的分类头输出 logits
        logits = self.output_head(fused_delta)  # [batch, num_classes]
        if logits.dim() != 2 or logits.shape[1] != 4:
            raise ValueError(f"[SelfLearning V6] 期望分类头输出 4 个类别，实际得到形状 {logits.shape}")
        
        # 将分类 logits 转换为 SEI 严重程度分数
        # 假设：Class 0 = 健康, Class 1 = 轻度退化, Class 2 = 中度退化, Class 3 = 严重退化
        probs = torch.softmax(logits, dim=1)  # [batch, 4]
        
        # 使用可训练的类别分数计算 SEI 严重程度分数（四分类版本）
        # v0=0 (固定): 完全健康
        # v1=sigmoid(beta1) (可学习): 轻度退化，自动在(0,1)之间
        # v2=sigmoid(beta2) (可学习): 中度退化，通过 beta2 = beta1 + softplus(beta2_offset) 确保 beta1 < beta2，从而保证 v1 < v2
        # v3=1 (固定): 完全失效
        v0 = 0.0
        v1 = torch.sigmoid(self.beta1)  # 通过sigmoid确保v1∈(0,1)
        beta2 = self.beta1 + F.softplus(self.beta2_offset)  # 确保 beta2 > beta1
        v2 = torch.sigmoid(beta2)  # 因为 beta2 > beta1，且 sigmoid 单调递增，所以 v2 > v1
        v3 = 1.0
        
        # Score = v0*p0 + v1*p1 + v2*p2 + v3*p3
        # 分数范围自动在 [0, 1] 之间
        score = v0*probs[:, 0] + v1*probs[:, 1] + v2*probs[:, 2] + v3*probs[:, 3]  # [batch]
        
        if return_probs:
            return score, probs
        return score


def compute_ranking_loss(
    model: SelfLearning,
    batch_data: tuple,
    device: torch.device,
    margin: float = 0.05,
    early_cycle_weight: float = 1.0,
    late_cycle_weight: float = 1.0,
    early_cycle_max: int = 1,
    late_cycle_min: int = 195,
    late_cycle_max: int = 200,
    late_cycle_target_score: float = 1.0) -> Dict[str, torch.Tensor]:
    """
    计算排序约束损失（V6版本：四分类，使用score约束）
    
    损失组成：
    1. 排序损失: L_rank = max(0, -(score_late - score_early) + margin)
    2. Early cycle约束: 要求 cycle<=early_cycle_max 的 score ≈ 0
    3. Late cycle约束: 要求 cycle∈[late_cycle_min, late_cycle_max] 的 score ≈ late_cycle_target_score
    
    Args:
        model: SelfLearning 模型
        batch_data: 来自 Dataset_com 的批次数据
        device: 设备
        margin: 排序损失的边界（默认0.05）
        early_cycle_weight: 早期cycle分数约束权重（默认1.0）
        late_cycle_weight: 晚期cycle分数约束权重（默认1.0）
        early_cycle_max: 早期cycle的最大值，约束cycle<=early_cycle_max（默认1）
        late_cycle_min: 晚期cycle的最小值（默认195）
        late_cycle_max: 晚期cycle的最大值（默认200）
        late_cycle_target_score: 晚期cycle的目标分数（默认1.0）
    
    Returns:
        包含各项损失的字典
    """
    # 解包数据
    # Dataset_com 返回: (after_i, before_i, after_j, before_j, cycle_i, cycle_j, labels)
    after_i, before_i, after_j, before_j, cycle_i, cycle_j, labels = batch_data
    
    # 移动到设备
    after_i = after_i.to(device)
    before_i = before_i.to(device)
    after_j = after_j.to(device)
    before_j = before_j.to(device)
    cycle_i = cycle_i.to(device) if isinstance(cycle_i, torch.Tensor) else torch.tensor(cycle_i, device=device)
    cycle_j = cycle_j.to(device) if isinstance(cycle_j, torch.Tensor) else torch.tensor(cycle_j, device=device)
    
    # 前向传播，得到两个分数和概率
    score_i, probs_i = model(after_i, before_i, return_probs=True)  # early 的分数和概率
    score_j, probs_j = model(after_j, before_j, return_probs=True)  # late 的分数和概率
    
    # === 损失 1: 排序约束损失 ===
    # 使用 MarginRankingLoss: L = max(0, -(score_j - score_i) + margin)
    rank_criterion = nn.MarginRankingLoss(margin=margin)
    target = torch.ones_like(score_i)
    rank_loss = rank_criterion(score_j, score_i, target)
    
    # === 损失 2: Early Cycle 分数约束损失 ===
    # 约束 cycle<=early_cycle_max 的样本，其分数应该接近 0
    early_cycle_loss = torch.tensor(0.0, device=device)
    
    mask_i_early = (cycle_i <= early_cycle_max)
    mask_j_early = (cycle_j <= early_cycle_max)
    
    if mask_i_early.any():
        # 约束分数接近0（完全健康）
        early_cycle_loss = early_cycle_loss + torch.mean(score_i[mask_i_early] ** 2)
    
    if mask_j_early.any():
        # 约束分数接近0（完全健康）
        early_cycle_loss = early_cycle_loss + torch.mean(score_j[mask_j_early] ** 2)
    
    # === 损失 3: Late Cycle 分数约束损失 ===
    # 约束 cycle∈[late_cycle_min, late_cycle_max] 的样本，其分数应该接近 late_cycle_target_score
    late_cycle_loss = torch.tensor(0.0, device=device)
    
    mask_i_late = (cycle_i >= late_cycle_min) & (cycle_i <= late_cycle_max)
    mask_j_late = (cycle_j >= late_cycle_min) & (cycle_j <= late_cycle_max)
    
    if mask_i_late.any():
        # 约束分数接近late_cycle_target_score（完全退化）
        late_cycle_loss = late_cycle_loss + torch.mean((score_i[mask_i_late] - late_cycle_target_score) ** 2)
    
    if mask_j_late.any():
        # 约束分数接近late_cycle_target_score（完全退化）
        late_cycle_loss = late_cycle_loss + torch.mean((score_j[mask_j_late] - late_cycle_target_score) ** 2)
    
    # === 总损失 ===
    total_loss = (rank_loss + 
                  early_cycle_weight * early_cycle_loss + 
                  late_cycle_weight * late_cycle_loss)
    
    return {
        'total_loss': total_loss,
        'rank_loss': rank_loss,
        'early_cycle_loss': early_cycle_loss,
        'late_cycle_loss': late_cycle_loss,
        'score_i': score_i,
        'score_j': score_j,
        'probs_i': probs_i,
        'probs_j': probs_j
    }


def evaluate_ranking_accuracy(
    model: SelfLearning,
    data_loader: DataLoader,
    device: torch.device) -> Dict[str, float]:
    """
    评估模型的排序准确率和分数分布
    
    核心功能：
    1. **排序准确率**: 计算 score_late > score_early 的比例
       - 这是最关键的指标，反映模型是否学会了时序关系
    
    2. **Early Cycle约束验证**: 检查 cycle<10 的分数
       - 应该接近 0，验证模型是否正确建立了"早期=低分"的概念
    
    3. **Late Cycle约束验证**: 检查 cycle∈[190,200] 的分数
       - 应该接近 1，验证模型是否正确建立了"晚期=高分"的概念
    
    4. **分数分布统计**: 分析分数的范围、均值、标准差
       - 帮助理解模型行为，检测异常值
    
    Args:
        model: SelfLearning 模型
        data_loader: 数据加载器（来自 Dataset_com）
        device: 计算设备
    
    Returns:
        评估指标字典，包含：
        - ranking_accuracy: 排序准确率（最关键指标）
        - mean_score, std_score, min_score, max_score: 所有分数的统计
        - early_cycle_mean_score, early_cycle_std_score: cycle<10 的分数统计
        - late_cycle_mean_score, late_cycle_std_score: cycle∈[190,200] 的分数统计
        - num_early_cycle_samples: cycle<10 的样本数
        - num_late_cycle_samples: cycle∈[190,200] 的样本数
    """
    model.eval()
    
    all_scores_i = []
    all_scores_j = []
    all_cycles_i = []
    all_cycles_j = []
    correct_rankings = 0
    total_pairs = 0
    
    cycle0_scores = []
    
    with torch.no_grad():
        for batch in data_loader:
            after_i, before_i, after_j, before_j, cycle_i, cycle_j, labels = batch
            
            # 移动信号到设备
            after_i = after_i.to(device)
            before_i = before_i.to(device)
            after_j = after_j.to(device)
            before_j = before_j.to(device)
            
            # 前向传播得到分数
            score_i = model(after_i, before_i)  # early 的分数
            score_j = model(after_j, before_j)  # late 的分数
            
            # === 1. 计算排序准确率 ===
            correct = (score_j > score_i).sum().item()
            correct_rankings += correct
            total_pairs += len(score_i)
            
            # === 2. 收集所有分数（转到 CPU 避免内存泄漏）===
            all_scores_i.extend(score_i.cpu().numpy())
            all_scores_j.extend(score_j.cpu().numpy())
            
            # 安全地转换 cycle 数据
            if isinstance(cycle_i, torch.Tensor):
                all_cycles_i.extend(cycle_i.cpu().numpy() if cycle_i.is_cuda else cycle_i.numpy())
            else:
                all_cycles_i.extend(np.array(cycle_i))
            
            if isinstance(cycle_j, torch.Tensor):
                all_cycles_j.extend(cycle_j.cpu().numpy() if cycle_j.is_cuda else cycle_j.numpy())
            else:
                all_cycles_j.extend(np.array(cycle_j))
            
            # === 3. 收集 cycle=0 的分数 ===
            # 确保 cycle 数据在正确的设备上进行比较
            cycle_i_tensor = cycle_i if isinstance(cycle_i, torch.Tensor) else torch.tensor(cycle_i)
            cycle_j_tensor = cycle_j if isinstance(cycle_j, torch.Tensor) else torch.tensor(cycle_j)
            
            mask_i = (cycle_i_tensor == 0)
            mask_j = (cycle_j_tensor == 0)
            
            if mask_i.any():
                cycle0_scores.extend(score_i[mask_i].cpu().numpy())
            if mask_j.any():
                cycle0_scores.extend(score_j[mask_j].cpu().numpy())
    
    # 转换为 numpy 数组
    all_scores_i = np.array(all_scores_i)
    all_scores_j = np.array(all_scores_j)
    all_scores = np.concatenate([all_scores_i, all_scores_j])
    
    # 计算指标
    metrics = {
        'ranking_accuracy': correct_rankings / total_pairs,
        'mean_score': float(np.mean(all_scores)),
        'std_score': float(np.std(all_scores)),
        'min_score': float(np.min(all_scores)),
        'max_score': float(np.max(all_scores)),
    }
    
    if len(cycle0_scores) > 0:
        metrics['cycle0_mean_score'] = float(np.mean(cycle0_scores))
        metrics['cycle0_std_score'] = float(np.std(cycle0_scores))
        metrics['num_cycle0_samples'] = len(cycle0_scores)
    else:
        metrics['cycle0_mean_score'] = None
        metrics['cycle0_std_score'] = None
        metrics['num_cycle0_samples'] = 0
    
    return metrics


def evaluate_ranking_detailed(
    model: SelfLearning,
    data_loader: DataLoader,
    device: torch.device,
    phase: str = 'Test') -> pd.DataFrame:
    """
    详细评估：逐样本输出预测分数及元数据
    
    Args:
        model: SelfLearning 模型
        data_loader: 数据加载器（来自 Dataset_com）
        device: 计算设备
        phase: 阶段名称（用于记录）
    
    Returns:
        DataFrame: 包含每个配对样本的详细信息
            - pair_idx: 配对索引（全局索引）
            - cycle_i: 样本i的cycle
            - cycle_j: 样本j的cycle
            - score_i: 样本i的预测分数
            - score_j: 样本j的预测分数
            - score_diff: score_j - score_i
            - correct_ranking: 是否正确排序（score_j > score_i）
            - label: 标签
            - current_density: 电流密度
            - frequency: 频率
            - after_path_i: 样本i的after路径
            - after_path_j: 样本j的after路径
            - prob_i_class0, prob_i_class1, prob_i_class2, prob_i_class3: 样本i的类别概率
            - prob_j_class0, prob_j_class1, prob_j_class2, prob_j_class3: 样本j的类别概率
    """
    model.eval()
    
    results = []
    pair_idx = 0
    
    # 获取数据集对象（用于访问元数据）
    dataset = data_loader.dataset
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            after_i, before_i, after_j, before_j, cycle_i, cycle_j, labels = batch
            
            # 移动信号到设备
            after_i = after_i.to(device)
            before_i = before_i.to(device)
            after_j = after_j.to(device)
            before_j = before_j.to(device)
            
            # 前向传播得到分数和概率
            score_i, probs_i = model(after_i, before_i, return_probs=True)  # early 的分数和概率
            score_j, probs_j = model(after_j, before_j, return_probs=True)  # late 的分数和概率
            
            # 转换为numpy
            score_i_np = score_i.cpu().numpy().flatten()
            score_j_np = score_j.cpu().numpy().flatten()
            probs_i_np = probs_i.cpu().numpy()  # [batch, num_classes]
            probs_j_np = probs_j.cpu().numpy()  # [batch, num_classes]
            
            # 安全地转换 cycle 数据
            if isinstance(cycle_i, torch.Tensor):
                cycle_i_np = cycle_i.cpu().numpy() if cycle_i.is_cuda else cycle_i.numpy()
            else:
                cycle_i_np = np.array(cycle_i)
            
            if isinstance(cycle_j, torch.Tensor):
                cycle_j_np = cycle_j.cpu().numpy() if cycle_j.is_cuda else cycle_j.numpy()
            else:
                cycle_j_np = np.array(cycle_j)
            
            if isinstance(labels, torch.Tensor):
                labels_np = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
            else:
                labels_np = np.array(labels)
            
            # 逐样本记录结果
            batch_size = len(score_i_np)
            for i in range(batch_size):
                # 计算数据集索引（batch_idx * batch_size + i）
                dataset_idx = batch_idx * data_loader.batch_size + i
                
                # 获取元数据
                metadata = {}
                if hasattr(dataset, 'get_metadata'):
                    metadata = dataset.get_metadata(dataset_idx)
                
                # 构建结果字典
                result = {
                    'pair_idx': pair_idx,
                    'batch_idx': batch_idx,
                    'in_batch_idx': i,
                    'cycle_i': int(cycle_i_np[i]),
                    'cycle_j': int(cycle_j_np[i]),
                    'score_i': float(score_i_np[i]),
                    'score_j': float(score_j_np[i]),
                    'score_diff': float(score_j_np[i] - score_i_np[i]),
                    'correct_ranking': bool(score_j_np[i] > score_i_np[i]),
                    'label': int(labels_np[i]),
                    'current_density': metadata.get('current_density', None),
                    'frequency': metadata.get('frequency', None),
                    'after_path_i': metadata.get('after_path_i', None),
                    'after_path_j': metadata.get('after_path_j', None),
                    'before_path': metadata.get('before_path', None),
                    'cycle_before': metadata.get('cycle_before', None),
                    'id_i': metadata.get('id_i', None),
                    'id_j': metadata.get('id_j', None),
                }
                
                # 添加概率值（样本i的概率分布）
                for class_idx in range(probs_i_np.shape[1]):
                    result[f'prob_i_class{class_idx}'] = float(probs_i_np[i, class_idx])
                
                # 添加概率值（样本j的概率分布）
                for class_idx in range(probs_j_np.shape[1]):
                    result[f'prob_j_class{class_idx}'] = float(probs_j_np[i, class_idx])
                
                results.append(result)
                pair_idx += 1
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    print(f"\n[{phase}] 详细评估完成:")
    print(f"  - 总配对数: {len(df)}")
    print(f"  - 排序准确率: {df['correct_ranking'].mean():.4f}")
    print(f"  - Score_i 范围: [{df['score_i'].min():.4f}, {df['score_i'].max():.4f}]")
    print(f"  - Score_j 范围: [{df['score_j'].min():.4f}, {df['score_j'].max():.4f}]")
    
    return df


def create_self_learning_model(
    checkpoint_path: str,
    config: Optional[Dict[str, Any]] = None,
    device: torch.device = torch.device('cpu')
) -> SelfLearning:
    """
    从预训练模型创建自监督学习模型
    
    Args:
        checkpoint_path: checkpoint 文件路径
        config: 模型配置（如果 checkpoint 中没有包含）
        device: 设备
    
    Returns:
        SelfLearning 模型
    """
    # 加载预训练模型
    pretrained_model = load_pretrained_model(checkpoint_path, config, device)
    
    # 创建自监督学习模型
    self_learning_model = SelfLearning(pretrained_model)
    
    return self_learning_model


__all__ = [
    'SelfLearning',
    'compute_ranking_loss',
    'evaluate_ranking_accuracy',
    'load_pretrained_model',
    'create_self_learning_model'
]

