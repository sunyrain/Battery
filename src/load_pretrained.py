"""
通用的预训练模型加载工具（基于 smartwave V9，支持频域动态通道）
"""

import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from src.smartwavev9 import DeltaBatteryModel, create_battery_model_from_config


def load_pretrained_model(
    checkpoint_path: str,
    config: Optional[Dict[str, Any]] = None,
    device: torch.device = torch.device("cpu"),
) -> DeltaBatteryModel:
    """
    加载预训练的分类模型（仅加载 encoder，output_head 根据新配置重新创建）

    - 为 freq_branch 的动态通道 projection 提前创建 Linear 层，再加载权重
    - 只加载 encoder（time_branch/freq_branch/alpha），跳过 output_head

    Args:
        checkpoint_path: checkpoint 文件路径
        config: 新的模型配置（必须提供，用于创建 output_head）
        device: 设备

    Returns:
        DeltaBatteryModel 实例（encoder 权重已加载，output_head 重新初始化）
    """
    logging.info(f"正在加载预训练模型: {checkpoint_path}")

    # 加载 checkpoint（需包含 state_dict）
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    # 兼容 SelfLearning 保存格式：可能带有 "model." 前缀
    if any(k.startswith("model.") for k in raw_state_dict.keys()):
        state_dict = {k[6:]: v for k, v in raw_state_dict.items() if k.startswith("model.")}
    else:
        state_dict = raw_state_dict

    # 从提供的 config 获取模型配置（必须包含 output_head 信息）
    if config is None:
        raise ValueError("必须提供 config 参数，用于指定新的 output_head 结构（num_classes等）")
    # 优先使用 config.model.model_config，其次 config.model，最后 config
    if hasattr(config, "model") and hasattr(config.model, "model_config"):
        base_cfg = config.model.model_config
    elif hasattr(config, "model"):
        base_cfg = config.model
    else:
        base_cfg = config

    # 展平为字典，移除无关键
    if isinstance(base_cfg, dict):
        final_config = {k: v for k, v in base_cfg.items() if not k.startswith("_")}
    else:
        final_config = {k: getattr(base_cfg, k) for k in dir(base_cfg) if not k.startswith("_") and not callable(getattr(base_cfg, k))}
    # 去掉 checkpoint 路径等无关项
    final_config.pop("pretrained_checkpoint", None)
    final_config.pop("model_config", None)

    logging.info("根据新配置创建模型（支持 V9 动态频域通道）...")
    model = create_battery_model_from_config(final_config)
    model.to(device)

    # ===== 预创建频域动态通道以便正确加载权重 =====
    # 在 V9 中，channel_projections 是在 forward 中按需动态创建的。
    # 这里提前根据 checkpoint 里的权重 shape 创建对应的 Linear，
    # 确保 load_state_dict 能把权重正确加载进去。
    d_model = model.d_model
    freq_branch = model.freq_branch
    created_channels = 0
    for key, value in state_dict.items():
        if not (key.startswith("freq_branch.channel_projections.") and key.endswith(".weight")):
            continue
        parts = key.split(".")
        # 格式: freq_branch.channel_projections.<name>.weight
        if len(parts) >= 4:
            channel_name = parts[2]
            in_features = value.shape[1]
            if channel_name not in freq_branch.channel_projections:
                freq_branch.channel_projections[channel_name] = nn.Linear(in_features, d_model).to(device)
                created_channels += 1
    logging.info(f"为频域动态通道预创建 Linear 层: {created_channels} 个")

    # 仅加载 encoder 权重（time_branch, freq_branch, alpha），跳过 output_head
    logging.info("加载 encoder 权重，跳过 output_head ...")
    encoder_state_dict = {k: v for k, v in state_dict.items() if k.startswith(("time_branch.", "freq_branch.", "alpha"))}
    missing_keys, unexpected_keys = model.load_state_dict(encoder_state_dict, strict=False)

    skipped_keys = [k for k in state_dict.keys() if k not in encoder_state_dict]
    logging.info(f"  ✓ 成功加载 encoder 权重：{len(encoder_state_dict)} 个参数")
    logging.info(f"    - 跳过的键（output_head 等）: {len(skipped_keys)} 个")
    if missing_keys:
        logging.warning(f"    ⚠ 缺失的键: {missing_keys}")
    if unexpected_keys:
        logging.warning(f"    ⚠ 未预期的键: {unexpected_keys}")

    model.eval()

    logging.info("=" * 60)
    logging.info("✓ 模型加载完成:")
    logging.info(f"  - Encoder: 从 checkpoint 加载（冻结）")
    logging.info(f"  - Output Head: 根据新配置创建（num_classes={final_config.get('num_classes')}）")
    logging.info("=" * 60)

    return model


def load_finetuned_model(
    checkpoint_path: str,
    device: torch.device = torch.device("cpu"),
) -> tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    加载微调后的 SelfLearning 模型（包含 output_head 与 beta 参数）

    - 先用通用加载器恢复基础 encoder（支持 V9 动态通道与前缀剥离）
    - 再包装为 SelfLearning，并加载完整 state_dict

    Returns:
        model: SelfLearning 实例
        config: checkpoint 中的 config（原样 dict）
        checkpoint_info: {epoch, best_val_metric}
    """
    logging.info(f"正在加载微调模型: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    checkpoint_info = {
        "epoch": checkpoint.get("epoch", -1),
        "best_val_metric": checkpoint.get("best_val_metric", None),
    }

    # 1) 基础 encoder
    base_model = load_pretrained_model(checkpoint_path, config, device)

    # 2) SelfLearning 包装
    from src.model.finetune.finetuneV6_fourcls import SelfLearning

    model = SelfLearning(base_model).to(device)

    # 3) 处理前缀并加载全量权重
    raw_state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    state_dict = (
        {k[6:]: v for k, v in raw_state_dict.items() if k.startswith("model.")}
        if any(k.startswith("model.") for k in raw_state_dict.keys())
        else raw_state_dict
    )

    logging.info("加载完整 finetune 权重（包含 output_head 与 beta 参数）")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        logging.info(f"  - 忽略的额外键: {len(unexpected_keys)} 个")
    if missing_keys:
        logging.warning(f"  - 缺失的键: {len(missing_keys)} 个（若仅 output_head/beta 缺失请检查配置）")

    model.eval()
    return model, config, checkpoint_info


__all__ = ["load_pretrained_model", "load_finetuned_model"]
