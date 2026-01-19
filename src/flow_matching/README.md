# Battery Flow Matching 模型

## 概述

这是一个基于 **Flow Matching** 的电池生命周期预测模型,通过学习潜空间中的分布迁移路径,实现从电池初始状态预测完整生命周期轨迹。

## 架构特点

### 🔬 核心创新

1. **潜空间 Flow Matching**: 在 SmartWave Encoder 的潜空间中学习分布迁移
2. **最优传输线性路径**: 使用 OT-CFM (Optimal Transport Conditional Flow Matching)
3. **高精度 ODE 求解器**: 支持 Euler, RK4, Dopri5 等多种求解器
4. **条件生成**: 支持基于 cycle 的条件生成

### 📐 数学公式

**Flow Matching 目标函数**:
$$\mathcal{L}_{FM} = \mathbb{E}_{t, z_0, z_1} \left\| v_\theta(z_t, t) - u_t(z_t | z_0, z_1) \right\|^2$$

其中:
- $z_t = (1-t) z_0 + t z_1$ (OT 线性插值路径)
- $u_t = z_1 - z_0$ (目标速度场)
- $v_\theta$ 是需要学习的速度场网络

**轨迹推理 (ODE)**:
$$\frac{d z}{d t} = v_\theta(z, t), \quad z(0) = z_0$$

## 项目结构

```
src/flow_matching/
├── __init__.py                 # 包初始化
├── models/                     # 模型定义
│   ├── embeddings.py           # 时间/条件嵌入
│   ├── velocity_net.py         # 速度场网络 (U-Net MLP)
│   └── flow_model.py           # 主模型 (BatteryFlowModel)
├── core/                       # 核心算法
│   ├── ode_solver.py           # ODE 求解器
│   ├── flow_matching_loss.py   # FM 损失函数
│   └── optimal_transport.py    # Sinkhorn OT
├── data/                       # 数据处理
│   ├── preprocessing.py        # 信号预处理
│   ├── dataset.py              # 数据集类
│   └── latent_cache.py         # 潜向量缓存
├── training/                   # 训练模块
│   ├── trainer.py              # 训练器 (EMA, 混合精度)
│   └── callbacks.py            # 回调函数
├── inference/                  # 推理模块
│   ├── predictor.py            # 生命周期预测器
│   └── visualizer.py           # 轨迹可视化
└── utils/                      # 工具函数
    ├── config.py               # 配置管理
    └── metrics.py              # 评估指标
```

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchdiffeq numpy pandas matplotlib scikit-learn tensorboard
```

### 2. 运行测试

```bash
cd sharejj0115
python scripts/test_flow_matching.py
```

### 3. 训练模型

```bash
# 使用默认配置
python scripts/train_flow.py --config configs/flow_matching_config.yaml

# 使用潜空间缓存加速
python scripts/train_flow.py --config configs/flow_matching_config.yaml --compute_cache --use_cache

# 恢复训练
python scripts/train_flow.py --config configs/flow_matching_config.yaml --resume checkpoints/last.pt
```

### 4. 推理预测

```bash
# 预测完整生命周期
python scripts/inference_flow.py --mode lifecycle \
    --checkpoint checkpoints/best_model.pt \
    --signal_after data/after.csv \
    --signal_before data/before.csv \
    --visualize

# 预测 RUL
python scripts/inference_flow.py --mode rul \
    --checkpoint checkpoints/best_model.pt \
    --current_cycle 50
```

## 使用示例

### Python API

```python
from src.flow_matching.models import BatteryFlowModel, BatteryFlowConfig
from src.flow_matching.inference import LifecyclePredictor

# 创建模型
config = BatteryFlowConfig(
    latent_dim=128,
    hidden_dim=512,
    num_layers=6,
    solver_type='dopri5',
)
model = BatteryFlowModel(config, encoder=your_encoder)

# 从检查点加载
predictor = LifecyclePredictor.from_checkpoint(
    'checkpoints/best_model.pt',
    encoder=encoder,
)

# 预测完整生命周期
result = predictor.predict_full_lifecycle(
    signal_after, signal_before,
    num_steps=200,
)

print(f"健康评分轨迹: {result['health_scores']}")
print(f"潜空间轨迹: {result['trajectory'].shape}")

# 预测 RUL
rul_result = predictor.predict_rul(
    signal_after, signal_before,
    current_cycle=50,
)
print(f"剩余寿命: {rul_result['rul']} cycles")
```

## 配置说明

配置文件 `configs/flow_matching_config.yaml`:

```yaml
model:
  latent_dim: 128           # 潜空间维度 (匹配 SmartWave)
  velocity_net:
    hidden_dim: 512         # 隐藏层维度
    num_layers: 6           # 网络层数
    use_adaln: true         # 使用 AdaLN
  solver:
    type: dopri5            # ODE 求解器
    rtol: 1.0e-5            # 相对容差
    atol: 1.0e-5            # 绝对容差

training:
  batch_size: 32
  num_epochs: 100
  optimizer:
    lr: 1.0e-4
    weight_decay: 0.01
```

## 技术细节

### VelocityNetwork 架构

```
输入: [z_t, t_emb, c_emb]
    ↓
Linear → LayerNorm → GELU
    ↓
ResidualBlock × N (with AdaLN)
    ↓ (skip connections)
Linear → 输出 v_θ
```

### ODE 求解器

| 求解器 | 精度 | 速度 | 推荐场景 |
|--------|------|------|----------|
| Euler | 低 | 快 | 快速测试 |
| Midpoint | 中 | 中 | 平衡选择 |
| RK4 | 高 | 中 | 通用 |
| Dopri5 | 高 | 自适应 | **推荐** |
| Adaptive Heun | 高 | 自适应 | 高精度需求 |

### 训练技巧

1. **EMA**: 使用指数移动平均稳定推理
2. **混合精度**: FP16 加速训练
3. **潜空间缓存**: 预计算 Encoder 输出
4. **梯度裁剪**: 防止梯度爆炸

## 评估指标

- **MAE/RMSE**: 健康评分预测误差
- **Wasserstein 距离**: 分布匹配质量
- **RUL 准确率**: 剩余寿命预测
- **轨迹 MSE**: 潜空间轨迹误差

## 参考文献

1. Lipman et al. "Flow Matching for Generative Modeling" (ICLR 2023)
2. Tong et al. "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport" (2023)
3. Liu et al. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (2022)

## License

MIT License
