# CVRP训练项目 - Mamba + DPO

## 项目概述

本项目将TSP的Mamba+DPO训练框架扩展到CVRP（带容量约束的车辆路径问题）。

**核心特性：**
- 🔥 基于Mamba架构的高效序列建模
- 🎯 DPO（Direct Preference Optimization）迭代训练
- 📊 LKH-3求解器生成高质量训练数据
- 🚀 支持混合数据训练（离线+在线）
- 📈 温度退火策略优化探索/利用平衡

## 完整训练流程

### 阶段0：数据生成（使用LKH-3）

```bash
# 生成1000个CVRP-100的训练样本
python generate_dataset.py --num_samples 1000 --problem_size 100 --save_dir data
```

**LKH-3配置说明：**
- 在`config.py`中设置`lkh_path`指向LKH-3.exe
- Windows: `r"D:\lkh-w\LKHWin-3.0.13\x64\Release\LKH-3.exe"`
- Linux: `"LKH"` (假设在系统PATH中)

**生成的数据包含：**
- `depot_xy`: depot坐标 [N, 1, 2]
- `node_xy`: 客户节点坐标 [N, problem_size, 2]
- `node_demand`: 客户需求量 [N, problem_size]
- `tours`: LKH-3求解的最优路径 [N, max_tour_len]
- `lengths`: 路径长度统计

### 阶段1：SFT（Supervised Fine-Tuning）

**目标：** 让模型学习基础的CVRP规则
- 如何返回depot补货
- 如何遵守容量约束
- 基本的路径构建逻辑

**训练方式：**
1. 使用LKH-3生成的高质量数据（推荐）
2. 使用启发式算法在线生成数据（备选）

```python
# 在config.py中设置
sft_data_path = "data/sft_data_lkh_cvrp100_n1000_xxx.pt"  # LKH-3生成的数据
# 或
sft_data_path = "None"  # 使用启发式算法在线生成
```

**SFT训练参数：**
```python
sft_lr = 4.5e-4
sft_batch_size = 512
sft_epochs = 60
```

### 阶段2-5：DPO迭代训练

**DPO核心思想：**
1. 从当前策略采样K个候选解（K=128）
2. 根据路径长度排序，构建winner/loser偏好对
3. 使用DPO loss优化策略，使其偏好短路径
4. 定期更新reference model防止过度偏离

**迭代流程：**
```
for iteration in 1..100:
    1. 采样128个候选解（使用温度退火控制探索/利用）
    2. 生成32个不同难度的偏好对
    3. 训练5个epoch（每个epoch 20步）
    4. 评估模型性能
    5. 每10轮更新一次reference model
```

**DPO训练参数：**
```python
dpo_lr = 1e-4
dpo_batch_size = 48
dpo_beta = 0.3
total_iterations = 100
epochs_per_iter = 5
num_samples = 128
num_pairs_per_sample = 32
```

**温度退火策略：**
```python
use_temperature_annealing = True
temperature_start = 1.5  # 初始高探索
temperature_end = 0.8    # 最终高利用
temperature_decay = 'linear'  # linear/exponential/cosine
```

### 完整训练命令

```bash
# 1. 生成SFT数据（可选，也可在线生成）
python generate_dataset.py --num_samples 1000 --problem_size 100

# 2. 修改config.py中的配置
# - 设置sft_data_path为生成的数据文件路径
# - 调整其他超参数

# 3. 开始训练（SFT + DPO）
python train.py
```

## 文件结构

```
CVRP/
├── __init__.py              # 包初始化
├── config.py                # 配置文件（所有超参数）
├── cvrp_env.py              # CVRP环境（问题生成、长度计算）
├── model.py                 # Mamba模型架构
├── dpo_loss.py              # DPO损失函数
├── data_sampler.py          # 偏好对采样器
├── lkh_solver.py            # LKH-3求解器封装
├── heuristics.py            # 启发式算法（NN、Sweep）
├── temperature_scheduler.py # 温度调度器
├── train.py                 # 训练主脚本
├── generate_dataset.py      # 数据生成脚本
├── README.md                # 本文件
└── data/                    # 数据目录
    └── sft_data_*.pt        # 生成的SFT数据
```

## 核心组件说明

### 1. CVRPEnv（cvrp_env.py）

**功能：**
- 生成随机CVRP问题实例
- 计算路径总长度
- 验证解的可行性（容量约束）

**关键方法：**
```python
env = CVRPEnv(device)

# 生成问题
depot_xy, node_xy, node_demand = env.get_random_problems(batch_size, problem_size)

# 计算路径长度
length = env.get_tour_length(depot_xy, node_xy, tour_indices)

# 检查可行性
is_feasible, violations = env.check_feasibility(node_demand, tour_indices)
```

### 2. CVRPModel（model.py）

**架构：**
- **Encoder**: Mamba编码器，处理 [depot+nodes, 3维特征]
  - depot特征: [x, y, 0]
  - node特征: [x, y, demand]
- **Decoder**: 自回归解码器，逐步构建路径
  - 考虑当前载重
  - 应用容量约束mask
  - 自动处理depot补货

**前向传播：**
```python
model = CVRPModel(embedding_dim=256, n_encode_layers=6)

# Teacher forcing模式（SFT）
tour, log_probs = model(depot_xy, node_xy, node_demand, target_tour, teacher_forcing=True)

# 采样模式（DPO）
tour, log_probs = model(depot_xy, node_xy, node_demand, teacher_forcing=False, temperature=1.2)
```

### 3. LKHSolver（lkh_solver.py）

**功能：** 封装LKH-3求解器，生成CVRP的近最优解

**使用方法：**
```python
solver = LKHSolver(lkh_path=Config.lkh_path, runs=10)

# 求解单个问题
tour = solver.solve(depot_coord, node_coords, node_demands, capacity=1.0)

# 批量求解
tours = solver.solve_batch(depot_coords, node_coords_batch, node_demands_batch)
```

### 4. PreferenceSampler（data_sampler.py）

**功能：** 生成DPO训练所需的偏好对

**采样策略：**
1. 从模型采样K个候选解（K=128）
2. 按路径长度排序
3. 生成多个不同难度的winner/loser对：
   - 简单对：Top1 vs Bottom1（大gap）
   - 中等对：Top区 vs Mid区
   - 困难对：Top区内部（小gap）

**使用方法：**
```python
sampler = PreferenceSampler(model, env)

depot_xy, node_xy, node_demand, winner_tours, loser_tours = sampler.sample_dpo_data(
    depot_xy, node_xy, node_demand, temperature=1.2
)
```

## 配置参数说明

### 问题设置
```python
problem_size = 100        # 客户节点数量
vehicle_capacity = 1.0    # 车辆容量（归一化）
demand_min = 1            # 最小需求量
demand_max = 9            # 最大需求量
demand_scaler = 50        # 需求量缩放因子
```

### 模型设置
```python
embedding_dim = 256       # 嵌入维度
n_encode_layers = 6       # 编码器层数
```

### SFT设置
```python
sft_lr = 4.5e-4          # 学习率
sft_batch_size = 512     # 批大小
sft_epochs = 60          # 训练轮数
sft_data_path = "..."    # 数据路径（或"None"在线生成）
```

### DPO设置
```python
dpo_lr = 1e-4                    # 学习率
dpo_batch_size = 48              # 批大小
dpo_beta = 0.3                   # DPO温度参数
total_iterations = 100           # 总迭代次数
epochs_per_iter = 5              # 每次迭代的epoch数
num_samples = 128                # 采样候选解数量
num_pairs_per_sample = 32        # 每个样本生成的偏好对数
ref_update_interval = 10         # Reference model更新间隔
normalize_logp_by_tour_len = True # 是否按路径长度归一化
```

### 混合数据训练
```python
use_hybrid_data = True           # 是否启用混合训练
hybrid_offline_ratio = 0.3       # 离线数据占比
hybrid_data_path = None          # 离线数据路径（None则使用sft_data_path）
```

## CVRP与TSP的关键区别

| 维度 | TSP | CVRP |
|------|-----|------|
| **问题复杂度** | 访问所有节点一次 | 访问所有节点+容量约束+多次返回depot |
| **输入特征** | [x, y] 2维 | [x, y, demand] 3维 |
| **输出路径** | 固定长度N | 可变长度（含多次depot访问）|
| **约束** | 无 | 车辆容量约束 |
| **Mask机制** | 仅visited mask | visited mask + capacity mask |
| **状态管理** | 当前节点 | 当前节点 + 当前载重 |

## 实验建议

### 小规模快速验证（CVRP-20）
```python
problem_size = 20
sft_epochs = 20
total_iterations = 20
num_samples = 32
```

### 中等规模（CVRP-50）
```python
problem_size = 50
sft_epochs = 40
total_iterations = 50
num_samples = 64
```

### 大规模高质量（CVRP-100）
```python
problem_size = 100
sft_epochs = 60
total_iterations = 100
num_samples = 128
```

## 性能优化建议

1. **GPU显存优化：**
   - CVRP比TSP需要更多显存（额外的demand维度）
   - 减小`dpo_batch_size`或`sft_batch_size`
   - 减小`num_samples`

2. **训练速度优化：**
   - 使用预生成的LKH-3数据（避免SFT阶段在线计算）
   - 减小`epochs_per_iter`
   - 使用更小的`problem_size`先验证

3. **解质量优化：**
   - 增大`num_samples`（更多候选解）
   - 增大`num_pairs_per_sample`（更多偏好对）
   - 使用温度退火策略
   - 增加`lkh_runs`（更高质量的SFT数据）

## 常见问题

**Q: LKH-3求解失败怎么办？**
A: 检查LKH路径配置，或使用启发式算法替代（设置`sft_data_path="None"`）

**Q: 训练loss震荡严重？**
A: 尝试降低`dpo_lr`，增大`dpo_beta`，增大`ref_update_interval`

**Q: 显存不足？**
A: 减小batch_size和num_samples，或使用更小的embedding_dim

**Q: 生成的路径不可行（违反容量约束）？**
A: 增加SFT训练时间，或使用更多高质量LKH-3数据

## 参考

本项目基于以下论文和方法：
- Mamba: Linear-Time Sequence Modeling
- DPO: Direct Preference Optimization
- POMO: Policy Optimization with Multiple Optima
- LKH-3: Lin-Kernighan-Helsgaun Heuristic

## License

MIT License
