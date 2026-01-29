# CVRP项目完整指南

## 项目架构总览

本项目将TSP的Mamba+DPO训练框架完整扩展到CVRP任务，包含从数据生成到模型训练的全流程。

## 目录结构

```
CVRP/
├── config.py                    # 核心配置文件
├── cvrp_env.py                  # CVRP环境（问题生成、长度计算、可行性检查）
├── model.py                     # Mamba模型架构
├── dpo_loss.py                  # DPO损失函数
├── data_sampler.py              # 偏好对采样器
├── lkh_solver.py                # LKH-3求解器封装
├── heuristics.py                # 启发式算法（NN、Sweep）
├── temperature_scheduler.py     # 温度调度器
├── train.py                     # 主训练脚本
├── generate_dataset.py          # 数据生成脚本
├── test.py                      # 组件测试脚本
├── README.md                    # 使用说明
├── PROJECT_GUIDE.md             # 本文件
├── configs/
│   └── config_presets.py        # 预设配置（CVRP-20/50/100）
├── data/                        # 数据目录
│   └── *.pt                     # 生成的数据集
└── result/                      # 训练结果目录
    └── train_*/                 # 每次训练的结果
```

## 完整工作流程

### Step 1: 环境准备

1. **安装依赖**
```bash
pip install torch numpy tqdm
```

2. **安装LKH-3（可选但推荐）**
   - Windows: 下载LKH-3.exe，在`config.py`中设置路径
   - Linux: 编译安装LKH-3，确保在PATH中

3. **验证环境**
```bash
cd CVRP
python test.py
```

### Step 2: 数据生成（使用LKH-3）

**方式1：生成小数据集（快速验证）**
```bash
python generate_dataset.py --num_samples 100 --problem_size 20
```

**方式2：生成中等数据集**
```bash
python generate_dataset.py --num_samples 500 --problem_size 50
```

**方式3：生成大数据集（推荐，用于正式训练）**
```bash
python generate_dataset.py --num_samples 1000 --problem_size 100
```

**生成的数据文件：**
- 位置：`data/sft_data_lkh_cvrp{size}_n{num}_{timestamp}.pt`
- 包含：问题实例（depot、nodes、demands）+ LKH-3解（tours、lengths）

### Step 3: 配置训练参数

**方式1：直接修改config.py**
```python
# 设置问题规模
problem_size = 100

# 设置SFT数据路径
sft_data_path = "data/sft_data_lkh_cvrp100_n1000_xxx.pt"

# 调整其他参数...
```

**方式2：使用预设配置**
```python
from configs.config_presets import ConfigPresets
ConfigPresets.apply_preset(Config, 'cvrp100')
```

### Step 4: 训练模型

```bash
python train.py
```

**训练过程：**
1. **Phase 0 - SFT (约30-60分钟)**
   - 加载LKH-3数据或在线生成启发式解
   - 训练模型模仿高质量路径
   - 学习基本CVRP规则（容量约束、depot补货）

2. **Phase 1-4 - DPO (约2-6小时)**
   - 100轮迭代训练
   - 每轮：采样128个候选→构建偏好对→优化模型→评估
   - 温度从1.5退火到0.8（探索→利用）

**训练输出：**
- `result/train_{timestamp}_cvrp{size}/`
  - `sft_cvrp_model.pth` - SFT阶段模型
  - `best_cvrp_model.pth` - 最佳模型
  - `checkpoint_iter*.pth` - 定期checkpoint
  - `training_history.pt` - 训练历史

### Step 5: 评估和分析

```python
import torch
from config import Config
from cvrp_env import CVRPEnv
from model import CVRPModel

# 加载最佳模型
model = CVRPModel(
    embedding_dim=Config.embedding_dim,
    n_encode_layers=Config.n_encode_layers
).to(Config.device)

model.load_state_dict(torch.load("result/train_xxx/best_cvrp_model.pth"))
model.eval()

# 生成测试问题
env = CVRPEnv(Config.device)
depot_xy, node_xy, node_demand = env.get_random_problems(100, Config.problem_size)

# 求解
with torch.no_grad():
    tours, _ = model(depot_xy, node_xy, node_demand, teacher_forcing=False)
    lengths = env.get_tour_length(depot_xy, node_xy, tours)

print(f"Average length: {lengths.mean().item():.4f}")
print(f"Std length: {lengths.std().item():.4f}")
```

## 核心算法详解

### 1. CVRP问题定义

**输入：**
- Depot坐标：$(x_0, y_0)$
- 客户节点坐标：$(x_1, y_1), ..., (x_N, y_N)$
- 客户需求量：$d_1, ..., d_N$ （归一化到[0,1]）
- 车辆容量：$C = 1.0$

**目标：** 最小化总路径长度，满足：
1. 所有客户恰好被访问一次
2. 每条子路径的总需求 ≤ 车辆容量
3. 每条子路径从depot开始并返回depot

**输出：** 访问序列，例如：
```
[0, 3, 5, 8, 0, 1, 2, 6, 0, 4, 7, 9, 0]
  └─────路径1────┘ └────路径2───┘ └─路径3─┘
```

### 2. Mamba编码器

**输入特征：**
```python
# Depot: [x, y, 0]
depot_features = [depot_x, depot_y, 0]

# Customers: [x, y, demand]
customer_features = [node_x, node_y, node_demand]

# 合并: [B, N+1, 3]
all_features = concat([depot_features, customer_features])
```

**编码过程：**
```python
# Linear投影: [B, N+1, 3] -> [B, N+1, D]
h = linear_embed(all_features)

# Mamba层（L层）
for layer in mamba_layers:
    h = layer(h)  # 线性时间O(N)，保持形状

# 输出: [B, N+1, D]
embeddings = layer_norm(h)
```

### 3. 自回归解码器

**状态维护：**
- `current_load`: 当前载重（剩余容量）
- `visited_mask`: 已访问的客户节点
- `ssm_state`: SSM隐状态

**单步解码：**
```python
for step in range(max_steps):
    # 1. 更新decoder状态（结合载重信息）
    decoder_vec = SSM_update(last_node_emb + capacity_emb)
    
    # 2. 计算每个节点的得分
    scores = dot(decoder_vec, node_embeddings)
    
    # 3. 应用mask
    scores[visited_mask] = -inf  # 已访问不能再访问
    scores[current_load < demand] = -inf  # 容量不足不能访问
    scores[depot] = valid  # depot永远可访问（补货）
    
    # 4. 采样或贪婪选择
    probs = softmax(scores / temperature)
    selected = sample(probs)
    
    # 5. 更新状态
    if selected == depot:
        current_load = 1.0  # 补满
    else:
        current_load -= demand[selected]
        visited_mask[selected] = True
    
    # 6. 终止条件：所有客户访问完且回到depot
    if all_visited and selected == depot:
        break
```

### 4. DPO训练原理

**核心思想：** 让模型偏好短路径，避免长路径

**偏好对构建：**
```python
# 1. 采样K个候选解
tours = [sample_from_model() for _ in range(K)]

# 2. 计算每个解的长度
lengths = [compute_length(tour) for tour in tours]

# 3. 排序并构建偏好对
sorted_tours = sort_by_length(tours)
winner = sorted_tours[0]    # 最短路径
loser = sorted_tours[-1]    # 最长路径
```

**DPO Loss：**
```python
# 1. 计算log概率
log_pi_w = model.log_prob(winner)
log_pi_l = model.log_prob(loser)
log_ref_w = ref_model.log_prob(winner)
log_ref_l = ref_model.log_prob(loser)

# 2. 计算log ratio
logr_w = log_pi_w - log_ref_w
logr_l = log_pi_l - log_ref_l

# 3. DPO loss
loss = -log_sigmoid(beta * (logr_w - logr_l))
```

**直观理解：**
- 增大 $\pi(winner)$ 相对于 $ref(winner)$ 的概率
- 减小 $\pi(loser)$ 相对于 $ref(loser)$ 的概率
- $\beta$ 控制相对于reference的偏离程度

### 5. 温度退火策略

**目的：** 平衡探索与利用

**策略：**
```python
# 初期（iter 0-33）：高温探索
temperature = 1.5  # 更随机，发现多样解

# 中期（iter 34-66）：平衡
temperature = 1.15  # 逐渐降低

# 后期（iter 67-100）：低温利用  
temperature = 0.8  # 更确定，集中优化
```

**效果：**
- 避免过早收敛到局部最优
- 后期稳定地优化解质量

## 关键技术细节

### 1. Mask机制

**CVRP需要两种mask：**

```python
# Visited mask: 客户节点只能访问一次
visited_mask[selected_customer] = True

# Capacity mask: 容量不足的节点不能访问
capacity_mask = (current_load < node_demand)

# Depot特殊处理
capacity_mask[depot] = False  # depot永远可访问
visited_mask[depot] = False   # depot可重复访问

# 综合mask
final_scores[visited_mask | capacity_mask] = -inf
```

### 2. 路径长度归一化

**问题：** CVRP路径长度可变（含多次depot访问），log_prob的尺度随路径长度变化

**解决：** Per-step归一化
```python
# 计算实际路径长度
tour_len = (tour != 0).sum()  # 不计padding

# 归一化log_prob
normalized_logp = sum_logp / tour_len
```

### 3. 混合数据训练

**策略：** 30%离线高质量数据 + 70%在线生成数据

**优势：**
- 离线数据（LKH-3）：提供高质量基准
- 在线数据（模型采样）：提供探索多样性

```python
# 离线数据
depot_off, nodes_off, demands_off = load_from_lkh_dataset()

# 在线数据
depot_on, nodes_on, demands_on = generate_random()

# 混合
depot = concat([depot_off, depot_on])
```

## 性能对比

### 与TSP的区别

| 维度 | TSP | CVRP |
|------|-----|------|
| 输入维度 | 2 (x,y) | 3 (x,y,demand) |
| 路径长度 | 固定N | 可变(N~2N) |
| 约束 | 无 | 容量约束 |
| Mask | 1种 | 2种 |
| 状态 | 1维 | 2维(位置+载重) |
| 复杂度 | 中 | 高 |

### 与CVRP-POMO的对比

| 维度 | CVRP-POMO | 本项目 |
|------|-----------|--------|
| 编码器 | Attention | **Mamba** |
| 解码器 | Attention | **SSM** |
| 训练 | RL (REINFORCE) | **SFT+DPO** |
| 复杂度 | O(N²) | **O(N)** |
| 显存 | 高 | **低** |
| 训练稳定性 | 一般 | **好** |

## 常见问题诊断

### 问题1：训练loss不下降

**可能原因：**
- 学习率太低
- batch_size太小
- SFT训练不充分

**解决方案：**
```python
# 增大学习率
sft_lr = 5e-4  # 原4.5e-4
dpo_lr = 1.5e-4  # 原1e-4

# 增大batch size
sft_batch_size = 768  # 原512
dpo_batch_size = 64  # 原48

# 增加SFT epochs
sft_epochs = 80  # 原60
```

### 问题2：生成的路径不可行

**可能原因：**
- SFT数据质量差
- Mask逻辑错误
- 模型容量不足

**解决方案：**
```python
# 使用LKH-3生成高质量SFT数据
sft_data_path = "data/sft_data_lkh_cvrp100_n1000.pt"

# 增加模型容量
embedding_dim = 384  # 原256
n_encode_layers = 8  # 原6

# 延长SFT训练
sft_epochs = 100
```

### 问题3：DPO阶段震荡

**可能原因：**
- DPO学习率太高
- Reference model更新太频繁
- Beta参数不合适

**解决方案：**
```python
# 降低学习率
dpo_lr = 5e-5  # 原1e-4

# 减少更新频率
ref_update_interval = 20  # 原10

# 增大beta（增强约束）
dpo_beta = 0.5  # 原0.3
```

### 问题4：显存不足

**解决方案：**
```python
# 减小batch size
sft_batch_size = 256
dpo_batch_size = 32

# 减小模型
embedding_dim = 128
n_encode_layers = 3

# 减少采样数
num_samples = 64
```

## 实验建议

### 快速验证流程（CVRP-20，1小时）

```python
# 1. 生成少量数据
python generate_dataset.py --num_samples 100 --problem_size 20

# 2. 快速训练
problem_size = 20
sft_epochs = 10
total_iterations = 10

# 3. 验证训练
python test.py
python train.py
```

### 正式实验流程（CVRP-100，6-12小时）

```python
# 1. 生成高质量数据
python generate_dataset.py --num_samples 1000 --problem_size 100

# 2. 完整训练
python train.py

# 3. 评估性能
# 对比LKH-3、启发式算法、训练模型
```

## 进阶优化

### 1. 多GPU训练

```python
# train.py
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
```

### 2. 学习率调度

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=Config.total_iterations
)
```

### 3. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = ...
    
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 总结

本项目成功将TSP的Mamba+DPO框架扩展到CVRP，核心创新：

1. **O(N)复杂度**：Mamba编码器替代Attention
2. **容量感知解码**：结合载重信息的自回归解码
3. **DPO优化**：直接优化路径长度偏好
4. **混合训练**：LKH-3数据+在线采样

**适用场景：**
- 大规模CVRP问题（N≥100）
- 需要快速推理的实时场景
- 资源受限环境（低显存）

**未来方向：**
- 扩展到VRPTW（带时间窗）
- 多depot CVRP
- 动态CVRP
