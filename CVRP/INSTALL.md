# CVRP Mamba+DPO 训练项目 - 快速开始指南

## 📋 项目简介

本项目将TSP的Mamba+DPO训练框架完整扩展到CVRP（带容量约束的车辆路径问题）。

**核心特性：**
- 🚀 基于Mamba的O(N)高效编码器
- 🎯 DPO直接偏好优化训练
- 📊 LKH-3生成高质量训练数据
- 🔥 完整的SFT+DPO训练流程

## 🛠️ 安装步骤

### 1. 环境要求
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (推荐GPU训练)

### 2. 安装依赖
```bash
pip install torch numpy tqdm
```

### 3. 安装LKH-3（可选但推荐）

**Windows:**
1. 下载 LKH-3.exe
2. 在 `config.py` 中设置路径：
   ```python
   lkh_path = r"D:\lkh-w\LKHWin-3.0.13\x64\Release\LKH-3.exe"
   ```

**Linux:**
```bash
# 下载并编译LKH-3
wget http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.9.tgz
tar -xzf LKH-3.0.9.tgz
cd LKH-3.0.9
make

# 添加到PATH或在config.py中设置路径
```

### 4. 初始化目录
```bash
cd CVRP
python init_dirs.py
```

## 🚀 快速开始

### 方式1: 一键演示（推荐新手）
```bash
python quick_start.py --mode demo
```
这将运行CVRP-20的快速演示（约30分钟），包括数据生成和训练。

### 方式2: 测试组件
```bash
python test.py
```
验证所有组件是否正常工作。

### 方式3: 完整训练流程

#### Step 1: 生成训练数据
```bash
# CVRP-20 (快速验证)
python generate_dataset.py --num_samples 100 --problem_size 20

# CVRP-100 (正式训练)
python generate_dataset.py --num_samples 1000 --problem_size 100
```

#### Step 2: 配置训练参数
编辑 `config.py`：
```python
# 设置数据路径
sft_data_path = "data/sft_data_lkh_cvrp100_n1000_xxx.pt"

# 或使用预设配置
from configs.config_presets import ConfigPresets
ConfigPresets.apply_preset(Config, 'cvrp100')
```

#### Step 3: 开始训练
```bash
python train.py
```

## 📊 训练流程详解

### Phase 0: SFT (Supervised Fine-Tuning)
**目标：** 学习基础CVRP规则
- 如何返回depot补货
- 如何遵守容量约束
- 基本路径构建逻辑

**训练方式：**
- 使用LKH-3生成的高质量数据（推荐）
- 或使用启发式算法在线生成

**预期时间：** 30分钟 - 2小时

### Phase 1-4: DPO (Direct Preference Optimization)
**目标：** 优化路径质量
- 从模型采样多个候选解
- 构建winner/loser偏好对
- 使用DPO loss优化

**训练流程：**
```
for iteration in 1..100:
    1. 采样128个候选解
    2. 生成32个偏好对
    3. 训练5个epoch
    4. 评估性能
    5. 更新reference model
```

**预期时间：** 2-10小时（取决于问题规模）

## 📁 文件结构

```
CVRP/
├── 核心文件
│   ├── config.py              # 配置参数
│   ├── cvrp_env.py            # CVRP环境
│   ├── model.py               # Mamba模型
│   ├── train.py               # 训练脚本
│   └── ...
├── 工具脚本
│   ├── generate_dataset.py    # 数据生成
│   ├── test.py                # 组件测试
│   ├── quick_start.py         # 快速开始
│   └── init_dirs.py           # 初始化目录
├── 文档
│   ├── README.md              # 基础说明
│   ├── PROJECT_GUIDE.md       # 详细指南
│   ├── INSTALL.md             # 本文件
│   └── SUMMARY.md             # 项目总结
└── 目录
    ├── data/                  # 训练数据
    ├── result/                # 训练结果
    └── configs/               # 配置预设
```

## ⚙️ 配置说明

### 快速配置（CVRP-20）
```python
problem_size = 20
sft_epochs = 10
total_iterations = 10
num_samples = 32
embedding_dim = 128
n_encode_layers = 3
```

### 标准配置（CVRP-100）
```python
problem_size = 100
sft_epochs = 60
total_iterations = 100
num_samples = 128
embedding_dim = 256
n_encode_layers = 6
```

### 使用预设配置
```python
from configs.config_presets import ConfigPresets

# 应用CVRP-20预设
ConfigPresets.apply_preset(Config, 'cvrp20')

# 应用CVRP-50预设
ConfigPresets.apply_preset(Config, 'cvrp50')

# 应用CVRP-100预设
ConfigPresets.apply_preset(Config, 'cvrp100')
```

## 🐛 常见问题

### Q1: LKH-3求解失败
**解决：**
- 检查LKH路径配置
- 或设置 `sft_data_path = "None"` 使用启发式算法

### Q2: 显存不足
**解决：**
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

### Q3: 训练loss不下降
**解决：**
```python
# 增大学习率
sft_lr = 5e-4
dpo_lr = 1.5e-4

# 延长SFT训练
sft_epochs = 80

# 使用LKH-3数据
sft_data_path = "data/sft_data_lkh_cvrp100_n1000.pt"
```

### Q4: 生成的路径不可行
**解决：**
- 增加SFT训练时间
- 使用更多高质量LKH-3数据
- 检查mask逻辑是否正确

## 📈 性能预期

| 问题规模 | 训练时间 | SFT数据 | DPO迭代 | 预期Gap |
|---------|---------|---------|---------|---------|
| CVRP-20 | ~30分钟  | 50-100  | 10-20   | 5-10%   |
| CVRP-50 | ~2-4小时 | 500-1000| 50      | 3-8%    |
| CVRP-100| ~6-12小时| 1000-2000| 100    | 2-5%    |

*Gap: 相对于LKH-3求解的差距

## 📚 进一步学习

- **详细指南：** [PROJECT_GUIDE.md](PROJECT_GUIDE.md)
- **使用说明：** [README.md](README.md)
- **项目总结：** [SUMMARY.md](SUMMARY.md)

## 🔗 相关资源

- [TSP项目](../TSP/)
- [CVRP-POMO参考](../CVRP-POMO/)
- [Mamba论文](https://arxiv.org/abs/2312.00752)
- [DPO论文](https://arxiv.org/abs/2305.18290)
- [LKH-3主页](http://webhotel4.ruc.dk/~keld/research/LKH-3/)

## 💡 使用建议

### 第一次使用
```bash
# 1. 测试组件
python test.py

# 2. 运行演示
python quick_start.py --mode demo

# 3. 查看结果
ls result/
```

### 正式实验
```bash
# 1. 生成数据
python generate_dataset.py --num_samples 1000 --problem_size 100

# 2. 修改配置
# 编辑 config.py，设置 sft_data_path

# 3. 完整训练
python train.py

# 4. 评估结果
# 检查 result/train_xxx/ 目录
```

## ✅ 检查清单

安装完成后，请确认：
- [ ] Python环境正确（3.8+）
- [ ] PyTorch安装成功
- [ ] 目录结构已创建（data/, result/, configs/）
- [ ] 测试脚本运行成功（python test.py）
- [ ] LKH-3可用（可选，或使用启发式算法）

## 🎯 下一步

1. ✅ 运行 `python test.py` 验证安装
2. ✅ 尝试 `python quick_start.py --mode demo`
3. ✅ 阅读 `PROJECT_GUIDE.md` 了解详细原理
4. ✅ 开始自己的实验！

---

**祝你训练顺利！** 🚀
