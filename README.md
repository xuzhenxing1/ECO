# Graph Mamba-DPO for Traveling Salesperson Problem

A novel approach combining **State Space Models (Mamba)** with **Direct Preference Optimization (DPO)** for solving large-scale TSP instances (up to TSP-1000).

## 🚀 Key Features

- **Linear Complexity**: Mamba-based encoder achieves O(N) complexity vs O(N²) in Transformers
- **Stable Training**: DPO eliminates high-variance RL gradients (no value network needed)
- **Iterative Self-Improvement**: Progressive policy refinement through iterative reference updates
- **Hybrid Data Training**: Combines expert solutions (LKH) with self-generated explorations

## 📁 Project Structure

```
tsp_dpo-mamba-1.14/
├── 🎯 Core Components
│   ├── model.py              # Graph Mamba Encoder-Decoder architecture
│   ├── train.py              # Main training pipeline (SFT + Iterative DPO)
│   ├── config.py             # Configuration parameters
│   ├── tsp_env.py            # TSP environment (problem generation & evaluation)
│   ├── dpo_loss.py           # Direct Preference Optimization loss
│   └── data_sampler.py       # Preference pair sampling strategy
│
├── 🔧 Utilities
│   ├── heuristics.py         # Nearest neighbor & greedy heuristics
│   ├── lkh_solver.py         # LKH-3 solver wrapper
│   └── temperature_scheduler.py  # Temperature annealing scheduler
│
├── 📂 Organized Folders
│   ├── scripts/              # Data generation scripts
│   │   ├── generate_sft_data.py
│   │   ├── generate_sft_data_lkh.py
│   │   └── generate_sft_data_lehd.py
│   │
│   ├── tests/                # Unit & integration tests
│   │   ├── test.py
│   │   ├── test_lkh.py
│   │   ├── test_lehd_setup.py
│   │   └── test_performance.py
│   │
│   ├── debug/                # Debugging & diagnostic tools
│   │   ├── quick_debug_dpo.py
│   │   ├── smoke_dpo.py
│   │   ├── diagnose_training.py
│   │   ├── dpo_diagnostics.py
│   │   └── compare_algorithms.py
│   │
│   ├── demos/                # Demo scripts
│   │   ├── demo_workflow.py
│   │   └── demo_temperature.py
│   │
│   ├── configs/              # Legacy configurations
│   │   ├── config_stable.py
│   │   ├── config_presets.py
│   │   └── train_stable_reference.py
│   │
│   └── docs/                 # Documentation
│       ├── README_DATA.md
│       ├── LKH_GUIDE.md
│       ├── LEHD_GUIDE.md
│       ├── OPTIMIZATION_GUIDE.md
│       └── 优化总结.md
│
├── 💾 Data & Results
│   ├── data/                 # Pre-generated SFT datasets
│   ├── result/               # Training outputs & checkpoints
│   └── LEHD/                 # LEHD reference implementation
│
└── __pycache__/              # Python cache files
```

## 🏃 Quick Start

### 1. Generate High-Quality SFT Data (Optional)
```bash
# Using LKH solver (recommended for best quality)
python scripts/generate_sft_data_lkh.py

# Or using nearest neighbor (faster, lower quality)
python scripts/generate_sft_data.py
```

### 2. Train the Model
```bash
# Edit config.py to set:
# - tsp_size (e.g., 100, 500, 1000)
# - sft_data_path (path to generated data or "None" for online generation)
# - GPU settings

python train.py
```

### 3. Evaluate Performance
```bash
python tests/test_performance.py
```

## 📊 Model Architecture

### Encoder: Graph Mamba
- **Input**: 2D coordinates → Linear projection to D-dim
- **Processing**: L stacked Mamba layers (d_state=16, d_conv=4, expand=2)
- **Normalization**: RMSNorm after final layer
- **Complexity**: O(N) linear time & memory

### Decoder: Recurrent SSM
- **State**: Fixed-size hidden state (eliminates KV-cache overhead)
- **Scoring**: Bilinear pointer mechanism (query-key matching)
- **Masking**: Enforces valid Hamiltonian cycle constraints

## 🎓 Training Paradigm

### Phase 0: Supervised Fine-Tuning (SFT)
- **Goal**: Learn TSP validity constraints
- **Teacher**: LKH solver or greedy heuristics
- **Loss**: Negative log-likelihood (behavior cloning)

### Phase 1-4: Iterative DPO
1. **Sample**: Generate K candidates per instance
2. **Label**: Construct winner/loser pairs (shorter vs longer tours)
3. **Update**: Optimize DPO objective with KL regularization
4. **Sync**: Periodically update reference model (every T iterations)

## 🔬 Key Innovations

1. **Linear-Time Graph Encoding**: First application of Mamba to combinatorial optimization
2. **DPO for CO**: Novel adaptation of LLM alignment techniques to discrete optimization
3. **Hybrid Curriculum**: Balances expert guidance (offline data) with exploration (self-play)
4. **Iterative Reference Update**: Enables progressive self-improvement beyond static baselines



## 🛠️ Configuration Highlights

Key parameters in `config.py`:
- `embedding_dim`: 128/256 (model capacity)
- `n_encode_layers`: 3/6 (depth)
- `dpo_beta`: 0.3 (KL penalty weight)
- `num_samples`: 128 (candidates per instance)
- `ref_update_interval`: 10 (reference sync frequency)

## 📝 Citation


## 📧 Contact



## 📜 License

MIT License - See LICENSE file for details
