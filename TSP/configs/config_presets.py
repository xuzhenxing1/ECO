"""
性能优化配置预设

包含不同场景下的推荐配置：
- 快速训练（低资源）
- 标准训练（平衡）
- 高质量训练（高资源）
"""

import torch
import platform

class ConfigFast:
    """快速训练配置 - 适合调试和快速实验"""
    
    tsp_size = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lkh_path = (
        r"D:\lkh-w\LKHWin-3.0.13\x64\Release\LKH-3.exe" 
        if platform.system() == "Windows" 
        else "LKH"
    )
    
    # 模型容量：小
    embedding_dim = 128
    n_encode_layers = 3
    
    # SFT 设置
    sft_lr = 5e-4
    sft_batch_size = 64
    sft_epochs = 5  # 快速训练
    sft_data_path = None
    
    # DPO 设置
    dpo_lr = 1e-4
    dpo_batch_size = 64
    dpo_beta = 0.3
    
    total_iterations = 50  # 减少迭代
    epochs_per_iter = 1
    num_samples = 16  # 较少采样
    sampling_temperature = 1.0
    
    update_ref_model = True
    ref_update_interval = 10
    normalize_logp_by_tour_len = True
    
    dpo_log_stats = True
    dpo_log_every_steps = 10
    
    # 评估设置
    eval_batch_size = 500
    eval_use_fixed_set = True
    use_pomo_eval = False  # 快速评估
    pomo_size = 10


class ConfigStandard:
    """标准训练配置 - 推荐的平衡配置"""
    
    tsp_size = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lkh_path = (
        r"D:\lkh-w\LKHWin-3.0.13\x64\Release\LKH-3.exe" 
        if platform.system() == "Windows" 
        else "LKH"
    )
    
    # 模型容量：中
    embedding_dim = 128
    n_encode_layers = 3
    
    # SFT 设置
    sft_lr = 5e-4
    sft_batch_size = 64
    sft_epochs = 10
    sft_data_path = None  # 建议使用 LKH 数据
    
    # DPO 设置
    dpo_lr = 1e-4
    dpo_batch_size = 64
    dpo_beta = 0.3
    
    total_iterations = 100
    epochs_per_iter = 1
    num_samples = 32  # 增加采样
    sampling_temperature = 1.0
    
    update_ref_model = True
    ref_update_interval = 10
    normalize_logp_by_tour_len = True
    
    dpo_log_stats = True
    dpo_log_every_steps = 10
    
    # 评估设置
    eval_batch_size = 1000
    eval_use_fixed_set = True
    use_pomo_eval = True  # 启用 POMO
    pomo_size = 20


class ConfigHighQuality:
    """高质量训练配置 - 追求最佳性能（需要更多资源）"""
    
    tsp_size = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lkh_path = (
        r"D:\lkh-w\LKHWin-3.0.13\x64\Release\LKH-3.exe" 
        if platform.system() == "Windows" 
        else "LKH"
    )
    
    # 模型容量：大
    embedding_dim = 256  # 增大
    n_encode_layers = 6  # 增深
    
    # SFT 设置
    sft_lr = 3e-4  # 更小的学习率
    sft_batch_size = 32  # 减小 batch size 以适应更大模型
    sft_epochs = 15
    sft_data_path = None  # **必须**使用 LKH 数据
    
    # DPO 设置
    dpo_lr = 5e-5  # 更小的学习率
    dpo_batch_size = 32
    dpo_beta = 0.5  # 更强的约束
    
    total_iterations = 200  # 更多迭代
    epochs_per_iter = 1
    num_samples = 64  # 更多采样
    sampling_temperature = 1.1  # 增加探索
    
    update_ref_model = True
    ref_update_interval = 5  # 更频繁更新
    normalize_logp_by_tour_len = True
    
    dpo_log_stats = True
    dpo_log_every_steps = 5
    
    # 评估设置
    eval_batch_size = 2000
    eval_use_fixed_set = True
    use_pomo_eval = True
    pomo_size = 50  # 更多起点


class ConfigTSP100:
    """TSP-100 专用配置"""
    
    tsp_size = 100  # 更大规模
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lkh_path = (
        r"D:\lkh-w\LKHWin-3.0.13\x64\Release\LKH-3.exe" 
        if platform.system() == "Windows" 
        else "LKH"
    )
    
    # 模型容量：大（TSP-100 需要更强的模型）
    embedding_dim = 256
    n_encode_layers = 6
    
    # SFT 设置
    sft_lr = 3e-4
    sft_batch_size = 16  # TSP-100 占用更多内存
    sft_epochs = 20
    sft_data_path = None  # 必须使用 LKH 数据
    
    # DPO 设置
    dpo_lr = 3e-5  # 更小的学习率
    dpo_batch_size = 16
    dpo_beta = 0.5
    
    total_iterations = 300
    epochs_per_iter = 1
    num_samples = 32  # 平衡性能和内存
    sampling_temperature = 1.2  # TSP-100 需要更多探索
    
    update_ref_model = True
    ref_update_interval = 5
    normalize_logp_by_tour_len = True
    
    dpo_log_stats = True
    dpo_log_every_steps = 5
    
    # 评估设置
    eval_batch_size = 500  # TSP-100 评估更慢
    eval_use_fixed_set = True
    use_pomo_eval = True
    pomo_size = 40


# ========== 使用指南 ==========
"""
如何使用这些配置：

1. 在 train.py 中导入：
   from config_presets import ConfigStandard as Config
   
2. 或者复制到 config.py：
   将 ConfigStandard 的内容复制到 config.py 的 Config 类中

3. 根据场景选择：
   - 调试/快速实验：ConfigFast
   - 正常训练：ConfigStandard
   - 追求性能：ConfigHighQuality
   - TSP-100：ConfigTSP100

4. 生成对应的 LKH 数据：
   
   # TSP-50 标准配置
   python generate_sft_data_lkh.py --tsp_size 50 --num_samples 5000 --lkh_runs 10
   
   # TSP-100 高质量配置
   python generate_sft_data_lkh.py --tsp_size 100 --num_samples 3000 --lkh_runs 15

5. 训练流程：
   
   # 第一步：生成 LKH 数据
   python generate_sft_data_lkh.py --tsp_size 50 --num_samples 5000 --lkh_runs 10
   
   # 第二步：修改 config.py 设置 sft_data_path
   # sft_data_path = "data/sft_data_lkh_tsp50_n5000_xxx.pt"
   
   # 第三步：训练
   python train.py
   
   # 第四步：测试性能
   python test_performance.py
"""
