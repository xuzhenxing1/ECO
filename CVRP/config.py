import torch
import platform
import os
import glob

class Config:
    # ==================== 基础设置 ====================
    problem_size = 100      # 客户节点数量（不包括depot）
    vehicle_capacity = 1.0  # 车辆容量（归一化为1.0）
    
    # GPU选择
    gpu_id = 0
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # ==================== 模型容量设置 ====================
    embedding_dim = 256     # 嵌入维度
    n_encode_layers = 6     # 编码器层数
    
    # ==================== LKH求解器路径 ====================
    lkh_path = (
        r"D:\lkh-w\LKHWin-3.0.13\x64\Release\LKH-3.exe" 
        if platform.system() == "Windows" 
        else "LKH"
    )
    
    # ==================== Phase 0: SFT 设置 ====================
    sft_lr = 4.5e-4
    sft_batch_size = 512
    sft_epochs = 60
    
    # 自动检测data文件夹中最新的数据文件
    @staticmethod
    def get_latest_data_file(problem_size=None):
        """自动检测data文件夹中最新的数据文件"""
        # 使用当前文件所在目录的data子目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "data")
        if not os.path.exists(data_dir):
            return None
        
        # 构建搜索模式
        if problem_size is not None:
            pattern = os.path.join(data_dir, f"sft_data_*_cvrp{problem_size}_*.pt")
        else:
            pattern = os.path.join(data_dir, "sft_data_*.pt")
        
        # 查找所有匹配的文件
        files = glob.glob(pattern)
        if not files:
            return None
        
        # 返回最新的文件（按修改时间）
        latest_file = max(files, key=os.path.getmtime)
        return latest_file
    
    # SFT数据路径：None表示自动检测，或设置具体路径
    sft_data_path = None
    
    # ==================== Phase 1-4: DPO 设置 ====================
    dpo_lr = 1e-4
    dpo_batch_size = 48
    dpo_beta = 0.3
    
    total_iterations = 100
    epochs_per_iter = 5
    num_samples = 128
    num_pairs_per_sample = 32
    
    # ==================== Reference Model ====================
    update_ref_model = True
    ref_update_interval = 10
    normalize_logp_by_tour_len = True
    
    # ==================== 采样策略 ====================
    sampling_temperature = 1.2
    use_temperature_annealing = True
    temperature_start = 1.5
    temperature_end = 0.8
    temperature_decay = 'linear'
    
    # ==================== 训练监控 ====================
    dpo_log_stats = True
    dpo_log_every_steps = 10
    
    # ==================== 评估设置 ====================
    eval_batch_size = 500
    eval_use_fixed_set = True
    
    # ==================== 混合数据训练 ====================
    use_hybrid_data = True
    hybrid_offline_ratio = 0.3
    hybrid_data_path = None
    
    # ==================== CVRP特定参数 ====================
    demand_min = 1
    demand_max = 9
    
    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    else:
        demand_scaler = 50
    
    lkh_runs = 10
    lkh_max_trials = 1000
    
    # ==================== 数据目录 ====================
    # 使用当前文件所在目录作为基准
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(_current_dir, "data")         # 数据保存目录: CVRP/data/
    result_dir = os.path.join(_current_dir, "result")     # 训练结果保存目录: CVRP/result/
