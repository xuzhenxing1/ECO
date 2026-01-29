"""
稳定的DPO训练配置 - 解决震荡问题

主要改进：
1. 降低学习率，避免参数更新过大
2. 增加beta值，增强约束防止偏离reference model太远
3. 优化reference model更新策略
4. 增加采样多样性
5. 添加更多监控和诊断
"""

import torch
import platform

class Config:
    # 基础设置
    tsp_size = 100         # 城市数量（建议先用小规模调试，如20或50）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # LKH求解器路径配置
    lkh_path = (
        r"D:\lkh-w\LKHWin-3.0.13\x64\Release\LKH-3.exe" 
        if platform.system() == "Windows" 
        else "LKH"
    )
    
    # ==================== Phase 0: SFT 设置 ====================
    sft_lr = 5e-4
    sft_batch_size = 64
    sft_epochs = 10         # 充分训练SFT，打好基础
    sft_data_path = "data/sft_data_lkh_tsp100_n1000_20260114_140122.pt"
    
    # ==================== Phase 1-4: DPO 设置 ====================
    # 🔧 关键参数1: 学习率 - 降低学习率提高稳定性
    dpo_lr = 1e-4  # ✓ 从3e-4降到1e-4，减少震荡
    
    # 🔧 关键参数2: Beta - 控制与reference model的偏离程度
    dpo_beta = 0.3  # ✓ 从0.1增加到0.3，增强约束（beta越大越保守）
    
    # 🔧 关键参数3: 批大小
    dpo_batch_size = 64  # ✓ 从128降到64，更稳定的梯度估计
    
    # 🔧 关键参数4: 迭代参数
    total_iterations = 100
    epochs_per_iter = 1  # ✓ 从2降到1，避免在同一数据上过拟合
    
    # 🔧 关键参数5: 采样数量
    num_samples = 16  # ✓ 从8增加到16，增加多样性，winner/loser差异更明显
    
    # ==================== Reference Model 更新策略 ====================
    update_ref_model = True
    ref_update_interval = 10  # ✓ 从20降到10，更频繁同步但不是每轮
    
    # 🔧 新增：使用EMA (Exponential Moving Average) 更新reference
    use_ref_ema = True  # 使用指数移动平均而不是直接复制
    ref_ema_decay = 0.95  # EMA衰减系数
    
    # ==================== 数值稳定性 ====================
    normalize_logp_by_tour_len = True
    gradient_clip_norm = 0.5  # ✓ 从1.0降到0.5，更强的梯度裁剪
    
    # ==================== 监控和诊断 ====================
    dpo_log_stats = True  # ✓ 开启详细日志
    dpo_log_every_steps = 10  # ✓ 更频繁的日志
    
    # 🔧 新增：Early stopping
    use_early_stopping = True
    patience = 15  # 15个iteration没有改善就停止
    min_delta = 0.001  # 最小改善幅度
    
    # ==================== 评估设置 ====================
    eval_batch_size = 1000  # ✓ 增加到1000，更准确的评估
    eval_use_fixed_set = True
    eval_every_n_iters = 1  # 每个iteration都评估
    
    # ==================== 温度采样（提高多样性）====================
    use_temperature_sampling = True
    sampling_temperature = 1.5  # 温度越高，采样越随机
    
    # ==================== Warmup策略 ====================
    use_lr_warmup = True
    warmup_iterations = 5  # 前5个iteration线性增加学习率
    
    # ==================== 保存策略 ====================
    save_every_n_iters = 5
    save_best_only = False  # 保存所有checkpoint便于回滚
