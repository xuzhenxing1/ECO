import torch
import platform

class Config:
    # 基础设置
    tsp_size = 500       # 城市数量
    
    # GPU选择：修改gpu_id来指定使用哪张GPU
    gpu_id = 0  # 0表示第一张GPU, 1表示第二张GPU, 等等
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # 模型容量设置（增大可提升性能，但训练更慢）
    # TSP-500需要更大模型才能充分学习
    embedding_dim = 256     # ✓ 从128增加到256，提升表达能力
    n_encode_layers = 6     # ✓ 从3增加到6，更深的网络学习复杂模式
                            # 注意：模型参数量增加约4倍，需要更多GPU内存
    
    # LKH求解器路径配置（用于生成高质量SFT数据）
    # Windows: 指定完整路径
    # Ubuntu: 如果LKH在系统PATH中，使用 "LKH" 即可
    lkh_path = (
        r"D:\lkh-w\LKHWin-3.0.13\x64\Release\LKH-3.exe" 
        if platform.system() == "Windows" 
        else "LKH"  # Ubuntu/Linux
    )
    
    # Phase 0: SFT 设置
    # TSP-500 + 大模型 + 100k数据 + 80GB GPU配置
    sft_lr = 4.5e-4         # ✓ batch_size翻倍 → 学习率增加1.5倍 (Linear Scaling Rule)
    sft_batch_size = 1024   # ✓ 从512→1024，充分利用80GB显存 (512时仅用33GB)
                            # 预计显存占用：65-70GB，每个epoch = 100k/1024 ≈ 98步
    sft_epochs = 60         # ✓ batch_size翻倍，步数减半，减少epochs加速收敛
                            # 总训练步数：98×60 ≈ 5880步 vs 之前 195×80 ≈ 15600步
    sft_data_path = "None"   # SFT数据文件路径，如果为None则在线生成数据
                            # 建议使用高质量数据：LEHD或LKH生成的数据
    
    # Phase 1-4: Iterative DPO 设置
    # 🔧 关键参数优化 - 解决震荡问题
    dpo_lr = 1e-4           # ✓ 从3e-4降到1e-4，减少震荡
    dpo_batch_size = 64     # ✓ 从96降到64，避免采样时tensor索引溢出
                            # 计算：64 × 128 × 500 × 256 ≈ 1.05B < 2.14B ✓
                            # 注意：使用hybrid_data时，在线数据会扩展num_samples倍
                            # 实际在线batch：64×0.7=45，采样tensor：45×128×500×256≈737M
    dpo_beta = 0.3          # ✓ 从0.1增加到0.3，增强约束
    
    # 核心迭代参数
    total_iterations = 100  # 总共进行多少轮 "采样-训练-更新Ref" 的大循环
    epochs_per_iter = 5     # ✓ 从1增加到5，充分利用采样的128个候选解
                            # 注意：train.py中实际步数 = epochs_per_iter × 20
                            # 5×20=100步/迭代，充分学习每批采样数据
    
    num_samples = 128       # ✓ 从32增加到128，显著提升DPO多样性
                            # Tensor索引检查：96 × 128 × 0.7 × 500 × 256 ≈ 1.1B < 2.14B ✓
                            # 更多候选解 → 更好的winner/loser对比 → 更强的训练信号
    
    # ========== 多偏好对采样策略（提高采样利用率） ==========
    num_pairs_per_sample = 32  # ✓ 从128个候选中生成32个不同难度的偏好对
                                # 采样利用率：32对×2解=64/128 = 50%（充分利用）
                                # 难度分布：
                                #   - 前8对：简单对比（Top vs Bottom，大gap）
                                #   - 中16对：中等难度（Top区 vs Mid区）
                                #   - 后8对：困难对比（Top区内部，小gap）
                                # 实际训练偏好对：96×32×100步×100轮 = 3072万对！
                                # 显存占用：96×32=3072对/批，可控范围
    # ============================================================
    
    # Reference Model 更新频率 (在每个 Iteration 结束时更新)
    update_ref_model = True

    # DPO 稳定性相关
    # 1) Reference model 不要每轮都追着 policy 更新，避免目标漂移导致震荡
    ref_update_interval = 10  # ✓ 从20降到10，更频繁同步

    # 2) 由于 log-prob 是整条序列的和（随 N 线性变大），建议在 DPO 中做 per-step 归一化
    normalize_logp_by_tour_len = True

    # DPO 采样策略优化
    sampling_temperature = 1.2  # 采样温度：>1 更随机(探索), <1 更确定(利用), =1 标准采样
    
    # 温度退火策略（可选）
    use_temperature_annealing = True   # 是否使用温度退火
    temperature_start = 1.5            # 初始温度（高探索）
    temperature_end = 0.8              # 最终温度（高利用）
    temperature_decay = 'linear'       # 退火方式: 'linear', 'exponential', 'cosine'
    
    # DPO 训练监控（帮助定位 loss 震荡原因）
    dpo_log_stats = True    # ✓ 开启详细日志
    dpo_log_every_steps = 10  # ✓ 从20降到10，更频繁监控

    # 评估设置：用于降低评估 length 的抖动
    # 建议 eval_batch_size>=1000（更稳，但更慢）
    eval_batch_size = 1000  # ✓ 从500增加到1000，更准确
    # True: 在 DPO 开始时生成一次固定验证集，后续每次 eval 复用
    eval_use_fixed_set = True
    
    # ========== 混合数据训练（Hybrid Data Training） ==========
    # 在DPO训练时混合使用高质量离线数据和在线生成数据
    use_hybrid_data = True           # 是否启用混合数据训练
    hybrid_offline_ratio = 0.3       # 离线数据占比（0.0-1.0）
                                      # 0.3 表示30%使用SFT数据，70%使用在线生成数据
    hybrid_data_path = None           # 离线数据路径，None则使用sft_data_path
    # ========================================================
    
    # POMO 评估策略（多起点搜索）
    use_pomo_eval = True    # ✓ 启用 POMO 评估，显著提升解质量
    pomo_size = 20          # POMO 起点数量（推荐 tsp_size 的 20%-40%）