"""
快速诊断当前训练问题

运行这个脚本来分析现有的训练结果
"""

import torch
import os
import glob
import json


def diagnose_current_training():
    """诊断当前训练状态"""
    print("\n" + "="*60)
    print("DPO训练震荡问题诊断")
    print("="*60 + "\n")
    
    # 检查配置
    from config import Config
    
    print("1. 当前配置检查:")
    print(f"   TSP规模: {Config.tsp_size}")
    print(f"   DPO学习率: {Config.dpo_lr}")
    print(f"   DPO Beta: {Config.dpo_beta}")
    print(f"   批大小: {Config.dpo_batch_size}")
    print(f"   采样数量: {Config.num_samples}")
    print(f"   Ref更新间隔: {Config.ref_update_interval}")
    print(f"   归一化logp: {Config.normalize_logp_by_tour_len}")
    
    # 潜在问题诊断
    print("\n2. 潜在问题诊断:")
    issues = []
    
    if Config.tsp_size > 200:
        issues.append("⚠ TSP规模很大(>200)，建议先用小规模(20-100)调试")
    
    if Config.dpo_lr >= 3e-4:
        issues.append(f"⚠ 学习率较高({Config.dpo_lr})，建议降到1e-4")
    
    if Config.dpo_beta < 0.2:
        issues.append(f"⚠ Beta较小({Config.dpo_beta})，约束弱，建议增加到0.3-0.5")
    
    if Config.num_samples < 16:
        issues.append(f"⚠ 采样数量较少({Config.num_samples})，建议增加到16-32")
    
    if Config.ref_update_interval <= 5:
        issues.append(f"⚠ Ref更新太频繁({Config.ref_update_interval})，建议10-20")
    
    if not Config.normalize_logp_by_tour_len:
        issues.append("⚠ 未启用logp归一化，大规模TSP可能梯度不稳定")
    
    if issues:
        for issue in issues:
            print(f"   {issue}")
    else:
        print("   ✓ 配置看起来合理")
    
    print("\n3. 建议的配置修改:")
    print("-" * 60)
    print("在config.py中修改以下参数:")
    print("")
    print("# 降低学习率")
    print("dpo_lr = 1e-4  # 从3e-4降低")
    print("")
    print("# 增加Beta约束")
    print("dpo_beta = 0.3  # 从0.1增加")
    print("")
    print("# 增加采样多样性")
    print("num_samples = 16  # 从8增加")
    print("")
    print("# 调整Ref更新频率")
    print("ref_update_interval = 10  # 从20降低")
    print("")
    print("# 开启详细日志")
    print("dpo_log_stats = True")
    print("dpo_log_every_steps = 10")
    print("-" * 60)
    
    print("\n4. 下一步行动:")
    print("   1. 修改config.py中的参数（参考上面的建议）")
    print("   2. 重新运行训练: python train.py")
    print("   3. 或者直接使用稳定配置: 将config_stable.py复制为config.py")
    print("   4. 观察训练日志中的Logits和Gap值是否正常")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    diagnose_current_training()
