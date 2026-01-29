# -*- coding: utf-8 -*-
"""
温度调度策略演示和可视化

运行此脚本查看不同温度策略的效果
"""

import matplotlib.pyplot as plt
import numpy as np
from temperature_scheduler import TemperatureScheduler

def visualize_temperature_schedules():
    """可视化不同温度调度策略"""
    
    total_iterations = 100
    iterations = np.arange(total_iterations)
    
    # 创建不同的调度器
    schedulers = {
        'Linear (推荐新手)': TemperatureScheduler(1.5, 0.8, total_iterations, 'linear'),
        'Exponential (快速收敛)': TemperatureScheduler(1.5, 0.8, total_iterations, 'exponential'),
        'Cosine (最平滑)': TemperatureScheduler(1.5, 0.8, total_iterations, 'cosine'),
        'Step (阶梯式)': TemperatureScheduler(1.5, 0.8, total_iterations, 'step'),
    }
    
    # 收集温度值
    temps = {}
    for name, scheduler in schedulers.items():
        temps[name] = [scheduler.get_temperature(i) for i in iterations]
    
    # 绘图
    plt.figure(figsize=(14, 8))
    
    # 子图1: 温度曲线
    plt.subplot(2, 1, 1)
    for name, temp_values in temps.items():
        plt.plot(iterations, temp_values, label=name, linewidth=2)
    
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (T=1.0)')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Temperature', fontsize=12)
    plt.title('温度调度策略对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 添加阶段标注
    plt.axvspan(0, 33, alpha=0.1, color='green', label='Exploration')
    plt.axvspan(33, 67, alpha=0.1, color='yellow')
    plt.axvspan(67, 100, alpha=0.1, color='red')
    plt.text(16, 1.55, 'Exploration', ha='center', fontsize=10, alpha=0.7)
    plt.text(50, 1.55, 'Balance', ha='center', fontsize=10, alpha=0.7)
    plt.text(83, 1.55, 'Exploitation', ha='center', fontsize=10, alpha=0.7)
    
    # 子图2: 温度变化率
    plt.subplot(2, 1, 2)
    for name, temp_values in temps.items():
        deltas = np.diff(temp_values)
        plt.plot(iterations[:-1], deltas, label=name, linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Temperature Change Rate', fontsize=12)
    plt.title('温度变化率（平滑度对比）', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temperature_schedules.png', dpi=150, bbox_inches='tight')
    print("✓ 图表已保存到 temperature_schedules.png")
    plt.show()


def print_temperature_table():
    """打印温度值表格"""
    
    total_iterations = 100
    schedulers = {
        'Linear': TemperatureScheduler(1.5, 0.8, total_iterations, 'linear'),
        'Exponential': TemperatureScheduler(1.5, 0.8, total_iterations, 'exponential'),
        'Cosine': TemperatureScheduler(1.5, 0.8, total_iterations, 'cosine'),
        'Step': TemperatureScheduler(1.5, 0.8, total_iterations, 'step'),
    }
    
    print("\n" + "="*80)
    print("温度调度策略对比表")
    print("="*80)
    print(f"{'Iteration':<12} {'Linear':<12} {'Exponential':<14} {'Cosine':<12} {'Step':<12} {'阶段':<15}")
    print("-"*80)
    
    key_iterations = [0, 10, 20, 33, 50, 67, 80, 90, 99]
    
    for iter_num in key_iterations:
        temps = [scheduler.get_temperature(iter_num) for scheduler in schedulers.values()]
        
        # 判断阶段
        progress = iter_num / total_iterations
        if progress < 0.33:
            phase = "探索 🔍"
        elif progress < 0.67:
            phase = "平衡 ⚖️"
        else:
            phase = "利用 🎯"
        
        print(f"{iter_num:<12} {temps[0]:<12.3f} {temps[1]:<14.3f} {temps[2]:<12.3f} {temps[3]:<12.3f} {phase:<15}")
    
    print("="*80)


def explain_temperature_effects():
    """解释不同温度的效果"""
    
    print("\n" + "="*80)
    print("温度参数对采样的影响")
    print("="*80)
    
    print("\n【温度 > 1.0】- 高探索")
    print("  ✓ 更随机的采样")
    print("  ✓ 增加路径多样性")
    print("  ✓ 容易跳出局部最优")
    print("  ✗ 可能采样到较差的路径")
    print("  适用：训练初期，寻找新的好解")
    
    print("\n【温度 = 1.0】- 标准采样")
    print("  ✓ 平衡探索与利用")
    print("  ✓ 按模型输出的概率分布采样")
    print("  适用：中期训练，稳定优化")
    
    print("\n【温度 < 1.0】- 高利用")
    print("  ✓ 更确定性的采样")
    print("  ✓ 倾向于选择高概率动作")
    print("  ✓ 稳定收敛到好解")
    print("  ✗ 可能过早收敛")
    print("  适用：训练后期，精细优化")
    
    print("\n" + "="*80)
    print("推荐策略")
    print("="*80)
    
    print("\n1. Cosine 退火（最推荐）⭐⭐⭐⭐⭐")
    print("   - 平滑过渡，不会突变")
    print("   - 前期充分探索，后期稳定收敛")
    print("   - 配置: temperature_decay = 'cosine'")
    
    print("\n2. Linear 退火（简单有效）⭐⭐⭐⭐")
    print("   - 线性递减，容易理解")
    print("   - 适合新手")
    print("   - 配置: temperature_decay = 'linear'")
    
    print("\n3. Exponential 退火（快速收敛）⭐⭐⭐")
    print("   - 前期降温慢，后期降温快")
    print("   - 适合时间有限的场景")
    print("   - 配置: temperature_decay = 'exponential'")
    
    print("\n4. Fixed 温度（调试用）⭐⭐")
    print("   - 不使用退火，固定温度")
    print("   - 配置: use_temperature_annealing = False")
    
    print("="*80)


def recommend_settings():
    """推荐不同场景的温度设置"""
    
    print("\n" + "="*80)
    print("不同场景的温度设置推荐")
    print("="*80)
    
    scenarios = [
        {
            'name': 'TSP-20/50 (小规模)',
            'start': 1.2,
            'end': 0.9,
            'decay': 'cosine',
            'reason': '小规模问题较简单，温度范围可以小一些'
        },
        {
            'name': 'TSP-100 (中等规模)',
            'start': 1.5,
            'end': 0.8,
            'decay': 'cosine',
            'reason': '中等规模需要更多探索，推荐默认设置'
        },
        {
            'name': 'TSP-200+ (大规模)',
            'start': 1.8,
            'end': 0.7,
            'decay': 'cosine',
            'reason': '大规模问题复杂度高，需要更强的探索'
        },
        {
            'name': '快速调试',
            'start': 1.0,
            'end': 1.0,
            'decay': 'fixed',
            'reason': '固定温度，减少变量，便于调试其他超参数'
        },
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   temperature_start = {scenario['start']}")
        print(f"   temperature_end = {scenario['end']}")
        print(f"   temperature_decay = '{scenario['decay']}'")
        print(f"   原因: {scenario['reason']}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("="*80)
    print("温度调度策略完整演示")
    print("="*80)
    
    # 1. 打印表格
    print_temperature_table()
    
    # 2. 解释效果
    explain_temperature_effects()
    
    # 3. 场景推荐
    recommend_settings()
    
    # 4. 可视化
    print("\n生成可视化图表...")
    try:
        visualize_temperature_schedules()
    except Exception as e:
        print(f"可视化失败: {e}")
        print("可能需要安装 matplotlib: pip install matplotlib")
    
    print("\n" + "="*80)
    print("演示完成！")
    print("="*80)
    print("\n💡 快速开始：")
    print("1. 在 config.py 中已经配置好了温度退火")
    print("2. 直接运行 python train.py 即可使用")
    print("3. 查看训练日志中的温度变化")
