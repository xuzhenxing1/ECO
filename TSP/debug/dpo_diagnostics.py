"""
DPO训练诊断工具 - 帮助定位震荡原因
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os


class DPODiagnostics:
    """DPO训练诊断类"""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.history = defaultdict(list)
        
    def log_iteration(self, iteration, metrics):
        """
        记录每个iteration的指标
        
        metrics应包含:
        - loss: DPO loss
        - eval_length: 评估路径长度
        - policy_chosen_logp: 平均
        - policy_rejected_logp: 平均
        - ref_chosen_logp: 平均
        - ref_rejected_logp: 平均
        - logits: DPO logits (preference strength)
        - winner_loser_gap: winner和loser的路径长度差异
        """
        for key, value in metrics.items():
            self.history[key].append(value)
        
    def plot_diagnostics(self):
        """绘制诊断图表"""
        iters = range(len(self.history['loss']))
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. Loss曲线
        ax = axes[0, 0]
        ax.plot(iters, self.history['loss'], 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('DPO Loss')
        ax.set_title('DPO Loss (应该逐渐下降)')
        ax.grid(True, alpha=0.3)
        
        # 2. 评估路径长度
        ax = axes[0, 1]
        ax.plot(iters, self.history['eval_length'], 'g-', linewidth=2)
        ax.axhline(y=min(self.history['eval_length']), color='r', linestyle='--', label='Best')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Tour Length')
        ax.set_title('评估路径长度 (应该逐渐下降)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. LogProbs变化
        ax = axes[1, 0]
        ax.plot(iters, self.history['policy_chosen_logp'], label='Policy Chosen', linewidth=2)
        ax.plot(iters, self.history['policy_rejected_logp'], label='Policy Rejected', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Log Probability')
        ax.set_title('Policy LogProbs (chosen应该上升，rejected应该下降)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Reference LogProbs变化
        ax = axes[1, 1]
        ax.plot(iters, self.history['ref_chosen_logp'], label='Ref Chosen', linewidth=2)
        ax.plot(iters, self.history['ref_rejected_logp'], label='Ref Rejected', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Log Probability')
        ax.set_title('Reference LogProbs (应该相对稳定)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. DPO Logits (preference strength)
        ax = axes[2, 0]
        ax.plot(iters, self.history['logits'], 'purple', linewidth=2)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('DPO Logits')
        ax.set_title('DPO Logits (应该为正且逐渐增大)')
        ax.grid(True, alpha=0.3)
        
        # 6. Winner-Loser Gap
        ax = axes[2, 1]
        ax.plot(iters, self.history['winner_loser_gap'], 'orange', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Length Gap')
        ax.set_title('Winner-Loser路径差异 (应该>0且稳定)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'dpo_diagnostics.png'), dpi=150)
        plt.close()
        
    def print_summary(self):
        """打印诊断总结"""
        print("\n" + "="*60)
        print("DPO训练诊断总结")
        print("="*60)
        
        # Loss趋势
        loss_arr = np.array(self.history['loss'])
        loss_trend = np.polyfit(range(len(loss_arr)), loss_arr, 1)[0]
        print(f"\n1. Loss趋势:")
        print(f"   起始Loss: {loss_arr[0]:.4f}")
        print(f"   最终Loss: {loss_arr[-1]:.4f}")
        print(f"   趋势: {'下降✓' if loss_trend < 0 else '上升或震荡✗'}")
        print(f"   斜率: {loss_trend:.6f}")
        
        # 路径长度趋势
        length_arr = np.array(self.history['eval_length'])
        length_trend = np.polyfit(range(len(length_arr)), length_arr, 1)[0]
        best_idx = np.argmin(length_arr)
        print(f"\n2. 路径长度趋势:")
        print(f"   起始长度: {length_arr[0]:.4f}")
        print(f"   最优长度: {length_arr[best_idx]:.4f} (iter {best_idx})")
        print(f"   最终长度: {length_arr[-1]:.4f}")
        print(f"   趋势: {'优化✓' if length_trend < 0 else '未改善✗'}")
        print(f"   改善率: {((length_arr[0] - length_arr[best_idx]) / length_arr[0] * 100):.2f}%")
        
        # Logits检查
        logits_arr = np.array(self.history['logits'])
        print(f"\n3. DPO Logits (preference strength):")
        print(f"   平均值: {logits_arr.mean():.4f} ({'正常✓' if logits_arr.mean() > 0 else '异常✗，模型偏好错误'})")
        print(f"   标准差: {logits_arr.std():.4f} ({'稳定✓' if logits_arr.std() < 1.0 else '震荡✗'})")
        
        # Winner-Loser Gap
        gap_arr = np.array(self.history['winner_loser_gap'])
        print(f"\n4. Winner-Loser路径差异:")
        print(f"   平均差异: {gap_arr.mean():.4f}")
        print(f"   最小差异: {gap_arr.min():.4f} ({'足够✓' if gap_arr.min() > 0.001 else '太小✗，采样质量差'})")
        
        # 震荡检测
        length_volatility = np.std(np.diff(length_arr))
        print(f"\n5. 震荡程度:")
        print(f"   路径长度波动: {length_volatility:.4f} ({'平稳✓' if length_volatility < 0.1 else '震荡✗'})")
        
        # 建议
        print(f"\n{'='*60}")
        print("诊断建议:")
        print("="*60)
        
        if loss_trend > 0:
            print("⚠ Loss上升，可能原因:")
            print("  - 学习率太高，尝试降低dpo_lr")
            print("  - Beta太小，尝试增加dpo_beta")
            print("  - Reference model更新太频繁")
            
        if logits_arr.mean() < 0:
            print("⚠ Logits为负，模型偏好反转:")
            print("  - 检查winner/loser标签是否正确")
            print("  - 检查采样是否有问题")
            
        if gap_arr.min() < 0.001:
            print("⚠ Winner-Loser差异太小:")
            print("  - 增加num_samples获得更多样化的解")
            print("  - 检查模型是否退化为固定策略")
            
        if length_volatility > 0.1:
            print("⚠ 路径长度震荡严重:")
            print("  - 降低学习率")
            print("  - 增加梯度裁剪")
            print("  - 减少ref_update_interval")
            
        print("="*60 + "\n")
        
    def save_history(self):
        """保存训练历史"""
        import json
        with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:
            # 转换为可序列化的格式
            history_dict = {k: [float(v) for v in vals] for k, vals in self.history.items()}
            json.dump(history_dict, f, indent=2)


def analyze_dpo_batch(x, winner_tours, loser_tours, env, policy_model, ref_model):
    """
    分析单个batch的DPO数据质量
    返回诊断指标
    """
    with torch.no_grad():
        # 计算路径长度
        winner_lengths = env.get_tour_length(x, winner_tours)
        loser_lengths = env.get_tour_length(x, loser_tours)
        
        # 计算logprobs
        _, policy_chosen_logps = policy_model(x, winner_tours, teacher_forcing=True)
        _, policy_rejected_logps = policy_model(x, loser_tours, teacher_forcing=True)
        _, ref_chosen_logps = ref_model(x, winner_tours, teacher_forcing=True)
        _, ref_rejected_logps = ref_model(x, loser_tours, teacher_forcing=True)
        
        # 计算指标
        metrics = {
            'policy_chosen_logp': policy_chosen_logps.mean().item(),
            'policy_rejected_logp': policy_rejected_logps.mean().item(),
            'ref_chosen_logp': ref_chosen_logps.mean().item(),
            'ref_rejected_logp': ref_rejected_logps.mean().item(),
            'logits': ((policy_chosen_logps - ref_chosen_logps) - 
                      (policy_rejected_logps - ref_rejected_logps)).mean().item(),
            'winner_loser_gap': (loser_lengths - winner_lengths).mean().item(),
            'winner_avg_length': winner_lengths.mean().item(),
            'loser_avg_length': loser_lengths.mean().item(),
        }
        
    return metrics
