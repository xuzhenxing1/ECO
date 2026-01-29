import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps, 
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    DPO Loss for CVRP
    
    公式: -log sigmoid(beta * (log(r_w) - log(r_l)))
    其中 log(r) = log(pi / ref) = log_pi - log_ref
    
    Args:
        policy_chosen_logps: [B] 策略模型在优选解上的对数概率
        policy_rejected_logps: [B] 策略模型在劣选解上的对数概率
        ref_chosen_logps: [B] 参考模型在优选解上的对数概率
        ref_rejected_logps: [B] 参考模型在劣选解上的对数概率
        beta: DPO温度参数
    
    Returns:
        mean_loss: 标量loss
        loss_value: loss数值（用于日志）
    """
    
    # 1. 计算Log Ratio
    policy_chosen_logr = policy_chosen_logps - ref_chosen_logps
    policy_rejected_logr = policy_rejected_logps - ref_rejected_logps
    
    # 2. 计算Logits
    logits = policy_chosen_logr - policy_rejected_logr

    # 数值稳定：避免beta*logits过大导致梯度饱和
    scaled = torch.clamp(beta * logits, min=-50.0, max=50.0)
    
    # 3. Loss
    losses = -F.logsigmoid(scaled)
    
    mean_loss = losses.mean()
    return mean_loss, mean_loss.detach().item()
