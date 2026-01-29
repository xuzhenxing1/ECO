import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps, 
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    DPO Loss 公式: -log sigmoid ( beta * (log(r_w) - log(r_l)) )
    其中 log(r) = log(pi / ref) = log_pi - log_ref
    """
    
    # 1. 计算 Log Ratio
    # pi_logr_chosen = log(pi(yw|x)) - log(ref(yw|x))
    policy_chosen_logr = policy_chosen_logps - ref_chosen_logps
    policy_rejected_logr = policy_rejected_logps - ref_rejected_logps
    
    # 2. 计算 Logits
    logits = policy_chosen_logr - policy_rejected_logr

    # 数值稳定：避免 beta*logits 过大导致梯度极度饱和
    scaled = torch.clamp(beta * logits, min=-50.0, max=50.0)
    
    # 3. Loss
    # F.logsigmoid(x) 等价于 log(1 / (1 + exp(-x)))
    losses = -F.logsigmoid(scaled)
    
    mean_loss = losses.mean()
    return mean_loss, mean_loss.detach().item()