"""
稳定版DPO训练脚本 - 针对震荡问题的改进

主要改进点:
1. 使用EMA更新reference model而不是直接复制
2. 添加学习率warmup
3. 增强的梯度裁剪
4. 详细的训练监控
5. Early stopping
6. 温度采样增加多样性
"""

# 在你的train.py中，可以添加这些改进：

# ==================== 改进1: EMA Reference Model 更新 ====================
def update_ref_model_ema(ref_model, policy_model, decay=0.95):
    """使用指数移动平均更新reference model，更平滑"""
    with torch.no_grad():
        for ref_param, policy_param in zip(ref_model.parameters(), policy_model.parameters()):
            ref_param.data.mul_(decay).add_(policy_param.data, alpha=1-decay)


# ==================== 改进2: 学习率Warmup ====================
def get_lr_schedule(iteration, warmup_iters=5, base_lr=1e-4):
    """学习率warmup，前几轮逐渐增加学习率"""
    if iteration < warmup_iters:
        return base_lr * (iteration + 1) / warmup_iters
    else:
        return base_lr


# ==================== 改进3: 增强采样多样性 ====================
# 在model.py的forward函数中添加温度参数
def sample_with_temperature(logits, temperature=1.0):
    """使用温度采样而不是argmax，增加多样性"""
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


# ==================== 改进4: 改进的DPO训练循环 ====================
def run_iterative_dpo_stable(policy_model, env, run_dir: str, config):
    """稳定版DPO训练"""
    from dpo_diagnostics import DPODiagnostics, analyze_dpo_batch
    
    print(f">>> Starting Stable Iterative DPO for {config.total_iterations} iterations...")
    
    # 初始化诊断工具
    diagnostics = DPODiagnostics(run_dir)
    
    # 固定验证集
    x_eval = env.get_random_problems(config.eval_batch_size, config.tsp_size)
    
    # Reference Model
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    
    # Optimizer
    optimizer = optim.Adam(policy_model.parameters(), lr=config.dpo_lr)
    sampler = PreferenceSampler(policy_model, env)
    
    # Early stopping
    best_eval_length = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(run_dir, "best_tsp_model.pth")
    
    for iteration in range(config.total_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration+1}/{config.total_iterations}")
        print(f"{'='*60}")
        
        # Learning rate warmup
        if config.use_lr_warmup:
            current_lr = get_lr_schedule(iteration, config.warmup_iterations, config.dpo_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"Learning Rate: {current_lr:.6f}")
        
        policy_model.train()
        iter_loss = 0
        iter_metrics = []
        
        # 训练步骤
        num_steps = config.epochs_per_iter * 20
        pbar = tqdm(range(num_steps), desc=f"Training")
        
        for step in pbar:
            # 生成问题
            x = env.get_random_problems(config.dpo_batch_size, config.tsp_size)
            
            # 采样winner/loser
            x, winner_tours, loser_tours = sampler.sample_dpo_data(x)
            
            # 过滤无效样本
            valid = (winner_tours != loser_tours).any(dim=1)
            if not valid.any():
                continue
                
            x = x[valid]
            winner_tours = winner_tours[valid]
            loser_tours = loser_tours[valid]
            
            # 计算logprobs
            _, policy_chosen_logps = policy_model(x, winner_tours, teacher_forcing=True)
            _, policy_rejected_logps = policy_model(x, loser_tours, teacher_forcing=True)
            
            with torch.no_grad():
                _, ref_chosen_logps = ref_model(x, winner_tours, teacher_forcing=True)
                _, ref_rejected_logps = ref_model(x, loser_tours, teacher_forcing=True)
            
            # Normalize
            if config.normalize_logp_by_tour_len:
                denom = float(config.tsp_size)
                policy_chosen_logps /= denom
                policy_rejected_logps /= denom
                ref_chosen_logps /= denom
                ref_rejected_logps /= denom
            
            # DPO Loss
            loss, loss_val = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                beta=config.dpo_beta
            )
            
            # 梯度更新
            optimizer.zero_grad()
            loss.backward()
            
            # 强梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(), 
                max_norm=config.gradient_clip_norm
            )
            
            optimizer.step()
            
            iter_loss += loss_val
            
            # 收集诊断指标
            if config.dpo_log_stats and (step % config.dpo_log_every_steps == 0):
                batch_metrics = analyze_dpo_batch(
                    x, winner_tours, loser_tours, env, policy_model, ref_model
                )
                iter_metrics.append(batch_metrics)
                
                pbar.set_postfix({
                    "Loss": f"{loss_val:.4f}",
                    "GradNorm": f"{grad_norm:.3f}",
                    "Logits": f"{batch_metrics['logits']:.3f}",
                    "Gap": f"{batch_metrics['winner_loser_gap']:.4f}"
                })
            else:
                pbar.set_postfix({"Loss": f"{loss_val:.4f}"})
        
        # 平均loss
        avg_loss = iter_loss / num_steps
        
        # 评估
        avg_len = evaluate_model(policy_model, env, x_eval=x_eval)
        
        # 记录诊断指标
        if iter_metrics:
            avg_metrics = {k: np.mean([m[k] for m in iter_metrics]) 
                          for k in iter_metrics[0].keys()}
            avg_metrics['loss'] = avg_loss
            avg_metrics['eval_length'] = avg_len
            diagnostics.log_iteration(iteration, avg_metrics)
        
        # Update Reference Model
        if config.update_ref_model and (iteration + 1) % config.ref_update_interval == 0:
            if config.use_ref_ema:
                print(f">>> Updating Reference Model (EMA, decay={config.ref_ema_decay})")
                update_ref_model_ema(ref_model, policy_model, config.ref_ema_decay)
            else:
                print(">>> Updating Reference Model (Copy)")
                ref_model.load_state_dict(policy_model.state_dict())
        
        # Early Stopping Check
        if config.use_early_stopping:
            if avg_len < best_eval_length - config.min_delta:
                best_eval_length = avg_len
                patience_counter = 0
                save_model(policy_model, best_model_path)
                print(f"✓ New best model! Length: {avg_len:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} iterations")
                
                if patience_counter >= config.patience:
                    print(f"Early stopping triggered after {iteration+1} iterations")
                    break
        else:
            if avg_len < best_eval_length:
                best_eval_length = avg_len
                save_model(policy_model, best_model_path)
        
        # 定期保存
        if (iteration + 1) % config.save_every_n_iters == 0:
            ckpt_path = os.path.join(run_dir, f"checkpoint_iter{iteration+1}.pth")
            save_model(policy_model, ckpt_path)
        
        # 定期生成诊断图
        if (iteration + 1) % 5 == 0:
            diagnostics.plot_diagnostics()
    
    # 训练结束后的诊断
    diagnostics.plot_diagnostics()
    diagnostics.print_summary()
    diagnostics.save_history()
    
    return diagnostics.history['loss']


# 使用示例：
# 在main函数中替换原来的run_iterative_dpo为：
# from config_stable import Config  # 使用稳定配置
# loss_history = run_iterative_dpo_stable(policy_model, env, run_dir, Config)
