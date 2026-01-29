import torch
from config import Config

class PreferenceSampler:
    def __init__(self, model, env):
        self.model = model
        self.env = env
        
    def sample_dpo_data(self, x, temperature=None, reference_tour=None):
        """
        输入: x [Batch, N, 2]
        temperature: 采样温度（如果为 None，从 Config 读取）
        reference_tour: [Batch, N] 可选的参考tour（用于混合训练）
        输出: (x_repeat, winner_tour, loser_tour)
        
        新增：支持从K个候选中生成多个偏好对，提高采样利用率
        """
        B, N, _ = x.size()
        K = Config.num_samples
        num_pairs_per_sample = getattr(Config, 'num_pairs_per_sample', 1)
        
        if temperature is None:
            temperature = getattr(Config, 'sampling_temperature', 1.0)
        
        # 1. 复制数据 K 份以并行采样
        # x_repeated: [B*K, N, 2]
        x_repeated = x.repeat_interleave(K, dim=0)
        
        # 2. 模型采样 (必须开启 No Grad)
        # NOTE: AttentionModel 在 eval() 下会走 greedy(argmax) 解码，
        # 会导致 K 次采样完全一样 => winner==loser => DPO logits 恒为 0 => loss 恒为 ln(2).
        # 这里临时切到 train() 仅用于启用随机采样逻辑（无 dropout/BN 的副作用）。
        was_training = self.model.training
        self.model.train()
        with torch.no_grad():
            sampled_tours, _ = self.model(x_repeated, teacher_forcing=False, temperature=temperature)
        self.model.train(was_training)
        
        # 3. 计算所有采样解的长度
        # lengths: [B*K]
        lengths = self.env.get_tour_length(x_repeated, sampled_tours)
        
        # 4. Reshape 回 [B, K] 以便组内比较
        lengths = lengths.view(B, K)
        sampled_tours = sampled_tours.view(B, K, N)
        
        # 5. 如果提供了reference_tour，将其加入候选中进行比较
        if reference_tour is not None:
            # 计算reference tour的长度
            ref_lengths = self.env.get_tour_length(x, reference_tour).unsqueeze(1)  # [B, 1]
            # 合并所有候选（模型采样的K个 + 1个reference）
            all_lengths = torch.cat([lengths, ref_lengths], dim=1)  # [B, K+1]
            all_tours = torch.cat([sampled_tours, reference_tour.unsqueeze(1)], dim=1)  # [B, K+1, N]
        else:
            all_lengths = lengths
            all_tours = sampled_tours
        
        # 6. 根据长度排序所有候选解
        # order: [B, K or K+1]，从小到大排序的索引
        order = torch.argsort(all_lengths, dim=1)  # [B, K+1] 小->大
        num_candidates = all_lengths.size(1)
        
        # 7. 生成多个不同难度的偏好对
        if num_pairs_per_sample == 1:
            # 原始策略：只生成min vs max（最简单）
            min_indices = order[:, 0:1]  # [B, 1]
            max_indices = order[:, -1:]  # [B, 1]
            
        else:
            # 新策略：生成多个不同难度的偏好对
            # 策略：从排序后的候选中选择不同位置的对比
            # 例如K=128, num_pairs=8时：
            #   - Pair 1: Top1 vs Bottom1 (gap最大，最简单)
            #   - Pair 2: Top1 vs Median (中等难度)
            #   - Pair 3: Top5 vs Top20 (较难)
            #   - Pair 4: Top10 vs Top30 (困难)
            #   等等，提供渐进式的学习信号
            
            min_indices_list = []
            max_indices_list = []
            
            for i in range(num_pairs_per_sample):
                if i == 0:
                    # 第一对：始终是最好 vs 最差（基础对比）
                    min_idx = 0
                    max_idx = num_candidates - 1
                else:
                    # 后续对：使用分层采样策略
                    # winner从前25%选择，loser从后75%选择
                    top_25_pct = max(1, num_candidates // 4)
                    bottom_75_pct = num_candidates - top_25_pct
                    
                    # 为每对设置不同的难度
                    # i越大，难度越高（winner和loser距离越近）
                    difficulty_ratio = min(i / num_pairs_per_sample, 0.8)  # 最高80%难度
                    
                    # Winner位置：从Top区域选择，难度越高位置越靠后
                    winner_pos = int(top_25_pct * difficulty_ratio)
                    winner_pos = min(winner_pos, top_25_pct - 1)
                    
                    # Loser位置：从Bottom区域选择，难度越高位置越靠前
                    loser_pos = bottom_75_pct + int((num_candidates - bottom_75_pct - 1) * (1 - difficulty_ratio))
                    loser_pos = min(loser_pos, num_candidates - 1)
                    
                    min_idx = winner_pos
                    max_idx = loser_pos
                
                min_indices_list.append(order[:, min_idx:min_idx+1])
                max_indices_list.append(order[:, max_idx:max_idx+1])
            
            # 合并所有偏好对
            min_indices = torch.cat(min_indices_list, dim=1)  # [B, num_pairs]
            max_indices = torch.cat(max_indices_list, dim=1)  # [B, num_pairs]
        
        # 8. 提取对应的 Tour
        # 扩展到 [B, num_pairs, N]
        num_pairs = min_indices.size(1)
        winner_tours = all_tours.gather(1, min_indices.view(B, num_pairs, 1).expand(-1, -1, N))
        loser_tours = all_tours.gather(1, max_indices.view(B, num_pairs, 1).expand(-1, -1, N))
        
        # 9. 将多个偏好对展平为batch维度
        # x: [B, N, 2] -> [B*num_pairs, N, 2]
        x_expanded = x.unsqueeze(1).expand(-1, num_pairs, -1, -1).reshape(B * num_pairs, N, 2)
        winner_tours = winner_tours.reshape(B * num_pairs, N)
        loser_tours = loser_tours.reshape(B * num_pairs, N)
        
        # 若 winner/loser 仍完全相同（可能是模型坍缩或重复采样），上层训练循环会跳过该样本
        
        return x_expanded, winner_tours, loser_tours