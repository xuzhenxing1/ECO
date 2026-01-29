import torch
from config import Config

class PreferenceSampler:
    """
    偏好对采样器 for CVRP
    
    功能：
    1. 从模型采样K个候选解
    2. 根据路径长度排序
    3. 生成多个不同难度的winner/loser对
    """
    
    def __init__(self, model, env):
        self.model = model
        self.env = env
        
    def sample_dpo_data(self, depot_xy, node_xy, node_demand, 
                       temperature=None, reference_tour=None):
        """
        采样DPO训练数据
        
        Args:
            depot_xy: [B, 1, 2]
            node_xy: [B, N, 2]
            node_demand: [B, N]
            temperature: 采样温度
            reference_tour: [B, L] 可选的参考tour（用于混合训练）
        
        Returns:
            depot_xy_expanded: [B*num_pairs, 1, 2]
            node_xy_expanded: [B*num_pairs, N, 2]
            node_demand_expanded: [B*num_pairs, N]
            winner_tours: [B*num_pairs, L_max]
            loser_tours: [B*num_pairs, L_max]
        """
        B = depot_xy.size(0)
        N = node_xy.size(1)
        K = Config.num_samples
        num_pairs_per_sample = getattr(Config, 'num_pairs_per_sample', 1)
        
        if temperature is None:
            temperature = getattr(Config, 'sampling_temperature', 1.0)
        
        # 1. 复制数据K份以并行采样
        depot_xy_repeated = depot_xy.repeat_interleave(K, dim=0)  # [B*K, 1, 2]
        node_xy_repeated = node_xy.repeat_interleave(K, dim=0)    # [B*K, N, 2]
        node_demand_repeated = node_demand.repeat_interleave(K, dim=0)  # [B*K, N]
        
        # 2. 模型采样
        was_training = self.model.training
        self.model.train()  # 启用随机采样
        
        with torch.no_grad():
            sampled_tours, _ = self.model(
                depot_xy_repeated, 
                node_xy_repeated, 
                node_demand_repeated,
                teacher_forcing=False, 
                temperature=temperature
            )
        
        self.model.train(was_training)
        
        # 3. 计算所有采样解的长度
        lengths = self.env.get_tour_length(
            depot_xy_repeated, 
            node_xy_repeated, 
            sampled_tours
        )  # [B*K]
        
        # 4. Reshape回[B, K]以便组内比较
        lengths = lengths.view(B, K)
        
        # sampled_tours可能长度不同，需要padding
        # 找到最大长度
        tour_lengths = [sampled_tours[i].size(0) for i in range(sampled_tours.size(0))]
        max_tour_len = max(tour_lengths)
        
        # Padding到相同长度（用0填充，但实际使用时会被mask）
        padded_tours = []
        for i in range(sampled_tours.size(0)):
            tour = sampled_tours[i]
            if tour.size(0) < max_tour_len:
                padding = torch.zeros(
                    max_tour_len - tour.size(0), 
                    dtype=tour.dtype, 
                    device=tour.device
                )
                tour = torch.cat([tour, padding])
            padded_tours.append(tour)
        
        sampled_tours = torch.stack(padded_tours, dim=0).view(B, K, max_tour_len)
        
        # 5. 如果提供了reference_tour，加入候选中
        if reference_tour is not None:
            ref_lengths = self.env.get_tour_length(depot_xy, node_xy, reference_tour).unsqueeze(1)
            
            # Padding reference_tour
            if reference_tour.size(1) < max_tour_len:
                padding = torch.zeros(
                    B, max_tour_len - reference_tour.size(1),
                    dtype=reference_tour.dtype,
                    device=reference_tour.device
                )
                reference_tour = torch.cat([reference_tour, padding], dim=1)
            elif reference_tour.size(1) > max_tour_len:
                reference_tour = reference_tour[:, :max_tour_len]
            
            all_lengths = torch.cat([lengths, ref_lengths], dim=1)  # [B, K+1]
            all_tours = torch.cat([sampled_tours, reference_tour.unsqueeze(1)], dim=1)  # [B, K+1, L]
        else:
            all_lengths = lengths
            all_tours = sampled_tours
        
        # 6. 排序
        order = torch.argsort(all_lengths, dim=1)  # [B, K or K+1] 小->大
        num_candidates = all_lengths.size(1)
        
        # 7. 生成多个不同难度的偏好对
        if num_pairs_per_sample == 1:
            # 简单策略：只生成min vs max
            min_indices = order[:, 0:1]
            max_indices = order[:, -1:]
        else:
            # 多样化策略：生成不同难度的偏好对
            min_indices_list = []
            max_indices_list = []
            
            for i in range(num_pairs_per_sample):
                if i == 0:
                    # 第一对：最好 vs 最差
                    min_idx = 0
                    max_idx = num_candidates - 1
                else:
                    # 后续对：分层采样
                    top_25_pct = max(1, num_candidates // 4)
                    bottom_75_pct = num_candidates - top_25_pct
                    
                    difficulty_ratio = min(i / num_pairs_per_sample, 0.8)
                    
                    winner_pos = int(top_25_pct * difficulty_ratio)
                    winner_pos = min(winner_pos, top_25_pct - 1)
                    
                    loser_pos = bottom_75_pct + int((num_candidates - bottom_75_pct - 1) * (1 - difficulty_ratio))
                    loser_pos = min(loser_pos, num_candidates - 1)
                    
                    min_idx = winner_pos
                    max_idx = loser_pos
                
                min_indices_list.append(order[:, min_idx:min_idx+1])
                max_indices_list.append(order[:, max_idx:max_idx+1])
            
            min_indices = torch.cat(min_indices_list, dim=1)  # [B, num_pairs]
            max_indices = torch.cat(max_indices_list, dim=1)  # [B, num_pairs]
        
        # 8. 提取对应的Tour
        num_pairs = min_indices.size(1)
        winner_tours = all_tours.gather(1, min_indices.view(B, num_pairs, 1).expand(-1, -1, max_tour_len))
        loser_tours = all_tours.gather(1, max_indices.view(B, num_pairs, 1).expand(-1, -1, max_tour_len))
        
        # 9. 展平为batch维度
        depot_xy_expanded = depot_xy.unsqueeze(1).expand(-1, num_pairs, -1, -1).reshape(B * num_pairs, 1, 2)
        node_xy_expanded = node_xy.unsqueeze(1).expand(-1, num_pairs, -1, -1).reshape(B * num_pairs, N, 2)
        node_demand_expanded = node_demand.unsqueeze(1).expand(-1, num_pairs, -1).reshape(B * num_pairs, N)
        winner_tours = winner_tours.reshape(B * num_pairs, max_tour_len)
        loser_tours = loser_tours.reshape(B * num_pairs, max_tour_len)
        
        return depot_xy_expanded, node_xy_expanded, node_demand_expanded, winner_tours, loser_tours
