import torch
from config import Config

class TSPEnv:
    def __init__(self, device):
        self.device = device

    def get_random_problems(self, batch_size, tsp_size):
        """生成 [Batch, N, 2] 的随机坐标"""
        return torch.rand(batch_size, tsp_size, 2).to(self.device)

    def get_tour_length(self, nodes, tour_indices):
        """
        计算路径长度
        nodes: [Batch, N, 2]
        tour_indices: [Batch, N] (访问顺序)
        """
        batch_size, tsp_size, _ = nodes.size()
        
        # 按照 tour_indices 重新排列节点
        # gathering: [Batch, N, 2]
        gathered_nodes = nodes.gather(1, tour_indices.unsqueeze(-1).expand(-1, -1, 2))
        
        # 计算相邻节点距离
        # 错位一步: 0->1, 1->2, ..., (N-1)->0
        next_nodes = torch.roll(gathered_nodes, shifts=-1, dims=1)
        
        # 欧氏距离
        dist = torch.norm(gathered_nodes - next_nodes, dim=2) # [Batch, N]
        
        return dist.sum(dim=1) # [Batch] 总长度