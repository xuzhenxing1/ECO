import torch
from config import Config

class CVRPEnv:
    """
    CVRP环境类
    
    核心功能：
    1. 生成随机CVRP问题实例（depot + nodes + demands）
    2. 计算路径总长度
    3. 验证解的可行性（容量约束、访问完整性）
    """
    
    def __init__(self, device):
        self.device = device
        self.capacity = Config.vehicle_capacity
    
    def get_random_problems(self, batch_size, problem_size):
        """
        生成随机CVRP问题
        
        Args:
            batch_size: 批次大小
            problem_size: 客户节点数量（不包括depot）
            
        Returns:
            depot_xy: [Batch, 1, 2] depot坐标
            node_xy: [Batch, N, 2] 客户节点坐标
            node_demand: [Batch, N] 客户需求量（归一化到[0,1]）
        """
        # Depot坐标 [0,1]范围内
        depot_xy = torch.rand(batch_size, 1, 2, device=self.device)
        
        # 客户节点坐标
        node_xy = torch.rand(batch_size, problem_size, 2, device=self.device)
        
        # 生成需求量（整数，然后归一化）
        demand_scaler = Config.demand_scaler
        node_demand = torch.randint(
            Config.demand_min, 
            Config.demand_max + 1, 
            size=(batch_size, problem_size),
            device=self.device
        ).float() / float(demand_scaler)
        
        return depot_xy, node_xy, node_demand
    
    def get_tour_length(self, depot_xy, node_xy, tour_indices):
        """
        计算CVRP路径总长度
        
        Args:
            depot_xy: [Batch, 1, 2]
            node_xy: [Batch, N, 2]
            tour_indices: [Batch, L] 访问序列
                         0表示depot，1~N表示客户节点
                         例如：[0, 3, 5, 0, 1, 2, 0] 表示两条子路径
            
        Returns:
            total_length: [Batch] 总路径长度
        """
        batch_size = depot_xy.size(0)
        
        # 合并depot和nodes: [Batch, N+1, 2]
        # 索引0=depot, 1~N=客户节点
        all_nodes = torch.cat([depot_xy, node_xy], dim=1)
        
        # 按照tour_indices获取访问序列的坐标
        tour_len = tour_indices.size(1)
        gathering_index = tour_indices.unsqueeze(-1).expand(-1, -1, 2)
        
        # 获取访问序列的坐标: [Batch, L, 2]
        ordered_nodes = all_nodes.gather(1, gathering_index)
        
        # 计算相邻节点间的距离
        next_nodes = torch.roll(ordered_nodes, shifts=-1, dims=1)
        distances = torch.norm(ordered_nodes - next_nodes, dim=2)
        
        # 总长度（最后一步是回到起点，已包含在roll中）
        total_length = distances.sum(dim=1)
        
        return total_length
    
    def check_feasibility(self, node_demand, tour_indices):
        """
        检查解的可行性（容量约束）
        
        Args:
            node_demand: [Batch, N] 客户需求量
            tour_indices: [Batch, L] 访问序列，0表示depot
            
        Returns:
            is_feasible: [Batch] bool tensor
            violations: [Batch] 违反约束的次数
        """
        batch_size = node_demand.size(0)
        
        # 添加depot的需求量（0）
        depot_demand = torch.zeros(batch_size, 1, device=self.device)
        all_demands = torch.cat([depot_demand, node_demand], dim=1)
        
        # 获取tour中每个位置的需求量
        tour_demands = all_demands.gather(1, tour_indices)
        
        # 计算每条子路径的累积需求
        is_depot = (tour_indices == 0)
        
        violations = torch.zeros(batch_size, device=self.device)
        current_loads = torch.zeros(batch_size, device=self.device)
        
        tour_len = tour_indices.size(1)
        for step in range(tour_len):
            demand = tour_demands[:, step]
            
            # 如果不是depot，累加需求
            current_loads = torch.where(
                is_depot[:, step],
                torch.zeros_like(current_loads),
                current_loads + demand
            )
            
            # 检查是否超出容量
            over_capacity = current_loads > (self.capacity + 1e-5)  # 容忍小误差
            violations += over_capacity.float()
        
        is_feasible = (violations == 0)
        
        return is_feasible, violations
