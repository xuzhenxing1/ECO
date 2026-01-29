import torch

def get_nearest_neighbor_tour(depot_xy, node_xy, node_demand, capacity=1.0):
    """
    贪婪启发式算法 for CVRP（用作SFT的teacher）
    
    策略：
    1. 从depot开始
    2. 重复选择最近的未访问节点（满足容量约束）
    3. 容量不足时返回depot补货
    
    Args:
        depot_xy: [Batch, 1, 2]
        node_xy: [Batch, N, 2]
        node_demand: [Batch, N]
        capacity: 车辆容量
        
    Returns:
        tours: [Batch, L] 访问序列（包含多次depot访问）
    """
    batch_size = depot_xy.size(0)
    n_nodes = node_xy.size(1)
    device = depot_xy.device
    
    # 合并所有节点：[Batch, N+1, 2]
    # 索引0=depot, 1~N=customers
    all_nodes = torch.cat([depot_xy, node_xy], dim=1)
    
    # 计算距离矩阵：[Batch, N+1, N+1]
    diff = all_nodes.unsqueeze(2) - all_nodes.unsqueeze(1)
    dists = torch.sum(diff ** 2, dim=-1).sqrt()
    
    # 对角线设为无穷大（避免自己到自己）
    dists.diagonal(dim1=1, dim2=2).fill_(float('inf'))
    
    tours = []  # 每个batch的tour
    
    for b in range(batch_size):
        tour = []
        visited = torch.zeros(n_nodes + 1, dtype=torch.bool, device=device)
        current_load = 0.0
        current_node = 0  # 从depot开始
        
        tour.append(current_node)
        visited[0] = True  # depot标记为已访问（但可以重复访问）
        
        unvisited_customers = n_nodes  # 剩余未访问的客户数
        
        while unvisited_customers > 0:
            # 找到最近的可访问节点
            # 1) 未访问过
            # 2) 容量足够
            
            best_node = None
            best_dist = float('inf')
            
            for node_idx in range(1, n_nodes + 1):
                if visited[node_idx]:
                    continue
                
                demand = node_demand[b, node_idx - 1].item()
                
                # 检查容量
                if current_load + demand <= capacity + 1e-5:
                    dist = dists[b, current_node, node_idx].item()
                    if dist < best_dist:
                        best_dist = dist
                        best_node = node_idx
            
            if best_node is not None:
                # 访问该节点
                tour.append(best_node)
                visited[best_node] = True
                current_load += node_demand[b, best_node - 1].item()
                current_node = best_node
                unvisited_customers -= 1
            else:
                # 没有可访问的节点（容量不足），返回depot
                tour.append(0)
                current_load = 0.0
                current_node = 0
        
        # 最后返回depot
        if tour[-1] != 0:
            tour.append(0)
        
        tours.append(torch.tensor(tour, dtype=torch.long, device=device))
    
    # Padding到相同长度
    max_len = max(len(t) for t in tours)
    padded_tours = []
    for tour in tours:
        if len(tour) < max_len:
            padding = torch.zeros(max_len - len(tour), dtype=torch.long, device=device)
            tour = torch.cat([tour, padding])
        padded_tours.append(tour)
    
    return torch.stack(padded_tours, dim=0)


def get_greedy_tour(depot_xy, node_xy, node_demand, capacity=1.0):
    """
    另一种贪婪策略：扫描法（Sweep Algorithm）
    
    1. 按极角排序客户节点
    2. 顺序访问，容量不足时返回depot
    
    Args:
        depot_xy: [Batch, 1, 2]
        node_xy: [Batch, N, 2]
        node_demand: [Batch, N]
        capacity: 车辆容量
        
    Returns:
        tours: [Batch, L]
    """
    batch_size = depot_xy.size(0)
    n_nodes = node_xy.size(1)
    device = depot_xy.device
    
    tours = []
    
    for b in range(batch_size):
        # 计算极角
        depot = depot_xy[b, 0]  # [2]
        nodes = node_xy[b]  # [N, 2]
        
        # 相对坐标
        relative = nodes - depot.unsqueeze(0)
        angles = torch.atan2(relative[:, 1], relative[:, 0])
        
        # 按角度排序
        sorted_indices = torch.argsort(angles)
        
        # 构建tour
        tour = [0]  # 从depot开始
        current_load = 0.0
        
        for idx in sorted_indices:
            customer_idx = idx.item() + 1  # 转为1-indexed
            demand = node_demand[b, idx].item()
            
            if current_load + demand <= capacity + 1e-5:
                # 可以访问
                tour.append(customer_idx)
                current_load += demand
            else:
                # 容量不足，返回depot
                tour.append(0)
                tour.append(customer_idx)
                current_load = demand
        
        # 最后返回depot
        if tour[-1] != 0:
            tour.append(0)
        
        tours.append(torch.tensor(tour, dtype=torch.long, device=device))
    
    # Padding
    max_len = max(len(t) for t in tours)
    padded_tours = []
    for tour in tours:
        if len(tour) < max_len:
            padding = torch.zeros(max_len - len(tour), dtype=torch.long, device=device)
            tour = torch.cat([tour, padding])
        padded_tours.append(tour)
    
    return torch.stack(padded_tours, dim=0)
