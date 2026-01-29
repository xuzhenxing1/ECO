import torch

def get_nearest_neighbor_tour(nodes):
    """
    一个简单的贪婪策略作为 SFT 的老师。
    nodes: [Batch, N, 2]
    Returns: [Batch, N] (indices)
    """
    batch_size, n_nodes, _ = nodes.size()
    # 距离矩阵: [B, N, N]
    diff = nodes.unsqueeze(2) - nodes.unsqueeze(1)
    dists = torch.sum(diff ** 2, dim=-1).sqrt()
    
    # 设对角线为无穷大 (自己不能访问自己)
    dists.diagonal(dim1=1, dim2=2).fill_(float('inf'))
    
    tours = []
    mask = torch.ones(batch_size, n_nodes, dtype=torch.bool, device=nodes.device)
    
    # 随机选择起点 (为了增加 SFT 数据的多样性，不要总是从索引0开始)
    current_node = torch.randint(0, n_nodes, (batch_size,), device=nodes.device)
    
    for _ in range(n_nodes):
        tours.append(current_node)
        # 标记当前节点已访问
        mask.scatter_(1, current_node.unsqueeze(1), False)
        
        # 将已访问列的距离设为无穷大
        dists.scatter_(2, current_node.unsqueeze(1).unsqueeze(2).expand(-1, n_nodes, -1), float('inf'))
        
        # 找最近的下一个点
        # 注意：最后一步时 mask 全为 False，argmin 会报错，需要处理
        if len(tours) < n_nodes:
            # 只在未访问的节点中找最近的
            # dists [B, current, all_next]
            current_dists = dists.gather(1, current_node.view(batch_size, 1, 1).expand(-1, 1, n_nodes)).squeeze(1)
            next_node = current_dists.argmin(dim=1)
            current_node = next_node
            
    return torch.stack(tours, dim=1)