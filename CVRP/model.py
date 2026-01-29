import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Mamba实现 - 与TSP版本相同
_HAS_MAMBA_SSM = False
try:
    from mamba_ssm.modules.mamba_simple import Mamba as _MambaSSM
    _HAS_MAMBA_SSM = True
except Exception:
    try:
        from mamba_ssm import Mamba as _MambaSSM
        _HAS_MAMBA_SSM = True
    except Exception:
        _MambaSSM = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm_x * self.weight


class SSMSequenceBlock(nn.Module):
    """纯PyTorch实现的SSM块（当mamba_ssm不可用时）"""
    
    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.norm = RMSNorm(d_model)
        self.A_log = nn.Parameter(torch.zeros(d_state))
        self.B = nn.Linear(d_model, d_state, bias=False)
        self.C = nn.Linear(d_state, d_model, bias=False)
        self.D = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        batch_size, n, _ = x.size()
        state = x.new_zeros(batch_size, self.d_state)
        a = torch.exp(-torch.exp(self.A_log)).unsqueeze(0)

        outputs = []
        for t in range(n):
            xt = x[:, t, :]
            state = state * a + self.B(xt)
            yt = self.C(state) + self.D(xt)
            outputs.append(yt)

        y = torch.stack(outputs, dim=1)
        y = F.silu(self.gate(x)) * y
        return residual + y


class GraphMambaEncoder(nn.Module):
    """
    Mamba编码器
    
    Input:  x [B, N, input_dim]
    Output: h [B, N, embed_dim]
    """

    def __init__(self, input_dim=3, embed_dim=128, n_layers=3):
        super(GraphMambaEncoder, self).__init__()
        self.init_embed = nn.Linear(input_dim, embed_dim)

        if _HAS_MAMBA_SSM:
            self.layers = nn.ModuleList([
                _MambaSSM(
                    d_model=embed_dim,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                )
                for _ in range(n_layers)
            ])
            self.norm = RMSNorm(embed_dim)
            self._uses_external_mamba = True
        else:
            self.layers = nn.ModuleList([
                SSMSequenceBlock(embed_dim, d_state=64) 
                for _ in range(n_layers)
            ])
            self.norm = RMSNorm(embed_dim)
            self._uses_external_mamba = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.init_embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return h


class CVRPModel(nn.Module):
    """
    CVRP模型 - 基于Mamba的自回归解码器
    
    输入：
        - depot_xy: [B, 1, 2]
        - node_xy: [B, N, 2]
        - node_demand: [B, N]
    
    输出：
        - tour: [B, L] 访问序列（包含多次返回depot）
        - log_probs: [B] 总对数概率
    """
    
    def __init__(self, 
                 embedding_dim=128, 
                 hidden_dim=128, 
                 n_encode_layers=3):
        super(CVRPModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Encoder: 将 [depot+nodes, 3] -> [depot+nodes, D]
        # depot: [x, y, 0], nodes: [x, y, demand]
        self.encoder = GraphMambaEncoder(
            input_dim=3, 
            embed_dim=embedding_dim, 
            n_layers=n_encode_layers
        )
        
        # Decoder SSM状态
        self.decoder_A_log = nn.Parameter(torch.zeros(hidden_dim))
        self.decoder_B = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.decoder_C = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.decoder_D = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.node_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.state_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        # 容量信息投影（用于decoder状态更新）
        self.capacity_proj = nn.Linear(1, embedding_dim, bias=False)
        
        # 可学习的初始上下文
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)

    def _decoder_step(self, ssm_state: torch.Tensor, x_in: torch.Tensor):
        """SSM解码器单步更新"""
        a = torch.exp(-torch.exp(self.decoder_A_log)).unsqueeze(0)
        ssm_state = ssm_state * a + self.decoder_B(x_in)
        dec_vec = self.decoder_C(ssm_state) + self.decoder_D(x_in)
        return ssm_state, dec_vec

    def forward(self, depot_xy, node_xy, node_demand, 
                target_tour=None, teacher_forcing=True, temperature=1.0):
        """
        前向传播
        
        Args:
            depot_xy: [B, 1, 2]
            node_xy: [B, N, 2]
            node_demand: [B, N]
            target_tour: [B, L] (可选) 目标路径，用于teacher forcing
            teacher_forcing: 是否使用teacher forcing
            temperature: 采样温度
            
        Returns:
            tour_indices: [B, L] 生成的路径
            sum_log_probs: [B] 总对数概率
        """
        batch_size = depot_xy.size(0)
        n_nodes = node_xy.size(1)
        
        # --- 1. 编码 ---
        # 准备输入：[B, N+1, 3]
        # depot: [x, y, 0], nodes: [x, y, demand]
        depot_features = torch.cat([
            depot_xy, 
            torch.zeros(batch_size, 1, 1, device=depot_xy.device)
        ], dim=2)  # [B, 1, 3]
        
        node_features = torch.cat([
            node_xy, 
            node_demand.unsqueeze(2)
        ], dim=2)  # [B, N, 3]
        
        all_features = torch.cat([depot_features, node_features], dim=1)  # [B, N+1, 3]
        
        # 编码: [B, N+1, D]
        embeddings = self.encoder(all_features)
        
        # 全局上下文
        fixed_context = embeddings.mean(dim=1)
        
        # 预计算节点投影
        node_proj = self.node_proj(embeddings)

        # --- 2. 解码初始化 ---
        # 当前载重（剩余容量）
        current_load = torch.ones(batch_size, device=depot_xy.device)
        
        # Mask: True表示已访问（不可再访问）
        visited_mask = torch.zeros(batch_size, n_nodes + 1, dtype=torch.bool, device=depot_xy.device)
        # Depot(索引0)初始时可访问
        
        log_probs_list = []
        tour_indices = []
        
        # 初始节点embedding
        last_node_embedding = self.W_placeholder[None, :self.embedding_dim].expand(batch_size, -1)
        
        # Decoder SSM状态
        ssm_state = depot_xy.new_zeros(batch_size, self.hidden_dim)
        
        # 最大步数：问题规模N + 可能的depot访问次数
        # 保守估计：每个节点最多需要一次depot往返，即最多N次额外depot访问
        max_steps = n_nodes * 3  # N个客户 + 最多2N个depot访问
        
        # --- 3. 自回归解码 ---
        for step in range(max_steps):
            # A. 更新decoder状态
            # 结合当前载重信息
            capacity_emb = self.capacity_proj(current_load.unsqueeze(1))  # [B, D]
            decoder_input = last_node_embedding + capacity_emb
            
            ssm_state, decoder_vec = self._decoder_step(ssm_state, decoder_input)
            decoder_vec = decoder_vec + fixed_context

            # B. 计算每个节点的得分
            state_proj = self.state_proj(decoder_vec)
            final_scores = (node_proj * state_proj[:, None, :]).sum(dim=-1) / math.sqrt(self.embedding_dim)

            # C. 应用mask
            # 1) 已访问的节点不能再访问
            final_scores = final_scores.masked_fill(visited_mask, float('-inf'))
            
            # 2) 容量不足的节点不能访问
            # 计算访问每个节点需要的容量
            all_demands = torch.cat([
                torch.zeros(batch_size, 1, device=depot_xy.device),  # depot需求=0
                node_demand
            ], dim=1)  # [B, N+1]
            
            capacity_mask = (current_load.unsqueeze(1) + 1e-5) < all_demands
            capacity_mask[:, 0] = False  # depot永远可访问（用于补货）
            final_scores = final_scores.masked_fill(capacity_mask, float('-inf'))
            
            # 应用温度
            if not (target_tour is not None and teacher_forcing):
                final_scores = final_scores / temperature
            
            step_probs = F.softmax(final_scores, dim=-1)
            step_log_probs = F.log_softmax(final_scores, dim=-1)
            
            # D. 选择节点
            if target_tour is not None and teacher_forcing:
                if step < target_tour.size(1):
                    selected_idx = target_tour[:, step]
                else:
                    # target_tour已经结束，停止
                    break
            else:
                # 采样或贪婪
                if self.training:
                    m = torch.distributions.Categorical(step_probs)
                    selected_idx = m.sample()
                else:
                    selected_idx = step_probs.argmax(dim=1)
            
            # E. 记录
            selected_log_prob = step_log_probs.gather(1, selected_idx.unsqueeze(1)).squeeze(1)
            log_probs_list.append(selected_log_prob)
            tour_indices.append(selected_idx)
            
            # F. 更新状态
            # 1) 更新visited mask（客户节点）
            is_customer = selected_idx > 0
            visited_mask.scatter_(
                1, 
                selected_idx.unsqueeze(1), 
                is_customer.unsqueeze(1)
            )
            
            # 2) 更新载重
            selected_demand = all_demands.gather(1, selected_idx.unsqueeze(1)).squeeze(1)
            is_depot = (selected_idx == 0)
            current_load = torch.where(
                is_depot,
                torch.ones_like(current_load),  # 在depot时补满
                current_load - selected_demand   # 否则减少
            )
            
            # 3) 更新last_node_embedding
            last_node_embedding = embeddings.gather(
                1, 
                selected_idx.view(batch_size, 1, 1).expand(-1, -1, self.embedding_dim)
            ).squeeze(1)
            
            # G. 终止条件：所有客户都已访问且回到depot
            all_customers_visited = visited_mask[:, 1:].all(dim=1)  # 除了depot外都访问了
            currently_at_depot = (selected_idx == 0)
            done = all_customers_visited & currently_at_depot
            
            if done.all():
                break
        
        # 结果
        sum_log_probs = torch.stack(log_probs_list, dim=1).sum(dim=1)
        out_tour = torch.stack(tour_indices, dim=1)
        
        return out_tour, sum_log_probs


# 兼容性别名
AttentionModel = CVRPModel
