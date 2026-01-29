import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Optional dependency: real Mamba implementation (mamba-ssm).
# If unavailable (common on Windows without a matching build toolchain), we fall back
# to a pure-PyTorch linear-time "Mamba-like" block so the project keeps running.
_HAS_MAMBA_SSM = False
try:
    # Most common import path
    from mamba_ssm.modules.mamba_simple import Mamba as _MambaSSM  # type: ignore

    _HAS_MAMBA_SSM = True
except Exception:
    try:
        # Alternate import path in some versions
        from mamba_ssm import Mamba as _MambaSSM  # type: ignore

        _HAS_MAMBA_SSM = True
    except Exception:
        _MambaSSM = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm_x * self.weight


class SSMSequenceBlock(nn.Module):
    """A simple state-space sequence mixer with explicit A/B/C/D parameters.

    This is a pure-PyTorch fallback when `mamba_ssm` isn't available.
    It removes attention/QKV entirely while still mixing information along the
    sequence dimension in O(N) time.

    Shapes:
      x: [B, N, D]
      y: [B, N, D]
    """

    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.norm = RMSNorm(d_model)

        # A: diagonal stable dynamics (negative exp)
        self.A_log = nn.Parameter(torch.zeros(d_state))
        # B: input -> state
        self.B = nn.Linear(d_model, d_state, bias=False)
        # C: state -> output
        self.C = nn.Linear(d_state, d_model, bias=False)
        # D: skip connection on input
        self.D = nn.Linear(d_model, d_model, bias=False)

        self.gate = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        batch_size, n, _ = x.size()
        # State: [B, d_state]
        state = x.new_zeros(batch_size, self.d_state)

        # Stable diagonal A in (0,1): a = exp(-exp(A_log))
        a = torch.exp(-torch.exp(self.A_log)).unsqueeze(0)  # [1, d_state]

        outputs = []
        for t in range(n):
            xt = x[:, t, :]  # [B, D]
            state = state * a + self.B(xt)
            yt = self.C(state) + self.D(xt)
            outputs.append(yt)

        y = torch.stack(outputs, dim=1)  # [B, N, D]
        y = F.silu(self.gate(x)) * y
        return residual + y

class GraphAttentionEncoder(nn.Module):
    """Deprecated: kept only for backwards compatibility with older checkpoints.

    The current code path uses GraphMambaEncoder.
    """

    def __init__(self, embed_dim=128, n_heads=8, n_layers=3):
        super(GraphAttentionEncoder, self).__init__()
        self.init_embed = nn.Linear(2, embed_dim)
        self.layers = nn.ModuleList([nn.Identity() for _ in range(n_layers)])

    def forward(self, x):
        return self.init_embed(x)


class GraphMambaEncoder(nn.Module):
    """Drop-in replacement for GraphAttentionEncoder.

    Input:  x [B, N, 2]
    Output: h [B, N, D]

    If `mamba_ssm` is installed, uses the real Mamba block.
    Otherwise uses a pure-PyTorch Mamba-like linear-time block.
    """

    def __init__(self, embed_dim=128, n_layers=3):
        super(GraphMambaEncoder, self).__init__()
        self.init_embed = nn.Linear(2, embed_dim)

        if _HAS_MAMBA_SSM:
            self.layers = nn.ModuleList(
                [
                    _MambaSSM(
                        d_model=embed_dim,
                        d_state=16,
                        d_conv=4,
                        expand=2,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.norm = RMSNorm(embed_dim)
            self._uses_external_mamba = True
        else:
            # Pure PyTorch fallback with explicit A/B/C/D params.
            self.layers = nn.ModuleList([SSMSequenceBlock(embed_dim, d_state=64) for _ in range(n_layers)])
            self.norm = RMSNorm(embed_dim)
            self._uses_external_mamba = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.init_embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return h

class AttentionModel(nn.Module):
    """
    Kool et al. (2019) Attention Model (used in POMO).
    Adapted for DPO Training (supports teacher_forcing evaluation).
    """
    def __init__(self, 
                 embedding_dim=128, 
                 hidden_dim=128, 
                 n_heads=8, 
                 n_encode_layers=3):
        super(AttentionModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Encoder (Mamba for speed; keeps output shape identical)
        # NOTE: `n_heads` is kept for API compatibility with the original Transformer-style encoder.
        self.encoder = GraphMambaEncoder(embed_dim=embedding_dim, n_layers=n_encode_layers)
        
        # Decoder (SSM-based; removes attention/QKV)
        # We keep the same external behavior: sequential node selection with masking.
        self.decoder_A_log = nn.Parameter(torch.zeros(hidden_dim))
        self.decoder_B = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.decoder_C = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.decoder_D = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.node_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.state_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        # Learnable initial context
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)

    def _make_heads(self, v, num_steps=None):
        # v: [B, N, Dim] or [B, Dim]
        assert v.size(-1) == 3 * self.embedding_dim
        batch_size = v.size(0)
        n = v.size(1) if num_steps is None else num_steps
        v = v.view(batch_size, n, self.n_heads, -1).permute(0, 2, 1, 3)
        return v

    def _decoder_step(self, ssm_state: torch.Tensor, x_in: torch.Tensor):
        """One recurrent SSM update.

        ssm_state: [B, H]
        x_in:      [B, D]
        returns: (new_state [B, H], decoder_vec [B, D])
        """
        a = torch.exp(-torch.exp(self.decoder_A_log)).unsqueeze(0)  # [1, H]
        ssm_state = ssm_state * a + self.decoder_B(x_in)
        dec_vec = self.decoder_C(ssm_state) + self.decoder_D(x_in)
        return ssm_state, dec_vec

    def forward(self, x, target_tour=None, teacher_forcing=True, temperature=1.0):
        """
        x: [Batch, N, 2]
        temperature: 采样温度参数 (仅在非 teacher_forcing 模式下生效)
                    - temperature > 1: 更随机，增加探索
                    - temperature < 1: 更确定，增加利用
                    - temperature = 1: 标准采样
        target_tour: [Batch, N] (Optional) 用于 DPO 计算 LogProb
        teacher_forcing: bool
        """
        batch_size, n_nodes, _ = x.size()
        
        # --- 1. Encoding ---
        # embeddings: [B, N, D]
        embeddings = self.encoder(x)
        
        # Graph Context: global average pooling [B, D]
        fixed_context = embeddings.mean(dim=1)
        
        # Precompute node projections once for fast scoring: [B, N, D]
        node_proj = self.node_proj(embeddings)

        # --- 2. Decoding State Initialization ---
        # Mask: 1 means unvisited (True), 0 means visited (False) 
        # Note: Logits masking usually uses -inf for masked positions
        mask = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=x.device)
        
        log_probs_list = []
        tour_indices = []
        
        # Last node (for context)
        last_node_embedding = self.W_placeholder[None, :self.embedding_dim].expand(batch_size, -1)

        # Decoder SSM state
        ssm_state = x.new_zeros(batch_size, self.hidden_dim)
        
        # Loop Steps
        for i in range(n_nodes):
            # A. Update decoder state using an SSM (A/B/C/D) recurrence.
            ssm_state, decoder_vec = self._decoder_step(ssm_state, last_node_embedding)

            # (Optional) include a global graph context without attention
            decoder_vec = decoder_vec + fixed_context

            # B. Score each node with a bilinear form (no Q/K/V, no softmax-attn).
            state_proj = self.state_proj(decoder_vec)  # [B, D]
            final_scores = (node_proj * state_proj[:, None, :]).sum(dim=-1) / math.sqrt(self.embedding_dim)  # [B, N]

            # D. Masking & Probs
            final_scores = final_scores.masked_fill(mask, float('-inf'))
            
            # 应用温度缩放（仅在采样模式下）
            if not (target_tour is not None and teacher_forcing):
                final_scores = final_scores / temperature
            
            step_probs = F.softmax(final_scores, dim=-1)
            step_log_probs = F.log_softmax(final_scores, dim=-1)
            
            # E. Selection (Sampling or Teacher Forcing)
            if target_tour is not None and teacher_forcing:
                # DPO Training: Use the given path
                selected_idx = target_tour[:, i]
            else:
                # Inference / Sampling
                if self.training:
                    m = torch.distributions.Categorical(step_probs)
                    selected_idx = m.sample()
                else:
                    selected_idx = step_probs.argmax(dim=1)
            
            # Collect data
            # Gather log_prob of the *selected* action
            selected_log_prob = step_log_probs.gather(1, selected_idx.unsqueeze(1)).squeeze(1)
            log_probs_list.append(selected_log_prob)
            tour_indices.append(selected_idx)
            
            # F. Update State
            # IMPORTANT: do not update `mask` in-place. Autograd may have saved the old mask
            # for backward (e.g., through masked_fill), and in-place modification breaks it.
            mask = mask.scatter(
                1,
                selected_idx.unsqueeze(1),
                torch.ones(batch_size, 1, dtype=torch.bool, device=x.device),
            )
            
            # Update 'last_node_embedding' for next step context
            # embeddings: [B, N, D]
            last_node_embedding = embeddings.gather(1, selected_idx.view(batch_size, 1, 1).expand(-1, -1, self.embedding_dim)).squeeze(1)

        # Result
        sum_log_probs = torch.stack(log_probs_list, dim=1).sum(dim=1)
        out_tour = torch.stack(tour_indices, dim=1)
        
        return out_tour, sum_log_probs

# 兼容性重命名，方便主程序调用
SimplifiedPointerNetwork = AttentionModel