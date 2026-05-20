"""
STBP-mHC: Scaled Transport Birkhoff Polytope Manifold-Constrained Hyper-Connections

This module implements manifold-constrained hyper-connections using a Scaled
Transport parameterization for EXACT doubly stochastic matrices. It extends
the sequential Birkhoff construction with a scaled sigmoid for improved
gradient flow and transport-like behavior.

===============================================================================
MOTIVATION
===============================================================================

Existing approaches have trade-offs:
- mHC (Sinkhorn-Knopp): Only APPROXIMATE DS, error accumulates across layers
- mHC-lite (BvN): Exact DS, but O(n!) parameter/memory complexity
- KromHC (Kronecker): Exact DS, O(n²C) params, but requires power-of-2 streams

STBP-mHC offers a new trade-off:
- EXACT doubly stochastic matrices (no approximation error)
- Only (n-1)² free parameters per DS matrix (vs n! for mHC-lite)
- Works for ANY n >= 2 (no power-of-2 restriction like KromHC)
- Scaled transport parameterization: X[i,j] = L_ij + (U_ij - L_ij) * sigmoid(β*t/(U_ij-L_ij+ε))

===============================================================================
SCALED TRANSPORT BIRKHOFF PARAMETERIZATION
===============================================================================

Instead of projecting (SK) or using convex combinations of permutations (BvN),
we directly CONSTRUCT a doubly stochastic matrix from (n-1)² free parameters.

The construction maintains row remainders r and column remainders c:
1. Initialize r = [1, 1, ..., 1], c = [1, 1, ..., 1]
2. For each entry (i,j) with i,j < n-1:
   - Compute feasible interval [L_ij, U_ij] from remainders
   - Set X[i,j] = L_ij + (U_ij - L_ij) * sigmoid(β*t[i,j]/(U_ij - L_ij + ε))
     with β=4, ε=1e-3 (scaled transport for improved gradient flow)
   - Update remainders: r[i] -= X[i,j], c[j] -= X[i,j]
3. Last column/row determined by remainders

This GUARANTEES X >= 0, X @ 1 = 1, X.T @ 1 = 1 exactly (up to float precision).

Optional strict positivity via uniform minorization:
    H = (1 - δ) * X + δ * (1/n) * 11^T,  where δ = sigmoid(s)

===============================================================================
COMPARISON WITH OTHER METHODS
===============================================================================

| Method   | Exact DS? | Params for H^res | Memory      | Restriction |
|----------|-----------|------------------|-------------|-------------|
| mHC      | No (~SK)  | O(n²)            | O(n²)       | None        |
| mHC-lite | Yes       | O(n!)            | O(n² × n!)  | n ≤ ~6      |
| KromHC   | Yes       | O(K × 4) = O(log n) | O(n²)   | n = 2^K     |
| STBP-mHC | Yes       | O((n-1)²)        | O(n²)       | n ≥ 2       |

===============================================================================
NOTATION
===============================================================================
b - batch dimension
d - feature dimension (C in paper)
s - number of residual streams (n in paper)
f - number of fractions
v - number of input views
"""

from __future__ import annotations
from typing import Callable, Optional

from functools import partial
from random import randrange

import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module

from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange

# Import shared utilities and base class
from .utils import (
    exists,
    divisible_by,
    default,
    add,
    Residual,
    get_expand_reduce_stream_functions,
)
from .base import BaseHyperConnections


# ============================================================================
# CORE: SCALED TRANSPORT BIRKHOFF PARAMETERIZATION
# ============================================================================

def _suffix_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Suffix sums with an appended 0 at the end.

    Input:  x shape (..., n)
    Output: s shape (..., n+1) where
        s[..., j] = sum_{k=j}^{n-1} x[..., k]
        s[..., n] = 0
    """
    s = torch.flip(torch.cumsum(torch.flip(x, dims=(-1,)), dim=-1), dims=(-1,))
    zero = torch.zeros_like(s[..., :1])
    return torch.cat([s, zero], dim=-1)


# Scaled Transport Birkhoff constants
STBP_BETA = 4.0
STBP_EPSILON = 1e-3


def sequential_birkhoff(
    t: torch.Tensor,
    *,
    delta_logit: Optional[torch.Tensor] = None,
    eps: float = 1e-7,
    beta: float = STBP_BETA,
    epsilon: float = STBP_EPSILON,
) -> torch.Tensor:
    """
    Map free parameters t to an exactly doubly stochastic matrix using
    Scaled Transport Birkhoff Polytope parameterization.
    
    Entry formula: X[i,j] = L_ij + (U_ij - L_ij) * sigmoid(β*t/(U_ij - L_ij + ε))

    Args:
        t: (..., n-1, n-1) unconstrained parameters
        delta_logit: optional scalar/broadcastable logit for minorization
        eps: small safety clamp for numerical stability (width)
        beta: scale factor for sigmoid argument (default 4)
        epsilon: denominator offset for scaled transport (default 1e-3)

    Returns:
        X: (..., n, n) exactly doubly stochastic matrix
    """
    if t.ndim < 2:
        raise ValueError(f"t must have shape (..., n-1, n-1), got {tuple(t.shape)}")

    n_minus_1_row = t.shape[-2]
    n_minus_1_col = t.shape[-1]
    if n_minus_1_row != n_minus_1_col:
        raise ValueError(f"t must be square in last two dims, got {tuple(t.shape)}")
    n = n_minus_1_row + 1

    batch_shape = t.shape[:-2]
    B = int(torch.tensor(batch_shape).prod().item()) if len(batch_shape) > 0 else 1
    tt = t.reshape(B, n - 1, n - 1)

    device = tt.device
    dtype = tt.dtype

    # Use lists to collect rows, then stack at the end (no in-place ops)
    X_rows = []
    
    # Track remainders as tensors (will be updated functionally)
    r = torch.ones((B, n), device=device, dtype=dtype)
    c = torch.ones((B, n), device=device, dtype=dtype)

    for i in range(n - 1):
        row_entries = []
        ri = r[:, i]
        
        for j in range(n - 1):
            cj = c[:, j]
            
            # Compute suffix sums for remaining columns/rows
            c_tail = c[:, j+1:].sum(dim=-1)  # sum of remaining column capacities
            r_tail = r[:, i+1:].sum(dim=-1)  # sum of remaining row capacities
            
            # Feasible interval [L, U]
            L = torch.clamp(torch.maximum(ri - c_tail, cj - r_tail), min=0)
            U = torch.minimum(ri, cj)
            width = (U - L).clamp_min(eps)
            
            # Scaled Transport entry: X[i,j] = L + (U-L) * sigmoid(β*t/(U-L+ε))
            scaled_arg = beta * tt[:, i, j] / (width + epsilon)
            x = L + width * torch.sigmoid(scaled_arg)
            row_entries.append(x)
            
            # Update remainders (functional, not in-place)
            ri = ri - x
            c = c.clone()
            c[:, j] = c[:, j] - x
        
        # Last column of this row = remaining row capacity
        row_entries.append(ri)
        c = c.clone()
        c[:, n-1] = c[:, n-1] - ri
        
        # Update r for next iteration
        r = r.clone()
        r[:, i] = torch.zeros(B, device=device, dtype=dtype)
        
        # Stack row entries: (B, n)
        X_rows.append(torch.stack(row_entries, dim=-1))
    
    # Last row = remaining column capacities
    X_rows.append(c)
    
    # Stack all rows: (B, n, n)
    X = torch.stack(X_rows, dim=-2)
    X = X.reshape(*batch_shape, n, n)

    # Optional strict positivity via uniform minorization
    if exists(delta_logit):
        delta = torch.sigmoid(delta_logit).to(dtype=dtype, device=device)
        while delta.ndim < X.ndim:
            delta = delta.unsqueeze(0)
        X = (1 - delta) * X + delta * (1.0 / n)

    return X


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def get_init_and_expand_reduce_stream_functions(
    num_streams,
    num_fracs = 1,
    dim = None,
    add_stream_embed = False,
    disable = None,
    **kwargs
):
    """Get initializer for STBP-mHC layers plus expand/reduce functions."""
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = STBP_MHC if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, num_fracs=num_fracs, **kwargs)
    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams, add_stream_embed=add_stream_embed, dim=dim, disable=disable
    )

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim=dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


# ============================================================================
# STBP-mHC IMPLEMENTATION
# ============================================================================

class STBP_MHC(BaseHyperConnections):
    """
    STBP-mHC: Scaled Transport Birkhoff Polytope Manifold-Constrained Hyper-Connections.
    
    Uses the scaled transport Birkhoff parameterization to construct EXACTLY doubly
    stochastic H^res matrices from only (n-1)² free parameters.
    
    KEY ADVANTAGES:
    - Exact DS (no approximation error like mHC's Sinkhorn-Knopp)
    - O((n-1)²) params for H^res (vs O(n!) for mHC-lite)
    - Works for any n >= 2 (no power-of-2 restriction like KromHC)
    - Pure PyTorch (no custom kernels needed)
    
    The sequential construction guarantees:
    - All entries non-negative: H^res >= 0
    - Rows sum to 1: H^res @ 1 = 1
    - Columns sum to 1: H^res.T @ 1 = 1
    
    INHERITANCE:
    ============
    Extends BaseHyperConnections, inheriting shared functionality.
    Implements _init_hyper_params() and _compute_alpha_beta() for sequential Birkhoff.
    """
    
    def __init__(
        self,
        num_residual_streams: int,
        *,
        dim: int,
        branch: Module | None = None,
        layer_index: int | None = None,
        channel_first: bool = False,
        dropout: float = 0.,
        residual_transform: Module | None = None,
        add_branch_out_to_residual: bool = True,
        num_input_views: int = 1,
        depth_residual_fn = add,
        num_fracs: int = 1,
        make_dse: bool = True,
    ):
        """
        Initialize LSB-mHC layer.
        
        Args:
            num_residual_streams: n, number of streams
            dim: C, feature dimension
            branch: Optional branch module F(·)
            layer_index: For deterministic initialization
            channel_first: If True, expect (batch, dim, ...) layout
            dropout: Dropout probability
            residual_transform: Transform on residual
            add_branch_out_to_residual: Enable depth connections
            num_input_views: Number of input views for branch
            depth_residual_fn: Function for output + residual
            num_fracs: Fractions for frac-connections
            make_dse: If True, add uniform minorization for strict positivity
        """
        assert num_residual_streams >= 2, "STBP-mHC requires at least 2 streams"
        self.make_dse = make_dse
        
        super().__init__(
            num_residual_streams,
            dim=dim,
            branch=branch,
            layer_index=layer_index,
            channel_first=channel_first,
            dropout=dropout,
            residual_transform=residual_transform,
            add_branch_out_to_residual=add_branch_out_to_residual,
            num_input_views=num_input_views,
            depth_residual_fn=depth_residual_fn,
            num_fracs=num_fracs,
        )
    
    def _init_hyper_params(self):
        """Initialize STBP-mHC parameters: H^pre, H^res (n²), H^post, delta."""
        n = self.num_residual_streams
        f = self.num_fracs
        v = self.num_input_views
        d = self.dim_per_frac
        
        # H^pre: mostly -1, one stream at +1
        init_alpha_pre = torch.ones((n * f, v * f)) * -1
        init_alpha_pre[self.init_residual_index, :] = 1.
        
        # H^res: n² params (extract (n-1)² for sequential_birkhoff)
        init_alpha_res = torch.zeros((n * f, n * f))
        
        self.static_alpha = nn.Parameter(cat((init_alpha_pre, init_alpha_res), dim=1))
        
        # Dynamic weights
        self.dynamic_alpha_fn = nn.Parameter(
            torch.zeros(d * n, f * (n * n + n * v))
        )
        
        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
        self.residual_scale = nn.Parameter(torch.ones(1) * 1e-2)
        
        # Delta for strict positivity (DSE)
        if self.make_dse:
            self.delta_logit = nn.Parameter(torch.tensor(-8.0))
        else:
            self.delta_logit = None
        
        # H^post
        if self.add_branch_out_to_residual:
            beta_init = torch.ones(n * f) * -1.
            beta_init[self.init_residual_index] = 1.
            self.static_beta = nn.Parameter(beta_init)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(d * n, f * n))
            self.h_post_scale = nn.Parameter(torch.ones(()) * 1e-2)
    
    def _compute_alpha_beta(self, normed: torch.Tensor, device: torch.device):
        """
        Compute H^pre, H^res (sequential Birkhoff), and H^post matrices.
        
        Uses scaled transport sequential_birkhoff to construct EXACT DS matrices from (n-1)² params.
        """
        n = self.num_residual_streams
        f = self.num_fracs
        v = self.num_input_views
        
        wc_weight = normed @ self.dynamic_alpha_fn
        wc_weight = rearrange(wc_weight, '... (s t) -> ... s t', s=n)
        
        # Apply scales
        pre_scale = repeat(self.pre_branch_scale, '1 -> v', v=v * f)
        res_scale = repeat(self.residual_scale, '1 -> s', s=f * n)
        alpha_scale = cat((pre_scale, res_scale))
        
        dynamic_alpha = wc_weight * alpha_scale
        static_alpha = rearrange(self.static_alpha, '(f s) t -> f s t', s=n)
        alpha = dynamic_alpha + static_alpha
        alpha = self.split_fracs(alpha)
        
        alpha_pre, alpha_res = alpha[..., :v], alpha[..., v:]
        alpha_pre = alpha_pre.sigmoid()
        
        # Sequential Birkhoff for exact DS
        alpha_res = rearrange(alpha_res, '... f s g t -> ... f g s t')
        birkhoff_params = alpha_res[..., :n-1, :n-1]
        
        orig_shape = birkhoff_params.shape[:-2]
        birkhoff_flat = birkhoff_params.reshape(-1, n-1, n-1)
        
        ds_flat = sequential_birkhoff(
            birkhoff_flat,
            delta_logit=self.delta_logit if self.make_dse else None
        )
        
        alpha_res = ds_flat.reshape(*orig_shape, n, n)
        alpha_res = rearrange(alpha_res, '... f g s t -> ... f s g t')
        
        alpha = cat((alpha_pre, alpha_res), dim=-1)
        
        # H^post
        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = normed @ self.dynamic_beta_fn
            dc_weight = rearrange(dc_weight, '... (s f) -> ... s f', s=n)
            dynamic_beta = dc_weight * self.h_post_scale
            static_beta = rearrange(self.static_beta, '(s f) -> s f', s=n)
            beta = (dynamic_beta + static_beta).sigmoid() * 2
        
        return alpha, beta


# Static method attachments
STBP_MHC.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
STBP_MHC.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)


# ============================================================================
# LEGACY WRAPPER (for compatibility)
# ============================================================================

class ScaledTransportBirkhoff(nn.Module):
    """
    Standalone nn.Module wrapper for scaled transport sequential_birkhoff().
    
    Usage:
        mapper = ScaledTransportBirkhoff(num_streams=n, make_dse=True)
        H = mapper(t)  # t shape (..., n-1, n-1), returns (..., n, n)
    """

    def __init__(
        self,
        num_streams: int,
        make_dse: bool = True,
        eps: float = 1e-7
    ):
        super().__init__()
        if num_streams < 2:
            raise ValueError("num_streams must be >= 2")
        self.num_streams = num_streams
        self.eps = eps

        self.make_dse = make_dse
        if make_dse:
            self.delta_logit = nn.Parameter(torch.tensor(-8.0))
        else:
            self.delta_logit = None

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        n = self.num_streams
        if t.shape[-2:] != (n - 1, n - 1):
            raise ValueError(f"Expected t shape (..., {n-1}, {n-1}), got {tuple(t.shape)}")

        return sequential_birkhoff(
            t,
            delta_logit=self.delta_logit if self.make_dse else None,
            eps=self.eps
        )
