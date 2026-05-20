"""
MSTBP-mHC: Margined Scaled Transport Birkhoff Polytope Manifold-Constrained Hyper-Connections

This module implements manifold-constrained hyper-connections using a Margined
Scaled Transport parameterization for EXACT doubly stochastic matrices. It extends
STBP-mHC by adding a margin (rho) that keeps entries away from the interval boundaries,
improving gradient flow and numerical stability.

===============================================================================
MOTIVATION
===============================================================================

STBP-mHC uses sigmoid output in [0, 1], which can saturate near boundaries.
MSTBP-mHC constrains the fraction to [ρ, 1-ρ] with ρ=1e-4, ensuring:
- Entries never exactly hit L or U (avoids vanishing gradients at boundaries)
- Better gradient flow during training
- Improved numerical stability

===============================================================================
MARGINED SCALED TRANSPORT BIRKHOFF PARAMETERIZATION
===============================================================================

Entry formula:
    X[i,j] = L_ij + (U_ij - L_ij) * (ρ + (1 - 2ρ) * sigmoid(β*t[i,j]/(U_ij - L_ij + ε)))

With β=4, ε=1e-3, ρ=1e-4. The fraction maps to [ρ, 1-ρ] instead of [0, 1].

===============================================================================
COMPARISON WITH OTHER METHODS
===============================================================================

| Method    | Exact DS? | Params for H^res | Memory      | Restriction |
|-----------|-----------|------------------|-------------|-------------|
| mHC       | No (~SK)  | O(n²)            | O(n²)       | None        |
| mHC-lite  | Yes       | O(n² × n!)       | O(n² × n!)  | n ≤ ~6      |
| KromHC    | Yes       | O(K × 4) = O(log n) | O(n²)   | n = 2^K     |
| MSTBP-mHC | Yes       | O((n-1)²)        | O(n²)       | n ≥ 2       |
"""

from __future__ import annotations
from typing import Callable, Optional

from functools import partial

import torch
from torch import nn, cat
from torch.nn import Module

from einops import rearrange, repeat

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
# CORE: MARGINED SCALED TRANSPORT BIRKHOFF PARAMETERIZATION
# ============================================================================

# Margined STBP constants
MSTBP_BETA = 4.0
MSTBP_EPSILON = 1e-3
MSTBP_RHO = 1e-4


def margined_sequential_birkhoff(
    t: torch.Tensor,
    *,
    delta_logit: Optional[torch.Tensor] = None,
    eps: float = 1e-7,
    beta: float = MSTBP_BETA,
    epsilon: float = MSTBP_EPSILON,
    rho: float = MSTBP_RHO,
) -> torch.Tensor:
    """
    Map free parameters t to an exactly doubly stochastic matrix using
    Margined Scaled Transport Birkhoff Polytope parameterization.
    
    Entry formula: X[i,j] = L_ij + (U_ij - L_ij) * (ρ + (1-2ρ)*sigmoid(β*t/(U_ij - L_ij + ε)))

    Args:
        t: (..., n-1, n-1) unconstrained parameters
        delta_logit: optional scalar/broadcastable logit for minorization
        eps: small safety clamp for numerical stability (width)
        beta: scale factor for sigmoid argument (default 4)
        epsilon: denominator offset for scaled transport (default 1e-3)
        rho: margin keeping fraction in [ρ, 1-ρ] (default 1e-4)

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

    X_rows = []
    r = torch.ones((B, n), device=device, dtype=dtype)
    c = torch.ones((B, n), device=device, dtype=dtype)

    for i in range(n - 1):
        row_entries = []
        ri = r[:, i]
        
        for j in range(n - 1):
            cj = c[:, j]
            
            c_tail = c[:, j+1:].sum(dim=-1)
            r_tail = r[:, i+1:].sum(dim=-1)
            
            L = torch.clamp(torch.maximum(ri - c_tail, cj - r_tail), min=0)
            U = torch.minimum(ri, cj)
            width = (U - L).clamp_min(eps)
            
            # Margined entry: X[i,j] = L + (U-L) * (ρ + (1-2ρ)*sigmoid(β*t/(U-L+ε)))
            scaled_arg = beta * tt[:, i, j] / (width + epsilon)
            frac = rho + (1.0 - 2.0 * rho) * torch.sigmoid(scaled_arg)
            x = L + width * frac
            row_entries.append(x)
            
            ri = ri - x
            c = c.clone()
            c[:, j] = c[:, j] - x
        
        row_entries.append(ri)
        c = c.clone()
        c[:, n-1] = c[:, n-1] - ri
        
        r = r.clone()
        r[:, i] = torch.zeros(B, device=device, dtype=dtype)
        
        X_rows.append(torch.stack(row_entries, dim=-1))
    
    X_rows.append(c)
    X = torch.stack(X_rows, dim=-2)
    X = X.reshape(*batch_shape, n, n)

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
    """Get initializer for MSTBP-mHC layers plus expand/reduce functions."""
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = MSTBP_MHC if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, num_fracs=num_fracs, **kwargs)
    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams, add_stream_embed=add_stream_embed, dim=dim, disable=disable
    )

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim=dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


# ============================================================================
# MSTBP-mHC IMPLEMENTATION
# ============================================================================

class MSTBP_MHC(BaseHyperConnections):
    """
    MSTBP-mHC: Margined Scaled Transport Birkhoff Polytope Manifold-Constrained Hyper-Connections.
    
    Uses the margined scaled transport Birkhoff parameterization to construct EXACTLY doubly
    stochastic H^res matrices from only (n-1)² free parameters.
    
    The margin ρ keeps entries away from interval boundaries for better gradient flow.
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
        assert num_residual_streams >= 2, "MSTBP-mHC requires at least 2 streams"
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
        """Initialize MSTBP-mHC parameters: H^pre, H^res (n²), H^post, delta."""
        n = self.num_residual_streams
        f = self.num_fracs
        v = self.num_input_views
        d = self.dim_per_frac
        
        init_alpha_pre = torch.ones((n * f, v * f)) * -1
        init_alpha_pre[self.init_residual_index, :] = 1.
        
        init_alpha_res = torch.zeros((n * f, n * f))
        
        self.static_alpha = nn.Parameter(cat((init_alpha_pre, init_alpha_res), dim=1))
        
        self.dynamic_alpha_fn = nn.Parameter(
            torch.zeros(d * n, f * (n * n + n * v))
        )
        
        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
        self.residual_scale = nn.Parameter(torch.ones(1) * 1e-2)
        
        if self.make_dse:
            self.delta_logit = nn.Parameter(torch.tensor(-8.0))
        else:
            self.delta_logit = None
        
        if self.add_branch_out_to_residual:
            beta_init = torch.ones(n * f) * -1.
            beta_init[self.init_residual_index] = 1.
            self.static_beta = nn.Parameter(beta_init)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(d * n, f * n))
            self.h_post_scale = nn.Parameter(torch.ones(()) * 1e-2)
    
    def _compute_alpha_beta(self, normed: torch.Tensor, device: torch.device):
        """Compute H^pre, H^res (margined sequential Birkhoff), and H^post matrices."""
        n = self.num_residual_streams
        f = self.num_fracs
        v = self.num_input_views
        
        wc_weight = normed @ self.dynamic_alpha_fn
        wc_weight = rearrange(wc_weight, '... (s t) -> ... s t', s=n)
        
        pre_scale = repeat(self.pre_branch_scale, '1 -> v', v=v * f)
        res_scale = repeat(self.residual_scale, '1 -> s', s=f * n)
        alpha_scale = cat((pre_scale, res_scale))
        
        dynamic_alpha = wc_weight * alpha_scale
        static_alpha = rearrange(self.static_alpha, '(f s) t -> f s t', s=n)
        alpha = dynamic_alpha + static_alpha
        alpha = self.split_fracs(alpha)
        
        alpha_pre, alpha_res = alpha[..., :v], alpha[..., v:]
        alpha_pre = alpha_pre.sigmoid()
        
        alpha_res = rearrange(alpha_res, '... f s g t -> ... f g s t')
        birkhoff_params = alpha_res[..., :n-1, :n-1]
        
        orig_shape = birkhoff_params.shape[:-2]
        birkhoff_flat = birkhoff_params.reshape(-1, n-1, n-1)
        
        ds_flat = margined_sequential_birkhoff(
            birkhoff_flat,
            delta_logit=self.delta_logit if self.make_dse else None
        )
        
        alpha_res = ds_flat.reshape(*orig_shape, n, n)
        alpha_res = rearrange(alpha_res, '... f g s t -> ... f s g t')
        
        alpha = cat((alpha_pre, alpha_res), dim=-1)
        
        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = normed @ self.dynamic_beta_fn
            dc_weight = rearrange(dc_weight, '... (s f) -> ... s f', s=n)
            dynamic_beta = dc_weight * self.h_post_scale
            static_beta = rearrange(self.static_beta, '(s f) -> s f', s=n)
            beta = (dynamic_beta + static_beta).sigmoid() * 2
        
        return alpha, beta


# Static method attachments
MSTBP_MHC.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
MSTBP_MHC.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)


# ============================================================================
# LEGACY WRAPPER
# ============================================================================

class MarginedScaledTransportBirkhoff(nn.Module):
    """
    Standalone nn.Module wrapper for margined_sequential_birkhoff().
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

        return margined_sequential_birkhoff(
            t,
            delta_logit=self.delta_logit if self.make_dse else None,
            eps=self.eps
        )
