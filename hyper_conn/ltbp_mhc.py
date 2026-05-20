"""
LTBP-mHC: Linear Transport Birkhoff Polytope Manifold-Constrained Hyper-Connections

This module implements a variant of LSB-mHC with a linear parameterization:
    X[i,j] = L_ij + (U_ij - L_ij) * t[i,j],  with t[i,j] ∈ [0, 1]

Instead of sigmoid, t is stored as explicit (n-1)² parameters and clamped to [0,1]
after each optimizer step. Simpler gradient flow, no saturation.
"""

from __future__ import annotations
from typing import Optional

from functools import partial

import torch
from torch import nn, cat
from torch.nn import Module

from einops import rearrange, repeat

from .utils import exists, default, add, get_expand_reduce_stream_functions
from .base import BaseHyperConnections


# ============================================================================
# CORE: LINEAR SEQUENTIAL BIRKHOFF
# ============================================================================

def linear_sequential_birkhoff(
    t: torch.Tensor,
    *,
    delta_logit: Optional[torch.Tensor] = None,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Map t ∈ [0,1]^(n-1)×(n-1) to an exactly doubly stochastic matrix.
    X[i,j] = L_ij + (U_ij - L_ij) * t[i,j]  (linear, no sigmoid)

    Args:
        t: (..., n-1, n-1) parameters in [0, 1] (clamp in forward for safety)
        delta_logit: optional scalar for uniform minorization
        eps: small safety clamp for numerical stability

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

    # Clamp t to [0,1] for valid fractions (safety; main clamp is after optimizer step)
    tt = tt.clamp(0.0, 1.0)

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

            # Linear: X[i,j] = L + (U-L) * t[i,j]
            x = L + width * tt[:, i, j]
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
    num_fracs=1,
    dim=None,
    add_stream_embed=False,
    disable=None,
    **kwargs
):
    """Get initializer for LTBP-mHC layers plus expand/reduce functions."""
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    init_hyper_conn_fn = partial(LTBP_MHC, num_streams, num_fracs=num_fracs, **kwargs)
    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams, add_stream_embed=add_stream_embed, dim=dim, disable=disable
    )

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim=dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


# ============================================================================
# LTBP-mHC IMPLEMENTATION
# ============================================================================

class LTBP_MHC(BaseHyperConnections):
    """
    LTBP-mHC: Linear Transport Birkhoff Polytope.

    Uses explicit (n-1)² parameters t ∈ [0,1] with linear formula
    X[i,j] = L_ij + (U_ij - L_ij) * t[i,j]. Call clamp_birkhoff_params()
    after each optimizer.step() to project t back to [0,1].
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
        depth_residual_fn=add,
        num_fracs: int = 1,
        make_dse: bool = True,
    ):
        assert num_residual_streams >= 2, "LTBP-mHC requires at least 2 streams"
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

    def clamp_birkhoff_params(self):
        """Project t to [0,1] after optimizer step. Call this after optimizer.step()."""
        self.birkhoff_params.data.clamp_(0.0, 1.0)

    def _init_hyper_params(self):
        """Initialize LTBP-mHC parameters: H^pre, explicit birkhoff_params, H^post."""
        n = self.num_residual_streams
        f = self.num_fracs
        v = self.num_input_views
        d = self.dim_per_frac

        # H^pre
        init_alpha_pre = torch.ones((n * f, v * f)) * -1
        init_alpha_pre[self.init_residual_index, :] = 1.
        init_alpha_res = torch.zeros((n * f, n * f))
        self.static_alpha = nn.Parameter(cat((init_alpha_pre, init_alpha_res), dim=1))

        # Explicit (n-1)² parameters for H^res, init to 0.5 (in [0,1])
        self.birkhoff_params = nn.Parameter(torch.ones(n - 1, n - 1) * 0.5)

        # Dynamic weights for H^pre only (H^res uses birkhoff_params)
        self.dynamic_alpha_fn = nn.Parameter(torch.zeros(d * n, f * (n * v)))
        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)

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
        """Compute H^pre, H^res (linear Birkhoff from explicit params), and H^post."""
        n = self.num_residual_streams
        f = self.num_fracs
        v = self.num_input_views

        # H^pre: dynamic + static
        wc_weight = normed @ self.dynamic_alpha_fn
        wc_weight = rearrange(wc_weight, '... (s t) -> ... s t', s=n)
        pre_scale = repeat(self.pre_branch_scale, '1 -> v', v=v * f)
        dynamic_pre = wc_weight[..., :v * f] * pre_scale
        static_pre = rearrange(self.static_alpha[:, :v * f], '(f s) v -> f s v', s=n)
        alpha_pre = (rearrange(dynamic_pre, '... n (f v) -> ... f n v', f=f) + static_pre).sigmoid()
        alpha_pre = alpha_pre.unsqueeze(-2)  # (..., f, n, 1, v)

        # H^res: linear Birkhoff from explicit params (same matrix for all positions)
        ds = linear_sequential_birkhoff(
            self.birkhoff_params.unsqueeze(0),
            delta_logit=self.delta_logit if self.make_dse else None
        )
        ds = ds.squeeze(0)  # (n, n)
        batch_shape = normed.shape[:-1]
        # alpha_res needs (..., f1, s, f2, t) = (..., 1, n, 1, n)
        alpha_res = ds.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-2)  # (1, 1, 1, n, 1, n)
        alpha_res = alpha_res.expand(*batch_shape, 1, n, 1, n)

        alpha = cat((alpha_pre, alpha_res), dim=-1)
        while alpha.ndim > 6 and alpha.shape[-5] == 1:
            alpha = alpha.squeeze(-5)

        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = normed @ self.dynamic_beta_fn
            dc_weight = rearrange(dc_weight, '... (s f) -> ... s f', s=n)
            dynamic_beta = dc_weight * self.h_post_scale
            static_beta = rearrange(self.static_beta, '(s f) -> s f', s=n)
            beta = (dynamic_beta + static_beta).sigmoid() * 2

        return alpha, beta


# Static method attachments
LTBP_MHC.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
LTBP_MHC.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)
