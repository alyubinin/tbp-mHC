"""
RTBP-mHC: Recursive Transportation Birkhoff Polytope Manifold-Constrained Hyper-Connections

This module implements manifold-constrained hyper-connections using the complete
recursive transportation-polytope parameterization described by the user. It
constructs EXACTLY doubly stochastic matrices by recursively splitting a unit
transport plan into smaller transport polytopes.

Compared with the sequential Birkhoff charts already in this project:
- LSB/STBP/MSTBP fill entries in a fixed sequential order
- RTBP recursively partitions the problem into balanced subproblems
- For even n, the four same-size child problems are evaluated in one stacked
  batched call to improve GPU utilization

The public `recursive_transport_birkhoff()` function maps unconstrained
parameters of shape (..., n-1, n-1) to an exactly doubly stochastic matrix of
shape (..., n, n).
"""

from __future__ import annotations

import math
from functools import partial
from typing import Optional

import torch
from torch import nn, cat
from torch.nn import Module

from einops import rearrange, repeat

from .utils import (
    exists,
    default,
    add,
    Residual,
    get_expand_reduce_stream_functions,
)
from .base import BaseHyperConnections


def _choose_in_interval(
    logits: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Map unconstrained logits into a closed interval [lower, upper]."""
    width = (upper - lower).clamp_min(0.0)
    return lower + width * torch.sigmoid(logits)


def _split_vector(
    logits: torch.Tensor,
    total: torch.Tensor,
    capacities: torch.Tensor,
) -> torch.Tensor:
    """
    Batched differentiable version of SplitVector(m, w).

    Args:
        logits: (B, p-1) free parameters
        total: (B,) amount to distribute
        capacities: (B, p) nonnegative capacities

    Returns:
        z: (B, p) with z >= 0, z <= capacities, sum(z) = total
    """
    batch, p = capacities.shape

    if p == 1:
        return total.unsqueeze(-1)

    if logits.shape != (batch, p - 1):
        raise ValueError(
            f"Expected logits shape {(batch, p - 1)}, got {tuple(logits.shape)}"
        )

    parts = []
    assigned = torch.zeros_like(total)

    for i in range(p - 1):
        remaining_capacity = capacities[:, i + 1 :].sum(dim=-1)
        lower = (total - remaining_capacity - assigned).clamp_min(0.0)
        upper = torch.minimum(capacities[:, i], total - assigned)
        zi = _choose_in_interval(logits[:, i], lower, upper)
        parts.append(zi)
        assigned = assigned + zi

    parts.append(total - assigned)
    return torch.stack(parts, dim=-1)


def _recursive_transport_tbp(
    theta: torch.Tensor,
    r: torch.Tensor,
    c: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """
    Internal batched recursive TBP constructor.

    Args:
        theta: (B, (n-1)^2) free parameters for the current problem
        r: (B, n) row margins
        c: (B, n) column margins
        n: matrix size
    """
    batch = theta.shape[0]
    expected = (n - 1) * (n - 1)
    if theta.shape[-1] != expected:
        raise ValueError(
            f"Expected {(n - 1)}^2={expected} parameters for n={n}, got {theta.shape[-1]}"
        )

    if n == 1:
        return r.unsqueeze(-1)

    if n == 2:
        x11_logit = theta[:, 0]
        r1, r2 = r[:, 0], r[:, 1]
        c1, c2 = c[:, 0], c[:, 1]

        lower = torch.maximum(torch.maximum(torch.zeros_like(r1), r1 - c2), c1 - r2)
        upper = torch.minimum(r1, c1)
        x11 = _choose_in_interval(x11_logit, lower, upper)
        x12 = r1 - x11
        x21 = c1 - x11
        x22 = r2 - c1 + x11

        row1 = torch.stack([x11, x12], dim=-1)
        row2 = torch.stack([x21, x22], dim=-1)
        return torch.stack([row1, row2], dim=-2)

    if n % 2 == 0:
        k = n // 2
        child_params = (k - 1) * (k - 1)
        idx = 0

        r_top = r[:, :k]
        r_bottom = r[:, k:]
        c_left = c[:, :k]
        c_right = c[:, k:]

        R1 = r_top.sum(dim=-1)
        R2 = r_bottom.sum(dim=-1)
        C1 = c_left.sum(dim=-1)
        C2 = c_right.sum(dim=-1)

        lower = torch.maximum(torch.maximum(torch.zeros_like(R1), R1 - C2), C1 - R2)
        upper = torch.minimum(R1, C1)
        m11 = _choose_in_interval(theta[:, idx], lower, upper)
        idx += 1

        m12 = R1 - m11
        m21 = C1 - m11

        a = _split_vector(theta[:, idx : idx + k - 1], m11, r_top)
        idx += k - 1
        b = r_top - a

        gamma = _split_vector(theta[:, idx : idx + k - 1], m21, r_bottom)
        idx += k - 1
        d = r_bottom - gamma

        u = _split_vector(theta[:, idx : idx + k - 1], m11, c_left)
        idx += k - 1
        v = c_left - u

        s = _split_vector(theta[:, idx : idx + k - 1], m12, c_right)
        idx += k - 1
        t = c_right - s

        theta_a = theta[:, idx : idx + child_params]
        idx += child_params
        theta_b = theta[:, idx : idx + child_params]
        idx += child_params
        theta_c = theta[:, idx : idx + child_params]
        idx += child_params
        theta_d = theta[:, idx : idx + child_params]
        idx += child_params

        if idx != expected:
            raise RuntimeError(f"RTBP parameter bookkeeping mismatch for n={n}")

        stacked_theta = torch.cat([theta_a, theta_b, theta_c, theta_d], dim=0)
        stacked_r = torch.cat([a, b, gamma, d], dim=0)
        stacked_c = torch.cat([u, s, v, t], dim=0)
        stacked_x = _recursive_transport_tbp(stacked_theta, stacked_r, stacked_c, k)

        A, B, C, D = stacked_x.split(batch, dim=0)
        top = torch.cat([A, B], dim=-1)
        bottom = torch.cat([C, D], dim=-1)
        return torch.cat([top, bottom], dim=-2)

    idx = 0
    last = n - 1

    row_prefix = r[:, :last]
    col_prefix = c[:, :last]
    rn = r[:, last]
    cn = c[:, last]

    Sr = row_prefix.sum(dim=-1)
    Sc = col_prefix.sum(dim=-1)

    lower = torch.maximum(torch.maximum(torch.zeros_like(rn), rn - Sc), cn - Sr)
    upper = torch.minimum(rn, cn)
    xnn = _choose_in_interval(theta[:, idx], lower, upper)
    idx += 1

    row_rest = rn - xnn
    col_rest = cn - xnn

    q = _split_vector(theta[:, idx : idx + last - 1], col_rest, row_prefix)
    idx += last - 1
    p = _split_vector(theta[:, idx : idx + last - 1], row_rest, col_prefix)
    idx += last - 1

    child_theta = theta[:, idx:]
    child = _recursive_transport_tbp(child_theta, row_prefix - q, col_prefix - p, last)

    last_col = q.unsqueeze(-1)
    top = torch.cat([child, last_col], dim=-1)

    last_row = torch.cat([p, xnn.unsqueeze(-1)], dim=-1).unsqueeze(-2)
    return torch.cat([top, last_row], dim=-2)


def recursive_transport_birkhoff(
    t: torch.Tensor,
    *,
    delta_logit: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Map free parameters t to an exactly doubly stochastic matrix using the
    recursive transportation-polytope parameterization.

    Args:
        t: (..., n-1, n-1) unconstrained parameters
        delta_logit: optional scalar/broadcastable logit for uniform minorization

    Returns:
        X: (..., n, n) doubly stochastic matrix
    """
    if t.ndim < 2:
        raise ValueError(f"t must have shape (..., n-1, n-1), got {tuple(t.shape)}")

    n_minus_1_row = t.shape[-2]
    n_minus_1_col = t.shape[-1]
    if n_minus_1_row != n_minus_1_col:
        raise ValueError(f"t must be square in last two dims, got {tuple(t.shape)}")

    n = n_minus_1_row + 1
    batch_shape = t.shape[:-2]
    batch = math.prod(batch_shape) if batch_shape else 1

    theta = t.reshape(batch, (n - 1) * (n - 1))
    device = theta.device
    dtype = theta.dtype

    r = torch.ones((batch, n), device=device, dtype=dtype)
    c = torch.ones((batch, n), device=device, dtype=dtype)

    X = _recursive_transport_tbp(theta, r, c, n)
    X = X.reshape(*batch_shape, n, n)

    if exists(delta_logit):
        delta = torch.sigmoid(delta_logit).to(dtype=dtype, device=device)
        while delta.ndim < X.ndim:
            delta = delta.unsqueeze(0)
        X = (1 - delta) * X + delta * (1.0 / n)

    return X


def get_init_and_expand_reduce_stream_functions(
    num_streams,
    num_fracs=1,
    dim=None,
    add_stream_embed=False,
    disable=None,
    **kwargs,
):
    """Get initializer for RTBP-mHC layers plus expand/reduce functions."""
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = RTBP_MHC if not disable else Residual

    init_hyper_conn_fn = partial(
        hyper_conn_klass,
        num_streams,
        num_fracs=num_fracs,
        **kwargs,
    )
    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams,
        add_stream_embed=add_stream_embed,
        dim=dim,
        disable=disable,
    )

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim=dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


class RTBP_MHC(BaseHyperConnections):
    """
    RTBP-mHC: Recursive Transportation Birkhoff Polytope.

    This variant uses the complete recursive transport-polytope construction
    instead of the fixed-entry sequential chart. It still uses only (n-1)^2
    effective free parameters, but allocates them according to the recursive
    split tree rather than by scanning the matrix.
    """

    def __init__(
        self,
        num_residual_streams: int,
        *,
        dim: int,
        branch: Module | None = None,
        layer_index: int | None = None,
        channel_first: bool = False,
        dropout: float = 0.0,
        residual_transform: Module | None = None,
        add_branch_out_to_residual: bool = True,
        num_input_views: int = 1,
        depth_residual_fn=add,
        num_fracs: int = 1,
        make_dse: bool = True,
    ):
        assert num_residual_streams >= 2, "RTBP-mHC requires at least 2 streams"
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
        """Initialize RTBP-mHC parameters: H^pre, H^res, H^post, delta."""
        n = self.num_residual_streams
        f = self.num_fracs
        v = self.num_input_views
        d = self.dim_per_frac

        init_alpha_pre = torch.ones((n * f, v * f)) * -1
        init_alpha_pre[self.init_residual_index, :] = 1.0

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
            beta_init = torch.ones(n * f) * -1.0
            beta_init[self.init_residual_index] = 1.0
            self.static_beta = nn.Parameter(beta_init)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(d * n, f * n))
            self.h_post_scale = nn.Parameter(torch.ones(()) * 1e-2)

    def _compute_alpha_beta(self, normed: torch.Tensor, device: torch.device):
        """Compute H^pre, H^res (recursive TBP), and H^post matrices."""
        n = self.num_residual_streams
        f = self.num_fracs
        v = self.num_input_views

        wc_weight = normed @ self.dynamic_alpha_fn
        wc_weight = rearrange(wc_weight, "... (s t) -> ... s t", s=n)

        pre_scale = repeat(self.pre_branch_scale, "1 -> v", v=v * f)
        res_scale = repeat(self.residual_scale, "1 -> s", s=f * n)
        alpha_scale = cat((pre_scale, res_scale))

        dynamic_alpha = wc_weight * alpha_scale
        static_alpha = rearrange(self.static_alpha, "(f s) t -> f s t", s=n)
        alpha = dynamic_alpha + static_alpha
        alpha = self.split_fracs(alpha)

        alpha_pre, alpha_res = alpha[..., :v], alpha[..., v:]
        alpha_pre = alpha_pre.sigmoid()

        alpha_res = rearrange(alpha_res, "... f s g t -> ... f g s t")
        rtbp_params = alpha_res[..., : n - 1, : n - 1]

        orig_shape = rtbp_params.shape[:-2]
        rtbp_flat = rtbp_params.reshape(-1, n - 1, n - 1)

        ds_flat = recursive_transport_birkhoff(
            rtbp_flat,
            delta_logit=self.delta_logit if self.make_dse else None,
        )

        alpha_res = ds_flat.reshape(*orig_shape, n, n)
        alpha_res = rearrange(alpha_res, "... f g s t -> ... f s g t")
        alpha = cat((alpha_pre, alpha_res), dim=-1)

        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = normed @ self.dynamic_beta_fn
            dc_weight = rearrange(dc_weight, "... (s f) -> ... s f", s=n)
            dynamic_beta = dc_weight * self.h_post_scale
            static_beta = rearrange(self.static_beta, "(s f) -> s f", s=n)
            beta = (dynamic_beta + static_beta).sigmoid() * 2

        return alpha, beta


RTBP_MHC.get_expand_reduce_stream_functions = staticmethod(
    get_expand_reduce_stream_functions
)
RTBP_MHC.get_init_and_expand_reduce_stream_functions = staticmethod(
    get_init_and_expand_reduce_stream_functions
)


class RecursiveTransportBirkhoff(nn.Module):
    """Standalone wrapper for recursive_transport_birkhoff()."""

    def __init__(self, num_streams: int, make_dse: bool = True):
        super().__init__()
        if num_streams < 2:
            raise ValueError("num_streams must be >= 2")
        self.num_streams = num_streams
        self.make_dse = make_dse
        if make_dse:
            self.delta_logit = nn.Parameter(torch.tensor(-8.0))
        else:
            self.delta_logit = None

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        n = self.num_streams
        if t.shape[-2:] != (n - 1, n - 1):
            raise ValueError(f"Expected t shape (..., {n-1}, {n-1}), got {tuple(t.shape)}")
        return recursive_transport_birkhoff(
            t,
            delta_logit=self.delta_logit if self.make_dse else None,
        )


RTBPHC = RTBP_MHC
