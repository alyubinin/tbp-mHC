"""
MSRTDP2N-mHC: Margined Scaled Power-of-2 Recursive Transportation Birkhoff Polytope

This module is the margined-scaled analogue of SRTBP2N-mHC. It keeps the same
power-of-2 recursive transportation-polytope construction, but every free
interval choice

    x in [L, U]

is parameterized with the margined scaled transport formula

    x = L + (U - L) * (rho + (1 - 2*rho) * sigmoid(beta * t / (U - L + epsilon)))

so the interpolation fraction stays in [rho, 1-rho] instead of [0, 1].
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


MSRTDP2N_BETA = 4.0
MSRTDP2N_EPSILON = 1e-3
MSRTDP2N_RHO = 1e-4


def _is_power_of_two(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _choose_in_interval(
    logits: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    eps: float = 1e-7,
    beta: float = MSRTDP2N_BETA,
    epsilon: float = MSRTDP2N_EPSILON,
    rho: float = MSRTDP2N_RHO,
) -> torch.Tensor:
    """Map unconstrained logits into [lower, upper] with the MSTBP-style margin."""
    interval_width = (upper - lower).clamp_min(0.0)
    scale_width = interval_width.clamp_min(eps)
    scaled_arg = beta * logits / (scale_width + epsilon)
    frac = rho + (1.0 - 2.0 * rho) * torch.sigmoid(scaled_arg)
    return lower + interval_width * frac


def _binary_split_vector(
    logits: torch.Tensor,
    total: torch.Tensor,
    capacities: torch.Tensor,
    *,
    eps: float = 1e-7,
    beta: float = MSRTDP2N_BETA,
    epsilon: float = MSRTDP2N_EPSILON,
    rho: float = MSRTDP2N_RHO,
) -> torch.Tensor:
    """
    Balanced binary split for vectors of power-of-2 length using the margined
    scaled RTBP2N interval map.
    """
    batch, k = capacities.shape

    if not _is_power_of_two(k):
        raise ValueError(f"Binary split requires power-of-2 length, got {k}")

    if k == 1:
        if logits.shape != (batch, 0):
            raise ValueError(f"Expected logits shape {(batch, 0)}, got {tuple(logits.shape)}")
        return total.unsqueeze(-1)

    if logits.shape != (batch, k - 1):
        raise ValueError(f"Expected logits shape {(batch, k - 1)}, got {tuple(logits.shape)}")

    half = k // 2
    left_caps = capacities[:, :half]
    right_caps = capacities[:, half:]

    left_capacity = left_caps.sum(dim=-1)
    right_capacity = right_caps.sum(dim=-1)

    lower = (total - right_capacity).clamp_min(0.0)
    upper = torch.minimum(total, left_capacity)
    left_total = _choose_in_interval(
        logits[:, 0],
        lower,
        upper,
        eps=eps,
        beta=beta,
        epsilon=epsilon,
        rho=rho,
    )
    right_total = total - left_total

    left_param_count = half - 1
    left_logits = logits[:, 1 : 1 + left_param_count]
    right_logits = logits[:, 1 + left_param_count :]

    left = _binary_split_vector(
        left_logits,
        left_total,
        left_caps,
        eps=eps,
        beta=beta,
        epsilon=epsilon,
        rho=rho,
    )
    right = _binary_split_vector(
        right_logits,
        right_total,
        right_caps,
        eps=eps,
        beta=beta,
        epsilon=epsilon,
        rho=rho,
    )
    return torch.cat([left, right], dim=-1)


def _recursive_transport_tbp_power2(
    theta: torch.Tensor,
    r: torch.Tensor,
    c: torch.Tensor,
    n: int,
    *,
    eps: float = 1e-7,
    beta: float = MSRTDP2N_BETA,
    epsilon: float = MSRTDP2N_EPSILON,
    rho: float = MSRTDP2N_RHO,
) -> torch.Tensor:
    """Internal power-of-2 recursive TBP constructor for the margined scaled chart."""
    if not _is_power_of_two(n):
        raise ValueError(f"Power-of-2 MSRTDP2N requires n=2^L, got {n}")

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
        x11 = _choose_in_interval(
            x11_logit,
            lower,
            upper,
            eps=eps,
            beta=beta,
            epsilon=epsilon,
            rho=rho,
        )
        x12 = r1 - x11
        x21 = c1 - x11
        x22 = r2 - c1 + x11

        row1 = torch.stack([x11, x12], dim=-1)
        row2 = torch.stack([x21, x22], dim=-1)
        return torch.stack([row1, row2], dim=-2)

    half = n // 2
    split_param_count = half - 1
    child_param_count = (half - 1) * (half - 1)

    idx = 0
    r_top = r[:, :half]
    r_bottom = r[:, half:]
    c_left = c[:, :half]
    c_right = c[:, half:]

    R1 = r_top.sum(dim=-1)
    R2 = r_bottom.sum(dim=-1)
    C1 = c_left.sum(dim=-1)
    C2 = c_right.sum(dim=-1)

    lower = torch.maximum(torch.maximum(torch.zeros_like(R1), R1 - C2), C1 - R2)
    upper = torch.minimum(R1, C1)
    m11 = _choose_in_interval(
        theta[:, idx],
        lower,
        upper,
        eps=eps,
        beta=beta,
        epsilon=epsilon,
        rho=rho,
    )
    idx += 1

    m12 = R1 - m11
    m21 = C1 - m11

    a = _binary_split_vector(
        theta[:, idx : idx + split_param_count],
        m11,
        r_top,
        eps=eps,
        beta=beta,
        epsilon=epsilon,
        rho=rho,
    )
    idx += split_param_count
    b = r_top - a

    gamma = _binary_split_vector(
        theta[:, idx : idx + split_param_count],
        m21,
        r_bottom,
        eps=eps,
        beta=beta,
        epsilon=epsilon,
        rho=rho,
    )
    idx += split_param_count
    d = r_bottom - gamma

    u = _binary_split_vector(
        theta[:, idx : idx + split_param_count],
        m11,
        c_left,
        eps=eps,
        beta=beta,
        epsilon=epsilon,
        rho=rho,
    )
    idx += split_param_count
    v = c_left - u

    s = _binary_split_vector(
        theta[:, idx : idx + split_param_count],
        m12,
        c_right,
        eps=eps,
        beta=beta,
        epsilon=epsilon,
        rho=rho,
    )
    idx += split_param_count
    t = c_right - s

    theta_a = theta[:, idx : idx + child_param_count]
    idx += child_param_count
    theta_b = theta[:, idx : idx + child_param_count]
    idx += child_param_count
    theta_c = theta[:, idx : idx + child_param_count]
    idx += child_param_count
    theta_d = theta[:, idx : idx + child_param_count]
    idx += child_param_count

    if idx != expected:
        raise RuntimeError(f"MSRTDP2N parameter bookkeeping mismatch for n={n}")

    stacked_theta = torch.cat([theta_a, theta_b, theta_c, theta_d], dim=0)
    stacked_r = torch.cat([a, b, gamma, d], dim=0)
    stacked_c = torch.cat([u, s, v, t], dim=0)
    stacked_x = _recursive_transport_tbp_power2(
        stacked_theta,
        stacked_r,
        stacked_c,
        half,
        eps=eps,
        beta=beta,
        epsilon=epsilon,
        rho=rho,
    )

    A, B, C, D = stacked_x.split(batch, dim=0)
    top = torch.cat([A, B], dim=-1)
    bottom = torch.cat([C, D], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def margined_scaled_recursive_transport_birkhoff_power2(
    t: torch.Tensor,
    *,
    delta_logit: Optional[torch.Tensor] = None,
    eps: float = 1e-7,
    beta: float = MSRTDP2N_BETA,
    epsilon: float = MSRTDP2N_EPSILON,
    rho: float = MSRTDP2N_RHO,
) -> torch.Tensor:
    """
    Map free parameters t to an exactly doubly stochastic matrix for n = 2^L
    using the margined scaled power-of-2 recursive transportation-polytope chart.
    """
    if t.ndim < 2:
        raise ValueError(f"t must have shape (..., n-1, n-1), got {tuple(t.shape)}")

    n_minus_1_row = t.shape[-2]
    n_minus_1_col = t.shape[-1]
    if n_minus_1_row != n_minus_1_col:
        raise ValueError(f"t must be square in last two dims, got {tuple(t.shape)}")

    n = n_minus_1_row + 1
    if not _is_power_of_two(n):
        raise ValueError(f"Power-of-2 MSRTDP2N requires n=2^L, got {n}")

    batch_shape = t.shape[:-2]
    batch = math.prod(batch_shape) if batch_shape else 1

    theta = t.reshape(batch, (n - 1) * (n - 1))
    device = theta.device
    dtype = theta.dtype

    r = torch.ones((batch, n), device=device, dtype=dtype)
    c = torch.ones((batch, n), device=device, dtype=dtype)

    X = _recursive_transport_tbp_power2(
        theta,
        r,
        c,
        n,
        eps=eps,
        beta=beta,
        epsilon=epsilon,
        rho=rho,
    )
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
    """Get initializer for MSRTDP2N-mHC layers plus expand/reduce functions."""
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = MSRTDP2N_MHC if not disable else Residual

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


class MSRTDP2N_MHC(BaseHyperConnections):
    """
    MSRTDP2N-mHC: margined scaled power-of-2 recursive transport Birkhoff parameterization.

    Same recursive power-of-2 chart as RTBP2N-mHC, but every interval choice uses:

        x = L + (U-L) * (rho + (1 - 2*rho) * sigmoid(beta * t / (U-L + epsilon)))
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
        assert num_residual_streams >= 2, "MSRTDP2N-mHC requires at least 2 streams"
        assert _is_power_of_two(num_residual_streams), (
            f"MSRTDP2N-mHC requires a power-of-2 stream count, got {num_residual_streams}"
        )
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
        """Initialize MSRTDP2N-mHC parameters: H^pre, H^res, H^post, delta."""
        n = self.num_residual_streams
        f = self.num_fracs
        v = self.num_input_views
        d = self.dim_per_frac

        init_alpha_pre = torch.ones((n * f, v * f)) * -1
        init_alpha_pre[self.init_residual_index, :] = 1.0

        init_alpha_res = torch.zeros((n * f, n * f))
        self.static_alpha = nn.Parameter(cat((init_alpha_pre, init_alpha_res), dim=1))

        self.dynamic_alpha_fn = nn.Parameter(torch.zeros(d * n, f * (n * n + n * v)))

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
        """Compute H^pre, H^res (margined scaled power-of-2 RTBP), and H^post matrices."""
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
        msrtdp2n_params = alpha_res[..., : n - 1, : n - 1]

        orig_shape = msrtdp2n_params.shape[:-2]
        msrtdp2n_flat = msrtdp2n_params.reshape(-1, n - 1, n - 1)

        ds_flat = margined_scaled_recursive_transport_birkhoff_power2(
            msrtdp2n_flat,
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


MSRTDP2N_MHC.get_expand_reduce_stream_functions = staticmethod(
    get_init_and_expand_reduce_stream_functions
)
MSRTDP2N_MHC.get_init_and_expand_reduce_stream_functions = staticmethod(
    get_init_and_expand_reduce_stream_functions
)


class MarginedScaledPowerOfTwoRecursiveTransportBirkhoff(nn.Module):
    """Standalone wrapper for margined_scaled_recursive_transport_birkhoff_power2()."""

    def __init__(
        self,
        num_streams: int,
        make_dse: bool = True,
        eps: float = 1e-7,
        beta: float = MSRTDP2N_BETA,
        epsilon: float = MSRTDP2N_EPSILON,
        rho: float = MSRTDP2N_RHO,
    ):
        super().__init__()
        if num_streams < 2:
            raise ValueError("num_streams must be >= 2")
        if not _is_power_of_two(num_streams):
            raise ValueError(f"num_streams must be a power of 2, got {num_streams}")
        self.num_streams = num_streams
        self.eps = eps
        self.beta = beta
        self.epsilon = epsilon
        self.rho = rho
        self.make_dse = make_dse
        if make_dse:
            self.delta_logit = nn.Parameter(torch.tensor(-8.0))
        else:
            self.delta_logit = None

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        n = self.num_streams
        if t.shape[-2:] != (n - 1, n - 1):
            raise ValueError(f"Expected t shape (..., {n-1}, {n-1}), got {tuple(t.shape)}")
        return margined_scaled_recursive_transport_birkhoff_power2(
            t,
            delta_logit=self.delta_logit if self.make_dse else None,
            eps=self.eps,
            beta=self.beta,
            epsilon=self.epsilon,
            rho=self.rho,
        )


MSRTDP2NHC = MSRTDP2N_MHC
