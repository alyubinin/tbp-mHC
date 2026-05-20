"""
RTBP2N-mHC: Power-of-2 Recursive Transportation Birkhoff Polytope

This module specializes the recursive transportation-polytope parameterization
to the case n = 2^L. In this regime, the odd-size branch disappears entirely
and the vector splitting subroutine can be replaced by a balanced binary split
tree instead of the sequential SplitVector routine.

The resulting parameterization still uses exactly (n-1)^2 free parameters, but
all split decisions follow the power-of-2 recursion tree.

===============================================================================
HIGH-LEVEL IDEA
===============================================================================

We want to map unconstrained parameters t ∈ R^{(n-1)×(n-1)} to a matrix
X ∈ R^{n×n} that is:

    1. entrywise nonnegative
    2. row-stochastic:  X 1 = 1
    3. column-stochastic: X^T 1 = 1

so X lies in the Birkhoff polytope.

For general n, the recursive transport construction has two cases:

    - even n: split the problem into four n/2 × n/2 child transport problems
    - odd  n: peel off the final row/column and recurse on n-1

This file handles only the even branch recursively, which is possible exactly
when n is a power of 2. That removes the odd case entirely and gives a regular
quadtree decomposition:

    n × n
      -> four (n/2) × (n/2) blocks
      -> sixteen (n/4) × (n/4) blocks
      -> ...
      -> 2 × 2 leaves

===============================================================================
LOCAL FORMULAS
===============================================================================

At each node of size n, split the row margins and column margins into halves:

    r = [r_top, r_bottom]
    c = [c_left, c_right]

Define the block totals:

    R1 = sum(r_top)      R2 = sum(r_bottom)
    C1 = sum(c_left)     C2 = sum(c_right)

The upper-left block mass m11 must satisfy the standard 2×2 transport bounds:

    L_M = max(0, R1 - C2, C1 - R2)
    U_M = min(R1, C1)

We parameterize it by a logit ξ:

    m11 = L_M + (U_M - L_M) * sigmoid(ξ)

and then the remaining block masses are forced:

    m12 = R1 - m11
    m21 = C1 - m11
    m22 = R2 - C1 + m11

The vectors a, gamma, u, s distribute these block totals back to individual
rows/columns:

    a      sums to m11 over the top-half rows
    gamma  sums to m21 over the bottom-half rows
    u      sums to m11 over the left-half columns
    s      sums to m12 over the right-half columns

The remaining row/column capacities become:

    b = r_top    - a
    d = r_bottom - gamma
    v = c_left   - u
    t = c_right  - s

The four child transport problems are then:

    A ~ TBP(a,      u)
    B ~ TBP(b,      s)
    C ~ TBP(gamma,  v)
    D ~ TBP(d,      t)

and the parent matrix is assembled as:

    X = [ A  B ]
        [ C  D ]

===============================================================================
WHY THE PARAMETER COUNT IS STILL (n-1)^2
===============================================================================

For a node of size n = 2m, this implementation uses:

    1         parameter for m11
    (m-1)     parameters for binary-splitting a
    (m-1)     parameters for binary-splitting gamma
    (m-1)     parameters for binary-splitting u
    (m-1)     parameters for binary-splitting s
    4(m-1)^2  parameters for the four child nodes

Total:

    1 + 4(m-1) + 4(m-1)^2
    = 1 + 4m - 4 + 4m^2 - 8m + 4
    = 4m^2 - 4m + 1
    = (2m - 1)^2
    = (n - 1)^2

So replacing the sequential SplitVector routine with a balanced binary split
tree changes the structure of the chart, but not the number of degrees of
freedom.
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


def _is_power_of_two(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _choose_in_interval(
    logits: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """
    Map unconstrained logits into a closed interval [lower, upper].

    Formula:

        x = lower + (upper - lower) * sigmoid(logit)

    This is the basic "choose any feasible value in [L, U]" primitive used
    throughout the transport parameterization. When lower == upper, the interval
    collapses and x is forced to that value.
    """
    width = (upper - lower).clamp_min(0.0)
    return lower + width * torch.sigmoid(logits)


def _binary_split_vector(
    logits: torch.Tensor,
    total: torch.Tensor,
    capacities: torch.Tensor,
) -> torch.Tensor:
    """
    Balanced binary split for vectors of power-of-2 length.

    Args:
        logits: (B, k-1) free parameters where k is a power of 2
        total: (B,) amount to distribute
        capacities: (B, k) nonnegative capacities

    Returns:
        z: (B, k) with z >= 0, z <= capacities, sum(z) = total

    This is the power-of-2 replacement for the generic left-to-right
    SplitVector routine.

    If capacities has length k = 2^L, we recursively split it into two halves:

        capacities = [w_left, w_right]

    Let:

        W_left  = sum(w_left)
        W_right = sum(w_right)

    To distribute total mass m over the whole vector, the left-half mass m_left
    must lie in:

        m_left ∈ [ max(0, m - W_right), min(m, W_left) ]

    because:
        - the right half can absorb at most W_right
        - the left half can absorb at most W_left

    Then:

        m_right = m - m_left

    and we recurse on each half independently.

    A binary split tree with k leaves has exactly k-1 internal nodes, so this
    routine uses exactly k-1 free scalar choices, matching the generic
    SplitVector parameter count.
    """
    batch, k = capacities.shape

    if not _is_power_of_two(k):
        raise ValueError(f"Binary split requires power-of-2 length, got {k}")

    if k == 1:
        # Leaf of the binary split tree: there is only one feasible assignment.
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

    # Feasible interval for the total mass routed into the left half:
    #
    #   lower = max(0, total - right_capacity)
    #   upper = min(total, left_capacity)
    #
    # This is exactly the 2-bin transportation feasibility condition.
    lower = (total - right_capacity).clamp_min(0.0)
    upper = torch.minimum(total, left_capacity)
    left_total = _choose_in_interval(logits[:, 0], lower, upper)
    right_total = total - left_total

    # The remaining logits are split between the left and right subtrees.
    # A subtree with `half` leaves needs `half - 1` logits.
    left_param_count = half - 1
    left_logits = logits[:, 1 : 1 + left_param_count]
    right_logits = logits[:, 1 + left_param_count :]

    # Recurse independently on the two halves, then concatenate the assignments
    # back together to recover the original vector order.
    left = _binary_split_vector(left_logits, left_total, left_caps)
    right = _binary_split_vector(right_logits, right_total, right_caps)
    return torch.cat([left, right], dim=-1)


def _recursive_transport_tbp_power2(
    theta: torch.Tensor,
    r: torch.Tensor,
    c: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """
    Internal power-of-2 recursive TBP constructor.

    Args:
        theta: (B, (n-1)^2) free parameters for current node
        r: (B, n) row margins
        c: (B, n) col margins
        n: matrix size, must be a power of 2

    Invariant:

        r_i >= 0, c_j >= 0, and sum_i r_i = sum_j c_j

    The return value X satisfies:

        X >= 0
        X 1 = r
        X^T 1 = c

    For the top-level call in `recursive_transport_birkhoff_power2`, we set
    r = c = 1, so the result is doubly stochastic.
    """
    if not _is_power_of_two(n):
        raise ValueError(f"Power-of-2 RTBP requires n=2^L, got {n}")

    batch = theta.shape[0]
    expected = (n - 1) * (n - 1)
    if theta.shape[-1] != expected:
        raise ValueError(
            f"Expected {(n - 1)}^2={expected} parameters for n={n}, got {theta.shape[-1]}"
        )

    if n == 1:
        # Trivial transport polytope: the only feasible matrix is [r1] = [c1].
        return r.unsqueeze(-1)

    if n == 2:
        # Closed-form 2×2 transportation polytope.
        #
        # For margins (r1, r2) and (c1, c2), x11 is the only free variable, with:
        #
        #   x11 ∈ [ max(0, r1 - c2, c1 - r2), min(r1, c1) ]
        #
        # Once x11 is chosen, the other entries are forced by the margins:
        #
        #   x12 = r1 - x11
        #   x21 = c1 - x11
        #   x22 = r2 - c1 + x11
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

    # Recursive even case. Since n is a power of 2 and n > 2, n can be split
    # cleanly into two equal halves.
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

    # Choose the upper-left block total m11 as the free variable of the 2×2
    # transport problem on the aggregated block margins:
    #
    #       [m11  m12]
    #       [m21  m22]
    #
    # with row sums (R1, R2) and column sums (C1, C2).
    #
    # Feasible interval:
    #
    #   m11 ∈ [ max(0, R1 - C2, C1 - R2), min(R1, C1) ]
    lower = torch.maximum(torch.maximum(torch.zeros_like(R1), R1 - C2), C1 - R2)
    upper = torch.minimum(R1, C1)
    m11 = _choose_in_interval(theta[:, idx], lower, upper)
    idx += 1

    # The remaining block totals are forced by the block margins.
    m12 = R1 - m11
    m21 = C1 - m11

    # Distribute each block total back to individual rows/columns using the
    # balanced binary split chart:
    #
    #   a      : top rows -> left columns    (sum(a)     = m11)
    #   gamma  : bottom rows -> left columns (sum(gamma) = m21)
    #   u      : left cols receives from top rows    (sum(u) = m11)
    #   s      : right cols receives from top rows   (sum(s) = m12)
    #
    # The complementary row/column masses become:
    #
    #   b = r_top    - a
    #   d = r_bottom - gamma
    #   v = c_left   - u
    #   t = c_right  - s
    a = _binary_split_vector(theta[:, idx : idx + split_param_count], m11, r_top)
    idx += split_param_count
    b = r_top - a

    gamma = _binary_split_vector(theta[:, idx : idx + split_param_count], m21, r_bottom)
    idx += split_param_count
    d = r_bottom - gamma

    u = _binary_split_vector(theta[:, idx : idx + split_param_count], m11, c_left)
    idx += split_param_count
    v = c_left - u

    s = _binary_split_vector(theta[:, idx : idx + split_param_count], m12, c_right)
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
        raise RuntimeError(f"RTBP2N parameter bookkeeping mismatch for n={n}")

    # To reduce Python overhead and increase tensor work per call, recurse on the
    # four child transport problems as one larger batched problem:
    #
    #   A ~ TBP(a,     u)
    #   B ~ TBP(b,     s)
    #   C ~ TBP(gamma, v)
    #   D ~ TBP(d,     t)
    #
    # Rather than calling the child function four times, stack the four child
    # batches together and recurse once.
    stacked_theta = torch.cat([theta_a, theta_b, theta_c, theta_d], dim=0)
    stacked_r = torch.cat([a, b, gamma, d], dim=0)
    stacked_c = torch.cat([u, s, v, t], dim=0)
    stacked_x = _recursive_transport_tbp_power2(stacked_theta, stacked_r, stacked_c, half)

    # Reassemble the four child blocks into the parent matrix:
    #
    #   X = [ A  B ]
    #       [ C  D ]
    A, B, C, D = stacked_x.split(batch, dim=0)
    top = torch.cat([A, B], dim=-1)
    bottom = torch.cat([C, D], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def recursive_transport_birkhoff_power2(
    t: torch.Tensor,
    *,
    delta_logit: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Map free parameters t to an exactly doubly stochastic matrix for n = 2^L.

    Args:
        t: (..., n-1, n-1) unconstrained parameters with n a power of 2
        delta_logit: optional scalar/broadcastable logit for uniform minorization

    The input uses the same `(n-1) × (n-1)` parameter budget as the other exact
    Birkhoff charts in this repo, but this function interprets those parameters
    according to the recursive transport tree rather than a fixed row/column scan.

    The top-level margins are initialized to all ones:

        r = (1, ..., 1),    c = (1, ..., 1)

    so the resulting matrix X is doubly stochastic.

    Optional strict-positivity / minorization:

        H = (1 - δ) X + δ (1/n) 11^T,
        δ = sigmoid(delta_logit)

    This keeps the matrix doubly stochastic while moving it away from the
    boundary of the Birkhoff polytope.
    """
    if t.ndim < 2:
        raise ValueError(f"t must have shape (..., n-1, n-1), got {tuple(t.shape)}")

    n_minus_1_row = t.shape[-2]
    n_minus_1_col = t.shape[-1]
    if n_minus_1_row != n_minus_1_col:
        raise ValueError(f"t must be square in last two dims, got {tuple(t.shape)}")

    n = n_minus_1_row + 1
    if not _is_power_of_two(n):
        raise ValueError(f"Power-of-2 RTBP requires n=2^L, got {n}")

    batch_shape = t.shape[:-2]
    batch = math.prod(batch_shape) if batch_shape else 1

    # Flatten the chart parameters so each batch item owns a contiguous bank of
    # (n-1)^2 scalars. The recursive solver consumes this bank in tree order.
    theta = t.reshape(batch, (n - 1) * (n - 1))
    device = theta.device
    dtype = theta.dtype

    # Unit row/column margins produce a doubly stochastic output matrix.
    r = torch.ones((batch, n), device=device, dtype=dtype)
    c = torch.ones((batch, n), device=device, dtype=dtype)

    X = _recursive_transport_tbp_power2(theta, r, c, n)
    X = X.reshape(*batch_shape, n, n)

    if exists(delta_logit):
        delta = torch.sigmoid(delta_logit).to(dtype=dtype, device=device)
        while delta.ndim < X.ndim:
            delta = delta.unsqueeze(0)
        # Uniform minorization preserves doubly stochasticity because both X and
        # the uniform matrix J/n are doubly stochastic.
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
    """Get initializer for RTBP2N-mHC layers plus expand/reduce functions."""
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = RTBP2N_MHC if not disable else Residual

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


class RTBP2N_MHC(BaseHyperConnections):
    """
    RTBP2N-mHC: power-of-2 recursive transport Birkhoff parameterization.

    This variant is restricted to n = 2^L and replaces the generic sequential
    SplitVector routine with balanced binary split trees.

    As in the other exact-DS hyper-connection variants in this repo:

        - H^pre is produced by sigmoid-transformed dynamic/static logits
        - H^res is produced by an exact Birkhoff chart
        - H^post is produced by sigmoid-transformed dynamic/static logits

    Here the exact Birkhoff chart is `recursive_transport_birkhoff_power2()`.
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
        assert num_residual_streams >= 2, "RTBP2N-mHC requires at least 2 streams"
        assert _is_power_of_two(num_residual_streams), (
            f"RTBP2N-mHC requires a power-of-2 stream count, got {num_residual_streams}"
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
        """
        Initialize RTBP2N-mHC parameters: H^pre, H^res, H^post, delta.

        The parameter layout mirrors the other transport-style HC variants:

            static_alpha = [ H^pre biases | H^res logits ]
            dynamic_alpha_fn projects normalized residuals to the same shape

        The H^res part is stored as an n×n block for compatibility with the
        shared base-class plumbing, but only the top-left (n-1)×(n-1) slice is
        consumed by the recursive transport chart.
        """
        n = self.num_residual_streams
        f = self.num_fracs
        v = self.num_input_views
        d = self.dim_per_frac

        # H^pre starts near a single dominant residual stream, matching the
        # initialization pattern used by the other HC variants in this project.
        init_alpha_pre = torch.ones((n * f, v * f)) * -1
        init_alpha_pre[self.init_residual_index, :] = 1.0

        # H^res logits start at 0, which maps every local interval choice to its
        # midpoint because sigmoid(0) = 0.5.
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
        """
        Compute H^pre, H^res (power-of-2 RTBP), and H^post matrices.

        The shared HC pattern is:

            alpha = [H^pre | H^res]

        where `alpha` is later consumed by the shared einsum in
        `BaseHyperConnections._apply_mixing()`.

        In this variant:
            1. Build dynamic + static logits
            2. Apply sigmoid to H^pre
            3. Interpret the H^res logits as RTBP2N chart parameters
            4. Convert them to an exact doubly stochastic matrix with
               `recursive_transport_birkhoff_power2`
            5. Build H^post as in the other variants
        """
        n = self.num_residual_streams
        f = self.num_fracs
        v = self.num_input_views

        # Dynamic logits from the normalized residual state.
        wc_weight = normed @ self.dynamic_alpha_fn
        wc_weight = rearrange(wc_weight, "... (s t) -> ... s t", s=n)

        # Shared scalar gates controlling the magnitudes of the pre/residual
        # dynamic components.
        pre_scale = repeat(self.pre_branch_scale, "1 -> v", v=v * f)
        res_scale = repeat(self.residual_scale, "1 -> s", s=f * n)
        alpha_scale = cat((pre_scale, res_scale))

        dynamic_alpha = wc_weight * alpha_scale
        static_alpha = rearrange(self.static_alpha, "(f s) t -> f s t", s=n)
        alpha = dynamic_alpha + static_alpha
        alpha = self.split_fracs(alpha)

        alpha_pre, alpha_res = alpha[..., :v], alpha[..., v:]
        alpha_pre = alpha_pre.sigmoid()

        # The transport chart consumes only the top-left (n-1)×(n-1) slice.
        # This keeps the parameter budget aligned with the theoretical RTBP2N
        # chart dimension while preserving the existing HC tensor plumbing.
        alpha_res = rearrange(alpha_res, "... f s g t -> ... f g s t")
        rtbp_params = alpha_res[..., : n - 1, : n - 1]

        orig_shape = rtbp_params.shape[:-2]
        rtbp_flat = rtbp_params.reshape(-1, n - 1, n - 1)

        # Convert the unconstrained chart parameters into an exact doubly
        # stochastic residual mixing matrix.
        ds_flat = recursive_transport_birkhoff_power2(
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


RTBP2N_MHC.get_expand_reduce_stream_functions = staticmethod(
    get_expand_reduce_stream_functions
)
RTBP2N_MHC.get_init_and_expand_reduce_stream_functions = staticmethod(
    get_init_and_expand_reduce_stream_functions
)


class PowerOfTwoRecursiveTransportBirkhoff(nn.Module):
    """Standalone wrapper for recursive_transport_birkhoff_power2()."""

    def __init__(self, num_streams: int, make_dse: bool = True):
        super().__init__()
        if num_streams < 2:
            raise ValueError("num_streams must be >= 2")
        if not _is_power_of_two(num_streams):
            raise ValueError(f"num_streams must be a power of 2, got {num_streams}")
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
        return recursive_transport_birkhoff_power2(
            t,
            delta_logit=self.delta_logit if self.make_dse else None,
        )


RTBP2NHC = RTBP2N_MHC
