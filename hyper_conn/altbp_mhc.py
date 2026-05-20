"""
ALTBP-mHC: Averaged Linear Transport Birkhoff Polytope Manifold-Constrained Hyper-Connections

This module implements the averaged chart mixture for LTBP-mHC. It runs the linear
sequential Birkhoff algorithm with different row/column orderings (permutations),
then averages the resulting doubly stochastic matrices with learnable weights.
Same as ALSB but with linear X[i,j] = L + (U-L)*t instead of sigmoid, and explicit
t parameters clamped to [0,1] after each optimizer step.
"""

from __future__ import annotations
from typing import Optional, List, Union

from functools import partial

import torch
from torch import nn, cat
from torch.nn import Module

from einops import rearrange, repeat

from .utils import exists, default, add, get_expand_reduce_stream_functions
from .base import BaseHyperConnections
from .ltbp_mhc import linear_sequential_birkhoff


# ============================================================================
# PERMUTATION HELPERS (shared with ALSB/ASTBP/AMSTBP)
# ============================================================================

def _resolve_permutation(spec: Union[str, List[int]], n: int) -> List[int]:
    """Resolve permutation spec to list of n indices. 'direct'=[0..n-1], 'reverse'=[n-1..0]."""
    if isinstance(spec, str):
        if spec == "direct":
            return list(range(n))
        elif spec == "reverse":
            return list(range(n - 1, -1, -1))
        else:
            raise ValueError(f"Unknown permutation name: {spec}. Use 'direct', 'reverse', or a list of {n} indices.")
    elif isinstance(spec, (list, tuple)):
        arr = list(spec)
        if len(arr) != n:
            raise ValueError(f"Permutation must have length {n}, got {len(arr)}")
        if set(arr) != set(range(n)):
            raise ValueError(f"Permutation must contain each of 0..{n-1} exactly once")
        return arr
    else:
        raise ValueError(f"Permutation spec must be str or list, got {type(spec)}")


def _permutation_matrix(pi: List[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build P where P[i, π[i]] = 1. P^T @ M @ P conjugates M by permutation π."""
    n = len(pi)
    P = torch.zeros(n, n, device=device, dtype=dtype)
    for i in range(n):
        P[i, pi[i]] = 1.0
    return P


def _conjugate_by_permutation(M: torch.Tensor, pi: List[int]) -> torch.Tensor:
    """Return P^T @ M @ P so that result[π(i), π(j)] = M[i,j]."""
    n = M.shape[-1]
    device, dtype = M.device, M.dtype
    P = _permutation_matrix(pi, device, dtype)
    return torch.einsum("ij,...jk,kl->...il", P.T, M, P)


# ============================================================================
# AVERAGED LINEAR SEQUENTIAL BIRKHOFF
# ============================================================================

def averaged_linear_sequential_birkhoff(
    t_list: List[torch.Tensor],
    permutations: List[Union[str, List[int]]],
    weights: torch.Tensor,
    n: int,
    *,
    delta_logit: Optional[torch.Tensor] = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute H̃ = Σ αₖ H_k where H_k = P_πₖ^T S_πₖ(t⁽ᵏ⁾) P_πₖ.
    Uses linear formula: X[i,j] = L + (U-L)*t[i,j], t ∈ [0,1].
    """
    K = len(permutations)
    assert len(t_list) == K, f"Need {K} parameter tensors for {K} permutations"

    device = t_list[0].device
    dtype = t_list[0].dtype
    batch_shape = t_list[0].shape[:-2]

    alpha = torch.softmax(weights.float(), dim=-1).to(dtype=dtype)
    while alpha.ndim < len(batch_shape) + 1:
        alpha = alpha.unsqueeze(0)

    orig_shape = t_list[0].shape[:-2]
    B = int(torch.tensor(orig_shape).prod().item()) if len(orig_shape) > 0 else 1

    H_sum = None
    for k in range(K):
        pi = _resolve_permutation(permutations[k], n)
        t_k = t_list[k]

        t_flat = t_k.reshape(B, n - 1, n - 1)
        S_k = linear_sequential_birkhoff(t_flat, delta_logit=None, eps=eps)
        H_k = _conjugate_by_permutation(S_k, pi)
        H_k = H_k.reshape(*orig_shape, n, n)

        a_k = alpha[..., k]
        while a_k.ndim < H_k.ndim:
            a_k = a_k.unsqueeze(-1)

        if H_sum is None:
            H_sum = a_k * H_k
        else:
            H_sum = H_sum + a_k * H_k

    if exists(delta_logit):
        delta = torch.sigmoid(delta_logit).to(dtype=dtype, device=device)
        while delta.ndim < H_sum.ndim:
            delta = delta.unsqueeze(0)
        H_sum = (1 - delta) * H_sum + delta * (1.0 / n)

    return H_sum


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def get_init_and_expand_reduce_stream_functions(
    num_streams,
    num_fracs=1,
    dim=None,
    add_stream_embed=False,
    disable=None,
    permutations=None,
    **kwargs
):
    """Get initializer for ALTBP-mHC layers plus expand/reduce functions."""
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    if permutations is None:
        permutations = ["direct", "reverse"]

    init_hyper_conn_fn = partial(
        ALTBP_MHC,
        num_streams,
        permutations=permutations,
        num_fracs=num_fracs,
        **kwargs
    )
    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams, add_stream_embed=add_stream_embed, dim=dim, disable=disable
    )

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim=dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


# ============================================================================
# ALTBP-mHC IMPLEMENTATION
# ============================================================================

class ALTBP_MHC(BaseHyperConnections):
    """
    ALTBP-mHC: Averaged Linear TBP with mixture of permutation charts.

    K permutations, each with explicit (n-1)² params t ∈ [0,1], averaged with
    learnable weights. Call clamp_birkhoff_params() after each optimizer.step().
    """

    def __init__(
        self,
        num_residual_streams: int,
        *,
        dim: int,
        permutations: Optional[List[Union[str, List[int]]]] = None,
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
        assert num_residual_streams >= 2, "ALTBP-mHC requires at least 2 streams"
        self.make_dse = make_dse
        self.permutations = default(permutations, ["direct", "reverse"])
        self.K = len(self.permutations)

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
        """Initialize ALTBP-mHC parameters: K charts × (n-1)² params, chart weights, H^pre, H^post."""
        n = self.num_residual_streams
        f = self.num_fracs
        v = self.num_input_views
        d = self.dim_per_frac
        K = self.K

        init_alpha_pre = torch.ones((n * f, v * f)) * -1
        init_alpha_pre[self.init_residual_index, :] = 1.

        self.static_alpha = nn.Parameter(init_alpha_pre)

        # Explicit K × (n-1)² parameters for H^res, init to 0.5
        self.birkhoff_params = nn.Parameter(torch.ones(K, n - 1, n - 1) * 0.5)

        # Chart weights (logits for softmax)
        self.chart_weights = nn.Parameter(torch.zeros(K))

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
        """Compute H^pre, averaged H^res (linear Birkhoff), and H^post."""
        n = self.num_residual_streams
        f = self.num_fracs
        v = self.num_input_views
        K = self.K

        # H^pre: dynamic + static
        wc_weight = normed @ self.dynamic_alpha_fn
        wc_weight = rearrange(wc_weight, '... (s t) -> ... s t', s=n)
        pre_scale = repeat(self.pre_branch_scale, '1 -> v', v=v * f)
        dynamic_pre = wc_weight[..., :v * f] * pre_scale
        static_pre = rearrange(self.static_alpha, '(f s) v -> f s v', s=n)
        alpha_pre = (rearrange(dynamic_pre, '... n (f v) -> ... f n v', f=f) + static_pre).sigmoid()
        alpha_pre = alpha_pre.unsqueeze(-2)  # (..., f, n, 1, v)

        # H^res: averaged linear Birkhoff from explicit params
        t_list = [self.birkhoff_params[k] for k in range(K)]
        weights = self.chart_weights  # (K,)
        ds = averaged_linear_sequential_birkhoff(
            t_list,
            self.permutations,
            weights,
            n,
            delta_logit=self.delta_logit if self.make_dse else None,
        )
        batch_shape = normed.shape[:-1]
        alpha_res = ds.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-2)
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
ALTBP_MHC.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
ALTBP_MHC.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)
