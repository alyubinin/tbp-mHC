"""
LMAMSTBP-mHC: Lazyfied-Minorized Averaged Margined Scaled Transport Birkhoff Polytope

This module extends AMSTBP-mHC by applying a lazyfied-minorized transformation to the
residual mixing matrix:

    H_n = (1 - λ_n - μ_n) * I + λ_n * X_n + μ_n * (J/n)

where X_n is the averaged MSTBP matrix from AMSTBP, I is identity, J = 1·1^T, and
λ_n, μ_n are learnable scalars per layer that control the spectral gap.

===============================================================================
MOTIVATION
===============================================================================

- λ_n: weight on the learned mixing X_n (lazy = more identity when λ small)
- μ_n: weight on uniform mixing J/n (minorization toward uniform)
- (1-λ-μ): weight on identity (stay put)
- Convex combination preserves doubly stochasticity
"""

from __future__ import annotations
import math
from typing import Optional, List, Union

from functools import partial

import torch
from torch import nn, cat
from torch.nn import Module

from einops import rearrange, repeat

from .utils import exists, default, add, Residual, get_expand_reduce_stream_functions
from .base import BaseHyperConnections
from .amstbp_mhc import (
    AMSTBP_MHC,
    averaged_margined_sequential_birkhoff,
    get_init_and_expand_reduce_stream_functions as amstbp_get_init_and_expand_reduce_stream_functions,
)


# ============================================================================
# LAZYFIED-MINORIZED TRANSFORM
# ============================================================================

def _lazyfied_minorized(X: torch.Tensor, lambda_val: torch.Tensor, mu_val: torch.Tensor, n: int) -> torch.Tensor:
    """
    Apply H = (1 - λ - μ) * I + λ * X + μ * (J/n).
    X: (..., n, n), lambda_val and mu_val: scalars (or broadcastable).
    """
    device = X.device
    dtype = X.dtype
    I = torch.eye(n, device=device, dtype=dtype)
    J_over_n = torch.ones(n, n, device=device, dtype=dtype) / n

    # Ensure lambda, mu broadcast correctly
    while lambda_val.ndim < X.ndim:
        lambda_val = lambda_val.unsqueeze(0)
    while mu_val.ndim < X.ndim:
        mu_val = mu_val.unsqueeze(0)

    H = (1 - lambda_val - mu_val) * I + lambda_val * X + mu_val * J_over_n
    return H


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
    lambda_init=None,
    mu_init=None,
    **kwargs
):
    """Get initializer for LMAMSTBP-mHC layers plus expand/reduce functions."""
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    if permutations is None:
        permutations = ["direct", "reverse"]

    lmamstbp_kwargs = dict(permutations=permutations, num_fracs=num_fracs, **kwargs)
    if lambda_init is not None:
        lmamstbp_kwargs["lambda_init"] = lambda_init
    if mu_init is not None:
        lmamstbp_kwargs["mu_init"] = mu_init

    init_hyper_conn_fn = partial(LMAMSTBP_MHC, num_streams, **lmamstbp_kwargs)
    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams, add_stream_embed=add_stream_embed, dim=dim, disable=disable
    )

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim=dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


# ============================================================================
# LMAMSTBP-mHC IMPLEMENTATION
# ============================================================================

class LMAMSTBP_MHC(AMSTBP_MHC):
    """
    LMAMSTBP-mHC: Lazyfied-Minorized Averaged MSTBP.

    Same as AMSTBP but applies H_n = (1-λ-μ)*I + λ*X_n + μ*(J/n) to the
    residual mixing matrix. λ and μ are learnable scalars per layer.
    """

    def __init__(
        self,
        num_residual_streams: int,
        *,
        dim: int,
        permutations: Optional[List[Union[str, List[int]]]] = None,
        lambda_init: float = 0.05,
        mu_init: float = 0.01,
        **kwargs
    ):
        self.lambda_init = lambda_init
        self.mu_init = mu_init
        super().__init__(
            num_residual_streams,
            dim=dim,
            permutations=permutations,
            **kwargs
        )

    def _init_hyper_params(self):
        """Initialize AMSTBP params plus λ_n and μ_n (scalars per layer)."""
        super()._init_hyper_params()

        # Use simplex param: λ=sigmoid(λ_logit), μ=(1-λ)*sigmoid(μ_logit)
        lam = self.lambda_init
        mu_target = self.mu_init
        lambda_logit = math.log(lam / (1 - lam))
        mu_logit = math.log((mu_target / (1 - lam)) / (1 - mu_target / (1 - lam)))

        self.lambda_logit = nn.Parameter(torch.tensor(lambda_logit))
        self.mu_logit = nn.Parameter(torch.tensor(mu_logit))

    def _compute_alpha_beta(self, normed: torch.Tensor, device: torch.device):
        """Compute H^pre, lazyfied-minorized H^res (LMAMSTBP), and H^post."""
        alpha, beta = super()._compute_alpha_beta(normed, device)

        n = self.num_residual_streams
        v = self.num_input_views

        # alpha shape: (..., f1, s, f2, v+s). Residual part is alpha[..., v:]
        alpha_pre = alpha[..., :v]
        alpha_res = alpha[..., v:]  # (..., f1, s, f2, n) - H^res[target, source] = alpha_res[..., f1, source, f2, target]

        # Apply lazyfied-minorized: H = (1-λ-μ)*I + λ*X + μ*(J/n)
        # alpha_res has (s, n) = (n, n) at dims -3 and -1; rearrange so last two dims are (n, n)
        lambda_val = torch.sigmoid(self.lambda_logit).to(device)
        mu_val = (1 - lambda_val) * torch.sigmoid(self.mu_logit).to(device)

        alpha_res = rearrange(alpha_res, '... f1 s f2 n -> ... f1 f2 s n')
        alpha_res = _lazyfied_minorized(alpha_res, lambda_val, mu_val, n)
        alpha_res = rearrange(alpha_res, '... f1 f2 s n -> ... f1 s f2 n')

        alpha = cat((alpha_pre, alpha_res), dim=-1)

        return alpha, beta


# Static method attachments
LMAMSTBP_MHC.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
LMAMSTBP_MHC.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)
