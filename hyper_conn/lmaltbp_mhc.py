"""
LMALTBP-mHC: Lazyfied-Minorized Averaged Linear Transport Birkhoff Polytope

This module extends ALTBP-mHC by applying a lazyfied-minorized transformation to the
residual mixing matrix:

    H_n = (1 - λ_n - μ_n) * I + λ_n * X_n + μ_n * (J/n)

where X_n is the averaged LTBP matrix from ALTBP, I is identity, J = 1·1^T, and
λ_n, μ_n are learnable scalars per layer that control the spectral gap.
"""

from __future__ import annotations
import math
from typing import Optional, List, Union

from functools import partial

import torch
from torch import nn, cat
from torch.nn import Module

from einops import rearrange, repeat

from .utils import exists, default, add, get_expand_reduce_stream_functions
from .base import BaseHyperConnections
from .altbp_mhc import (
    ALTBP_MHC,
    get_init_and_expand_reduce_stream_functions as altbp_get_init_and_expand_reduce_stream_functions,
)
from .lmamstbp_mhc import _lazyfied_minorized


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
    """Get initializer for LMALTBP-mHC layers plus expand/reduce functions."""
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    if permutations is None:
        permutations = ["direct", "reverse"]

    lmaltbp_kwargs = dict(permutations=permutations, num_fracs=num_fracs, **kwargs)
    if lambda_init is not None:
        lmaltbp_kwargs["lambda_init"] = lambda_init
    if mu_init is not None:
        lmaltbp_kwargs["mu_init"] = mu_init

    init_hyper_conn_fn = partial(LMALTBP_MHC, num_streams, **lmaltbp_kwargs)
    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams, add_stream_embed=add_stream_embed, dim=dim, disable=disable
    )

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim=dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


# ============================================================================
# LMALTBP-mHC IMPLEMENTATION
# ============================================================================

class LMALTBP_MHC(ALTBP_MHC):
    """
    LMALTBP-mHC: Lazyfied-Minorized Averaged Linear TBP.

    Same as ALTBP but applies H_n = (1-λ-μ)*I + λ*X_n + μ*(J/n) to the
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
        """Initialize ALTBP params plus λ_n and μ_n (scalars per layer)."""
        super()._init_hyper_params()

        # Use simplex param: λ=sigmoid(λ_logit), μ=(1-λ)*sigmoid(μ_logit)
        lam = self.lambda_init
        mu_target = self.mu_init
        lambda_logit = math.log(lam / (1 - lam))
        mu_logit = math.log((mu_target / (1 - lam)) / (1 - mu_target / (1 - lam)))

        self.lambda_logit = nn.Parameter(torch.tensor(lambda_logit))
        self.mu_logit = nn.Parameter(torch.tensor(mu_logit))

    def _compute_alpha_beta(self, normed: torch.Tensor, device: torch.device):
        """Compute H^pre, lazyfied-minorized H^res (LMALTBP), and H^post."""
        alpha, beta = super()._compute_alpha_beta(normed, device)

        n = self.num_residual_streams
        v = self.num_input_views

        # alpha shape: (..., f1, s, f2, v+s). Residual part is alpha[..., v:]
        alpha_pre = alpha[..., :v]
        alpha_res = alpha[..., v:]

        # Apply lazyfied-minorized: H = (1-λ-μ)*I + λ*X + μ*(J/n)
        lambda_val = torch.sigmoid(self.lambda_logit).to(device)
        mu_val = (1 - lambda_val) * torch.sigmoid(self.mu_logit).to(device)

        alpha_res = rearrange(alpha_res, '... f1 s f2 n -> ... f1 f2 s n')
        alpha_res = _lazyfied_minorized(alpha_res, lambda_val, mu_val, n)
        alpha_res = rearrange(alpha_res, '... f1 f2 s n -> ... f1 s f2 n')

        alpha = cat((alpha_pre, alpha_res), dim=-1)

        return alpha, beta


# Static method attachments
LMALTBP_MHC.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
LMALTBP_MHC.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)
