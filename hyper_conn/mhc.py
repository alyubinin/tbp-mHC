"""
mHC: Manifold-Constrained Hyper-Connections (Sinkhorn-Knopp variant)

This module implements the original mHC method from DeepSeek as described in:
    "mHC: Manifold-Constrained Hyper-Connections"
    Xie et al., 2025 (arXiv:2512.24880)

Also referenced in the KromHC paper (Zhou et al., 2026) for comparison.

===============================================================================
BACKGROUND: HYPER-CONNECTIONS (HC)
===============================================================================

Standard residual connections (He et al., 2016) have limited expressivity:
    x_{l+1} = x_l + F(x_l)

Hyper-Connections (Zhu et al., 2025) expand this by using n residual streams:
    X_{l+1} = H^res_l @ X_l + H^post_l^T @ F(H^pre_l @ X_l)   (Equation 1)

where:
    - X_l ∈ R^{n×C}: n streams, each of dimension C
    - H^res_l ∈ R^{n×n}: learnable residual mixing matrix
    - H^pre_l ∈ R^{1×n}: aggregates streams for branch input
    - H^post_l ∈ R^{1×n}: distributes branch output to streams

PROBLEM: Unconstrained H^res leads to training instability because
the product ∏_{i=1}^{L-1} H^res_{L-i} fails to preserve the identity
mapping property (Equation 2 in paper). Feature norms can explode or vanish.

===============================================================================
MANIFOLD-CONSTRAINED HC (mHC) SOLUTION
===============================================================================

mHC constrains H^res_l to lie on the BIRKHOFF POLYTOPE B_n, which is the
set of all n×n DOUBLY STOCHASTIC matrices. A matrix is doubly stochastic if:
    1. All entries are non-negative: H^res_l >= 0
    2. Rows sum to 1: H^res_l @ 1_n = 1_n  
    3. Columns sum to 1: H^res_l^T @ 1_n = 1_n   (Equation 3)

WHY DOUBLY STOCHASTIC?
- Spectral norm is bounded: ||H^res_l|| <= 1 (norm preservation)
- Product of DS matrices is DS (compositional closure)
- Residual mixing becomes a CONVEX COMBINATION of inputs
- Feature mean is preserved across layers -> stable training

SINKHORN-KNOPP ALGORITHM:
mHC uses the Sinkhorn-Knopp (SK) algorithm to project arbitrary matrices
onto the Birkhoff polytope. SK alternates row and column normalization:

    for iteration in range(num_iters):
        M = normalize_rows(M)    # Rows sum to 1
        M = normalize_cols(M)    # Columns sum to 1

After many iterations, M converges to a doubly stochastic matrix.

LIMITATION (addressed by KromHC):
The SK algorithm only APPROXIMATES double stochasticity. With finite
iterations (typically 20), there's residual error that accumulates
across layers (see Figure 2 in KromHC paper - MAE grows to ~0.05).

===============================================================================
PARAMETRIZATION (Equation 4 in KromHC paper)
===============================================================================

Given flattened input x_l = vec(X_l) ∈ R^{1×nC}:

1. x'_l = RMSNorm(x_l)

2. H^pre_l = σ(α^pre_l * x'_l @ W^pre_l + b^pre_l)

3. H^post_l = 2σ(α^post_l * x'_l @ W^post_l + b^post_l)

4. H^res_l = SK(α^res_l * mat(x'_l @ W^res_l) + b^res_l)
   
   where SK(·) is 20 iterations of Sinkhorn-Knopp normalization

Note: W^res_l ∈ R^{nC × n²} - parameter complexity is O(n³C)!

===============================================================================
COMPARISON WITH OTHER VARIANTS
===============================================================================

| Method    | Exact DS? | Param Complexity | Custom Kernels |
|-----------|-----------|------------------|----------------|
| mHC (this)| No        | O(n³C)           | Yes (for SK)   |
| mHC-lite  | Yes       | O(nC × n!)       | No             |
| KromHC    | Yes       | O(n²C)           | No             |

===============================================================================
NOTATION (Einstein summation via einops)
===============================================================================
b - batch dimension
d - feature dimension (C in paper)
s - number of residual streams (n in paper)
f - number of fractions
v - number of input views
"""

from __future__ import annotations
from typing import Callable

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
# SINKHORN-KNOPP ALGORITHM
# ============================================================================

def l1norm(t, dim):
    """
    L1 normalization along a dimension.
    Makes the specified dimension sum to 1.
    """
    return F.normalize(t, p = 1, dim = dim)


def sinkhorn_knopps(log_alpha, iters=20):
    """
    Sinkhorn-Knopp algorithm for projecting onto the Birkhoff polytope.
    
    This implements the iterative normalization procedure from:
        Sinkhorn & Knopp (1967) "Concerning nonnegative matrices and
        doubly stochastic matrices"
    
    The algorithm alternates between normalizing rows and columns:
        M^(t+1) = D_r^(-1) @ M^(t) @ D_c^(-1)
    
    where D_r and D_c are diagonal matrices that normalize rows/cols.
    
    LOG-SPACE IMPLEMENTATION:
    Working in log-space is more numerically stable. The updates become:
        log_M = log_M - logsumexp(log_M, dim=rows)
        log_M = log_M - logsumexp(log_M, dim=cols)
    
    After `iters` iterations, exp(log_M) is approximately doubly stochastic.
    
    LIMITATION (from KromHC paper, Figure 2):
    With finite iterations, there's residual error in the DS property.
    This error accumulates when multiplying H^res across many layers,
    leading to training instabilities in very deep networks.
    
    Args:
        log_alpha: Log of the pre-normalized matrix, shape (..., n, n)
        iters: Number of SK iterations (default 20, as in mHC paper)
        
    Returns:
        Approximately doubly stochastic matrix of same shape
    """
    for _ in range(iters):
        # Row normalization in log-space
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        # Column normalization in log-space
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)

    return log_alpha.exp()


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def get_init_and_expand_reduce_stream_functions(
    num_streams,
    num_fracs = 1,
    dim = None,
    add_stream_embed = False,
    disable = None,
    sinkhorn_iters = 20,
    **kwargs
):
    """
    Get initializer for mHC layers plus expand/reduce functions.
    
    Args:
        num_streams: n, number of residual streams
        num_fracs: Number of fractions for frac-connections
        dim: Feature dimension C
        add_stream_embed: Whether to add learnable stream embeddings
        disable: Force disable hyper-connections
        sinkhorn_iters: Number of Sinkhorn-Knopp iterations (default 20)
        **kwargs: Additional arguments passed to mHC constructor
        
    Returns:
        (init_fn, expand_fn, reduce_fn) tuple
    """
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = ManifoldConstrainedHyperConnections if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, num_fracs = num_fracs, sinkhorn_iters = sinkhorn_iters, **kwargs)
    expand_reduce_fns = get_expand_reduce_stream_functions(num_streams, add_stream_embed = add_stream_embed, dim = dim, disable = disable)

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim = dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


# ============================================================================
# MANIFOLD-CONSTRAINED HYPER-CONNECTIONS (mHC)
# ============================================================================

class ManifoldConstrainedHyperConnections(BaseHyperConnections):
    """
    Manifold-Constrained Hyper-Connections using Sinkhorn-Knopp projection.
    
    This is the original mHC from DeepSeek (Xie et al., 2025) that projects
    the residual matrix onto the Birkhoff polytope using the Sinkhorn-Knopp
    algorithm.
    
    KEY INSIGHT (from KromHC paper):
    mHC addresses HC's instability by constraining H^res to be doubly
    stochastic, which:
    - Bounds the spectral norm to 1
    - Preserves feature mean across layers
    - Ensures composability (product of DS matrices is DS)
    
    LIMITATION:
    The Sinkhorn-Knopp algorithm only APPROXIMATES double stochasticity.
    With 20 iterations, there's residual error (~0.05 MAE per layer) that
    accumulates across the network depth, potentially causing instabilities
    in very deep models (Figure 2 in KromHC paper).
    
    PARAMETER COMPLEXITY: O(n³C)
    W^res_l ∈ R^{nC × n²} - this is cubic in stream width n, limiting
    scalability compared to KromHC's O(n²C).
    
    INHERITANCE:
    ============
    Extends BaseHyperConnections, inheriting shared functionality.
    Implements _init_hyper_params() and _compute_alpha_beta() for SK projection.
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
        sinkhorn_iters: int = 20
    ):
        """
        Initialize mHC layer.
        
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
            sinkhorn_iters: Number of SK iterations (default 20)
        """
        self.sinkhorn_iters = sinkhorn_iters
        
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
        """Initialize mHC parameters: H^pre, H^res (n²), H^post."""
        n = self.num_residual_streams
        f = self.num_fracs
        v = self.num_input_views
        d = self.dim_per_frac
        
        # H^pre: mostly -1, one stream at +1
        init_alpha_pre = torch.ones((n * f, v * f)) * -1
        init_alpha_pre[self.init_residual_index, :] = 1.
        
        # H^res: mostly -8, diagonal at 0 -> SK approx identity
        init_alpha_res = torch.ones((n * f, n * f)) * -8
        init_alpha_res.fill_diagonal_(0.)
        
        self.static_alpha = nn.Parameter(cat((init_alpha_pre, init_alpha_res), dim=1))
        
        # Dynamic weights: O(n³C) bottleneck
        self.dynamic_alpha_fn = nn.Parameter(
            torch.zeros(d * n, f * (n * n + n * v))
        )
        
        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
        self.residual_scale = nn.Parameter(torch.ones(1) * 1e-2)
        
        # H^post
        if self.add_branch_out_to_residual:
            beta_init = torch.ones(n * f) * -1.
            beta_init[self.init_residual_index] = 1.
            self.static_beta = nn.Parameter(beta_init)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(d * n, f * n))
            self.h_post_scale = nn.Parameter(torch.ones(()) * 1e-2)
    
    def _compute_alpha_beta(self, normed: torch.Tensor, device: torch.device):
        """
        Compute H^pre, H^res (Sinkhorn-Knopp), and H^post matrices.
        
        H^res is projected onto the Birkhoff polytope via SK iterations.
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
        
        # Sinkhorn-Knopp projection for approximate DS
        alpha_res = rearrange(alpha_res, '... f s g t -> ... f g s t')
        alpha_res = sinkhorn_knopps(alpha_res, self.sinkhorn_iters)
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


ManifoldConstrainedHyperConnections.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
ManifoldConstrainedHyperConnections.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)
