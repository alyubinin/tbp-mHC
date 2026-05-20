"""
Hyper-Connections (HC): The Base Architecture

This module implements the original Hyper-Connections from:
    "Hyper-Connections"
    Zhu et al., 2025 (ICLR 2025)
    https://arxiv.org/abs/2409.19606

This is the BASELINE that mHC, mHC-lite, and KromHC all build upon.

===============================================================================
MOTIVATION: BEYOND STANDARD RESIDUAL CONNECTIONS
===============================================================================

The standard residual connection (ResNet, He et al., 2016) is:
    x_{l+1} = x_l + F(x_l)

This has been incredibly successful but has limitations:
1. Single pathway for information flow
2. All features mixed equally (no learned weighting)
3. Limited topological complexity

Hyper-Connections address this by:
1. Expanding the residual stream to n > 1 parallel streams
2. Learning how to mix streams (H^res matrix)
3. Learning how to aggregate/distribute for the branch (H^pre, H^post)

===============================================================================
HYPER-CONNECTIONS FORMULATION (Equation 1 in KromHC paper)
===============================================================================

A single HC layer is defined as:

    X_{l+1} = H^res_l @ X_l + H^post_l^T @ F(H^pre_l @ X_l)

where:
    - X_l ∈ R^{n×C}: n residual streams, each of dimension C
    - H^res_l ∈ R^{n×n}: learnable RESIDUAL MIXING matrix
    - H^pre_l ∈ R^{1×n}: AGGREGATION - combines n streams into 1 for F(·)
    - H^post_l ∈ R^{1×n}: DISTRIBUTION - spreads branch output to n streams
    - F(·): branch function (e.g., attention, FFN)

The key insight is that X_l has MORE CAPACITY than standard residuals
(n streams vs 1), and the mixing matrices allow learning the optimal
way to route information.

===============================================================================
WIDTH vs DEPTH CONNECTIONS
===============================================================================

HC has two types of connections:

1. WIDTH CONNECTION (horizontal):
   - H^res: mixes residual streams among themselves
   - H^pre: aggregates streams for branch input
   
2. DEPTH CONNECTION (vertical):
   - H^post: distributes branch output back to streams
   - Combined with residuals via addition

The terminology comes from viewing the network as a 2D grid:
- Width = number of parallel streams
- Depth = number of sequential layers

===============================================================================
INSTABILITY ISSUE (motivation for mHC)
===============================================================================

The original HC has UNCONSTRAINED H^res matrices. This causes problems:

From Equation 2 in KromHC paper, the output at layer L depends on:
    X_L = (∏_{i=1}^{L-l} H^res_{L-i}) @ X_l + ...

The product of unconstrained matrices can:
- Explode (spectral norm > 1)
- Vanish (spectral norm < 1)
- Distort feature statistics

This motivates MANIFOLD-CONSTRAINED HC (mHC), which restricts H^res
to the Birkhoff polytope (doubly stochastic matrices) to preserve
the identity mapping property.

===============================================================================
PARAMETRIZATION (Appendix E in KromHC paper)
===============================================================================

This implementation uses DYNAMIC mappings that depend on the input:

    X'_l = RMSNorm(X_l)
    H^pre_l = α^pre * tanh(W^pre @ X'^T) + b^pre
    H^post_l = α^post * tanh(W^post @ X'^T) + b^post
    H^res_l = α^res * tanh(W^res @ X'^T) + b^res

This is different from mHC's parametrization (Equation 4) which uses
sigmoid and Sinkhorn-Knopp. The original HC uses tanh without constraints.

===============================================================================
COMPARISON WITH VARIANTS
===============================================================================

| Variant   | H^res Constraint | Exact DS? | Stability |
|-----------|------------------|-----------|-----------|
| HC (this) | None (tanh)      | No        | Unstable  |
| mHC       | Sinkhorn-Knopp   | ≈ Yes     | Better    |
| mHC-lite  | BvN convex combo | Yes       | Best      |
| KromHC    | Kronecker DS     | Yes       | Best      |

===============================================================================
NOTATION (Einstein summation via einops)
===============================================================================
b - batch dimension
d - feature dimension (C in paper)
s - number of residual streams (n in paper)
f - number of fractions
v - number of input views
t - total indices (streams + views)
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
    RMSNorm,
    Residual,
    get_expand_reduce_stream_functions,
)
from .base import BaseHyperConnections


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def get_init_and_expand_reduce_stream_functions(
    num_streams,
    num_fracs = 1,
    dim = None,
    add_stream_embed = False,
    disable = None
):
    """
    Get initializer for HC layers plus expand/reduce functions.
    
    This is the main factory function for creating HyperConnections.
    Returns a partial function that can be called with layer-specific
    parameters to create HC instances.
    
    Args:
        num_streams: n, number of residual streams
        num_fracs: Number of fractions for frac-connections extension
        dim: Feature dimension C
        add_stream_embed: Whether to add learnable stream embeddings
        disable: Force disable hyper-connections
        
    Returns:
        (init_fn, expand_fn, reduce_fn) tuple
    """
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = HyperConnections if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, num_fracs = num_fracs)
    expand_reduce_fns = get_expand_reduce_stream_functions(num_streams, add_stream_embed = add_stream_embed, dim = dim, disable = disable)

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim = dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


# ============================================================================
# HYPER-CONNECTIONS (Original, Unconstrained)
# ============================================================================

class HyperConnections(BaseHyperConnections):
    """
    Hyper-Connections: Learnable Residual Stream Mixing (Zhu et al., 2025)
    
    This is the ORIGINAL Hyper-Connections implementation with unconstrained
    mixing matrices. It expands the residual stream to n parallel streams
    and learns how to mix them dynamically.
    
    ARCHITECTURE:
    ============
    Input X_l ∈ R^{n×C} (n streams, C features)
    
    1. NORMALIZATION:
       X'_l = RMSNorm(X_l)  -- NOTE: Per-stream, not joint!
    
    2. WIDTH CONNECTION (compute mixing matrices):
       H^pre_l = α^pre * tanh(X' @ W^pre) + b^pre   (aggregation)
       H^res_l = α^res * tanh(X' @ W^res) + b^res   (residual mixing)
       
    3. APPLY MIXING:
       branch_input = H^pre_l @ X_l   (aggregate n streams -> 1)
       mixed_residuals = H^res_l @ X_l (mix streams)
       
    4. BRANCH:
       branch_output = F(branch_input)
       
    5. DEPTH CONNECTION:
       output = mixed_residuals + H^post_l^T @ branch_output
    
    INSTABILITY ISSUE:
    ==================
    The H^res matrix is UNCONSTRAINED - it can have arbitrary spectral norm.
    When stacking L layers, the product:
        ∏_{i=1}^{L} H^res_{L-i}
    can explode or vanish, causing training instability.
    
    This is why mHC constrains H^res to the Birkhoff polytope (DS matrices).
    
    KEY DIFFERENCES FROM mHC/KromHC:
    ===============================
    1. Per-stream normalization (not joint normalization)
    2. Tanh activation (not sigmoid)
    3. No doubly stochastic constraint on H^res
    4. Different beta (H^post) parameter structure
    
    INHERITANCE:
    ============
    Extends BaseHyperConnections, overriding:
    - _init_hyper_params(): Different parameter structure
    - width_connection(): Per-stream normalization + tanh activation
    - depth_connection(): Different beta computation
    """
    
    def __init__(
        self,
        num_residual_streams: int,
        *,
        dim: int,
        branch: Module | None = None,
        layer_index: int | None = None,
        tanh: bool = True,
        channel_first: bool = False,
        dropout: float = 0.,
        residual_transform: Module | None = None,
        add_branch_out_to_residual: bool = True,
        num_input_views: int = 1,
        depth_residual_fn = add,
        num_fracs: int = 1
    ):
        """
        Initialize HyperConnections layer.
        
        Args:
            num_residual_streams: n, number of parallel streams
            dim: C, feature dimension
            branch: Optional branch module F(·) to wrap
            layer_index: Deterministic initialization index
            tanh: Use tanh activation (True) or identity (False)
            channel_first: If True, expect (batch, dim, ...) layout
            dropout: Dropout probability
            residual_transform: Transform on residual (for dim changes)
            add_branch_out_to_residual: Enable depth connections
            num_input_views: Number of views for branch input
            depth_residual_fn: Function for output + residual
            num_fracs: Fractions for frac-connections (typically 1)
        """
        self.use_tanh = tanh
        
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
        
        # Override norm from base class - HC uses per-stream normalization
        self.norm = RMSNorm(self.dim_per_frac)
        
        # Activation function
        self.act = nn.Tanh() if tanh else nn.Identity()
    
    def _init_hyper_params(self):
        """Initialize HyperConnections parameters (tanh-based, unconstrained)."""
        n = self.num_residual_streams
        f = self.num_fracs
        v = self.num_input_views
        d = self.dim_per_frac
        
        # H^pre: zeros with one stream at 1
        init_alpha_pre = torch.zeros((n * f, v * f))
        init_alpha_pre[self.init_residual_index, :] = 1.
        
        # H^res: identity (unconstrained, not DS)
        init_alpha_res = torch.eye(n * f)
        
        self.static_alpha = nn.Parameter(cat((init_alpha_pre, init_alpha_res), dim=1))
        
        # Dynamic weights - simpler than mHC
        self.dynamic_alpha_fn = nn.Parameter(torch.zeros(d, n * f + v * f))
        self.dynamic_alpha_scale = nn.Parameter(torch.ones(()) * 1e-2)
        
        # H^post - different structure from mHC
        if self.add_branch_out_to_residual:
            self.static_beta = nn.Parameter(torch.ones(n * f))
            dynamic_beta_shape = (d,) if f == 1 else (d, f)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dynamic_beta_shape))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(()) * 1e-2)
    
    def width_connection(self, residuals: torch.Tensor):
        """
        Compute width connection with per-stream normalization and tanh.
        
        This differs from other variants:
        - Normalization is per-stream (not joint over all streams)
        - Uses tanh activation (not sigmoid)
        - H^res is unconstrained (no DS guarantee)
        """
        streams = self.num_residual_streams
        
        if self.channel_first:
            residuals = rearrange(residuals, 'b d ... -> b ... d')
        
        residuals = self.split_fracs(residuals)
        residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s=streams)
        
        # Per-stream normalization (different from mHC's joint normalization)
        normed = self.norm(residuals)
        
        # Tanh-based dynamic weights
        wc_weight = self.act(normed @ self.dynamic_alpha_fn)
        dynamic_alpha = wc_weight * self.dynamic_alpha_scale
        
        static_alpha = rearrange(self.static_alpha, '(f s) d -> f s d', s=streams)
        alpha = dynamic_alpha + static_alpha
        alpha = self.split_fracs(alpha)
        
        # H^post (beta) - different computation than mHC
        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = self.act(normed @ self.dynamic_beta_fn)
            if not self.has_fracs:
                dc_weight = rearrange(dc_weight, '... -> ... 1')
            dynamic_beta = dc_weight * self.dynamic_beta_scale
            static_beta = rearrange(self.static_beta, '(s f) -> s f', s=streams)
            beta = dynamic_beta + static_beta
        
        # Apply mixing
        mix_h = einsum(alpha, residuals, '... f1 s f2 t, ... f1 s d -> ... f2 t d')
        
        if self.num_input_views == 1:
            branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]
        else:
            branch_input = mix_h[..., :self.num_input_views, :]
            residuals = mix_h[..., self.num_input_views:, :]
            branch_input = rearrange(branch_input, 'b ... v d -> v b ... d')
        
        if self.channel_first:
            branch_input = rearrange(branch_input, 'b ... d -> b d ...')
        branch_input = self.merge_fracs(branch_input)
        
        residuals = rearrange(residuals, 'b ... s d -> (b s) ... d')
        if self.channel_first:
            residuals = rearrange(residuals, 'b ... d -> b d ...')
        residuals = self.merge_fracs(residuals)
        
        return branch_input, residuals, dict(beta=beta)
    
    def depth_connection(self, branch_output: torch.Tensor, residuals: torch.Tensor, *, beta: torch.Tensor):
        """
        Compute depth connection for original HC.
        
        Note: Different einsum pattern than mHC due to different beta shape.
        """
        assert self.add_branch_out_to_residual
        
        branch_output = self.split_fracs(branch_output)
        if self.channel_first:
            branch_output = rearrange(branch_output, 'b d ... -> b ... d')
        
        # Different einsum than mHC (beta has different shape)
        output = einsum(branch_output, beta, 'b ... f1 d, b ... f1 s f2 -> b ... f2 s d')
        output = rearrange(output, 'b ... s d -> (b s) ... d')
        output = self.merge_fracs(output)
        
        if self.channel_first:
            output = rearrange(output, 'b ... d -> b d ...')
        
        residuals = self.depth_residual_fn(output, residuals)
        return self.dropout(residuals)
    
    def _compute_alpha_beta(self, normed: torch.Tensor, device: torch.device):
        """Not used - HyperConnections overrides width_connection directly."""
        raise NotImplementedError("HyperConnections overrides width_connection directly")


# Attach factory functions as static methods
HyperConnections.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
HyperConnections.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)
