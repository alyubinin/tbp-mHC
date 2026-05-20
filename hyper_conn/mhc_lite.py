"""
mHC-lite: Manifold-Constrained Hyper-Connections via Birkhoff-von-Neumann

This module implements the mHC-lite method as described in:
    "mHC-lite: You Don't Need 20 Sinkhorn-Knopp Iterations"
    Yang & Gao, 2026 (arXiv:2601.05732)

Also referenced in the KromHC paper (Zhou et al., 2026) for comparison.

===============================================================================
MOTIVATION
===============================================================================

mHC (Xie et al., 2025) uses Sinkhorn-Knopp to project H^res onto the Birkhoff
polytope, but:
1. SK only APPROXIMATES double stochasticity (finite iterations)
2. Requires custom CUDA kernels for efficiency
3. Error accumulates across layers (Figure 2 in KromHC paper)

mHC-lite addresses these issues using the BIRKHOFF-VON-NEUMANN THEOREM.

===============================================================================
BIRKHOFF-VON-NEUMANN THEOREM (Theorem 4.1 in KromHC paper)
===============================================================================

Any n×n DOUBLY STOCHASTIC matrix X can be written as:

    X = Σ_{k=1}^{n!} a_k * P_k

where:
    - P_1, ..., P_{n!} are ALL n×n permutation matrices
    - a_k >= 0 for all k (non-negative coefficients)
    - Σ_k a_k = 1 (coefficients sum to 1)

In other words, the Birkhoff polytope B_n is the CONVEX HULL of the n!
permutation matrices. Any doubly stochastic matrix is a convex combination
of permutation matrices.

===============================================================================
mHC-lite APPROACH
===============================================================================

Instead of SK projection, mHC-lite directly parameterizes H^res as:

    H^res_l = Σ_{k=1}^{n!} a_l(k) * P_k

where the coefficients come from a softmax:
    a_l = Softmax(α^res_l * x'_l @ W^res_l + b^res_l)

This GUARANTEES exact double stochasticity by construction!

ADVANTAGES:
- Exact doubly stochastic matrices (no approximation error)
- PyTorch native operations (no custom kernels needed)
- Simpler implementation than SK iterations

===============================================================================
LIMITATION: FACTORIAL EXPLOSION
===============================================================================

The number of permutation matrices for n×n is n! (n factorial):
    - n=2: 2 matrices
    - n=3: 6 matrices
    - n=4: 24 matrices
    - n=8: 40,320 matrices
    - n=16: 20,922,789,888,000 matrices!

mHC-lite requires storing all n! permutation matrices and learning n!
coefficients per layer. This creates:
    - Parameter complexity: O(nC × n!)
    - Memory: O(n² × n!) for storing permutation matrices

This factorial explosion makes mHC-lite INFEASIBLE for n > ~6 or so.
(See Figure 3 in KromHC paper)

===============================================================================
KromHC SOLUTION
===============================================================================

KromHC avoids the factorial explosion by using KRONECKER PRODUCTS of smaller
doubly stochastic matrices. For n = 2^K:
    - K factors of size 2×2
    - Each 2×2 has only 2 permutations
    - Total: 2K parameters vs n! for mHC-lite

===============================================================================
PARAMETRIZATION (Equation 26 in KromHC paper, Appendix F)
===============================================================================

Given flattened input x_l = vec(X_l):

1. x'_l = RMSNorm(x_l)

2. H^pre_l = sigmoid(α^pre_l * x'_l @ W^pre_l + b^pre_l)

3. H^post_l = 2 * sigmoid(α^post_l * x'_l @ W^post_l + b^post_l)

4. a_l = softmax(α^res_l * x'_l @ W^res_l + b^res_l)  # n! coefficients
   H^res_l = Σ_{k=1}^{n!} a_l(k) * P_k

===============================================================================
NOTATION (Einstein summation via einops)
===============================================================================
b - batch dimension
d - feature dimension (C in paper)
s - number of residual streams (n in paper)
f - number of fractions
v - number of input views
r - permutation index (1 to n!)
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
import itertools

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
# PERMUTATION MATRIX GENERATION
# ============================================================================

def get_all_permutations(n: int):
    """
    Generate ALL n×n permutation matrices for Birkhoff-von-Neumann decomposition.
    
    A permutation matrix is a 0-1 matrix with exactly one 1 in each row and
    column. For n×n, there are exactly n! such matrices (one for each
    permutation of {0, 1, ..., n-1}).
    
    The Birkhoff-von-Neumann theorem states that these n! matrices form the
    vertices of the Birkhoff polytope, and any doubly stochastic matrix
    can be written as a convex combination of these vertices.
    
    WARNING: This grows FACTORIALLY!
        n=4: 24 matrices (OK)
        n=5: 120 matrices (OK)
        n=6: 720 matrices (starting to get large)
        n=8: 40,320 matrices (significant memory)
        n=10: 3,628,800 matrices (infeasible!)
    
    Args:
        n: Size of permutation matrices
        
    Returns:
        Tensor of shape (n!, n, n) containing all n! permutation matrices
        
    Example for n=2:
        Returns 2 matrices:
        P_0 = [[1, 0], [0, 1]]  (identity - maps 0->0, 1->1)
        P_1 = [[0, 1], [1, 0]]  (swap - maps 0->1, 1->0)
    
    Example for n=3:
        Returns 6 matrices corresponding to permutations:
        (0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)
    """
    assert n >= 1, "n must be a positive integer"

    # Generate all permutations of [0, 1, ..., n-1]
    perms = list(itertools.permutations(range(n)))
    index = torch.tensor(perms, dtype=torch.long)

    # Convert permutation indices to permutation matrices
    # P[perm][i, j] = 1 iff perm[i] = j
    eye = torch.eye(n, dtype=torch.float32)
    perm_mats = eye[index]  # (n!, n, n)

    return perm_mats


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
    """
    Get initializer for mHC-lite layers plus expand/reduce functions.
    """
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = MHCLite if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, num_fracs = num_fracs, **kwargs)
    expand_reduce_fns = get_expand_reduce_stream_functions(num_streams, add_stream_embed = add_stream_embed, dim = dim, disable = disable)

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim = dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


# ============================================================================
# PERMUTATION MATRIX CACHE
# ============================================================================
# Global cache to avoid regenerating permutation matrices on every forward pass.
# Key: (n, device_string) -> permutation matrices tensor

perm_mats = {}


# ============================================================================
# mHC-lite IMPLEMENTATION
# ============================================================================

class MHCLite(BaseHyperConnections):
    """
    mHC-lite: Manifold-Constrained Hyper-Connections via Birkhoff-von-Neumann.
    
    This implements the exact doubly stochastic approach using the Birkhoff-
    von-Neumann theorem, which states that any DS matrix is a convex
    combination of permutation matrices.
    
    KEY EQUATION (Theorem 4.1, Equation 26):
    ==========================================
    H^res_l = Σ_{k=1}^{n!} a_l(k) * P_k
    
    where:
    - P_1, ..., P_{n!}: All n×n permutation matrices (precomputed)
    - a_l = softmax(α^res * x'_l @ W^res + b^res): Convex combination weights
    
    Since a_l is a valid probability distribution (softmax outputs) and
    P_k are permutation matrices (which are DS), the convex combination
    is GUARANTEED to be exactly doubly stochastic.
    
    COMPARISON WITH OTHER METHODS:
    ==============================
    | Method   | Exact DS? | Param Complexity |
    |----------|-----------|------------------|
    | mHC      | No        | O(n³C)           |
    | mHC-lite | YES       | O(nC × n!)       | <-- This method
    | KromHC   | YES       | O(n²C)           |
    
    FACTORIAL EXPLOSION PROBLEM:
    ============================
    mHC-lite requires O(n!) permutation matrices and coefficients.
    This is INFEASIBLE for large n:
    
    n=4:  24 perms, ~2KB matrices
    n=6:  720 perms, ~100KB matrices  
    n=8:  40,320 perms, ~10MB matrices
    n=10: 3,628,800 perms, ~1.4GB matrices (!)
    
    KromHC solves this by using Kronecker products of small (2×2) DS matrices,
    achieving O(n²C) complexity while still guaranteeing exact DS.
    
    INHERITANCE:
    ============
    Extends BaseHyperConnections, inheriting shared functionality.
    Implements _init_hyper_params() and _compute_alpha_beta() for BvN logic.
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
    ):
        """
        Initialize mHC-lite layer.
        
        Args:
            num_residual_streams: n, number of streams (WARNING: n! complexity!)
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
        """
        # Precompute permutation matrices BEFORE super().__init__
        if (num_residual_streams, "cpu") not in perm_mats:
            _perm_mats = get_all_permutations(num_residual_streams).to("cpu")
            perm_mats[(num_residual_streams, "cpu")] = _perm_mats
        self.num_perms = len(perm_mats[(num_residual_streams, "cpu")])  # n!
        
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
        """Initialize mHC-lite parameters: H^pre, H^res (BvN), H^post."""
        n = self.num_residual_streams
        f = self.num_fracs
        v = self.num_input_views
        d = self.dim_per_frac
        
        # H^pre biases: mostly -1, one stream at +1
        init_alpha_pre = torch.ones((n * f, v * f)) * -1
        init_alpha_pre[self.init_residual_index, :] = 1.
        
        # H^res biases: identity permutation (P_0) dominates
        init_alpha_res = torch.ones(self.num_perms * f) * -8
        init_alpha_res[0] = 0.  # Identity permutation gets highest weight
        
        self.static_alpha = nn.Parameter(cat([
            init_alpha_pre.view(-1),
            init_alpha_res
        ], dim=-1))
        
        # Dynamic weights: O(nC × n!) bottleneck
        self.dynamic_alpha_fn = nn.Parameter(
            torch.zeros(d * n, f * (self.num_perms + n * v))
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
        Compute H^pre, H^res (BvN), and H^post matrices.
        
        H^res is computed as a convex combination of permutation matrices:
            a_l = softmax(α^res * x'_l @ W^res + b^res)
            H^res_l = Σ_{k=1}^{n!} a_l(k) * P_k
        """
        n = self.num_residual_streams
        f = self.num_fracs
        v = self.num_input_views
        
        wc_weight = normed @ self.dynamic_alpha_fn
        
        psize = v * n * f
        dynamic_pre = wc_weight[..., :psize]
        dynamic_res = wc_weight[..., psize:]
        
        static_pre = self.static_alpha[:psize]
        static_res = self.static_alpha[psize:]
        
        # Get cached permutation matrices
        dev = str(device)
        if (n, dev) not in perm_mats:
            perm_mats[(n, dev)] = get_all_permutations(n).to(device)
        perms = perm_mats[(n, dev)]
        
        # H^res via Birkhoff-von-Neumann (softmax + einsum)
        res_coeff = self.residual_scale * dynamic_res + static_res
        res_coeff = torch.softmax(res_coeff, dim=-1)
        alpha_res = einsum(res_coeff, perms, '... r, r i j -> ... i j')
        alpha_res = self.split_fracs(alpha_res)
        
        # H^pre: sigmoid
        alpha_pre = self.pre_branch_scale * dynamic_pre + static_pre
        alpha_pre = rearrange(alpha_pre, '... (f s v) -> ... s f v', v=v, f=f)
        alpha_pre = alpha_pre.sigmoid()
        
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


MHCLite.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
MHCLite.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)
