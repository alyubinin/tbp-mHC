"""
KromHC: Manifold-Constrained Hyper-Connections with Kronecker-Product Residual Matrices

This module implements the KromHC method as described in:
    "KromHC: Manifold-Constrained Hyper-Connections with Kronecker-Product Residual Matrices"
    Zhou et al., 2026 (arXiv:2601.21579)

===============================================================================
PAPER BACKGROUND AND MOTIVATION
===============================================================================

Hyper-Connections (HC) expand the residual stream width from 1 to n, allowing
more expressive feature propagation through learnable mixing matrices. A single
HC layer is defined as (Equation 1 in the paper):

    X_{l+1} = H^res_l * X_l + H^post_l^T * F(H^pre_l * X_l)

where:
    - X_l ∈ R^{n×C}: expanded input at layer l (n streams, C feature dim)
    - H^res_l ∈ R^{n×n}: residual mixing matrix
    - H^pre_l ∈ R^{1×n}: aggregates n streams into 1 for the branch F(·)
    - H^post_l ∈ R^{1×n}: distributes branch output back to n streams

PROBLEM: Unconstrained H^res_l can cause training instability in deep networks
because the product ∏H^res_l doesn't preserve the identity mapping property.

SOLUTION (mHC): Constrain H^res_l to be doubly stochastic (rows/cols sum to 1,
non-negative). This preserves feature mean across layers. mHC uses the
Sinkhorn-Knopp algorithm, but it doesn't guarantee EXACT double stochasticity.

SOLUTION (mHC-lite): Use Birkhoff-von-Neumann theorem to represent H^res_l as
a convex combination of n! permutation matrices. Guarantees exact DS, but has
O(n! * nC) parameter complexity - factorial explosion!

SOLUTION (KromHC - this implementation): Use Kronecker products of smaller
doubly stochastic matrices. For n = ∏_{k=1}^K i_k:

    H^res_l = U^K_l ⊗ U^{K-1}_l ⊗ ... ⊗ U^1_l    (Equation 10)

where each U^k_l ∈ R^{i_k × i_k} is a small doubly stochastic matrix.

KEY THEOREM (Theorem 4.2 - Kronecker Closure):
The Kronecker product of doubly stochastic matrices is also doubly stochastic.
This GUARANTEES exact double stochasticity with O(n²C) parameter complexity.

For n = 2^K (power of 2), we use K factors of size 2×2. Each 2×2 DS matrix
is a convex combination of only 2 permutation matrices:
    P_1 = [[1,0],[0,1]] (identity)
    P_2 = [[0,1],[1,0]] (swap)

So U^k_l = a * P_1 + (1-a) * P_2 = [[a, 1-a], [1-a, a]] where a ∈ [0,1]

===============================================================================
PARAMETRIZATION (Section 4.3, Equation 14)
===============================================================================

Given flattened input x_l = vec(X_l) ∈ R^{1×nC}:

1. Normalize: x'_l = RMSNorm(x_l)

2. Pre-mapping (aggregates streams for branch input):
   H^pre_l = σ(α^pre_l * x'_l * W^pre_l + b^pre_l)

3. Post-mapping (distributes branch output to streams):
   H^post_l = 2σ(α^post_l * x'_l * W^post_l + b^post_l)

4. Residual matrix via Kronecker structure:
   For each factor k = 1, ..., K:
       a^k_l = Softmax(α^res_l * x'_l * W^{res,k}_l + b^{res,k}_l)
       U^k_l = Σ_{m=1}^{i_k!} a^k_l(m) * P_m

   Then: H^res_l = ⊗_{k=K}^1 U^k_l

===============================================================================
INITIALIZATION (Section 5.1)
===============================================================================

Following Yang & Gao (2026):
- W^{res,k}_l, W^pre_l, W^post_l initialized to zero
- b^pre_l, b^post_l: all -1 except one index = 1
- α^pre_l, α^post_l = 0.01

For 2×2 factors:
- b^{res,k}_l = [0, -8]^T
- α^res_l = 0.01
- This yields a^k_l(1) ≈ 1, a^k_l(2) ≈ 0
- So U^k_l ≈ I_2×2, and H^res_l ≈ I_n×n at initialization

===============================================================================
NOTATION (Einstein summation via einops)
===============================================================================
b - batch dimension
d - feature dimension (C in paper)
s - number of residual streams (n in paper)
f - number of fractions (for frac-connections extension)
v - number of input views for branch
t - total indices (streams + views)
"""

from __future__ import annotations
from typing import Callable

from functools import partial
from random import randrange
import math

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
# PERMUTATION MATRIX GENERATION (for Birkhoff-von-Neumann decomposition)
# ============================================================================

def get_2x2_perm_matrices(device='cpu'):
    """
    Returns the two 2×2 permutation matrices for Birkhoff-von-Neumann decomposition.
    
    Per the Birkhoff-von-Neumann theorem (Theorem 4.1), any doubly stochastic 
    matrix can be written as a convex combination of permutation matrices.
    
    For 2×2, there are exactly 2! = 2 permutation matrices:
        P_1 = [[1, 0], [0, 1]]  (identity - preserves order)
        P_2 = [[0, 1], [1, 0]]  (swap - reverses order)
    
    Any 2×2 doubly stochastic matrix U can be written as:
        U = a * P_1 + (1-a) * P_2 = [[a, 1-a], [1-a, a]]
    where a ∈ [0, 1] is learned via softmax over 2 logits.
    
    Returns:
        Tensor of shape (2, 2, 2) - [num_permutations, rows, cols]
    """
    perms = torch.tensor([
        [[1., 0.], [0., 1.]],  # P_1: Identity matrix
        [[0., 1.], [1., 0.]]   # P_2: Swap/exchange matrix
    ], dtype=torch.float32, device=device)
    return perms


def factorize_into_twos(n: int):
    """
    Factorize n into a product of factors, preferring 2s.
    
    This implements the factorization n = ∏_{k=1}^K i_k from Section 4.1.
    
    For n = 2^K (power of 2), returns [2, 2, ..., 2] (K times).
    This is the OPTIMAL factorization (Remark 4.5) because:
    - Each 2×2 factor only needs 2! = 2 permutation matrices
    - Total params scale as Σ i_k! = 2K (minimal)
    
    For non-powers of 2, includes the remaining factor.
    NOTE: Current implementation only fully supports powers of 2.
    
    Args:
        n: The number of residual streams to factorize
        
    Returns:
        List of factors [i_1, i_2, ..., i_K] such that n = ∏ i_k
    
    Example:
        factorize_into_twos(8) -> [2, 2, 2]  (8 = 2×2×2, K=3)
        factorize_into_twos(16) -> [2, 2, 2, 2]  (16 = 2^4, K=4)
    """
    if n == 1:
        return []
    
    factors = []
    remaining = n
    
    # Extract all factors of 2
    while remaining % 2 == 0:
        factors.append(2)
        remaining //= 2
    
    # Handle non-power-of-2 remainder
    # NOTE: Full support for arbitrary n is planned for future versions
    if remaining > 1:
        factors.append(remaining)
    
    return factors


def get_all_permutations(n: int):
    """
    Generate all n×n permutation matrices for arbitrary factor size.
    
    Used for factors i_k > 2 in the Kronecker decomposition.
    Returns n! matrices of shape (n, n).
    
    For n=2, prefer get_2x2_perm_matrices() for efficiency.
    For n=3, returns 6 matrices; for n=4, returns 24; etc.
    
    WARNING: This grows factorially! Only use for small n.
    
    Args:
        n: Size of permutation matrices
        
    Returns:
        Tensor of shape (n!, n, n)
    """
    assert n >= 1, "n must be a positive integer"

    perms = list(itertools.permutations(range(n)))
    index = torch.tensor(perms, dtype=torch.long, device="cpu")

    eye = torch.eye(n, dtype=torch.float32, device="cpu")
    perm_mats = eye[index]  # (n!, n, n)

    return perm_mats


# ============================================================================
# CACHES FOR PERMUTATION MATRICES
# ============================================================================
# These caches avoid recomputing permutation matrices on every forward pass.
# Keyed by (device_string,) to handle multi-GPU training.

perm_mats_2x2 = {}      # Cache for 2×2 permutation matrices {device: tensor}
perm_mats_general = {}  # Cache for general n×n perms {(n, device): tensor}

def get_cached_2x2_perms(device):
    """Get cached 2×2 permutation matrices for given device."""
    dev_key = str(device)
    if dev_key not in perm_mats_2x2:
        perm_mats_2x2[dev_key] = get_2x2_perm_matrices(device)
    return perm_mats_2x2[dev_key]


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
    Get initializer for KromHC layers plus expand/reduce functions.
    
    This is the main factory function for creating KromHC instances.
    Returns a partial function that can be called with layer-specific
    parameters (like layer_index, branch module, etc.)
    
    Args:
        num_streams: n, number of residual streams (must be power of 2)
        num_fracs: Number of fractions for frac-connections (typically 1)
        dim: Feature dimension C
        add_stream_embed: Whether to add learnable stream embeddings
        disable: Force disable hyper-connections (use standard residual)
        **kwargs: Additional arguments passed to KromHC constructor
        
    Returns:
        (init_fn, expand_fn, reduce_fn) tuple where init_fn creates KromHC layers
    """
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = KromHC if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, num_fracs = num_fracs, **kwargs)
    expand_reduce_fns = get_expand_reduce_stream_functions(num_streams, add_stream_embed = add_stream_embed, dim = dim, disable = disable)

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim = dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


# ============================================================================
# KROMHC: MAIN IMPLEMENTATION
# ============================================================================

class KromHC(BaseHyperConnections):
    """
    Kronecker-Product Manifold-Constrained Hyper-Connections (KromHC)
    
    This is the main contribution of the paper. KromHC addresses two issues
    with prior manifold-constrained hyper-connections:
    
    1. mHC: Uses Sinkhorn-Knopp algorithm which doesn't guarantee EXACT
       double stochasticity (see Figure 2 in paper - MAE accumulates)
       
    2. mHC-lite: Uses Birkhoff-von-Neumann theorem with n! permutation
       matrices - factorial parameter explosion (see Figure 3)
    
    KromHC SOLUTION (Section 4):
    ==========================================================================
    Instead of directly parameterizing the n×n residual matrix H^res_l,
    KromHC represents it as a Kronecker product of K smaller matrices:
    
        H^res_l = U^K_l ⊗ U^{K-1}_l ⊗ ... ⊗ U^1_l    (Equation 10)
    
    where n = i_1 × i_2 × ... × i_K (factorization of stream count)
    
    THEOREM 4.2 (Kronecker Closure): The Kronecker product of doubly
    stochastic matrices is itself doubly stochastic. This GUARANTEES
    exact double stochasticity without Sinkhorn iterations!
    
    For n = 2^K (power of 2 streams):
    - Each factor U^k_l is 2×2
    - Each 2×2 DS matrix needs only 2 parameters (convex combo of I and swap)
    - Total: K × 2 = 2log₂(n) parameters for H^res (vs n² for full matrix)
    
    TUCKER DECOMPOSITION INTERPRETATION (Section 4.1):
    ==========================================================================
    The Kronecker structure can be viewed as Tucker decomposition:
    
    1. Tensorize residual stream: X_l ∈ R^{n×C} -> X ∈ R^{i_1×...×i_K×C}
    2. Apply mode-k products: X ×_1 U^1 ×_2 U^2 ... ×_K U^K ×_{K+1} I_C
    3. Unfold back to matrix form
    
    This is equivalent to Equation 9:
        H^res_l X_l = mat(X ×_1 U^1 ×_2 U^2 ... ×_K U^K ×_{K+1} I_{C×C})
    
    PARAMETER COMPLEXITY (Figure 3, Section 4.3):
    ==========================================================================
    - mHC: O(n³C) - cubic in stream width
    - mHC-lite: O(nC × n!) - factorial explosion  
    - KromHC: O(n²C) - quadratic, with Σi_k! additional for residual coeffs
    
    For n=16, C=512:
    - mHC: ~2.1B params
    - mHC-lite: ~10^14 params (infeasible!)
    - KromHC: ~135M params
    
    INHERITANCE:
    ==========================================================================
    Extends BaseHyperConnections, inheriting:
    - Common initialization (fracs, norm, dropout, etc.)
    - _prepare_inputs(), _apply_mixing(), _finalize_outputs()
    - depth_connection(), decorate_branch(), forward()
    
    Implements:
    - _init_hyper_params(): Kronecker-structured parameter initialization
    - _compute_alpha_beta(): Kronecker product H^res construction
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
        Initialize KromHC layer.
        
        Args:
            num_residual_streams: n, number of streams (MUST be power of 2)
            dim: C, feature dimension
            branch: Optional branch module F(·) to wrap
            layer_index: Index for deterministic initialization (else random)
            channel_first: If True, expect (batch, dim, ...) not (batch, ..., dim)
            dropout: Dropout probability on output
            residual_transform: Transform on residual (e.g., for dim change)
            add_branch_out_to_residual: If False, disable depth connections
            num_input_views: Number of views for branch input (typically 1)
            depth_residual_fn: Function for combining output + residual
            num_fracs: Fractions for frac-connections extension
        """
        # =====================================================================
        # KRONECKER-SPECIFIC SETUP (must be done BEFORE super().__init__)
        # =====================================================================
        # Validate stream count - must be power of 2 for current implementation
        # (Remark 4.5: prime factorization with 2s is most efficient)
        assert num_residual_streams >= 2, '`num_residual_streams` must be at least 2'
        assert num_residual_streams & (num_residual_streams - 1) == 0, \
            f'`num_residual_streams` must be a power of 2, got {num_residual_streams}'
        
        # Factorize n into product of 2s: n = 2^K means K factors of size 2
        # This implements the tensorization from Section 4.1
        self.factors = factorize_into_twos(num_residual_streams)
        self.num_factors = len(self.factors)  # K in the paper
        
        # For each factor i_k, count permutation matrices needed (i_k!)
        # For 2×2 factors, that's just 2 (identity and swap)
        self.factor_perms = []
        self.total_res_coeffs = 0
        for f in self.factors:
            num_perms = math.factorial(f)
            self.factor_perms.append(num_perms)
            self.total_res_coeffs += num_perms  # Σ i_k! total coefficients
        
        # Pre-cache permutation matrices for non-2×2 factors
        for f in self.factors:
            if f > 2 and (f, "cpu") not in perm_mats_general:
                perm_mats_general[(f, "cpu")] = get_all_permutations(f).to("cpu")
        
        # =====================================================================
        # CALL BASE CLASS INIT (handles common setup + calls _init_hyper_params)
        # =====================================================================
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
        Initialize KromHC parameters: H^pre, H^res (Kronecker), H^post.
        
        This implements Section 5.1 initialization:
        - W^{res,k}_l, W^pre_l, W^post_l initialized to zero
        - b^pre_l, b^post_l: all -1 except one index = 1
        - α^pre_l, α^post_l = 0.01
        - For 2×2 factors: b^{res,k}_l = [0, -8]^T -> U^k_l ≈ I_2×2
        """
        n = self.num_residual_streams
        f = self.num_fracs
        v = self.num_input_views
        d = self.dim_per_frac
        
        # =====================================================================
        # H^pre INITIALIZATION (aggregation mapping)
        # =====================================================================
        # H^pre_l ∈ R^{v×n}: aggregates n streams into v for branch input
        # Init: all -1 except one stream at 1 (that stream dominates initially)
        init_alpha_pre = torch.ones((n * f, v * f)) * -1
        init_alpha_pre[self.init_residual_index, :] = 1.
        
        # =====================================================================
        # H^res INITIALIZATION (Kronecker residual mixing - Section 5.1)
        # =====================================================================
        # For 2×2 factors: b^{res,k}_l = [0, -8]^T
        # After softmax: a^k_l ≈ [1, 0] -> U^k_l ≈ I_2×2 -> H^res_l ≈ I_n×n
        init_alpha_res = torch.ones(self.total_res_coeffs * f) * -8
        
        # Set first permutation (identity) coefficient to 0 for each factor
        coeff_idx = 0
        for num_perms in self.factor_perms:
            init_alpha_res[coeff_idx] = 0.  # Identity permutation weight
            coeff_idx += num_perms
        
        # Combined static biases for H^pre and H^res
        self.static_alpha = nn.Parameter(cat([
            init_alpha_pre.view(-1),  # H^pre biases
            init_alpha_res            # H^res biases
        ], dim=-1))
        
        # =====================================================================
        # DYNAMIC WEIGHT MATRICES (W^pre, W^res,k in Equation 14)
        # =====================================================================
        # Projects normalized input to mapping coefficients
        self.dynamic_alpha_fn = nn.Parameter(
            torch.zeros(d * n, f * (self.total_res_coeffs + n * v))
        )
        
        # =====================================================================
        # LEARNABLE SCALARS (α^pre, α^res in Equation 14)
        # =====================================================================
        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
        # α^res_l: SHARED scalar for ALL factors k (Section 5.6 - shared outperforms unique)
        self.residual_scale = nn.Parameter(torch.ones(1) * 1e-2)
        
        # =====================================================================
        # H^post INITIALIZATION (distribution mapping)
        # =====================================================================
        if self.add_branch_out_to_residual:
            beta_init = torch.ones(n * f) * -1.
            beta_init[self.init_residual_index] = 1.
            self.static_beta = nn.Parameter(beta_init)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(d * n, f * n))
            self.h_post_scale = nn.Parameter(torch.ones(()) * 1e-2)
    
    def _compute_alpha_beta(self, normed: torch.Tensor, device: torch.device):
        """
        Compute H^pre, H^res (Kronecker), and H^post matrices.
        
        Implements Equation 14:
        - H^pre_l = σ(α^pre_l * x'_l @ W^pre_l + b^pre_l)
        - H^res_l = ⊗_{k=K}^1 U^k_l where U^k_l = Σ_m a^k_l(m) * P_m
        - H^post_l = 2σ(α^post_l * x'_l @ W^post_l + b^post_l)
        
        Args:
            normed: Normalized input x'_l = RMSNorm(flatten(X_l))
            device: Target device
        
        Returns:
            (alpha, beta) where alpha combines H^pre and H^res
        """
        n = self.num_residual_streams
        f = self.num_fracs
        v = self.num_input_views
        
        # =====================================================================
        # COMPUTE DYNAMIC WEIGHTS (x'_l @ W)
        # =====================================================================
        wc_weight = normed @ self.dynamic_alpha_fn
        
        # Separate H^pre and H^res dynamic coefficients
        psize = v * n * f
        dynamic_pre = wc_weight[..., :psize]
        dynamic_res = wc_weight[..., psize:]
        
        static_pre = self.static_alpha[:psize]
        static_res = self.static_alpha[psize:]
        
        # =====================================================================
        # BUILD H^pre (aggregation via sigmoid)
        # =====================================================================
        alpha_pre = self.pre_branch_scale * dynamic_pre + static_pre
        alpha_pre = rearrange(alpha_pre, '... (f s v) -> ... s f v', v=v, f=f)
        alpha_pre = alpha_pre.sigmoid()
        
        # =====================================================================
        # BUILD H^res (Kronecker-structured doubly stochastic)
        # =====================================================================
        alpha_res = self._build_kronecker_hres(dynamic_res, static_res, device)
        alpha_res = self.split_fracs(alpha_res)
        
        # Combine H^pre and H^res
        alpha = cat((alpha_pre, alpha_res), dim=-1)
        
        # =====================================================================
        # BUILD H^post (distribution via sigmoid * 2)
        # =====================================================================
        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = normed @ self.dynamic_beta_fn
            dc_weight = rearrange(dc_weight, '... (s f) -> ... s f', s=n)
            
            dynamic_beta = dc_weight * self.h_post_scale
            static_beta = rearrange(self.static_beta, '(s f) -> s f', s=n)
            
            beta = (dynamic_beta + static_beta).sigmoid() * 2
        
        return alpha, beta

    def _get_factor_perms(self, factor_size, device):
        """
        Get permutation matrices for a factor of given size.
        
        For factor_size=2, uses efficient 2×2 cache.
        For larger factors, uses general permutation generation.
        
        Args:
            factor_size: i_k, the size of this Kronecker factor
            device: Target device for the tensors
            
        Returns:
            Tensor of shape (i_k!, i_k, i_k) containing all permutations
        """
        if factor_size == 2:
            return get_cached_2x2_perms(device)
        else:
            dev_key = str(device)
            if (factor_size, dev_key) not in perm_mats_general:
                perm_mats_general[(factor_size, dev_key)] = get_all_permutations(factor_size).to(device)
            return perm_mats_general[(factor_size, dev_key)]

    def _build_kronecker_hres(self, dynamic_coeffs, static_coeffs, device):
        """
        Build H^res matrix using Kronecker product of factor matrices.
        
        This implements the core of KromHC (Equations 10, 14):
        
        For each factor k = 1, ..., K:
            1. Compute mixing coefficients: a^k_l = Softmax(α^res * x'W^{res,k} + b^{res,k})
            2. Build factor matrix: U^k_l = Σ_m a^k_l(m) * P_m
        
        Then compute: H^res_l = U^K_l ⊗ U^{K-1}_l ⊗ ... ⊗ U^1_l
        
        OPTIMIZATION FOR 2×2 FACTORS:
        For 2×2, the convex combination simplifies:
            U = a * I + (1-a) * swap = [[a, 1-a], [1-a, a]]
        where a = softmax([c1, c2])[0].
        
        This avoids materializing the permutation matrices entirely!
        
        Args:
            dynamic_coeffs: (..., Σ i_k!) tensor of x'W coefficients
            static_coeffs: (Σ i_k!,) tensor of bias coefficients
            device: Device for output tensor
            
        Returns:
            Tensor of shape (..., n, n) - the residual mixing matrix H^res_l
        """
        if len(self.factors) == 0:
            # Edge case: n=1 -> H^res is 1×1 identity
            return dynamic_coeffs.new_ones(dynamic_coeffs.shape[:-1] + (1, 1))
        
        # Apply shared residual scale α^res_l (Equation 14)
        # Note: Section 5.6 ablation shows shared α outperforms per-factor α^{res,k}
        combined_coeffs = self.residual_scale * dynamic_coeffs + static_coeffs
        
        # Check if all factors are 2×2 for optimized path
        all_2x2 = all(f == 2 for f in self.factors)
        
        if all_2x2:
            # =================================================================
            # OPTIMIZED 2×2 PATH (most common case for n = 2^K)
            # =================================================================
            batch_shape = combined_coeffs.shape[:-1]
            
            # Reshape to (batch..., K, 2) - one pair of logits per factor
            coeffs_reshaped = combined_coeffs.view(*batch_shape, self.num_factors, 2)

            # Softmax over 2 permutations -> convex combination weights
            weights = F.softmax(coeffs_reshaped, dim=-1)  # (..., K, 2)
            p = weights[..., 0]  # Weight for identity matrix
            
            # For 2×2 doubly stochastic: [[p, 1-p], [1-p, p]]
            # This is the convex combo: p * I + (1-p) * swap
            one_minus_p = 1.0 - p
            
            # Build all K factor matrices at once: (..., K, 2, 2)
            row0 = torch.stack([p, one_minus_p], dim=-1)
            row1 = torch.stack([one_minus_p, p], dim=-1)
            all_factor_matrices = torch.stack([row0, row1], dim=-2)
            
            # =================================================================
            # KRONECKER PRODUCT COMPUTATION
            # =================================================================
            # Iteratively compute: result = U^1 ⊗ U^2 ⊗ ... ⊗ U^K
            # After k iterations: result is (2^k × 2^k)
            
            result = all_factor_matrices[..., 0, :, :]  # U^1: (..., 2, 2)
            
            for k in range(1, self.num_factors):
                mat = all_factor_matrices[..., k, :, :]  # U^{k+1}: (..., 2, 2)
                
                # Kronecker product: result ⊗ mat
                # result ∈ R^{a×a}, mat ∈ R^{2×2} -> result ∈ R^{2a×2a}
                #
                # (A ⊗ B)_{(i1,j1), (i2,j2)} = A_{i1,i2} * B_{j1,j2}
                #
                # Broadcast multiplication via unsqueeze:
                result_exp = result.unsqueeze(-1).unsqueeze(-3)  # (..., a, 1, a, 1)
                mat_exp = mat.unsqueeze(-4).unsqueeze(-2)        # (..., 1, 2, 1, 2)
                kron = result_exp * mat_exp  # (..., a, 2, a, 2)
                
                # Reshape to (..., 2a, 2a)
                result = kron.reshape(*batch_shape, result.shape[-2] * 2, result.shape[-1] * 2)
            
            return result
            
        else:
            # =================================================================
            # GENERAL PATH (for factors > 2, future extension)
            # =================================================================
            factor_matrices = []
            coeff_idx = 0
            
            for k, (factor_size, num_perms) in enumerate(zip(self.factors, self.factor_perms)):
                # Extract coefficients for this factor
                factor_coeffs = combined_coeffs[..., coeff_idx:coeff_idx + num_perms]
                coeff_idx += num_perms
                
                # Get permutation matrices P_1, ..., P_{i_k!}
                perms = self._get_factor_perms(factor_size, device)
                
                # Softmax to get convex combination weights
                weights = F.softmax(factor_coeffs, dim=-1)
                
                # U^k_l = Σ_m a^k_l(m) * P_m (Equation 14)
                U_k = einsum(weights, perms, '... r, r i j -> ... i j')
                factor_matrices.append(U_k)
            
            # Compute Kronecker product iteratively
            result = factor_matrices[0]
            for mat in factor_matrices[1:]:
                result_exp = rearrange(result, '... a1 a2 -> ... a1 1 a2 1')
                mat_exp = rearrange(mat, '... b1 b2 -> ... 1 b1 1 b2')
                kron = result_exp * mat_exp
                result = rearrange(kron, '... a b c d -> ... (a b) (c d)')
            
            return result



# Attach factory functions as static methods for convenience
KromHC.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
KromHC.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)
