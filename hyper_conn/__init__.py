"""
KromHC: Manifold-Constrained Hyper-Connections with Kronecker-Product Residual Matrices

This package implements hyper-connection variants for neural network residual streams,
as described in the paper:

    "KromHC: Manifold-Constrained Hyper-Connections with Kronecker-Product Residual Matrices"
    Zhou et al., 2026 (arXiv:2601.21579)

===============================================================================
PACKAGE OVERVIEW
===============================================================================

Hyper-Connections (HC) expand the standard residual connection from 1 to n
parallel streams, with learnable mixing matrices:

    X_{l+1} = H^res_l @ X_l + H^post_l^T @ F(H^pre_l @ X_l)

This package provides multiple variants with different trade-offs:

VARIANT COMPARISON TABLE (from Table 1 in paper):
=================================================
| Variant        | Exact DS? | Param Efficient | PyTorch Native |
|----------------|-----------|-----------------|----------------|
| HC             | No        | Yes             | Yes            |
| mHC            | ≈ Yes     | No (O(n³C))     | No (SK kernel) |
| mHC-lite       | Yes       | No (O(n!))      | Yes            |
| KromHC (ours)  | Yes       | Yes (O(n²C))    | Yes            |

DS = Doubly Stochastic (rows and columns sum to 1)

===============================================================================
AVAILABLE IMPLEMENTATIONS
===============================================================================

1. hyper_connections.py - HyperConnections
   Original HC with unconstrained mixing matrices.
   Can cause training instability in deep networks.
   
2. mhc.py - ManifoldConstrainedHyperConnections
   Uses Sinkhorn-Knopp algorithm to project H^res onto Birkhoff polytope.
   Approximately doubly stochastic. O(n³C) params.
   
3. mhc_lite.py - MHCLite
   Uses Birkhoff-von-Neumann theorem: H^res = Σ a_k * P_k (convex combo of perms).
   Exactly doubly stochastic but O(n!) permutation matrices needed.
   
4. Kromhc.py - KromHC [MAIN CONTRIBUTION]
   Uses Kronecker products: H^res = U^K ⊗ ... ⊗ U^1.
   Exactly doubly stochastic with only O(n²C) params.
   For n=2^K, uses K 2×2 factors with just 2 perms each.

5. mhc_analysis.py - MHCAnalysis
   mHC variant with logging hooks for analyzing DS error accumulation.
   Used for Figure 2 in the paper.

6. residuals.py - GatedResidual, GRUGatedResidual
   Alternative gating mechanisms for single-stream residuals.

===============================================================================
USAGE PATTERNS
===============================================================================

PATTERN 1: Factory functions (recommended)
------------------------------------------
    from hyper_conn import kromhc_get_init_and_expand_reduce_stream_functions
    
    # Get initializer + expand/reduce functions
    init_hc, expand_fn, reduce_fn = kromhc_get_init_and_expand_reduce_stream_functions(
        num_streams=4,
        dim=512
    )
    
    # Expand input to streams
    x = expand_fn(x)  # (batch, seq, dim) -> (batch*4, seq, dim)
    
    # Create and apply HC layer
    hc_layer = init_hc(layer_index=0)
    branch_in, add_residual = hc_layer(x)
    branch_out = attention(branch_in)
    x = add_residual(branch_out)
    
    # Reduce back to single stream
    x = reduce_fn(x)  # (batch*4, seq, dim) -> (batch, seq, dim)

PATTERN 2: Unified interface
----------------------------
    from hyper_conn import hyper_conn_init_func
    
    # Choose variant by name
    init_hc, expand_fn, reduce_fn = hyper_conn_init_func(
        hyper_conn_type="kromhc",  # or "hc", "mhc", "mhc_lite", "analysis", "none"
        hyper_conn_n=4
    )

PATTERN 3: Direct class usage
-----------------------------
    from hyper_conn import KromHC
    
    layer = KromHC(
        num_residual_streams=4,
        dim=512,
        layer_index=0
    )
    
    branch_input, add_residual_fn = layer(residuals)
    branch_output = my_attention(branch_input)
    output = add_residual_fn(branch_output)

PATTERN 4: Decorator pattern
----------------------------
    from hyper_conn import KromHC
    
    layer = KromHC(num_residual_streams=4, dim=512)
    wrapped_attn = layer.decorate_branch(attention_layer)
    
    output = wrapped_attn(residuals)  # Automatically applies HC

===============================================================================
KEY CONCEPTS FROM THE PAPER
===============================================================================

DOUBLY STOCHASTIC MATRICES (Section 2):
- Rows sum to 1: H @ 1 = 1
- Columns sum to 1: H^T @ 1 = 1
- Non-negative entries: H >= 0
- Preserves feature mean when multiplied
- Spectral norm bounded by 1

BIRKHOFF POLYTOPE B_n:
- Set of all n×n doubly stochastic matrices
- Vertices are the n! permutation matrices (BvN theorem)
- mHC projects onto this via Sinkhorn-Knopp
- mHC-lite uses convex combinations of vertices
- KromHC uses Kronecker products of smaller DS matrices

KRONECKER CLOSURE (Theorem 4.2):
- If A ∈ B_m and B ∈ B_n, then A ⊗ B ∈ B_{mn}
- Kronecker product of DS matrices is DS
- Enables decomposition: H^res = U^K ⊗ ... ⊗ U^1
- For n = 2^K: K 2×2 factors, each with 2 perms

PARAMETER COMPLEXITY:
- mHC: O(n³C) - cubic in stream width
- mHC-lite: O(nC × n!) - factorial explosion
- KromHC: O(n²C + Σ i_k!) ≈ O(n²C) for small factors

===============================================================================
"""

# ============================================================================
# SHARED UTILITIES (Phase 1 Refactoring)
# ============================================================================
# These utilities are shared across all hyper-connection variants.
# Extracted to utils.py to eliminate ~1000 lines of duplicated code.

from .utils import (
    # Helper functions
    exists,
    divisible_by,
    default,
    add,
    # Shared classes
    RMSNorm,
    Residual,
    StreamEmbed,
    AttentionPoolReduceStream,
    # Shared factory function
    get_expand_reduce_stream_functions,
    # Debugging utilities
    print_trainable_parameters,
    count_parameters,
)

# ============================================================================
# BASE CLASS (Phase 2 Refactoring)
# ============================================================================
# Abstract base class for all hyper-connection variants.
# Provides shared implementation for ~70% of code that was previously duplicated.

from .base import BaseHyperConnections

# ============================================================================
# HYPER-CONNECTIONS (Original, unconstrained)
# ============================================================================

from .hyper_connections import (
    HyperConnections,
    get_init_and_expand_reduce_stream_functions,
)

# ============================================================================
# mHC: MANIFOLD-CONSTRAINED (Sinkhorn-Knopp)
# ============================================================================
# Uses iterative SK algorithm to approximately project onto Birkhoff polytope.
# O(n³C) parameter complexity. Requires custom kernels for efficiency.

from .mhc import (
    ManifoldConstrainedHyperConnections,
    get_init_and_expand_reduce_stream_functions as mc_get_init_and_expand_reduce_stream_functions,
)

# ============================================================================
# mHC-lite: BIRKHOFF-VON-NEUMANN DECOMPOSITION
# ============================================================================
# Exact doubly stochastic via convex combination of permutation matrices.
# O(nC × n!) parameter complexity - factorial explosion limits scalability.

from .mhc_lite import (
    MHCLite,
    get_init_and_expand_reduce_stream_functions as mhclite_get_init_and_expand_reduce_stream_functions,
)

# ============================================================================
# mHC ANALYSIS: LOGGING VARIANT
# ============================================================================
# Same as mHC but with hooks to capture H^res matrices for analysis.
# Used for generating Figure 2 (DS error accumulation analysis).

from .mhc_analysis import (
    MHCAnalysis,
    get_init_and_expand_reduce_stream_functions as mhc_analysis_get_init_and_expand_reduce_stream_functions,
)

# ============================================================================
# KromHC: KRONECKER-PRODUCT RESIDUAL MATRICES (Main Contribution)
# ============================================================================
# Exact doubly stochastic via Kronecker products of small DS matrices.
# O(n²C) parameter complexity - efficient and exact!
# For n = 2^K, uses K 2×2 factors with 2 permutations each.

from .Kromhc import (
    KromHC,
    get_init_and_expand_reduce_stream_functions as kromhc_get_init_and_expand_reduce_stream_functions,
)

# ============================================================================
# TBP-mHC: TRANSPORTATION BIRKHOFF POLYTOPE
# ============================================================================
# Exact doubly stochastic via transportation-polytope construction with (n-1)² free params.
# Works for any n >= 2 (no power-of-2 restriction). O((n-1)²) params for H^res.

from .tbp_mhc import (
    TBP_MHC,
    TransportationBirkhoff,
    get_init_and_expand_reduce_stream_functions as tbp_get_init_and_expand_reduce_stream_functions,
)
LSB_MHC = TBP_MHC
SequentialBirkhoff = TransportationBirkhoff
lsb_get_init_and_expand_reduce_stream_functions = tbp_get_init_and_expand_reduce_stream_functions

# ============================================================================
# RTBP-mHC: RECURSIVE TRANSPORTATION BIRKHOFF POLYTOPE
# ============================================================================
# Exact doubly stochastic via recursive transportation-polytope splitting.
# Works for any n >= 2 and batches the four even-case child problems together.

from .rtbpHC import (
    RTBP_MHC,
    RTBPHC,
    RecursiveTransportBirkhoff,
    get_init_and_expand_reduce_stream_functions as rtbp_get_init_and_expand_reduce_stream_functions,
)

# ============================================================================
# SRTBP-mHC: SCALED RECURSIVE TRANSPORTATION BIRKHOFF POLYTOPE
# ============================================================================
# Exact doubly stochastic via recursive transportation-polytope splitting, with
# the STBP-style scaled sigmoid applied to every local interval choice.

from .srtbpHC import (
    SRTBP_MHC,
    SRTBPHC,
    ScaledRecursiveTransportBirkhoff,
    get_init_and_expand_reduce_stream_functions as srtbp_get_init_and_expand_reduce_stream_functions,
)

# ============================================================================
# RTBP2N-mHC: POWER-OF-2 RECURSIVE TRANSPORTATION BIRKHOFF POLYTOPE
# ============================================================================
# Exact doubly stochastic via the recursive transport construction restricted to
# n = 2^L, using balanced binary split trees instead of sequential SplitVector.

from .rtbp2n_HC import (
    RTBP2N_MHC,
    RTBP2NHC,
    PowerOfTwoRecursiveTransportBirkhoff,
    get_init_and_expand_reduce_stream_functions as rtbp2n_get_init_and_expand_reduce_stream_functions,
)

# ============================================================================
# SRTBP2N-mHC: SCALED POWER-OF-2 RECURSIVE TRANSPORTATION BIRKHOFF POLYTOPE
# ============================================================================
# Exact doubly stochastic via the power-of-2 recursive transport construction,
# with the STBP-style scaled sigmoid applied to every local interval choice.

from .srtbp2n_mhc import (
    SRTBP2N_MHC,
    SRTBP2NHC,
    ScaledPowerOfTwoRecursiveTransportBirkhoff,
    get_init_and_expand_reduce_stream_functions as srtbp2n_get_init_and_expand_reduce_stream_functions,
)

# ============================================================================
# ORTBP2N-mHC: OPTIMIZED SCALED POWER-OF-2 RECURSIVE TRANSPORT BIRKHOFF
# ============================================================================
# Same scaled power-of-2 recursive transport chart as SRTBP2N, but with the
# residual-chart parameters factored so optimizer code can place them in a
# dedicated zero-weight-decay group.

from .ortbp2n_mhc import (
    ORTBP2N_MHC,
    ORTBP2NHC,
    OptimizedPowerOfTwoRecursiveTransportBirkhoff,
    get_init_and_expand_reduce_stream_functions as ortbp2n_get_init_and_expand_reduce_stream_functions,
)

# ============================================================================
# DORTBP2N-mHC: DEPTH-WEIGHTED OPTIMIZED POWER-OF-2 RECURSIVE TRANSPORT BIRKHOFF
# ============================================================================
# Same chart and parameter layout as ORTBP2N, with a static per-coordinate gain
# g_d = p^d (mean-normalized) that weights each local transport choice by its
# recursion depth, biasing the chart toward localized rearrangements.

from .dortbp2n_mhc import (
    DORTBP2N_MHC,
    DORTBP2NHC,
    DepthWeightedPowerOfTwoRecursiveTransportBirkhoff,
    build_chart_gain,
    chart_depth_labels,
    get_init_and_expand_reduce_stream_functions as dortbp2n_get_init_and_expand_reduce_stream_functions,
)

# ============================================================================
# MSRTBP2N-mHC: MARGINED SCALED POWER-OF-2 RECURSIVE TRANSPORT BIRKHOFF
# ============================================================================
# Exact doubly stochastic via the power-of-2 recursive transport construction,
# with an MSTBP-style margin rho keeping local choices away from interval edges.

from .msrtbp2n_mhc import (
    MSRTBP2N_MHC,
    MSRTBP2NHC,
    MarginedScaledPowerOfTwoRecursiveTransportBirkhoff,
    get_init_and_expand_reduce_stream_functions as msrtbp2n_get_init_and_expand_reduce_stream_functions,
)

# ============================================================================
# AMSRTBP2N-mHC: AVERAGED MARGINED SCALED POWER-OF-2 RECURSIVE TRANSPORT
# ============================================================================
# Averaged MSRTBP2N: mixture of permutation charts built from the power-of-2
# margined scaled recursive transport parameterization.

from .amsrtbp2n_mhc import (
    AMSRTBP2N_MHC,
    get_init_and_expand_reduce_stream_functions as amsrtbp2n_get_init_and_expand_reduce_stream_functions,
)

# ============================================================================
# STBP-mHC: SCALED TRANSPORT BIRKHOFF POLYTOPE
# ============================================================================
# Exact doubly stochastic via scaled transport parameterization. Same as TBP-mHC
# but with X[i,j] = L_ij + (U_ij - L_ij) * sigmoid(β*t/(U_ij - L_ij + ε)).

from .stbp_mhc import (
    STBP_MHC,
    ScaledTransportBirkhoff,
    get_init_and_expand_reduce_stream_functions as stbp_get_init_and_expand_reduce_stream_functions,
)

# ============================================================================
# MSTBP-mHC: MARGINED SCALED TRANSPORT BIRKHOFF POLYTOPE
# ============================================================================
# Exact doubly stochastic via margined scaled transport. Same as STBP-mHC but
# with fraction in [ρ, 1-ρ] for improved gradient flow: X[i,j] = L + (U-L)*(ρ+(1-2ρ)*sigmoid(...)).

from .mstbp_mhc import (
    MSTBP_MHC,
    MarginedScaledTransportBirkhoff,
    get_init_and_expand_reduce_stream_functions as mstbp_get_init_and_expand_reduce_stream_functions,
)

# ============================================================================
# ATBP-mHC: AVERAGED TRANSPORTATION BIRKHOFF POLYTOPE
# ============================================================================
# Exact doubly stochastic via mixture of TBP permutation charts. Reduces order bias.

from .atbp_mhc import (
    ATBP_MHC,
    ALSB_MHC,
    get_init_and_expand_reduce_stream_functions as atbp_get_init_and_expand_reduce_stream_functions,
)
alsb_get_init_and_expand_reduce_stream_functions = atbp_get_init_and_expand_reduce_stream_functions

# ============================================================================
# ASTBP-mHC: AVERAGED SCALED TRANSPORT BIRKHOFF POLYTOPE
# ============================================================================
# Averaged STBP: mixture of permutation charts with scaled transport formula.

from .astbp_mhc import (
    ASTBP_MHC,
    get_init_and_expand_reduce_stream_functions as astbp_get_init_and_expand_reduce_stream_functions,
)

# ============================================================================
# AMSTBP-mHC: AVERAGED MARGINED SCALED TRANSPORT BIRKHOFF POLYTOPE
# ============================================================================
# Averaged MSTBP: mixture of permutation charts with margined scaled transport.

from .amstbp_mhc import (
    AMSTBP_MHC,
    get_init_and_expand_reduce_stream_functions as amstbp_get_init_and_expand_reduce_stream_functions,
)

# ============================================================================
# LMAMSTBP-mHC: LAZYFIED-MINORIZED AVERAGED MSTBP
# ============================================================================
# AMSTBP + H_n = (1-λ-μ)*I + λ*X_n + μ*(J/n) with learnable λ, μ per layer.

from .lmamstbp_mhc import (
    LMAMSTBP_MHC,
    get_init_and_expand_reduce_stream_functions as lmamstbp_get_init_and_expand_reduce_stream_functions,
)

# ============================================================================
# LTBP-mHC: LINEAR TRANSPORT BIRKHOFF POLYTOPE
# ============================================================================
# LSB variant with linear X[i,j]=L+(U-L)*t, t∈[0,1] as explicit params. Clamp after optimizer.step().

from .ltbp_mhc import (
    LTBP_MHC,
    get_init_and_expand_reduce_stream_functions as ltbp_get_init_and_expand_reduce_stream_functions,
)

# ============================================================================
# ALTBP-mHC: AVERAGED LINEAR TRANSPORT BIRKHOFF POLYTOPE
# ============================================================================
# Averaged LTBP: mixture of permutation charts with linear formula. Clamp after optimizer.step().

from .altbp_mhc import (
    ALTBP_MHC,
    get_init_and_expand_reduce_stream_functions as altbp_get_init_and_expand_reduce_stream_functions,
)

# ============================================================================
# LMALTBP-mHC: LAZYFIED-MINORIZED AVERAGED LINEAR TBP
# ============================================================================
# ALTBP + H_n = (1-λ-μ)*I + λ*X_n + μ*(J/n) with learnable λ, μ per layer.

from .lmaltbp_mhc import (
    LMALTBP_MHC,
    get_init_and_expand_reduce_stream_functions as lmaltbp_get_init_and_expand_reduce_stream_functions,
)

def clamp_ltbp_params(model):
    """Clamp LTBP/ALTBP/LMALTBP birkhoff_params to [0,1] after optimizer.step(). Call when hyper_conn_type in ["ltbp_mhc","altbp_mhc","lmaltbp_mhc"]."""
    for m in model.modules():
        if isinstance(m, (LTBP_MHC, ALTBP_MHC)):
            m.clamp_birkhoff_params()


# ============================================================================
# TESTS (Phase 5 - run with: python -m hyper_conn.tests)
# ============================================================================

from .tests import run_all_tests


# ============================================================================
# UNIFIED INTERFACE
# ============================================================================

# Flag to prevent repeated logging of configuration
flag = False

def hyper_conn_init_func(hyper_conn_type: str, hyper_conn_n: int, atbp_permutations=None, alsb_permutations=None, astbp_permutations=None, amstbp_permutations=None, altbp_permutations=None, lmamstbp_lambda_init=None, lmamstbp_mu_init=None, lmaltbp_lambda_init=None, lmaltbp_mu_init=None, ortbp_log_stats=False, ortbp_depth_gain_base=1.0, ortbp_depth_gains=None):
    """
    Unified factory function for all hyper-connection variants.
    
    This provides a single entry point to select and initialize any
    hyper-connection variant by name, making it easy to switch between
    methods in experiments.
    
    Args:
        hyper_conn_type: One of:
            - "none": Standard residual (no hyper-connections)
            - "hc": Original HyperConnections (unconstrained)
            - "mhc": Manifold-Constrained HC (Sinkhorn-Knopp)
            - "mhc_lite": mHC-lite (Birkhoff-von-Neumann)
            - "analysis": mHC with logging hooks
            - "kromhc": KromHC (Kronecker products) [RECOMMENDED]
            - "tbp_mhc": TBP-mHC (Transportation Birkhoff Polytope - exact DS)
            - "rtbp_mhc": RTBP-mHC (Recursive transportation Birkhoff - exact DS)
            - "srtbp_mhc": SRTBP-mHC (Scaled recursive transport - exact DS)
            - "rtbp2n_mhc": RTBP2N-mHC (Power-of-2 recursive transport - exact DS)
            - "srtbp2n_mhc": SRTBP2N-mHC (Scaled power-of-2 recursive transport - exact DS)
            - "ortbp2n_mhc": ORTBP2N-mHC (Optimizer-friendly SRTBP2N with split residual chart params)
            - "dortbp2n_mhc": DORTBP2N-mHC (ORTBP2N with depth-weighted transport chart)
            - "msrtbp2n_mhc": MSRTBP2N-mHC (Margined scaled power-of-2 recursive transport - exact DS)
            - "amsrtbp2n_mhc": AMSRTBP2N-mHC (Averaged margined scaled power-of-2 recursive transport)
            - "stbp_mhc": STBP-mHC (Scaled Transport Birkhoff - exact DS)
            - "mstbp_mhc": MSTBP-mHC (Margined Scaled Transport Birkhoff - exact DS)
            - "atbp_mhc": ATBP-mHC (Averaged TBP - mixture of permutation charts)
            - "astbp_mhc": ASTBP-mHC (Averaged STBP - mixture of permutation charts)
            - "amstbp_mhc": AMSTBP-mHC (Averaged MSTBP - mixture of permutation charts)
            - "lmamstbp_mhc": LMAMSTBP-mHC (Lazyfied-minorized AMSTBP - spectral gap control)
            - "ltbp_mhc": LTBP-mHC (Linear TBP - explicit t params, clamp after optimizer step)
            - "altbp_mhc": ALTBP-mHC (Averaged LTBP - mixture of permutation charts)
            - "lmaltbp_mhc": LMALTBP-mHC (Lazyfied-minorized ALTBP - spectral gap control)
            
        hyper_conn_n: Number of residual streams (n in paper)
            - For KromHC, should be a power of 2 (2, 4, 8, 16, ...)
            - For mHC-lite, keep small (n! grows fast!)
            
    Returns:
        (init_fn, expand_fn, reduce_fn) tuple:
            - init_fn: Partial function to create HC layer instances
            - expand_fn: Expands input to n streams
            - reduce_fn: Reduces n streams back to 1
            
    Example:
        >>> init_hc, expand, reduce = hyper_conn_init_func("kromhc", 4)
        >>> x = expand(x)  # (batch, seq, dim) -> (batch*4, seq, dim)
        >>> layer = init_hc(dim=512, layer_index=0)
        >>> branch_in, add_res = layer(x)
        >>> branch_out = my_attention(branch_in)
        >>> x = add_res(branch_out)
        >>> x = reduce(x)  # (batch*4, seq, dim) -> (batch, seq, dim)
    """
    global flag
    if not flag:
        print(f"HYPER_CONN: USING {hyper_conn_type} with {hyper_conn_n} streams")
        flag = True

    if hyper_conn_type == "none":
        # Standard residual - disable hyper-connections
        return get_init_and_expand_reduce_stream_functions(hyper_conn_n, disable = True)
    
    elif hyper_conn_type == "hc":
        # Original HyperConnections (unconstrained H^res)
        return get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    
    elif hyper_conn_type == "mhc":
        # Manifold-Constrained HC (Sinkhorn-Knopp projection)
        return mc_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    
    elif hyper_conn_type == "mhc_lite":
        # mHC-lite (Birkhoff-von-Neumann convex combination)
        return mhclite_get_init_and_expand_reduce_stream_functions(hyper_conn_n)

    elif hyper_conn_type == "analysis":
        # mHC with analysis logging hooks
        return mhc_analysis_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    
    elif hyper_conn_type == "kromhc":
        # KromHC (Kronecker-product residual matrices) [RECOMMENDED]
        return kromhc_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    
    elif hyper_conn_type in {"tbp_mhc", "lsb_mhc"}:
        # TBP-mHC (Transportation Birkhoff Polytope - exact DS with (n-1)² params)
        return tbp_get_init_and_expand_reduce_stream_functions(hyper_conn_n)

    elif hyper_conn_type in {"rtbp_mhc", "rtbphc"}:
        # RTBP-mHC (Recursive transportation Birkhoff - exact DS with (n-1)² params)
        return rtbp_get_init_and_expand_reduce_stream_functions(hyper_conn_n)

    elif hyper_conn_type in {"srtbp_mhc", "srtbphc"}:
        # SRTBP-mHC (Scaled recursive transport - exact DS with (n-1)² params)
        return srtbp_get_init_and_expand_reduce_stream_functions(hyper_conn_n)

    elif hyper_conn_type in {"rtbp2n_mhc", "rtbp2n_hc"}:
        # RTBP2N-mHC (Power-of-2 recursive transport - exact DS with (n-1)² params)
        return rtbp2n_get_init_and_expand_reduce_stream_functions(hyper_conn_n)

    elif hyper_conn_type in {"srtbp2n_mhc", "srtbp2n_hc"}:
        # SRTBP2N-mHC (Scaled power-of-2 recursive transport - exact DS with (n-1)² params)
        return srtbp2n_get_init_and_expand_reduce_stream_functions(hyper_conn_n)

    elif hyper_conn_type in {"ortbp2n_mhc", "ortbp2n_hc"}:
        # ORTBP2N-mHC (Optimizer-friendly SRTBP2N with split residual chart params)
        return ortbp2n_get_init_and_expand_reduce_stream_functions(
            hyper_conn_n,
            log_stats=ortbp_log_stats,
        )

    elif hyper_conn_type in {"dortbp2n_mhc", "dortbp2n_hc"}:
        # DORTBP2N-mHC (ORTBP2N with depth-weighted p^d gain on the transport chart)
        return dortbp2n_get_init_and_expand_reduce_stream_functions(
            hyper_conn_n,
            log_stats=ortbp_log_stats,
            depth_gain_base=ortbp_depth_gain_base,
            depth_gains=ortbp_depth_gains,
        )

    elif hyper_conn_type in {"msrtbp2n_mhc", "msrtbp2n_hc", "msrtdp2n_mhc", "msrtdp2n_hc"}:
        # MSRTBP2N-mHC (Margined scaled power-of-2 recursive transport - exact DS with (n-1)² params)
        return msrtbp2n_get_init_and_expand_reduce_stream_functions(hyper_conn_n)

    elif hyper_conn_type == "amsrtbp2n_mhc":
        # AMSRTBP2N-mHC (Averaged MSRTBP2N - mixture of permutation charts)
        return amsrtbp2n_get_init_and_expand_reduce_stream_functions(
            hyper_conn_n, permutations=amstbp_permutations or astbp_permutations or atbp_permutations or alsb_permutations
        )
    
    elif hyper_conn_type == "stbp_mhc":
        # STBP-mHC (Scaled Transport Birkhoff - exact DS with (n-1)² params)
        return stbp_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    
    elif hyper_conn_type == "mstbp_mhc":
        # MSTBP-mHC (Margined Scaled Transport Birkhoff - exact DS with (n-1)² params)
        return mstbp_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    
    elif hyper_conn_type in {"atbp_mhc", "alsb_mhc"}:
        # ATBP-mHC (Averaged TBP - mixture of permutation charts)
        return atbp_get_init_and_expand_reduce_stream_functions(
            hyper_conn_n, permutations=atbp_permutations or alsb_permutations
        )
    
    elif hyper_conn_type == "astbp_mhc":
        # ASTBP-mHC (Averaged STBP - mixture of permutation charts)
        return astbp_get_init_and_expand_reduce_stream_functions(
            hyper_conn_n, permutations=astbp_permutations
        )
    
    elif hyper_conn_type == "amstbp_mhc":
        # AMSTBP-mHC (Averaged MSTBP - mixture of permutation charts)
        return amstbp_get_init_and_expand_reduce_stream_functions(
            hyper_conn_n, permutations=amstbp_permutations
        )
    
    elif hyper_conn_type == "lmamstbp_mhc":
        # LMAMSTBP-mHC (Lazyfied-minorized AMSTBP - λ, μ control spectral gap)
        return lmamstbp_get_init_and_expand_reduce_stream_functions(
            hyper_conn_n,
            permutations=amstbp_permutations,
            lambda_init=lmamstbp_lambda_init,
            mu_init=lmamstbp_mu_init,
        )
    
    elif hyper_conn_type == "ltbp_mhc":
        # LTBP-mHC (Linear TBP - explicit t params, clamp after optimizer step)
        return ltbp_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    
    elif hyper_conn_type == "altbp_mhc":
        # ALTBP-mHC (Averaged LTBP - mixture of permutation charts)
        return altbp_get_init_and_expand_reduce_stream_functions(
            hyper_conn_n, permutations=altbp_permutations or atbp_permutations or alsb_permutations
        )
    
    elif hyper_conn_type == "lmaltbp_mhc":
        # LMALTBP-mHC (Lazyfied-minorized ALTBP - λ, μ control spectral gap)
        return lmaltbp_get_init_and_expand_reduce_stream_functions(
            hyper_conn_n,
            permutations=altbp_permutations or atbp_permutations or alsb_permutations,
            lambda_init=lmaltbp_lambda_init,
            mu_init=lmaltbp_mu_init,
        )
    
    else:
        raise ValueError(f"Invalid hyper connection type: {hyper_conn_type}")
