"""
mHC Analysis: Manifold-Constrained Hyper-Connections with Logging/Analysis

This module implements the mHC method with additional logging capabilities
for analyzing the behavior of the residual matrices during training.

This is used for the numerical stability analysis shown in Figure 2 of the
KromHC paper (Zhou et al., 2026), which demonstrates:
- mHC: MAE between column sum and 1 grows to ~0.05 across 24 layers
- mHC-lite: MAE is exactly 0 (exact double stochasticity)
- KromHC: MAE is exactly 0 (exact double stochasticity)

===============================================================================
PURPOSE
===============================================================================

This variant adds logging hooks to capture:
1. H^res BEFORE Sinkhorn-Knopp projection (raw learned logits)
2. H^res AFTER Sinkhorn-Knopp projection (approximately DS matrix)

These can be used to:
- Verify double stochasticity (row/column sum = 1)
- Analyze how far from DS the raw logits are
- Track the "error" introduced by finite SK iterations
- Compare different methods' stability properties

===============================================================================
USAGE
===============================================================================

analysis_mhc = MHCAnalysis(n, dim=d)
analysis_mhc.log_info = True  # Enable logging

# Forward pass
output = analysis_mhc(input)

# Access logged data
h_res_before = analysis_mhc.info["H_res_bef"]  # Before SK
h_res_after = analysis_mhc.info["H_res"]        # After SK

# Clear for next batch
analysis_mhc.clear_info()

===============================================================================
ANALYSIS METRICS (from Figure 2)
===============================================================================

For a doubly stochastic matrix H, we measure:
    MAE = mean(|column_sum(H) - 1|)

For a product of L matrices:
    MAE_L = mean(|column_sum(∏_{i=1}^L H^res_{L-i}) - 1|)

The KromHC paper shows:
- Standard mHC: MAE_L ≈ 0.05 for L=24 (accumulating error)
- mHC-lite: MAE_L = 0 for all L (exact DS)
- KromHC: MAE_L = 0 for all L (exact DS via Kronecker structure)

===============================================================================
IMPLEMENTATION NOTES
===============================================================================

This is nearly identical to mhc.py, but with:
1. self.log_info flag to control logging
2. self.info dict to store captured matrices
3. clear_info() method to reset between batches
4. Hooks in width_connection to capture H^res before/after SK

The SK algorithm and parameter complexity are identical to mHC.
"""

from __future__ import annotations
from typing import Callable

from functools import partial

import torch
from torch import nn, cat

from einops import rearrange, repeat

# Import shared utilities and parent class
from .utils import (
    exists,
    default,
    Residual,
    get_expand_reduce_stream_functions,
)
from .mhc import ManifoldConstrainedHyperConnections, sinkhorn_knopps


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
    """Get initializer for MHCAnalysis layers plus expand/reduce functions."""
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = MHCAnalysis if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, num_fracs = num_fracs, sinkhorn_iters = sinkhorn_iters, **kwargs)
    expand_reduce_fns = get_expand_reduce_stream_functions(num_streams, add_stream_embed = add_stream_embed, dim = dim, disable = disable)

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim = dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


# ============================================================================
# mHC ANALYSIS IMPLEMENTATION
# ============================================================================

class MHCAnalysis(ManifoldConstrainedHyperConnections):
    """
    Manifold-Constrained Hyper-Connections with Analysis/Logging capabilities.
    
    This variant of mHC includes hooks for capturing the residual matrices
    before and after Sinkhorn-Knopp projection, enabling analysis of:
    
    1. DOUBLE STOCHASTICITY ERROR:
       The Sinkhorn-Knopp algorithm only approximates DS matrices with finite
       iterations. We can measure how close the output is to being DS:
           error = ||H^res @ 1 - 1|| + ||H^res^T @ 1 - 1||
    
    2. ERROR ACCUMULATION (Figure 2 in KromHC paper):
       When multiplying H^res matrices across layers, small errors compound:
           ∏_{l=1}^L H^res_l should have column sums = 1
       But with finite SK iterations, the error grows with L.
    
    3. GRADIENT STABILITY:
       The gradient norms can be tracked to verify that DS constraints
       improve training stability (Figure 5 in KromHC paper).
    
    USAGE:
    ======
        layer = MHCAnalysis(n=4, dim=512)
        layer.log_info = True  # Enable capture
        
        output = layer(input)
        
        # Access captured matrices (as numpy arrays)
        h_res_before_sk = layer.info["H_res_bef"]  # (batch, n, n)
        h_res_after_sk = layer.info["H_res"]       # (batch, n, n) - approx DS
        
        # Compute column sum error
        col_sums = h_res_after_sk.sum(axis=-2)  # Should be close to 1
        mae = np.abs(col_sums - 1).mean()
        
        layer.clear_info()  # Reset for next batch
    
    INHERITANCE:
    ============
    Extends ManifoldConstrainedHyperConnections, adding:
    1. self.log_info boolean flag to toggle logging
    2. self.info dictionary for captured data
    3. clear_info() method
    4. Logging hooks in _compute_alpha_beta()
    5. I_AM_ANALYSIS_BLOCK marker for identification
    """
    
    def __init__(self, num_residual_streams, **kwargs):
        """
        Initialize MHCAnalysis layer.
        
        Same parameters as ManifoldConstrainedHyperConnections.
        
        Additional attributes:
            log_info (bool): When True, captures H^res matrices in self.info
            info (dict): Dictionary storing captured analysis data
            I_AM_ANALYSIS_BLOCK (bool): Marker for identifying analysis layers
        """
        super().__init__(num_residual_streams, **kwargs)
        
        # Analysis-specific attributes
        self.I_AM_ANALYSIS_BLOCK = True
        self.log_info = False
        self.info = {}
    
    def clear_info(self):
        """Clear captured analysis data."""
        self.info = {}
    
    def _compute_alpha_beta(self, normed: torch.Tensor, device: torch.device):
        """
        Compute H^pre, H^res (Sinkhorn-Knopp), and H^post with logging.
        
        When self.log_info is True, captures:
        - "H_res_bef": H^res matrix BEFORE Sinkhorn-Knopp
        - "H_res": H^res matrix AFTER Sinkhorn-Knopp
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
        
        # H^res: reshape for SK
        alpha_res = rearrange(alpha_res, '... f s g t -> ... f g s t')
        
        # Logging: capture BEFORE Sinkhorn-Knopp
        if self.log_info:
            _alpha = rearrange(alpha_res, 'b s f g n m -> (b s f g) n m')
            self.info["H_res_bef"] = _alpha.clone().detach().cpu().numpy()
        
        # Sinkhorn-Knopp projection
        alpha_res = sinkhorn_knopps(alpha_res, self.sinkhorn_iters)
        
        # Logging: capture AFTER Sinkhorn-Knopp
        if self.log_info:
            _alpha = rearrange(alpha_res, 'b s f g n m -> (b s f g) n m')
            self.info["H_res"] = _alpha.clone().detach().cpu().numpy()
        
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


MHCAnalysis.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
MHCAnalysis.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)
