"""
BaseHyperConnections: Abstract Base Class for All Hyper-Connection Variants

This module provides the shared foundation for all hyper-connection implementations.
The base class encapsulates ~70% of the code that was previously duplicated across
6 different variant files.

===============================================================================
DESIGN PHILOSOPHY
===============================================================================

The base class handles STRUCTURAL concerns that are identical across all variants:
- Common initialization (fracs, channel_first, dropout, norm, etc.)
- Input preparation (rearrange, normalize)
- Output finalization (rearrange back)
- Mixing application (einsum)
- depth_connection, decorate_branch, forward (100% shared)

Subclasses implement VARIANT-SPECIFIC logic:
1. _init_hyper_params(): Initialize H^pre, H^res, H^post parameters
2. _compute_alpha_beta(): Compute the mixing matrices

===============================================================================
ARCHITECTURAL DECISION: H^pre/H^post IN SUBCLASSES
===============================================================================

H^pre and H^post are intentionally kept in model-specific subclasses (not the
base class) because:

1. Different variants use different activations (tanh for HC, sigmoid for mHC/*)
2. Parameter shapes may vary (vector vs matrix parameterization)
3. Initialization strategies differ between variants
4. This keeps the base class minimal and assumption-free

===============================================================================
USAGE PATTERN
===============================================================================

class MyHyperConnection(BaseHyperConnections):
    def __init__(self, num_streams, **kwargs):
        # Pre-compute any variant-specific attributes BEFORE super().__init__
        self.my_special_thing = compute_something(num_streams)
        super().__init__(num_streams, **kwargs)
    
    def _init_hyper_params(self):
        # Initialize ALL parameters: H^pre, H^res, H^post
        self.static_alpha_pre = nn.Parameter(...)
        self.static_alpha_res = nn.Parameter(...)
        self.static_beta = nn.Parameter(...)  # if add_branch_out_to_residual
        ...
    
    def _compute_alpha_beta(self, normed, device):
        # Compute mixing matrices from parameters
        alpha = ...  # Combined H^pre and H^res
        beta = ...   # H^post (or None)
        return alpha, beta

===============================================================================
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable
from random import randrange

import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module
from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange

from .utils import (
    exists,
    divisible_by,
    default,
    add,
    RMSNorm,
)


class BaseHyperConnections(Module, ABC):
    """
    Abstract base class encapsulating shared hyper-connection functionality.
    
    This class implements the Template Method pattern:
    - The overall algorithm structure is defined in width_connection()
    - Subclasses customize behavior via _init_hyper_params() and _compute_alpha_beta()
    
    HYPER-CONNECTION EQUATION (from the papers):
        X_{l+1} = H^res_l @ X_l + H^post_l^T @ F(H^pre_l @ X_l)
    
    where:
        - X_l ∈ R^{n×C}: n residual streams, each of dimension C
        - H^res_l ∈ R^{n×n}: residual mixing matrix
        - H^pre_l ∈ R^{v×n}: aggregates n streams into v for branch F(·)
        - H^post_l ∈ R^{1×n}: distributes branch output to n streams
        - F(·): branch function (attention, FFN, etc.)
    
    SHARED FUNCTIONALITY (implemented here):
        - __init__: Common initialization (fracs, norms, dropout, etc.)
        - _prepare_inputs: Channel handling, frac splitting, normalization
        - _apply_mixing: einsum for stream mixing
        - _finalize_outputs: Rearrange back to expected format
        - width_connection: Template method combining the above
        - depth_connection: Add branch output back to residuals
        - decorate_branch: Decorator pattern for wrapping branches
        - forward: Main entry point
    
    VARIANT-SPECIFIC FUNCTIONALITY (implemented by subclasses):
        - _init_hyper_params: Initialize ALL parameters (H^pre, H^res, H^post)
        - _compute_alpha_beta: Compute mixing matrices from parameters
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
        depth_residual_fn: Callable = add,
        num_fracs: int = 1,
    ):
        """
        Initialize base hyper-connection layer.
        
        Args:
            num_residual_streams: n, number of parallel residual streams
            dim: C, feature dimension
            branch: Optional branch module F(·) to wrap
            layer_index: For deterministic initialization (default: random)
            channel_first: If True, expect (batch, dim, ...) layout
            dropout: Dropout probability for regularization
            residual_transform: Optional transform on residuals
            add_branch_out_to_residual: If True, add branch output to residuals
            num_input_views: v, number of views for branch input
            depth_residual_fn: Function for combining output + residual
            num_fracs: For frac-connections extension (typically 1)
        """
        super().__init__()
        
        # =====================================================================
        # SHARED INITIALIZATION (identical across all variants)
        # =====================================================================
        
        # Store configuration
        self.branch = branch
        self.num_residual_streams = num_residual_streams
        self.num_fracs = num_fracs
        self.has_fracs = num_fracs > 1
        self.num_input_views = num_input_views
        self.channel_first = channel_first
        self.add_branch_out_to_residual = add_branch_out_to_residual
        
        # Fraction handling (for frac-connections extension)
        self.split_fracs = Rearrange('b ... (f d) -> b ... f d', f=num_fracs)
        self.merge_fracs = Rearrange('b ... f d -> b ... (f d)')
        
        # Dimension validation
        assert divisible_by(dim, num_fracs), \
            f"dim ({dim}) must be divisible by num_fracs ({num_fracs})"
        self.dim_per_frac = dim // num_fracs
        
        # Normalization (RMSNorm on flattened streams)
        num_streams_fracs = num_residual_streams * num_fracs
        self.norm = RMSNorm(self.dim_per_frac * num_streams_fracs)
        
        # Deterministic initialization index
        # Uses layer_index for reproducibility, random if not provided
        self.init_residual_index = default(
            layer_index,
            randrange(num_residual_streams)
        ) % num_residual_streams
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.residual_transform = default(residual_transform, nn.Identity())
        self.depth_residual_fn = depth_residual_fn
        
        # =====================================================================
        # VARIANT-SPECIFIC INITIALIZATION
        # Subclass must initialize ALL parameters: H^pre, H^res, H^post
        # =====================================================================
        self._init_hyper_params()
    
    # =========================================================================
    # ABSTRACT METHODS (must be implemented by subclasses)
    # =========================================================================
    
    @abstractmethod
    def _init_hyper_params(self):
        """
        Initialize ALL hyper-connection parameters.
        
        Subclass must initialize:
        - H^pre parameters: static weights, dynamic projection, scale
        - H^res parameters: variant-specific (Sinkhorn/BvN/Kronecker/etc.)
        - H^post parameters: if add_branch_out_to_residual is True
        
        This is called at the end of __init__ after all shared setup.
        
        Example for a manifold-constrained variant:
            # H^pre
            self.static_alpha_pre = nn.Parameter(torch.ones(...) * -1)
            self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
            
            # H^res (variant-specific)
            self.static_alpha_res = nn.Parameter(...)
            self.residual_scale = nn.Parameter(...)
            
            # Combined dynamic weights
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros(...))
            
            # H^post
            if self.add_branch_out_to_residual:
                self.static_beta = nn.Parameter(...)
                self.dynamic_beta_fn = nn.Parameter(...)
                self.h_post_scale = nn.Parameter(...)
        """
        pass
    
    @abstractmethod
    def _compute_alpha_beta(self, normed: torch.Tensor, device: torch.device):
        """
        Compute H^pre, H^res, and H^post matrices.
        
        This is where each variant implements its unique logic:
        - HC: tanh activation, no doubly stochastic constraint
        - mHC: sigmoid + Sinkhorn-Knopp projection
        - mHC-lite: sigmoid + softmax over n! permutation matrices
        - KromHC: sigmoid + Kronecker product of small DS matrices
        - LSB-mHC: sigmoid + sequential Birkhoff construction
        
        Args:
            normed: Normalized input x'_l = RMSNorm(flatten(X_l))
                   Shape: (batch, ..., num_streams * num_fracs * dim_per_frac)
            device: Target device for any newly created tensors
        
        Returns:
            (alpha, beta) tuple where:
            - alpha: Combined H^pre and H^res
                    Shape: (..., num_fracs, num_streams, num_fracs, num_views + num_streams)
            - beta: H^post for depth connection, or None if add_branch_out_to_residual=False
                   Shape: (..., num_fracs, num_streams, num_fracs) when not None
        """
        pass
    
    # =========================================================================
    # SHARED IMPLEMENTATION (identical across all variants)
    # =========================================================================
    
    def _prepare_inputs(self, residuals: torch.Tensor):
        """
        Prepare inputs for width connection.
        
        Steps:
        1. Handle channel_first layout if needed
        2. Split into fractions (for frac-connections)
        3. Rearrange to expose stream dimension
        4. Normalize via RMSNorm
        
        Args:
            residuals: Input tensor, shape depends on channel_first:
                      - channel_first=False: (batch*streams, ..., dim)
                      - channel_first=True: (batch*streams, dim, ...)
        
        Returns:
            (residuals, normed) tuple where:
            - residuals: Rearranged to (batch, ..., streams, dim_per_frac)
            - normed: Normalized flattened input for dynamic weight computation
        """
        streams = self.num_residual_streams
        
        # Handle channel_first layout
        if self.channel_first:
            residuals = rearrange(residuals, 'b d ... -> b ... d')
        
        # Split fractions and expose stream dimension
        residuals = self.split_fracs(residuals)
        residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s=streams)
        
        # Normalize for dynamic weight computation
        normed = rearrange(residuals, 'b ... s d -> b ... (s d)', s=streams)
        normed = self.norm(normed)
        
        return residuals, normed
    
    def _apply_mixing(self, alpha: torch.Tensor, residuals: torch.Tensor):
        """
        Apply stream mixing via einsum.
        
        This implements the matrix multiplication:
            [branch_input; mixed_residuals] = [H^pre; H^res] @ X_l
        
        Args:
            alpha: Combined H^pre and H^res matrix
                  Shape: (..., f1, s, f2, v+s) where v=num_views, s=num_streams
            residuals: Input streams
                      Shape: (..., f1, s, d)
        
        Returns:
            (branch_input, residuals) tuple:
            - branch_input: Input for branch F(·)
            - residuals: Mixed residual streams
        """
        # The einsum performs: mix_h[..., f2, t, d] = sum over f1,s of alpha[..., f1, s, f2, t] * residuals[..., f1, s, d]
        mix_h = einsum(alpha, residuals, '... f1 s f2 t, ... f1 s d -> ... f2 t d')
        
        # Split into branch input (first v entries) and residuals (remaining s entries)
        if self.num_input_views == 1:
            branch_input = mix_h[..., 0, :]
            residuals = mix_h[..., 1:, :]
        else:
            branch_input = mix_h[..., :self.num_input_views, :]
            residuals = mix_h[..., self.num_input_views:, :]
            # Rearrange multiple views for branch
            branch_input = rearrange(branch_input, 'b ... v d -> v b ... d')
        
        return branch_input, residuals
    
    def _finalize_outputs(self, branch_input: torch.Tensor, residuals: torch.Tensor):
        """
        Finalize outputs after mixing.
        
        Steps:
        1. Handle channel_first layout for branch_input
        2. Merge fractions back
        3. Rearrange residuals to expected format
        
        Args:
            branch_input: Mixed input for branch
            residuals: Mixed residual streams
        
        Returns:
            (branch_input, residuals) in the expected output format
        """
        # Finalize branch_input
        if self.channel_first:
            branch_input = rearrange(branch_input, 'b ... d -> b d ...')
        branch_input = self.merge_fracs(branch_input)
        
        # Finalize residuals
        residuals = rearrange(residuals, 'b ... s d -> (b s) ... d')
        if self.channel_first:
            residuals = rearrange(residuals, 'b ... d -> b d ...')
        residuals = self.merge_fracs(residuals)
        
        return branch_input, residuals
    
    def width_connection(self, residuals: torch.Tensor):
        """
        Width connection: mix streams via H^pre and H^res.
        
        This is a Template Method that combines:
        1. _prepare_inputs() - shared input preparation
        2. _compute_alpha_beta() - SUBCLASS-SPECIFIC mixing matrix computation
        3. _apply_mixing() - shared einsum-based mixing
        4. _finalize_outputs() - shared output formatting
        
        Args:
            residuals: Input tensor
        
        Returns:
            (branch_input, residuals, kwargs) tuple where kwargs contains
            any additional data needed by depth_connection (e.g., beta)
        """
        # Step 1: Prepare inputs (shared)
        residuals, normed = self._prepare_inputs(residuals)
        
        # Step 2: Compute mixing matrices (subclass-specific)
        alpha, beta = self._compute_alpha_beta(normed, residuals.device)
        
        # Step 3: Apply mixing (shared)
        branch_input, residuals = self._apply_mixing(alpha, residuals)
        
        # Step 4: Finalize outputs (shared)
        branch_input, residuals = self._finalize_outputs(branch_input, residuals)
        
        return branch_input, residuals, dict(beta=beta)
    
    def depth_connection(self, branch_output: torch.Tensor, residuals: torch.Tensor, *, beta: torch.Tensor):
        """
        Depth connection: distribute branch output via H^post.
        
        This implements:
            output = H^post^T @ F(branch_input)
            X_{l+1} = depth_residual_fn(output, residuals)
        
        Args:
            branch_output: Output from branch F(·)
            residuals: Mixed residual streams from width_connection
            beta: H^post matrix computed in width_connection
        
        Returns:
            Updated residuals after adding branch contribution
        """
        assert self.add_branch_out_to_residual, \
            "depth_connection called but add_branch_out_to_residual=False"
        
        # Prepare branch output
        branch_output = self.split_fracs(branch_output)
        if self.channel_first:
            branch_output = rearrange(branch_output, 'b d ... -> b ... d')
        
        # Apply H^post via einsum
        output = einsum(branch_output, beta, 'b ... f1 d, b ... f1 s f2 -> b ... f2 s d')
        
        # Finalize output
        output = rearrange(output, 'b ... s d -> (b s) ... d')
        output = self.merge_fracs(output)
        if self.channel_first:
            output = rearrange(output, 'b ... d -> b d ...')
        
        # Combine with residuals
        residuals = self.depth_residual_fn(output, residuals)
        return self.dropout(residuals)
    
    def decorate_branch(self, branch: Callable):
        """
        Decorator pattern for wrapping a branch function.
        
        This allows using the hyper-connection as a decorator:
            @hc_layer.decorate_branch
            def my_attention(x):
                return attention(x)
            
            output = my_attention(residuals)  # Automatically applies HC
        
        Args:
            branch: Function or module to wrap
        
        Returns:
            Wrapped function that applies HC + branch + residual
        """
        assert not exists(self.branch), 'branch was already wrapped on init'
        
        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)
            branch_output = branch(branch_input, *args, **kwargs)
            residual = add_residual(branch_output)
            return residual
        
        return forward_and_add_residual
    
    def forward(self, residuals: torch.Tensor, *branch_args, **branch_kwargs):
        """
        Forward pass through hyper-connection layer.
        
        Two usage modes:
        
        1. With branch (set in __init__):
            output = layer(residuals)  # Returns processed output
        
        2. Without branch (functional style):
            branch_input, add_residual = layer(residuals)
            branch_output = my_branch(branch_input)
            output = add_residual(branch_output)
        
        Args:
            residuals: Input residual streams
            *branch_args, **branch_kwargs: Passed to branch if present
        
        Returns:
            If branch is set: Processed output
            If no branch: (branch_input, add_residual_fn) tuple
        """
        # Width connection: mix streams, compute branch input
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)
        
        def add_residual_fn(branch_out):
            """Closure to add branch output via depth connection."""
            if not self.add_branch_out_to_residual:
                return branch_out
            
            # Handle tuple outputs (e.g., attention returning (output, weights))
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)
            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)
            return tree_unflatten((branch_out, *rest), tree_spec)
        
        # If no branch, return components for external handling
        if not exists(self.branch):
            return branch_input, add_residual_fn
        
        # Apply branch and depth connection
        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)
        return add_residual_fn(branch_output)
