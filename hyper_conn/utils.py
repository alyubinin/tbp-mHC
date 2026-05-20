"""
Shared utilities for all hyper-connection variants.

This module contains helper functions and classes that are identical across
all hyper-connection implementations (HC, mHC, mHC-lite, KromHC, TBP-mHC, etc.).

Extracting these to a single location:
- Eliminates ~1,000 lines of duplicated code
- Provides a single source of truth for bug fixes
- Makes it easier to add new variants
"""

from __future__ import annotations
from typing import Callable
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module
from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def exists(v):
    """Check if value is not None."""
    return v is not None


def divisible_by(num, den):
    """Check if num is evenly divisible by den."""
    return (num % den) == 0


def default(v, d):
    """Return v if not None, else return default d."""
    return v if exists(v) else d


def identity(t):
    """Identity function."""
    return t


def add(x, y):
    """Simple addition for depth residual connection."""
    return x + y


# ============================================================================
# RMS NORMALIZATION
# ============================================================================

class RMSNorm(Module):
    """
    Root Mean Square Layer Normalization.
    
    Normalizes the input by its RMS and applies learned scale.
    Used for x'_l = RMSNorm(x_l) in the hyper-connection equations.
    
    More efficient than LayerNorm (no mean subtraction) and works
    well in modern transformer architectures.
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1)


# ============================================================================
# STANDARD RESIDUAL CONNECTION (baseline/fallback)
# ============================================================================

class Residual(Module):
    """
    Standard residual connection: x_{l+1} = x_l + F(x_l)
    
    The foundational residual connection from ResNet (He et al., 2016).
    Used as fallback when hyper-connections are disabled (hyper_conn_type="none").
    
    This class implements the same interface as hyper-connection classes
    to allow seamless switching between standard residuals and HC variants.
    """
    def __init__(
        self,
        *args,
        branch: Module | None = None,
        residual_transform: Module | None = None,
        **kwargs
    ):
        super().__init__()
        self.branch = branch
        self.residual_transform = default(residual_transform, nn.Identity())

    def width_connection(self, residuals):
        """Width connection is trivial for standard residual."""
        return residuals, residuals, dict()

    def depth_connection(self, branch_output, residuals):
        """Depth connection: simply add branch output to residual."""
        return branch_output + self.residual_transform(residuals)

    def decorate_branch(self, branch: Callable):
        """Decorator pattern for wrapping a branch function."""
        assert not exists(self.branch), 'branch was already wrapped on init'

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)
            branch_output = branch(branch_input, *args, **kwargs)
            residual = add_residual(branch_output)
            return residual

        return forward_and_add_residual

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)
            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)
            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)
        return add_residual_fn(branch_output)


# ============================================================================
# STREAM EMBEDDING
# ============================================================================

class StreamEmbed(Module):
    """
    Learnable embeddings added to each residual stream.
    
    When expanding a single input to n streams, the default behavior
    just replicates the input. StreamEmbed adds unique learned vectors
    to each stream to break symmetry.
    
    This allows streams to specialize for different aspects of the
    representation from the very first layer.
    """
    def __init__(
        self,
        num_streams,
        dim,
        channel_first=False,
        expand_to_streams=False
    ):
        super().__init__()
        self.channel_first = channel_first
        self.num_streams = num_streams
        self.expand_to_streams = expand_to_streams
        self.stream_embed = nn.Parameter(torch.zeros(num_streams, dim))

    def forward(self, residuals):
        if self.expand_to_streams:
            residuals = repeat(residuals, 'b ... -> (b s) ...', s=self.num_streams)

        if self.channel_first:
            residuals = rearrange(residuals, '(b s) d ... -> b ... s d', s=self.num_streams)
        else:
            residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s=self.num_streams)

        residuals = residuals + self.stream_embed

        if self.channel_first:
            residuals = rearrange(residuals, 'b ... s d -> (b s) d ...', s=self.num_streams)
        else:
            residuals = rearrange(residuals, 'b ... s d -> (b s) ... d', s=self.num_streams)

        return residuals


# ============================================================================
# ATTENTION-BASED STREAM POOLING
# ============================================================================

class AttentionPoolReduceStream(Module):
    """
    Attention-based pooling to reduce multiple streams to one.
    
    Instead of simple summation over streams, uses learned attention
    weights based on content similarity. This allows the model to
    dynamically select which streams are most relevant at each position.
    
    Reference: Similar to attention pooling in Enformer
    (Avsec et al., Nature Methods 2021)
    """
    def __init__(
        self,
        num_streams,
        dim,
        channel_first=False
    ):
        super().__init__()
        self.num_streams = num_streams
        self.channel_first = channel_first

        # Initialize to identity for gradual learning
        self.to_attn_logits = nn.Linear(dim, dim, bias=False)
        self.to_attn_logits.weight.data.copy_(torch.eye(dim))

    def forward(self, residuals):
        if self.channel_first:
            residuals = rearrange(residuals, '(b s) d ... -> b ... s d', s=self.num_streams)
        else:
            residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s=self.num_streams)

        attn_logits = self.to_attn_logits(residuals)
        attn = attn_logits.softmax(dim=-2)  # Softmax over streams

        residuals = reduce(residuals * attn, 'b ... s d -> b ... d', 'sum')

        if self.channel_first:
            residuals = rearrange(residuals, 'b ... d -> b d ...')

        return residuals


# ============================================================================
# STREAM EXPANSION/REDUCTION FUNCTIONS
# ============================================================================

def get_expand_reduce_stream_functions(
    num_streams,
    add_stream_embed=False,
    dim=None,
    disable=False
):
    """
    Get functions to expand input to n streams and reduce back to 1.
    
    EXPANSION: Replicates input across n streams
        (batch, ..., dim) -> (batch * n, ..., dim)
    
    REDUCTION: Sums across streams
        (batch * n, ..., dim) -> (batch, ..., dim)
    
    This is part of the public API used by hyper_conn_init_func().
    
    Args:
        num_streams: n, number of parallel residual streams
        add_stream_embed: If True, add learnable per-stream embeddings
        dim: Feature dimension (required if add_stream_embed=True)
        disable: If True, return identity functions
        
    Returns:
        (expand_fn, reduce_fn) tuple of nn.Module
    """
    if num_streams == 1 or disable:
        return (nn.Identity(), nn.Identity())

    if add_stream_embed:
        assert exists(dim), '`dim` must be passed for stream embeddings'
        expand_fn = StreamEmbed(num_streams, dim, expand_to_streams=True)
    else:
        # Simple replication via einops Reduce with 'repeat'
        expand_fn = Reduce(pattern='b ... -> (b s) ...', reduction='repeat', s=num_streams)

    # Sum reduction across streams
    reduce_fn = Reduce(pattern='(b s) ... -> b ...', reduction='sum', s=num_streams)

    return expand_fn, reduce_fn


# ============================================================================
# DEBUGGING UTILITIES
# ============================================================================

def print_trainable_parameters(
    model, 
    prefix_filter: str = None, 
    contains_filter: str = None,
    show_shapes: bool = True
):
    """
    Print all trainable parameters in a model.
    
    This is useful for verifying exactly what's being learned in a given
    configuration, especially when using hyper-connections where different
    variants have different parameter structures.
    
    Args:
        model: PyTorch model (nn.Module)
        prefix_filter: Optional string to filter parameters by name prefix
                      (e.g., "transformer.h.0" to see only first layer)
        contains_filter: Optional string to filter parameters by substring
                        (e.g., "hc_" to see all hyper-connection params)
        show_shapes: If True, show parameter shapes; if False, just names
        
    Returns:
        dict: {param_name: param_shape} for all trainable parameters
        
    Example:
        >>> from hyper_conn import print_trainable_parameters
        >>> print_trainable_parameters(model)
        Trainable parameters (125,000,000 total):
          transformer.wte.weight                (50304, 768)
          transformer.wpe.weight                (1024, 768)
          transformer.h.0.attn.c_attn.weight    (768, 2304)
          ...
          
        >>> # Filter to see only hyper-connection params (by substring)
        >>> print_trainable_parameters(model, contains_filter="hc_")
        
        >>> # Filter by prefix (first transformer block)
        >>> print_trainable_parameters(model, prefix_filter="transformer.h.0")
    """
    trainable_params = {}
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Apply filters
            if prefix_filter is not None and not name.startswith(prefix_filter):
                continue
            if contains_filter is not None and contains_filter not in name:
                continue
            trainable_params[name] = tuple(param.shape)
            total_params += param.numel()
    
    # Format total with commas
    total_str = f"{total_params:,}"
    
    # Build filter description
    filter_desc = ""
    if prefix_filter:
        filter_desc = f" starting with '{prefix_filter}'"
    if contains_filter:
        filter_desc = f" containing '{contains_filter}'"
    
    if filter_desc:
        print(f"Trainable parameters{filter_desc} ({total_str} total):")
    else:
        print(f"Trainable parameters ({total_str} total):")
    
    for name, shape in trainable_params.items():
        if show_shapes:
            print(f"  {name:60s} {shape}")
        else:
            print(f"  {name}")
    
    return trainable_params


def count_parameters(model, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model (nn.Module)
        trainable_only: If True, count only trainable parameters
        
    Returns:
        int: Number of parameters
        
    Example:
        >>> total = count_parameters(model)
        >>> trainable = count_parameters(model, trainable_only=True)
        >>> print(f"Total: {total:,}, Trainable: {trainable:,}")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())
