"""
DORTBP2N-mHC: Depth-Weighted Optimized Scaled Power-of-2 Recursive Transport Birkhoff

This module reuses the ORTBP2N layer and the scaled power-of-2 recursive transport
chart unchanged, and adds a single static per-coordinate multiplier that depends on
how localized the corresponding transport decision is.

MOTIVATION
==========

The recursive chart is hierarchical: a decision taken near the root changes the
margins handed to every descendant, so it moves a large part of the resulting
matrix. A decision taken deep in the tree only rearranges mass inside one small
block. Weighting deeper coordinates more strongly biases the chart toward
"rearrange locally first, move coarse mass later".

FORMULA
=======

Every local interval choice in SRTBP2N uses

    x = L + (U - L) * sigmoid(beta * t / (U - L + epsilon))

This module replaces the sigmoid argument with

    x = L + (U - L) * sigmoid(g_d * beta * t / (U - L + epsilon))

where `d` is the recursion depth of that particular choice and `g_d` is a static
gain. Because the gain enters multiplicatively,

    beta * (g_d * t) / (U - L + epsilon) == g_d * beta * t / (U - L + epsilon)

scaling the chart parameters before the chart call is exactly equivalent to
threading a depth argument through the recursion. That is what this module does,
which leaves `srtbp2n_mhc` and `ortbp2n_mhc` untouched.

At the midpoint initialization t = 0 and with beta = 4, the derivative is

    dx/dt = g_d * (beta / 4) * (U - L) / (U - L + epsilon) ~= g_d

so `g_d` is very close to a per-depth unit sensitivity. Under Adam, whose updates
are approximately sign-like, it acts as a per-depth step size in chart-output
space rather than as a gradient rescaling that the optimizer would normalize away.

DEPTH
=====

Depth is defined by spatial scale,

    d = log2(N / s)

where N is the stream count and s is the size of the group a decision operates on.
The chart contains two interleaved recursions - the matrix quadrant split and the
margin vector split - and this definition gives equal gain to decisions acting at
the same resolution in either of them. The root decision has d = 0.

NORMALIZATION
=============

Gains are mean-normalized by default. The depth histogram is heavily weighted
toward the deepest level (for n = 8, 40 of the 49 coordinates sit at d = 2), so a
raw p^d would mostly act as a uniform increase of beta with only a small
differential between levels. Normalizing keeps the effective beta fixed so the
experiment isolates the depth bias.

With `depth_gain_base = 1.0` the gain vector is all ones and this variant is
numerically identical to ORTBP2N-mHC.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Optional, Sequence

import torch
from torch import nn, cat
from torch.nn import Module

from einops import rearrange, repeat

from .utils import exists, default, add, Residual, get_expand_reduce_stream_functions
from .ortbp2n_mhc import ORTBP2N_MHC
from .srtbp2n_mhc import (
    SRTBP2N_BETA,
    SRTBP2N_EPSILON,
    scaled_recursive_transport_birkhoff_power2,
)


def _is_power_of_two(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _split_depths(k: int, n_root: int) -> list[int]:
    """
    Depth labels for the k-1 nodes of a balanced binary split over k leaves.

    `_binary_split_vector` consumes its logits in pre-order: the node's own choice
    first, then the whole left subtree, then the whole right subtree. A node that
    splits a vector of length k acts at spatial scale k, hence depth log2(N / k).
    """
    if k <= 1:
        return []
    depth = int(math.log2(n_root // k))
    child = _split_depths(k // 2, n_root)
    return [depth] + child + child


def chart_depth_labels(n: int, n_root: Optional[int] = None, depth: int = 0) -> list[int]:
    """
    Depth label for every chart coordinate, in the order the chart consumes them.

    Mirrors the parameter bookkeeping of `_recursive_transport_tbp_power2`:
    the block mass m11, then the four margin splits (a, gamma, u, s), then the
    four child blocks. The children share one layout because the chart recurses
    on them as a single stacked batch.

    Returns a list of length (n-1)^2.
    """
    n_root = default(n_root, n)
    if n <= 1:
        return []
    if n == 2:
        return [depth]

    half = n // 2
    labels = [depth]
    for _ in range(4):  # a, gamma, u, s
        labels += _split_depths(half, n_root)
    labels += 4 * chart_depth_labels(half, n_root, depth + 1)
    return labels


def build_chart_gain(
    n: int,
    depth_gain_base: float = 1.0,
    depth_gains: Optional[Sequence[float]] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Build the (n-1, n-1) static gain applied to the chart parameters.

    Args:
        n: stream count, must be a power of 2
        depth_gain_base: p in the geometric law g_d = p^d
        depth_gains: optional explicit per-depth gains, overriding the geometric law
        normalize: divide by the mean gain so the effective beta is unchanged

    The returned tensor is laid out to match the chart's row-major flattening
    (`t.reshape(batch, (n-1)*(n-1))`), so tree index k lands at
    (k // (n-1), k % (n-1)).
    """
    if not _is_power_of_two(n) or n < 2:
        raise ValueError(f"DORTBP2N requires a power-of-2 stream count >= 2, got {n}")

    labels = chart_depth_labels(n)
    expected = (n - 1) * (n - 1)
    if len(labels) != expected:
        raise RuntimeError(
            f"depth labelling produced {len(labels)} entries, expected {expected} for n={n}"
        )

    if exists(depth_gains):
        gains = list(depth_gains)
        needed = max(labels) + 1
        if len(gains) < needed:
            raise ValueError(
                f"depth_gains needs at least {needed} entries for n={n}, got {len(gains)}"
            )
        if any(g <= 0 for g in gains[:needed]):
            raise ValueError(f"depth_gains must be positive, got {gains[:needed]}")
        values = [float(gains[d]) for d in labels]
    else:
        if depth_gain_base <= 0:
            raise ValueError(f"depth_gain_base must be positive, got {depth_gain_base}")
        values = [float(depth_gain_base) ** d for d in labels]

    gain = torch.tensor(values, dtype=torch.float32)
    if normalize:
        gain = gain / gain.mean()
    return gain.reshape(n - 1, n - 1)


def build_chart_depth_ids(n: int) -> torch.Tensor:
    """Per-coordinate depth labels shaped like the chart parameter block."""
    labels = chart_depth_labels(n)
    return torch.tensor(labels, dtype=torch.long).reshape(n - 1, n - 1)


def get_init_and_expand_reduce_stream_functions(
    num_streams,
    num_fracs=1,
    dim=None,
    add_stream_embed=False,
    disable=None,
    **kwargs,
):
    """Get initializer for DORTBP2N-mHC layers plus expand/reduce functions."""
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = DORTBP2N_MHC if not disable else Residual

    init_hyper_conn_fn = partial(
        hyper_conn_klass,
        num_streams,
        num_fracs=num_fracs,
        **kwargs,
    )
    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams,
        add_stream_embed=add_stream_embed,
        dim=dim,
        disable=disable,
    )

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim=dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


class DORTBP2N_MHC(ORTBP2N_MHC):
    """
    DORTBP2N-mHC: ORTBP2N with a depth-weighted transport chart.

    Inherits the parameter layout, the optimizer group contract, and the stats
    accessors from `ORTBP2N_MHC`. The only functional change is that the chart
    parameters are multiplied by a static per-coordinate gain before the exact
    doubly stochastic residual matrix is built.
    """

    def __init__(
        self,
        num_residual_streams: int,
        *,
        dim: int,
        branch: Module | None = None,
        layer_index: int | None = None,
        channel_first: bool = False,
        dropout: float = 0.0,
        residual_transform: Module | None = None,
        add_branch_out_to_residual: bool = True,
        num_input_views: int = 1,
        depth_residual_fn=add,
        num_fracs: int = 1,
        make_dse: bool = True,
        log_stats: bool = False,
        depth_gain_base: float = 1.0,
        depth_gains: Optional[Sequence[float]] = None,
        normalize_depth_gain: bool = True,
    ):
        # Consumed by _init_hyper_params, which the base constructor calls.
        self.depth_gain_base = depth_gain_base
        self.depth_gains = list(depth_gains) if exists(depth_gains) else None
        self.normalize_depth_gain = normalize_depth_gain

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
            make_dse=make_dse,
            log_stats=log_stats,
        )

    def _init_hyper_params(self):
        """Initialize ORTBP2N parameters, then register the static depth gain."""
        super()._init_hyper_params()

        n = self.num_residual_streams
        gain = build_chart_gain(
            n,
            depth_gain_base=self.depth_gain_base,
            depth_gains=self.depth_gains,
            normalize=self.normalize_depth_gain,
        )
        depth_ids = build_chart_depth_ids(n)

        # Non-persistent: the gain is derived from config, so it must not be
        # restored from a checkpoint and silently override the current setting.
        self.register_buffer("chart_gain", gain, persistent=False)
        self.register_buffer("chart_depth_ids", depth_ids, persistent=False)
        self.chart_depth_values = sorted(set(depth_ids.flatten().tolist()))

    def _record_stats(self, ortbp2n_flat: torch.Tensor, ds_flat: torch.Tensor):
        """
        Record ORTBP diagnostics plus a per-depth breakdown.

        `ortbp2n_flat` is the gained tensor, so the saturation fractions describe
        the actual sigmoid argument scale rather than the raw logits, which stop
        being comparable across depths once the gains differ.
        """
        super()._record_stats(ortbp2n_flat, ds_flat)

        if not self.log_stats:
            return

        with torch.no_grad():
            abs_flat = ortbp2n_flat.detach().abs()
            for depth in self.chart_depth_values:
                mask = self.chart_depth_ids == depth
                self._latest_stats[f"chart_abs_mean_d{depth}"] = float(
                    abs_flat[..., mask].mean().item()
                )
                self._latest_stats[f"chart_gain_d{depth}"] = float(
                    self.chart_gain[mask][0].item()
                )

    def _compute_alpha_beta(self, normed: torch.Tensor, device: torch.device):
        """Compute H^pre, H^res (depth-weighted scaled power-of-2 RTBP), and H^post."""
        n = self.num_residual_streams
        f = self.num_fracs
        v = self.num_input_views

        wc_pre = normed @ self.dynamic_pre_alpha_fn
        wc_pre = rearrange(wc_pre, "... (s t) -> ... s t", s=n)
        pre_scale = repeat(self.pre_branch_scale, "1 -> v", v=v * f)
        dynamic_pre = wc_pre * pre_scale

        wc_res = normed @ self.dynamic_res_alpha_fn
        wc_res = rearrange(wc_res, "... (s t) -> ... s t", s=n)
        res_scale = repeat(self.residual_scale, "1 -> s", s=f * n)
        dynamic_res = wc_res * res_scale

        dynamic_alpha = cat((dynamic_pre, dynamic_res), dim=-1)
        static_alpha = cat((self.static_alpha_pre, self.static_alpha_res), dim=1)
        static_alpha = rearrange(static_alpha, "(f s) t -> f s t", s=n)

        alpha = dynamic_alpha + static_alpha
        alpha = self.split_fracs(alpha)

        alpha_pre, alpha_res = alpha[..., :v], alpha[..., v:]
        alpha_pre = alpha_pre.sigmoid()

        alpha_res = rearrange(alpha_res, "... f s g t -> ... f g s t")
        dortbp2n_params = alpha_res[..., : n - 1, : n - 1]

        orig_shape = dortbp2n_params.shape[:-2]
        dortbp2n_flat = dortbp2n_params.reshape(-1, n - 1, n - 1)

        # Depth weighting: equivalent to scaling the sigmoid argument by g_d.
        dortbp2n_flat = dortbp2n_flat * self.chart_gain.to(dtype=dortbp2n_flat.dtype)

        ds_flat = scaled_recursive_transport_birkhoff_power2(
            dortbp2n_flat,
            delta_logit=self.delta_logit if self.make_dse else None,
            beta=SRTBP2N_BETA,
            epsilon=SRTBP2N_EPSILON,
        )
        self._record_stats(dortbp2n_flat, ds_flat)

        alpha_res = ds_flat.reshape(*orig_shape, n, n)
        alpha_res = rearrange(alpha_res, "... f g s t -> ... f s g t")
        alpha = cat((alpha_pre, alpha_res), dim=-1)

        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = normed @ self.dynamic_beta_fn
            dc_weight = rearrange(dc_weight, "... (s f) -> ... s f", s=n)
            dynamic_beta = dc_weight * self.h_post_scale
            static_beta = rearrange(self.static_beta, "(s f) -> s f", s=n)
            beta = (dynamic_beta + static_beta).sigmoid() * 2

        return alpha, beta


DORTBP2N_MHC.get_expand_reduce_stream_functions = staticmethod(
    get_init_and_expand_reduce_stream_functions
)
DORTBP2N_MHC.get_init_and_expand_reduce_stream_functions = staticmethod(
    get_init_and_expand_reduce_stream_functions
)


class DepthWeightedPowerOfTwoRecursiveTransportBirkhoff(nn.Module):
    """Standalone depth-weighted wrapper around the scaled RTBP2N transport chart."""

    def __init__(
        self,
        num_streams: int,
        make_dse: bool = True,
        eps: float = 1e-7,
        beta: float = SRTBP2N_BETA,
        epsilon: float = SRTBP2N_EPSILON,
        depth_gain_base: float = 1.0,
        depth_gains: Optional[Sequence[float]] = None,
        normalize_depth_gain: bool = True,
    ):
        super().__init__()
        if num_streams < 2:
            raise ValueError("num_streams must be >= 2")
        if not _is_power_of_two(num_streams):
            raise ValueError(f"num_streams must be a power of 2, got {num_streams}")
        self.num_streams = num_streams
        self.eps = eps
        self.beta = beta
        self.epsilon = epsilon
        self.make_dse = make_dse
        if make_dse:
            self.delta_logit = nn.Parameter(torch.tensor(-8.0))
        else:
            self.delta_logit = None

        gain = build_chart_gain(
            num_streams,
            depth_gain_base=depth_gain_base,
            depth_gains=depth_gains,
            normalize=normalize_depth_gain,
        )
        self.register_buffer("chart_gain", gain, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        n = self.num_streams
        if t.shape[-2:] != (n - 1, n - 1):
            raise ValueError(f"Expected t shape (..., {n-1}, {n-1}), got {tuple(t.shape)}")
        return scaled_recursive_transport_birkhoff_power2(
            t * self.chart_gain.to(dtype=t.dtype),
            delta_logit=self.delta_logit if self.make_dse else None,
            eps=self.eps,
            beta=self.beta,
            epsilon=self.epsilon,
        )


DORTBP2NHC = DORTBP2N_MHC
