"""
ORTBP2N-mHC: Optimized Scaled Power-of-2 Recursive Transportation Birkhoff Polytope

This module keeps the same scaled power-of-2 recursive transport chart as
`srtbp2n_mhc`, but reorganizes the learnable parameters so the residual-chart
projection can be isolated from the H^pre projection. This is useful for
optimizer setup, because AdamW weight decay on the residual transport logits
does not correspond to a natural geometric regularization of the transport
chart; it mainly biases local choices back toward midpoint decisions.

Compared with `SRTBP2N_MHC`, the key architectural change is:

    dynamic_alpha_fn  ->  dynamic_pre_alpha_fn + dynamic_res_alpha_fn
    static_alpha      ->  static_alpha_pre     + static_alpha_res

The chart itself is unchanged. The module exposes both
`no_weight_decay_param_names()` and `optimizer_param_group_names()` so
`model.configure_optimizers()` can isolate the RTBP residual chart in optimizer
setup without also affecting H^pre.
"""

from __future__ import annotations

from functools import partial

import torch
from torch import nn, cat
from torch.nn import Module

from einops import rearrange, repeat

from .utils import exists, default, add, Residual, get_expand_reduce_stream_functions
from .base import BaseHyperConnections
from .srtbp2n_mhc import (
    SRTBP2N_BETA,
    SRTBP2N_EPSILON,
    scaled_recursive_transport_birkhoff_power2,
)


def get_init_and_expand_reduce_stream_functions(
    num_streams,
    num_fracs=1,
    dim=None,
    add_stream_embed=False,
    disable=None,
    **kwargs,
):
    """Get initializer for ORTBP2N-mHC layers plus expand/reduce functions."""
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = ORTBP2N_MHC if not disable else Residual

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


class ORTBP2N_MHC(BaseHyperConnections):
    """
    ORTBP2N-mHC: optimizer-friendly scaled power-of-2 recursive transport Birkhoff.

    The transport chart is identical to `SRTBP2N_MHC`. The optimization-focused
    change is that residual-chart parameters are stored separately so they can be
    placed in a zero-weight-decay optimizer group without also affecting H^pre.
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
    ):
        assert num_residual_streams >= 2, "ORTBP2N-mHC requires at least 2 streams"
        assert num_residual_streams & (num_residual_streams - 1) == 0, (
            f"ORTBP2N-mHC requires a power-of-2 stream count, got {num_residual_streams}"
        )
        self.make_dse = make_dse
        self.log_stats = log_stats
        self._latest_stats = {}

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
        """Initialize ORTBP2N-mHC parameters with split pre/residual projections."""
        n = self.num_residual_streams
        f = self.num_fracs
        v = self.num_input_views
        d = self.dim_per_frac

        init_alpha_pre = torch.ones((n * f, v * f)) * -1
        init_alpha_pre[self.init_residual_index, :] = 1.0
        self.static_alpha_pre = nn.Parameter(init_alpha_pre)

        # Residual chart logits start at 0, giving midpoint local transport choices.
        self.static_alpha_res = nn.Parameter(torch.zeros((n * f, n * f)))

        # Split the dynamic projection so the residual chart can be placed in a
        # zero-weight-decay group independently of H^pre.
        self.dynamic_pre_alpha_fn = nn.Parameter(torch.zeros(d * n, f * (n * v)))
        self.dynamic_res_alpha_fn = nn.Parameter(torch.zeros(d * n, f * (n * n)))

        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
        self.residual_scale = nn.Parameter(torch.ones(1) * 1e-2)

        if self.make_dse:
            self.delta_logit = nn.Parameter(torch.tensor(-8.0))
        else:
            self.delta_logit = None

        if self.add_branch_out_to_residual:
            beta_init = torch.ones(n * f) * -1.0
            beta_init[self.init_residual_index] = 1.0
            self.static_beta = nn.Parameter(beta_init)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(d * n, f * n))
            self.h_post_scale = nn.Parameter(torch.ones(()) * 1e-2)

    def no_weight_decay_param_names(self) -> set[str]:
        """
        Return local parameter names that should bypass AdamW weight decay.

        The important one is `dynamic_res_alpha_fn`: decaying it pushes transport
        logits back toward zero, which means midpoint chart choices rather than a
        meaningful shrinkage of the doubly stochastic residual matrix.
        """
        names = {
            "static_alpha_res",
            "dynamic_res_alpha_fn",
            "residual_scale",
        }
        if self.make_dse and exists(self.delta_logit):
            names.add("delta_logit")
        return names

    def optimizer_param_group_names(self) -> dict[str, set[str]]:
        """
        Return named local parameter groups for optimizer customization.

        These names are local to the module and are expanded to full parameter
        names by `GPT.configure_optimizers()`. Keeping the residual chart, its
        global scale, and the optional minorization scalar separate allows later
        training code to assign different learning rates or betas.
        """
        groups = {
            "ortbp_residual_chart": {
                "static_alpha_res",
                "dynamic_res_alpha_fn",
            },
            "ortbp_residual_scale": {"residual_scale"},
        }
        if self.make_dse and exists(self.delta_logit):
            groups["ortbp_delta"] = {"delta_logit"}
        return groups

    def clear_stats(self):
        """Clear the last recorded ORTBP diagnostics."""
        self._latest_stats = {}

    def get_stats(self) -> dict[str, float]:
        """Return the most recent detached ORTBP diagnostics."""
        return dict(self._latest_stats)

    def _record_stats(self, ortbp2n_flat: torch.Tensor, ds_flat: torch.Tensor):
        """Record lightweight scalar diagnostics for training-time logging."""
        if not self.log_stats:
            return

        with torch.no_grad():
            flat = ortbp2n_flat.detach()
            abs_flat = flat.abs()
            ds = ds_flat.detach().clamp_min(1e-12)

            stats = {
                "residual_scale": float(self.residual_scale.detach().item()),
                "chart_abs_mean": float(abs_flat.mean().item()),
                "chart_sat5": float((abs_flat > 5).float().mean().item()),
                "chart_sat8": float((abs_flat > 8).float().mean().item()),
                "row_entropy": float((-(ds * ds.log()).sum(dim=-1).mean()).item()),
            }

            if self.make_dse and exists(self.delta_logit):
                stats["delta"] = float(torch.sigmoid(self.delta_logit.detach()).item())

            self._latest_stats = stats

    def _compute_alpha_beta(self, normed: torch.Tensor, device: torch.device):
        """Compute H^pre, H^res (scaled power-of-2 RTBP), and H^post matrices."""
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
        ortbp2n_params = alpha_res[..., : n - 1, : n - 1]

        orig_shape = ortbp2n_params.shape[:-2]
        ortbp2n_flat = ortbp2n_params.reshape(-1, n - 1, n - 1)

        ds_flat = scaled_recursive_transport_birkhoff_power2(
            ortbp2n_flat,
            delta_logit=self.delta_logit if self.make_dse else None,
            beta=SRTBP2N_BETA,
            epsilon=SRTBP2N_EPSILON,
        )
        self._record_stats(ortbp2n_flat, ds_flat)

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


ORTBP2N_MHC.get_expand_reduce_stream_functions = staticmethod(
    get_init_and_expand_reduce_stream_functions
)
ORTBP2N_MHC.get_init_and_expand_reduce_stream_functions = staticmethod(
    get_init_and_expand_reduce_stream_functions
)


class OptimizedPowerOfTwoRecursiveTransportBirkhoff(nn.Module):
    """Standalone wrapper around the shared scaled RTBP2N transport chart."""

    def __init__(
        self,
        num_streams: int,
        make_dse: bool = True,
        eps: float = 1e-7,
        beta: float = SRTBP2N_BETA,
        epsilon: float = SRTBP2N_EPSILON,
    ):
        super().__init__()
        if num_streams < 2:
            raise ValueError("num_streams must be >= 2")
        if num_streams & (num_streams - 1) != 0:
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

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        n = self.num_streams
        if t.shape[-2:] != (n - 1, n - 1):
            raise ValueError(f"Expected t shape (..., {n-1}, {n-1}), got {tuple(t.shape)}")
        return scaled_recursive_transport_birkhoff_power2(
            t,
            delta_logit=self.delta_logit if self.make_dse else None,
            eps=self.eps,
            beta=self.beta,
            epsilon=self.epsilon,
        )


ORTBP2NHC = ORTBP2N_MHC
