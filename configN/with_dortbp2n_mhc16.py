hyper_conn_type = "dortbp2n_mhc"
hyper_conn_n = 16  # Number of residual streams; DORTBP2N requires a power of 2.

# Depth weighting of the transport chart: each local interval choice at recursion
# depth d gets gain g_d = p^d, mean-normalized so the effective beta is unchanged.
# p > 1 favours localized (deep) decisions; p = 1.0 reproduces ORTBP2N exactly.
# With n=8 there are three depth levels (1 coordinate at d=0, 8 at d=1, 40 at d=2),
# so p=1.3 spans a 1.69x range between the root and the leaves.
ortbp_depth_gain_base = 1.5

# Alternative to the geometric law: explicit per-depth gains, one entry per level.
# ortbp_depth_gains = [0.7, 0.9, 1.05]

# Enable ORTBP-specific optimizer groups from train.py / model.configure_optimizers().
ortbp_use_custom_optimizer = True

# Uniform scale on all ORTBP subgroup LRs (multiplies the three *_lr_mult values below).
ortbp_lr_mult = 1.0

# Learning-rate multipliers relative to the global learning_rate:
# residual_chart -> transport-chart logits (static_alpha_res, dynamic_res_alpha_fn)
# residual_scale -> scalar controlling overall RTBP chart stiffness
# delta -> minorization / uniform-mixing scalar delta_logit
ortbp_residual_chart_lr_mult = 1.0 / 6.0
ortbp_residual_scale_lr_mult = 0.187
ortbp_delta_lr_mult = 0.05

# Adam/AdamW betas used only for the ORTBP-specific optimizer groups.
ortbp_beta1 = 0.8
ortbp_beta2 = 0.95

# ORTBP-specific gradient clipping:
# ortbp_grad_clip applies to the residual chart + residual_scale groups
# ortbp_delta_grad_clip applies only to delta_logit
ortbp_use_custom_grad_clip = True
ortbp_grad_clip = 0.3
ortbp_delta_grad_clip = 0.05

# ORTBP diagnostics/logging:
# logs residual_scale, delta, chart_abs_mean, saturation fractions, row entropy,
# and the per-depth chart_abs_mean_d{k} / chart_gain_d{k} breakdown
ortbp_log_stats = True

wandb_notes = "DORTBP2N-mHC"
out_prefix_method = "dortbp2n_mhc"
