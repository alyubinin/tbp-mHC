hyper_conn_type = "ortbp2n_mhc"
hyper_conn_n = 4  # Number of residual streams; ORTBP2N requires a power of 2.

# Enable ORTBP-specific optimizer groups from train.py / model.configure_optimizers().
ortbp_use_custom_optimizer = True

# Uniform scale on all ORTBP subgroup LRs (multiplies the three *_lr_mult values below).
ortbp_lr_mult = 1.0

# Learning-rate multipliers relative to the global learning_rate:
# residual_chart -> transport-chart logits (static_alpha_res, dynamic_res_alpha_fn)
# residual_scale -> scalar controlling overall RTBP chart stiffness
# delta -> minorization / uniform-mixing scalar delta_logit
ortbp_residual_chart_lr_mult = 1.0 / 6.0
ortbp_residual_scale_lr_mult = 0.05
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
# logs residual_scale, delta, chart_abs_mean, saturation fractions, and row entropy
ortbp_log_stats = True

wandb_notes = "ORTBP2N-mHC"
out_prefix_method = "ortbp2n_mhc"
