# LTBP-mHC: Linear Transport Birkhoff Polytope
#
# LSB variant with X[i,j] = L_ij + (U_ij - L_ij) * t[i,j], t ∈ [0,1].
# Explicit (n-1)² params per layer, clamped after each optimizer step.
#
hyper_conn_type = "ltbp_mhc"
hyper_conn_n = 4

wandb_notes = "LTBP-mHC"
out_prefix_method = "ltbp_mhc"
