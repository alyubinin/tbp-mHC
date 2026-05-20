# LMALTBP-mHC: Lazyfied-Minorized Averaged Linear Transport Birkhoff Polytope
#
# Same as ALTBP but applies H_n = (1-λ-μ)*I + λ*X_n + μ*(J/n) to control spectral gap.
# λ and μ are learnable scalars per layer. Initial values configurable below.
# Uses altbp_permutations for the underlying chart mixture.
#
hyper_conn_type = "lmaltbp_mhc"
hyper_conn_n = 4
altbp_permutations = ["direct", "reverse"]
lmaltbp_lambda_init = 0.8
lmaltbp_mu_init = 0.01

wandb_notes = "LMALTBP-mHC"
out_prefix_method = "lmaltbp_mhc"
