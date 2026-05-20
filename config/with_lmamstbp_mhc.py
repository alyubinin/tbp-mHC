# LMAMSTBP-mHC: Lazyfied-Minorized Averaged Margined Scaled Transport Birkhoff Polytope
#
# Same as AMSTBP but applies H_n = (1-λ-μ)*I + λ*X_n + μ*(J/n) to control spectral gap.
# λ and μ are learnable scalars per layer. Initial values configurable below.
# Uses amstbp_permutations for the underlying chart mixture.
#
hyper_conn_type = "lmamstbp_mhc"
hyper_conn_n = 4
amstbp_permutations = ["direct", "reverse"]
lmamstbp_lambda_init = 0.8
lmamstbp_mu_init = 0.01

wandb_notes = "LMAMSTBP-mHC"
out_prefix_method = "lmamstbp_mhc"
