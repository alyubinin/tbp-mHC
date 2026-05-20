# ASTBP-mHC: Averaged Scaled Transport Birkhoff Polytope
#
# Permutation format for astbp_permutations (same as alsb_permutations):
#   - List of permutation specs. Each spec is either:
#     - str: "direct" = identity [0,1,...,n-1], "reverse" = [n-1,...,0]
#     - list of n ints: explicit permutation of {0,...,n-1}
#   - Default (if not set): ["direct", "reverse"]
#
hyper_conn_type = "astbp_mhc"
hyper_conn_n = 4
astbp_permutations = ["direct", "reverse"]

wandb_notes = "ASTBP-mHC"
out_prefix_method = "astbp_mhc"
