# ATBP-mHC: Averaged Transportation Birkhoff Polytope
#
# Permutation format for atbp_permutations:
#   - List of permutation specs. Each spec is either:
#     - str: "direct" = identity [0,1,...,n-1], "reverse" = [n-1,...,0]
#     - list of n ints: explicit permutation of {0,...,n-1}
#   - Default (if not set): ["direct", "reverse"] = two charts, forward + reverse order
#
# Example for n=4:
#   atbp_permutations = ["direct", "reverse"]           # default, 2 charts
#   atbp_permutations = ["direct", "reverse", [3,1,0,2]]  # 3 charts, third is custom
#
hyper_conn_type = "atbp_mhc"
hyper_conn_n = 4
atbp_permutations = ["direct", "reverse"]

wandb_notes = "ATBP-mHC"
out_prefix_method = "atbp_mhc"
