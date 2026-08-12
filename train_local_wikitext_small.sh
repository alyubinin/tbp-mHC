#!/usr/bin/env bash
set -euo pipefail
cd /workspace/tbp-mHC

echo "Training mHC (small model, wikitext103 n=4)..."
# torchrun --standalone --nproc_per_node=8 train.py configN/small_model.py configN/with_mhc4.py configN/train_wikitext103.py

echo
echo "Training mHC_lite (small model, wikitext103 n=4)..."
torchrun --standalone --nproc_per_node=8 train.py configN/small_model.py configN/with_mhc_lite4.py configN/train_wikitext103.py

echo
echo "Training KromHC (small model, wikitext103 n=4)..."
torchrun --standalone --nproc_per_node=8 train.py configN/small_model.py configN/with_KromHC4.py configN/train_wikitext103.py

echo
echo "Training DORTBP2N (small model, wikitext103 n=4)..."
torchrun --standalone --nproc_per_node=8 train.py configN/small_model.py configN/with_dortbp2n_mhc4.py configN/train_wikitext103.py

echo
echo "Done."
