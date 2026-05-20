# TBP-mHC

This repository contains the experiment code for the **TBP-mHC** paper [**TBP-mHC: full expressivity for manifold-constrained hyper connections through transportation polytopes**]().
It extends the previous [mHC-lite](https://github.com/FFTYYY/mhc-lite), [KromHC](https://github.com/wz1119/KromHC) and related mHC code repositories with TBP/RTBP parametrizations and experiments.

---

## Previous Base: mHC-lite

This repository contains the experiment code for the paper [**mHC-lite: You Don’t Need 20 Sinkhorn-Knopp Iterations**](https://arxiv.org/abs/2601.05732). The codebase is adapted from [nanoGPT](https://github.com/karpathy/nanoGPT).

## Preparation

Install the required packages:

```sh
pip install torch numpy transformers datasets tiktoken wandb tqdm einops
```

#### Data preparation

To prepare the datasets, enter the corresponding dataset folder and run `prepare.py`:

```sh
cd data/shakespeare_char
python prepare.py

cd ../fineweb_edu
python prepare.py

cd ../openwebtext
python prepare.py
```

Data preparation typically takes **~30 minutes** (depending on your machine and disk speed).

## Training

To train a model, run `train.py`. Use `torchrun` to enable distributed training (see the original nanoGPT project for details). You can combine multiple config files to specify the dataset, model scale, and method.

### Available config files

* **Model scales**:

  * S: `config/small_model.py`
  * M: `config/medium_model.py`
  * L: `config/large_model.py`

* **Methods**:

  * HC: `config/with_hc.py`
  * mHC: `config/with_mhc.py`
  * mHC-lite: `config/with_mhc_lite.py`
  * Residual: (default)

* **Datasets**:

  * OpenWebText: `config/train_owt.py`
  * FineWeb-Edu: `config/train_fineweb_edu.py`

### Example

Train a **small (S)** model with **mHC-lite** on **OpenWebText**:

```sh
torchrun --standalone --nproc_per_node=8 train.py \
  config/train_owt.py config/small_model.py config/with_mhc_lite.py
```

* Set `--nproc_per_node` to the number of GPUs you have.

Alternatively, run `run.sh` to reproduce all experiments reported in Table 1 of the paper.

## Analyze

Run `train_analysis.py` with `config/with_mhc_analysis.py` to perform analysis using a checkpoint. 

Experiment runs automatically create output directories and save checkpoints. For analysis runs, please specify the checkpoint directory via `--out_dir` so the script can resume from it. Anakysis can only be performed on checkpoints with mHC enabled.

### Example

To analyze a checkpoint from a **small** model trained on **OpenWebText**, set `--out_dir` to `out-owt-small-mhc`:

```sh
python train_analysis.py \
  config/train_owt.py config/with_mhc_analysis.py config/small_model.py \
  --out_dir=out-owt-small-mhc --init_from=resume
```

After the analysis run, results will be saved to `log_out/infos.pkl`. Then run:

```sh
python -m analyze.h_and_nu
```
This produces the analysis figures in `analyze/`.

## Acknowledgements

This codebase is adapted from [nanoGPT](https://github.com/karpathy/nanoGPT).

Our Hyper-Connection implementation is from the [hyper-connections](https://github.com/lucidrains/hyper-connections) library. Note that `hyper-connections` also provides an mHC implementation; however, since it does not exactly match the details described in the mHC paper, we implemented our own version.

We would also like to thank [this mHC reproduction](https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections), which (to the best of our knowledge) is the earliest public reproduction of mHC. While we do not directly use code from it, some of our early explorations were inspired by that project.
