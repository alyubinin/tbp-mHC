"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from hyper_conn import clamp_ltbp_params
from pprint import pprint
import warnings
import json
import glob
import random 

# suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# ----- hyper conn start -----
hyper_conn_type = "none" # none, hc, mhc, mhc_lite, kromhc, tbp_mhc, atbp_mhc, stbp_mhc, mstbp_mhc, ltbp_mhc, altbp_mhc, lmaltbp_mhc, astbp_mhc, amstbp_mhc, lmamstbp_mhc, analysis
hyper_conn_n = 1 # num_streams

# ----- hyper conn end -----

# I/O
seed = 1337
out_prefix_dataset = ""
out_prefix_model = ""
out_prefix_method = "residual"
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'owt'
wandb_run_name = 'exp' # 'run' + str(time.time())
wandb_group = "default"
wandb_notes = ""
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
use_loss_mask = False # if True, load loss masks from {split}_mask.bin (for ARC-style training)
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# -----------------------------------------------------------------------------
# ORTBP2N-mHC training knobs (Phase 5 — all overridable via config/*.py or --key=value)
# -----------------------------------------------------------------------------
# ortbp_use_custom_optimizer: if True, ORTBP params get dedicated AdamW param groups.
# ortbp_lr_mult: uniform scale on all ORTBP group LRs (multiplies the per-subgroup mults below).
# ortbp_residual_chart_lr_mult: LR scale for static_alpha_res + dynamic_res_alpha_fn vs learning_rate.
# ortbp_residual_scale_lr_mult: LR scale for residual_scale vs learning_rate.
# ortbp_delta_lr_mult: LR scale for delta_logit vs learning_rate.
# ortbp_beta1 / ortbp_beta2: Adam betas for ORTBP-only groups (defaults match global beta1/beta2).
# ortbp_use_custom_grad_clip: if True, use subgroup clipping (Phase 3).
# ortbp_grad_clip: max norm for residual chart + residual_scale params.
# ortbp_delta_grad_clip: max norm for delta_logit.
# ortbp_log_stats: if True, ORTBP layers record diagnostics; train logs ortbp/* metrics (Phase 4).
# -----------------------------------------------------------------------------
ortbp_use_custom_optimizer = False
ortbp_lr_mult = 1.0
ortbp_residual_chart_lr_mult = 1.0
ortbp_residual_scale_lr_mult = 1.0
ortbp_delta_lr_mult = 1.0
ortbp_beta1 = 0.9
ortbp_beta2 = 0.95
ortbp_use_custom_grad_clip = False
ortbp_grad_clip = 0.3
ortbp_delta_grad_clip = 0.05
ortbp_log_stats = False
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(
        project = wandb_project , 
        name    = wandb_run_name + '-' + str(random.randint(1000,9999)), 
        group   = wandb_group,
        config  = config , 
        notes   = wandb_notes,
    )

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if out_dir == 'out':
    out_dir = f"out-{out_prefix_dataset}-{out_prefix_model}-{out_prefix_method}"
if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)

# Check if loss mask files exist (for ARC-style training)
train_mask_path = os.path.join(data_dir, 'train_mask.bin')
val_mask_path = os.path.join(data_dir, 'val_mask.bin')
has_loss_mask = use_loss_mask and os.path.exists(train_mask_path) and os.path.exists(val_mask_path)
if use_loss_mask and not has_loss_mask:
    print(f"Warning: use_loss_mask=True but mask files not found in {data_dir}")

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        mask_path = train_mask_path
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        mask_path = val_mask_path
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    # Load loss mask if available
    mask = None
    if has_loss_mask:
        mask_data = np.memmap(mask_path, dtype=np.uint8, mode='r')
        # Mask for targets (shifted by 1 from input)
        mask = torch.stack([torch.from_numpy((mask_data[i+1:i+1+block_size]).astype(np.float32)) for i in ix])
    
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        if mask is not None:
            mask = mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        if mask is not None:
            mask = mask.to(device)
    
    return x, y, mask

# -----------------------------------------------------------------------------
# Batch sampling (simple random contiguous windows)




# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta = {}
meta_vocab_size = None
train_tokens_per_byte = None
val_tokens_per_byte = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta.get('vocab_size')
    train_tokens_per_byte = meta.get('train_tokens_per_byte')
    val_tokens_per_byte = meta.get('val_tokens_per_byte')
    if meta_vocab_size is not None:
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
vocab_size = meta_vocab_size

# model init
model_args = dict(
    n_layer=n_layer, 
    n_head=n_head, 
    n_embd=n_embd, 
    block_size=block_size,
    bias=bias, 
    vocab_size=vocab_size, 
    dropout=dropout,
    hyper_conn_n=hyper_conn_n,
    hyper_conn_type=hyper_conn_type
)
if hyper_conn_type == "ortbp2n_mhc":
    model_args["ortbp_log_stats"] = globals().get("ortbp_log_stats", False)
if hyper_conn_type in {"atbp_mhc", "alsb_mhc"}:
    model_args["atbp_permutations"] = globals().get("atbp_permutations", globals().get("alsb_permutations", None))
if hyper_conn_type == "astbp_mhc":
    model_args["astbp_permutations"] = globals().get("astbp_permutations", None)
if hyper_conn_type == "amstbp_mhc":
    model_args["amstbp_permutations"] = globals().get("amstbp_permutations", None)
if hyper_conn_type == "lmamstbp_mhc":
    model_args["amstbp_permutations"] = globals().get("amstbp_permutations", None)
    model_args["lmamstbp_lambda_init"] = globals().get("lmamstbp_lambda_init", None)
    model_args["lmamstbp_mu_init"] = globals().get("lmamstbp_mu_init", None)
if hyper_conn_type == "altbp_mhc":
    model_args["altbp_permutations"] = globals().get("altbp_permutations", globals().get("atbp_permutations", globals().get("alsb_permutations", None)))
if hyper_conn_type == "lmaltbp_mhc":
    model_args["altbp_permutations"] = globals().get("altbp_permutations", globals().get("atbp_permutations", globals().get("alsb_permutations", None)))
    model_args["lmaltbp_lambda_init"] = globals().get("lmaltbp_lambda_init", None)
    model_args["lmaltbp_mu_init"] = globals().get("lmaltbp_mu_init", None)
if master_process:
    print ("="*100)
    for k, v in model_args.items():
        print (f"{k} = {v}")
    print ("="*100)

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    if 'atbp_permutations' in checkpoint_model_args:
        model_args['atbp_permutations'] = checkpoint_model_args['atbp_permutations']
    elif 'alsb_permutations' in checkpoint_model_args:
        model_args['atbp_permutations'] = checkpoint_model_args['alsb_permutations']
    if 'astbp_permutations' in checkpoint_model_args:
        model_args['astbp_permutations'] = checkpoint_model_args['astbp_permutations']
    if 'amstbp_permutations' in checkpoint_model_args:
        model_args['amstbp_permutations'] = checkpoint_model_args['amstbp_permutations']
    if 'lmamstbp_lambda_init' in checkpoint_model_args:
        model_args['lmamstbp_lambda_init'] = checkpoint_model_args['lmamstbp_lambda_init']
    if 'lmamstbp_mu_init' in checkpoint_model_args:
        model_args['lmamstbp_mu_init'] = checkpoint_model_args['lmamstbp_mu_init']
    if 'altbp_permutations' in checkpoint_model_args:
        model_args['altbp_permutations'] = checkpoint_model_args['altbp_permutations']
    if 'lmaltbp_lambda_init' in checkpoint_model_args:
        model_args['lmaltbp_lambda_init'] = checkpoint_model_args['lmaltbp_lambda_init']
    if 'lmaltbp_mu_init' in checkpoint_model_args:
        model_args['lmaltbp_mu_init'] = checkpoint_model_args['lmaltbp_mu_init']
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# print and log trainable parameters
if master_process:
    from hyper_conn import count_parameters
    
    # Collect trainable parameter info
    param_info = [
        (name, tuple(p.shape), p.numel())
        for name, p in model.named_parameters() if p.requires_grad
    ]
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = sum(numel for _, _, numel in param_info)
    hc_params = sum(numel for name, _, numel in param_info if 'hc_' in name)
    
    print(f"\nModel parameters:")
    print(f"  Total:     {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    if hc_params > 0:
        print(f"  Hyper-conn: {hc_params:,} ({100*hc_params/trainable_params:.1f}% of trainable)")
    
    # Log to wandb
    if wandb_log:
        import wandb
        # Log counts to config/summary
        wandb.config.update({
            "params/total": total_params,
            "params/trainable": trainable_params,
            "params/hyper_conn": hc_params,
        }, allow_val_change=True)
        wandb.run.summary["params/total"] = total_params
        wandb.run.summary["params/trainable"] = trainable_params
        wandb.run.summary["params/hyper_conn"] = hc_params
        
        # Log parameter names as a table
        param_table = wandb.Table(columns=["name", "shape", "numel"])
        for name, shape, numel in param_info:
            param_table.add_data(name, str(shape), numel)
        wandb.log({"params/trainable_list": param_table}, step=0)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

def get_optimizer_group_overrides():
    """
    Optional per-module optimizer overrides for variants that expose named groups.

    For ORTBP2N this lets the residual chart, residual scale, and delta scalar
    use separate learning-rate multipliers and betas while leaving the rest of
    the model on the default optimizer settings.
    """
    if hyper_conn_type != "ortbp2n_mhc" or not ortbp_use_custom_optimizer:
        return None

    ortbp_betas = (ortbp_beta1, ortbp_beta2)
    m = ortbp_lr_mult
    return {
        "ortbp_residual_chart": {
            "lr": learning_rate * m * ortbp_residual_chart_lr_mult,
            "weight_decay": 0.0,
            "betas": ortbp_betas,
        },
        "ortbp_residual_scale": {
            "lr": learning_rate * m * ortbp_residual_scale_lr_mult,
            "weight_decay": 0.0,
            "betas": ortbp_betas,
        },
        "ortbp_delta": {
            "lr": learning_rate * m * ortbp_delta_lr_mult,
            "weight_decay": 0.0,
            "betas": ortbp_betas,
        },
    }

def set_optimizer_group_lrs(optimizer, base_lr):
    """
    Update optimizer-group learning rates while preserving ORTBP-specific
    multipliers for groups created by `module_param_group_overrides`.
    """
    for param_group in optimizer.param_groups:
        group_name = param_group.get("group_name", "")
        if group_name == "ortbp_residual_chart":
            param_group["lr"] = base_lr * ortbp_lr_mult * ortbp_residual_chart_lr_mult
        elif group_name == "ortbp_residual_scale":
            param_group["lr"] = base_lr * ortbp_lr_mult * ortbp_residual_scale_lr_mult
        elif group_name == "ortbp_delta":
            param_group["lr"] = base_lr * ortbp_lr_mult * ortbp_delta_lr_mult
        else:
            param_group["lr"] = base_lr

def get_optimizer_group_lrs(optimizer):
    """Collect current optimizer-group learning rates for logging."""
    return {
        param_group.get("group_name", f"group_{idx}"): param_group["lr"]
        for idx, param_group in enumerate(optimizer.param_groups)
    }

def collect_ortbp_stats(model):
    """
    Aggregate ORTBP diagnostics across all ORTBP modules in the model.

    Each ORTBP layer stores only the latest detached scalar stats from its most
    recent forward pass, so aggregation here is cheap and logging-friendly.
    """
    if hyper_conn_type != "ortbp2n_mhc" or not ortbp_log_stats:
        return {}

    stats_by_name = {}
    for module in model.modules():
        if not hasattr(module, "get_stats"):
            continue
        module_stats = module.get_stats()
        if not module_stats:
            continue
        for stat_name, stat_value in module_stats.items():
            stats_by_name.setdefault(stat_name, []).append(stat_value)

    return {
        f"ortbp/{stat_name}": float(np.mean(stat_values))
        for stat_name, stat_values in stats_by_name.items()
        if len(stat_values) > 0
    }

def clip_gradients(optimizer, model):
    """
    Clip gradients with optional ORTBP-specific subgroup thresholds.

    Default behavior matches the old global clip. When ORTBP custom clipping is
    enabled, the optimizer groups form a disjoint partition:
    - default_decay + default_nodecay use the global grad_clip
    - ortbp_residual_chart + ortbp_residual_scale use ortbp_grad_clip
    - ortbp_delta uses ortbp_delta_grad_clip
    """
    grad_stats = {
        "global": -1.0,
        "default": -1.0,
        "ortbp": -1.0,
        "ortbp_delta": -1.0,
    }

    use_custom_ortbp_clip = (
        hyper_conn_type == "ortbp2n_mhc"
        and ortbp_use_custom_optimizer
        and ortbp_use_custom_grad_clip
    )

    if not use_custom_ortbp_clip:
        if grad_clip > 0.0:
            ret_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_stats["global"] = float(ret_grad_norm)
        return grad_stats

    optimizer_groups = {
        param_group.get("group_name", f"group_{idx}"): list(param_group["params"])
        for idx, param_group in enumerate(optimizer.param_groups)
    }

    default_params = (
        optimizer_groups.get("default_decay", [])
        + optimizer_groups.get("default_nodecay", [])
    )
    ortbp_params = (
        optimizer_groups.get("ortbp_residual_chart", [])
        + optimizer_groups.get("ortbp_residual_scale", [])
    )
    ortbp_delta_params = optimizer_groups.get("ortbp_delta", [])

    squared_norm = 0.0

    if grad_clip > 0.0 and default_params:
        default_norm = torch.nn.utils.clip_grad_norm_(default_params, grad_clip)
        grad_stats["default"] = float(default_norm)
        squared_norm += grad_stats["default"] ** 2

    if ortbp_grad_clip > 0.0 and ortbp_params:
        ortbp_norm = torch.nn.utils.clip_grad_norm_(ortbp_params, ortbp_grad_clip)
        grad_stats["ortbp"] = float(ortbp_norm)
        squared_norm += grad_stats["ortbp"] ** 2

    if ortbp_delta_grad_clip > 0.0 and ortbp_delta_params:
        ortbp_delta_norm = torch.nn.utils.clip_grad_norm_(
            ortbp_delta_params,
            ortbp_delta_grad_clip,
        )
        grad_stats["ortbp_delta"] = float(ortbp_delta_norm)
        squared_norm += grad_stats["ortbp_delta"] ** 2

    if squared_norm > 0.0:
        grad_stats["global"] = math.sqrt(squared_norm)

    return grad_stats

# optimizer
optimizer_group_overrides = get_optimizer_group_overrides()
optimizer = model.configure_optimizers(
    weight_decay,
    learning_rate,
    (beta1, beta2),
    device_type,
    module_param_group_overrides=optimizer_group_overrides,
)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, mask = get_batch(split)
            with ctx:
                logits, loss = model(X, Y, loss_mask=mask)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# training loop
X, Y, mask = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
train_losses = []
tpss = [] # token per second
grad_norms = []
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    set_optimizer_group_lrs(optimizer, lr)
    optimizer_group_lrs = get_optimizer_group_lrs(optimizer)

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        ortbp_stats = collect_ortbp_stats(raw_model)
        avg_train_loss = [np.mean(train_losses), np.std(train_losses)] if len(train_losses) > 0 else [0, 0]
        avg_grad_norm = [np.mean(grad_norms), np.std(grad_norms)] if len(grad_norms) > 0 else [0, 0]
        train_bpb = losses['train'] / math.log(2) * train_tokens_per_byte if train_tokens_per_byte is not None else None
        val_bpb = losses['val'] / math.log(2) * val_tokens_per_byte if val_tokens_per_byte is not None else None

        desc_parts = [
            f"train loss: {losses['train']:.4f}" , 
            f"val loss: {losses['val']:.4f}" , 
        ]
        if train_bpb is not None:
            desc_parts.append(f"train BPB: {train_bpb:.4f}")
        if val_bpb is not None:
            desc_parts.append(f"val BPB: {val_bpb:.4f}")
        desc_parts.extend([
            f"(avg train loss: {avg_train_loss[0]:.4f} ± {avg_train_loss[1]:.4f})" , 
            f"(avg grad norm: {avg_grad_norm[0]:.4f} ± {avg_grad_norm[1]:.4f})" , 
        ])
        desc = f"[step {iter_num}]" + ", ".join(desc_parts)
        print(desc)


        if wandb_log:
            log_data = {
                "iter": iter_num,
                "train/loss_est": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
                "train/avg_loss": avg_train_loss[0],
                "train/avg_gnorm": avg_grad_norm[0],
            }
            if ortbp_use_custom_optimizer:
                log_data.update({
                    f"lr/{group_name}": group_lr
                    for group_name, group_lr in optimizer_group_lrs.items()
                })
            if ortbp_stats:
                log_data.update(ortbp_stats)
            if train_bpb is not None:
                log_data["train/bpb"] = train_bpb
            if val_bpb is not None:
                log_data["val/bpb"] = val_bpb
            wandb.log(log_data, step=iter_num)
        if (losses['val'] < best_val_loss or always_save_checkpoint) and not eval_only:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if eval_only:
        break
    
    train_time_start = time.time()
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y, loss_mask=mask)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, mask = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    grad_norm = -1.
    ortbp_grad_norm = -1.
    ortbp_delta_norm = -1.
    should_unscale = grad_clip > 0. or (
        hyper_conn_type == "ortbp2n_mhc"
        and ortbp_use_custom_optimizer
        and ortbp_use_custom_grad_clip
    )
    should_clip = grad_clip > 0. or (
        hyper_conn_type == "ortbp2n_mhc"
        and ortbp_use_custom_optimizer
        and ortbp_use_custom_grad_clip
    )
    if should_unscale:
        scaler.unscale_(optimizer)
        if should_clip:
            grad_clip_stats = clip_gradients(optimizer, model)
            grad_norm = grad_clip_stats["global"]
            ortbp_grad_norm = grad_clip_stats["ortbp"]
            ortbp_delta_norm = grad_clip_stats["ortbp_delta"]
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # LTBP/ALTBP/LMALTBP-mHC: clamp t params to [0,1] after each optimizer step
    if hyper_conn_type in ["ltbp_mhc", "altbp_mhc", "lmaltbp_mhc"]:
        clamp_ltbp_params(raw_model)
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    train_time_end = time.time()
    d_train_time = train_time_end - train_time_start
    tokens_per_sec = tokens_per_iter / d_train_time
    
    tpss.append(tokens_per_sec)
    train_losses.append(loss.detach().item())
    grad_norms.append(float(grad_norm))
    if len(train_losses) > 200:
        train_losses.pop(0)
    if len(grad_norms) > 200:
        grad_norms.pop(0)


    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        
        tokens_seen = iter_num * tokens_per_iter

        desc = f"[iter {iter_num}]" + ", ".join([
            f"loss: {lossf:.4f}" , 
            f"tokens/sec: {np.mean(tpss):.2f} ± {np.std(tpss):.2f}" if len(tpss) > 0 else "0.00 ± 0.00" , 
            f"tokens seen: {tokens_seen}" , 
            f"grad norm: {grad_norm:.4f}" , 
        ])
        if wandb_log:
            ortbp_stats = collect_ortbp_stats(raw_model)
            log_data = {
                "train/loss": lossf,
                "train/tokens_per_sec": np.mean(tpss) if len(tpss) > 0 else 0,
                "train/tokens_seen": tokens_seen,
                "train/grad_norm": grad_norm,
            }
            if ortbp_use_custom_grad_clip:
                log_data["train/grad_norm_ortbp"] = ortbp_grad_norm
                log_data["train/grad_norm_ortbp_delta"] = ortbp_delta_norm
            if ortbp_stats:
                log_data.update(ortbp_stats)
            wandb.log(log_data, step=iter_num)

        print(desc)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
