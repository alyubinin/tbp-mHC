wandb_group = "small"
out_prefix_model = "small"

n_layer = 6
n_head = 8
n_embd = 512
dropout = 0.0

learning_rate = 1e-3
min_lr = 1e-4
max_iters = 10000
lr_decay_iters = 10000
warmup_iters = 200
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
