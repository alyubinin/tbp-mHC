wandb_group = "medium"
out_prefix_model = "medium"

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0

learning_rate = 6e-4
min_lr = 6e-5
max_iters = 10000
lr_decay_iters = 10000
warmup_iters = 200
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
