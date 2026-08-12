# WikiText-103 (raw), GPT-2 BPE — combine with a model + method config
# e.g. python train.py configN/small_model.py configN/with_mhc.py configN/train_wikitext103.py

eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

wandb_log = True
wandb_project = 'local-wikitext103'
wandb_group = 'wikitext103'

dataset = 'wikitext-103-raw-v1'
out_prefix_dataset = 'wikitext103'

gradient_accumulation_steps = 8
batch_size = 16
block_size = 1024

dtype = 'bfloat16'
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually