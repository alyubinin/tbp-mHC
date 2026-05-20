# OpenBookQA dataset configuration (use with config/small_model.py)
eval_interval = 100
eval_iters = 50
log_interval = 10
always_save_checkpoint = False  # only save when val improves (small dataset)

wandb_log = True
wandb_project = 'mhc-lite'
out_prefix_dataset = "openbookqa"

dataset = 'openbookqa'
gradient_accumulation_steps = 4
batch_size = 16
block_size = 512  # QA examples are short

dtype = 'bfloat16'
