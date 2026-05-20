eval_interval = 500
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

wandb_log = True
wandb_project = 'mhc-lite'
out_prefix_dataset = "finewebedu"

dataset = 'fineweb_edu'
gradient_accumulation_steps = 8
batch_size = 16 
block_size = 1024

dtype = 'bfloat16'
