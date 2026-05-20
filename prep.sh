apt install p7zip-full mc
pip install numpy transformers datasets tiktoken wandb tqdm einops gdown
cd /workspace
mkdir mhc-lite
mkdir mhc-lite/data
mkdir mhc-lite/data/openwebtext
cd /workspace/mhc-lite/data/openwebtext
gdown 1WoEcJuTsqDm3qrpiGz7foDTbchQrC_Fj
7z x openwebtext.7z

nvidia-smi
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"