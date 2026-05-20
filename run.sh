# fineweb_edu
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/small_model.py config/with_hc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/small_model.py config/with_mhc_lite.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/small_model.py config/with_mhc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/small_model.py  

torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/medium_model.py config/with_hc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/medium_model.py config/with_mhc_lite.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/medium_model.py config/with_mhc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/medium_model.py  

torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/large_model.py config/with_hc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/large_model.py config/with_mhc_lite.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/large_model.py config/with_mhc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/large_model.py  


# owt
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/small_model.py config/with_hc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/small_model.py config/with_mhc_light.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/small_model.py config/with_mhc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/small_model.py  

torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/medium_model.py config/with_hc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/medium_model.py config/with_mhc_lite.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/medium_model.py config/with_mhc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/medium_model.py  

torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/large_model.py config/with_hc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/large_model.py config/with_mhc_lite.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/large_model.py config/with_mhc.py 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/large_model.py  





