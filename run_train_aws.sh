#!/bin/sh


# Activate Anaconda work environment for OpenDrift
# source /.bashrc
source activate torch 

python --version

cd IFRNet
export OMP_NUM_THREADS=1
nohup bash scrap.sh &
# python -m torch.distributed.launch --nproc_per_node=2 train_vimeo90k.py --world_size 2 --model_name 'IFRNet' --epochs 50 --batch_size 48  --lr_start 1e-4 --lr_end 1e-5
python -m torch.distributed.launch --nproc_per_node=1 --master_port='29501' train_vimeo90k.py --world_size 1 --model_name 'IFRNet' --resume_epoch 34 --resume_path drive/.shared/11785_Project/checkpoints/IFRVAN1/120422_050846/ --vimeo90k_dir vimeo_triplet --epochs 50 --batch_size 48  --lr_start 1e-4 --lr_end 1e-5