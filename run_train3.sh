#!/bin/sh


# Activate Anaconda work environment for OpenDrift
source /jet/home/${USER}/.bashrc
source activate torch 

python --version

cd /jet/home/aramakri/project/IFRNet
export OMP_NUM_THREADS=1
nohup bash scrap.sh &
# python -m torch.distributed.launch --nproc_per_node=2 train_vimeo90k.py --world_size 2 --model_name 'IFRNet' --epochs 50 --batch_size 48  --lr_start 1e-4 --lr_end 1e-5
python -m torch.distributed.launch --nproc_per_node=4 --master_port='29501' train_vimeo90k.py --world_size 4 --model_name 'IFRNet' --epochs 50 --batch_size 48  --lr_start 1e-4 --lr_end 1e-5