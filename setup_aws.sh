#!/bin/sh

# Install miniconda (conda already exists on the AMI image)
# curl -o https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
# bash Miniconda3-py39_4.12.0-Linux-x86_64.sh

# Create environment with python 3.7 and torch
conda create -n torch python=3.7
conda activate torch
conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install tqdm wandb

# Log in to wandb
wandb login '2fe54db29c2f3b04c92b08a6fa9872847ec64c80'

# Connect Google Drive
sudo add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
sudo apt-get update -qq 2>&1 > /dev/null
sudo apt -y install -qq google-drive-ocamlfuse 2>&1 > /dev/null
# google-drive-ocamlfuse
sudo apt-get install -qq w3m # to act as web browser 
# xdg-settings set default-web-browser w3m.desktop # to set default browser
mkdir drive
google-drive-ocamlfuse drive

# Download and unzip the dataset
cp drive/.shared/11785_Project/vimeo_triplet.zip /vimeo_triplet.zip
unzip -q vimeo_triplet.zip

# Clone IFRNet
# git clone https://github.com/varunjain3/IFRNet
# git checkout IFRVAN # TODO: change to branch of choice