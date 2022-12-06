#!/bin/sh

# Install miniconda
curl -o https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
bash Miniconda3-py39_4.12.0-Linux-x86_64.sh

# Create environment with python 3.7 and torch
conda create -n torch python=3.7
conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia

# Activate the environment
conda activate torch

# Install tdqm, wandb
conda install tqdm wandb
conda deactivate

# Log in to wandb
wandb login '2fe54db29c2f3b04c92b08a6fa9872847ec64c80'

# Connect Google Drive
sudo add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
sudo apt-get update -qq 2>&1 > /dev/null
sudo apt -y install -qq google-drive-ocamlfuse 2>&1 > /dev/null
google-drive-ocamlfuse
sudo apt-get install -qq w3m # to act as web browser 
xdg-settings set default-web-browser w3m.desktop # to set default browser
cd /content
mkdir drive
cd drive
mkdir MyDrive
cd ..
cd ..
google-drive-ocamlfuse /content/drive/MyDrive

# Download and unzip the dataset
cp /content/drive/MyDrive/11785_Project/vimeo_triplet.zip /vimeo_triplet.zip
unzip -q vimeo_triplet.zip

# Clone IFRNet
# git clone https://github.com/varunjain3/IFRNet
# git checkout IFRVAN # TODO: change to branch of choice