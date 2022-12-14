import os
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import Vimeo90K_Train_Dataset, Vimeo90K_Test_Dataset
from metric import calculate_psnr, calculate_ssim
from utils import AverageMeter
import matplotlib.pyplot as plt
import logging
import wandb

def visualize_VAE(args, model):
    model.eval()

    dataset_train = Vimeo90K_Train_Dataset(dataset_dir=args.vimeo90k_dir, augment=False)
    # sampler = DistributedSampler(dataset_train)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True, shuffle=False)#, sampler=sampler)
    dataset_val = Vimeo90K_Test_Dataset(dataset_dir=args.vimeo90k_dir)
    dataloader_val = DataLoader(dataset_val, batch_size=16, num_workers=2, pin_memory=True, shuffle=False, drop_last=True)

    scaler1 = torch.cuda.amp.GradScaler()
    
    time_stamp = time.time()
    avg_rec = AverageMeter()
    avg_vae = AverageMeter()
    avg_gan = AverageMeter()
    best_psnr = 0.0

    # Tracked Variables (declaration)
    psnr = 0
    img0 = img1 = imgt = imgt_pred = None

    # Get one batch
    for i, data in enumerate(dataloader_train):
        for l in range(len(data)):
            data[l] = data[l].to(args.device)
        img0, imgt, img1, embt = data
        break
        # data_time_interval = time.time() - time_stamp
        # time_stamp = time.time()
        
        # with torch.cuda.amp.autocast():

    DIM = 0 # channel to be analyzed
    result = torch.empty((args.batch_size, 7, 3, img0.shape[2], img0.shape[3]))
    with torch.inference_mode():
        for i in range(7):
            imgt_pred = model(img0, img1, embt, imgt, DIM, i - 3)
            print(f"Batch shape: {imgt_pred.shape}")
            result[:, i, :, :, :] = imgt_pred
        # imgt_pred should be (B x 7 x W x H)

    # Result will be B x 8 grid of images. First column is ground truth
    # Subsequent columns walk along the Gaussian for one channel
    B = args.batch_size
    fig, = plt.figure()
    for i in range(7 + 1):
        for j in range(B):
            # Plot ground truth or the data samples
            ax = fig.add_subplot(B, 7, i * 8 + j)
            if i == 0:
                img_arr = imgt.numpy(force = True)
            else:
                img_arr = imgt_pred[j, i].numpy(force = True)
            imgplot = plt.imshow(img_arr)
            # Column labels
            if j == 0:
                if i == 0:
                    ax.set_title(f"Ground Truth")
                else:
                    ax.set_title(f"sigma = {i - 4}")
            if i == 0:
                ax.set_ylabel(f"Image {j}")

    fig.tight_layout()
    plt.show()
    
    # Evaluation
    # psnr = 0
    # if (epoch+1) % args.eval_interval == 0:
    #     psnr, disc_accuracy = evaluate(args, ddp_generator, ddp_discriminator, dataloader_val, epoch, logger)
    #     # TODO: run ssim test here

def main(args):
    # print(f"{args.local_rank}: before init_process_group")
    # dist.init_process_group(backend='nccl', world_size=args.world_size)
    # torch.cuda.set_device(args.local_rank)
    args.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu") #torch.device('cuda', args.local_rank)

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    # if args.model_name == 'IFRNet':
    #     from models.IFRNet import IFRNet
    # elif args.model_name == 'IFRNet_GB': # not supported
    #     from models.IFRNet_GB import IFRNet_GB
    if args.model_name == 'IFRVAE': # not supported
        from models.IFRVAE import IFRVAE

    args.num_workers = 4#10 #args.batch_size

    model = IFRVAE().to(args.device)
    
    # Load the trained checkpoints
    model.load_state_dict(torch.load(args.resume_path + f"/{args.model_name}_latest_gen.pth", map_location='cpu'))
    
    visualize_VAE(args, model)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IFRNet')
    parser.add_argument('--model_name', default='IFRNet', type=str, help='IFRNet, IFRNet_GB, IFRVAE, IFRVAN')
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--resume_path', default='checkpoint', type=str)
    parser.add_argument('--vimeo90k_dir', default='/ocean/projects/cis220078p/vjain1/data/vimeo_triplet', type=str) # TODO: change to data directory)
    args = parser.parse_args()

    main(args)