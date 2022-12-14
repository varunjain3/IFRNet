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
from models.IFRNet import gradient_penalty
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
    with torch.inference_mode():
        imgt_pred, loss_rec, loss_kl = model(img0, img1, embt, imgt, DIM)
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



def evaluate(args, ddp_generator, ddp_discriminator, dataloader_val, epoch, logger):
    loss_rec_list = []
    loss_vae_list = []
    loss_disc_list = []
    psnr_list = []
    time_stamp = time.time()

    disc_correct_cnt = 0
    total = 0
    ddp_discriminator.eval()
    ddp_generator.eval()
    for i, data in enumerate(dataloader_val):
        for l in range(len(data)):
            data[l] = data[l].to(args.device)
        img0, imgt, img1, embt = data

        with torch.inference_mode():
            imgt_pred, loss_rec, loss_kl = ddp_generator(img0, img1, embt, imgt, ret_loss = True)
            # gp = gradient_penalty(ddp_discriminator, imgt, imgt_pred, args.device)
            
            true_discriminator_out = ddp_discriminator(imgt)
            true_disc_loss = -torch.mean(true_discriminator_out)
            disc_correct_cnt += (torch.count_nonzero(true_discriminator_out > 0))

            gen_discriminator_out = ddp_discriminator(imgt_pred.detach())
            gen_disc_loss = torch.mean(gen_discriminator_out)
            disc_correct_cnt += (torch.count_nonzero(gen_discriminator_out < 0))
            total += 2 * imgt_pred.size(0)
            
            loss_disc = (true_disc_loss + gen_disc_loss) / 2

        loss_rec_list.append(loss_rec.cpu().numpy())
        loss_vae_list.append(loss_kl.cpu().numpy())
        loss_disc_list.append(loss_disc.cpu().numpy())

        for j in range(img0.shape[0]):
            psnr = calculate_psnr(imgt_pred[j].unsqueeze(0), imgt[j].unsqueeze(0)).cpu().data
            psnr_list.append(psnr)

    eval_time_interval = time.time() - time_stamp
    disc_accuracy = disc_correct_cnt / total
    logger.info('eval epoch:{}/{} time:{:.2f} loss_rec:{:.4e} loss_geo:{:.4e} loss_dis:{:.4e} psnr:{:.3f} disc_acc:{:.3f}'.format(
        epoch+1, args.epochs, eval_time_interval, 
        np.array(loss_rec_list).mean(), np.array(loss_vae_list).mean(), 
        np.array(loss_disc_list).mean(), np.array(psnr_list).mean(),
        disc_accuracy))
    return np.array(psnr_list).mean(), disc_accuracy


torch.autograd.set_detect_anomaly(True)

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

    if args.model_name == 'IFRNet':
        from models.IFRNet import IFRNet
    elif args.model_name == 'IFRNet_GB': # not supported
        from models.IFRNet_GB import IFRNet_GB
    elif args.model_name == 'IFRVAE': # not supported
        from models.IFRVAE import IFRVAE

    args.log_path = args.log_path
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