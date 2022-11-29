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
import logging
import wandb


# Implements Cosine Annealing Scheduler
def get_lr(args, iters):
    ratio = 0.5 * (1.0 + np.cos(iters / (args.epochs * args.iters_per_epoch) * math.pi))
    lr = (args.lr_start - args.lr_end) * ratio + args.lr_end
    return lr


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def generate_true_labels(batch_size, label_smoothing):
    labels = torch.ones((batch_size,1))
    smoothing = torch.zeros((batch_size,1))
    if label_smoothing > 0:
        smoothing.random_(to=label_smoothing)
    return labels + smoothing

def train(args, ddp_generator,model, ddp_discriminator):
    model.train()
    local_rank = args.local_rank
    print('Distributed Data Parallel Training IFRNet on Rank {}'.format(local_rank))

    if local_rank == 0:
        os.makedirs(args.log_path, exist_ok=True)
        log_path = os.path.join(args.log_path, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        os.makedirs(log_path, exist_ok=True)
        logger = logging.getLogger()
        logger.setLevel('INFO')
        BASIC_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'
        DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
        chlr = logging.StreamHandler()
        chlr.setFormatter(formatter)
        chlr.setLevel('INFO')
        fhlr = logging.FileHandler(os.path.join(log_path, 'train.log'))
        fhlr.setFormatter(formatter)
        logger.addHandler(chlr)
        logger.addHandler(fhlr)
        logger.info(args)

    vimeo90k_dir = '/ocean/projects/cis220078p/vjain1/data/vimeo_triplet' # TODO: change to data directory
    dataset_train = Vimeo90K_Train_Dataset(dataset_dir=vimeo90k_dir, augment=True)
    sampler = DistributedSampler(dataset_train)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=sampler)
    args.iters_per_epoch = dataloader_train.__len__()
    iters = args.resume_epoch * args.iters_per_epoch
    
    dataset_val = Vimeo90K_Test_Dataset(dataset_dir=vimeo90k_dir)
    dataloader_val = DataLoader(dataset_val, batch_size=16, num_workers=2, pin_memory=True, shuffle=False, drop_last=True)

    gen_optimizer = optim.AdamW(ddp_generator.parameters(), lr=args.lr_start, weight_decay=0)
    # Additions: Discriminator optimizer
    disc_optimizer = optim.AdamW(ddp_discriminator.parameters(), lr=args.lr_start, weight_decay=0)
    GAN_loss = JSD()

    time_stamp = time.time()
    avg_rec = AverageMeter()
    avg_vae = AverageMeter()
    avg_gan = AverageMeter()
    best_psnr = 0.0

    for epoch in range(args.resume_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for i, data in enumerate(dataloader_train):
            for l in range(len(data)):
                data[l] = data[l].to(args.device)
            img0, imgt, img1, flow, embt = data
            print(f"Iteration {i}")
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            lr = get_lr(args, iters)
            set_lr(gen_optimizer, lr)
            set_lr(disc_optimizer, lr)

            gen_optimizer.zero_grad()
            disc_optimizer.zero_grad()
            # GAN Training Flow derived from 
            # https://www.run.ai/guides/deep-learning-for-computer-vision/pytorch-gan#GAN-Tutorial
            # If run into difficulties: use https://github.com/soumith/ganhacks
            # Generator Training Step
            print("Generator Training")
            imgt_pred, loss_rec, loss_kl = ddp_generator(img0, img1, embt, imgt, flow, ret_loss = True)
            discriminator_logits = ddp_discriminator(imgt_pred)
            true_labels = generate_true_labels(args.batch_size, args.label_smoothing).to(args.device).float()
            loss_adv = GAN_loss(true_labels, discriminator_logits)
            generator_loss = loss_rec + loss_kl + loss_adv
            generator_loss.backward()
            gen_optimizer.step()

            # Discriminator Training Step
            print("Discriminator Training")
            disc_optimizer.zero_grad()
            # gen_optimizer.zero_grad()
            true_labels2 = generate_true_labels(args.batch_size, args.label_smoothing).to(args.device)
            false_labels = 1-true_labels2

            # Training set
            full_training_set = torch.concat([imgt, imgt_pred.detach()])
            full_training_labels = torch.concat([true_labels2, false_labels])
            discriminator_out = ddp_discriminator(full_training_set)
            loss_disc = GAN_loss(full_training_labels, discriminator_out)
            # TODO: Check if this is the correct order of the arguments.
            
            loss_disc.backward()
            disc_optimizer.step()

            # loss = loss_rec + loss_kl
            # loss.backward()
            # gen_optimizer.step()

            avg_rec.update(loss_rec.cpu().data)
            avg_vae.update(loss_kl.cpu().data)
            avg_gan.update(loss_disc.cpu().data)
            train_time_interval = time.time() - time_stamp
            

            if (iters+1) % 100 == 0 and local_rank == 0:
                wandb.log({
                    "loss_rec": loss_rec,
                    "loss_kl": loss_kl,
                    "loss_dis": loss_disc,
                    # "loss": loss,
                    "example": [wandb.Image(img0[0]), wandb.Image(imgt[0]), wandb.Image(img1[0]), wandb.Image(imgt_pred[0])],
                    "flow": [wandb.Image(flow[0]), wandb.Image(embt[0])],
                    "epoch": epoch+1,
                    "iter": iters+1,
                    "lr" : lr,
                })
                logger.info('epoch:{}/{} iter:{}/{} time:{:.2f}+{:.2f} lr:{:.5e} loss_rec:{:.4e} loss_geo:{:.4e} loss_dis:{:.4e}'.format(
                    epoch+1, args.epochs, iters+1, args.epochs * args.iters_per_epoch,
                    data_time_interval, train_time_interval, lr, 
                    avg_rec.avg, avg_vae.avg, avg_gan.avg))
                avg_rec.reset()
                avg_vae.reset()
                avg_gan.reset()

            iters += 1
            time_stamp = time.time()

        if (epoch+1) % args.eval_interval == 0 and local_rank == 0:
            psnr = evaluate(args, ddp_generator, ddp_discriminator, GAN_loss, dataloader_val, epoch, logger)
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(ddp_generator.module.state_dict(), '{}/{}_{}.pth'.format(log_path, args.model_name, 'best'))
            torch.save(ddp_generator.module.state_dict(), '{}/{}_{}.pth'.format(log_path, args.model_name, 'latest'))

        dist.barrier()


def evaluate(args, ddp_generator, ddp_discriminator, GAN_loss, dataloader_val, epoch, logger):
    loss_rec_list = []
    loss_vae_list = []
    loss_disc_list = []
    psnr_list = []
    time_stamp = time.time()
    ddp_discriminator.eval()
    ddp_generator.eval()
    for i, data in enumerate(dataloader_val):
        for l in range(len(data)):
            data[l] = data[l].to(args.device)
        img0, imgt, img1, flow, embt = data

        with torch.inference_mode():
            imgt_pred, loss_rec, loss_kl = ddp_generator(img0, img1, embt, imgt, flow)
            true_labels = generate_true_labels(args.batch_size, 0).to(args.device)

            true_discriminator_out = ddp_discriminator(imgt)
            true_disc_loss = GAN_loss(true_labels, true_discriminator_out)

            gen_discriminator_out = ddp_discriminator(imgt_pred.detach())
            gen_disc_loss = GAN_loss(1-true_disc_loss, gen_discriminator_out)
            
            loss_disc = (true_disc_loss + gen_disc_loss) / 2

        loss_rec_list.append(loss_rec.cpu().numpy())
        loss_vae_list.append(loss_kl.cpu().numpy())
        loss_disc_list.append(loss_disc.cpu().numpy())

        for j in range(img0.shape[0]):
            psnr = calculate_psnr(imgt_pred[j].unsqueeze(0), imgt[j].unsqueeze(0)).cpu().data
            psnr_list.append(psnr)

    eval_time_interval = time.time() - time_stamp
    
    logger.info('eval epoch:{}/{} time:{:.2f} loss_rec:{:.4e} loss_geo:{:.4e} loss_dis:{:.4e} psnr:{:.3f}'.format(
        epoch+1, args.epochs, eval_time_interval, 
        np.array(loss_rec_list).mean(), np.array(loss_vae_list).mean(), 
        np.array(loss_disc_list).mean(), np.array(psnr_list).mean()))
    return np.array(psnr_list).mean()


torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='IFRNet')
    parser.add_argument('--model_name', default='IFRNet', type=str, help='IFRNet, IFRNet_L, IFRNet_S')
    parser.add_argument('--local_rank', default=1, type=int)
    parser.add_argument('--world_size', default=4, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--eval_interval', default=1, type=int)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--lr_start', default=1e-4, type=float)
    parser.add_argument('--lr_end', default=1e-5, type=float)
    parser.add_argument('--log_path', default='checkpoint', type=str)
    parser.add_argument('--resume_epoch', default=0, type=int)
    parser.add_argument('--resume_path', default=None, type=str)
    parser.add_argument('--label_smoothing', default=0, type=float)
    args = parser.parse_args()

    dist.init_process_group(backend='nccl', world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda', args.local_rank)

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    if args.model_name == 'IFRNet':
        from models.IFRNet import Generator, ConvNeXt, JSD
    elif args.model_name == 'IFRNet_L': # not supported
        from models.IFRNet_L import Generator
    elif args.model_name == 'IFRNet_S': # not supported
        from models.IFRNet_S import Generator

    args.log_path = args.log_path + '/' + args.model_name
    args.num_workers = 10 #args.batch_size

    model = Generator().to(args.device)
    
    if args.local_rank == 0:
        wandb.init(name="VAN1", project="IDL Project", entity="11785_cmu")
        wandb.watch(model)
    
    if args.resume_epoch != 0:
        model.load_state_dict(torch.load(args.resume_path, map_location='cpu'))
    # TODO: Will DDP react well to being used as a non-distributed model
    ddp_generator = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)#, find_unused_parameters=True)
    discriminator = ConvNeXt(96, [3,3,9,3], [0.0,0.0,0.0,0.0]).to(args.device)
    ddp_discriminator = DDP(discriminator, device_ids=[args.local_rank])#, find_unused_parameters=True)
    
    train(args, ddp_generator, model, ddp_discriminator)
    
    dist.destroy_process_group()
