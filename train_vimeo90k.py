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
from tqdm import tqdm

# Implements Cosine Annealing Scheduler
def get_lr(args, iters):
    epoch = iters / args.iters_per_epoch
    lr = 0
    if epoch < args.warm_up: # Linear warm-up from 1e-8
        lr = epoch * (args.lr_start - 1e-8) / 20 + 1e-8
    else:
        iters -= args.warm_up * args.iters_per_epoch
        ratio = 0.5 * (1.0 + np.cos(iters / (args.epochs * args.iters_per_epoch) * math.pi))
        lr = (args.lr_start - args.lr_end) * ratio + args.lr_end
    return lr


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args, ddp_generator,model):
    # ddp_generator.train()
    local_rank = args.local_rank
    print('Distributed Data Parallel Training IFRNet on Rank {}'.format(local_rank))
    print(f"wandb run id is {args.resume_run_id}")

    if local_rank == 0:
        os.makedirs(args.log_path, exist_ok=True)
        log_path = os.path.join(args.log_path, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
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

    dataset_train = Vimeo90K_Train_Dataset(dataset_dir=args.vimeo90k_dir, augment=True)
    sampler = DistributedSampler(dataset_train)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=sampler)
    args.iters_per_epoch = dataloader_train.__len__()
    iters = args.resume_epoch * args.iters_per_epoch
    
    dataset_val = Vimeo90K_Test_Dataset(dataset_dir=args.vimeo90k_dir)
    dataloader_val = DataLoader(dataset_val, batch_size=16, num_workers=2, pin_memory=True, shuffle=False, drop_last=True)

    gen_optimizer = optim.AdamW(ddp_generator.parameters(), lr=args.lr_start, weight_decay=0)
    # Additions: Beta Scaling for Beta-VAE
    Beta = args.beta
    
    epoch = 0
    if args.evaluate_only:
        evaluate(args, ddp_generator, dataloader_val, epoch, logger)
        return
    
    scaler1 = torch.cuda.amp.GradScaler()
    
    time_stamp = time.time()
    avg_rec = AverageMeter()
    avg_vae = AverageMeter()
    best_psnr = 0.0

    # Tracked Variables (declaration)
    psnr = 0
    img0 = img1 = imgt = imgt_pred = None

    for epoch in range(args.resume_epoch, args.epochs):
        if local_rank == 0:
            logger.info(f"Epoch {epoch}")
        sampler.set_epoch(epoch)

        ddp_generator.train()

        if local_rank ==0:
            batch_bar = tqdm(total=len(dataloader_train), dynamic_ncols=True, leave=False, position=0, desc='Train')
        for i, data in enumerate(dataloader_train):
            if local_rank == 0:
                logger.info(f"Iteration {i}")
            for l in range(len(data)):
                data[l] = data[l].to(args.device)
            img0, imgt, img1, embt = data
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            lr = get_lr(args, iters)
            set_lr(gen_optimizer, lr)
            
            gen_optimizer.zero_grad()
            
            # GAN Training Flow derived from 
            # https://www.run.ai/guides/deep-learning-for-computer-vision/pytorch-gan#GAN-Tutorial
            # If run into difficulties: use https://github.com/soumith/ganhacks
            # Generator Training Step
            # with torch.cuda.amp.autocast():    
            imgt_pred, loss_rec, loss_kl = ddp_generator(img0, img1, embt, imgt, ret_loss = True)
            generator_loss = loss_rec + loss_kl * Beta

            generator_loss.backward()
            gen_optimizer.step()
            # scaler1.scale(generator_loss).backward()
            # scaler1.step(gen_optimizer)
            # scaler1.update()

            avg_rec.update(loss_rec.cpu().data)
            avg_vae.update(loss_kl.cpu().data)
            train_time_interval = time.time() - time_stamp

            iters += 1
            if local_rank ==0:
                batch_bar.set_postfix(
                    rec_loss = f"{avg_rec.avg: .6f}",
                    vae_loss = f"{avg_vae.avg: .6f}"
                    )
                batch_bar.update()
            time_stamp = time.time()
        if local_rank ==0:
            batch_bar.close()
        print(f"Rank: {local_rank}, Closed batch bar")
        
        # Evaluation
        psnr = 0
        if (epoch+1) % args.eval_interval == 0:
            psnr = evaluate(args, ddp_generator, dataloader_val, epoch, logger)
            if local_rank == 0 and psnr > best_psnr:
                best_psnr = psnr
                torch.save(ddp_generator.module.state_dict(), '{}/{}_{}.pth'.format(log_path, args.model_name, 'best_gen'))
            torch.save(ddp_generator.module.state_dict(), '{}/{}_{}.pth'.format(log_path, args.model_name, 'latest_gen'))

        # Wandb logging
        if local_rank == 0:
            wandb.log({
                "loss_rec": avg_rec.avg,
                "loss_kl": avg_vae.avg,
                "psnr": psnr,
                "example": [wandb.Image(img0[0]), wandb.Image(imgt[0]), wandb.Image(img1[0]), wandb.Image(imgt_pred[0])],
                # "flow": [wandb.Image(flow[0])],#, wandb.Image(embt[0])],
                "epoch": epoch+1,
                "iter": iters+1,
                "lr" : lr,
            })
            logger.info('epoch:{}/{} iter:{}/{} time:{:.2f}+{:.2f} lr:{:.5e}'
                'loss_rec:{:.4e} loss_vae:{:.4e} '.format(
                epoch+1, args.epochs, iters+1, args.epochs * args.iters_per_epoch,
                data_time_interval, train_time_interval, lr, 
                avg_rec.avg, avg_vae.avg))
            avg_rec.reset()
            avg_vae.reset()

        dist.barrier()


def evaluate(args, ddp_generator, dataloader_val, epoch, logger):
    loss_rec_list = []
    loss_vae_list = []
    psnr_list = []
    time_stamp = time.time()

    ddp_generator.eval()
    for i, data in enumerate(dataloader_val):
        for l in range(len(data)):
            data[l] = data[l].to(args.device)
        img0, imgt, img1, embt = data

        with torch.inference_mode():
            imgt_pred, loss_rec, loss_kl = ddp_generator(img0, img1, embt, imgt, ret_loss = True)

        loss_rec_list.append(loss_rec.cpu().numpy())
        loss_vae_list.append(loss_kl.cpu().numpy())

        for j in range(img0.shape[0]):
            psnr = calculate_psnr(imgt_pred[j].unsqueeze(0), imgt[j].unsqueeze(0)).cpu().data
            psnr_list.append(psnr)

    eval_time_interval = time.time() - time_stamp
    logger.info('eval epoch:{}/{} time:{:.2f} loss_rec:{:.4e} loss_vae:{:.4e} psnr:{:.3f}'.format(
        epoch+1, args.epochs, eval_time_interval, 
        np.array(loss_rec_list).mean(), np.array(loss_vae_list).mean(), 
         np.array(psnr_list).mean()))
    return np.array(psnr_list).mean()


torch.autograd.set_detect_anomaly(True)

def main(args):
    print(f"{args.local_rank}: before init_process_group")
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
        from models.IFRNet import Generator
    elif args.model_name == 'IFRNet_L': # not supported
        from models.IFRNet_L import Generator
    elif args.model_name == 'IFRNet_S': # not supported
        from models.IFRNet_S import Generator

    args.log_path = args.log_path + '/' + args.model_name
    args.num_workers = 4#10 #args.batch_size

    model = Generator().to(args.device)
    if args.local_rank == 0:
        if args.resume_epoch > 0 and args.resume_run_id:
            wandb.init(name="VAE1", run_id = args.resume_run_id, resume="must", project="IDL Project", entity="11785_cmu")
        else:
            args.resume_run_id = wandb.util.generate_id()
            wandb.init(name="VAE1", project="IDL Project", entity="11785_cmu")
        wandb.watch(model)
    
    if args.resume_epoch != 0:
        model.load_state_dict(torch.load(args.resume_path + f"/{args.model_name}_best_gen.pth", map_location='cpu'))

    # TODO: Will DDP react well to being used as a non-distributed model
    ddp_generator = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)#, find_unused_parameters=True)
    
    train(args, ddp_generator, model)
    
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IFRNet')
    parser.add_argument('--model_name', default='IFRNet', type=str, help='IFRNet, IFRNet_L, IFRNet_S')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--world_size', default=4, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--eval_interval', default=1, type=int)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--warm_up', default=0, type=float)
    parser.add_argument('--lr_start', default=1e-4, type=float)
    parser.add_argument('--lr_end', default=1e-5, type=float)
    parser.add_argument('--log_path', default='checkpoint', type=str)
    parser.add_argument('--resume_epoch', default=0, type=int)
    parser.add_argument('--resume_path', default=None, type=str)
    parser.add_argument('--label_smoothing', default=0, type=float)
    parser.add_argument('--vimeo90k_dir', default='/ocean/projects/cis220078p/vjain1/data/vimeo_triplet', type=str) # TODO: change to data directory)
    parser.add_argument('--beta', default=1, type=int)
    parser.add_argument('--evaluate_only', default=False, type=bool)
    parser.add_argument('--resume_run_id', default='', type=str)
    args = parser.parse_args()

    main(args)