import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import warp, get_robust_weight
from loss import *


def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)


def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.PReLU(out_channels)
    )


class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out[:, -self.side_channels:, :, :] = self.conv2(out[:, -self.side_channels:, :, :].clone())
        out = self.conv3(out)
        out[:, -self.side_channels:, :, :] = self.conv4(out[:, -self.side_channels:, :, :].clone())
        out = self.prelu(x + self.conv5(out))
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(3, 32, 3, 2, 1), 
            convrelu(32, 32, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(32, 48, 3, 2, 1), 
            convrelu(48, 48, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(48, 72, 3, 2, 1), 
            convrelu(72, 72, 3, 1, 1)
        )
        self.pyramid4 = nn.Sequential(
            convrelu(72, 96, 3, 2, 1), 
            convrelu(96, 96, 3, 1, 1)
        )
        
    def forward(self, img):
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4


class Decoder4(nn.Module):
    def __init__(self):
        super(Decoder4, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(192+1, 192), 
            ResBlock(192, 32), 
            # MODIFIED: Force Decoder 4 to output mean and variance of the features
            nn.ConvTranspose2d(192, 72* 2 + 2 * 2, 4, 2, 1, bias=True)
        )
        
    def forward(self, f0, f1, embt):
        b, c, h, w = f0.shape
        embt = embt.repeat(1, 1, h, w)
        f_in = torch.cat([f0, f1, embt], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder3(nn.Module):
    def __init__(self):
        super(Decoder3, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(220, 216), 
            ResBlock(216, 32), 
            nn.ConvTranspose2d(216, 52, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(148, 144), 
            ResBlock(144, 32), 
            nn.ConvTranspose2d(144, 36, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(100, 96), 
            ResBlock(96, 32), 
            nn.ConvTranspose2d(96, 8, 4, 2, 1, bias=True)
        )
        
    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out

class ConvNeXt_Block(torch.nn.Module):

    def __init__(self, chan_ct, prob, downsample = False):
        super().__init__()
        if downsample: # used at the boundary between block groups
            stride = 2
        else:
            stride = 1
            
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = chan_ct // stride, out_channels = chan_ct, kernel_size = 7, stride = stride, groups = chan_ct // stride, padding = 3, bias = True),
            torch.nn.BatchNorm2d(chan_ct),
            torch.nn.Conv2d(in_channels = chan_ct, out_channels = chan_ct * 4, kernel_size = 1, bias = True),
            torch.nn.GELU(), # TODO: Consider Prelu/LeakyReLU?
            torch.nn.Conv2d(in_channels = chan_ct * 4, out_channels = chan_ct, kernel_size = 1, bias = True),
        )
        self.conv.apply(self.init_blocks)

        self.StochDepth = torchvision.ops.StochasticDepth(p = prob, mode ='batch')

        if downsample:
            self.shortcut = torch.nn.Conv2d(in_channels = chan_ct // 2, out_channels = chan_ct, kernel_size = 2, stride = stride, groups = chan_ct // 2, bias = False)
            self.shortcut.apply(self.init_blocks)
        else:
            self.shortcut = torch.nn.Identity()

    def init_blocks(self, m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight, gain = 0.5 ** 0.5)#, mode = 'fan_out')

    # Initialize the projections to identity (theoretical ideal for residual learning).
    def init_shortcuts(self, m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight, gain = 0.5 ** 0.5)
    
    def forward(self, x):
        return self.shortcut(x) + self.StochDepth(self.conv(x))

class ConvNeXt(torch.nn.Module):
    def __init__(self, root_channels, block_pattern, p_pattern, num_classes = 1):
        super().__init__()

        # Init backbone: stem and body
        # Stem layer
        self.stem = torch.nn.Conv2d(in_channels = 3, out_channels = root_channels, kernel_size = 4, stride = 4, bias = True)
        self.stem.apply(self.init_conv)

        self.stem_norm = torch.nn.BatchNorm2d(root_channels)

        # Main Body
        blocks = []
        self.shortcuts = []
        downsample = False # first block does not downsample
        chan_ct = root_channels
        for i, block_count in enumerate(block_pattern):
            depth_p = p_pattern[i]
            for j in range(block_count):
                blocks.append(ConvNeXt_Block(chan_ct, depth_p, downsample))
            
                # All non-first blocks of the block group will not downsample
                if downsample:
                    downsample = False
            chan_ct *= 2
            downsample = True # make sure first block of next block group downsamples

        self.blocks = torch.nn.ModuleList(blocks)
        
        chan_ct //= 2
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.avg_norm = torch.nn.BatchNorm2d(chan_ct)
        self.flatten = torch.nn.Flatten()
        
        self.cls_layer = torch.nn.Linear(chan_ct, num_classes)
        self.cls_layer.apply(self.init_linear)

    def init_conv(self, m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight, gain = 0.5 ** 0.5)

    def init_linear(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain = 0.5 ** 0.5)

    def forward(self, x):
        # Pass through backbone
        printed = False
        z = x
        z = self.stem_norm(self.stem(z))
        if torch.any(torch.isnan(z)) and not printed:
            printed = True
            print(f"Nan after stem")

        for i, block in enumerate(self.blocks):
            z = block(z)
            if torch.any(torch.isnan(z)) and not printed:
                printed = True
                print(f"Nan in block {i}")
        z = self.avg_norm(self.avg_pool(z))
        if torch.any(torch.isnan(z)) and not printed:
            printed = True
            print(f"Nan after average")
        feats = self.flatten(z) # N x 768
        # weight of cls: 768 x 1
        # dz/dW = X (N x 768)
        # dz/dX = W^T (1 x 768)
        # dz/db = 1

        logits = self.cls_layer(feats)
        return logits

# Jensen-Shannon Divergence.
# Source: https://discuss.pytorch.org/t/jensen-shannon-divergence/2626/13
class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=False)

    # p - prob values between [0,1]. This is the ground truth label
    # q - logit values in the real values [-inf, inf]. This is the discriminator output
    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1)).sigmoid()
        m = (0.5 * (p + q))
        return 0.5 * (self.kl(m, p) + self.kl(m, q))

# Code credit to Aladdin Persson
# Source: https://www.youtube.com/watch?v=pG0QZ7OddX4&t=3s&ab_channel=AladdinPersson
def gradient_penalty(critic, real, fake, device):
    B, C, H, W = real.shape
    epsilon = torch.rand((B, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * epsilon + fake * (1- epsilon)

    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs = interpolated_images,
        outputs = mixed_scores,
        grad_outputs = torch.ones_like(mixed_scores),
        create_graph = True,
        retain_graph = True,
    )[0]

    gradient_v = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient_v.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

class IFRVAE(nn.Module):
    def __init__(self, local_rank=-1, lr=1e-4):
        super(IFRVAE, self).__init__()
        self.encoder = Encoder()
        self.decoder4 = Decoder4()
        self.decoder3 = Decoder3()
        self.decoder2 = Decoder2()
        self.decoder1 = Decoder1()
        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        self.rb_loss = Charbonnier_Ada()
        self.gc_loss = Geometry(3)

        # MODIFIED: For VAE reparametrization
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0
        
    # def inference(self, img0, img1, embt, scale_factor=1.0):
    #     mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
    #     img0 = img0 - mean_
    #     img1 = img1 - mean_

    #     img0_ = resize(img0, scale_factor=scale_factor)
    #     img1_ = resize(img1, scale_factor=scale_factor)

    #     f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_)
    #     f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_)

    #     out4 = self.decoder4(f0_4, f1_4, embt)
    #     up_flow0_4 = out4[:, 0:2]
    #     up_flow1_4 = out4[:, 2:4]
    #     # MODIFIED: VAE Variational Reparametrization
    #     # Source: https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b
    #     ft_3_mean = out4[:, 4:4+72]
    #     ft_3_var = torch.exp(out4[:, 4+72:4+72*2]) # snap to positive values
    #     ft_3_ = ft_3_mean + ft_3_var * self.N.sample(ft_3_mean.shape)

    #     out3 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
    #     up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
    #     up_flow1_3 = out3[:, 2:4] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
    #     ft_2_ = out3[:, 4:]

    #     out2 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
    #     up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
    #     up_flow1_2 = out2[:, 2:4] + 2.0 * resize(up_flow1_3, scale_factor=2.0)
    #     ft_1_ = out2[:, 4:]

    #     out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
    #     up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)
    #     up_flow1_1 = out1[:, 2:4] + 2.0 * resize(up_flow1_2, scale_factor=2.0)
    #     up_mask_1 = torch.sigmoid(out1[:, 4:5])
    #     up_res_1 = out1[:, 5:]

    #     up_flow0_1 = resize(up_flow0_1, scale_factor=(1.0/scale_factor)) * (1.0/scale_factor)
    #     up_flow1_1 = resize(up_flow1_1, scale_factor=(1.0/scale_factor)) * (1.0/scale_factor)
    #     up_mask_1 = resize(up_mask_1, scale_factor=(1.0/scale_factor))
    #     up_res_1 = resize(up_res_1, scale_factor=(1.0/scale_factor))

    #     img0_warp = warp(img0, up_flow0_1)
    #     img1_warp = warp(img1, up_flow1_1)
    #     imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_
    #     imgt_pred = imgt_merge + up_res_1
    #     imgt_pred = torch.clamp(imgt_pred, 0, 1)
    #     return imgt_pred

    def inference(self, img0, img1, embt, imgt, dim, z):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        # imgt_ = imgt - mean_

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1)
        # ft_1, ft_2, ft_3, ft_4 = self.encoder(imgt_)

        out4 = self.decoder4(f0_4, f1_4, embt)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        # MODIFIED: VAE Variational Reparametrization
        # Source: https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b
        ft_3_mean = out4[:, 4:4+72]
        ft_3_var = torch.exp(out4[:, 4+72:4+72*2]) # snap to positive values

        # Modify the sample
        sample = self.N.sample(ft_3_mean.shape) # (B x C x W x H)
        sample[:, dim] = z
        print(sample[0, dim])
        print(f"Variance is {ft_3_var[0, dim]}")

        # (B x 7 x C x W x H)
        ft_3_ = ft_3_mean + ft_3_var * sample

        # KL Loss: Constrain intermediate features to look like standard Normal distributions
        # New paradigm: the encoder should not just condense information about the input images but also fuse them
        # It should take both images in at once and then output information that is decoded into an intermediate frame
        # kl = (ft_3_var ** 2 + ft_3_mean ** 2 - torch.log(ft_3_var) - 1/2).mean()

        out3 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow1_3 = out3[:, 2:4] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        ft_2_ = out3[:, 4:]

        out2 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        up_flow1_2 = out2[:, 2:4] + 2.0 * resize(up_flow1_3, scale_factor=2.0)
        ft_1_ = out2[:, 4:]

        out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)
        up_flow1_1 = out1[:, 2:4] + 2.0 * resize(up_flow1_2, scale_factor=2.0)
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1 = out1[:, 5:]
        
        img0_warp = warp(img0, up_flow0_1)
        img1_warp = warp(img1, up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_
        imgt_pred = imgt_merge + up_res_1
        imgt_pred = torch.clamp(imgt_pred, 0, 1)
        # Reconstruction Loss: Enforces closeness between the target image and the predicted
        loss_rec = self.l1_loss(imgt_pred - imgt) + self.tr_loss(imgt_pred, imgt)

        # return imgt_pred, loss_rec, loss_geo, loss_dis
        return [imgt_pred, sample[0, dim], ft_3_var[0, dim]]

    def forward(self, img0, img1, embt, imgt, dim, z):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        # imgt_ = imgt - mean_

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1)
        # ft_1, ft_2, ft_3, ft_4 = self.encoder(imgt_)

        out4 = self.decoder4(f0_4, f1_4, embt)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        # MODIFIED: VAE Variational Reparametrization
        # Source: https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b
        ft_3_mean = out4[:, 4:4+72]
        ft_3_var = torch.exp(out4[:, 4+72:4+72*2]) # snap to positive values

        # Modify the sample
        sample = self.N.sample(ft_3_mean.shape) # (B x C x W x H)
        sample[:, dim] = z
        print(sample[0, dim])
        print(f"Variance is {ft_3_var[0, dim]}")

        # (B x 7 x C x W x H)
        ft_3_ = ft_3_mean + ft_3_var * sample

        # KL Loss: Constrain intermediate features to look like standard Normal distributions
        # New paradigm: the encoder should not just condense information about the input images but also fuse them
        # It should take both images in at once and then output information that is decoded into an intermediate frame
        # kl = (ft_3_var ** 2 + ft_3_mean ** 2 - torch.log(ft_3_var) - 1/2).mean()

        out3 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow1_3 = out3[:, 2:4] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        ft_2_ = out3[:, 4:]

        out2 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        up_flow1_2 = out2[:, 2:4] + 2.0 * resize(up_flow1_3, scale_factor=2.0)
        ft_1_ = out2[:, 4:]

        out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)
        up_flow1_1 = out1[:, 2:4] + 2.0 * resize(up_flow1_2, scale_factor=2.0)
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_res_1 = out1[:, 5:]
        
        img0_warp = warp(img0, up_flow0_1)
        img1_warp = warp(img1, up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_
        imgt_pred = imgt_merge + up_res_1
        imgt_pred = torch.clamp(imgt_pred, 0, 1)
        # Reconstruction Loss: Enforces closeness between the target image and the predicted
        loss_rec = self.l1_loss(imgt_pred - imgt) + self.tr_loss(imgt_pred, imgt)

        # return imgt_pred, loss_rec, loss_geo, loss_dis
        return [imgt_pred, sample[0, dim], ft_3_var[0, dim]]