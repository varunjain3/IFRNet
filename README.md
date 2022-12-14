# Exploring High Resolution Video Frame Interpolation
A PyTorch adaptation of [IFRNet](https://arxiv.org/abs/2205.14620) (CVPR 2022).

Authors: Varun Jain, Kevin Liu, , Aditya Ramakrishnan, Madankumar 

## Abstract
Over the past decade, a surge in video-based services has been observed in major industries from engineering to entertainment to telecommunications. As a consequence, several highly proficient learning-based video optimization techniques have been devised to meet the demand for optimization solutions. There are a lot of ways to optimize video data. In this work, we focus our experiments towards intermediate frame generation tasks, also known as Video Frame Interpolation (VFI). The goal of Video Frame Interpolation is to produce frame(s) between adjacent frames. Video Frame Interpolation can be used for a variety of applications ranging from video frame rate boosting, frame recovery in video streaming, and slow motion video capture. We take a compact and efficient net, IFRNet\cite{IFRNet}, which is among the State of the Art works for VFI. IFRNet is an encoder-decoder network employing optical flow field paradigms to achieve successful VFI for a variety of standard datasets. We further discuss what makes IFRNet special, and some methods that can be experimented to generate even more consistent frames.

![](./figures/vimeo90k.png)

## YouTube Demos
[[4K60p] うたわれるもの 偽りの仮面 OP フレーム補間+超解像 (IFRnetとReal-CUGAN)](https://www.youtube.com/watch?v=tV2imgGS-5Q)

[[4K60p] 天神乱漫 -LUCKY or UNLUCKY!?- OP (IFRnetとReal-CUGAN)](https://www.youtube.com/watch?v=NtpJqDZaM-4)

[RIFE IFRnet 比較](https://www.youtube.com/watch?v=lHqnOQgpZHQ)

[IFRNet frame interpolation](https://www.youtube.com/watch?v=ygSdCCZCsZU)

## Preparation
1. PyTorch >= 1.3.0 (We have verified that this repository supports Python 3.6/3.7, PyTorch 1.3.0/1.9.1).
2. Download training and test datasets: [Vimeo90K](http://toflow.csail.mit.edu/) - Triplet Dataset.

## Download Pre-trained Models and Play with Demos
Figures from left to right are overlaid input frames, 2x and 8x video interpolation results respectively.
<p float="left">
  <img src=./figures/img_overlaid.png width=270 />
  <img src=./figures/out_2x.gif width=270 />
  <img src=./figures/out_8x.gif width=270 /> 
</p>

1. Download our pre-trained models in this [link](https://www.dropbox.com/sh/hrewbpedd2cgdp3/AADbEivu0-CKDQcHtKdMNJPJa?dl=0), and then put file <code> checkpoints</code> into the root dir.

2. Run the following scripts to generate 2x and 8x frame interpolation demos
<pre><code>$ python demo_2x.py</code>
<code>$ python demo_8x.py</code></pre>


## Training on Vimeo90K Triplet Dataset for 2x Frame Interpolation
1. First, run this script to generate optical flow pseudo label
<pre><code>$ python generate_flow.py</code></pre>

2. Then, start training by executing one of the following commands with selected model
<pre><code>$ python -m torch.distributed.launch --nproc_per_node=4 train_vimeo90k.py --world_size 4 --model_name 'IFRNet' --epochs 300 --batch_size 6 --lr_start 1e-4 --lr_end 1e-5</code>
<code>$ python -m torch.distributed.launch --nproc_per_node=4 train_vimeo90k.py --world_size 4 --model_name 'IFRNet_L' --epochs 300 --batch_size 6 --lr_start 1e-4 --lr_end 1e-5</code>
<code>$ python -m torch.distributed.launch --nproc_per_node=4 train_vimeo90k.py --world_size 4 --model_name 'IFRNet_S' --epochs 300 --batch_size 6 --lr_start 1e-4 --lr_end 1e-5</code></pre>

## Benchmarks for 2x Frame Interpolation
To test running time and model parameters, you can run
<pre><code>$ python benchmarks/speed_parameters.py</code></pre>

To test frame interpolation accuracy on Vimeo90K, UCF101 and SNU-FILM datasets, you can run
<pre><code>$ python benchmarks/Vimeo90K.py</code>
<code>$ python benchmarks/UCF101.py</code>
<code>$ python benchmarks/SNU_FILM.py</code></pre>

## Quantitative Comparison for 2x Frame Interpolation
Proposed IFRNet achieves state-of-the-art frame interpolation accuracy with less inference time and computation complexity. We expect proposed single encoder-decoder joint refinement based IFRNet to be a useful component for many frame rate up-conversion, video compression and intermediate view synthesis systems. Time and FLOPs are measured on 1280 x 720 resolution.

![](./figures/benchmarks.png)


## Qualitative Comparison for 2x Frame Interpolation
Video comparison for 2x interpolation of methods using 2 input frames on SNU-FILM dataset.

![](./figures/fig2_1.gif)

![](./figures/fig2_2.gif)


## Middlebury Benchmark
Results on the [Middlebury](https://vision.middlebury.edu/flow/eval/results/results-i1.php) online benchmark.

![](./figures/middlebury.png)

Results on the Middlebury Other dataset.

![](./figures/middlebury_other.png)


## Training on GoPro Dataset for 8x Frame Interpolation
1. Start training by executing one of the following commands with selected model
<pre><code>$ python -m torch.distributed.launch --nproc_per_node=4 train_vimeo90k.py --world_size 4 --model_name 'IFRNet' --epochs 300 --batch_size 6 --lr_start 1e-4 --lr_end 1e-5</code>
<code>$ python -m torch.distributed.launch --nproc_per_node=4 train_vimeo90k.py --world_size 4 --model_name 'IFRNet_L' --epochs 300 --batch_size 6 --lr_start 1e-4 --lr_end 1e-5</code>
<code>$ python -m torch.distributed.launch --nproc_per_node=4 train_vimeo90k.py --world_size 4 --model_name 'IFRNet_S' --epochs 300 --batch_size 6 --lr_start 1e-4 --lr_end 1e-5</code></pre>

Since inter-frame motion in 8x interpolation setting is relatively small, task-oriented flow distillation loss is omitted here. Due to the GoPro training set is a relatively small dataset, we suggest to use your specific datasets to train slow-motion generation for better results.

## Quantitative Comparison for 8x Frame Interpolation

<img src=./figures/8x_interpolation.png width=480 />

## Qualitative Results on GoPro and Adobe240 Datasets for 8x Frame Interpolation
Each video has 9 frames, where the first and the last frames are input, and the middle 7 frames are predicted by IFRNet.

<p float="left">
  <img src=./figures/fig1_1.gif width=270 />
  <img src=./figures/fig1_2.gif width=270 />
  <img src=./figures/fig1_3.gif width=270 /> 
</p>
