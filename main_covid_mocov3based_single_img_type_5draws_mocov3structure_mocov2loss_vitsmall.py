#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from functools import partial
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter

# import moco.builder_vit_mocov3structure_mocov2loss_noprediction_q as builder_vit
import moco.builder_vit_mocov3structure_mocov2loss as builder_vit
import moco.loader
import moco.optimizer

import vits
from _internally_replaced_utils import load_state_dict_from_url

import aihc_utils.storage_util as storage_util
import aihc_utils.image_transform as image_transform


torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_small_ori', 'vit_base_ori', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


parser = argparse.ArgumentParser(description='MoCo ImageNet Pre-Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.6, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://192.168.1.6:10003', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=256, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--moco-mlp-dim', default=4096, type=int,
                    help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--moco-m', default=0.99, type=float,
                    help='moco momentum of updating momentum encoder (default: 0.99)')
parser.add_argument('--moco-m-cos', action='store_true',
                    help='gradually increase moco momentum to 1 with a '
                         'half-cycle cosine schedule')
parser.add_argument('--moco-t', default=1.0, type=float,
                    help='softmax temperature (default: 1.0)')

# vit specific configs:
parser.add_argument('--stop-grad-conv1', action='store_true',
                    help='stop-grad after first conv, or patch embedding')

# other upgrades
parser.add_argument('--optimizer', default='lars', type=str,
                    choices=['lars', 'adamw', 'adam'],
                    help='optimizer used (default: lars)')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--crop-min', default=0.08, type=float,
                    help='minimum scale for random cropping (default: 0.08)')

# modification

parser.add_argument('--exp-name', dest='exp_name', type=str, default='exp',
                    help='Experiment name')

parser.add_argument('--aug-setting', default='chexpert',
                    choices=['moco_v1', 'moco_v2', 'chexpert'],
                    help='version of data augmentation to use')

parser.add_argument('--train_data', metavar='DIR',
                    help='path to train folder')

parser.add_argument('--save-epoch', dest='save_epoch', type=int, default=30,
                    help='Number of epochs per checkpoint save')

# parser.add_argument('--from-imagenet', dest='from_imagenet', action='store_true',
#                     help='use pre-trained ImageNet model')

parser.add_argument('--img-size', dest='img_size', type=int, default=224,
                    help='image size (Chexpert=320)')

parser.add_argument('--crop', dest='crop', type=int, default=224,
                    help='image crop (Chexpert=320)')

parser.add_argument('--maintain-ratio', dest='maintain_ratio', 
                    default=True,
                    action='store_true',
                    help='whether to maintain aspect ratio or scale the image')

parser.add_argument('--rotate', dest='rotate', type=int, default=10,
                    help='degree to rotate image')

parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--schedule', default=[12, 18, 24], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')

def main():
    args = parser.parse_args()

    checkpoint_folder = storage_util.get_storage_folder(args.exp_name, f'moco{"v3"}')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, checkpoint_folder))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, checkpoint_folder)


def main_worker(gpu, ngpus_per_node, args, checkpoint_folder):
    with open(os.path.join(checkpoint_folder,'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    args.gpu = gpu

    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        
        ## not includded in previous##
        torch.distributed.barrier()

    # parameters
    # parameters
    # ALL_SEMI_RATIO =  [0.00075, 0.0009, 0.001, 0.0025, 0.005, 0.01, 0.1, 0.2, 0.3]
    ALL_SEMI_RATIO =  [1]#, 0.1, 0.2, 0.3] #miss 0.1 iteration 4(last)
    SEMI_ITERATIONS = { 0.0005: 5,
                        0.00075: 5,
                        0.0009: 5,
                        0.001: 5,
                        0.0025: 5,
                        0.005: 5,
                        0.01: 5,
                        0.1: 5,
                        0.2: 5,
                        0.3: 5,
                        0.5: 5,
                        0.7: 5,
                        0.9: 5,
                        1:5
                    } 
    
    
    
    for s in ALL_SEMI_RATIO:
        print ('ratio = {}'.format(s))
        for it in range(SEMI_ITERATIONS[s]):    
            print ('iteration = {}'.format(it))
            
            sub_checkpoint_folder = storage_util.get_storage_sub_folder(checkpoint_folder, s, it)
       
            # create model
            print("=> creating model '{}'".format(args.arch))
            if args.arch.startswith('vit'):
                model = builder_vit.MoCo_ViT(
                    partial(vits.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1), args,
                    args.moco_dim, args.moco_mlp_dim, args.moco_t)
            else:
                # base_encoder = 
                # state_dict = load_state_dict_from_url(model_urls[args.arch], progress=progress)
                # base_encoder.load_state_dict(state_dict)
                model = builder_vit.MoCo_ResNet(
                    partial(torchvision_models.__dict__[args.arch], zero_init_residual=True, pretrained=True), args, 
                    args.moco_dim, args.moco_mlp_dim, args.moco_t)
            
            ## not included in previous ##
            # infer learning rate before changing batch size
            if args.cos:
                print ('cos')
                lr = args.lr * args.batch_size / 4 #bs=32/8=4 or bs16/4=4 or bs=1024/256=4
            else:
                lr = args.lr
        
            if not torch.cuda.is_available():
                print('using CPU, this will be slow')
            elif args.distributed:
                # not included in previous
                # apply SyncBN
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) # not included in previous
                # For multiprocessing distributed, DistributedDataParallel constructor
                # should always set the single device scope, otherwise,
                # DistributedDataParallel will use all available devices.
                if args.gpu is not None:
                    torch.cuda.set_device(args.gpu)
                    model.cuda(args.gpu)
                    # When using a single GPU per process and per
                    # DistributedDataParallel, we need to divide the batch size
                    # ourselves based on the total number of GPUs we have
                    print ('world size = {}'.format(args.world_size))
                    print ('initial bs = {}'.format(args.batch_size))
                    batch_size = int(args.batch_size / args.world_size)
                    print ('paralled bs = {}'.format(batch_size))
                    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                else:
                    model.cuda()
                    # DistributedDataParallel will divide and allocate batch_size to all
                    # available GPUs if device_ids are not set
                    model = torch.nn.parallel.DistributedDataParallel(model)
            elif args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model = model.cuda(args.gpu)
                # comment out the following line for debugging
                raise NotImplementedError("Only DistributedDataParallel is supported.")
            else:
                # AllGather/rank implementation in this code only supports DistributedDataParallel.
                raise NotImplementedError("Only DistributedDataParallel is supported.")
            print(model) # print model after SyncBatchNorm
            print('Distributed model defined')
        
            # define loss function (criterion) and optimizer
            criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        
            print('Loss defined')
        
            if args.optimizer == 'lars':
                optimizer = moco.optimizer.LARS(model.parameters(), lr,
                                                weight_decay=args.weight_decay,
                                                momentum=args.momentum)
            elif args.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr,
                                        weight_decay=args.weight_decay)

            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr,
                                             betas=(0.9, 0.999),
                                             weight_decay=args.weight_decay)
            # print (lr)
            print('Optimizer defined')
            
            scaler = torch.cuda.amp.GradScaler() #not included in mocov2
            
            summary_writer = SummaryWriter(os.path.join(checkpoint_folder, 'tb_train_val_test_'+str(s)+'_'+str(it))) \
                if args.rank == 0 else None
            # summary_writer = SummaryWriter() if args.rank == 0 else None
        
            # optionally resume from a checkpoint
            if args.resume:
                if os.path.isfile(args.resume):
                    print("=> loading checkpoint '{}'".format(args.resume))
                    if args.gpu is None:
                        checkpoint = torch.load(args.resume)
                    else:
                        # Map model to be loaded to specified single gpu.
                        loc = 'cuda:{}'.format(args.gpu)
                        checkpoint = torch.load(args.resume, map_location=loc)
                    args.start_epoch = checkpoint['epoch']
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scaler.load_state_dict(checkpoint['scaler'])
                    print("=> loaded checkpoint '{}' (epoch {})"
                          .format(args.resume, checkpoint['epoch']))
                else:
                    print("=> no checkpoint found at '{}'".format(args.resume))
        
            cudnn.benchmark = True
        
            # Data loading code
            if s != 1:
                img_csv = 'create_covid_dataset/'+str(s)+'_unlabeled_train_'+str(it)+'.txt'
            else:
                img_csv = 'create_covid_dataset/'+str(s)+'_labeled_train_'+str(it)+'.txt'
            traindir = args.train_data
            # disease_name = args.class_name
            
            # traindir = os.path.join(args.data, 'train')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        
            # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
            if args.aug_setting == 'aug1':
                augmentation1 = [
                    transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=1.0),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]
            elif args.aug_setting == 'aug2':
                augmentation2 = [
                    transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.1),
                    transforms.RandomApply([moco.loader.Solarize()], p=0.2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]
            elif args.aug_setting == 'chexpert':
                augmentation = image_transform.get_transform_type(args, training=True, 
                                                                  img_type=traindir)
        
            print('Augmentation defined')
            
            # train_dataset = datasets.ImageFolder(
            #     traindir,
            #     moco.loader.TwoCropsTransform(transforms.Compose(augmentation1), 
            #                                   transforms.Compose(augmentation2)))
        
            train_dataset = moco.loader.Dataset_covid(traindir, img_csv,  
                                        transforms.Compose(augmentation)
                                        )
            print(len(train_dataset))
            # datasets.ImageFolder(
            #      traindir,
            #      moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))
        
            print('Training dataset defined')
        
            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            else:
                train_sampler = None
        
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
            
            print (batch_size)
            print (len(train_loader))
            print('Training dataloader defined')
            
            ep_smallest_loss = float('inf')
            for epoch in range(args.start_epoch, args.epochs):
                if args.distributed:
                    train_sampler.set_epoch(epoch)
        
                # train for one epoch
                ep_loss, train_imgs = train(train_loader, model, criterion, 
                                            optimizer, scaler, summary_writer, epoch, args, lr)
                print (ep_loss, train_imgs)
                
                if ep_loss < ep_smallest_loss:
                    ep_smallest_loss = ep_loss

                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, is_best=False, filename=os.path.join(
                        sub_checkpoint_folder, 'checkpoint_smallest_loss.pth.tar'))  
                
                if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0 and \
                     ((epoch == args.epochs - 1))):
                    #((epoch % args.save_epoch == 0) or (epoch == args.epochs - 1))):
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, is_best=False, filename=os.path.join(
                        sub_checkpoint_folder, 'checkpoint_{:04d}.pth.tar'.format(epoch)))
        
            if args.rank == 0:
                summary_writer.close()

def train(train_loader, model, criterion, optimizer, scaler, summary_writer, epoch, args, lr):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    
    running_loss = 0.0
    num_imgs = 0    
    for i, (images, _) in enumerate(train_loader):
        # print (i, images[0].shape, images[1].shape)
        # # print (i)
        # # print (len(images))
        # print (images[0].shape, images[1].shape)
        
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        if args.cos:
            _lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args, lr)
            # learning_rates.update(lr_)
        else:
            # lr_ = lr
            # print ('input lr = {}'.format(lr))
            _lr = adjust_learning_rate(optimizer, epoch, args, lr)
            # print ('adjust lr = {}'.format(_lr))
            # lr = lr_
        learning_rates.update(_lr)
        
        
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(True):
            output, target = model(images[0], images[1], moco_m)
            loss = criterion(output, target)

        running_loss += loss.item() * images[0].size(0)
        num_imgs += images[0].size(0)

        losses.update(loss.item(), images[0].size(0))
        if args.rank == 0:
            summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)
            summary_writer.add_scalar("lr", _lr, epoch * iters_per_epoch + i)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    epoch_loss = running_loss/num_imgs
    
    return epoch_loss, num_imgs

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args, lr):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if args.cos:
        if epoch < args.warmup_epochs:
            lr_ = lr * epoch / args.warmup_epochs 
        else:
            lr_ = lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    else:  # stepwise lr schedule
        lr_ = lr
        # print (lr_)
        for milestone in args.schedule:
            lr_ *= 0.1 if epoch >= milestone else 1.
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_
    return lr_


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


if __name__ == '__main__':
    main()
