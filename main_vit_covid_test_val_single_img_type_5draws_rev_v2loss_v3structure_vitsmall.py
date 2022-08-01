#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import math
import copy
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import pickle
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
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import moco.loader
import moco.builder
import training_tools.evaluator as eval_tools
from training_tools.meters import AverageMeter
from training_tools.meters import ProgressMeter

import aihc_utils.storage_util as storage_util
import aihc_utils.image_transform as image_transform


import torchvision.models as torchvision_models
import vits
import matplotlib.pyplot as plt

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_small_ori', 
               'vit_base_ori', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
# torch.cuda.device_count()  # print 1

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
# JBY: Decrease number of workers
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=48, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')

# parser.add_argument('-p', '--print-freq', default=100, type=int,
#                     metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')

# parser.add_argument('--world-size', default=-1, type=int,
#                     help='number of nodes for distributed training')
# parser.add_argument('--rank', default=-1, type=int,
#                     help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://172.31.226.232:10001', type=str,
#                     help='url used to set up distributed training')  #172.31.206.108
# parser.add_argument('--dist-backend', default='gloo', type=str,
#                     help='distributed backend')
# parser.add_argument('--seed', default=None, type=int,
#                     help='seed for initializing training. ')
# parser.add_argument('--gpu', default=None, type=int,
#                     help='GPU id to use.')
# parser.add_argument('--multiprocessing-distributed', action='store_true',
#                     help='Use multi-processing distributed training to launch '
#                          'N processes per node, which has N GPUs. This is the '
#                          'fastest way to use PyTorch for either single node or '
#                          'multi node data parallel training')

parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')


# Stanford AIHC modification
parser.add_argument('--exp-name', dest='exp_name', type=str, default='exp',
                    help='Experiment name')
parser.add_argument('--train_data', metavar='DIR',
                    help='path to train folder')
# parser.add_argument('--val_data', metavar='DIR',
#                     help='path to val folder')
# parser.add_argument('--test_data', metavar='DIR',
#                     help='path to test folder')
parser.add_argument('--class_name', type=str, default='Pleural Effusion',
                    help = 'disease name')

parser.add_argument('--save-epoch', dest='save_epoch', type=int, default=1,
                    help='Number of epochs per checkpoint save')
parser.add_argument('--from-imagenet', dest='from_imagenet', action='store_true',
                    help='use pre-trained ImageNet model')
parser.add_argument('--best-metric', dest='best_metric', type=str, default='auc',
                    help='metric to use for best model')
parser.add_argument('--semi-supervised', dest='semi_supervised', action='store_true',
                    help='allow the entire model to fine-tune')

# parser.add_argument('--binary', dest='binary', action='store_true', help='change network to binary classif')

parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
# parser.add_argument('--cos-rate', default=4, type=float, metavar='CR',
#                     help='Scaling factor for cos, higher the slower the decay')

parser.add_argument('--img-size', dest='img_size', type=int, default=224,
                    help='image size (Chexpert=320)')
parser.add_argument('--crop', dest='crop', type=int, default=224,
                    help='image crop (Chexpert=320)')
parser.add_argument('--maintain-ratio', dest='maintain_ratio', action='store_true',
                    help='whether to maintain aspect ratio or scale the image')
parser.add_argument('--rotate', dest='rotate', action='store_true',
                    help='to rotate image')
parser.add_argument('--optimizer', dest='optimizer', default='adam',
                    help='optimizer to use, chexpert=adam, moco=sgd')
parser.add_argument('--aug-setting', default='chexpert',
                    choices=['moco_v1', 'moco_v2', 'chexpert'],
                    help='version of data augmentation to use')
parser.add_argument('--crop-min', default=0.08, type=float,
                    help='minimum scale for random cropping (default: 0.08)')                 
# best_metrics = {'acc@1': {'func': 'topk_acc', 'format': ':6.2f', 'args': [1]}}#,#}
#                 # 'acc@5': {'func': 'topk_acc', 'format': ':6.2f', 'args': [5]},
#                 # 'auc': {'func': 'compute_auc_binary', 'format': ':6.2f', 'args': []}}
# best_metrics = {}
# best_metric_val = 0



def main():

    args = parser.parse_args()
    # print(args)
    checkpoint_folder = storage_util.get_storage_folder(args.exp_name, f'mocov3_test_val')
    print (checkpoint_folder)
    main_worker(args, checkpoint_folder)

    # if args.seed is not None:
    #     random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     cudnn.deterministic = True
    #     warnings.warn('You have chosen to seed training. '
    #                   'This will turn on the CUDNN deterministic setting, '
    #                   'which can slow down your training considerably! '
    #                   'You may see unexpected behavior when restarting '
    #                   'from checkpoints.')

    # if args.gpu is not None:
    #     warnings.warn('You have chosen a specific GPU. This will completely '
    #                   'disable data parallelism.')

    # if args.dist_url == "env://" and args.world_size == -1:
    #     args.world_size = int(os.environ["WORLD_SIZE"])

    # args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # ngpus_per_node = 1 #torch.cuda.device_count()
    # if args.multiprocessing_distributed:
    #     # Since we have ngpus_per_node processes per node, the total world_size
    #     # needs to be adjusted accordingly
    #     args.world_size = ngpus_per_node * args.world_size
    #     # Use torch.multiprocessing.spawn to launch distributed processes: the
    #     # main_worker process function
    #     mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, checkpoint_folder))
    # else:
    #     # Simply call main_worker function
    #     main_worker(args.gpu, ngpus_per_node, args, checkpoint_folder)


def main_worker(args, checkpoint_folder):
    
    with open(os.path.join(checkpoint_folder,'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    # if args.binary:
    #     best_metrics.update({'auc' : {'func': 'compute_auc_binary', 'format': ':6.2f', 'args': []}})
    
    # args.gpu = gpu
    # print (gpu)
    
    # # suppress printing if not master
    # if args.multiprocessing_distributed and args.gpu != 0:
    #     def print_pass(*args):
    #         pass
    #     builtins.print = print_pass

    # if args.gpu is not None:
    #     print("Use GPU: {} for training".format(args.gpu))

    # if args.distributed:
    #     if args.dist_url == "env://" and args.rank == -1:
    #         args.rank = int(os.environ["RANK"])
    #     if args.multiprocessing_distributed:
    #         # For multiprocessing distributed training, rank needs to be the
    #         # global rank among all the processes
    #         args.rank = args.rank * ngpus_per_node + gpu
    #     dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                             world_size=args.world_size, rank=args.rank)
    
    # device = torch.device('cuda:1')
    
    # parameters
    # ALL_SEMI_RATIO =  [0.00075, 0.0009, 0.001, 0.0025, 0.001, 0.005, 0.01, 0.1, 0.2, 0.3]
    ALL_SEMI_RATIO =  [1]
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
                        1:1
                    } 
    
    all_test_auc = []
    all_test_acc = []
    for s in ALL_SEMI_RATIO:
        print ('ratio = {}'.format(s))
        
        ratio_test_auc = []
        ratio_test_acc = []
        
        for it in range(SEMI_ITERATIONS[s]):    
            print ('iteration = {}'.format(it))
            
            summary_writer = SummaryWriter(os.path.join(checkpoint_folder, 'tb_train_val_test_'+str(s)+'_'+str(it)))
            sub_checkpoint_folder = storage_util.get_storage_sub_folder(checkpoint_folder, s, it)
            sub_checkpoint_folder_acc = storage_util.get_storage_sub_folder_acc(checkpoint_folder, s, it)
            
            # create model
            print("=> creating model '{}'".format(args.arch))
            if args.arch.startswith('vit'):
                model = vits.__dict__[args.arch]()
                linear_keyword = 'head'
            else:
                model = torchvision_models.__dict__[args.arch]()
                linear_keyword = 'fc'
        
            # freeze all layers but the last fc
            if not args.semi_supervised:
                for name, param in model.named_parameters():
                    if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
                        param.requires_grad = False

            num_classes = 3 #len(os.listdir(args.val_data)) #assume in imagenet format, so length == num folders/classes
            if linear_keyword == 'head':
                model.head = nn.Linear(model.head.in_features, num_classes)
            elif linear_keyword == 'fc':
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            
            # init the fc layer
            getattr(model, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
            getattr(model, linear_keyword).bias.data.zero_()    
        

            # if num_classes == 2 and not args.binary:
            #     raise ValueError(f'Folder has {num_classes} classes, but you did not use "--binary" flag')
            # elif num_classes != 2 and args.binary:
            #     raise ValueError(f'Folder has {num_classes} classes, but you used "--binary" flag')
        
            # init the fc layer
            # if args.binary:
            # model.fc = nn.Linear(model.fc.in_features, num_classes)
                
            # model.fc.weight.data.normal_(mean=0.0, std=0.01)
            # model.fc.bias.data.zero_()
        
            # load from pre-trained, before DistributedDataParallel constructor
            if args.pretrained:                  
                pretrained_path = os.path.join(args.pretrained, 
                                                    'train_'+str(s)+'_'+str(it),
                                                    'checkpoint_smallest_loss.pth.tar')
                # pretrained_path = os.path.join(args.pretrained, 
                #                                'train_'+str(s)+'_'+str(it),
                #                                'checkpoint_0029.pth.tar')

                print (pretrained_path)
                if os.path.isfile(pretrained_path):
                    print("=> loading checkpoint '{}'".format(pretrained_path))
                    checkpoint = torch.load(pretrained_path, map_location="cpu")
        
                    # rename moco pre-trained keys
                    state_dict = checkpoint['state_dict']
                    for k in list(state_dict.keys()):
                        # retain only base_encoder up to before the embedding layer
                        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                            # remove prefix
                            state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                        # delete renamed or unused k
                        del state_dict[k]
        
                    args.start_epoch = 0
                    msg = model.load_state_dict(state_dict, strict=False)
                    assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
        
                    print("=> loaded pre-trained model '{}'".format(args.pretrained))
                else:
                    print("=> no checkpoint found at '{}'".format(args.pretrained))
        
            # infer learning rate before changing batch size
            if args.cos:
                init_lr = args.lr * args.batch_size / 8
            else:
                init_lr = args.lr #* args.batch_size / 8    
        
            # if args.distributed:
            #     # For multiprocessing distributed, DistributedDataParallel constructor
            #     # should always set the single device scope, otherwise,
            #     # DistributedDataParallel will use all available devices.
            #     if args.gpu is not None:
            #         torch.cuda.set_device(args.gpu)
            #         model.cuda(args.gpu)
            #         # When using a single GPU per process and per
            #         # DistributedDataParallel, we need to divide the batch size
            #         # ourselves based on the total number of GPUs we have
            #         args.batch_size = int(args.batch_size / ngpus_per_node)
            #         args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            #         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            #     else:
            #         model.cuda()
            #         # DistributedDataParallel will divide and allocate batch_size to all
            #         # available GPUs if device_ids are not set
            #         model = torch.nn.parallel.DistributedDataParallel(model)
            # elif args.gpu is not None:
            #     torch.cuda.set_device(args.gpu)
            #     model = model.cuda(args.gpu)
            # else:
            #     # DataParallel will divide and allocate batch_size to all available GPUs
            #     if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            #         model.features = torch.nn.DataParallel(model.features)
            #         model.cuda()
            #     else:
            #         model = torch.nn.DataParallel(model).cuda()
        
            # define loss function (criterion) and optimizer
            criterion = nn.CrossEntropyLoss()#.cuda(args.gpu)
        
            # optimize only the linear classifier
            parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
            if not args.semi_supervised:
                assert len(parameters) == 2  # fc.weight, fc.bias
            
            if args.optimizer == 'sgd':
                # optimizer = torch.optim.SGD(model.parameters(), args.lr,
                #                             momentum=args.momentum,
                #                             weight_decay=args.weight_decay)
                 optimizer = torch.optim.SGD(parameters, init_lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)               
                
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                             betas=(0.9, 0.999),
                                             weight_decay=args.weight_decay)
        
        
            # # optionally resume from a checkpoint
            # if args.resume:
            #     if os.path.isfile(args.resume):
            #         print("=> loading checkpoint '{}'".format(args.resume))
            #         if args.gpu is None:
            #             checkpoint = torch.load(args.resume)
            #         else:
            #             # Map model to be loaded to specified single gpu.
            #             loc = 'cuda:{}'.format(args.gpu)
            #             checkpoint = torch.load(args.resume, map_location=loc)
            #         args.start_epoch = checkpoint['epoch']
        
            #         # TODO JBY: Handle resume for current metrics setup
            #         raise NotImplementedError('Resuming not supported yet!')
        
            #         for metric in best_metrics:
            #             best_metrics[metric][0] = checkpoint[f'best_metrics'][metric]
            #         if args.gpu is not None:
            #             # best_acc_val may be from a checkpoint from a different GPU
            #             # best_acc_val = best_acc_val.to(args.gpu)
            #             # best_acc_test = best_acc_test.to(args.gpu)
            #             for metric in best_metrics:
            #                 best_metrics[metric][0] = best_metrics[metric][0].to(args.gpu)
        
            #         model.load_state_dict(checkpoint['state_dict'])
            #         optimizer.load_state_dict(checkpoint['optimizer'])
            #         print("=> loaded checkpoint '{}' (epoch {})"
            #               .format(args.resume, checkpoint['epoch']))
            #     else:
            #         print("=> no checkpoint found at '{}'".format(args.resume))
        
            # cudnn.benchmark = True
        
            # Data loading code
            # traindir = os.path.join(args.data, 'train')
            # valdir = os.path.join(args.data, 'val')
            # disease_name = args.class_name
            
            train_csv = os.path.join('create_covid_dataset', str(s)+'_labeled_train_'+str(it)+'.txt')
            traindir = args.train_data
            
            valid_csv = 'create_covid_dataset/val_ds.txt'
            # validdir = args.train_data
            
            test_csv = 'create_covid_dataset/test_ds.txt'
            testdir = args.train_data
        
            if args.aug_setting == 'moco_v2':
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                train_augmentation = [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
        
                test_augmentation = [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            elif args.aug_setting == 'chexpert':
                train_augmentation = image_transform.get_transform_type(args, training=True,
                                                                        img_type= traindir)
                
                test_augmentation = image_transform.get_transform_type(args, training=False,
                                                                       img_type= testdir)
                
                data_transforms = {'train': transforms.Compose(train_augmentation),
                                    'val': transforms.Compose(test_augmentation),}
                
        
            # train_dataset = moco.loader.Dataset(traindir, train_csv,  
            #                                     transforms.Compose(train_augmentation), 
            #                                     disease_name)
            # num_train_imgs = len(train_dataset)
            # valid_dataset = moco.loader.Dataset(validdir, valid_csv,  
            #                                     transforms.Compose(test_augmentation), 
            #                                     disease_name)
            # num_valid_imgs = len(valid_dataset)
            
            image_datasets = {'train': moco.loader.Dataset_covid(traindir, train_csv,  
                                                     data_transforms['train']),
                              'val': moco.loader.Dataset_covid(traindir, valid_csv,  
                                                     data_transforms['val'])}
            
            dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
            
            test_dataset = moco.loader.Dataset_covid(testdir, test_csv,  
                                               transforms.Compose(test_augmentation))
            num_test_imgs = len(test_dataset)            
            # train_dataset = datasets.ImageFolder(
            #     traindir, transforms.Compose(train_augmentation))
        
            # if args.distributed:
            #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            # else:
            #     train_sampler = None
        
            # train_loader = torch.utils.data.DataLoader(
            #     train_dataset, 
            #     batch_size=args.batch_size, shuffle=True,
            #     num_workers=args.workers, pin_memory=True)
        
            # val_loader = torch.utils.data.DataLoader(
            #     valid_dataset,
            #     batch_size=args.batch_size, shuffle=False,
            #     num_workers=args.workers, pin_memory=True)
            
            
            dataloaders = {x: torch.utils.data.DataLoader(
                                image_datasets[x], 
                                batch_size=args.batch_size,
                                shuffle=True, 
                                num_workers=args.workers, pin_memory=True)
                            for x in ['train', 'val']}

        
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
        
            # evaluator = eval_tools.Evaluator(model, criterion, best_metrics,\
            #                                  {'train': train_loader,\
            #                                   'valid': val_loader,\
            #                                   'test': test_loader}, args)
        
            # if args.evaluate:
            #     evaluator.evaluate('valid', 0)
            #     evaluator.evaluate('test', 0)
            #     return
            
            # evaluator.evaluate('test', 0)
            best_val_auc = 0.0
            best_test_auc = 0.0

            best_val_acc = 0.0
            best_test_acc = 0.0
            
            lr_allep_1per_1iter = []
            for epoch in range(args.start_epoch, args.epochs):
                # if args.distributed:
                #     train_sampler.set_epoch(epoch)
                lr_ = adjust_learning_rate(optimizer, init_lr, epoch, args)
                lr_allep_1per_1iter.append(lr_)
                summary_writer.add_scalar('lr', lr_, epoch)
                print(optimizer.param_groups[0]['lr'])
                # train for one epoch
                val_loss, val_auc, val_acc, summary_writer, model = train(dataloaders, model, criterion, 
                                                optimizer, epoch, args, dataset_sizes, 
                                                summary_writer)
                                
                if val_auc > best_val_auc:
                    
                    best_val_auc = val_auc
                    
                    test_loss, test_auc, _ =test(test_loader, model, criterion, optimizer, epoch, num_test_imgs)
                    
                    if test_auc > best_test_auc:
                        best_test_auc = test_auc
                    
                    summary_writer.add_scalar('test/all_test_loss_auc', test_loss, epoch)
                    summary_writer.add_scalar('test/all_test_auc', test_auc, epoch)
                        
                    save_checkpoint(sub_checkpoint_folder, {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_metric_val_test': test_auc,
                        'best_metric_val': best_val_auc,
                        'best_metric_test': best_test_auc,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best=True)
                    
                    if epoch == args.start_epoch and args.pretrained:
                        sanity_check(model.state_dict(), pretrained_path,
                                     args.semi_supervised, linear_keyword)
        
                if val_acc>best_val_acc:
                    
                    best_val_acc = val_acc
                    
                    test_loss, _, test_acc =test(test_loader, model, criterion, optimizer, epoch, num_test_imgs)
                    
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                    
                    summary_writer.add_scalar('test/all_test_loss_acc', test_loss, epoch)
                    summary_writer.add_scalar('test/all_test_acc', test_acc, epoch)
                        
                    save_checkpoint(sub_checkpoint_folder_acc, {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_metric_val_test': test_acc,
                        'best_metric_val': best_val_acc,
                        'best_metric_test': best_test_acc,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best=True)
                    
                    if epoch == args.start_epoch and args.pretrained:
                        sanity_check(model.state_dict(), pretrained_path,
                                     args.semi_supervised, linear_keyword)
            
            # last is in the auc folder
            save_checkpoint(sub_checkpoint_folder, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                
                'best_metric_val_auc': best_val_auc,
                'last_metric_val_auc': val_auc,
                'best_metric_test_auc': best_test_auc,
                'best_metric_val_test_auc': test_auc,
                
                'best_metric_val_acc': best_val_acc,
                'last_metric_val_acc': val_acc,
                'best_metric_test_acc': best_test_acc,
                'best_metric_val_test_acc': test_acc,
                
                'optimizer' : optimizer.state_dict(),
            }, is_best=False)
            
            if epoch == args.start_epoch and args.pretrained:
                sanity_check(model.state_dict(), pretrained_path,
                             args.semi_supervised, linear_keyword)
                
            print('Best_Auc: {:.4f} Best_Val_Test_Auc: {:.4f} Best_Acc: {:.4f} Best_Val_Test_Acc: {:.4f}'
              .format(best_test_auc, test_auc, best_test_acc, test_acc))
            plt.plot(lr_allep_1per_1iter)
            plt.savefig(os.path.join(checkpoint_folder, "lr.jpg"))
            
            ratio_test_auc.append(test_auc)
            ratio_test_acc.append(test_acc)
            
        all_test_auc.append(ratio_test_auc)
        all_test_acc.append(ratio_test_acc)
        
    file_1 = open(os.path.join(checkpoint_folder, args.exp_name+'_auc.pickle'),'wb')
    pickle.dump(all_test_auc,  file_1)
    file_2 = open(os.path.join(checkpoint_folder, args.exp_name+'_acc.pickle'),'wb')
    pickle.dump(all_test_acc,  file_2)
        
# def train(train_loader, model, criterion, optimizer, epoch, args, best_metrics):
def train(train_loader, model, criterion, optimizer, epoch, args, num_imgs, summary_writer):
    
    print(f'==> Training, epoch {epoch}')

    # batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')

    # metric_meters = {metric: AverageMeter(metric,
    #                                       best_metrics[metric]['format']) \
    #                         for metric in best_metrics}
        
    # list_meters = [metric_meters[m] for m in metric_meters]
    
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, *list_meters],
    #     prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())
    
    # device = torch.device('cuda:1')
    device = torch.device('cuda:0')

    for phase in ['train', 'val']: 
        # JBY: If semi-supervised, we tune on the entire model instead
        if args.semi_supervised:
            model.train().to(device)
        else:
            model.eval().to(device)
            
        # all_output = []
        # all_gt = []  
    
        # end = time.time()
        
        running_loss = 0.0
        # best_auc = 0.0
        
        for i, (image, target) in enumerate(tqdm(train_loader[phase])):
            # print (len(image))
            # print (image[0].shape)
            images, images2  = image
            # print (target)
            # measure data loading time
            # data_time.update(time.time() - end)
    
            # if args.gpu is not None:
            images = images.to(device)
            target = target.long().to(device)
            
            # compute gradient and do SGD step
            optimizer.zero_grad()
    
            # compute output
            output = model(images)
            # print (output.shape)
            _, preds = torch.max(output, 1)
            loss = criterion(output, target)
            
            if phase == 'train':
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            pred_np = preds.view(-1).cpu().detach().numpy()
            val_np = np.array(output.cpu().detach().numpy())
            # print (val_np.shape)
            gt_np = target.view(-1).cpu().detach().numpy()
            # print (gt_np.shape)
            
            if i == 0:
                all_pred = pred_np
                all_gt = gt_np
                all_val = val_np
            else:
                all_pred = np.append(all_pred, pred_np)
                all_gt = np.append(all_gt, gt_np)
                all_val = np.concatenate((all_val, val_np), axis=0)
        
        all_gt_one_hot = label_binarize(all_gt, classes=[0,1,2])
        # print (all_gt)
        
        fpr = {}; tpr={}; thresholds={}; epoch_auc_class={}
        for i in range(3):
            fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(all_gt_one_hot[:, i], all_val[:, i])     
            epoch_auc_class[i] = metrics.auc(fpr[i], tpr[i])
        
        epoch_auc = np.mean([epoch_auc_class[0], epoch_auc_class[1], epoch_auc_class[2]])
        epoch_acc = np.sum((all_pred == all_gt))/num_imgs[phase]
        epoch_loss = running_loss/num_imgs[phase]
        
        print('{} Loss: {:.4f} Auc: {:.4f} Acc: {:.4f}'
              .format(phase, epoch_loss, epoch_auc, epoch_acc))
        
        if phase == 'train':
            summary_writer.add_scalar('train/loss', epoch_loss, epoch)
            summary_writer.add_scalar('train/auc', epoch_auc, epoch)
            summary_writer.add_scalar('train/acc', epoch_acc, epoch)        
        elif phase == 'val':
            summary_writer.add_scalar('val/loss', epoch_loss, epoch)
            summary_writer.add_scalar('val/auc', epoch_auc, epoch)
            summary_writer.add_scalar('val/acc', epoch_acc, epoch)
        
        
    return epoch_loss, epoch_auc, epoch_acc, summary_writer, model
        
# def train(train_loader, model, criterion, optimizer, epoch, args, best_metrics):
def test(train_loader, model, criterion, optimizer, epoch, num_imgs):
    
    print(f'==> Testing, epoch {epoch}')
   
    # device = torch.device('cuda:1')
    device = torch.device('cuda:0')

    model.eval().to(device)

    running_loss = 0.0
    # best_auc = 0.0
    for i, (image, target) in enumerate(tqdm(train_loader)):
        
        images, images2  = image
        # measure data loading time
        # data_time.update(time.time() - end)

        # if args.gpu is not None:
        images = images.to(device)
        target = target.long().to(device)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()

        # compute output
        output = model(images)
        # print (output.shape)
        _, preds = torch.max(output, 1)
        loss = criterion(output, target)
               
        running_loss += loss.item() * images.size(0)
        
        pred_np = preds.view(-1).cpu().detach().numpy()
        val_np = np.array(output.cpu().detach().numpy())
        # print (val_np.shape)
        gt_np = target.view(-1).cpu().detach().numpy()
        
        if i == 0:
            all_pred = pred_np
            all_gt = gt_np
            all_val = val_np
        else:
            all_pred = np.append(all_pred, pred_np)
            all_gt = np.append(all_gt, gt_np)
            all_val = np.concatenate((all_val, val_np), axis=0)
            
    all_gt_one_hot = label_binarize(all_gt, classes=[0,1,2])
    # print (all_gt)
    
    fpr = {}; tpr={}; thresholds={}; epoch_auc_class={}
    for i in range(3):
        fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(all_gt_one_hot[:, i], all_val[:, i])     
        epoch_auc_class[i] = metrics.auc(fpr[i], tpr[i])
    
    auc = np.mean([epoch_auc_class[0], epoch_auc_class[1], epoch_auc_class[2]])
    epoch_loss = running_loss/num_imgs
    acc = np.sum((all_pred == all_gt))/num_imgs
    
    print('{} Loss: {:.4f} Auc: {:.4f} Acc: {:.4f}'
          .format('Test', epoch_loss, auc, acc))
    
    return epoch_loss, auc, acc


def save_checkpoint(checkpoint_folder, state, is_best, filename='last_checkpoint.pth.tar'):
    # torch.save(state, os.path.join(checkpoint_folder, filename))
    # if is_best:
    #     shutil.copyfile(os.path.join(checkpoint_folder, filename),
    #                     os.path.join(checkpoint_folder, 'model_best.pth.tar'))

    if is_best:
        filename = 'model_best.pth.tar'
        torch.save(state, os.path.join(checkpoint_folder, filename))
    else:
        torch.save(state, os.path.join(checkpoint_folder, filename))

def sanity_check(state_dict, pretrained_weights, semi_supervised, linear_keyword):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    if semi_supervised:
        print('SKIPPING SANITY CHECK for semi-supervised learning')
        return

    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore linear layer
        if '%s.weight' % linear_keyword in k or '%s.bias' % linear_keyword in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.base_encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


# JBY: Ported over support for Cosine learning rate
def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = init_lr
    if args.cos:  # cosine lr schedule
        # TODO, JBY, is /4 an appropriate scale?
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

if __name__ == '__main__':
    main()

#%% 

# import pickle

# file = open("finetune_test.pickle",'rb')
# tttt = pickle.load(file)