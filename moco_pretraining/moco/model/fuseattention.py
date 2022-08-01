#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 13:51:51 2022

@author: endiqq
"""


import math
from collections import deque

import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import models


# transformer base structure
class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        B, T, C = x.size()

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x

# transfuer
class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head, block_exp, n_layer, 
                    vert_anchors, horz_anchors, seq_len, 
                    embd_pdrop, attn_pdrop, resid_pdrop, args, config):
        super().__init__()
        
        self.n_embd = n_embd #input feature
        self.seq_len = seq_len
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        
        self.config = config
        self.args = args
        
        # positional embedding parameter (learnable), image + lidar
        if args.arch.startswith('res'):
            self.pos_emb = nn.Parameter(torch.zeros(1, (self.config.n_views + 1) * seq_len * vert_anchors * horz_anchors, n_embd))
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, ((self.config.n_views + 1) * seq_len * vert_anchors * horz_anchors)+2, n_embd))
        
        # velocity embedding
        # self.vel_emb = nn.Linear(1, n_embd)
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, 
                        block_exp, attn_pdrop, resid_pdrop)
                        for layer in range(n_layer)])
        
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = seq_len
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # def configure_optimizers(self):
    #     # separate out all parameters to those that will and won't experience regularizing weight decay
    #     decay = set()
    #     no_decay = set()
    #     whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
    #     blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
    #     for mn, m in self.named_modules():
    #         for pn, p in m.named_parameters():
    #             fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

    #             if pn.endswith('bias'):
    #                 # all biases will not be decayed
    #                 no_decay.add(fpn)
    #             elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
    #                 # weights of whitelist modules will be weight decayed
    #                 decay.add(fpn)
    #             elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
    #                 # weights of blacklist modules will NOT be weight decayed
    #                 no_decay.add(fpn)

    #     # special case the position embedding parameter in the root GPT module as not decayed
    #     no_decay.add('pos_emb')

    #     # create the pytorch optimizer object
    #     param_dict = {pn: p for pn, p in self.named_parameters()}
    #     optim_groups = [
    #         {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
    #         {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    #     ]

    #     return optim_groups

    def forward(self, cxr_tensor, enh_tensor):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
            velocity (tensor): ego-velocity
        """
        
        bz = cxr_tensor.shape[0]
        
        if self.args.arch.startswith('res'):
        
            h, w = cxr_tensor.shape[2:4]
            
            # forward the image model for token embeddings
            image_tensor = cxr_tensor.view(bz, self.config.n_views * self.seq_len, -1, h, w)
            lidar_tensor = enh_tensor.view(bz, self.seq_len, -1, h, w)
    
            # pad token embeddings along number of tokens dimension
            token_embeddings = torch.cat([image_tensor, lidar_tensor], dim=1).permute(0,1,3,4,2).contiguous()
            token_embeddings = token_embeddings.view(bz, -1, self.n_embd) # (B, an * T, C)
        else:
            _, ftrs, _ = cxr_tensor.shape
            token_embeddings = torch.cat([cxr_tensor, enh_tensor], dim=1)

        # # project velocity to n_embed
        # velocity_embeddings = self.vel_emb(velocity.unsqueeze(1)) # (B, C)

        # add (learnable) positional embedding and velocity embedding for all tokens
        if self.args.pos_embed:
            x = self.drop(self.pos_emb + token_embeddings) # (B, an * T, C)
        else:
            x = self.drop(token_embeddings) # (B, an * T, C)
        # x = self.drop(token_embeddings + velocity_embeddings.unsqueeze(1)) # (B, an * T, C)
        x = self.blocks(x) # (B, an * T, C)
        x = self.ln_f(x) # (B, an * T, C)
        
        if self.args.arch.startswith('res'):
            x = x.view(bz, (self.config.n_views + 1) * self.seq_len, self.vert_anchors, self.horz_anchors, self.n_embd)
            x = x.permute(0,1,4,2,3).contiguous() # same as token_embeddings
    
            image_tensor_out = x[:, :self.config.n_views*self.seq_len, :, :, :].contiguous().view(bz * self.config.n_views * self.seq_len, -1, h, w)
            lidar_tensor_out = x[:, self.config.n_views*self.seq_len:, :, :, :].contiguous().view(bz * self.seq_len, -1, h, w)
        else:
            image_tensor_out = x[:, :ftrs, :]
            lidar_tensor_out = x[:, ftrs:, :]
        
        return image_tensor_out, lidar_tensor_out


class Encoder(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    """

    def __init__(self, model_cxr, model_enh, config, args):
        super().__init__()
        
        # if not semi_supervised:
        #     for params in model_cxr.parameters():
        #         params.requires_grad = False
                
        #     for params in model_enh.parameters():
        #         params.requires_grad = False
            
        
        self.config = config
        self.args = args
        self.avgpool = nn.AdaptiveAvgPool2d((self.config.vert_anchors, self.config.horz_anchors))
        
        # if self.args.arch.startswith('res'):
        #     self.cxr_encoder = nn.Sequential(*list(model_cxr.model.children())[:-2])
        #     self.enh_encoder = nn.Sequential(*list(model_enh.model.children())[:-2])            
        # else:
        #     self.cxr_encoder = model_cxr.model.features3D
        #     self.enh_encoder = model_enh.model.features3D
        if self.args.arch.startswith('res'):
            self.cxr_encoder = nn.Sequential(*list(model_cxr.children())[:-2])
            self.enh_encoder = nn.Sequential(*list(model_enh.children())[:-2])            
        else:
            self.cxr_encoder = model_cxr.features3D
            self.enh_encoder = model_enh.features3D
        
        
        self.transformer4 = GPT(n_embd=config.n_embd,
                            n_head=config.n_head, 
                            block_exp=config.block_exp, 
                            n_layer=config.n_layer, 
                            vert_anchors=config.vert_anchors, 
                            horz_anchors=config.horz_anchors, 
                            seq_len=config.seq_len, 
                            embd_pdrop=config.embd_pdrop, 
                            attn_pdrop=config.attn_pdrop, 
                            resid_pdrop=config.resid_pdrop,
                            args = args,
                            config=config)

        
    def forward(self, cxr_image, enh_image):
        '''
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            velocity (tensor): input velocity from speedometer
        '''
        # # image normalization
        # if self.image_encoder.normalize:
        #     image_list = [normalize_imagenet(image_input) for image_input in image_list]

        bz, _, h, w = cxr_image.shape
        
        # img_channel = cxr_image.shape[1]
        # lidar_channel = enh_image.shape[1]
        
        # self.config.n_views = len(image_list) // self.config.seq_len

        # image_tensor = torch.stack(image_list, dim=1).view(bz * self.config.n_views * self.config.seq_len, img_channel, h, w)
        # lidar_tensor = torch.stack(lidar_list, dim=1).view(bz * self.config.seq_len, lidar_channel, h, w)

        image_features = self.cxr_encoder(cxr_image)
        # print (image_features.shape)
        # image_features = self.image_encoder.features.bn1(image_features)
        # image_features = self.image_encoder.features.relu(image_features)
        # image_features = self.image_encoder.features.maxpool(image_features)
        
        lidar_features = self.enh_encoder(enh_image)
        # print (lidar_features.shape)
        # lidar_features = self.lidar_encoder._model.bn1(lidar_features)
        # lidar_features = self.lidar_encoder._model.relu(lidar_features)
        # lidar_features = self.lidar_encoder._model.maxpool(lidar_features)

        
        # fusion at (B, 512, 8, 8)
        if self.args.arch.startswith('res'):
            image_embd_layer4 = self.avgpool(image_features)
            lidar_embd_layer4 = self.avgpool(lidar_features)
        else:
            image_embd_layer4 = image_features
            lidar_embd_layer4 = lidar_features           
        
        image_features_layer4, lidar_features_layer4 = self.transformer4(image_embd_layer4, lidar_embd_layer4)
        # print (image_features_layer4.shape, lidar_features_layer4.shape)
        image_features = image_features + image_features_layer4
        lidar_features = lidar_features + lidar_features_layer4
        
        if self.args.arch.startswith('res'):
            image_features = F.relu(image_features, inplace=True)
            image_features = F.adaptive_avg_pool2d(image_features, (1, 1))
            image_features = torch.flatten(image_features, 1)
            image_features = image_features.view(bz, self.config.n_views * self.config.seq_len, -1)
            
            lidar_features = F.relu(lidar_features, inplace=True)
            lidar_features = F.adaptive_avg_pool2d(lidar_features, (1, 1))
            lidar_features = torch.flatten(lidar_features, 1)
            lidar_features = lidar_features.view(bz, self.config.seq_len, -1)

        else:
            image_features = image_features[:,0].view(bz, self.config.n_views * self.config.seq_len, -1)
            lidar_features = lidar_features[:,0].view(bz, self.config.seq_len, -1)
    
        fused_features = torch.cat([image_features, lidar_features], dim=1)
        fused_features = torch.sum(fused_features, dim=1)
        return fused_features
    
class TransFuser(nn.Module):
    '''
    Transformer-based feature fusion followed by GRU-based waypoint prediction network and PID controller
    '''

    def __init__(self, model_cxr, model_enh, config, args):
        super().__init__()
        
        # self.device = device
        self.config = config
        # self.pred_len = config.pred_len
        # self.cfg = cfg
        # self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        # self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)
        self.args = args
        self.encoder = Encoder(model_cxr, model_enh, config, args)#.to(self.device)
        
        # if cfg.extend == True:
        #     self.join = nn.Sequential(
        #                         nn.Linear(1024, 512),
        #                         nn.ReLU(inplace=True),
        #                         nn.Linear(512, 256),
        #                         nn.ReLU(inplace=True),
        #                         nn.Linear(256, 128),
        #                         nn.ReLU(inplace=True),
        #                     )#.to(self.device)
            
        
        #     # self.decoder = nn.GRUCell(input_size=2, hidden_size=64).to(self.device)
        #     self.output = ClassificationHead(cfg, 128) #nn.Linear(64, 2).to(self.device)
        # else:
        #     self.output = ClassificationHead(cfg, 1024) #nn.Linear(64, 2).to(self.device)
        
        
        # num_classes = 3 #len(os.listdir(args.val_data)) #assume in imagenet format, so length == num folders/classes
        if self.args.arch.startswith('vit'):
            self.output = nn.Linear(model_cxr.head.in_features, 3)
            # model_enh.head = nn.Linear(model_enh.head.in_features, num_classes)
        else:
            self.output = nn.Linear(model_cxr.fc.in_features, 3)
            # model_enh.fc = nn.Linear(model_enh.fc.in_features, num_classes)
        
        # init the fc layer
        self.output.weight.data.normal_(mean=0.0, std=0.01)
        self.output.bias.data.zero_()
        # getattr(model_cxr, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
        # getattr(model_enh, linear_keyword).bias.data.zero_()  
        
    def forward(self, image_list, lidar_list):
        '''
        Predicts waypoint from geometric feature projections of image + LiDAR input
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            target_point (tensor): goal location registered to ego-frame
            velocity (tensor): input velocity from speedometer
        '''
        fused_features = self.encoder(image_list, lidar_list)
        # if self.cfg.extend == True:
        #     z = self.join(fused_features)
        # else:
        #     z = fused_features
    
        logits = self.output(fused_features)
        
        return logits