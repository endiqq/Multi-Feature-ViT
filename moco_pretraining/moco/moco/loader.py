# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
from torch.utils.data import Dataset
import glob
import cv2
import numpy as np
from PIL import Image
import os
import random
random.seed(0)

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x



# rotation as augmentation method
class Dataset(Dataset):
    
    def __init__(self, folder, img_csv, transform_img, disease_name, mode='train'):
        
        self._image_paths = []
        self._labels = []
        self.transform = transform_img
        self._mode = mode
        
        self.dict = {'1.0': '1','': '0', '0.0': '0', '-1.0': '1'}
        
        with open(img_csv) as f:
            header = f.readline().strip('\n').split(',')
            idx = [i for i, h in enumerate(header) if h == disease_name]
            
            for line in f:
                # print (line)
                fields = line.strip('\n').split(',')
             
                img_path = os.path.join(folder, fields[1])
                # print (img_path)
                label = self.dict.get(fields[idx[0]])
                label = np.array(label)        
                self._image_paths.append(img_path)
                self._labels.append(label)

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        
        # if self.dataset == 'data':
        image = cv2.imread(self._image_paths[idx])          
        image = Image.fromarray(image)
        # enh_image = Image.fromarray(enh_image)
        
        if self._mode == 'train':
            image_q = self.transform(image)
            image_k = self.transform(image)
            # enh_image = self.transform_enh(enh_image)
            
        labels = self._labels[idx].astype(np.float32)
        # labels = self._labels[idx]
        images = [image_q, image_k]
        
        if self._mode == 'train' or self._mode == 'dev':
            return images, labels
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))
            


# rotation as augmentation method
class Dataset_covid(Dataset):
    
    def __init__(self, folder, img_csv, transform_img, mode='train'):
        
        self._image_paths = []
        self._labels = []
        self.transform = transform_img
        self._mode = mode
        
        # all_lines = []        
        with open(img_csv) as f:            
            for line in f:
                # print (line)
                # all_lines.append(line)
                
                fields = line.strip('\n').split(' ')
             
                img_path = os.path.join(fields[1], folder, fields[2])
                # print(img_path)
                
                # print (img_path)
                label = fields[-2]
                label = np.array(label)        
                self._image_paths.append(img_path)
                self._labels.append(label)

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        
        # if self.dataset == 'data':
        image = cv2.imread(self._image_paths[idx])
        # print (self._image_paths[idx])
        # print (image.shape)          
        image = Image.fromarray(image)
        # enh_image = Image.fromarray(enh_image)       
        
        if self._mode == 'train':
            image_q = self.transform(image)
            image_k = self.transform(image)
            # enh_image = self.transform_enh(enh_image)
            
        labels = self._labels[idx].astype(np.float32)
        # labels = self._labels[idx]
        images = [image_q, image_k]
        
        if self._mode == 'train' or self._mode == 'dev':
            return images, labels
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))

class Dataset_covid_4ch(Dataset):
    
    def __init__(self, img_csv, transform_img, mode='train'):
        
        self._image_paths_cxr = []
        self._image_paths_enh = []
        self._labels = []
        self.transform = transform_img
        self._mode = mode
        
        # all_lines = []        
        with open(img_csv) as f:            
            for line in f:
                # print (line)
                # all_lines.append(line)
                
                fields = line.strip('\n').split(' ')
             
                img_path_cxr = os.path.join(fields[1], 'data', fields[2])
                img_path_enh = os.path.join(fields[1], 'Train_Mix', fields[2])
                # print(img_path)
                
                # print (img_path)
                label = fields[-2]
                label = np.array(label)        
                self._image_paths_cxr.append(img_path_cxr)
                self._image_paths_enh.append(img_path_enh)
                self._labels.append(label)

    def __len__(self):
        return len(self._image_paths_cxr)

    def __getitem__(self, idx):
        
        # if self.dataset == 'data':
        image_cxr = cv2.imread(self._image_paths_cxr[idx])
        image_enh = cv2.imread(self._image_paths_enh[idx])
        # print (image.shape)
        image = np.concatenate((image_cxr, image_enh), axis=2)[:,:,2:]          
        image = Image.fromarray(image)
        # enh_image = Image.fromarray(enh_image)       
        
        if self._mode == 'train':
            image_q = self.transform(image)
            image_k = self.transform(image)
            # enh_image = self.transform_enh(enh_image)
            
        labels = self._labels[idx].astype(np.float32)
        # labels = self._labels[idx]
        images = [image_q, image_k]
        
        if self._mode == 'train' or self._mode == 'dev':
            return images, labels
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))


class Dataset_covid_LEnh_RCXR_2norms(Dataset):
    
    def __init__(self, img_csv, transform_img_cxr, transform_img_enh, mode='train'):
        
        self._image_paths_cxr = []
        self._image_paths_enh = []
        self._labels = []
        self.transform_cxr = transform_img_cxr
        self.transform_enh = transform_img_enh
        self._mode = mode
        
        # all_lines = []        
        with open(img_csv) as f:            
            for line in f:
                # print (line)
                # all_lines.append(line)
                
                fields = line.strip('\n').split(' ')
             
                img_path_cxr = os.path.join(fields[1], 'data', fields[2])
                img_path_enh = os.path.join(fields[1], 'Train_Mix', fields[2])
                # print(img_path)
                
                # print (img_path)
                label = fields[-2]
                label = np.array(label)        
                self._image_paths_cxr.append(img_path_cxr)
                self._image_paths_enh.append(img_path_enh)
                self._labels.append(label)

    def __len__(self):
        return len(self._image_paths_cxr)

    def __getitem__(self, idx):
        
        # if self.dataset == 'data':
        image_cxr = cv2.imread(self._image_paths_cxr[idx])
        image_cxr = Image.fromarray(image_cxr)
        image_enh = cv2.imread(self._image_paths_enh[idx])
        image_enh = Image.fromarray(image_enh)
        # print (image.shape)
        # image = np.concatenate((image_cxr, image_enh), axis=2)[:,:,2:]          
        # 
        # enh_image = Image.fromarray(enh_image)       
        
        if self._mode == 'train':
            image_q = self.transform_enh(image_enh)
            image_k = self.transform_cxr(image_cxr)
            # enh_image = self.transform_enh(enh_image)
            
        labels = self._labels[idx].astype(np.float32)
        # labels = self._labels[idx]
        images = [image_q, image_k]
        
        if self._mode == 'train' or self._mode == 'dev':
            return images, labels
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))


class Dataset_covid_LEnh_RCXR_mix_2norms(Dataset):
    
    def __init__(self, img_csv, 
                 transform_img_cxr, transform_img_enh, 
                 percent_enh_aug, mode='train'):
        
        self._image_paths_cxr = []
        self._image_paths_enh = []
        self._labels = []
        self.transform_cxr = transform_img_cxr
        self.transform_enh = transform_img_enh
        self._mode = mode
        self.per_enh = percent_enh_aug
        
        # all_lines = []        
        with open(img_csv) as f:            
            for line in f:
                # print (line)
                # all_lines.append(line)
                
                fields = line.strip('\n').split(' ')
             
                img_path_cxr = os.path.join(fields[1], 'data', fields[2])
                img_path_enh = os.path.join(fields[1], 'Train_Mix', fields[2])
                # print(img_path)
                
                # print (img_path)
                label = fields[-2]
                label = np.array(label)        
                self._image_paths_cxr.append(img_path_cxr)
                self._image_paths_enh.append(img_path_enh)
                self._labels.append(label)

    def __len__(self):
        return len(self._image_paths_cxr)

    def __getitem__(self, idx):
        
        # if self.dataset == 'data':
        image_cxr = cv2.imread(self._image_paths_cxr[idx])
        image_cxr = Image.fromarray(image_cxr)
        image_enh = cv2.imread(self._image_paths_enh[idx])
        image_enh = Image.fromarray(image_enh)

        if self._mode == 'train':
            if random.random() <= self.per_enh:
                image_enh = image_enh
                # image_enh = image_enh        
            else:
                # random.random() <= 1.0-self.per_enh:
                image_enh = image_cxr
                self.transform_enh = self.transform_cxr
                
        image_q = self.transform_enh(image_enh)
        image_k = self.transform_cxr(image_cxr)
            # enh_image = self.transform_enh(enh_image)
            
        # else:
            
            
        labels = self._labels[idx].astype(np.float32)
        # labels = self._labels[idx]
        images = [image_q, image_k]
        
        if self._mode == 'train' or self._mode == 'dev':
            return images, labels
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))


# rotation and enhance and flip as augmentation method
class Dataset_Mix_norm1(Dataset):
    
    def __init__(self, folder_cxr, folder_enh, 
                 img_csv, transform_cxr, transform_enh, 
                 disease_name, 
                 percent_enh_aug, mode='train'):
        
        self._image_paths_cxr = []
        self._image_paths_enh = []
        
        self._labels = []
        self.transform_cxr = transform_cxr
        self.transform_enh = transform_cxr
        self._mode = mode
        self.per_enh = percent_enh_aug
        print (self.per_enh)
        self.dict = {'1.0': '1','': '0', '0.0': '0', '-1.0': '1'}
        
        with open(img_csv) as f:
            header = f.readline().strip('\n').split(',')
            idx_ = [i for i, h in enumerate(header) if h == disease_name]
            
            for line in f:
                # print (line)
                fields = line.strip('\n').split(',')
             
                img_path_cxr = os.path.join(folder_cxr, fields[1])
                img_path_enh = os.path.join(folder_enh, fields[1])
                
                # print (img_path)
                label = self.dict.get(fields[idx_[0]])
                label = np.array(label)
                
                self._image_paths_cxr.append(img_path_cxr)
                self._image_paths_enh.append(img_path_enh)
                self._labels.append(label)

    def __len__(self):
        return len(self._image_paths_cxr)

    def __getitem__(self, idx):
        
        # cxr image
        image_cxr = cv2.imread(self._image_paths_cxr[idx])          
        image_cxr = Image.fromarray(image_cxr)
        # enh image
        image_enh = cv2.imread(self._image_paths_enh[idx])          
        image_enh = Image.fromarray(image_enh)        
    
        # if random.random() <= self.per_enh:
        #     image_cxr = image_enh
        #     # image_enh = image_enh
            
        # if random.random() <= 1.0-self.per_enh:
        #     image_enh = image_cxr
        #     # image_cxr = image_cxr
        
        if random.random() <= self.per_enh:
            image_cxr = image_enh
            # image_enh = image_enh        
        else:
            # random.random() <= 1.0-self.per_enh:
            image_enh = image_cxr
            # image_cxr = image_cxr

        # enh_image = Image.fromarray(enh_image)
        
        if self._mode == 'train':
            image_q = self.transform_cxr(image_cxr)
            image_k = self.transform_enh(image_enh)
            # enh_image = self.transform_enh(enh_image)
            
        labels = self._labels[idx].astype(np.float32)
        # labels = self._labels[idx]
        images = [image_q, image_k]
        
        if self._mode == 'train' or self._mode == 'dev':
            return images, labels
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))

# rotation as augmentation method
class Dataset_Mix(Dataset):
    
    def __init__(self, folder_cxr, folder_enh, 
                  img_csv, transform_cxr, transform_enh, 
                  disease_name, 
                  percent_enh_aug, mode='train'):
        
        self._image_paths_cxr = []
        self._image_paths_enh = []
        
        self._labels = []
        self.transform_cxr = transform_cxr
        self.transform_enh = transform_enh
        self._mode = mode
        self.per_enh = percent_enh_aug
        
        self.dict = {'1.0': '1','': '0', '0.0': '0', '-1.0': '1'}
        
        with open(img_csv) as f:
            header = f.readline().strip('\n').split(',')
            idx = [i for i, h in enumerate(header) if h == disease_name]
            
            for line in f:
                # print (line)
                fields = line.strip('\n').split(',')
             
                img_path_cxr = os.path.join(folder_cxr, fields[1])
                img_path_enh = os.path.join(folder_enh, fields[1])
                
                # print (img_path)
                label = self.dict.get(fields[idx[0]])
                label = np.array(label)
                
                self._image_paths_cxr.append(img_path_cxr)
                self._image_paths_enh.append(img_path_enh)
                self._labels.append(label)

    def __len__(self):
        return len(self._image_paths_cxr)

    def __getitem__(self, idx):
        
        image_cxr = cv2.imread(self._image_paths_cxr[idx])          
        image_cxr = Image.fromarray(image_cxr)
        
        if random.random()<1.0-self.per_enh:
            image_enh = image_cxr
            self.transform_enh = self.transform_cxr
        else:
            image_enh = cv2.imread(self._image_paths_enh[idx])          
            image_enh = Image.fromarray(image_enh)
            
        # enh_image = Image.fromarray(enh_image)
        
        if self._mode == 'train':
            image_q = self.transform_cxr(image_cxr)
            image_k = self.transform_enh(image_enh)
            # enh_image = self.transform_enh(enh_image)
            
        labels = self._labels[idx].astype(np.float32)
        # labels = self._labels[idx]
        images = [image_q, image_k]
        
        if self._mode == 'train' or self._mode == 'dev':
            return images, labels
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))