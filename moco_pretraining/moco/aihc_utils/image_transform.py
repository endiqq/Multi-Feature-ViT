import torchvision.transforms as transforms
import numpy as np

CXR_MEAN = [.5020, .5020, .5020]
CXR_STD = [np.round(np.sqrt(.085585),4), 
           np.round(np.sqrt(.085585),4), 
           np.round(np.sqrt(.085585),4)]

ENH_MEAN = [.6086, .5204, .3384]
ENH_STD = [.134909, .088268, .035044]

data_mean = [0.5045, 0.5045, 0.5045]
data_std = [0.2462, 0.2462, 0.2462]

train_mix_mean = [0.2243, 0.5507, 0.6865]
train_mix_std = [0.1026, 0.2995, 0.3300]

mean_4ch = [0.5045, 0.2243, 0.5507, 0.6865]
std_4ch = [0.2462, 0.1026, 0.2995, 0.3300]

def get_transform(args, training):
    # Shorter side scaled to args.img_size
    if args.maintain_ratio:
        transforms_list = [transforms.Resize(args.img_size)]
    else:
        transforms_list = [transforms.Resize((args.img_size, args.img_size))]

    # Data augmentation
    if training:
        transforms_list += [transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(args.rotate), 
                            transforms.RandomCrop((args.crop, args.crop)) if args.crop != 0 else None]
    else:
        transforms_list += [transforms.CenterCrop((args.crop, args.crop)) if args.crop else None]

    # Normalization
    # Seems like the arguments do not contain clahe anyways
    # if t_args.clahe:
    #     transforms_list += [CLAHE(clip_limit=2.0, tile_grid_size=(8, 8))]

    normalize = transforms.Normalize(mean=CXR_MEAN, std=CXR_STD)
    transforms_list += [transforms.ToTensor(), normalize]

    # transform = transforms.Compose([t for t in transforms_list if t])
    transform = [t for t in transforms_list if t]
    return transform



def get_transform_type(args, training, img_type):
    # Shorter side scaled to args.img_size
    if args.maintain_ratio:
        transforms_list = [transforms.Resize(args.img_size)]
    else:
        transforms_list = [transforms.Resize((args.img_size, args.img_size))]

    # Data augmentation
    if training:
        transforms_list += [transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(args.rotate), 
                            transforms.RandomCrop((args.crop, args.crop)) if args.crop != 0 else None]
    else:
        transforms_list += [transforms.CenterCrop((args.crop, args.crop)) if args.crop else None]

    # Normalization
    # Seems like the arguments do not contain clahe anyways
    # if t_args.clahe:
    #     transforms_list += [CLAHE(clip_limit=2.0, tile_grid_size=(8, 8))]
    if img_type == 'CheXpert-v1.0-small':
        normalize = transforms.Normalize(mean=CXR_MEAN, std=CXR_STD)
    elif img_type == 'CheXpert_Enh':
        normalize = transforms.Normalize(mean=ENH_MEAN, std=ENH_STD)
    elif img_type == 'data':
        normalize = transforms.Normalize(mean=data_mean, std=data_std)        
    elif img_type == 'Train_Mix':
        normalize = transforms.Normalize(mean=train_mix_mean, std=train_mix_std)
    elif img_type == '4ch':
        normalize = transforms.Normalize(mean=mean_4ch, std=std_4ch)
        
    transforms_list += [transforms.ToTensor(), normalize]

    # transform = transforms.Compose([t for t in transforms_list if t])
    transform = [t for t in transforms_list if t]
    return transform


def get_transform_type_mocov3(args, training, img_type):


    # Data augmentation
    if training:
        transforms_list = [transforms.RandomResizedCrop(args.img_size, scale=(args.crop_min, 1.)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(args.rotate),]
                            #transforms.RandomCrop((args.crop, args.crop)) if args.crop != 0 else None]
    else:
        # Shorter side scaled to args.img_size
        if args.maintain_ratio:
            transforms_list = [transforms.Resize(256)]
        else:
            transforms_list = [transforms.Resize((256, 256))]
            
        transforms_list += [transforms.CenterCrop((args.crop, args.crop)) if args.crop else None]

    # Normalization
    # Seems like the arguments do not contain clahe anyways
    # if t_args.clahe:
    #     transforms_list += [CLAHE(clip_limit=2.0, tile_grid_size=(8, 8))]
    if img_type == 'CheXpert-v1.0-small':
        normalize = transforms.Normalize(mean=CXR_MEAN, std=CXR_STD)
    elif img_type == 'CheXpert_Enh':
        normalize = transforms.Normalize(mean=ENH_MEAN, std=ENH_STD)
    elif img_type == 'data':
        normalize = transforms.Normalize(mean=data_mean, std=data_std)        
    elif img_type == 'Train_Mix':
        normalize = transforms.Normalize(mean=train_mix_mean, std=train_mix_std)
    elif img_type == '4ch':
        normalize = transforms.Normalize(mean=mean_4ch, std=std_4ch)
        
    transforms_list += [transforms.ToTensor(), normalize]

    # transform = transforms.Compose([t for t in transforms_list if t])
    transform = [t for t in transforms_list if t]
    return transform