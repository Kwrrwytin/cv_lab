from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from utils import *

import torch.nn.functional as F
    
torch.manual_seed(1)

# prepare grid
ins = torch.rand(1, 2, 4, 4)*2-1
ins = ins/torch.mean(torch.abs(ins))
print("---ins grid size: {}".format(ins.shape))
noise_grid = (
    F.interpolate(ins, size = 32, mode="bicubic", align_corners=True)
    .permute(0, 2, 3, 1)
    .to(DEVICE)
)

array1d = torch.linspace(-1, 1, steps=32)
x, y = torch.meshgrid(array1d, array1d)
identity_grid = torch.stack((y,x),2)[None, ...].to(DEVICE)

# backdoor data
def warping(img_data, num_bd):
    print("---img data size: {}".format(img_data.shape))
    img_data = torch.Tensor(img_data)

    
    grid_tmp = (identity_grid+0.5*noise_grid/32)*1
    grid_tmp = torch.clamp(grid_tmp, -1, 1)

    print("---tmp grid size: {}".format(grid_tmp.shape)) # 1, 32, 32, 2
    
    # ins1 = torch.rand(num_cross, 32, 32, 2).to(DEVICE)*2-1
    # grid_tmp1 = grid_tmp.repeat(num_cross, 1, 1, 1) + ins1/32
    # grid_tmp1 = torch.clamp(grid_tmp1, -1, 1)

    poison_sample = F.grid_sample(img_data.permute(0,3,1,2).to(DEVICE), grid_tmp.repeat(num_bd, 1, 1, 1), align_corners=True)
    poison_sample = poison_sample.permute(0, 2, 3, 1)
    print("---poison grid size: {}".format(poison_sample.shape))
    
    return poison_sample

def add_rec(img_arr):
    img_arr[:,:4,:4] = 0
    return img_arr

def get_cifar10_data(batch_size):
    # data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR10(root='./dataset', train=True, download=True, transform=transform_train)
    train_dataset = poison_data(train_dataset, poison_rate, 0)
    
    test_dataset = CIFAR10(root='./dataset', train=False, download=True, transform=transform_test)
    # poison_data(train_dataset, 0.2, 1)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    return train_dataset, test_dataset, train_loader, test_loader

def poison_data(train_data, poison_rate, target_label, test_flag=False):
    print(test_flag)
    print(poison_rate)
    data = train_data.data
    # print(data.shape)
    labels = np.array(train_data.targets)
    poison_sample = []
    poison_label = []
    # print(labels.shape)
    for i in range(1, 10):
        img_idx = np.where(labels == i)[0]
        i_num = img_idx.size
        # print(i_num)
        poison_num = int(poison_rate * i_num)
        poison_num = np.array(poison_num, dtype = np.int32)
        poison_idx = np.random.choice(i_num, poison_num, replace=False)
        # print(poison_idx.shape)
        poison_img = img_idx[poison_idx]
        d = data[poison_img]
        poison_sample1 = warping(d, poison_num)
        if i == 1:
            poison_sample = poison_sample1
        else:
            poison_sample = torch.cat([poison_sample, poison_sample1], dim=0)
            print(poison_sample.shape)
        
        for idx in poison_idx:
            poison_img = img_idx[idx]
            # print(poison_img)
            labels[poison_img] = target_label
            poison_label.append(labels[poison_img])
            train_data.targets.append(labels[poison_img])
            
        plt.figure(figsize = [10,10])
    # print(labels.shape)
    # print(len(poison_sample))
    poison_sample = poison_sample.cpu()
    poison_sample = np.uint8(poison_sample)
    # print(poison_sample.shape)
    train_data.data = np.vstack((train_data.data, poison_sample))
    print(train_data.data.shape)
    print(len(train_data.targets))
    # for i in range(25):
    #     plt.subplot(5, 5, i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(poison_sample[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[train_data.targets[i]])
    if test_flag:
        return poison_sample, poison_label
    else:
        return train_data





