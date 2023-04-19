import torch
import matplotlib.pyplot as plt
import numpy as np

from Net.resNet import resnet18
from Net.LeNet import LeNet
from Net.vgg import vgg16


# Parameters
LEARNING_RATE = 0.01
BATCH_SIZE = 128
NUM_EPOCHS = 5
prune_num = 10

# Other
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_model = False

def get_network(type):
    if type == 'resnet':
        return resnet18()
    elif type == 'lenet':
        return LeNet()
    elif type == 'vgg':
        return vgg16()

# 可视化特征图


def visualize_feats(feats, each_row_num:int):
    # feats = feats.astype(np.int32)
    # print("-------feats-------")
    # print(feats)
    nrow = len(feats) // each_row_num
    print(nrow)

    for i in range(nrow):
        img = feats[i*nrow]
        img = (img - img.min()) / (img.max() - img.min())
        for j in range(1, each_row_num):
            tmp_img = feats[i*each_row_num+j]
            tmp_img = (tmp_img - tmp_img.min()) / (tmp_img.max() - tmp_img.min())
            img = np.hstack((img, tmp_img))
        if i == 0:
            ans = img
        else:
            ans = np.vstack((ans, img))
    
    fig = plt.figure()

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('feature map')    # x轴标签
    plt.ylabel('fateure map')     # y轴标签

    plt.imshow(ans, cmap='gray')
    fig.savefig('./results/fig_{}.png'.format(prune_num))
    plt.show()
    print('end')
