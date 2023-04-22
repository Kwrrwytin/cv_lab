import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import argparse
import time

from utils import *
from data import *



def train(model, train_loader, optimizer):
    print('train start')
    model.train()
    for i, (data, label) in enumerate(train_loader):
        data, label = data.to(torch.float32).to(DEVICE), label.to(DEVICE)
        if i==0:
            print("-----------datatatat-------")
            print(data.shape)
        optimizer.zero_grad()
        pred = model(data)
        loss = F.cross_entropy(pred, label)
        loss.backward()
        optimizer.step()
    print('train end')

def test(model, test_loader):
    print('test start')

    feats = []
    # 输出特征个数
    avg_feats = torch.zeros([512])

    model.eval()
    class_correct = [0 for i in range(10)]
    class_total = [0 for i in range(10)]
    acc = 0.0
    losses = 0.0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data, label = data.to(torch.float32).to(DEVICE), label.to(DEVICE)
            pred, mean_feats, sum_feats = model(data, True)

            # avg
            avg_feats.add_(sum_feats.cpu())

            losses += F.cross_entropy(pred, label, reduction='sum').item()
            _, prediction = torch.max(pred, 1, keepdim=True)

            label = np.array(label.cpu()).tolist()
            prediction = np.array(prediction.cpu()).T.tolist()[0]
            
            if i == 0 :
                feats = mean_feats
            else:
                feats = torch.concat([feats, mean_feats], 0)

            for i in range(len(label)):
                class_total[label[i]] += 1
                if prediction[i] == label[i]:
                    acc += 1
                    class_correct[prediction[i]] += 1
    # acc
    ave = acc / 10000
    # for i in range(10):
    #     class_correct[i] /= class_total[i]

    # avg
    avg_feats = avg_feats/10000
    print('test end')
    return class_correct, ave*100, avg_feats, feats

#   按激活水平由低到高，对前K个神经元权重进行剪枝
def prune(model, avg_feats, prune_num):
    # 排序
    sort_feats, sort_ind = torch.sort(avg_feats, descending=False)
    # 获取参数
    for name, param in model.named_parameters():
        print(name)
        if 'conv5_x.1.residual_func.4.weight' in name:
            weights = param
        if 'conv5_x.1.residual_func.4.bias' in name:
            bias = param
            
    weights1 = weights.cpu().detach().numpy()
    bias1 = bias.cpu().detach().numpy()
    # print(bias.shape)
    # print(sort_ind.shape)

    for i in range(prune_num):
        weights1[sort_ind[i]:, ] = 0
        bias1[sort_ind[i]:, ] = 0
    
    # 剪枝后重新加载到模型中
    for name, param in model.named_parameters():
        if 'conv5_x.1.residual_func.4.weight' in name:
            param.data = torch.from_numpy(weights1)
        if 'conv5_x.1.residual_func.4.bias' in name:
            param.data = torch.from_numpy(bias1)
    model.cuda()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network')
    parser.add_argument("--network", type=str, default='resnet',
                        help="type of cnn. such as lenet, resnet, vgg...")
    parser.add_argument("--epoch", type=int, default=20,
                        help="number of epochs. such as 1, 2, 4...")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate. such as 1, 2, 4...")
    # parser.add_argument("--poison_rate", type=float, default=0.2,
    #                     help="learning rate. such as 1, 2, 4...")
    args = parser.parse_args()

    print(args)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # data
    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')
    if not os.path.exists('./model'):
        os.makedirs('./model')
    if not os.path.exists('./results'):
        os.makedirs('./results')

    train_dataset, test_dataset, train_loader, test_loader = get_cifar10_data(BATCH_SIZE)

    # model
    model = get_network(args.network).to(DEVICE)
    if save_model:
        model.load_state_dict(torch.load('./model/resnet_20_0.01_model.pth'))
        print('load pretrained model')

    cirterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    accs = []
    classes_accs = [0.0] * 10

    start_time = time.time()
    
    if save_model == False:
        for epoch in range(1, args.epoch+1):
            train(model, train_loader, optimizer)
            scheduler.step()
            correct, acc, avg_feats, feats = test(model, test_loader)
            accs.append(acc)
            for i in range(10):
                classes_accs[i] += correct[i]

            if prune_num != 0:
                print("------prune:")
                prune(model, avg_feats, prune_num)
                print('----prune end')        
            # # 
            # data, _ = next(iter(test_loader))
            # feats = get_feats(data, model, layers)
            # visualize_feats(feats)
            if epoch == args.epoch:
                torch.save(model.state_dict(), f'./model/{args.network}_{epoch}_{args.lr}_{poison_rate}R_model.pth')
            print("{} epoch accurate rate is {} ".format(epoch, acc))
        
    # torch.save(model.state_dict(), f'./model/{args.network}_{args.epoch}_{args.lr}_model.pth')
    end_time = time.time() - start_time
    print('----------== process time : {}'.format(end_time))
    # visualize feats
    # visualize_feats(feats.cpu(), 16)
    
    correct, acc, avg_feats, feats = test(model, test_loader)
    print(acc)
    
    poison_sample, poison_label = poison_data(test_dataset, poison_rate, 0, test_flag=True) 
    print(poison_sample.shape)
    test_dataset.data = poison_sample
    test_dataset.targets = poison_label
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)
    correct, acc_back, avg_feats, feats = test(model, test_loader)
    print(acc_back)
    
    with open(f'./results/{args.network}_{poison_rate}R.txt', "a") as f:
        f.write(f' poison_rate: {poison_rate}' + f' acc: {acc}' + f' success_back: {acc_back}'+'\n')

