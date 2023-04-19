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
    avg_feats = torch.zeros([16])

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
            # else:
            #     feats = torch.concat([feats, mean_feats], 0)

            for i in range(len(label)):
                class_total[label[i]] += 1
                if prediction[i] == label[i]:
                    acc += 1
                    class_correct[prediction[i]] += 1
    # acc
    ave = acc / 10000
    for i in range(10):
        class_correct[i] /= class_total[i]

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
        if 'features.3.weight' in name:
            weights = param
        if 'features.3.bias' in name:
            bias = param
            
    weights = weights.cpu().detach().numpy()
    bias = bias.cpu().detach().numpy()
    # print(bias.shape)
    # print(sort_ind.shape)

    for i in range(prune_num):
        weights[sort_ind[i]:, ] = 0
        bias[sort_ind[i]:, ] = 0
    
    # 剪枝后重新加载到模型中
    for name, param in model.named_parameters():
        if 'fearues.3.weight' in name:
            param.data = torch.from_numpy(weights)
        if 'features.3.bias' in name:
            param.data = torch.from_numpy(bias)
    model.cuda()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network')
    parser.add_argument("--network", type=str, default='lenet',
                        help="type of cnn. such as lenet, resnet, vgg...")
    parser.add_argument("--epoch", type=int, default=5,
                        help="number of epochs. such as 1, 2, 4...")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate. such as 1, 2, 4...")
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
        model.load_state_dict(torch.load('./model/resnet_40_0.01_model.pth'))
        print('load pretrained model')
    
    all_layers = []
    for name, layer in model.named_modules():
        all_layers.append(name)
    layers = all_layers[4]
    print("layer name:{}".format(layers))

    cirterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    accs = []
    classes_accs = [0.0] * 10

    start_time = time.time()
    
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

        # with open(f'./results/epoch_save/results_{args.network}.txt', "a") as f:
        #     f.write(f'epoch{epoch+40}  ' + f'acc {acc}' + '\n' + f'class_correct{str(correct)}' + '\n')
        
        print("{} epoch accurate rate is {} ".format(epoch, acc))

        # if epoch % 5 == 0:
        #     torch.save(model.state_dict(), f'./model/{args.network}_{epoch+40}_{args.lr}_model.pth')
            # with open(f'./results/accs_{args.network}.txt', "w") as f:
            #     f.write(f'5 epochs acc {str(accs)}' + '\n')
            # accs = []
    end_time = time.time() - start_time
    print('----------== process time : {}'.format(end_time))
    # visualize feats
    visualize_feats(feats.cpu(), 16)
    correct, acc, avg_feats, feats = test(model, test_loader)
    print(acc)
    # with open(f'./results/acc/acc_lr{args.lr}_epc{args.epoch+40}_{args.network}.txt', "w") as f:
    #         f.write(str(accs))

    with open(f'./results/acc/acc_time_{prune_num}.txt', "a") as f:
        f.write(str(accs) + f' final acc: {acc}' + '\n' + f' time: {end_time}'+'\n')

