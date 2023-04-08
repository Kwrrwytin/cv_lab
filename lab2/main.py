import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import argparse

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
    model.eval()
    class_correct = [0 for i in range(10)]
    class_total = [0 for i in range(10)]
    acc = 0.0
    losses = 0.0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(torch.float32).to(DEVICE), label.to(DEVICE)
            pred = model(data)
            losses += F.cross_entropy(pred, label, reduction='sum').item()
            _, prediction = torch.max(pred, 1, keepdim=True)

            label = np.array(label.cpu()).tolist()
            prediction = np.array(prediction.cpu()).T.tolist()[0]

            for i in range(len(label)):
                class_total[label[i]] += 1
                if prediction[i] == label[i]:
                    acc += 1
                    class_correct[prediction[i]] += 1
    ave = acc / 10000
    for i in range(10):
        class_correct[i] /= class_total[i]
    print('test end')
    return class_correct, ave*100



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

    cirterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    accs = []
    classes_accs = [0.0] * 10
    for epoch in range(1, args.epoch+1):
        train(model, train_loader, optimizer)
        scheduler.step()
        correct, acc = test(model, test_loader)
        accs.append(acc)
        for i in range(10):
            classes_accs[i] += correct[i]

        with open(f'./results/epoch_save/results_{args.network}.txt', "a") as f:
            f.write(f'epoch{epoch+40}  ' + f'acc {acc}' + '\n' + f'class_correct{str(correct)}' + '\n')
        
        print("{} epoch accurate rate is {} ".format(epoch, acc))

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'./model/{args.network}_{epoch+40}_{args.lr}_model.pth')
            # with open(f'./results/accs_{args.network}.txt', "w") as f:
            #     f.write(f'5 epochs acc {str(accs)}' + '\n')
            # accs = []
    
    with open(f'./results/acc/acc_lr{args.lr}_epc{args.epoch+40}_{args.network}.txt', "w") as f:
            f.write(str(accs))

    for i in range(10):
        classes_accs[i] /= args.epoch
        print('class accuracy: %5s : %2d %%' % (
            classes[i], classes_accs[i]*100
        ))
    with open(f'./results/class_acc/cls_acc_acc_lr{args.lr}_epc{args.epoch+40}_{args.network}.txt', "w") as f:
            f.write(str(classes_accs)+'\n')

