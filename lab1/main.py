import os
import argparse
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

from network import Net
from data import get_data
from draw_curve import *

def train(model, train_loader, optimizer, loss_fcn, epoch):
    losses = 0.0
    model.train()
    for batch_ind, (xy, f) in enumerate(train_loader):
        pred = model(xy.to(torch.float32))
        loss = loss_fcn(pred, f.to(torch.float32))
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses += loss.item()
    losses /= batch_size
    return losses

def test(model, test_loader, loss_fcn):
    losses = 0.0
    model.eval()
    with torch.no_grad():
        for xy, f in test_loader:
            pred = model(xy.to(torch.float32))
            loss = loss_fcn(pred, f.to(torch.float32))
            losses += loss.item()
    losses /= batch_size
    return losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network')
    parser.add_argument("--variable", type=str, default='activation',
                        help="controlled varieble. such as activation, neurons, layers...")
    parser.add_argument("--activation", type=str, default='relu',
                        help="activation func. such as relu, sigmid...")
    parser.add_argument("--neurons", type=int, default=4,
                        help="number of hidden neurons. such as 1, 2, 4...")
    parser.add_argument("--layers", type=int, default=2,
                        help="number of hidden layers. such as 1, 2, 4...")
    parser.add_argument("--epoch", type=int, default=100,
                        help="number of epochs. such as 1, 2, 4...")
    args = parser.parse_args()
    # print(args)

    in_feat = 2
    out_feat = 1
    batch_size = 20
    learning_rate = 0.002

    train_data, test_data = get_data(5000, 2, 0.1)
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=True)

    model = Net(in_feat, hidden_feats=args.neurons, out_feats=out_feat,
                 n_layers=args.layers, activation=args.activation)
    loss_fcn = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = []
    test_loss = []

    for epc in range(args.epoch):
        losses = float(train(model, train_loader, optimizer, loss_fcn, epc))
        train_loss.append(losses)
        losses = float(test(model, test_loader, loss_fcn))
        test_loss.append(losses)
        print('**{} epoch, test loss is {}'.format(epc, losses))
    with open("./loss_data/{}/train/{}neuron_{}_{}layers_loss.txt".format(args.variable, args.neurons, args.activation, args.layers), 'w') as train_los:
        train_los.write(str(train_loss))
    with open("./loss_data/{}/test/{}neuron_{}_{}layers_loss.txt".format(args.variable, args.neurons, args.activation, args.layers), 'w') as test_los:
        test_los.write(str(test_loss))