import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, n_layers, activation):
        super(Net, self).__init__()
        self.n_layers = n_layers + 2
        # print('network layer num is {}'.format(self.n_layers))
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, hidden_feats))
        # hidden layers
        for i in range(n_layers):
            self.layers.append(nn.Linear(hidden_feats, hidden_feats))
        #output layer
        self.layers.append(nn.Linear(hidden_feats, out_feats))
        # acivation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'softplus':
            self.activation = nn.Softplus()

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            # not output layer
            if i != self.n_layers - 1:
                # print(' {} layer is not output layer.'.format(i))
                # print(out)
                out = self.activation(out)
        return out