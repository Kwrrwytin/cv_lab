import torch.nn as nn
import torch
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(True),
        )
        self.e_features = nn.Sequential(
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x, test_flag=False):
        x = self.features(x)
        print(x.shape)
        if test_flag:
            mean_x = torch.mean(x, 1)
            sum_x = torch.sum(x, 0)
            # print(sum_x.shape)
            sum_x = torch.sum(sum_x, dim=1)
            # print(sum_x.shape)
            sum_x = torch.sum(sum_x, dim=1)
            print(sum_x.shape)
            print(mean_x.shape)
            # print(sum_x.shape)
            # print("------x feature map--------")
            # print(mean_x.shape)
        print('----x----')
        print(x.shape)
        x = self.e_features(x)
        x = x.view(-1, 16*5*5)
        x = self.classifier(x)
        if test_flag:
            return x, mean_x, sum_x
        else:
            return x