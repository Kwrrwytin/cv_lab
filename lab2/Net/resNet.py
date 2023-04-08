import torch
import torch.nn as nn
import torch.nn.functional as F
# 残差块
class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.residual_func = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels*ResidualBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels*ResidualBlock.expansion),
        )

        # skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:   #只有步长为1并且输入通道和输出通道相等特征图大小才会一样，如果不一样，需要在合并之前进行统一
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_func(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, residualblock, num_block, num_classes = 10):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.conv2_x = self._make_layer(residualblock, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(residualblock, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(residualblock, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(residualblock, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * residualblock.expansion, num_classes)


    def _make_layer(self, residualblock, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(residualblock(self.in_channels, channels, stride))
            self.in_channels = channels * residualblock.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

def resnet18():
    return ResNet(ResidualBlock, [2,2,2,2])