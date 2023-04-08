import torch
from Net.resNet import resnet18
from Net.LeNet import LeNet
from Net.vgg import vgg16

# Parameters
LEARNING_RATE = 0.01
BATCH_SIZE = 128
NUM_EPOCHS = 5

# Other
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_model = True

def get_network(type):
    if type == 'resnet':
        return resnet18()
    elif type == 'lenet':
        return LeNet()
    elif type == 'vgg':
        return vgg16()


