from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
    
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
    test_dataset = CIFAR10(root='./dataset', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    return train_dataset, test_dataset, train_loader, test_loader