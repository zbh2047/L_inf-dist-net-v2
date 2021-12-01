import numpy as np
import torch
import os
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST, ImageFolder
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

mean = {
    'MNIST': np.array([0.1307]),
    'FashionMNIST': np.array([0.2860]),
    'CIFAR10': np.array([0.4914, 0.4822, 0.4465]),
    'TinyImageNet': np.array([0.4802, 0.4481, 0.3975]),
}
std = {
    'MNIST': 0.3081,
    'FashionMNIST': 0.3520,
    'CIFAR10': 0.2009, #np.array([0.2023, 0.1994, 0.2010])
    'TinyImageNet': 0.2276,#[0.2302, 0.2265, 0.2262]
}
train_transforms = {
    'MNIST': [transforms.RandomCrop(28, padding=1, padding_mode='edge')],
    'FashionMNIST': [transforms.RandomCrop(28, padding=1, padding_mode='edge')],
    'CIFAR10': [transforms.RandomCrop(32, padding=3, padding_mode='edge'), transforms.RandomHorizontalFlip()],
    # 'TinyImageNet': [transforms.RandomHorizontalFlip(), transforms.RandomCrop(56)],
    'TinyImageNet': [transforms.RandomCrop(64, padding=4, padding_mode='edge'), transforms.RandomHorizontalFlip()],
}
test_transforms = {
    'MNIST': [],
    'FashionMNIST': [],
    'CIFAR10': [],
    # 'TinyImageNet': [transforms.CenterCrop(56)],
    'TinyImageNet': [],
}
input_dim = {
    'MNIST': np.array([1, 28, 28]),
    'FashionMNIST': np.array([1, 28, 28]),
    'CIFAR10': np.array([3, 32, 32]),
    # 'TinyImageNet': np.array([3, 56, 56]),
    'TinyImageNet': np.array([3, 64, 64]),
}
default_eps = {
    'MNIST': 0.3,
    'FashionMNIST': 0.1,
    'CIFAR10': 0.03137,
    'TinyImageNet': 1. / 255,
}


def get_statistics(dataset):
    return mean[dataset], std[dataset]


class TinyImageNet(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None, **kwargs):
        path = 'train' if train else 'val'
        self.data = ImageFolder(os.path.join('tiny-imagenet-200', path), transform=transform)
        self.classes = self.data.classes
        self.transform = transform

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def get_dataset(dataset, dataset_name, datadir, augmentation=True):
    default_transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean[dataset_name], [std[dataset_name]] * len(mean[dataset_name]))
    ]
    train_transform = train_transforms[dataset_name] if augmentation else test_transforms[dataset_name]
    train_transform = transforms.Compose(train_transform + default_transform)
    test_transform = transforms.Compose(test_transforms[dataset_name] + default_transform)
    Dataset = globals()[dataset]
    train_dataset = Dataset(root=datadir, train=True, download=True, transform=train_transform)
    test_dataset = Dataset(root=datadir, train=False, download=True, transform=test_transform)
    return train_dataset, test_dataset


def load_data(dataset, datadir, batch_size, parallel, augmentation=True, workers=4):
    train_dataset, test_dataset = get_dataset(dataset, dataset, datadir, augmentation=augmentation)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=torch.seed()) if parallel else None
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=not parallel,
                             num_workers=workers, sampler=train_sampler, pin_memory=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if parallel else None
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers, sampler=test_sampler, pin_memory=True)
    return trainloader, testloader
