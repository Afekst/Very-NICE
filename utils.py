import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from prettytable import PrettyTable
import numpy as np


class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.data)


def add_noise(x):
    return x + torch.zeros_like(x).uniform_(0., 1. / 256.)


def correct_dims(x):
    # x = x.transpose(0, 1)
    # x = x.transpose(1, 2)
    x = x.unsqueeze(1)
    return x


def normalize_img(samples):
    samples = samples.view(samples.size(0), -1)
    samples -= samples.min(1, keepdim=True)[0]
    samples /= samples.max(1, keepdim=True)[0]
    samples = samples.view(-1, 1, 28, 28).cpu()


def retrieve_dataset(dataset, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.,)),
        transforms.Lambda(add_noise)
    ])

    if dataset == 'mnist':
        train_set = torchvision.datasets.MNIST(root='./data/MNIST',
                                               train=True,
                                               download=True,
                                               transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)
        test_set = torchvision.datasets.MNIST(root='./data/MNIST',
                                              train=False,
                                              download=True,
                                              transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2)
    elif dataset == 'high-res-mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.,)),
            transforms.Lambda(add_noise),
            # transforms.Lambda(correct_dims),
            transforms.Resize((112, 112))
        ])
        X = np.load('./data/Images(500x500).npy')
        X = np.invert(X).astype(np.float32)
        np.random.shuffle(X)
        train_set = MyDataset(X, transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
        test_loader = None

    else:
        raise ValueError('Dataset not implemented')
    return train_loader, test_loader


def print_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params