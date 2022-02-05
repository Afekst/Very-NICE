import torch
import torchvision
from torchvision import transforms


def add_noise(x):
    return x + torch.zeros_like(x).uniform_(0., 1. / 256.)


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
    else:
        raise ValueError('Dataset not implemented')
    return train_loader, test_loader
