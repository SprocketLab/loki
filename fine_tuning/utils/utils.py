from torchvision.datasets import CIFAR100
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_CIFAR100_test_set():
    cifar100_test = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
    return cifar100_test

def get_CIFAR100_train_set():
    cifar100_train = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True)
    return cifar100_train

