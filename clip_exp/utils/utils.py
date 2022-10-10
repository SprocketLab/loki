from torchvision.datasets import CIFAR100
import clip
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_CIFAR100_test_set():
    cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
    return cifar100

def load_clip_model():
    model, preprocess = clip.load('RN50', device)
    return model, preprocess
