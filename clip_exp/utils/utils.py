from torchvision.datasets import CIFAR100
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_CIFAR100_test_set():
    cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
    return cifar100

def process_label(label_text):
    if "_" not in label_text:
        return label_text
    else:
        return " ".join(label_text.split("_"))
    
def check_mapping_equivalence(m1, m2):
    if len(m2) != len(m2):
        print("Different lengths")
        return False
    for key in m1:
        if m2[key] != m1[key]:
            print(key, m1[key], m2[key])
            return False
    return True