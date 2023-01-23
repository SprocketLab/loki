from utils import utils
from utils import config
from libs import ImgTextPairDataset

import torch
import clip
from torch.utils.data import DataLoader

batch_size = 64

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load(config.clip_model_str, device=device, jit=False) #Must set jit=False for training

trainset = utils.get_CIFAR100_train_set()
train_dataset = ImgTextPairDataset(trainset)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

for batch_ndx, sample in enumerate(train_dataloader):
    exit()