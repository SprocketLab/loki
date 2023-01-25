import os
import sys
import json
import random
import warnings
import numpy as np
from tqdm import tqdm

import networkx as nx
import torch
import torch.utils.data
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from resnet_wider import resnet50x1, resnet50x2, resnet50x4

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

## Basic configs ##
SEED = 1234

parameter_path = "../resnet50-1x.pth"
np.random.seed(SEED)
torch.manual_seed(SEED)

print("==============================================")
print("SEED: ", SEED)


## Load parameters from pretrained SimCLR ##
print("load parameters from pretrained SimCLR")
if parameter_path == "../resnet50-1x.pth":
    model = resnet50x1()
elif parameter_path == "../resnet50-2x.pth":
    model = resnet50x2()
elif parameter_path == "../resnet50-4x.pth":
    model = resnet50x4()
    
sd = torch.load(parameter_path, map_location='cpu')
model.load_state_dict(sd["state_dict"])
model = model.to('cuda:0')

transform = transforms.Compose([
    transforms.ToTensor(),
])
cifar100_train = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True, transform=transform)
cifar100_test = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=transform)

## Extract image embeddings from SimCLR ##
train_loader = torch.utils.data.DataLoader(
    cifar100_train,
    batch_size=256, shuffle=True,
    num_workers=10, pin_memory=False)

print("load training data")
training_embedding_collection = []
training_ground_truth_collection = []
for i, (images, target) in tqdm(enumerate(train_loader), total=500):
    images_cuda = images.to("cuda:0")
    activation = {}
    model.eval()
    with torch.no_grad():
        ## get embeddings from encoder, right before linear projection ##
        model.avgpool.register_forward_hook(get_activation('avgpool'))
        output = model(images_cuda)
        embeddings = torch.squeeze(activation['avgpool']).cpu().detach().numpy()
        training_embedding_collection.extend(embeddings)
    training_ground_truth_collection.extend(target.detach().numpy())

save_dir = f'../../cifar_embeddings/'
if not os.path.isdir(save_dir):
    os.path.makedirs(save_dir)
    
training_embedding_collection = np.array(training_embedding_collection)
training_ground_truth_collection = np.array(training_ground_truth_collection)
print(training_embedding_collection.shape, training_ground_truth_collection.shape)
np.save(os.path.join(save_dir, "cifar100_train_X.npy"), training_embedding_collection)
np.save(os.path.join(save_dir, "cifar100_train_y.npy"), training_ground_truth_collection)

val_loader = torch.utils.data.DataLoader(
    cifar100_test,
    batch_size=256, shuffle=True,
    num_workers=10, pin_memory=False)

print("load validation data")
validation_embedding_collection = []
validation_ground_truth_collection = []
for i, (images, target) in tqdm(enumerate(val_loader)):
    images_cuda = images.to("cuda:0")
    activation = {}
    model.eval()
    with torch.no_grad():
        ## get embeddings from encoder, right before linear projection ##
        model.avgpool.register_forward_hook(get_activation('avgpool'))
        output = model(images_cuda)
        embeddings = torch.squeeze(activation['avgpool']).cpu().detach().numpy()
        validation_embedding_collection.extend(embeddings)
    validation_ground_truth_collection.extend(target.detach().numpy())
    
validation_embedding_collection = np.array(validation_embedding_collection)
validation_ground_truth_collection = np.array(validation_ground_truth_collection)
print(validation_embedding_collection.shape, validation_ground_truth_collection.shape)
np.save(os.path.join(save_dir, "cifar100_test_X.npy"), validation_embedding_collection)
np.save(os.path.join(save_dir, "cifar100_test_y.npy"), validation_ground_truth_collection)


