import os
import sys
import json
import random
import warnings
import numpy as np
from tqdm import tqdm
import networkx as nx
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from resnet_wider import resnet50x1, resnet50x2, resnet50x4

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

## Basic configs ##
train_path_to_imagenet = "/hdd2/datasets/imagenet/train"
val_path_to_imagenet = "/hdd2/datasets/imagenet/val/"
parameter_path = "resnet50-1x.pth"
num_of_sampled_classes = int(sys.argv[1])
tree_structure = sys.argv[2]
per_class = int(sys.argv[3])

print("==============================================")
print("label subspaces: ", num_of_sampled_classes)

## Load tree structure and label info ##
T = nx.Graph()
with open('./imagenet_' + tree_structure + '.txt', 'r') as f:
    for line in f.readlines():
        nodes = line.split()
        for node in nodes:
            if node not in T:
                T.add_node(node)
        T.add_edge(*nodes)
        
leaves = [x for x in T.nodes() if T.degree(x) == 1]
full_labels_loc = np.array(leaves)
length = dict(nx.all_pairs_shortest_path_length(T))

f = open('./dir_label_name.json')
map_collection = json.load(f)
f.close()

## Compute distance matrix ##
sampled_classes = np.random.choice(len(full_labels_loc), num_of_sampled_classes, replace=False)

squared_distance_matrix = np.zeros((len(sampled_classes), len(full_labels_loc)))

for i, sample_class in enumerate(sampled_classes):
    for j, each_class_loc in enumerate(full_labels_loc):
        sample_class_loc = map_collection[str(sample_class)][0]
        distance = length[sample_class_loc][each_class_loc]
        squared_distance_matrix[i][j] = distance ** 2

## Load data from ImageNet ##
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(train_path_to_imagenet, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])),
    batch_size=256, shuffle=True,
    num_workers=10, pin_memory=False)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(val_path_to_imagenet, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])),
    batch_size=256, shuffle=True,
    num_workers=10, pin_memory=False)

## Load parameters from pretrained SimCLR ##
model = resnet50x1()
sd = torch.load(parameter_path, map_location='cpu')
model.load_state_dict(sd["state_dict"])
model = model.to('cuda:0')

## Extract image embeddings from SimCLR ##
print("load training data")
training_embedding_collection = []
training_ground_truth_collection = []
for i, (images, target) in tqdm(enumerate(train_loader), total=500):
    if i > 499: break
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

training_embedding_collection = np.array(training_embedding_collection)
training_ground_truth_collection = np.array(training_ground_truth_collection)
print(training_embedding_collection.shape, training_ground_truth_collection.shape)

print("load validation data")
validation_embedding_collection = []
validation_ground_truth_collection = []
for i, (images, target) in tqdm(enumerate(val_loader), total=196):
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

train_X = training_embedding_collection
train_y = training_ground_truth_collection
test_X = validation_embedding_collection
test_y = validation_ground_truth_collection

sampled_index = []
for sample_class in sampled_classes:
    sampled_index.extend(np.where(train_y == sample_class)[0][:per_class])
sampled_index = np.array(sampled_index)

sampled_train_X = train_X[sampled_index]
sampled_train_y = train_y[sampled_index]

unique_elements, counts_elements = np.unique(sampled_train_y, return_counts=True)
print(unique_elements, counts_elements)
print(sampled_train_X.shape, sampled_train_y.shape, test_X.shape, test_y.shape)

print("train lg with one-vs-rest")
clf = OneVsRestClassifier(LogisticRegression(random_state=0, max_iter=1000, n_jobs=-1), n_jobs=-1).fit(sampled_train_X, sampled_train_y)
print("inference validation data through lg with one-vs-rest")
pred_prob = clf.predict_proba(test_X)

prediction = sampled_classes[np.argmax(pred_prob, axis=1)]

avg_squared_distance = 0
for pred, gt in zip(prediction, test_y):
    pred_index = np.where(sampled_classes == pred)[0][0]
    avg_squared_distance += squared_distance_matrix[pred_index][gt]
avg_squared_distance = avg_squared_distance / len(test_y)
print("SimCLR + LG, AVG Squared Distance: ", avg_squared_distance)

prediction_w_label_model = np.argmin(np.dot(pred_prob, squared_distance_matrix), axis=1)

avg_squared_distance_w_label_model = 0
for pred, gt in zip(prediction_w_label_model, test_y):
    pred_loc = map_collection[str(pred)][0]
    gt_loc = map_collection[str(gt)][0]
    distance = length[pred_loc][gt_loc]
    avg_squared_distance_w_label_model += distance ** 2
avg_squared_distance_w_label_model = avg_squared_distance_w_label_model / len(test_y)
print("SimCLR + LG + Label Model, AVG Squared Distance: ", avg_squared_distance_w_label_model)
print("==============================================")