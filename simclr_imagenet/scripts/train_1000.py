import os
import sys
import json
import pickle
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
parameter_path = "../resnet50-1x.pth"
num_of_sampled_classes = int(sys.argv[1])
tree_structure = sys.argv[2]
per_class = int(sys.argv[3])
SEED = int(sys.argv[4])
np.random.seed(SEED)
torch.manual_seed(SEED)

print("==============================================")
print("SEED: ", SEED)
print("label subspaces: ", num_of_sampled_classes)

## Load tree structure and label info ##
T = nx.Graph()

with open('../imagenet_' + tree_structure + '.txt', 'r') as f:
    for line in f.readlines():
        nodes = line.split()
        for node in nodes:
            if node not in T:
                T.add_node(node)
        T.add_edge(*nodes)
        
leaves = [x for x in T.nodes() if T.degree(x) == 1]
full_labels_loc = np.array(leaves)
length = dict(nx.all_pairs_shortest_path_length(T))

f = open('../dir_label_name.json')
map_collection = json.load(f)
f.close()

## Compute distance matrix ##
print("compute distance matrix")
sampled_classes = np.random.choice(len(full_labels_loc), num_of_sampled_classes, replace=False)
sampled_classes = np.sort(sampled_classes)  

squared_distance_matrix = np.zeros((len(full_labels_loc), len(full_labels_loc)))

for i, each_class_loc_i in enumerate(full_labels_loc):
    for j, each_class_loc_j in enumerate(full_labels_loc):
        distance = length[each_class_loc_i][each_class_loc_j]
        squared_distance_matrix[i][j] = distance ** 2

## Load parameters from pretrained SimCLR ##
print("load parameters from pretrained SimCLR")
model = resnet50x1()
sd = torch.load(parameter_path, map_location='cpu')
model.load_state_dict(sd["state_dict"])
model = model.to('cuda:0')

## Extract image embeddings from SimCLR ##
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(train_path_to_imagenet, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])),
    batch_size=256, shuffle=True,
    num_workers=10, pin_memory=False)

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

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(val_path_to_imagenet, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])),
    batch_size=256, shuffle=True,
    num_workers=10, pin_memory=False)

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

print("save training np array and testing np array")
np.save("../saved/train_X_1000_classes", sampled_train_X)
np.save("../saved/train_y_1000_classes", sampled_train_y)
np.save("../saved/test_X_1000_classes", test_X)
np.save("../saved/test_y_1000_classes", test_y)

unique_elements, counts_elements = np.unique(sampled_train_y, return_counts=True)
print(unique_elements, counts_elements)
print(sampled_train_X.shape, sampled_train_y.shape, test_X.shape, test_y.shape)

print("train lg with one-vs-rest")
clf = OneVsRestClassifier(LogisticRegression(random_state=0, max_iter=500), n_jobs=10).fit(sampled_train_X, sampled_train_y)
print("save lg model")
pickle.dump(clf, open("../saved/lg_1000_classes_model.sav", 'wb'))

print("inference validation data through lg with one-vs-rest")
pred_prob = clf.predict_proba(test_X)

prediction = sampled_classes[np.argmax(pred_prob, axis=1)]

avg_squared_distance = 0
for pred, gt in zip(prediction, test_y):
    avg_squared_distance += squared_distance_matrix[pred][gt]
avg_squared_distance = avg_squared_distance / len(test_y)
print("SimCLR + LG, AVG Squared Distance: ", avg_squared_distance)

prediction_w_label_model = np.argmin(np.dot(pred_prob, squared_distance_matrix[sampled_classes]), axis=1)

avg_squared_distance_w_label_model = 0
for pred, gt in zip(prediction_w_label_model, test_y):
    avg_squared_distance_w_label_model += squared_distance_matrix[pred][gt]
avg_squared_distance_w_label_model = avg_squared_distance_w_label_model / len(test_y)
print("SimCLR + LG + Label Model, AVG Squared Distance: ", avg_squared_distance_w_label_model)

