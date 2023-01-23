import os
import json
import random
import pickle
import warnings
import numpy as np
from tqdm import tqdm
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from resnet_wider import resnet50x1, resnet50x2, resnet50x4

print("==============================================")
tree_structure = "mintree"
per_class = 50
SEED = 123

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

## Compute distance matrix ##
print("compute distance matrix")
squared_distance_matrix = np.zeros((len(full_labels_loc), len(full_labels_loc)))

for i, each_class_loc_i in enumerate(full_labels_loc):
    for j, each_class_loc_j in enumerate(full_labels_loc):
        distance = length[each_class_loc_i][each_class_loc_j]
        squared_distance_matrix[i][j] = distance ** 2
        
f = open('../dir_label_name.json')
map_collection = json.load(f)
f.close()

## Load training and testing data ##
print("load training and testing data")
train_X = np.load("../saved/train_X_1000_classes.npy")
train_y = np.load("../saved/train_y_1000_classes.npy")
test_X = np.load("../saved/test_X_1000_classes.npy")
test_y = np.load("../saved/test_y_1000_classes.npy")
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

## Load parameters from pretrained SimCLR ##
print("load saved lg model")
loaded_model = pickle.load(open("../saved/simclr1x_lg_1000_classes_model.sav", 'rb'))

classes_ = [i for i in range(50, 1000, 100)]
classes_.insert(0, 10)
classes_.insert(6, 500)
classes_.append(1000)

np.random.seed(SEED)
for num_of_sampled_classes in classes_:
    print(num_of_sampled_classes)
    
    ## Exponential Sampler ##
    print("exponential sampler")
    center_node_index = np.argmin(np.sum(squared_distance_matrix, axis=1))

    possibility = []
    variance = 0.5
    for node_index in range(len(full_labels_loc)):
        dist = squared_distance_matrix[node_index][center_node_index]
        possibility.append(math.exp(-1 * variance * dist))
    possibility = np.array(possibility) / sum(possibility)

    sampled_classes = np.random.choice(len(full_labels_loc), num_of_around_center, replace=False, p=possibility)
    distance_dict = {}
    for i in range(len(full_labels_loc)):
        for j in range(i + 1, len(full_labels_loc)):
            distance_dict[(i, j)] = squared_distance_matrix[i][j]
    sorted_distance_dict = dict(sorted(distance_dict.items(), key=lambda item: -item[1]))

    for key in sorted_distance_dict:
        if key[0] not in sampled_classes:
            if len(sampled_classes) + 1 <= num_of_sampled_classes:
                sampled_classes = np.append(sampled_classes, key[0])
            else:
                break
        if key[1] not in sampled_classes:
            if len(sampled_classes) + 1 <= num_of_sampled_classes:
                sampled_classes = np.append(sampled_classes, key[1])
            else:
                break
    sampled_classes = np.sort(sampled_classes)
    print(sampled_classes)
    
    pred_prob = []
    for c in tqdm(sampled_classes, total=len(sampled_classes)):
        prob = loaded_model.estimators_[c].predict_proba(test_X)[:, 1]
        pred_prob.append(prob)
    pred_prob = np.array(pred_prob).T
    print(pred_prob.shape)
    
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
    