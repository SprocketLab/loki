import sys
import pickle
import numpy as np
import networkx as nx

from tqdm import tqdm
from itertools import combinations
from scipy.sparse import csr_matrix, load_npz
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import json

from sklearn.model_selection import train_test_split

## load the saved files ##
d = np.load('features/dist_matrix_euc.npy')
## not squared yet ##
d = d ** 2
X_sparse = np.asarray(np.load('features/title.npy'))
# print(X_sparse)
f = open('class_mapping.json')
class_mapping = json.loads(json.load(f))
df = pd.read_csv('pubmed_reduced_label.csv')
Y_sparse = np.asarray(df['singleLabelId'].tolist())
print('total unique classes',len(np.unique(Y_sparse)))

print(X_sparse.shape, Y_sparse.shape)
inv_map_index_classes = {v: k for k, v in class_mapping.items()}
    
f = open('class_freq.json')
sorted_freq = json.loads(json.load(f))
sorted_freq = {int(k):int(v) for k,v in sorted_freq.items()}
sorted_freq = {k: v for k, v in sorted(sorted_freq.items(), key=lambda item: item[1], reverse=True)}
# for i, key in enumerate(sorted_freq):
#     if i >750:
#         break
#     print(key, sorted_freq[key])
# # print(sorted_freq[:1000])
# exit()
class_more_than_1 = {k: v for k, v in sorted_freq.items() if v > 2}
# print('n of classes appear more than once', len(class_more_than_1))
# exit()
# number of observed classes ##
k = int(sys.argv[1])

## get the indices the the k most frequent classes ##
topk_classes = []
for c in list(sorted_freq.keys()):
    if len(topk_classes) == k: break
    if c not in list(class_mapping.values()):
        continue
    else:
        topk_classes.append(c)
print(len(topk_classes))

inds = []
for i in range(Y_sparse.shape[0]):
    if Y_sparse[i] in topk_classes:
        inds.append(i)
inds = np.array(inds)
X = X_sparse[inds]
y = Y_sparse[inds]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)


print("Hello", X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

## train the knn model ##
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, Y_train)
print('X train', X_train.shape, 'Y train', Y_train.shape, len(np.unique(Y_train)))
## baseline v.s. proposed labeled model ##
batch_size = 5000
avg_squared_distance = 0
avg_squared_distance_w_label_model = 0

sampled_classes_index = topk_classes
print(len(sampled_classes_index))

count = 0
for i in tqdm(range(0, Y_test.shape[0], batch_size), total=int(Y_test.shape[0] / batch_size) + 1):
    print('X test', X_test[i : i + batch_size].shape)
    pred_prob = neigh.predict_proba(X_test[i : i + batch_size])
    prediction = np.argmax(pred_prob, axis=1)
    print(pred_prob.shape, d[sampled_classes_index].shape)
    prediction_w_label_model = np.argmin(np.dot(pred_prob, d[sampled_classes_index]), axis=1)
    
    for pred, pred_w_label_model, gt in zip(prediction, prediction_w_label_model, Y_test[i : i + batch_size]):
        avg_squared_distance += d[pred][gt]
        avg_squared_distance_w_label_model += d[pred_w_label_model][gt]
    
    count += pred_prob.shape[0]
    # print(count)
    # print("BASELINE: ", avg_squared_distance / count)
    # print("PROPOSED: ", avg_squared_distance_w_label_model / count)

avg_squared_distance = avg_squared_distance / Y_test.shape[0]
avg_squared_distance_w_label_model = avg_squared_distance_w_label_model / Y_test.shape[0]

print("topk: ", k)
print("5-nn, AVG Squared Distance: ", avg_squared_distance)
print("5-nn + Label Model, AVG Squared Distance: ", avg_squared_distance_w_label_model)
