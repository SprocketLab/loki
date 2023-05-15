import os
import json

import numpy as np
from tqdm import tqdm
import networkx as nx

import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from libs import TreeMetrics, CLIPLogitExtractor
from run_cifar import calc_clip_sq_dist_matrix, calc_clip_tree_metric


tree_structure = "mintree"
num_of_validation = 50000

train_path_to_imagenet = "/hdd2/datasets/imagenet/train/"

T = nx.Graph()

with open('/hdd4/brian/LabelSubspaces/simclr-converter/imagenet_' + tree_structure + '.txt', 'r') as f:
    for line in tqdm(f.readlines()):
        nodes = line.split()
        for node in nodes:
            if node not in T:
                T.add_node(node)
        T.add_edge(*nodes)
        
leaves = [x for x in T.nodes() if T.degree(x) == 1]
full_labels_loc = np.array(leaves)

f = open('/hdd4/brian/LabelSubspaces/simclr-converter/dir_label_name.json')
map_collection = json.load(f)

np.random.seed(123)
torch.manual_seed(123)

final_dataset = torch.utils.data.DataLoader(
    datasets.ImageFolder(train_path_to_imagenet, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])),
    batch_size=1, shuffle=True,
    num_workers=5, pin_memory=False)

clip = CLIPLogitExtractor()
label_text = [map_collection[str(key)][1] for key in map_collection]

def process_labels(label_text):
    processed_labels = []
    for text in label_text:
        text = text.lower()
        text = text.replace('_', ' ')
        processed_labels.append(text)
    return processed_labels

label_text = process_labels(label_text)
print(label_text)

print('getting label embeddings...')
labels_emb = clip.extract_label_text_features(label_text)
if 'logits.pt' not in os.listdir('.'):
    logits, y_true = clip.get_logits(final_dataset, text_features_all=labels_emb, stop_idx=num_of_validation)
else:
    logits = torch.load('logits.pt')
    y_true = torch.load('y.pt').detach().cpu().numpy().tolist()

sq_clip_dist_matrix = calc_clip_sq_dist_matrix(labels_emb)
preds = clip.get_preds(logits)

error_rate_vanilla = (preds != y_true).mean()
print("argmax prediction + complete graph dist: {}".format(error_rate_vanilla))
print("")

prediction_w_label_model = np.argmin(np.dot(logits, sq_clip_dist_matrix), axis=1)
tree_dist_CLIP = calc_clip_tree_metric(sq_clip_dist_matrix, prediction_w_label_model, y_true)
argmax_dist_CLIP = calc_clip_tree_metric(sq_clip_dist_matrix, preds, y_true)
print("argmax prediction + CLIP dist: {}".format(argmax_dist_CLIP))
print("CLIP dist prediction + CLIP dist: {}".format(tree_dist_CLIP))
print("error%: {}".format((prediction_w_label_model!=y_true).mean()))
print("% relative improvement {}".format((tree_dist_CLIP - argmax_dist_CLIP)/argmax_dist_CLIP))
print("")