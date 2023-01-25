import networkx as nx
import numpy as np
from tqdm import tqdm
from torchvision.datasets import CIFAR100
import os

class CIFARTreeParser:
    def __init__(self):
        self.class_to_idx = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False).class_to_idx
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.labels_id = list(idx_to_class.keys())

    def compute_dist_matrix(self, T, labels):
        d = np.zeros((len(labels), len(labels)))
        length = dict(nx.all_pairs_shortest_path_length(T))
        for i in range(len(labels)):
            for j in range(len(labels)):
                distance = length[i][j]
                d[i, j] = distance
        return d
        
    def preprocess_class_name_list_to_class_idx(self, subclasses):
        class_to_idx = self.class_to_idx
        subclasses_ids = []
        subclasses_names_corrected = []
        for class_ in subclasses:
            if class_ not in class_to_idx:
                class_ = self.interpolate_to_available_key(class_, class_to_idx)
            subclasses_ids.append(class_to_idx[class_])
            subclasses_names_corrected.append(class_)
        return subclasses_ids, subclasses_names_corrected

    def get_hierarchy_tree(self):
        class_to_idx = self.class_to_idx
        T = nx.Graph()
        T.add_node('root')
        labels = []
        mapping = {}
        with open('../cifar100/cifar100_hierarchy.txt', 'r') as f:
            for line in f.readlines():
                splitline = line.split('\t')
                superclass = splitline[0]
                subclasses = splitline[1].strip().split(', ')
                subclasses_idx, subclasses_corrected = self.preprocess_class_name_list_to_class_idx(class_to_idx, subclasses)
                labels += subclasses_idx
                for i, sub_ in enumerate(subclasses_idx):
                    mapping[sub_] = subclasses_corrected[i]
                T.add_node(superclass)
                T.add_edge('root', superclass)
                T.add_nodes_from(subclasses_idx)
                T.add_edges_from([(subclass, superclass) for subclass in subclasses_idx])
        labels = np.array(labels)
        return T, labels, mapping
    
    def get_parent_child_tree(self):
        T = nx.Graph()
        labels = []
        with open('../cifar100/cifar_parent_child.txt', 'r') as f:
            for line in f.readlines():
                nodes = line.split()
                nodes = [int(node) for node in nodes]
                for node in nodes:
                    if node not in T:
                        T.add_node(node)
                T.add_edge(*nodes)
                    
        leaves = [x for x in T.nodes() if T.degree(x) == 1]
        labels = np.array(leaves)
        return T, labels
        
