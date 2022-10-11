import networkx as nx
import numpy as np
from tqdm import tqdm

class TreeMetrics:
    def __init__(self):
        pass

    def compute_sq_dist_matrix(self, T, labels):
        d = np.zeros((len(labels), len(labels)))
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                d[i, j] = nx.shortest_path_length(T, labels[i], labels[j])
                d[j, i] = d[i, j]
        return np.square(d)

    def interpolate_to_available_key(self, class_, class_to_idx):
        if " " in class_:
            class_split = class_.split(" ")
            for c in class_split:
                if c in class_to_idx:
                    return c
            class_ = "_".join(class_.split(" "))
        if "-" in class_:
            class_ = class_.replace("-", "_")
        if class_[-3:]== "ies":
            class_ = class_[:-3]
            class_ += 'y'
        elif class_[-1] == 's':
            class_ = class_[:-1]
        if class_ not in class_to_idx:
            class_ += "_tree"
        return class_
        
    def preprocess_class_name_list_to_class_idx(self, class_to_idx, subclasses):
        subclasses_ids = []
        for class_ in subclasses:
            if class_ not in class_to_idx:
                class_ = self.interpolate_to_available_key(class_, class_to_idx)
            subclasses_ids.append(class_to_idx[class_])
        return subclasses_ids
    
    def preprocess_class_name_to_class_idx(self, class_to_idx, subclasses):
        subclasses_ids = []
        for class_ in subclasses:
            if class_ not in class_to_idx:
                class_ = self.interpolate_to_available_key(class_, class_to_idx)
            subclasses_ids.append(class_to_idx[class_])
        return subclasses_ids

    def get_hierarchy_graph(self, class_to_idx):
        T = nx.Graph()
        T.add_node('root')
        labels = []
        with open('../cifar100/cifar100_hierarchy.txt', 'r') as f:
            for line in f.readlines():
                splitline = line.split('\t')
                superclass = splitline[0]
                subclasses = splitline[1].strip().split(', ')
                subclasses = self.preprocess_class_name_list_to_class_idx(class_to_idx, subclasses)
                labels += subclasses
                T.add_node(superclass)
                T.add_edge('root', superclass)
                T.add_nodes_from(subclasses)
                T.add_edges_from([(subclass, superclass) for subclass in subclasses])
        labels = np.array(labels)
        return T, labels
    
    def get_parent_child_graph(self):
        T = nx.Graph()
        labels = []
        with open('../cifar_parent_child.txt', 'r') as f:
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
    
    def calc_tree_metric(self, T, y_pred, y_true):
        dists = []
        for pred, gt in zip(y_pred, y_true):
            dists.append(nx.shortest_path_length(T, pred, gt))
        dists = np.square(np.array(dists))
        expected_dists = np.mean(dists)
        return expected_dists
        
