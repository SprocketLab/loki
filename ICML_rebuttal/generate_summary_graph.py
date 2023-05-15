import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import fire

from tqdm import tqdm
from itertools import combinations
from scipy.sparse import csr_matrix

def load_dataset(filename="train-remapped.csv", nmax=1_000_000_000_000):
    with open(filename, "r") as f:
        lines = f.readlines()

    class_set = set()
    labels = []
    features = []
    for l, line in tqdm(enumerate(lines), total=len(lines)-1):
        if l > nmax: break
        if l == 0: continue
        line = line.strip().split(" ")
        label = []
        feature = {}
        for element in line:
            if ":" not in element:
                element = int(element.replace(",", ""))
                class_set.add(element)
                label.append(element)
            else:
                feature_id = int(element.split(":")[0])
                feature_value = int(element.split(":")[1])
                feature[feature_id] = feature_value
        labels.append(label)
        features.append(feature)
    return class_set, features, labels

def filter_dataset(classes, X, Y, f):
    Xnew, Ynew = [], []
    for _x, _y in zip(X, Y):
        if f(_x, _y):
            Xnew.append(_x)
            Ynew.append(_y)
    classes_new = set([val for sublist in Ynew for val in sublist])
    return classes_new, Xnew, Ynew

def get_graph(classes, hierarchy_file="hierarchy.txt"):
    with open(hierarchy_file, "r") as f:
        lines = f.readlines()
    G = nx.Graph()
    for l, line in tqdm(enumerate(lines), total=len(lines)-1):
        a, b = line.split(' ')
        a = int(a.strip())
        b = int(b.strip())
        if a in classes or b in classes:
            if a not in G.nodes():
                G.add_node(a)
            if b not in G.nodes():
                G.add_node(b)
            G.add_edge(a, b)
    return G

def main(target_graph_size=10_000, basedir="./lshtc", savedir="./output"):
    classes, X, Y = load_dataset(f"{basedir}/train-remapped.csv")

    # Get the largest connected component in the graph... 
    G = get_graph(classes, f"{basedir}/hierarchy.txt")
    G_components = [G.subgraph(cc_G) for cc_G in nx.connected_components(G)]
    G_ours = G_components[np.argmax([len(G_c.nodes()) for G_c in G_components])] 
    len(G_ours.nodes())

    # Graph summarization
    og_graph_size = len(G_ours.nodes)
    G_summary = nx.Graph(G_ours)
    manifest = {node: {node} for node in G_summary.nodes}

    # Iterate over all nodes in the absolute worst possible case
    pbar = tqdm(total=og_graph_size - target_graph_size)
    for _ in range(og_graph_size - target_graph_size):
        rand_node = np.random.choice(G_summary.nodes)
        neighborhood = list(G_summary.neighbors(rand_node))
        for neighbor_node in neighborhood:
            # Merge nodes. Note that ordering matters -- rand_node will be kept
            nx.contracted_nodes(G_summary, rand_node, neighbor_node,
                                self_loops=False, copy=False)
            # Merge manifest entries
            # Loop invariant: the manifest should *always* be a partition of nodes
            manifest[rand_node] = manifest[rand_node] | manifest[neighbor_node]
            manifest.pop(neighbor_node, None)

            # Update progress
            current_graph_size = len(G_summary.nodes)
            pbar.set_postfix_str(current_graph_size)
            pbar.update(1)

            if current_graph_size <= target_graph_size:
                break # The graph is sufficiently summarized

        if current_graph_size <= target_graph_size:
            break # The graph is sufficiently summarized

    # Save G_summary and manifest
    nx.write_adjlist(G_summary, f"{savedir}/G_summary.adjlist")
    with open('./output/manifest.pkl', 'wb') as f:
        pickle.dump(manifest, f)

if __name__ == "__main__":
    fire.Fire(main)
