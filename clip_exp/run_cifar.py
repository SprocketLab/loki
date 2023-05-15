from utils import utils
import numpy as np
from libs import TreeMetrics, ALIGNLogitExtractor
import os
import torch 
from tqdm import tqdm

device = utils.device

def calc_clip_sq_dist_matrix(labels_emb):
    labels_emb = labels_emb.detach().cpu().numpy()
    print('calculating clip dist matrix')
    dist_matrix = np.zeros((labels_emb.shape[1], labels_emb.shape[1]))
    visited = set()
    for i in tqdm(range(labels_emb.shape[1])):
        emb1 = labels_emb[:,i]
        for j in range(labels_emb.shape[1]):
            if i == j:
                continue
            if (i, j) in visited:
                continue
            emb2 = labels_emb[:,j]
            dist = np.square(np.linalg.norm(emb1-emb2, ord=2))
            dist_matrix[i, j] = dist
            dist_matrix[j,i] = dist
            visited.add((i, j))
            visited.add((j,i))
    return dist_matrix

def calc_clip_tree_metric(dist_matrix, y_pred, y_true):
    dists = []
    for pred, gt in zip(y_pred, y_true):
        pred = int(pred)
        gt = int(gt)
        dists.append(dist_matrix[pred, gt])
    expected_dists = np.mean(dists)
    return expected_dists


if __name__ == '__main__':
    cifar100_test = utils.get_CIFAR100_test_set()
    class_to_idx = cifar100_test.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    labels_id = list(idx_to_class.keys())

    tree_metric = TreeMetrics()
    T_pc, _ = tree_metric.get_parent_child_graph()
    T_hierarchy, _, mapping = tree_metric.get_hierarchy_graph(class_to_idx)
    T_random = tree_metric.get_random_graph(len(labels_id))
    print('checking cifar100 map vs hierarchy map equivalence', utils.check_mapping_equivalence(idx_to_class, mapping))
    label_text = ["a photo of a {}.".format(class_) for class_ in class_to_idx]
    
    # clip = CLIPLogitExtractor()
    clip = ALIGNLogitExtractor()
    # print('here')
    # exit()
    # labels_emb = clip.extract_label_text_features(label_text)
    # sq_clip_dist_matrix = calc_clip_sq_dist_matrix(labels_emb)

    if 'logits.pt' not in os.listdir('.'):
        logits, y_true = clip.get_logits(cifar100_test, label_text)
    else:
        logits = torch.load('logits.pt')
        y_true = torch.load('y.pt').detach().cpu().numpy().tolist()
    exit()
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

    argmax_dist_pc_tree = tree_metric.calc_tree_metric(T_pc, preds, y_true)
    argmax_dist_hierarchy_tree = tree_metric.calc_tree_metric(T_hierarchy, preds, y_true)
    argmax_dist_random_tree = tree_metric.calc_tree_metric(T_random, preds, y_true)

    squared_distance_matrix = tree_metric.compute_sq_dist_matrix(T_hierarchy, labels_id)
    prediction_w_label_model = np.argmin(np.dot(logits, squared_distance_matrix), axis=1)
    tree_dist_hierarcy_tree = tree_metric.calc_tree_metric(T_hierarchy, prediction_w_label_model, y_true)
    print("argmax prediction + hierarchy tree dist: {}".format(argmax_dist_hierarchy_tree))
    print("hierarchy tree prediction + hierarchy tree dist: {}".format(tree_dist_hierarcy_tree))
    print("error%: {}".format((prediction_w_label_model!=y_true).mean()))
    print("% relative improvement {}".format((tree_dist_hierarcy_tree - argmax_dist_hierarchy_tree)/argmax_dist_hierarchy_tree))
    print("")

    squared_distance_matrix = tree_metric.compute_sq_dist_matrix(T_pc, labels_id)
    prediction_w_label_model = np.argmin(np.dot(logits, squared_distance_matrix), axis=1)
    tree_dist_pc_tree = tree_metric.calc_tree_metric(T_pc, prediction_w_label_model, y_true)
    print("argmax prediction + pc tree dist: {}".format(argmax_dist_pc_tree))
    print("pc tree prediction + pc tree dist: {}".format(tree_dist_pc_tree))
    print("error%: {}".format((prediction_w_label_model!=y_true).mean()))
    print("% relative improvement {}".format((tree_dist_pc_tree - argmax_dist_pc_tree)/argmax_dist_pc_tree))
    print("")

    squared_distance_matrix = tree_metric.compute_sq_dist_matrix(T_random, labels_id)
    prediction_w_label_model = np.argmin(np.dot(logits, squared_distance_matrix), axis=1)
    tree_dist_random_tree = tree_metric.calc_tree_metric(T_random, prediction_w_label_model, y_true)
    print("argmax prediction + random tree dist: {}".format(argmax_dist_random_tree))
    print("random tree prediction + random tree dist: {}".format(tree_dist_random_tree))
    print("error%: {}".format((prediction_w_label_model!=y_true).mean()))
    print("% relative improvement {}".format((tree_dist_random_tree-argmax_dist_random_tree)/argmax_dist_random_tree))