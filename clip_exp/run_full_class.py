from utils import utils
import numpy as np
from libs import TreeMetrics, CLIPLogitExtractor

device = utils.device

if __name__ == '__main__':
    cifar100_test = utils.get_CIFAR100_test_set()
    class_to_idx = cifar100_test.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    labels_id = list(idx_to_class.keys())

    tree_metric = TreeMetrics()
    T_pc, labels_id_pc = tree_metric.get_parent_child_graph()
    T_hierarchy, labels_id_hierarchy, mapping = tree_metric.get_hierarchy_graph(class_to_idx)
    print('checking cifar100 map vs hierarchy map equivalence', utils.check_mapping_equivalence(idx_to_class, mapping))
    label_text = [f"a photo of a {class_}." for class_ in class_to_idx]

    clip = CLIPLogitExtractor()
    logits, y_true = clip.get_logits(cifar100_test, label_text)
    preds = clip.get_preds(logits)

    error_rate_vanilla = (preds != y_true).mean()
    print(f"argmax prediction + complete graph dist: {error_rate_vanilla}")

    argmax_dist_pc_tree = tree_metric.calc_tree_metric(T_pc, preds, y_true)
    argmax_dist_hierarchy_tree = tree_metric.calc_tree_metric(T_hierarchy, preds, y_true)

    squared_distance_matrix = tree_metric.compute_sq_dist_matrix(T_hierarchy, labels_id)
    prediction_w_label_model = np.argmin(np.dot(logits, squared_distance_matrix), axis=1)
    tree_dist_hierarcy_tree = tree_metric.calc_tree_metric(T_hierarchy, prediction_w_label_model, y_true)
    print(f"argmax prediction + hierarchy tree dist: {argmax_dist_hierarchy_tree}")
    print(f"hierarchy tree prediction + hierarchy tree dist: {tree_dist_hierarcy_tree}")

    squared_distance_matrix = tree_metric.compute_sq_dist_matrix(T_pc, labels_id)
    prediction_w_label_model = np.argmin(np.dot(logits, squared_distance_matrix), axis=1)
    tree_dist_pc_tree = tree_metric.calc_tree_metric(T_pc, prediction_w_label_model, y_true)
    print(f"argmax prediction + pc tree dist: {argmax_dist_pc_tree}")
    print(f"pc tree prediction + pc tree dist: {tree_dist_pc_tree}")