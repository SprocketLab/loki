from utils import utils, cifar100_labels
import torch
import clip
import numpy as np
from tqdm import tqdm
from libs import TreeMetrics

device = utils.device

def extract_label_text_features(label_text):
    zeroshot_weights = []
    for label_t in label_text:
        texts = clip.tokenize(label_t).to(device)
        class_embeddings = model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

def get_logits(dataset, label_text):
    image_features_all = []
    y_true_all = []
    print("Extracting image features...")
    for image, y_true in tqdm(dataset):
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feature = model.encode_image(image_input)
        image_feature /= image_feature.norm()
        image_features_all.append(image_feature)
        y_true_all.append(y_true)
    image_features_all = torch.stack(image_features_all, dim=1).to(device)
    image_features_all = image_features_all.squeeze()
    text_features_all = extract_label_text_features(label_text)
    logits = (100. * image_features_all @ text_features_all).softmax(dim=-1).detach().cpu()
    return logits, y_true_all

def get_preds(logits):
    return torch.argmax(logits, dim=1).detach().cpu().numpy()

def process_label(label_text):
    if "_" not in label_text:
        return label_text
    else:
        return " ".join(label_text.split("_"))

if __name__ == '__main__':
    cifar100_test = utils.get_CIFAR100_test_set()
    class_to_idx = cifar100_test.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    model, preprocess = utils.load_clip_model()

    tree_metric = TreeMetrics()
    T_pc, labels_id_pc = tree_metric.get_parent_child_graph()
    T_hierarchy, labels_id_hierarchy = tree_metric.get_hierarchy_graph(class_to_idx)
    label_text_pc = [f"a photo of a {process_label(idx_to_class[int(label_id)])}." for label_id in labels_id_pc]
    label_text_hierarchy = [f"a photo of a {process_label(idx_to_class[int(label_id)])}." for label_id in labels_id_hierarchy]

    logits_pc, y_true = get_logits(cifar100_test, label_text_pc)
    preds_pc = get_preds(logits_pc)
    logits_hierarchy, y_true = get_logits(cifar100_test, label_text_hierarchy)
    preds_hierarchy = get_preds(logits_hierarchy)

    error_rate_vanilla_1 = (preds_pc != y_true).mean()
    print(f"argmax prediction + complete graph dist: {error_rate_vanilla_1}")

    argmax_dist_pc_tree = tree_metric.calc_tree_metric(T_pc, preds_pc, y_true)
    argmax_dist_hierarchy_tree = tree_metric.calc_tree_metric(T_hierarchy, preds_hierarchy, y_true)

    squared_distance_matrix = tree_metric.compute_sq_dist_matrix(T_hierarchy, labels_id_hierarchy)
    prediction_w_label_model = np.argmin(np.dot(logits_hierarchy, squared_distance_matrix), axis=1)
    tree_dist_hierarcy_tree = tree_metric.calc_tree_metric(T_hierarchy, prediction_w_label_model, y_true)
    print(f"argmax prediction + hierarchy tree dist: {argmax_dist_hierarchy_tree}")
    print(f"hierarchy tree prediction + hierarchy tree dist: {tree_dist_hierarcy_tree}")


    squared_distance_matrix = tree_metric.compute_sq_dist_matrix(T_pc, labels_id_pc)
    prediction_w_label_model = np.argmin(np.dot(logits_pc, squared_distance_matrix), axis=1)
    tree_dist_pc_tree = tree_metric.calc_tree_metric(T_pc, prediction_w_label_model, y_true)
    print(f"argmax prediction + pc tree dist: {argmax_dist_pc_tree}")
    print(f"pc tree prediction + pc graph dist: {tree_dist_pc_tree}")