'''
https://github.com/openai/CLIP/issues/83
'''
from utils import utils
from utils import config
from libs import ImgTextPairDataset, loki_loss, TreeMetrics

import torch
import clip
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

batch_size = config.batch_size
n_epoch = config.n_epoch
lr = config.lr
betas = config.betas
eps = config.eps
weight_decay = config.weight_decay

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load(config.clip_model_str, device=device, jit=False) #Must set jit=False for training

trainset = utils.get_CIFAR100_train_set()
train_dataset = ImgTextPairDataset(trainset, preprocess)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
  
testset = utils.get_CIFAR100_test_set()
test_dataset = ImgTextPairDataset(testset, preprocess)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

def extract_label_text_features(model, label_text):
    label_features = []
    for label_t in label_text:
        texts = clip.tokenize(label_t).to(device)
        class_embeddings = model.encode_text(texts)
        class_embeddings_normalized = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        embedding = class_embeddings_normalized.mean(dim=0)
        embedding_norm = embedding / embedding.norm()
        label_features.append(embedding_norm)
    label_features = torch.stack(label_features, dim=1).to(device)
    return label_features

def get_logits(model, images, label_text):
    image_features_all = []
    for image in images:
        with torch.no_grad():
            image_feature_ = model.encode_image(image)
        image_feature = image_feature_ / image_feature_.norm()
        image_features_all.append(image_feature)
    image_features_all = torch.stack(image_features_all, dim=1).to(device)
    image_features_all = image_features_all.squeeze()
    text_features_all = extract_label_text_features(model, label_text)
    logits = (100. * image_features_all @ text_features_all)
    return logits.float()
    
def evaluate(epoch, model):
    model.eval()
    class_to_idx = testset.class_to_idx
    label_text = ["a photo of a {}.".format(class_) for class_ in class_to_idx]
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for images, targets in tqdm(test_dataloader):
            images = images.to(device)
            targets_all.extend(targets.detach().cpu())
            logits = get_logits(model, images, label_text).softmax(dim=-1)
            preds = np.argmin(np.dot(logits.detach().cpu().numpy(), sq_distances.detach().cpu().numpy()), axis=1)
            preds_all.extend(preds.tolist())
    preds_all = np.asarray(preds_all)
    targets_all = np.asarray(targets_all)
    tree_dist = tree_metric.calc_tree_metric(T_hierarchy, preds_all, targets_all)
    acc = (preds_all == targets_all).mean()
    print("Epoch {} eval: tree dist = {:.3f} | acc = {:.3f}".format(epoch, tree_dist, acc))
            

k = len(trainset.classes)
class_to_idx = trainset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
labels_id = list(idx_to_class.keys())
label_text = ["a photo of a {}.".format(class_) for class_ in class_to_idx]

tree_metric = TreeMetrics()
T_hierarchy, _, _ = tree_metric.get_hierarchy_graph(class_to_idx)
sq_distances = torch.FloatTensor(tree_metric.compute_dist_matrix(T_hierarchy, labels_id)).to(device)
distances = torch.sqrt(sq_distances)
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

#eval train (1 loop)
for epoch in tqdm(range(n_epoch)):
    evaluate(epoch, model)
    running_loss = 0.
    model.train()
    for images, targets in tqdm(train_dataloader):
        optimizer.zero_grad()
        images = images.to(device)
        targets = targets.to(device)

        logits = get_logits(model, images, label_text)
        probs = logits.softmax(dim=1).float()
        preds = loki_loss.loki_polytope_predict(probs, distances)
        
        # Apply "label smoothing" to the hard predictions
        eps = 0.001
        with torch.no_grad():
            smooth_onehot = (torch.ones(k, k) * (1 / (k - 1)) * eps) + (torch.eye(k) * ((1 - eps) - (1 / (k - 1)) * eps))
        smooth_onehot = smooth_onehot.to(device)
        preds_smoothed = preds @ smooth_onehot

        logs = torch.log(preds_smoothed)
        loss = loss_fn(logs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print("Epoch {} train loss: {:.3f}".format(epoch+1, epoch_loss))

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, f"model_checkpoint/model_{n_epoch}.pt") #just change to your preferred folder/filename
print('model saved to',  f"model_checkpoint/model_{n_epoch}.pt")