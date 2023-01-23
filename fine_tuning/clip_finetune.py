'''
https://github.com/openai/CLIP/issues/83
'''
from utils import utils
from utils import config
from libs import ImgTextPairDataset, loki_loss

import torch
import clip
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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

#https://github.com/openai/CLIP/issues/57
# def convert_models_to_fp32(model): 
#     for p in model.parameters(): 
#         p.data = p.data.float() 
#         p.grad.data = p.grad.data.float() 

# if device == "cpu":
#     model.float()
# else:
#     clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16
  
def extract_label_text_features(model, label_text):
    label_features = []
    for label_t in label_text:
        texts = clip.tokenize(label_t).to(device)
        class_embeddings = model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        label_features.append(class_embedding)
    label_features = torch.stack(label_features, dim=1).to(device)
    return label_features

def get_logits(model, images, label_text):
    with torch.no_grad():
        image_features_all = model.encode_image(images)
    image_features_all /= image_features_all.norm(dim=0)
    image_features_all = image_features_all.to(device)
    image_features_all = image_features_all.squeeze()
    text_features_all = extract_label_text_features(model, label_text)
    logits = (100. * image_features_all @ text_features_all).softmax(dim=-1)
    return logits

k = len(trainset.classes)
class_to_idx = trainset.class_to_idx
label_text = ["a photo of a {}.".format(class_) for class_ in class_to_idx]
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

for epoch in tqdm(range(n_epoch)):
    running_loss = 0.
    running_corrects = 0
    for images, targets in tqdm(train_dataloader):
        optimizer.zero_grad()
        images = images.to(device)
        targets = targets.to(device)

        logits = get_logits(model, images, label_text)
        preds = logits.argmax(dim=1).detach().cpu()
        preds = loki_loss.smooth_hard_preds(preds, k=k, eps=0.001)
        
        logs = torch.log(preds).to(device)
        loss = loss_fn(logs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * targets.size(0)
        running_corrects += torch.sum(preds == targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects / len(train_dataset) * 100.
    print("Epoch {} Loss: {:.3f} Acc: {:.3f}".format(epoch, epoch_loss, epoch_acc))

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, f"model_checkpoint/model_{n_epoch}.pt") #just change to your preferred folder/filename
print('model saved to',  f"model_checkpoint/model_{n_epoch}.pt")