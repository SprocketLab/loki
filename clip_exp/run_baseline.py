from utils import utils, cifar100_labels
import torch
import clip
import numpy as np
from tqdm import tqdm

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

if __name__ == '__main__':
    cifar100_test = utils.get_CIFAR100_test_set()
    model, preprocess = utils.load_clip_model()
    label_text = [f"a photo of a {y}." for y in cifar100_labels]
    logits, y_true = get_logits(cifar100_test, label_text)
    preds = get_preds(logits)
    error_rate_vanilla = (preds != y_true).mean()
    print(f"Error rate using complete graph: {error_rate_vanilla}")
    
