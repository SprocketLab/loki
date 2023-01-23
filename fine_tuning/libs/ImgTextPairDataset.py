from torch.utils.data import Dataset
import clip
import torch

from utils import config

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load(config.clip_model_str,device=device) #Must set jit=False for training

class ImgTextPairDataset(Dataset):
    def __init__(self, dataset):
        class_to_idx = dataset.class_to_idx
        label_text = ["a photo of a {}.".format(class_) for class_ in class_to_idx]
        self.dataset = dataset
        self.class_label_tokens = {class_to_idx[class_]: clip.tokenize(class_) for class_ in class_to_idx}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        image = preprocess(image).unsqueeze(0)
        text = self.class_label_tokens[target]
        return image, text