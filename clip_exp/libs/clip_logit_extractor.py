'''
https://github.com/openai/CLIP#zero-shot-prediction
'''
from utils import utils
import torch
import clip
from tqdm import tqdm
import torchvision.transforms as transforms

device = utils.device

class CLIPLogitExtractor:
    def __init__(self,):
        self.model, self.preprocess = clip.load('RN50', device)
    
    def extract_label_text_features(self, label_text):
        zeroshot_weights = []
        for label_t in tqdm(label_text):
            texts = clip.tokenize(label_t).to(device)
            class_embeddings = self.model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
        return zeroshot_weights

    def get_logits(self, dataset, label_text=None, text_features_all=None, stop_idx=None):
        image_features_all = []
        y_true_all = []
        print("Extracting image features...")
        for idx, (image, y_true) in tqdm(enumerate(dataset)):
            if stop_idx != None:
                if idx >= stop_idx:
                    break
            if torch.is_tensor(image):
                image = image[0]
                t = transforms.ToPILImage()
                image = t(image)
            image_input = self.preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_feature = self.model.encode_image(image_input)
            image_feature /= image_feature.norm()
            image_features_all.append(image_feature)
            y_true_all.append(y_true)
        image_features_all = torch.stack(image_features_all, dim=1).to(device)
        image_features_all = image_features_all.squeeze()
        if label_text != None:
            text_features_all = self.extract_label_text_features(label_text)
        logits = (100. * image_features_all @ text_features_all).softmax(dim=-1).detach().cpu()
        torch.save(logits, 'logits.pt')
        torch.save(torch.Tensor(y_true_all), 'y.pt')
        return logits, y_true_all
    
    def get_preds(self, logits):
        return torch.argmax(logits, dim=1).detach().cpu().numpy()