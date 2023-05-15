'''
https://github.com/openai/CLIP#zero-shot-prediction
'''
from utils import utils
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from transformers import AlignModel, AlignProcessor
# from transformers import AltCLIPModel, AltCLIPProcessor
import open_clip
from PIL import Image

device = utils.device

class ALIGNLogitExtractor:
    def __init__(self,):
        # self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion400m_e31')
        self.model = AlignModel.from_pretrained("kakaobrain/align-base")
        # AltCLIPModel.from_pretrained("BAAI/AltCLIP")
        # AlignModel.from_pretrained("kakaobrain/align-base")
        self.preprocess =AlignProcessor.from_pretrained("kakaobrain/align-base")
        # AltCLIPProcessor.from_pretrained("BAAI/AltCLIP")
        # AlignProcessor.from_pretrained("kakaobrain/align-base")
    
    def extract_label_text_features(self, label_text):
        zeroshot_weights = []
        for label_t in tqdm(label_text):
            text_embedding = self.model.encode_text(self.preprocess([label_t])).detach().cpu().numpy()
            # print(label_t)
            noun_entity_inputs = self.preprocess(text=label_t, images=torch.rand(3,48,48), return_tensors="pt", max_length=16, padding='max_length')
            clip_outputs = self.model(**noun_entity_inputs)
            class_embeddings = clip_outputs.text_embeds
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
            if len(image.size) !=3:
                image = image.convert('RGB')
            if stop_idx != None:
                if idx >= stop_idx:
                    break
            if torch.is_tensor(image):
                image = image[0]
                t = transforms.ToPILImage()
                image = t(image)
            noun_entity_inputs = self.preprocess(text=['a'], images=image, return_tensors="pt", max_length=1, padding='max_length')
            clip_outputs = self.model(**noun_entity_inputs)
            image_feature = clip_outputs.image_embeds
            # if idx == 10:
            #     break
            # print(image.shape)
            # image = self.preprocess(image).unsqueeze(0)
            # image_feature = self.model.encode_image(image)
            # image_feature /= image_feature.norm()
            image_features_all.append(image_feature)
            y_true_all.append(y_true)
        image_features_all = torch.stack(image_features_all, dim=1).to(device)
        image_features_all = image_features_all.squeeze()
        print(image_features_all.shape)
        torch.save(image_features_all, 'image_feats_ALIGN.pt')
        exit()
        if label_text != None:
            text_features_all = self.extract_label_text_features(label_text)
        logits = (100. * image_features_all @ text_features_all).softmax(dim=-1).detach().cpu()
        torch.save(logits, 'logits_align.pt')
        torch.save(torch.Tensor(y_true_all), 'y_align.pt')
        return logits, y_true_all
    
    def get_preds(self, logits):
        return torch.argmax(logits, dim=1).detach().cpu().numpy()