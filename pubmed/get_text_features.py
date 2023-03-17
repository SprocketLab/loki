from simcse import SimCSE
import pandas as pd
from tqdm import tqdm
import os
import torch
import numpy as np
import json

class TextFeatureExtractor:
    def __init__(self, data_path, bs=512, col_to_extract = 'Title'):
        self.model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
        self.df = pd.read_csv(data_path).dropna()
        self.col_to_extract = col_to_extract
        self.bs = bs

    def batch_texts(self,texts):
        print("batching texts...")
        texts_batched = []
        for i in tqdm(range(0, len(texts), self.bs)):
            texts_batched.append(texts[i:i+self.bs])
        return texts_batched

    def extract_features(self, texts):
        texts_batched = self.batch_texts(texts)
        features = []
        for batch in tqdm(texts_batched):
            try:
                feats = self.model.encode(batch)
                features.extend(feats.detach().cpu().numpy()) 
            except Exception as e:
                raise e
        features = np.asarray(features)
        return features
    
    def forward(self):
        texts = self.df[self.col_to_extract].tolist()
        features = self.extract_features(texts)
        print(f'features extracted, dim = {features.shape}')
        return features

class LabelFeatureExtractor(TextFeatureExtractor):
    def __init__(self, data_path, label_col):
        super().__init__(data_path)
        df = pd.read_csv(data_path).dropna()
        labels = df[label_col].tolist()
        self.labels = self.get_indiv_labels(labels)
        class_mapping = {label:i for i,label in enumerate(self.labels)}
        jsonstr = json.dumps(class_mapping)
        print('Class map')
        print(jsonstr)
        with open('class_mapping.json', 'w') as file_object:
            json.dump(jsonstr, file_object) 


    def get_indiv_labels(self, raw_labels):
        unique_labels = set()
        for label_ in raw_labels:
            label_ = label_.replace("'", "")
            labels_list = label_.strip('][').split(', ')
            unique_labels.update(labels_list)
        return list(unique_labels)
    
    def forward(self):
        return super().extract_features(self.labels)


def extract_label_features():
    data_path = 'PubMed Multi Label Text Classification Dataset Processed.csv'
    label_col = 'meshMajor'
    extractor = LabelFeatureExtractor(data_path, label_col)
    label_features = extractor.forward()
    np.save(os.path.join(store_dir, 'label.npy'), label_features)

def extract_data_features():
    data_path = 'PubMed Multi Label Text Classification Dataset Processed.csv'
    col_to_extract = 'abstractText'
    extractor = TextFeatureExtractor(data_path, col_to_extract=col_to_extract)
    text_features = extractor.forward()
    np.save(os.path.join(store_dir, f'{col_to_extract.lower()}.npy'), text_features)

if __name__ == '__main__':
    store_dir = 'features'
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)
    extract_label_features()
    
