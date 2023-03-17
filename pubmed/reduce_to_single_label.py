import pandas as pd
import numpy as np

from tqdm import tqdm
import json

def count_label_frequency(raw_labels, unique_labels):
    print('counting labels frequency...')
    label_freq = {l_: 0 for l_ in unique_labels}
    for multi_label in tqdm(raw_labels[:1000]):
        multi_label = multi_label.replace("'", "")
        labels_list = multi_label.strip('][').split(', ')
        for l_ in labels_list:
            l_id = class_mapping[l_]
            label_freq[l_id] += 1
    return label_freq

def reduce_label(raw_labels, class_mapping):
    print("reducing labels...")
    reduced_label = []
    reduced_label_id = []
    for label_ in tqdm(raw_labels):
        label_ = label_.replace("'", "")
        labels_list = label_.strip('][').split(', ')
        random_label = np.random.choice(labels_list)
        reduced_label.append(random_label)
        reduced_label_id.append(class_mapping[random_label])
    return reduced_label, reduced_label_id


if __name__ == '__main__':
    df = pd.read_csv('PubMed Multi Label Text Classification Dataset Processed.csv')
    df = df.dropna()
    labels_raw = df['meshMajor'].tolist()

    f = open('class_mapping.json')
    class_mapping = json.loads(json.load(f))

    labels_unique = list(class_mapping.values())
    
    labels_freq = count_label_frequency(labels_raw, labels_unique)
    jsonstr = json.dumps(labels_freq)
    with open('class_freq.json', 'w') as file_object:
        json.dump(jsonstr, file_object) 
    print(len(class_mapping))
    reduced_label, reduced_label_id = reduce_label(labels_raw, class_mapping)
    print('unique labels', len(np.unique(reduced_label_id)))
    df['singleLabel'] = reduced_label
    df['singleLabelId'] = reduced_label_id
    df = df[['Title', 'abstractText', 'meshMajor', 'singleLabel', 'singleLabelId']]
    print(df)
    df.to_csv('pubmed_reduced_label.csv', index=False)

