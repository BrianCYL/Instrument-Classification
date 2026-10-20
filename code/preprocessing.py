import numpy as np
import os
import json
from tqdm import tqdm

def load_data(dataset):
    labels_path = f'slakh/{dataset}_labels.json'
    with open(labels_path, 'r') as f:
        label_dict = json.load(f)

    data_path = f'slakh/{dataset}'
    dataset = []
    labels = []
    for file in tqdm(os.listdir(data_path)):
        if file.endswith('.npy'):
            path = os.path.join(data_path, file)
            data = np.load(path)
            dataset.append(data)
            label = label_dict[file]
            labels.append(label)
    dataset = np.array(dataset)
    labels = np.array(labels)
    return dataset, labels

def main():
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    for dataset in ['train', 'validation', 'test']:
        print(f'Processing {dataset} dataset ... ')
        data, labels = load_data(dataset)
        print(f'Saving {dataset} dataset with {data.shape[0]} samples ... ')
        np.save(f'dataset/{dataset}_data.npy', data)
        np.save(f'dataset/{dataset}_labels.npy', labels)

if __name__ == '__main__':
    main()