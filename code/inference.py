import os
from tqdm import tqdm
import numpy as np
import json
import argparse
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

from utils import AudioData, AudioDataLoader, MusicClassifier, CompleteModel

from transformers import Wav2Vec2FeatureExtractor, AutoModel

def load_data(args):
    labels_path =  os.path.join(args.folder_path, 'test_labels.json')
    with open(labels_path, 'r') as f:
        label_dict = json.load(f)

    data_path = os.path.join(args.folder_path, 'test')
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


def inference(args, model, processor):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model.to(device)
    model.eval()

    test_folder = os.path.join(args.folder_path, 'test')
    test_files = os.listdir(test_folder)
    label_path = os.path.join(args.folder_path, 'test_labels.json')
    with open(label_path, 'r') as f:
        label_file = json.load(f)
    preds = []
    prob_preds = []
    test_labels = []
    pred_tracks = {file.split('_')[0]: [] for file in test_files}
    for file in tqdm(test_files):
        if file[-4:] != '.npy':
            continue
        file_path = os.path.join(test_folder, file)
        data = np.load(file_path)
        data = torch.tensor(data).unsqueeze(0)
        label = torch.tensor(label_file[file]).float()
        input = processor(data, sampling_rate=24000, return_tensors="pt")
        input = input.input_values.squeeze(0).to(device)

        output = model(input)
        prob_pred = F.sigmoid(output)
        pred = torch.where(prob_pred >= args.threshold, torch.tensor(1.0), torch.tensor(0.0)).cpu().detach().numpy()
        prob_pred = prob_pred.cpu().detach().numpy()
        preds.append(pred)
        prob_preds.append(prob_pred)
        test_labels.append(label)
        pred_tracks[file.split('_')[0]].append([pred.flatten().tolist()] * 5)

    preds = np.concatenate(preds, axis=0)
    prob_preds = np.concatenate(prob_preds, axis=0)
    return preds, prob_preds, test_labels, pred_tracks

def accuracy(preds, labels):
    class_report = classification_report(labels, preds)
    print(class_report)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default='./slakh')
    parser.add_argument('--model_name', type=str, default="m-a-p/MERT-v1-95M")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--fine_tune', type=int, default=1)
    parser.add_argument('--balance_data', type=int, default=1)
    args = parser.parse_args()
    
    if args.fine_tune and args.balance_data:
        suffix = '_fine_tune' + '_balance'
    elif args.fine_tune:
        suffix = '_fine_tune'
    elif args.balance_data:
        suffix = '_balance'
    else:
        suffix = ''

    feature_extractor = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name, trust_remote_code=True)
    classifier = MusicClassifier()
    model = CompleteModel(feature_extractor, classifier)
    model.load_state_dict(torch.load(f'./checkpoint/model_best{suffix}.pth'))
    preds, prob_preds, test_labels, pred_tracks = inference(args, model, processor)

    accuracy(preds, test_labels)

    if not os.path.exists('./result'):
        os.makedirs('./result')
    
    np.save(f'./result/preds{suffix}.npy', preds)
    np.save(f'./result/prob_preds{suffix}.npy', prob_preds)
    np.save(f'./result/test_labels{suffix}.npy', test_labels)
    with open(f'./result/pred_tracks{suffix}.json', 'w') as f:
        json.dump(pred_tracks, f)

if __name__ == '__main__':
    main()
