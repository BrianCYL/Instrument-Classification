import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.functional as F
import torchaudio   
from torch.utils.data import Dataset, DataLoader

from transformers import Wav2Vec2FeatureExtractor, AutoModel

import argparse
from tqdm import tqdm
from utils import AudioData, AudioDataLoader, MusicClassifier, CompleteModel


def train(model, processor, optimizer, criterion, train_dataloader, valid_dataloader, args):
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        print('Using GPU')
    else:
        DEVICE = 'cpu'
        print('Using CPU')
    
    best_loss = float('inf')  
    best_epoch = 0
    train_loss = 0
    valid_loss = 0

    if args.tracking:
        total_train_loss = []
        total_valid_loss = []
    
    if args.fine_tune and args.balance_data:
        suffix = '_fine_tune' + '_balance'
    elif args.fine_tune:
        suffix = '_fine_tune'
    elif args.balance_data:
        suffix = '_balance'
    else:
        suffix = ''

    # Start Training!!!
    model.to(DEVICE)
    for epoch in tqdm(range(args.n_epochs)):
        model.train()
        for i, (data, label) in enumerate(train_dataloader):
            label = label.float().to(DEVICE)
            inputs = processor(data, sampling_rate=24000, return_tensors="pt") # (batch_size, 374, 1024)
            input = inputs.input_values.squeeze(0).to(DEVICE)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_dataloader)
        total_train_loss.append(train_loss)

        model.eval()
        for i, (data, label) in enumerate(valid_dataloader):
            inputs = processor(data, sampling_rate=24000, return_tensors="pt") # (batch_size, 374, 1024)
            input = inputs.input_values.squeeze(0).to(DEVICE)
            label = label.float().to(DEVICE)
            output = model(input)
            loss = criterion(output, label)
            valid_loss += loss.item()
        valid_loss /= len(valid_dataloader)
        total_valid_loss.append(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f'./checkpoint/model_best{suffix}.pth')

        if epoch % 5 == 0 and args.verbose:
            print('\n' + f'Epoch {epoch + 1}, Validation Loss: {valid_loss}')
            
        if epoch - best_epoch > args.early_stop:
            print('Early stopping!')
            break

    if args.tracking:
        plt.figure(figsize=(10, 8))
        plt.plot(total_train_loss, label='train_loss')
        plt.plot(total_valid_loss, label='valid_loss')
        plt.legend()
        plt.savefig(f'loss_tracking{suffix}.pdf')            
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="m-a-p/MERT-v1-95M")
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--fine_tune', type=int, default=1)
    parser.add_argument('--balance_data', type=int, default=1)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--folder_path', type=str, default='./dataset')
    parser.add_argument('--tracking', type=bool, default=True)
    args = parser.parse_args()
    
    if not os.path.exists('./checkpoint'):
        os.makedirs('./checkpoint')

    feature_extractor = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name, trust_remote_code=True)
    classifier = MusicClassifier()
    model = CompleteModel(feature_extractor, classifier)

    if args.fine_tune:
        print('Fine-tuning the whole model')
        for param in model.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        print('Freezing the feature extractor')
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    train_dataset = AudioData(args.folder_path, 'train')
    train_dataset.load_audio()
    train_dataloader = AudioDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset = AudioData(args.folder_path, 'validation')
    valid_dataset.load_audio()
    valid_dataloader = AudioDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    if args.balance_data:
        all_labels = []
        for _, labels in train_dataloader:
            all_labels.append(labels)

        # Concatenate the list of multi-hot label tensors into a single tensor
        all_labels = torch.cat(all_labels, dim=0)
        pos_freq = all_labels.sum(dim=0)
        neg_freq = len(all_labels) - pos_freq
        pos_weight = neg_freq / pos_freq
        pos_weight = pos_weight.to('cuda') if torch.cuda.is_available() else pos_weight

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if args.balance_data else nn.BCEWithLogitsLoss()

    train(model, processor, optimizer, criterion, train_dataloader, valid_dataloader, args)
    
if __name__ == '__main__':
    main()

