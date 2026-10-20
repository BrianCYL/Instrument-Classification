import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

import numpy as np


class AudioData(Dataset):
    def __init__(self, folder_path, type):
        self.folder_path = folder_path
        if type not in ['train', 'validation', 'test']:
            raise ValueError('type should be either train, validation, or test')
        self.type = type
        self.data = []
        self.labels = []

    def load_audio(self):
        self.data = np.load(f'{self.folder_path}/{self.type}_data.npy')
        self.labels = np.load(f'{self.folder_path}/{self.type}_labels.npy')
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        
class AudioDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        super(AudioDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle)

class MusicClassifier(nn.Module):
    def __init__(self):
        super(MusicClassifier, self).__init__()
        self.spec_bn = nn.BatchNorm2d(13)
        self.agg = nn.Conv2d(13, 1, kernel_size=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc = nn.Sequential(nn.Linear(187 * 384, 512),
                                nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Linear(256, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Linear(64, 9))
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = x.permute(1, 0, 2, 3)
        x = self.spec_bn(x)
        x = self.agg(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class CompleteModel(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(CompleteModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x):
        features = self.feature_extractor(x, output_hidden_states=True)
        features = torch.stack(features.hidden_states).squeeze()
        output = self.classifier(features)
        return output
