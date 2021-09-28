import pandas as pd
import numpy as np
import torch
import os
import librosa
from torch.utils.data import Dataset

from trans import get_transforms

BASE_DIR = '../data/'
TRAIN_PATH = os.path.join(BASE_DIR, 'train_dataset')
TEST_PATH = os.path.join(BASE_DIR, 'test_dataset')

class WavDatset(Dataset):

    def __init__(self, df, trans=None, is_train=True) -> None:
        super().__init__()

        self.df = df
        self.is_train = is_train
        self.trans = trans

    def read_wav2mel(self, path):
        y, sr = librosa.load(path)

        n_fft = 2048
        win_length = 2048
        hop_length = 1024
        n_mels = 128

        D = np.abs(librosa.stft(y, n_fft=n_fft, win_length = win_length, hop_length=hop_length))
        mel_spec = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length)

        return mel_spec

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        
        
        if self.is_train:
            path, label = self.df.iloc[index][['file_name', 'age_']]
            label -= 1
            path = os.path.join(TRAIN_PATH, path) + '.wav'
        else:
            path = self.df.iloc[index]['file_name']
            path = os.path.join(TEST_PATH, path) + '.wav'
        
        mel = self.read_wav2mel(path)
        if not self.trans is None:
            data = self.trans(image=mel)
            mel = data['image']
        
        if self.is_train:
            return mel, label
        else:
            return mel
    

if __name__ == '__main__':
    df = pd.read_csv('../data/train_label.csv')
    trans = get_transforms(data='train')
    dataset = WavDatset(df, trans=trans)
    mel, label = dataset[1]
    print(mel.shape)
    print(label)