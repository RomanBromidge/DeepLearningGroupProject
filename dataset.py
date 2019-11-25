import torch
from torch.utils import data
import numpy as np
import pickle


class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, mode):
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.mode = mode

    def __getitem__(self, index):
        item = self.dataset[index]
        item_features = item['features']
        cst = self.getCSTfeature(item_features)
        if self.mode == 'LMC':
            # Edit here to load and concatenate the neccessary features to
            # create the LMC feature
            lm = item_features['logmelspec']
            feature = np.concatenate((lm,cst))
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MC':
            # Edit here to load and concatenate the neccessary features to
            # create the MC feature
            mfcc = item_features['mfcc']
            feature = np.concatenate((mfcc,cst))
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MLMC':
            # Edit here to load and concatenate the neccessary features to
            # create the MLMC feature
            mfcc = item_features['mfcc']
            lm = item_features['logmelspec']
            feature = np.concatenate((mfcc,lm,cst))
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        else:
            raise ValueError('Illegal Argument: mode must be one of {LMC,MC,MLMC},' +
                            ' not ' + self.mode)

        label = item['classID']
        fname = item['filename']
        return feature, label, fname

    def getCSTfeature(self, features):
        return np.concatenate((features['chroma'],
                               features['spectral_contrast'],
                               features['tonnetz']))

    def __len__(self):
        return len(self.dataset)
