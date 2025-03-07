import torch
from torch.utils.data import Dataset
import numpy as np

class TabularDataset(Dataset):
    def __init__(self, data, weights):
        """
        dataframe: A pandas DataFrame where the last column is the target.
        """
        self.data = data[:, :-1].astype(np.float32)  # Exclude the target column
        self.targets = data[:, -1].astype(np.float32)  # Targets
        self.weights = weights
        self.indices = np.arange(len(data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'data': torch.tensor(self.data[idx], dtype=torch.float),
                'target': torch.tensor(self.targets[idx], dtype=torch.float),
                'weight': torch.tensor(self.weights[idx], dtype=torch.float),
                'index': idx}
