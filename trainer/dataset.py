import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TabularDataset(Dataset):
    def __init__(self, dataframe):
        """
        dataframe: A pandas DataFrame where the last column is the target.
        """
        self.data = dataframe.iloc[:, :-1].to_numpy().astype(np.float32)  # Exclude the target column
        self.targets = dataframe.iloc[:, -1].to_numpy().astype(np.float32)  # Targets
        self.indices = np.arange(len(dataframe))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'data': torch.tensor(self.data[idx], dtype=torch.float),
                'target': torch.tensor(self.targets[idx], dtype=torch.float),
                'index': idx}
