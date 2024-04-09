import torch.nn as nn
import pickle
from pathlib import Path


class AE(nn.Module):

    def __init__(self, in_dim, name, dropout=0):
        super(AE, self).__init__()
        if "cifar" in name or "svhn" in name or "mnist" in name:
            self.enc = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.Linear(512, 256),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.Linear(64, 32),
                nn.Dropout(dropout),
                nn.Linear(32, 16),
                nn.Linear(16, 8)
            )

        else:
            self.enc = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.Linear(64, 32),
                nn.Dropout(dropout),
                nn.Linear(32, 16),
                nn.Dropout(dropout),
                nn.Linear(16, 8)
            )
        self.name = name

        if "cifar" in name or "svhn" in name or "mnist" in name:
            self.dec = nn.Sequential(
                nn.Linear(8, 16),
                nn.Linear(16, 32),
                nn.Dropout(dropout),
                nn.Linear(32, 64),
                nn.Linear(64, 128),
                nn.Dropout(dropout),
                nn.Linear(128, 256),
                nn.Dropout(dropout),
                nn.Linear(256, 512),
                nn.Linear(512, in_dim)
            )
        else:
            self.dec = nn.Sequential(
                nn.Linear(8, 16),
                nn.Linear(16, 32),
                nn.Dropout(dropout),
                nn.Linear(32, 64),
                nn.Dropout(dropout),
                nn.Linear(64, in_dim)
            )

    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return decode

    def compute_l2_loss(self):
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        return l2_lambda * l2_norm

    def save(self):
        parent_name = "checkpoints"
        Path(parent_name).mkdir(parents=True, exist_ok=True)
        with open(parent_name + "/" + self.name + ".pickle", "wb") as fp:
            pickle.dump(self.state_dict(), fp)

    def load(self):
        with open("checkpoints/" + self.name + ".pickle", "rb") as fp:
            self.load_state_dict(pickle.load(fp))

    def load_from(self, file):
        with open("checkpoints/" + file + ".pickle", "rb") as fp:
            self.load_state_dict(pickle.load(fp))

