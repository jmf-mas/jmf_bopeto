from torch.utils.data.dataloader import DataLoader
import torch
from trainer.base import BaseTrainer


class TrainerSVDD(BaseTrainer):
    # code source: https://github.com/lukasruff/Deep-SVDD

    def __init__(self, params):
        self.params = params
        self.model = self.params.model
        self.model.to(params.device)
        super(TrainerSVDD, self).__init__(params)
        self.c = params.c
        self.R = params.R

    def train_iter(self, sample, w):
        outputs = self.model(sample)
        dist = w.unsqueeze(1) * torch.sum((outputs - self.c) ** 2, dim=1)
        return torch.mean(dist)

    def score(self, sample: torch.Tensor):
        outputs = self.model(sample)
        return torch.sum((outputs - self.c) ** 2, dim=1)

    def before_training(self, dataset: DataLoader):
        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print("Initializing center c...")
            self.c = self.init_center_c(dataset)
            print("Center c initialized.")

    def init_center_c(self, train_loader: DataLoader, eps=0.1):
        n_samples = 0
        c = torch.zeros(self.model.rep_dim, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for sample in train_loader:
                # get the inputs of the batch
                X = sample[0]
                X = X.to(self.device).float()
                outputs = self.model(X)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        if c.isnan().sum() > 0:
            raise Exception("NaN value encountered during init_center_c")

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def get_params(self) -> dict:
        return {'c': self.c, 'R': self.R, **self.model.get_params()}

