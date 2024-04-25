from .base import BaseTrainer
from .dataset import TabularDataset
import numpy as np
from torch.nn import Parameter
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)

class TrainerDSEBM(BaseTrainer):
    def __init__(self, params):
        self.params = params
        assert params.score_metric == "reconstruction" or params.score_metric == "energy"
        self.model = self.params.model
        self.model.to(params.device)
        super(TrainerDSEBM, self).__init__(params)
        self.score_metric = params.score_metric
        self.criterion = nn.BCEWithLogitsLoss()
        self.b_prime = Parameter(torch.Tensor(self.model.in_features).to(self.device))
        torch.nn.init.normal_(self.b_prime)
        self.optim = optim.Adam(
            list(self.model.parameters()) + [self.b_prime],
            lr=self.lr, betas=(0.5, 0.999)
        )

    def train_iter(self, X, w):
        noise = self.model.random_noise_like(X).to(self.device)
        X_noise = X + noise
        X.requires_grad_()
        X_noise.requires_grad_()
        out_noise = self.model(X_noise)
        energy_noise = self.energy(X_noise, out_noise)
        dEn_dX = torch.autograd.grad(energy_noise, X_noise, retain_graph=True, create_graph=True)
        fx_noise = (X_noise - dEn_dX[0])
        return self.loss(X, w, fx_noise)

    def score(self, sample: torch.Tensor):
        # Evaluation of the score based on the energy
        with torch.no_grad():
            flat = sample - self.b_prime
            out = self.model(sample)
            energies = 0.5 * torch.sum(torch.square(flat), dim=1) - torch.sum(out, dim=1)

        # Evaluation of the score based on the reconstruction error
        sample.requires_grad_()
        out = self.model(sample)
        energy = self.energy(sample, out)
        dEn_dX = torch.autograd.grad(energy, sample)[0]
        rec_errs = torch.linalg.norm(dEn_dX, 2, keepdim=False, dim=1)
        return energies.cpu().numpy(), rec_errs.cpu().numpy()

    def test(self, data):
        self.model.eval()
        test_set = TabularDataset(data, np.ones(data.shape[0]))
        test_loader = DataLoader(test_set, batch_size=self.params.batch_size, shuffle=True,
                                 num_workers=self.params.num_workers)
        y_true, scores = [], []
        scores_e, scores_r = [], []
        for row in test_loader:
            X = row['data'].to(self.params.device)
            y = row['target'].to(self.params.device)
            score_e, score_r = self.score(X)

            y_true.extend(y.cpu().tolist())
            scores_e.extend(score_e)
            scores_r.extend(score_r)

        scores = scores_r if self.score_metric == "reconstruction" else scores_e
        return np.array(y_true), np.array(scores)

    def ren_dict_keys(self, d: dict, prefix=''):
        d_ = {}
        for k in d.keys():
            d_[f"{prefix}_{k}"] = d[k]

        return d_

    def loss(self, X, w, fx_noise):
        out = w.unsqueeze(1)*torch.square(X - fx_noise)
        out = torch.sum(out, dim=-1)
        out = torch.mean(out)
        return out

    def energy(self, X, X_hat):
        return 0.5 * torch.sum(torch.square(X - self.b_prime.expand_as(X))) - torch.sum(X_hat)
