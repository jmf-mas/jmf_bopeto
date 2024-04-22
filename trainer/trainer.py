from torch.cuda.amp import GradScaler, autocast

from models.ae import AECleaning, AEDetecting
from models.alad import ALAD
from models.dagmm import DAGMM
from models.dsebm import DSEBM
from .base import BaseTrainer
from .dataset import TabularDataset
import numpy as np
from collections import defaultdict
from typing import Union
from torch.nn import Parameter
from sklearn import metrics
import torch
import torch.nn as nn
from tqdm import trange
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

torch.autograd.set_detect_anomaly(True)

class Trainer:

    def __init__(self, params):
        self.params = params
        self.data = torch.tensor(self.params.data[:, :-1], dtype=torch.float32)
        torch.cuda.empty_cache()

    def train(self, to_save=False):
        dataset = TabularDataset(self.params.data)
        data_loader = DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers)
        optimizer = torch.optim.Adam(self.params.model.parameters(), lr=1e-3)
        scaler = GradScaler()
        return self.train(optimizer, scaler, data_loader, to_save)
        self.params.model.to(self.params.device)
        self.data = self.data.to(self.params.device)
        reconstruction_errors = []

        for epoch in range(self.params.epochs):
            self.params.model.train()
            total_loss = 0

            for batch in data_loader:
                data = batch['data'].to(self.params.device)
                optimizer.zero_grad()
                noisy_data = add_noise(data)
                with torch.cuda.amp.autocast():
                    outputs = self.params.model(data)
                    loss = torch.nn.MSELoss()(outputs, data)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader)
            print(f'Epoch {epoch + 1}, Average Loss: {avg_loss}')

            outputs = self.params.model(self.data)
            if to_save:
                #errors = torch.nn.functional.mse_loss(outputs, self.data, reduction='none').mean(1)
                errors = torch.nn.functional.cosine_similarity(outputs, self.data, dim=1)
                errors = errors.cpu().detach()
                if len(reconstruction_errors)==0:
                    reconstruction_errors =  errors
                else:
                    reconstruction_errors = np.column_stack((reconstruction_errors, errors))
        return reconstruction_errors

class TrainerAE(BaseTrainer):
    def __init__(self, params):
        self.model = AEDetecting(params)
        self.model.to(params.device)
        super(TrainerAE, self).__init__(params)

    def train(self):
        dataset = TabularDataset(self.params.data)
        data_loader = DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scaler = GradScaler()
        self.model.to(self.params.device)

        for epoch in range(self.params.epochs):
            self.model.train()
            total_loss = 0

            for batch in data_loader:
                data = batch['data'].to(self.params.device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = self.model(data)
                    loss = torch.nn.MSELoss()(outputs, data)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader)
            print(f'Epoch {epoch + 1}, Average Loss: {avg_loss}')


    def test_(self):
        val = torch.tensor(self.params.val[:, :-1], dtype=torch.float32)
        test = torch.tensor(self.params.test[:, :-1], dtype=torch.float32)
        val = val.to(self.params.device)
        test = test.to(self.params.device)
        self.model.eval()
        outputs_val = self.model(val)
        outputs_test = self.model(test)
        self.params.val_scores = torch.nn.functional.mse_loss(outputs_val, val, reduction='none').mean(1).cpu().detach().numpy()
        self.params.test_scores = torch.nn.functional.mse_loss(outputs_test, test, reduction='none').mean(1).cpu().detach().numpy()

class TrainerSK:

    def __init__(self, params):
        self.params = params
        self.name = "sklearn"
        if params.model == "ocsvm":
            self.model = OneClassSVM(kernel='rbf', nu=0.1)
        elif params.model == "lof":
            self.model = LocalOutlierFactor(n_neighbors=20, novelty=True)
        else:
            self.model = IsolationForest(n_estimators=50, random_state=42)

    def train(self):
        self.model.fit(self.params.data[:, :-1])
    def test(self, data):
        y_pred = self.model.predict(data)
        y_pred = np.where(y_pred == 1, 0, y_pred)
        self.params.y_pred = np.where(y_pred == -1, 1, y_pred)

class TrainerDAGMM(BaseTrainer):
    def __init__(self, params) -> None:
        self.params = params
        self.model = DAGMM(params)
        self.model.to(params.device)
        super(TrainerDAGMM, self).__init__(params)
        self.lamb_1 = params.lambda_1
        self.lamb_2 = params.lambda_2
        self.phi = None
        self.mu = None
        self.cov_mat = None
        self.covs = None


    def train_iter(self, sample):
        z_c, x_prime, _, z_r, gamma_hat = self.model(sample)
        phi, mu, cov_mat = self.compute_params(z_r, gamma_hat)
        energy_result, pen_cov_mat = self.estimate_sample_energy(
            z_r, phi, mu, cov_mat
        )
        self.phi = phi.data
        self.mu = mu.data
        self.cov_mat = cov_mat
        return self.loss(sample, x_prime, energy_result, pen_cov_mat)

    def loss(self, x, x_prime, energy, pen_cov_mat):
        rec_err = ((x - x_prime) ** 2).mean()
        return rec_err + self.lamb_1 * energy + self.lamb_2 * pen_cov_mat

    def test(self, data):
        self.model.eval()
        test_set = TabularDataset(data)
        test_loader = DataLoader(test_set, batch_size=self.params.batch_size, shuffle=True,
                                 num_workers=self.params.num_workers)

        with torch.no_grad():
            scores, y_true = [], []
            for row in test_loader:
                X = row['data'].to(self.params.device)
                y = row['target'].to(self.params.device)
                # forward pass
                code, x_prime, cosim, z, gamma = self.model(X)
                sample_energy, pen_cov_mat = self.estimate_sample_energy(
                    z, self.phi, self.mu, self.cov_mat, average_energy=False
                )
                y_true.extend(y.cpu().numpy())
                scores.extend(sample_energy.cpu().numpy())

        return np.array(y_true), np.array(scores)

    def weighted_log_sum_exp(self, x, weights, dim):
        m, idx = torch.max(x, dim=dim, keepdim=True)
        return m.squeeze(dim) + torch.log(torch.sum(torch.exp(x - m) * (weights.unsqueeze(2)), dim=dim))

    def relative_euclidean_dist(self, x, x_prime):
        return (x - x_prime).norm(2, dim=1) / x.norm(2, dim=1)

    def compute_params(self, z: torch.Tensor, gamma: torch.Tensor):
        N = z.shape[0]
        # K
        gamma_sum = torch.sum(gamma, dim=0)
        phi = gamma_sum / N

        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(-1)

        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)
        cov_mat = mu_z.unsqueeze(-1) @ mu_z.unsqueeze(-2)
        cov_mat = gamma.unsqueeze(-1).unsqueeze(-1) * cov_mat
        cov_mat = torch.sum(cov_mat, dim=0) / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov_mat

    def estimate_sample_energy(self, z, phi=None, mu=None, cov_mat=None, average_energy=True, eps=1e-12):
        if phi is None:
            phi = self.phi
        if mu is None:
            mu = self.mu
        if cov_mat is None:
            cov_mat = self.cov_mat

        # Avoid non-invertible covariance matrix by adding small values (eps)
        d = z.shape[1]
        cov_mat = cov_mat + (torch.eye(d)).to(self.device) * eps
        # N x K x D
        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)

        # scaler
        inv_cov_mat = torch.cholesky_inverse(torch.linalg.cholesky(cov_mat))
        # inv_cov_mat = torch.linalg.inv(cov_mat)
        det_cov_mat = torch.linalg.cholesky(2 * np.pi * cov_mat)
        det_cov_mat = torch.diagonal(det_cov_mat, dim1=1, dim2=2)
        det_cov_mat = torch.prod(det_cov_mat, dim=1)

        exp_term = torch.matmul(mu_z.unsqueeze(-2), inv_cov_mat)
        exp_term = torch.matmul(exp_term, mu_z.unsqueeze(-1))
        exp_term = - 0.5 * exp_term.squeeze()

        # Applying log-sum-exp stability trick
        # https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        if exp_term.ndim == 1:
            exp_term = exp_term.unsqueeze(0)
        max_val = torch.max(exp_term.clamp(min=0), dim=1, keepdim=True)[0]
        exp_result = torch.exp(exp_term - max_val)

        log_term = phi * exp_result
        log_term /= det_cov_mat
        log_term = log_term.sum(axis=-1)

        # energy computation
        energy_result = - max_val.squeeze() - torch.log(log_term + eps)

        if average_energy:
            energy_result = energy_result.mean()

        # penalty term
        cov_diag = torch.diagonal(cov_mat, dim1=1, dim2=2)
        pen_cov_mat = (1 / cov_diag).sum()

        return energy_result, pen_cov_mat

    def score(self, sample: torch.Tensor):
        _, _, _, z, _ = self.model(sample)
        return self.estimate_sample_energy(z)
class TrainerDSEBM(BaseTrainer):
    def __init__(self, params):
        assert params.score_metric == "reconstruction" or params.score_metric == "energy"
        self.model = DSEBM(params)
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

    def train_iter(self, X):
        noise = self.model.random_noise_like(X).to(self.device)
        X_noise = X + noise
        X.requires_grad_()
        X_noise.requires_grad_()
        out_noise = self.model(X_noise)
        energy_noise = self.energy(X_noise, out_noise)
        dEn_dX = torch.autograd.grad(energy_noise, X_noise, retain_graph=True, create_graph=True)
        fx_noise = (X_noise - dEn_dX[0])
        return self.loss(X, fx_noise)

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
        test_set = TabularDataset(data)
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

    def evaluate(self, y_true: np.array, scores: np.array, threshold, pos_label: int = 1) -> dict:
        res = defaultdict()
        for score, name in zip(scores, ["score_e", "score_r"]):
            res[name] = {"Precision": -1, "Recall": -1, "F1-Score": -1, "AUROC": -1, "AUPR": -1}
            thresh = np.percentile(scores, threshold)
            y_pred = self.predict(scores, thresh)
            precision, recall, f1, _ = metrics.precision_recall_fscore_support(
                y_true, y_pred, average='binary', pos_label=pos_label
            )
            res[name]["Precision"], res[name]["Recall"], res[name]["F1-Score"] = precision, recall, f1
            res[name]["AUROC"] = metrics.roc_auc_score(y_true, scores)
            res[name]["AUPR"] = metrics.average_precision_score(y_true, scores)
        return res

    def ren_dict_keys(self, d: dict, prefix=''):
        d_ = {}
        for k in d.keys():
            d_[f"{prefix}_{k}"] = d[k]

        return d_

    def loss(self, X, fx_noise):
        out = torch.square(X - fx_noise)
        out = torch.sum(out, dim=-1)
        out = torch.mean(out)
        return out

    def energy(self, X, X_hat):
        return 0.5 * torch.sum(torch.square(X - self.b_prime.expand_as(X))) - torch.sum(X_hat)

class TrainerALAD(BaseTrainer):
    def __init__(self, params):
        self.params = params
        self.criterion = nn.BCEWithLogitsLoss()
        self.optim_ge, self.optim_d = None, None
        self.model = ALAD(params)
        self.model.to(params.device)
        super(TrainerALAD, self).__init__(params)


    def train_iter(self, sample):
        pass

    def score(self, sample):
        _, feature_real = self.model.D_xx(sample, sample)
        _, feature_gen = self.model.D_xx(sample, self.model.G(self.model.E(sample)))
        return torch.linalg.norm(feature_real - feature_gen, 2, keepdim=False, dim=1)

    def set_optimizer(self):
        self.optim_ge = optim.Adam(
            list(self.model.G.parameters()) + list(self.model.E.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )
        self.optim_d = optim.Adam(
            list(self.model.D_xz.parameters()) + list(self.model.D_zz.parameters()) + list(
                self.model.D_xx.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )

    def train_iter_dis(self, X):

        # Labels
        y_true = Variable(torch.zeros(X.size(0), 1)).to(self.device)
        y_fake = Variable(torch.ones(X.size(0), 1)).to(self.device)
        # Forward pass
        out_truexz, out_fakexz, out_truezz, out_fakezz, out_truexx, out_fakexx = self.model(X)
        # Compute loss
        # Discriminators Losses
        loss_dxz = self.criterion(out_truexz, y_true) + self.criterion(out_fakexz, y_fake)
        loss_dzz = self.criterion(out_truezz, y_true) + self.criterion(out_fakezz, y_fake)
        loss_dxx = self.criterion(out_truexx, y_true) + self.criterion(out_fakexx, y_fake)
        loss_d = loss_dxz + loss_dzz + loss_dxx

        return loss_d

    def train_iter_gen(self, X):
        # Labels
        y_true = Variable(torch.zeros(X.size(0), 1)).to(self.device)
        y_fake = Variable(torch.ones(X.size(0), 1)).to(self.device)
        # Forward pass
        out_truexz, out_fakexz, out_truezz, out_fakezz, out_truexx, out_fakexx = self.model(X)
        # Generator losses
        loss_gexz = self.criterion(out_fakexz, y_true) + self.criterion(out_truexz, y_fake)
        loss_gezz = self.criterion(out_fakezz, y_true) + self.criterion(out_truezz, y_fake)
        loss_gexx = self.criterion(out_fakexx, y_true) + self.criterion(out_truexx, y_fake)
        cycle_consistency = loss_gexx + loss_gezz
        loss_ge = loss_gexz + cycle_consistency

        return loss_ge

    def train(self):
        dataset = TabularDataset(self.params.data)
        data_loader = DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True,
                                 num_workers=self.params.num_workers)
        self.model.train()
        for epoch in range(self.n_epochs):
            ge_losses, d_losses = 0, 0
            with trange(len(data_loader)) as t:

                for batch in data_loader:
                    data = batch['data'].to(self.params.device)
                    X_dis, X_gen = data, data.clone().to(self.device).float()
                    # Forward pass

                    # Cleaning gradients
                    self.optim_d.zero_grad()
                    loss_d = self.train_iter_dis(X_dis)
                    # Backward pass
                    loss_d.backward()
                    self.optim_d.step()

                    # Cleaning gradients
                    self.optim_ge.zero_grad()
                    loss_ge = self.train_iter_gen(X_gen)
                    # Backward pass
                    loss_ge.backward()
                    self.optim_ge.step()

                    # Journaling
                    d_losses += loss_d.item()
                    ge_losses += loss_ge.item()
                    t.set_postfix(
                        ep=epoch + 1,
                        loss_d='{:05.4f}'.format(loss_d),
                        loss_ge='{:05.4f}'.format(loss_ge),
                    )
                    t.update()

    def eval(self, dataset: DataLoader):
        self.model.eval()
        with torch.no_grad():
            loss = 0
            for row in dataset:
                X = row['data'].to(self.params.device)
                X = X.to(self.device).float()
                loss_d = self.train_iter_dis(X)
                loss_ge = self.train_iter_gen(X)
                loss += loss_d.item() + loss_ge.item()
        self.model.train()
        return loss

def add_noise(data, noise_factor=0.5):
    noise = noise_factor * torch.randn_like(data)
    noisy_data = data + noise
    return noisy_data