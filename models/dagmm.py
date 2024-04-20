import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import pickle
from pathlib import Path


class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.linalg.cholesky(a)
        ctx.save_for_backward(l)
        return l

    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s


class DaGMM(nn.Module):
    def __init__(self, name, input_size=118, n_gmm=2, latent_dim=3):
        super(DaGMM, self).__init__()
        self.name = name
        layers = []
        layers += [nn.Linear(input_size, 60)]
        layers += [nn.Tanh()]
        layers += [nn.Linear(60, 30)]
        layers += [nn.Tanh()]
        layers += [nn.Linear(30, 10)]
        layers += [nn.Tanh()]
        layers += [nn.Linear(10, 1)]

        self.encoder = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(1, 10)]
        layers += [nn.Tanh()]
        layers += [nn.Linear(10, 30)]
        layers += [nn.Tanh()]
        layers += [nn.Linear(30, 60)]
        layers += [nn.Tanh()]
        layers += [nn.Linear(60, input_size)]

        self.decoder = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(latent_dim, 10)]
        layers += [nn.Tanh()]
        layers += [nn.Dropout(p=0.5)]
        layers += [nn.Linear(10, n_gmm)]
        layers += [nn.Softmax(dim=1)]

        self.estimation = nn.Sequential(*layers)

        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm, latent_dim))
        self.register_buffer("cov", torch.zeros(n_gmm, latent_dim, latent_dim))

    def forward(self, x):

        enc = self.encoder(x)
        dec = self.decoder(enc)
        rec_cosine = F.cosine_similarity(x, dec, dim=1)
        rec_euclidean = relative_euclidean_distance(x, dec)
        z = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)
        gamma = self.estimation(z)

        return enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        sum_gamma = torch.sum(gamma, dim=0)

        phi = (sum_gamma / N)

        self.phi = phi.data

        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data

        return phi, mu, cov


    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = to_var(self.phi)
        if mu is None:
            mu = to_var(self.mu)
        if cov is None:
            cov = to_var(self.cov)

        k, D, _ = cov.size()

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            cov_k = cov[i] + to_var(torch.eye(D) * eps)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2 * np.pi)).diag().prod()).unsqueeze(0))
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).cuda()

        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)

        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag

    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):

        recon_error = torch.mean((x - x_hat) ** 2)
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)
        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag

        return loss, sample_energy, recon_error, cov_diag

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

def relative_euclidean_distance(a, b):
    return (a - b).norm(2, dim=1) / a.norm(2, dim=1)

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
