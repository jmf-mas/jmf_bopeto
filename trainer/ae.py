from torch.cuda.amp import GradScaler, autocast
from .base import BaseTrainer, weighted_loss
from .dataset import TabularDataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange
torch.autograd.set_detect_anomaly(True)

class Trainer:

    def __init__(self, params):
        self.params = params
        self.data = torch.tensor(self.params.data[:, :-1], dtype=torch.float32)
        torch.cuda.empty_cache()

    def train(self, to_save=False):
        dataset = TabularDataset(self.params.data, self.params.weights)
        data_loader = DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers)
        optimizer = torch.optim.Adam(self.params.model.parameters(), lr=1e-3)
        scaler = GradScaler()
        self.params.model.to(self.params.device)
        self.data = self.data.to(self.params.device)
        reconstruction_errors = []

        for epoch in range(self.params.epochs):
            self.params.model.train()
            epoch_loss = 0.0
            counter = 1

            with trange(len(data_loader)) as t:
                for batch in data_loader:
                    x_in = batch['data'].to(self.params.device)
                    weight = batch['weight'].to(self.params.device)
                    optimizer.zero_grad()
                    x_in = add_noise(x_in)
                    with torch.cuda.amp.autocast():
                        outputs = self.params.model(x_in)
                        loss = weighted_loss(x_in, outputs, weight)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss += loss.item()
                    t.set_postfix(
                        loss='{:.8f}'.format(epoch_loss / counter),
                        epoch=epoch + 1
                    )
                    t.update()
                    counter += 1

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
        self.params = params
        self.params.model.to(params.device)
        self.model = self.params.model
        super(TrainerAE, self).__init__(params)

    def train_iter(self, x_in, weights):
        x_out = self.model(x_in)[1]
        loss = weighted_loss(x_in, x_out, weights)
        return loss
    def score(self, x_in):
        _, x_out = self.model(x_in)
        return ((x_in - x_out) ** 2).sum(axis=1)


def add_noise(data, noise_factor=0.5):
    noise = noise_factor * torch.randn_like(data)
    noisy_data = data + noise
    return noisy_data