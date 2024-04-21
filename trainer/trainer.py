from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from .dataset import TabularDataset
import torch
import numpy as np
import os
import time
import datetime
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:

    def __init__(self, params):
        self.params = params
        self.data = torch.tensor(self.params.data[:, :-1], dtype=torch.float32)
        torch.cuda.empty_cache()

    def run(self, to_save=False):
        dataset = TabularDataset(self.params.data)
        data_loader = DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers)
        optimizer = torch.optim.Adam(self.params.model.parameters(), lr=1e-3)
        scaler = GradScaler()
        return self.train(optimizer, scaler, data_loader, to_save)

    def train(self, optimizer, scaler, data_loader, to_save=False):
        self.params.model.to(device)
        self.data = self.data.to(device)
        reconstruction_errors = []

        for epoch in range(self.params.epochs):
            self.params.model.train()
            total_loss = 0

            for batch in data_loader:
                data = batch['data'].to(device)
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

class TrainerAE:

    def __init__(self, params):
        self.params = params
        torch.cuda.empty_cache()

    def run(self):
        dataset = TabularDataset(self.params.data)
        data_loader = DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers)
        optimizer = torch.optim.Adam(self.params.model.parameters(), lr=1e-3)
        scaler = GradScaler()
        return self.train(optimizer, scaler, data_loader)

    def train(self, optimizer, scaler, data_loader):
        val = torch.tensor(self.params.val[:, :-1], dtype=torch.float32)
        test = torch.tensor(self.params.test[:, :-1], dtype=torch.float32)
        self.params.model.to(device)
        val = val.to(device)
        test = test.to(device)

        for epoch in range(self.params.epochs):
            self.params.model.train()
            total_loss = 0

            for batch in data_loader:
                data = batch['data'].to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = self.params.model(data)
                    loss = torch.nn.MSELoss()(outputs, data)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader)
            print(f'Epoch {epoch + 1}, Average Loss: {avg_loss}')

        self.params.model.eval()
        outputs_val = self.params.model(val)
        outputs_test = self.params.model(test)
        self.params.val_scores = torch.nn.functional.mse_loss(outputs_val, val, reduction='none').mean(1).cpu().detach().numpy()
        self.params.test_scores = torch.nn.functional.mse_loss(outputs_test, test, reduction='none').mean(1).cpu().detach().numpy()

class TrainerSK:

    def __init__(self, params):
        self.params = params

    def run(self):
        return self.train()

    def train(self):
        self.params.model.fit(self.params.data[:, :-1])
        y_pred = self.params.model.predict(self.params.test[:, :-1])
        y_pred = np.where(y_pred == 1, 0, y_pred)
        self.params.y_pred = np.where(y_pred == -1, 1, y_pred)


class TrainerDaGMM(object):
    DEFAULTS = {}

    def __init__(self, params):
        self.params = params
        self.data = torch.tensor(self.params.data[:, :-1], dtype=torch.float32)
        torch.cuda.empty_cache()

    def run(self, to_save=False):
        dataset = TabularDataset(self.params.data)
        data_loader = DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers)
        optimizer = torch.optim.Adam(self.params.model.parameters(), lr=self.params.learning_rate)
        return self.train(optimizer, data_loader, to_save)

    def load_pretrained_model(self):
        self.params.model.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_dagmm.pth'.format(self.pretrained_model))))

        print("phi", self.dagmm.phi, "mu", self.params.model.mu, "cov", self.params.model.cov)

        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.params.model.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def train(self, optimizer, data_loader, to_save = False):
        iters_per_epoch = len(data_loader)

        start = 0
        iter_ctr = 0
        start_time = time.time()

        for e in range(start, self.params.epochs):
            for i, batch in enumerate(data_loader):
                data = batch['data'].to(device)
                iter_ctr += 1
                start = time.time()

                input_data = self.to_var(data)

                total_loss, sample_energy, recon_error, cov_diag = self.dagmm_step(input_data, optimizer)
                # Logging
                loss = {}
                loss['total_loss'] = total_loss.data.item()
                loss['sample_energy'] = sample_energy.item()
                loss['recon_error'] = recon_error.item()
                loss['cov_diag'] = cov_diag.item()

                # Print out log info
                if (i + 1) % self.params.log_step == 0:
                    elapsed = time.time() - start_time
                    total_time = ((self.params.epochs * iters_per_epoch) - (e * iters_per_epoch + i)) * elapsed / (
                                e * iters_per_epoch + i + 1)
                    epoch_time = (iters_per_epoch - i) * elapsed / (e * iters_per_epoch + i + 1)

                    epoch_time = str(datetime.timedelta(seconds=epoch_time))
                    total_time = str(datetime.timedelta(seconds=total_time))
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    lr_tmp = []
                    for param_group in optimizer.param_groups:
                        lr_tmp.append(param_group['lr'])
                    tmplr = np.squeeze(np.array(lr_tmp))

                    log = "Elapsed {}/{} -- {} , Epoch [{}/{}], Iter [{}/{}], lr {}".format(
                        elapsed, epoch_time, total_time, e + 1, self.params.epochs, i + 1, iters_per_epoch, tmplr)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)


    def dagmm_step(self, input_data, optimizer):
        self.params.model.train()
        enc, dec, z, gamma = self.params.model(input_data)
        total_loss, sample_energy, recon_error, cov_diag = self.params.model.loss_function(input_data, dec, z, gamma,
                                                                                    self.params.lambda_energy,
                                                                                    self.params.lambda_cov_diag)
        self.reset_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params.model.parameters(), 5)
        optimizer.step()

        return total_loss, sample_energy, recon_error, cov_diag

def add_noise(data, noise_factor=0.5):
    noise = noise_factor * torch.randn_like(data)
    noisy_data = data + noise
    return noisy_data