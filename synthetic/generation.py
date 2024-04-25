
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from trainer.dataset import TabularDataset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FGM:
    def __init__(self, params):
        self.params = params
        self.data = torch.tensor(self.params.data[:, :-1], dtype=torch.float32)
        torch.cuda.empty_cache()
    def generate(self):
        weights = np.ones(self.params.fragment.shape[0])
        dataset = TabularDataset(self.params.fragment, weights)
        data_loader = DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True,
                                 num_workers=self.params.num_workers)
        optimizer = torch.optim.Adam(self.params.model.parameters(), lr=1e-3)
        scaler = GradScaler()

        self.params.model.eval()

        adversarial_examples = []
        for batch in data_loader:
            data = batch['data'].to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.params.model(data)
                loss = torch.nn.MSELoss()(outputs, data)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            outputs = self.params.model(data)
            perturbed_inputs = outputs  + np.random.uniform(0, 0.1, 1)[0] * outputs  # FGSM perturbation
            perturbed_outputs = self.params.model(perturbed_inputs)
            perturbed_outputs = perturbed_outputs.cpu().detach()
            if len(adversarial_examples)==0:
                adversarial_examples =  perturbed_outputs
            else:
                adversarial_examples = np.vstack((adversarial_examples, perturbed_outputs))
        return adversarial_examples

class JMF:

    def __init__(self, params):
        self.gamma = params.gamma
        self.X = params.data[:, :-1]
    def generate(self):
        f = PCA(n_components=1)
        z_x = f.fit_transform(self.X)
        n = len(self.X)
        m = np.ceil(self.gamma*n)
        mu_z = np.mean(z_x)
        sigma_z = np.std(z_x)
        u0= np.random.uniform(low=mu_z - 3 * sigma_z, high=mu_z - 2 * sigma_z, size=int(m/2))
        u1 = np.random.uniform(low=mu_z + 2 * sigma_z, high=mu_z + 3 * sigma_z, size=int(m/2))
        u_gamma = np.concatenate((u0, u1))
        u_gamma = u_gamma.reshape((-1, 1))
        x_gamma = f.inverse_transform(u_gamma)
        return x_gamma