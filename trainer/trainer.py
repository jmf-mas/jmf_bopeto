import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from .dataset import TabularDataset

class Trainer:

    def __init__(self, params):
        self.params = params

    def run(self):
        dataset = TabularDataset(self.params.data)
        data_loader = DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers)
        optimizer = torch.optim.Adam(self.params.model.parameters(), lr=1e-3)
        scaler = GradScaler()
        return self.train(self.params.model, optimizer, scaler, data_loader)

    def train(self, optimizer, scaler, data_loader, epochs=10):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params.model.to(device)
        reconstruction_errors = []

        for epoch in range(epochs):
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
                errors = torch.nn.functional.mse_loss(outputs, data, reduction='none').mean(1)  # Mean over features
                for index, error in zip(batch['index'], errors):
                    reconstruction_errors.append((epoch, index, error.item()))

            avg_loss = total_loss / len(data_loader)
            print(f'Epoch {epoch + 1}, Average Loss: {avg_loss}')

        return reconstruction_errors
