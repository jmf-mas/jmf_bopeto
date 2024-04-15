import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA

class FGM:
    def __init__(self, X, gamma = 0.1):
        X = torch.tensor(X, dtype=torch.float32)
        y = np.array([0]*len(X))
        self.n = int(gamma * len(y))
        y = torch.tensor(y, dtype=torch.float32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = X.to(self.device)
        y = y.to(self.device)
        self.epochs = 10
        dataset = TensorDataset(X, y)
        self.train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(dataset, batch_size=1, shuffle=True)


    def generate(self):
        model = BinaryClassificationModel(input_size=20)
        model.to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epochs = 5
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)  # Squeeze to match the shape of labels
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(self.train_dataset)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}')

        test_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=False)
        classifier = PyTorchClassifier(
            model=model,
            loss=criterion,
            input_shape=(1, 20),
            nb_classes=2,
            optimizer=optimizer,
            clip_values=(0, 1),
            preprocessing=(0, 1)  # Ensure input values are in [0, 1] range
        )
        attack = FastGradientMethod(estimator=classifier, eps=15)

        adv_examples = []
        i = 0
        for data, target in test_loader:
            adv_data = attack.generate(data.cpu().numpy())
            adv_examples.append((adv_data, target))
            i += 1
            if i >= self.n:
                break
        # evaluate the model on the adversarial examples
        adv_correct = 0
        with torch.no_grad():
            for data, target in adv_examples:
                output = model(torch.tensor(data).to(self.device))
                predicted = (output >= 0.5).float()
                adv_correct += (predicted.to(self.device) == target.to(self.device)).sum().item()

        print(f'Accuracy on adversarial examples: {100 * adv_correct / self.n}%')
        return adv_examples


class BinaryClassificationModel(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return self.sigmoid(x)

class JMF:

    def __init__(self, X, gamma=0.1):
        self.gamma = gamma
        self.X = X
    def generate(self):
        f = PCA(n_components=1)
        Z_X = f.fit_transform(self.X)
        N = len(self.X)
        M = np.ceil(self.gamma*N)
        mu_Z = np.mean(Z_X)
        sigma_Z = np.std(Z_X)
        U0 = np.random.uniform(low=mu_Z - 3 * sigma_Z, high=mu_Z - 2 * sigma_Z, size=int(M/2))
        U1 = np.random.uniform(low=mu_Z + 2 * sigma_Z, high=mu_Z + 3 * sigma_Z, size=int(M/2))
        U_gamma = np.concatenate((U0, U1))
        U_gamma = U_gamma.reshape((-1, 1))
        X_gamma = f.inverse_transform(U_gamma)
        return X_gamma