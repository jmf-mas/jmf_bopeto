import numpy as np
import torch

from abc import ABC, abstractmethod
from typing import Union
from sklearn import metrics as sk_metrics
from torch.utils.data.dataloader import DataLoader
from torch import optim
from tqdm import trange
from torch.cuda.amp import GradScaler, autocast
from metric.utils import Patience
from trainer.dataset import TabularDataset


class BaseTrainer(ABC):

    def __init__(self, model):
        self.device = model.params.device
        self.model = model.to(self.device)
        self.batch_size = model.params.batch_size
        self.n_jobs_dataloader = model.params.n_jobs_dataloader
        self.n_epochs = model.params.epochs
        self.lr = model.params.learning_rate
        self.weight_decay = model.params.weight_decay
        self.optimizer = self.set_optimizer(self.weight_decay)

        patience = model.params.patience
        self.early_stopper = Patience(patience=patience, use_train_loss=False, model=self.model)
        self.params = model.params

    @abstractmethod
    def train_iter(self, sample: torch.Tensor, **kwargs):
        pass

    @abstractmethod
    def score(self, sample: torch.Tensor):
        pass

    def after_training(self):
        """
        Perform any action after training is done
        """
        pass

    def before_training(self, dataset: DataLoader):
        """
        Optionally perform pre-training or other operations.
        """
        pass

    def set_optimizer(self, weight_decay):
        return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)

    def train(self):
        dataset = TabularDataset(self.params.data)
        data_loader = DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers)
        val_set = TabularDataset(self.params.val)
        val_loader = DataLoader(val_set, batch_size=self.params.batch_size, shuffle=True,
                                 num_workers=self.params.num_workers)
        test_set = TabularDataset(self.params.val)
        test_loader = DataLoader(test_set, batch_size=self.params.batch_size, shuffle=True,
                                num_workers=self.params.num_workers)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scaler = GradScaler()

        self.model.train()
        should_stop = False
        for epoch in range(self.params.epochs):
            epoch_loss = 0.0
            len_trainloader = len(data_loader)
            counter = 1

            with trange(len_trainloader) as t:
                for batch in data_loader:
                    data = batch['data'].to(self.params.device)
                    self.optimizer.zero_grad()
                    loss = self.train_iter(data)

                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    t.set_postfix(
                        loss='{:.3f}'.format(epoch_loss / counter),
                        epoch=epoch + 1
                    )
                    t.update()
                    counter += 1
            if val_loader is not None:
                val_loss = self.eval(val_loader).item()
                results = self._eval(test_loader)

                if self.params.early_stopping:
                    should_stop = self.early_stopper.stop(epoch=epoch,
                                                          val_loss=val_loss,
                                                          train_loss=epoch_loss,
                                                          val_auc=results["proc1p"],
                                                          test_f1=results["f_score"])

                    print(f'Val loss :{val_loss} | Train loss: {self.eval(data_loader).item()} '
                          f'| early_stop? {should_stop} | patience:{self.early_stopper.counter}  ')
                print(results)

                if should_stop:
                    break
        if self.params.early_stopping:
            self.early_stopper.get_best_vl_metrics()

        self.after_training()

    def eval(self, dataset: DataLoader):
        self.model.eval()
        with torch.no_grad():
            loss = 0
            for row in dataset:
                data = row['data'].to(self.params.device)
                loss += self.train_iter(data)
        self.model.train()

        return loss

    def _eval(self, dataset: DataLoader):
        self.model.eval()
        y_true, scores = [], []
        with torch.no_grad():
            for row in dataset:
                X, y = row[0], row[1]
                X = X.to(self.device).float()
                # if len(X) < self.batch_size:
                #     break
                score = self.score(X)
                y_true.append(y.cpu().numpy())
                scores.append(score.cpu().numpy())
        self.model.train()

        y_true, scores = np.concatenate(y_true, axis=0), np.concatenate(scores, axis=0)
        # _estimate_threshold_metrics

        accuracy, precision, recall, f_score, roc, avgpr = _estimate_threshold_metrics(scores, y_true,
                                                                                       optimal=False)

        return {k: round(v, 3) for k, v in
                dict(accuracy=accuracy,
                     precision=precision, recall=recall, f_score=f_score, avgpr=avgpr, proc1p=roc).items()}

    def test(self, dataset: DataLoader) -> Union[np.array, np.array]:
        self.model.eval()
        y_true, scores = [], []
        with torch.no_grad():
            for row in dataset:
                X, y = row[0], row[1]
                X = X.to(self.device).float()
                score = self.score(X)
                y_true.extend(y.cpu().tolist())
                scores.extend(score.cpu().tolist())
        self.model.train()

        return np.array(y_true), np.array(scores)

    def test_return_all(self, dataset: DataLoader) -> Union[np.array, np.array]:
        self.model.eval()
        y_true, scores, xs = [], [], []
        with torch.no_grad():
            for row in dataset:
                X, y = row[0], row[1]
                X = X.to(self.device).float()
                # if len(X) < self.batch_size:
                #     break
                score = self.score(X)
                y_true.extend(y.cpu().tolist())
                scores.extend(score.cpu().tolist())
                xs.extend(X.cpu().tolist())
        self.model.train()

        return np.array(y_true), np.array(scores), np.array(xs)

    def get_params(self) -> dict:
        return {
            "learning_rate": self.lr,
            "epochs": self.n_epochs,
            "batch_size": self.batch_size,
            **self.model.get_params()
        }

    def predict(self, scores: np.array, thresh: float):
        return (scores >= thresh).astype(int)

