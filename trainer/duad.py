from copy import deepcopy
from torch import nn, optim
from tqdm import trange
import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from trainer.dataset import TabularDataset
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Sampler, Subset


class TrainerDUAD:

    def __init__(self, params):

        self.params = params
        self.model = self.params.model
        self.model.to(params.device)
        super(TrainerDUAD, self).__init__(params)
        self.metric_hist = []
        self.r = params.r
        self.p = params.p
        self.p0 = params.p0
        self.num_cluster = params.num_cluster
        self.lr = params.learning_rate
        self.n_epochs = params.epochs
        self.device = params.device
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr,weight_decay=params.weight_decay)

        self.criterion = nn.MSELoss()

    def re_evaluation(self, X, p, num_clusters=20):
        gmm = GaussianMixture(n_components=num_clusters, max_iter=400)
        gmm.fit(X)
        pred_label = gmm.predict(X)
        X_means = torch.from_numpy(gmm.means_)

        clusters_vars = []
        for i in range(num_clusters):
            var_ci = torch.sum((X[pred_label == i] - X_means[i].unsqueeze(dim=0)) ** 2)
            var_ci /= (pred_label == i).sum()
            clusters_vars.append(var_ci)

        clusters_vars = torch.stack(clusters_vars)
        qp = 100 - p
        q = torch.quantile(clusters_vars, qp / 100)

        selected_clusters = (clusters_vars <= q).nonzero().squeeze()

        selection_mask = [pred in list(selected_clusters.cpu().numpy()) for pred in pred_label]
        indices_selection = torch.from_numpy(
            np.array(selection_mask)).nonzero().squeeze()

        return indices_selection

    def train(self):
        dataset = TabularDataset(self.params.data)
        data_loader = DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True,
                                 num_workers=self.params.num_workers)

        mean_loss = np.inf
        self.dm.update_train_set(self.dm.get_selected_indices())
        train_ldr = self.dm.get_train_set()
        REEVAL_LIMIT = 10
        X = []
        y = []
        indices = []
        for i, X_i in enumerate(train_ldr, 0):
            X.append(X_i[0])
            indices.append(X_i[2])
            y.append(X_i[1])

        X = torch.cat(X, axis=0)
        y = torch.cat(y, axis=0)

        indices = torch.cat(indices, axis=0)
        train_ldr = self.dm.get_train_set()

        L = []
        L_old = [-1]
        reev_count = 0
        while len(set(L_old).difference(set(L))) <= 10 and reev_count < REEVAL_LIMIT:
            for epoch in range(self.n_epochs):
                print(f"\nEpoch: {epoch + 1} of {self.n_epochs}")
                if (epoch + 1) % self.r == 0:
                    self.model.eval()
                    L_old = deepcopy(L)
                    with torch.no_grad():
                        print("\nRe-evaluation")
                        indices = []
                        Z = []
                        X = []
                        y = []
                        X_loader = self.dm.get_init_train_loader()
                        for i, X_i in enumerate(X_loader, 0):
                            indices.append(X_i[2])
                            train_inputs = X_i[0].to(self.device).float()
                            code, X_prime, Z_r = self.model(train_inputs)
                            Z_i = torch.cat([code, Z_r.unsqueeze(-1)], axis=1)
                            Z.append(Z_i)
                            X.append(X_i)
                            y.append(X_i[1])

                        # X = torch.cat(X, axis=0)
                        indices = torch.cat(indices, axis=0)
                        Z = torch.cat(Z, axis=0)
                        y = torch.cat(y, axis=0).cpu().numpy()

                        # plot_2D_latent(Z.cpu(), y)

                        selection_mask = self.re_evaluation(Z.cpu(), self.p, self.num_cluster)
                        selected_indices = indices[selection_mask]
                        y_s = y[selection_mask.cpu().numpy()]

                        print(f"selected label 0 ratio:{(y_s == 0).sum() / len(y)}"
                              f"")
                        print(f"selected label 1 ratio:{(y_s == 1).sum() / len(y)}"
                              f"")

                        self.dm.update_train_set(selected_indices)
                        train_ldr = self.dm.get_train_set()

                    # switch back to train mode
                    self.model.train()
                    L = selected_indices.cpu().numpy()
                    print(
                        f"Back to training--size L_old:{len(L_old)}, L:{len(L)}, "
                        f"diff:{len(set(L_old).difference(set(L)))}\n")

                else:
                    # Train with the current trainset
                    loss = 0
                    with trange(len(train_ldr)) as t:
                        for i, X_i in enumerate(train_ldr, 0):
                            train_inputs = X_i[0].to(self.device).float()
                            loss += self.train_iter(train_inputs)
                            mean_loss = loss / (i + 1)
                            t.set_postfix(loss='{:.3f}'.format(mean_loss))
                            t.update()
            print(f'Reeval  count:{reev_count}\n')
            reev_count += 1
            # self.evaluate_on_test_set()
            # break
        return mean_loss

    def train_iter(self, sample):

        code, X_prime, Z_r = self.model(sample)
        l2_z = (torch.cat([code, Z_r.unsqueeze(-1)], axis=1).norm(2, dim=1)).mean()
        reg = 0.5
        loss = ((sample - X_prime) ** 2).sum(axis=-1).mean() + reg * l2_z
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate_on_test_set(self, pos_label=1, **kwargs):
        energy_threshold = kwargs.get('threshold', 80)
        test_loader = self.dm.get_test_set()
        self.model.eval()
        train_score = []

        with torch.no_grad():
            test_score = []
            test_labels = []
            test_z = []

            for data in test_loader:
                test_inputs, label_inputs = data[0].float().to(self.device), data[1]
                code, X_prime, h_x = self.model(test_inputs)

                test_score.append(((test_inputs - X_prime) ** 2).sum(axis=-1).squeeze().cpu().numpy())
                test_z.append(code.cpu().numpy())
                test_labels.append(label_inputs.numpy())

            test_score = np.concatenate(test_score, axis=0)
            test_z = np.concatenate(test_z, axis=0)
            test_labels = np.concatenate(test_labels, axis=0)

            self.model.train()

            return test_score, test_labels

    def setDataManager(self, dm):
        self.dm = dm

class MySubset(Subset):

    def __getitem__(self, idx):
        data = self.dataset[self.indices[idx]]
        return data['data'], data['target'], idx


class DataManager:

    def __init__(self, train_dataset: torch.utils.data.Dataset,
                 test_dataset: torch.utils.data.Dataset,
                 batch_size: int,
                 num_classes: int = None,
                 input_shape: tuple = None,
                 validation: float = 0.,
                 seed: int = 0,
                 **kwargs):

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.train_set = train_dataset
        self.test_set = test_dataset
        self.anomaly_ratio = sum(test_dataset.dataset.y == 1)/len(test_dataset)
        self.validation = validation
        self.kwargs = kwargs
        self.seed = seed

        # torch.manual_seed(seed)
        n = len(train_dataset)
        shuffled_idx = torch.randperm(n).long()

        # Create a mask to track selection process
        self.train_selection_mask = torch.ones_like(shuffled_idx)

        self.current_train_set = MySubset(train_dataset, self.train_selection_mask.nonzero().squeeze())

        # Create the loaders
        train_sampler, val_sampler = self.train_validation_split(
            len(self.train_set), self.validation, self.seed
        )

        self.init_train_loader = DataLoader(self.current_train_set, self.batch_size, sampler=train_sampler,
                                            **self.kwargs)
        self.train_loader = DataLoader(self.current_train_set, self.batch_size, sampler=train_sampler, **self.kwargs)
        self.validation_loader = DataLoader(self.train_set, self.batch_size, sampler=val_sampler, **self.kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size, shuffle=True, **kwargs)

    def get_current_training_set(self):
        return self.current_train_set

    def get_init_train_loader(self):
        return self.init_train_loader

    def get_selected_indices(self):
        return self.train_selection_mask.nonzero().squeeze()

    def update_train_set(self, selected_indices):
        self.train_selection_mask[:] = 0
        self.train_selection_mask[selected_indices] = 1
        lbl_sample_idx = self.train_selection_mask.nonzero().squeeze()
        self.current_train_set = MySubset(self.train_set, lbl_sample_idx)
        train_sampler, val_sampler = self.train_validation_split(len(self.current_train_set), self.validation,
                                                                 self.seed)
        self.train_loader = DataLoader(self.current_train_set, self.batch_size, sampler=train_sampler, **self.kwargs)
        self.validation_loader = DataLoader(self.current_train_set, self.batch_size, sampler=val_sampler, **self.kwargs)
        return self.train_loader, self.validation_loader

    @staticmethod
    def train_validation_split(num_samples, validation_ratio, seed=0):
        # torch.manual_seed(seed)
        num_val = int(num_samples * validation_ratio)
        shuffled_idx = torch.randperm(num_samples).long()
        train_idx = shuffled_idx[num_val:]
        val_idx = shuffled_idx[:num_val]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        return train_sampler, val_sampler

    def get_train_set(self):
        return self.train_loader

    def get_validation_set(self):
        return self.validation_loader

    def get_test_set(self):
        return self.test_loader

    def get_classes(self):
        return range(self.num_classes)

    def get_input_shape(self):
        return self.input_shape

    def get_batch_size(self):
        return self.batch_size

    def get_random_sample_from_test_set(self):
        indice = np.random.randint(0, len(self.test_set))
        return self.test_set[indice]
