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
        self.metric_hist = []
        self.name = "duad"
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.learning_rate,weight_decay=params.weight_decay)
        self.criterion = nn.MSELoss()
        self.dm = DataManager(params)


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
        mean_loss = np.inf
        self.dm.update_train_set(self.dm.get_selected_indices())
        train_ldr = self.dm.get_train_set()
        REEVAL_LIMIT = 10
        X = []
        y = []
        indices = []
        for i, batch in enumerate(train_ldr):
            data = batch[0]
            targets = batch[1]
            weights = batch[2]
            index = batch[3]
            X.append(data)
            indices.append(index)
            y.append(targets)
        X = torch.cat(X, axis=0)
        y = torch.cat(y, axis=0)

        indices = torch.cat(indices, axis=0)
        train_ldr = self.dm.get_train_set()

        L = []
        L_old = [-1]
        reev_count = 0
        while len(set(L_old).difference(set(L))) <= 10 and reev_count < REEVAL_LIMIT:
            for epoch in range(self.params.epochs):
                if (epoch + 1) % self.params.r == 0:
                    self.model.eval()
                    L_old = deepcopy(L)
                    with torch.no_grad():
                        indices = []
                        Z = []
                        X = []
                        y = []
                        X_loader = self.dm.get_init_train_loader()
                        for i, X_i in enumerate(X_loader, 0):
                            indices.append(X_i[2])
                            train_inputs = X_i[0].to(self.params.device).float()
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

                        selection_mask = self.re_evaluation(Z.cpu(), self.params.p, self.params.num_cluster)
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
                            train_inputs = X_i[0].to(self.params.device).float()
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

    def test(self, data):
        self.model.eval()
        test_set = TabularDataset(data, np.ones(data.shape[0]))
        test_loader = DataLoader(test_set, batch_size=self.params.batch_size, shuffle=True,
                                 num_workers=self.params.num_workers)

        with torch.no_grad():
            scores = []
            y_true = []

            for row in test_loader:
                X = row['data'].to(self.params.device)
                y = row['target'].to(self.params.device)
                code, X_prime, h_x = self.model(X)

                scores.append(((X - X_prime) ** 2).sum(axis=-1).squeeze().cpu().numpy())
                y_true.append(y.cpu().numpy())

            scores = np.concatenate(scores, axis=0)
            y_true = np.concatenate(y_true, axis=0)

            return y_true, scores

class MySubset(Subset):

    def __getitem__(self, idx):
        data = self.dataset[self.indices[idx]]
        return data['data'], data['target'], data['weight'], idx


class DataManager:

    def __init__(self, params):

        self.params = params
        self.train_set = TabularDataset(self.params.data, np.ones(self.params.data.shape[0]))
       # print("jordan = ", training_data)
        #self.train_set = DataLoader(training_data, batch_size=self.params.batch_size, shuffle=True,
         #                         num_workers=self.params.num_workers)
        self.test_set = TabularDataset(self.params.val, np.ones(self.params.val.shape[0]))
        #self.test_set = DataLoader(val_data, batch_size=self.params.batch_size, shuffle=True,
         #                       num_workers=self.params.num_workers)

        self.batch_size = params.batch_size
        self.num_classes = params.num_cluster
        self.input_shape = (params.data.shape[0], self.params.in_features)
        self.anomaly_ratio = params.contamination_rate
        self.validation = params.validation
        self.seed = params.seed
        n = len(self.train_set)
        shuffled_idx = torch.randperm(n).long()

        self.train_selection_mask = torch.ones_like(shuffled_idx)

        self.current_train_set = MySubset(self.train_set, self.train_selection_mask.nonzero().squeeze())

        train_sampler, val_sampler = self.train_validation_split(
            len(self.train_set), self.validation)

        self.init_train_loader = DataLoader(self.current_train_set, self.batch_size, sampler=train_sampler)
        self.train_loader = DataLoader(self.current_train_set, self.batch_size, sampler=train_sampler)
        self.validation_loader = DataLoader(self.train_set, self.batch_size, sampler=val_sampler)
        self.test_loader = DataLoader(self.test_set, params.batch_size, shuffle=True)
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
        train_sampler, val_sampler = self.train_validation_split(len(self.current_train_set), self.validation)
        self.train_loader = DataLoader(self.current_train_set, self.batch_size, sampler=train_sampler)
        self.validation_loader = DataLoader(self.current_train_set, self.batch_size, sampler=val_sampler)
        return self.train_loader, self.validation_loader

    @staticmethod
    def train_validation_split(num_samples, validation_ratio):
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
