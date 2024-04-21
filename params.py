import numpy as np
from models.ae import AE


class Params:

    def __init__(self, batch_size, learning_rate, weight_decay, num_workers, alpha, gamma, epochs, name, metric, synthetic):

        self.rate = 0
        self.id = None
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.gamma = gamma
        self.epochs = epochs
        self.dataset_name = name
        self.metric = metric
        self.data = None
        self.synthetic = synthetic
        self.model = None
        self.num_workers = num_workers
        self.dynamics = None
        self.fragment = None

    def set_model(self):
        self.id = self.dataset_name+"_"+self.synthetic + "_" +self.metric+"_ae_rate_"+str(self.rate)
        self.model = AE(self.data.shape[1]-1, self.dataset_name, 0.2)
        self.model.load()
        self.model.name = self.id

    def init_model(self, n, load=False):
        self.model = AE(n, self.dataset_name, 0.0)
        if load:
            self.model.load()
        self.model.save()


    def update_data(self, synthetic):
        y = [2]*len(synthetic)
        synthetic = np.column_stack((synthetic, y))
        self.data = np.vstack((self.data, synthetic))
        np.random.shuffle(self.data)

    def update_rate(self, rate):
        self.rate = rate

    def update_metric(self, metric):
        self.metric = metric

class ParamsAE:
    def __init__(self, batch_size, learning_rate, weight_decay, num_workers, epochs):
        self.id = None
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.data = None
        self.val = None
        self.test = None
        self.model = None
        self.num_workers = num_workers
        self.val_scores = None
        self.test_scores = None
        self.y_pred = None

class ParamsDaGMM:
    def __init__(self, batch_size, learning_rate, epochs, num_workers, gmm_k, lambda_energy, lambda_cov_diag, log_step, sample_step, model_save_step):
        self.id = None
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.epochs = epochs
        self.data = None
        self.val = None
        self.test = None
        self.model = None
        self.val_scores = None
        self.test_scores = None
        self.y_pred = None
        self.gmm_k = gmm_k
        self.lambda_energy = lambda_energy
        self.lambda_cov_diag = lambda_cov_diag
        self.log_step = log_step
        self.sample_step = sample_step
        self.model_save_step = model_save_step





