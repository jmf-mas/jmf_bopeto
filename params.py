import os
import numpy as np
from models.ae import AE

class Params:

    def __init__(self, rate, batch_size, learning_rate, weight_decay, num_workers, alpha, gamma, momentum, epochs, path, metric, synthetic, beta, phi_0, phi_1, phi_2):

        self.rate = rate
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.gamma = gamma
        self.momentum = momentum
        self.epochs = epochs
        self.path = path
        self.dataset_name = os.path.basename(path).split(".")[0]
        self.metric = metric
        self.synthetic = synthetic
        self.model = None
        self.num_workers = num_workers
        raw = np.load(path)
        key = list(raw.keys())[0]
        self.data = raw[key]
        self.beta = beta
        self.phi_0 = phi_0
        self.phi_1 = phi_1
        self.phi_2 = phi_2
        self.dynamics = None

    def set_model(self, load=False):
        self.model = AE(self.data.shape[1]-1, self.metric+"_ae_rate_"+str(self.rate))
        if load:
            self.model.load()

    def update_data(self, synthetic):
        y = [2]*len(synthetic)
        synthetic = np.column_stack((synthetic, y))
        self.data = np.vstack((self.data, synthetic))
        np.random.shuffle(self.data)




