import os
import numpy as np
from models.ae import AE

class Params:

    def __init__(self, rate, batch_size, learning_rate, weight_decay, num_workers, alpha, gamma, momentum, epochs, path, metric, synthetic, beta):

        self.rate = rate
        self.id = None
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
        self.dynamics = None

    def set_model(self, load=False):
        self.id = self.dataset_name+"_"+self.synthetic + "_" +self.metric+"_ae_rate_"+str(self.rate)
        self.model = AE(self.data.shape[1]-1, self.id)
        if load:
            self.model.load()

    def update_data(self, synthetic):
        y = [2]*len(synthetic)
        synthetic = np.column_stack((synthetic, y))
        self.data = np.vstack((self.data, synthetic))
        np.random.shuffle(self.data)

    def update_rate(self, rate):
        self.rate = rate




