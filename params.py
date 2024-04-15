import os
import numpy as np
from models.ae import AE

class Params:

    def __init__(self, rate, batch_size, learning_rate, weight_decay, num_workers, alpha, gamma, momentum, epochs, path, metric, synthetic):

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
        print(os.getcwd())
        raw = np.load(path)
        key = list(raw.keys())[0]
        self.data = raw[key]
        print(self.dataset_name)

    def set_model(self, load=False):
        self.model = AE(self.data.shape[1]-1, "ae_rate_"+str(self.rate))
        if load:
            self.model.load()

