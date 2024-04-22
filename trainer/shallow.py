import numpy as np
import torch

torch.autograd.set_detect_anomaly(True)

class TrainerSK:

    def __init__(self, params):
        self.params = params
        self.name = "sklearn"
    def train(self):
        self.params.model.fit(self.params.data[:, :-1])
    def test(self, data):
        y_pred = self.params.model.predict(data)
        y_pred = np.where(y_pred == 1, 0, y_pred)
        self.params.y_pred = np.where(y_pred == -1, 1, y_pred)
