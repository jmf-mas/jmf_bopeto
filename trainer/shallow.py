import numpy as np
import torch
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

torch.autograd.set_detect_anomaly(True)

class TrainerSK:

    def __init__(self, params):
        self.params = params
        self.name = "sklearn"
        if params.model == "ocsvm":
            self.model = OneClassSVM(kernel='rbf', nu=0.1)
        elif params.model == "lof":
            self.model = LocalOutlierFactor(n_neighbors=20, novelty=True)
        else:
            self.model = IsolationForest(n_estimators=50, random_state=42)

    def train(self):
        self.model.fit(self.params.data[:, :-1])
    def test(self, data):
        y_pred = self.model.predict(data)
        y_pred = np.where(y_pred == 1, 0, y_pred)
        self.params.y_pred = np.where(y_pred == -1, 1, y_pred)
