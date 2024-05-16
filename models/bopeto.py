import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class BOPETO:
    def __init__(self, params):
        self.params = params

    def sdc(self):
        return np.std(np.diff(self.params.dynamics, axis=1), axis=1)

    def refine(self):
        dynamics_scores = self.sdc()
        n = len(dynamics_scores)
        target = ["synthetic" if self.params.dynamics[i, -1] == 2 else "training" for i in range(n)]
        db = pd.DataFrame(data={'sample': range(n), "sdc": dynamics_scores, "class": target})
        values = db["sdc"].values.reshape(-1, 1)
        detector = IsolationForest(n_estimators=50, random_state=42)
        y_pred = detector.fit_predict(values)
        anomaly_scores = detector.decision_function(values)
        ood = anomaly_scores[y_pred == -1]
        in_ = anomaly_scores[y_pred == 1]
        threshold = (np.max(ood) + np.min(in_)) / 2
        threshold = np.percentile(ood, np.random.randint(60, 70, 1)[0])
        y_pred = anomaly_scores >= threshold
        indices = list(db[(y_pred == 1) & (db["class"] != "synthetic")].index)
        hard_weights = y_pred.astype(int).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        soft_weights = scaler.fit_transform(anomaly_scores.reshape(-1, 1))
        soft_weights[hard_weights == 1] = 1
        weights = np.hstack((hard_weights, soft_weights))
        training_indices = list(db[db["class"] != "synthetic"].index)
        return weights[training_indices], indices

