import numpy as np
import pandas as pd
from scipy.stats import variation as cv
from metric.metrics import Metrics

class BOPETO:


    def __init__(self, params):

        self.params = params
        self.m = Metrics(self.params.dynamics)
        self.metric = getattr(self.m, params.metric, None)


    def refine(self):
        std = self.metric()
        print("shape", std.shape)
        n = len(std)
        iclass = ["synthetic" if self.params.dynamics[i, -1] == 2 else "training" for i in range(n)]
        dbframe = pd.DataFrame(data={'sample': range(n), 'std': std, "class": iclass})
        cv1 = cv(dbframe[(dbframe["class"] == "synthetic")]["std"].values)
        cv2 = cv(dbframe[(dbframe["class"] != "synthetic")]["std"].values)
        kappa = cv1 / cv2

        if kappa < self.params.phi_0 or kappa >= self.params.phi_2:
            indices = list(dbframe[(dbframe["class"] != "synthetic")].index)
        elif kappa >= self.params.phi_0 and kappa < self.params.phi_1:
            threshold = dbframe[(dbframe["class"] == "synthetic")]["std"].quantile(.2)
            indices = list(dbframe[(dbframe["std"] < threshold) & (dbframe["class"] != "synthetic")].index)
        else:
            threshold_mean = dbframe[(dbframe["class"] == "synthetic")]["std"].mean()
            threshold_std = dbframe[(dbframe["class"] == "synthetic")]["std"].std()
            threshold_min = threshold_mean - self.params.beta*threshold_std
            threshold_max = threshold_mean + self.params.beta*threshold_std
            indices = list(dbframe[((dbframe["std"] < threshold_min) | (dbframe["std"] > threshold_max)) & (dbframe["class"] != "synthetic")].index)
        return indices




