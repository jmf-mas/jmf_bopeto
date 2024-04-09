import numpy as np
import pandas as pd
from scipy.stats import variation as cv
from metric.metrics import Metrics

class BOPETO:


    def __init__(self, dynamics, beta = 0.25, phi_0 = 0.5, phi_1 = 1., phi_2 = 2., metric_id = 0):

        self.dynamics = dynamics
        self.beta = beta
        self.phi_0 = phi_0
        self.phi_1 = phi_1
        self.phi_2 = phi_2

        self.m = Metrics(dynamics)
        self.metric_id = metric_id


    def refine(self):

        std = np.mean(np.abs(np.diff(self.dynamics[:, :-1].T, axis=0)), axis=0)
        std = self.metric_selection(self.metric_id)
        n = len(std)
        iclass = ["synthetic" if self.dynamics[i, -1] == 2 else "training" for i in range(n)]
        dbframe = pd.DataFrame(data={'sample': range(n), 'std': std, "class": iclass})
        cv1 = cv(dbframe[(dbframe["class"] == "synthetic")]["std"].values)
        cv2 = cv(dbframe[(dbframe["class"] != "synthetic")]["std"].values)
        kappa = cv1 / cv2

        if kappa < self.phi_0 or kappa >= self.phi_2:
            indices = list(dbframe[(dbframe["class"] != "synthetic")].index)
        elif kappa >= self.phi_0 and kappa < self.phi_1:
            threshold = dbframe[(dbframe["class"] == "synthetic")]["std"].quantile(.2)
            indices = list(dbframe[(dbframe["std"] < threshold) & (dbframe["class"] != "synthetic")].index)
        else:
            threshold_mean = dbframe[(dbframe["class"] == "synthetic")]["std"].mean()
            threshold_std = dbframe[(dbframe["class"] == "synthetic")]["std"].std()
            threshold_min = threshold_mean - self.beta*threshold_std
            threshold_max = threshold_mean + self.beta*threshold_std
            indices = list(dbframe[((dbframe["std"] < threshold_min) | (dbframe["std"] > threshold_max)) & (dbframe["class"] != "synthetic")].index)

        return indices

    def metric_selection(self, i):
        if i == 0:
            return self.m.mac()
        elif i==1:
            return self.m.msc()
        elif i==2:
            return self.m.sdc()
        elif i==3:
            return self.m.pc()
        elif i==4:
            return self.m.rmac()
        elif i==5:
            return self.m.cv()
        else:
            return self.m.iqr()



