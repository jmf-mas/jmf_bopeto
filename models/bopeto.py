from plot.filtering import plot_segmented_one_line, plot_segmented_two_lines, plot_segmented
import pandas as pd
from scipy.stats import variation as cv
from metric.metrics import Metrics
from sklearn.cluster import KMeans
import numpy as np


class BOPETO:
    def __init__(self, params):

        self.params = params
        self.m = Metrics(self.params.dynamics)
        self.metric = getattr(self.m, params.metric, None)

    def refine(self, plot=False):
        std = self.metric()
        n = len(std)
        iclass = ["synthetic" if self.params.dynamics[i, -1] == 2 else "training" for i in range(n)]
        dbframe = pd.DataFrame(data={'sample': range(n), self.params.metric: std, "class": iclass})
        if plot:
            target = ["in" if self.params.dynamics[i, -1] == 0 else ("out" if self.params.dynamics[i, -1] == 1 else "synthetic") for i in range(n)]
            db = pd.DataFrame(data={'sample': range(n), self.params.metric: std, "class": target})
        values = dbframe[(dbframe["class"] != "synthetic")][self.params.metric].values.reshape(-1, 1)
        kappa = np.std(values)/np.max(values)
        if kappa < 0.12:
            # no ood detected
            indices = list(dbframe[dbframe["class"] != "synthetic"].index)
            thresh = 1.5*np.max(values)
        else:
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(values)
            centroids = kmeans.cluster_centers_
            index = np.argmin(centroids)
            y_predict = kmeans.predict(values)
            thresh_1 = np.max(values[y_predict==index])
            thresh_2 = np.min(values[y_predict != index])
            thresh = (thresh_1 + thresh_2)/2
            indices = list(dbframe[(dbframe[self.params.metric] <= thresh) & (dbframe["class"] != "synthetic")].index)
        if plot:
            plot_segmented_one_line(self.params.id, db, thresh, self.params.metric)
        return indices

    def refine_old(self, plot=False):
        std = self.metric()
        n = len(std)
        iclass = ["synthetic" if self.params.dynamics[i, -1] == 2 else "training" for i in range(n)]
        dbframe = pd.DataFrame(data={'sample': range(n), self.params.metric: std, "class": iclass})
        if plot:
            target = ["in" if self.params.dynamics[i, -1] == 0 else ("out" if self.params.dynamics[i, -1] == 1 else "synthetic") for i in range(n)]
            db = pd.DataFrame(data={'sample': range(n), self.params.metric: std, "class": target})
        cv1 = cv(dbframe[(dbframe["class"] == "synthetic")][self.params.metric].values)
        cv2 = cv(dbframe[(dbframe["class"] != "synthetic")][self.params.metric].values)

        kappa = cv1 / cv2
        print(cv1, cv2, kappa)

        if kappa < self.params.phi_0 or kappa >= self.params.phi_2:
            indices = list(dbframe[(dbframe["class"] != "synthetic")].index)
            if plot:
                plot_segmented(self.params.id, db, self.params.metric)
        elif kappa >= self.params.phi_0 and kappa < self.params.phi_1:
            threshold = dbframe[(dbframe["class"] == "synthetic")][self.params.metric].quantile(.2)
            indices = list(dbframe[(dbframe[self.params.metric] < threshold) & (dbframe["class"] != "synthetic")].index)
            if plot:
                plot_segmented_one_line(self.params.id, db, threshold, self.params.metric)
        else:
            threshold_mean = dbframe[(dbframe["class"] == "synthetic")][self.params.metric].mean()
            threshold_std = dbframe[(dbframe["class"] == "synthetic")][self.params.metric].std()
            threshold_min = threshold_mean - self.params.beta*threshold_std
            threshold_max = threshold_mean + self.params.beta*threshold_std
            indices = list(dbframe[((dbframe[self.params.metric] < threshold_min) | (dbframe[self.params.metric] > threshold_max)) & (dbframe["class"] != "synthetic")].index)
            if plot:
                plot_segmented_two_lines(self.params.id, db, threshold_min, threshold_max, self.params.metric)
        return indices




