from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from models.base import BaseShallowModel


class IF(BaseShallowModel):

    def __init__(self, params):
        super(IF, self).__init__(params)
        self.clf = IsolationForest(n_estimators=50, contamination=params.contamination_rate, random_state=42)
        self.name = "IF"

    def get_params(self) -> dict:
        return {
            "n_estimators": self.clf.n_estimators
        }


class OCSVM(BaseShallowModel):
    def __init__(self, params):
        super(OCSVM, self).__init__(params)
        self.clf = OneClassSVM(
            kernel="rbf",
            gamma="scale",
            shrinking=False,
            nu=0.1
        )
        self.name = "OCSVM"

    def get_params(self) -> dict:
        return {
            "kernel": self.clf.kernel,
            "gamma": self.clf.gamma,
            "shrinking": self.clf.shrinking,
            "nu": self.clf.nu
        }


class LOF(BaseShallowModel):
    def __init__(self, params):
        super(LOF, self).__init__(params)
        self.clf = LocalOutlierFactor(novelty=params.novelty, contamination=params.contamination_rate, n_neighbors=20, n_jobs=-1)
        self.name = "LOF"

    def get_params(self) -> dict:
        return {
            "n_neighbors": self.clf.n_neighbors
        }


