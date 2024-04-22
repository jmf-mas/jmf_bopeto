from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from models.base import BaseSKModel


class LOF(BaseSKModel):

    def __init__(self, params):
        self.params = params
        super(LOF, self).__init__(params)
        model = LocalOutlierFactor(n_neighbors=20, novelty=True)

class IF(BaseSKModel):

    def __init__(self, params):
        self.params = params
        super(IF, self).__init__(params)
        model = IsolationForest(n_estimators=50, random_state=42)

class OCSVM(BaseSKModel):

    def __init__(self, params):
        self.params = params
        super(OCSVM, self).__init__(params)
        model = OneClassSVM(kernel='rbf', nu=0.1)


