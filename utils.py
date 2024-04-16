from trainer.trainer import Trainer
import numpy as np
from synthetic.generation import JMF, FGM

class Utils:
    def __init__(self, params):
        self.params = params

    def get_reconstruction_errors(self):
        trainer = Trainer(self.params)
        errors = trainer.run()
        return errors

    def generate_synthetic_data(self):
        data = self.params.data[:, :-1]
        if self.params.synthetic == "JMF":
            generator = JMF(data, self.params.gamma)
        else:
            generator = FGM(data, self.params.gamma)
        return generator.generate()

    def data_split(self):
        data = self.params.data
        in_dist = data[data[:, -1] == 0]
        oo_dist = data[data[:, -1] == 1]
        return in_dist, oo_dist

    def contaminate(self, in_dist, oo_dist):
        m = int(self.params.rate*len(in_dist)/(1-self.params.rate))
        indices = [i for i in range(len(oo_dist))]
        selected_indices = np.random.choice(indices, size=m, replace=False)
        data = np.vstack((in_dist, oo_dist[selected_indices]))
        return data

    def update_params(self, params):
        self.params = params




