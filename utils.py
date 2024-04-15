from trainer.trainer import Trainer

class Utils:
    def __init__(self, params):
        self.params = params

    def train(self):
        trainer = Trainer(self.params)
        trainer.run()
