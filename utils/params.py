import numpy as np
from models.ae import AECleaning
import torch

class Params:

    def __init__(self):
        self.i = 0
        self.rate = 0
        self.weights = None
        self.cleaning = "hard"
        self.num_contamination_subsets = 3
        self.patience = 10
        self.id = None
        self.batch_size = None
        self.learning_rate = None
        self.weight_decay = None
        self.alpha = None
        self.epochs = None
        self.dataset_name = None
        self.data = None
        self.model_name = None
        self.num_workers = None
        self.dynamics = None
        self.fragment = None
        self.val = None
        self.test = None
        self.val_scores = None
        self.test_scores = None
        self.y_pred = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.dropout = 0
        self.novelty = True
        # dagmm
        self.lambda_1 = 0.005
        self.lambda_2 =  0.1
        self.reg_covar = 0.01 #1e-12
        self.n_jobs_dataloader = 1
        self.early_stopping = True
        self.score_metric = "reconstruction"
        self.ae_latent_dim = 1
        self.in_features = None
        self.D = 8
        self.c = .8
        self.R = None
        # duad
        self.r = 10
        self.p0 = .35
        self.p = .30
        self.num_cluster = 20
        self.contamination_rate = 'auto'
        self.true_contamination_rate = .0
        self.drop_lastbatch = False
        self.validation = 0.
        self.seed = 0
        # neuralad
        self.n_layers = 3
        self.trans_type = 'res'
        self.temperature = 0.1
        self.rob = None
        self.warmup = 0.1
        self.rob_method = "loe"
        self.label = None

    def set_model(self):
        self.id = self.dataset_name+"_"+"_ae_rate_"+str(self.rate)
        self.model = AECleaning(self)
        self.model.load()
        self.model.name = self.id

    def init_model(self, load=False):
        self.model = AECleaning(self)
        if load:
            self.model.load()
        self.model.save()


    def update_data(self, synthetic):
        y = [2]*len(synthetic)
        synthetic = np.column_stack((synthetic, y))
        self.data = np.vstack((self.data, synthetic))
        np.random.shuffle(self.data)

    def update_rate(self, rate):
        self.rate = rate





