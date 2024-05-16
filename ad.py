import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from models.duad import DUAD
from models.neutralad import NeuTraLAD
from models.shallow import IF, LOF, OCSVM
from trainer.duad import TrainerDUAD
from trainer.neutralad import TrainerNeuTraLAD
from selector import mode_cleaning, mode_b_iad
from utils.params import Params
from trainer.ae import TrainerAE
from trainer.base import TrainerBaseShallow
from trainer.dagmm import TrainerDAGMM
from trainer.dsebm import TrainerDSEBM
from trainer.alad import TrainerALAD
from trainer.svdd import TrainerSVDD
from models.svdd import DeepSVDD
from models.alad import ALAD
from models.dsebm import DSEBM
from models.ae import AEDetecting, AECleaning
from models.dagmm import DAGMM
import logging

np.random.seed(42)


directory_model = "b_checkpoints/"
directory_data = "b_data/"
directory_output = "b_outputs/"
directory_log = "b_logs/"
directory_detection = "b_detection/"


Path(directory_model).mkdir(parents=True, exist_ok=True)
Path(directory_data).mkdir(parents=True, exist_ok=True)
Path(directory_output).mkdir(parents=True, exist_ok=True)
Path(directory_log).mkdir(parents=True, exist_ok=True)
Path(directory_detection).mkdir(parents=True, exist_ok=True)


logging.basicConfig(filename=directory_log+'/robustness.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_bopeto_map = {
    "alad": (TrainerALAD, ALAD),
    "dagmm": (TrainerDAGMM, DAGMM),
    "dsebm": (TrainerDSEBM, DSEBM),
    "if": (TrainerBaseShallow, IF),
    "lof": (TrainerBaseShallow, LOF),
    "ae": (TrainerAE, AEDetecting),
    "svdd": (TrainerSVDD, DeepSVDD),
}

model_iad_map = {
    "ae": (TrainerAE, AEDetecting),
    "svdd": (TrainerSVDD, DeepSVDD),
}

model_cleaning_map = {
    "ae": (TrainerAE, AECleaning)
}

model_impact_map = {
    "neutralad": (TrainerNeuTraLAD, NeuTraLAD),
    "duad": (TrainerDUAD, DUAD),
    "ocsvm": (TrainerBaseShallow, OCSVM),
    "if": (TrainerBaseShallow, IF),
    "lof": (TrainerBaseShallow, LOF)
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OoD detection", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch_size', nargs='?', const=1, type=int, default=64)
    parser.add_argument('-l', '--learning_rate', nargs='?', const=1, type=float, default=1e-3)
    parser.add_argument('-w', '--weight_decay', nargs='?', const=1, type=float, default=1e-3)
    parser.add_argument('-a', '--alpha', nargs='?', const=1, type=float, default=0.3)
    parser.add_argument('-e', '--epochs', nargs='?', const=1, type=int, default=100)
    parser.add_argument('-n', '--num_workers', nargs='?', const=1, type=int, default=4)
    parser.add_argument('--dataset', type=str, default='nsl', help='data set name')
    parser.add_argument('--model', type=str, default='svdd', help='model name')
    parser.add_argument('--cleaning', type=str, default='hard', help='type of cleaning (hard or soft)')
    parser.add_argument('--mode', type=str, default='bopeto', help='running modes: cleaning, bopeto or iad')
    parser.add_argument('-c', '--num_contamination_subsets', nargs='?', const=1, type=int, default=10)

    #DaGMM
    parser.add_argument('--gmm_k', type=int, default=4)
    parser.add_argument('--lambda_energy', type=float, default=0.1)
    parser.add_argument('--lambda_cov_diag', type=float, default=0.005)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=194)
    parser.add_argument('--model_save_step', type=int, default=194)

    parser.add_argument(
        '--n-runs',
        help='number of runs of the experiment',
        type=int,
        default=1
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help="The learning rate"
    )
    parser.add_argument(
        '--test_pct',
        type=float,
        default=0.5,
        help="The percentage of normal data used for training"
    )

    parser.add_argument(
        '--patience',
        type=float,
        default=5,
        help='Early stopping patience')

    parser.add_argument(
        "--pct",
        type=float,
        default=1.0,
        help="Percentage of original data to keep"
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Ratio of validation set from the training set"
    )

    parser.add_argument(
        "--hold_out",
        type=float,
        default=0.0,
        help="Percentage of anomalous data to holdout for possible contamination of the training set"
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.0,
        help="Anomaly ratio within training set"
    )

    parser.add_argument('--drop_lastbatch', dest='drop_lastbatch', action='store_true')
    parser.add_argument('--no-drop_lastbatch', dest='drop_lastbatch', action='store_false')
    parser.set_defaults(drop_lastbatch=False)

    # Robustness parameters
    parser.add_argument('--rob', dest='rob', action='store_true')
    parser.add_argument('--no-rob', dest='rob', action='store_false')
    parser.set_defaults(rob=False)

    parser.add_argument('--rob-sup', dest='rob_sup', action='store_true')
    parser.add_argument('--no-rob-sup', dest='rob_sup', action='store_false')
    parser.set_defaults(rob_sup=False)

    parser.add_argument('--rob-reg', dest='rob_reg', action='store_true')
    parser.add_argument('--no-rob-reg', dest='rob_reg', action='store_false')
    parser.set_defaults(rob_reg=False)

    parser.add_argument('--eval-test', dest='eval_test', action='store_true')
    parser.add_argument('--no-eval-test', dest='eval_test', action='store_false')
    parser.set_defaults(eval_test=False)

    parser.add_argument('--early_stopping', dest='early_stopping', action='store_true')
    parser.add_argument('--no-early_stopping', dest='early_stopping', action='store_false')
    parser.set_defaults(early_stopping=True)

    parser.add_argument(
        '--rob_method',
        type=str,
        choices=['refine', 'loe', 'our', 'sup'],
        default='daecd',
        help='methods used, either blind, refine, loe, daecd'
    )

    parser.add_argument(
        '--alpha-off-set',
        type=int,
        default=0.0,
        help='values between o and 1 used to offset the true value of the contamination ratio'
    )

    parser.add_argument(
        '--reg_n',
        type=float,
        default=1e-3,
        help='regulizer factor for the latent representation norm  '
    )

    parser.add_argument(
        '--reg_a',
        type=float,
        default=1e-3,
        help='regulizer factor for the anomalies loss'
    )

    parser.add_argument(
        '--num_clusters',
        type=int,
        default=3,
        help='number of clusters'
    )

    args = parser.parse_args()
    configs = vars(args)

    gmm_k = configs['gmm_k']
    lambda_energy = configs['lambda_energy']
    lambda_cov_diag = configs['lambda_cov_diag']
    log_step = configs['log_step']
    sample_step = configs['sample_step']
    model_save_step = configs['model_save_step']

    params = Params()
    params.patience = configs['patience']
    params.learning_rate = configs['learning_rate']
    params.weight_decay = configs['weight_decay']
    params.batch_size = configs['batch_size']
    params.num_workers = configs['num_workers']
    params.alpha = configs['alpha']
    params.epochs = configs['epochs']
    params.model_name = configs['model']
    params.early_stopping = configs['early_stopping']
    params.dataset_name = configs['dataset']
    params.cleaning = configs['cleaning']
    params.mode = configs['mode']
    params.num_contamination_subsets = configs['num_contamination_subsets']
    params.directory_model = directory_model
    params.directory_data = directory_data
    params.directory_output = directory_output
    params.directory_log = directory_log
    params.directory_detection = directory_detection

    if params.mode == "bopeto":
        params.model_trainer_map = model_bopeto_map
    else:
        params.model_trainer_map = model_iad_map

    if params.mode == "cleaning":
        mode_cleaning(params)
    else:
        mode_b_iad(params)















