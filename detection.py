import argparse
import numpy as np
import pandas as pd
import torch
from models.ae import AE
from models.dagmm import DaGMM
from params import ParamsAE, ParamsDaGMM
from trainer.trainer import TrainerAE, TrainerSK, TrainerDaGMM
from utils import estimate_optimal_threshold, compute_metrics, compute_metrics_binary
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

outputs = "outputs/"


def get_model(model):
    if model.upper()=="AE":
        backup = AE(params.val.shape[1]-1, name, 0.0)
        model_name = "AE"
        backup.load()
    elif model.upper()=="OC-SVM":
        backup = OneClassSVM(kernel='rbf', nu=0.1)
        model_name = "OC-SVM"
    elif model.upper()=="LOF":
        backup = LocalOutlierFactor(n_neighbors=20, novelty=True)
        model_name = "LOF"
    elif model.upper()=="IF":
        backup = IsolationForest(n_estimators=100, random_state=42)
        model_name = "IF"
    else:
        backup = DaGMM(name, params.val.shape[1]-1, params.gmm_k)
        if torch.cuda.is_available():
            backup.cuda()
        backup.load()
        model_name = "DAGMM"

    return backup, model_name

def get_contamination(key, model_name):
    if "bopeto" in key:
        model_name_ = "Bopeto_" + model_name
    else:
        model_name_ = model_name

    contamination = 0
    splits = key.split("_")
    if len(splits) >= 3:
        cont = splits[-1]
        contamination = float("." + cont.split(".")[1])
    return contamination, model_name_

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OoD detection",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch_size', nargs='?', const=1, type=int, default=64)
    parser.add_argument('-l', '--learning_rate', nargs='?', const=1, type=float, default=1e-3)
    parser.add_argument('-w', '--weight_decay', nargs='?', const=1, type=float, default=1e-3)
    parser.add_argument('-a', '--alpha', nargs='?', const=1, type=float, default=0.3)
    parser.add_argument('-e', '--epochs', nargs='?', const=1, type=int, default=20)
    parser.add_argument('-n', '--num_workers', nargs='?', const=1, type=int, default=4)
    parser.add_argument('--name', type=str, default='kdd', help='data set name')
    parser.add_argument('--model', type=str, default='AE', help='model name')

    #DaGMM
    parser.add_argument('--gmm_k', type=int, default=4)
    parser.add_argument('--lambda_energy', type=float, default=0.1)
    parser.add_argument('--lambda_cov_diag', type=float, default=0.005)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=194)
    parser.add_argument('--model_save_step', type=int, default=194)

    args = parser.parse_args()
    configs = vars(args)
    batch_size = configs['batch_size']
    lr = configs['learning_rate']
    wd = configs['weight_decay']
    nw = configs['num_workers']
    alpha = configs['alpha']
    epochs = configs['epochs']
    name = configs['name']

    gmm_k = configs['gmm_k']
    lambda_energy = configs['lambda_energy']
    lambda_cov_diag = configs['lambda_cov_diag']
    log_step = configs['log_step']
    sample_step = configs['sample_step']
    model_save_step = configs['model_save_step']


    data = np.load("detection/"+name+".npz", allow_pickle=True)
    keys = list(data.keys())
    filter_keys = list(filter(lambda s: "train" in s, keys))
    if configs['model']=="DAGMM":
        params = ParamsDaGMM(batch_size, lr, epochs, nw, gmm_k, lambda_energy, lambda_cov_diag, log_step, sample_step, model_save_step)
        backup = DaGMM(name, data[name + "_val"].shape[1] - 1, params.gmm_k)
        if torch.cuda.is_available():
            backup.cuda()
        backup.save()
    else:
        params = ParamsAE(batch_size, lr, wd, nw, epochs)
    params.test = data[name+"_test"]
    params.val = data[name + "_val"]
    performances = pd.DataFrame([], columns=["dataset", "contamination", "model", "accuracy","precision", "recall", "f1"])
    backup = None
    for key in filter_keys:
        print("training on "+key)
        params.model, model_name = get_model(configs['model'])
        params.data = data[key]
        if model_name == "AE":
            trainer = TrainerAE(params)
            trainer.run()
            y_val = params.val[:, -1]
            y_test = params.test[:, -1]
            threshold = estimate_optimal_threshold(trainer.params.val_scores, y_val, pos_label=1, nq=100)
            threshold = threshold["Thresh_star"]
            metrics = compute_metrics(trainer.params.test_scores, y_test, threshold, pos_label=1)

            contamination, model_name_ = get_contamination(key, model_name)
            perf = [name, contamination, model_name_, metrics[0], metrics[1], metrics[2], metrics[3]]
            performances.loc[len(performances)] = perf
            print("performance on", key, metrics[:4])
        elif model_name == "DAGMM":
            trainer = TrainerDaGMM(params)
            trainer.run()
        else:
            trainer = TrainerSK(params)
            trainer.run()
            y_test = params.test[:, -1]
            metrics = compute_metrics_binary(trainer.params.y_pred, y_test, pos_label=1)
            contamination, model_name_ = get_contamination(key, model_name)
            perf = [name, contamination, model_name_, metrics[0], metrics[1], metrics[2], metrics[3]]
            performances.loc[len(performances)] = perf
            print("performance on", key, metrics[:4])

    performances.to_csv("outputs/performances_"+name+"_"+model_name+".csv", header=True, index=False)












