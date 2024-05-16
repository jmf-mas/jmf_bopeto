import numpy as np
import pandas as pd
from copy import deepcopy
from models.bopeto import BOPETO
from trainer.split import Splitter
from utils.utils import Utils, contamination, compute_metrics_binary, estimate_optimal_threshold, compute_metrics, \
    find_match, get_contamination, resolve_model_trainer
import logging

np.random.seed(42)


def mode_cleaning(params):
    logging.basicConfig(filename=params.directory_log+'/cleaning.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    params.model_name = "AECleaning"
    splitter = Splitter(params.dataset_name, params.directory_data)

    data = splitter.split()
    rates = np.linspace(0, 1, params.num_contamination_subsets)
    cleaning = []
    in_dist = data[params.dataset_name + "_train"]
    params.in_features = in_dist.shape[1] - 1
    n_out = len(data[params.dataset_name + "_contamination"])
    params.init_model(load=False)
    utils = Utils(params)
    n_cases = len(rates)
    for i, rate in enumerate(rates):
        cont = 0
        try:
            print("{}/{}: cleaning on {}".format(i + 1, n_cases, params.dataset_name))
            ind = np.arange(int(rate * n_out))
            oo_dist = data[params.dataset_name + "_contamination"][ind]
            utils.params.data = utils.contaminate(in_dist, oo_dist)
            utils.params.weights = np.ones(utils.params.data.shape[0])
            utils.params.set_model()
            # training
            print("sub-optimal training ...")
            _ = utils.initial_train()
            before = contamination(utils.params.data)
            cont = before[1]
            print("synthetic data generation ...")
            n = len(utils.params.data)
            m = int(np.ceil(0.1 * n))
            utils.params.fragment = pd.DataFrame(utils.params.data).sample(m).values
            synthetic = utils.generate_synthetic_data()
            utils.params.update_data(synthetic)
            utils.params.weights = np.ones(utils.params.data.shape[0])
            np.savetxt(utils.params.dataset_name + '.csv', utils.params.data, delimiter=',')
            cond = utils.params.data[:, -1] != 2
            data[params.dataset_name + "_train_contamination_" + str(before[1])] = utils.params.data[cond]
            # filtering
            y = utils.params.data[:, -1]
            print("getting dynamics ...")
            utils.params.set_model()
            dynamics = utils.get_reconstruction_errors()
            utils.params.dynamics = np.column_stack((dynamics, y))
            np.savetxt(params.directory_output + utils.params.model.name + '.csv', utils.params.dynamics, delimiter=',')
            dynamics = np.loadtxt(params.directory_output + utils.params.model.name + '.csv', delimiter=',')
            utils.params.dynamics = dynamics
            b = BOPETO(utils.params)
            weights, indices = b.refine()
            refined_data = utils.params.data[indices]
            after = contamination(refined_data)
            data[params.dataset_name + "_train_bopeto_" + str(before[1])] = weights
            cleaning.append([before[0], after[0], before[1], after[1]])
            print("contamination from ", before[1], "to", after[1], "--size from", before[0], "to", after[0])
        except RuntimeError as e:
            logging.error(
                "Error for Bopeto cleaning on {} and contamination rate {}: {} ...".format(
                    params.dataset_name, cont, e))
        except Exception as e:
            logging.error(
                "Bopeto cleaning on {} and contamination rate {} unfinished caused by {} ...".format(
                    params.dataset_name, cont, e))

    db = pd.DataFrame(data=cleaning, columns=['n1', 'n2', 'r1', 'r2'])
    db.to_csv(params.directory_output + utils.params.dataset_name + ".csv", index=False)
    np.savez(params.directory_detection + utils.params.dataset_name + ".npz", **data)


def mode_b_iad(params):
    logging.basicConfig(filename=params.directory_log+'/bopeto.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    data = np.load(params.directory_detection + params.dataset_name + ".npz", allow_pickle=True)
    keys = list(data.keys())
    filter_keys = list(filter(lambda s: "train_" in s, keys))
    params.test = data[params.dataset_name + "_test"]
    params.val = data[params.dataset_name + "_val"]
    params.in_features = params.val.shape[1] - 1
    performances = pd.DataFrame([], columns=["dataset", "contamination", "model", "accuracy", "precision", "recall", "f1"])
    params.data = data[filter_keys[0]]

    tr, mo = resolve_model_trainer(params.model_trainer_map, params.model_name)
    mo = mo(params)
    params.model = mo
    tr = tr(params)
    if params.mode != "iad":
        params.T = 1
    n_cases = len(filter_keys)

    print("running mode: {} and cleaning mode {}".format(params.mode, params.cleaning))
    for i, key in enumerate(filter_keys):
        try:
            print("{}/{}: training on {}".format(i + 1, n_cases, key))
            model = deepcopy(mo)
            model.params.data = data[key]
            trainer = deepcopy(tr)
            contamination, model_name_ = get_contamination(key, params.model_name)
            if "contamination" in key:
                trainer.params.data = data[key]
                trainer.params.weights = np.ones(trainer.params.data.shape[0])
            else:
                match = find_match(filter_keys, contamination)
                if not match:
                    break
                trainer.params.data = data[match]
                weights = data[key]
                if trainer.params.cleaning == "hard":
                    weight = weights[:, 0]
                    trainer.params.data = trainer.params.data[weight == 1]
                    trainer.params.weights = np.ones(trainer.params.data.shape[0])
                else:
                    trainer.params.weights = weights[:, 1]

            trainer.params.model = model
            trainer.train()

            if trainer.name == "shallow":
                X, y_test = params.test[:, :-1], params.test[:, -1]
                y_pred = trainer.test(X)
                metrics = compute_metrics_binary(y_pred, y_test, pos_label=1)
            else:
                y_val, score_val = trainer.test(params.val)
                y_test, score_test = trainer.test(params.test)
                threshold = estimate_optimal_threshold(score_val, y_val, pos_label=1, nq=100)
                threshold = threshold["Thresh_star"]
                metrics = compute_metrics(score_test, y_test, threshold, pos_label=1)

            perf = [params.dataset_name, contamination, model_name_, metrics[0], metrics[1], metrics[2], metrics[3]]
            performances.loc[len(performances)] = perf
            print("performance on", key, metrics[:4])
        except RuntimeError as e:
            logging.error(
                "OoD detection on {} with {} and contamination rate {} unfinished caused by {} ...".format(
                    params.dataset_name,
                    params.model_name,
                    contamination, e))
        except Exception as e:
            logging.error(
                "Error for OoD detection on {} with {} and contamination rate {}: {} ...".format(
                    params.dataset_name,
                    params.model_name,
                    contamination, e))
    perf_path = params.directory_output + "/performances_" + params.cleaning + "_" + params.mode + "_" + params.dataset_name + "_" + params.model_name + ".csv"
    performances.to_csv(perf_path, header=True, index=False)












