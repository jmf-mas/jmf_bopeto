import argparse
from utils.params import Params
from models.bopeto import BOPETO
from utils.utils import Utils
from utils.utils import contamination
import numpy as np
import pandas as pd
from trainer.split import Splitter
import logging

logging.basicConfig(filename='logs/bopeto.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
np.random.seed(42)

outputs = "outputs/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bopeto",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--metrics', metavar='N', type=str, nargs='+', help='list of dynamics metrics', default=['sdc'])
    parser.add_argument('-b', '--batch_size', nargs='?', const=1, type=int, default=64)
    parser.add_argument('-l', '--learning_rate', nargs='?', const=1, type=float, default=1e-6)
    parser.add_argument('-w', '--weight_decay', nargs='?', const=1, type=float, default=1e-3)
    parser.add_argument('-a', '--alpha', nargs='?', const=1, type=float, default=0.3)
    parser.add_argument('-g', '--gamma', nargs='?', const=1, type=float, default=0.1)
    parser.add_argument('-e', '--epochs', nargs='?', const=1, type=int, default=10)
    parser.add_argument('-n', '--num_workers', nargs='?', const=1, type=int, default=4)
    parser.add_argument('-c', '--num_contamination_subsets', nargs='?', const=1, type=int, default=10)
    parser.add_argument('--dataset', type=str, default='kdd', help='data set name')
    parser.add_argument('--synthetic', type=str, default='FGM', help='path to data sets to use')

    args = parser.parse_args()
    configs = vars(args)
    params = Params()
    params.dataset_name = configs['dataset']
    params.batch_size = configs['batch_size']
    params.learning_rate = configs['learning_rate']
    params.weight_decay = configs['weight_decay']
    params.num_workers = configs['num_workers']
    params.alpha = configs['alpha']
    params.gamma = configs['gamma']
    params.epochs = configs['epochs']
    params.synthetic  = configs['synthetic']
    params.num_contamination_subsets = configs['num_contamination_subsets']
    params.metric = 'sdc'
    params.model_name = "AECleaning"
    splitter = Splitter(params.dataset_name)

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
            ind = np.arange(int(rate*n_out))
            oo_dist = data[params.dataset_name + "_contamination"][ind]
            utils.params.data = utils.contaminate(in_dist, oo_dist)
            utils.params.weights = np.ones(utils.params.data.shape[0])
            utils.params.set_model()
            #training
            print("initial training")
            _ = utils.initial_train()
            before = contamination(utils.params.data)
            print("before", before)
            cont = before[1]
            data[params.dataset_name + "_train_contamination_" + str(before[1])] = utils.params.data
            print("synthetic data generation")
            N = len(utils.params.data)
            M = int(np.ceil(utils.params.gamma * N))
            utils.params.fragment = pd.DataFrame(utils.params.data).sample(M).values
            synthetic = utils.generate_synthetic_data()
            utils.params.update_data(synthetic)
            utils.params.weights = np.ones(utils.params.data.shape[0])
            np.savetxt(utils.params.dataset_name + '.csv', utils.params.data, delimiter=',')
            # filtering
            y = utils.params.data[:, -1]
            print("training for dynamics")
            utils.params.set_model()
            dynamics = utils.get_reconstruction_errors()
            utils.params.dynamics = np.column_stack((dynamics, y))
            np.savetxt(outputs+utils.params.model.name+'.csv', utils.params.dynamics, delimiter=',')
            dynamics = np.loadtxt(outputs+utils.params.model.name+'.csv', delimiter=',')
            utils.params.dynamics = dynamics
            utils.params.update_metric(params.metric)
            b = BOPETO(utils.params)
            weights, indices = b.refine()
            refined_data = utils.params.data[indices]
            after = contamination(refined_data)
            print("after with", params.metric, after)
            data[params.dataset_name + "_train_bopeto_" +params.metric+"_"+ str(before[1])] = weights
            cleaning.append([params.metric, before[0], after[0], before[1], after[1]])
        except RuntimeError as e:
            logging.error(
                "Error for Bopeto cleaning on {} and contamination rate {}: {} ...".format(
                    params.dataset_name, cont, e))
        except Exception as e:
            logging.error(
                "Bopeto cleaning on {} and contamination rate {} unfinished caused by {} ...".format(
                    params.dataset_name,cont, e))

    name = utils.params.dataset_name+"_"+utils.params.synthetic
    db = pd.DataFrame(data=cleaning, columns=['utils', 'n1', 'n2', 'r1', 'r2'])
    db.to_csv(outputs + name+".csv", index=False)
    np.savez("detection/"+utils.params.dataset_name+".npz", **data)






