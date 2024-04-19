import argparse
from params import Params
from models.bopeto import BOPETO
from utils import Utils
from metric.metrics import contamination
import numpy as np
import pandas as pd

outputs = "outputs/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bopeto",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--contamination_rate', metavar='N', type=float, nargs='+', help='list of contamination rates', default=[0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    parser.add_argument('-m', '--metrics', metavar='N', type=str, nargs='+', help='list of dynamics metrics', default=['mac', 'sdc', 'msc', 'std'])
    parser.add_argument('-b', '--batch_size', nargs='?', const=1, type=int, default=64)
    parser.add_argument('-l', '--learning_rate', nargs='?', const=1, type=float, default=1e-3)
    parser.add_argument('-w', '--weight_decay', nargs='?', const=1, type=float, default=1e-3)
    parser.add_argument('-a', '--alpha', nargs='?', const=1, type=float, default=0.3)
    parser.add_argument('-g', '--gamma', nargs='?', const=1, type=float, default=0.1)
    parser.add_argument('-e', '--epochs', nargs='?', const=1, type=int, default=20)
    parser.add_argument('-n', '--num_workers', nargs='?', const=1, type=int, default=4)
    parser.add_argument('--path', type=str, default='/data', help='path to data sets to use')
    parser.add_argument('--synthetic', type=str, default='FGM', help='path to data sets to use')

    args = parser.parse_args()
    configs = vars(args)
    rates = configs['contamination_rate']
    metrics = configs['metrics']
    batch_size = configs['batch_size']
    lr = configs['learning_rate']
    wd = configs['weight_decay']
    nw = configs['num_workers']
    alpha = configs['alpha']
    gamma = configs['gamma']
    epochs = configs['epochs']
    path = configs['path']
    metric = 'mac'
    synthetic = configs['synthetic']

    params = Params(0, batch_size, lr, wd, nw, alpha, gamma, epochs, path, metric, synthetic)
    utils = Utils(params)
    in_dist, oo_dist = utils.data_split()
    cleaning = []
    for rate in rates:
        params.update_rate(rate)
        params.set_model(load=False)
        utils.update_params(params)
        #training
        utils.params.rate = rate
        utils.params.data = utils.contaminate(in_dist, oo_dist)
        print("initial training")
        _ = utils.initial_train()
        before = contamination(params.data)
        print("before", before)
        print("synthetic data generation")
        N = len(utils.params.data)
        M = int(np.ceil(utils.params.gamma * N))
        utils.params.fragment = pd.DataFrame(utils.params.data).sample(M).values
        synthetic = utils.generate_synthetic_data()
        utils.params.update_data(synthetic)
        np.savetxt(utils.params.dataset_name + '.csv', utils.params.data, delimiter=',')
        # filtering
        y = utils.params.data[:, -1]
        print("training for dynamics")
        utils.params.set_model(load=False)
        dynamics = utils.get_reconstruction_errors()
        utils.params.dynamics = np.column_stack((dynamics, y))
        np.savetxt(outputs+utils.params.model.name+'.csv', utils.params.dynamics, delimiter=',')

        dynamics = np.loadtxt(outputs+utils.params.model.name+'.csv', delimiter=',')
        utils.params.dynamics = dynamics
        for metric in metrics:
            utils.params.update_metric(metric)
            b = BOPETO(utils.params)
            indices = b.refine(True)
            refined_data = utils.params.data[indices]
            after = contamination(refined_data)
            print("after with", metric, after)
            cleaning.append([metric, before[0], after[0], before[1], after[1]])
    name = utils.params.dataset_name+"_"+utils.params.synthetic+".csv"
    db = pd.DataFrame(data=cleaning, columns=['metric', 'n1', 'n2', 'r1', 'r2'])
    db.to_csv(outputs + name, index=False)






