import argparse
from params import Params
from models.bopeto import BOPETO
from utils import Utils
from metric.metrics import contamination
import numpy as np
import pandas as pd
from trainer.split import Splitter

outputs = "outputs/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bopeto",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--metrics', metavar='N', type=str, nargs='+', help='list of dynamics metrics', default=['sdc'])
    parser.add_argument('-b', '--batch_size', nargs='?', const=1, type=int, default=64)
    parser.add_argument('-l', '--learning_rate', nargs='?', const=1, type=float, default=1e-3)
    parser.add_argument('-w', '--weight_decay', nargs='?', const=1, type=float, default=1e-3)
    parser.add_argument('-a', '--alpha', nargs='?', const=1, type=float, default=0.3)
    parser.add_argument('-g', '--gamma', nargs='?', const=1, type=float, default=0.1)
    parser.add_argument('-e', '--epochs', nargs='?', const=1, type=int, default=10)
    parser.add_argument('-n', '--num_workers', nargs='?', const=1, type=int, default=4)
    parser.add_argument('--name', type=str, default='kdd', help='data set name')
    parser.add_argument('--synthetic', type=str, default='FGM', help='path to data sets to use')

    args = parser.parse_args()
    configs = vars(args)
    params = Params()
    params.dataset_name = configs['name']
    params.batch_size = configs['batch_size']
    params.learning_rate = configs['learning_rate']
    params.weight_decay = configs['weight_decay']
    params.num_workers = configs['num_workers']
    params.alpha = configs['alpha']
    params.gamma = configs['gamma']
    params.epochs = configs['epochs']
    params.synthetic  = configs['synthetic']
    params.metric = 'sdc'
    splitter = Splitter(params.dataset_name)

    data = splitter.split()

    rates = np.linspace(0, 1, 11)[1:]
    cleaning = []
    in_dist = data[params.dataset_name + "_train"]
    params.in_features = in_dist.shape[1] - 1
    n_out = len(data[params.dataset_name + "_contamination"])
    params.init_model(load=False)
    utils = Utils(params)
    for rate in rates:
        ind = np.arange(int(rate*n_out))
        oo_dist = data[params.dataset_name + "_contamination"][ind]
        utils.params.data = utils.contaminate(in_dist, oo_dist)
        utils.params.set_model()
        #training
        print("initial training")
        _ = utils.initial_train()
        before = contamination(params.data)
        print("before", before)
        data[params.dataset_name + "_train_contamination_" + str(before[1])] = utils.params.data
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
        utils.params.set_model()
        dynamics = utils.get_reconstruction_errors()
        utils.params.dynamics = np.column_stack((dynamics, y))
        np.savetxt(outputs+utils.params.model.name+'.csv', utils.params.dynamics, delimiter=',')
        dynamics = np.loadtxt(outputs+utils.params.model.name+'.csv', delimiter=',')
        utils.params.dynamics = dynamics
        utils.params.update_metric(params.metric)
        b = BOPETO(utils.params)
        indices = b.refine(True)
        refined_data = utils.params.data[indices]
        after = contamination(refined_data)
        print("after with", params.metric, after)
        data[params.dataset_name + "_train_bopeto_" +params.metric+"_"+ str(before[1])] = refined_data
        cleaning.append([params.metric, before[0], after[0], before[1], after[1]])
    name = utils.params.dataset_name+"_"+utils.params.synthetic
    db = pd.DataFrame(data=cleaning, columns=['metric', 'n1', 'n2', 'r1', 'r2'])
    db.to_csv(outputs + name+".csv", index=False)
    np.savez("detection/"+utils.params.dataset_name+".npz", **data)






