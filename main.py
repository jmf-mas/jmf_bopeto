import argparse
from params import Params
from models.bopeto import BOPETO
from utils import Utils
from metric.metrics import contamination
import numpy as np

outputs = "outputs/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bopeto",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--is_bopeto", action="store_true", help="training mode", default=False)
    parser.add_argument('-r', '--contamination_rate', metavar='N', type=float, nargs='+', help='list of contamination rates', default=[0.1])
    parser.add_argument('-b', '--batch_size', nargs='?', const=1, type=int, default=64)
    parser.add_argument('-l', '--learning_rate', nargs='?', const=1, type=float, default=1e-3)
    parser.add_argument('-w', '--weight_decay', nargs='?', const=1, type=float, default=1e-3)
    parser.add_argument('-a', '--alpha', nargs='?', const=1, type=float, default=0.3)
    parser.add_argument('-g', '--gamma', nargs='?', const=1, type=float, default=0.1)
    parser.add_argument('-m', '--momentum', nargs='?', const=1, type=float, default=0.9)
    parser.add_argument('-e', '--epochs', nargs='?', const=1, type=int, default=20)
    parser.add_argument('-n', '--num_workers', nargs='?', const=1, type=int, default=4)
    parser.add_argument('--path', type=str, default='/data', help='path to data sets to use')
    parser.add_argument('--metric', type=str, default='rmac', help='dynamics metrics')
    parser.add_argument('--synthetic', type=str, default='JMF', help='path to data sets to use')
    parser.add_argument('-b1', '--beta', nargs='?', const=1, type=float, default=0.25)
    parser.add_argument('-p0', '--phi_0', nargs='?', const=1, type=float, default=0.5)
    parser.add_argument('-p1', '--phi_1', nargs='?', const=1, type=float, default=1.)
    parser.add_argument('-p2', '--phi_2', nargs='?', const=1, type=float, default=2.)

    args = parser.parse_args()
    configs = vars(args)
    is_bopeto = configs['is_bopeto']
    rates = configs['contamination_rate']
    batch_size = configs['batch_size']
    lr = configs['learning_rate']
    wd = configs['weight_decay']
    nw = configs['num_workers']
    alpha = configs['alpha']
    gamma = configs['gamma']
    momentum = configs['momentum']
    epochs = configs['epochs']
    path = configs['path']
    metric = configs['metric']
    synthetic = configs['synthetic']
    beta = configs['beta']
    phi_0 = configs['phi_0']
    phi_1 = configs['phi_1']
    phi_2 = configs['phi_2']

    params = Params(0, batch_size, lr, wd, nw, alpha, gamma, momentum, epochs, path, metric, synthetic, beta, phi_0,
                    phi_1, phi_2)
    utils = Utils(params)
    in_dist, oo_dist = utils.data_split()
    cleaning = {}
    for rate in rates:
        params.update_rate(rate)
        params.set_model(load=False)
        utils.update_params(params)
        #training
        utils.params.rate = rate
        utils.params.data = utils.contaminate(in_dist, oo_dist)
        before = contamination(params.data)
        print("before", before)
        synthetic = utils.generate_synthetic_data()
        utils.params.update_data(synthetic)
        if not is_bopeto:
            y = utils.params.data[:, -1]
            dynamics = utils.get_reconstruction_errors()
            utils.params.dynamics = np.column_stack((dynamics, y))
            np.savetxt(outputs+utils.params.model.name+'.csv', utils.params.dynamics, delimiter=',')

        #filtering
        dynamics = np.loadtxt(outputs+utils.params.model.name+'.csv', delimiter=',')
        utils.params.dynamics = dynamics
        b = BOPETO(utils.params)
        indices = b.refine(True)
        refined_data = utils.params.data[indices]
        after = contamination(refined_data)
        print("after", after)





