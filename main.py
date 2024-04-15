import argparse
from params import Params
from utils import Utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bopeto",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--contamination_rate', nargs='?', const=1, type=float, default=0.05)
    parser.add_argument('-b', '--batch_size', nargs='?', const=1, type=int, default=64)
    parser.add_argument('-l', '--learning_rate', nargs='?', const=1, type=float, default=1e-3)
    parser.add_argument('-w', '--weight_decay', nargs='?', const=1, type=float, default=1e-3)
    parser.add_argument('-a', '--alpha', nargs='?', const=1, type=float, default=0.3)
    parser.add_argument('-g', '--gamma', nargs='?', const=1, type=float, default=0.1)
    parser.add_argument('-m', '--momentum', nargs='?', const=1, type=float, default=0.9)
    parser.add_argument('-e', '--epochs', nargs='?', const=1, type=int, default=20)
    parser.add_argument('-n', '--num_workers', nargs='?', const=1, type=int, default=4)
    parser.add_argument('--path', type=str, default='/data', help='path to data sets to use')
    parser.add_argument('--metric', type=str, default='IQR', help='dynamics metrics')
    parser.add_argument('--synthetic', type=str, default='ours', help='path to data sets to use')

    args = parser.parse_args()
    configs = vars(args)
    rate = configs['contamination_rate']
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

    params = Params(rate, batch_size, lr, wd, nw, alpha, gamma, momentum, epochs, path, metric, synthetic)
    params.set_model(load=False)
    utils = Utils(params)
    utils.train()


