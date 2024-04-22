from sklearn import metrics as sk_metrics
import torch
import numpy as np
from trainer.trainer import *
from synthetic.generation import JMF, FGM
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score

class Utils:
    def __init__(self, params):
        self.params = params

    def get_reconstruction_errors(self):
        trainer = Trainer(self.params)
        errors = trainer.train(True)
        return errors

    def initial_train(self):
        trainer = Trainer(self.params)
        errors = trainer.train(False)
        return errors

    def generate_synthetic_data(self):
        if self.params.synthetic == "JMF":
            generator = JMF(self.params)
        else:
            generator = FGM(self.params)
        return generator.generate()

    def contaminate(self, in_dist, oo_dist):
        data = np.vstack((in_dist, oo_dist))
        np.random.shuffle(data)
        self.params.rate = oo_dist.shape[0]/(in_dist.shape[0]+oo_dist.shape[0])
        return data

    def optimal_threshold(self, model):
        pass

def compute_metrics(val_score, y_val, thresh, pos_label=1):
    y_pred = (val_score >= thresh).astype(int)
    y_true = y_val.astype(int)

    accuracy = sk_metrics.accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=pos_label
    )
    avgpr = sk_metrics.average_precision_score(y_true, val_score)
    roc = sk_metrics.roc_auc_score(y_true, val_score)
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])

    return accuracy, precision, recall, f_score, roc, avgpr, cm

def compute_metrics_binary(y_pred, y_val, pos_label=1):
    y_true = y_val.astype(int)

    accuracy = sk_metrics.accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=pos_label
    )
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])

    return accuracy, precision, recall, f_score, cm
def estimate_optimal_threshold(val_score, y_val, pos_label=1, nq=100):
    ratio = 100 * sum(y_val == 0) / len(y_val)
    q = np.linspace(ratio - 5, min(ratio + 5, 100), nq)
    thresholds = np.percentile(val_score, q)

    result_search = []
    confusion_matrices = []
    f1 = np.zeros(shape=nq)
    r = np.zeros(shape=nq)
    p = np.zeros(shape=nq)
    auc = np.zeros(shape=nq)
    aupr = np.zeros(shape=nq)
    qis = np.zeros(shape=nq)

    for i, (thresh, qi) in enumerate(zip(thresholds, q)):
        accuracy, precision, recall, f_score, roc, avgpr, cm = compute_metrics(val_score, y_val, thresh, pos_label)

        confusion_matrices.append(cm)
        result_search.append([accuracy, precision, recall, f_score])
        f1[i] = f_score
        r[i] = recall
        p[i] = precision
        auc[i] = roc
        aupr[i] = avgpr
        qis[i] = qi

    arm = np.argmax(f1)

    return {
        "Precision": p[arm],
        "Recall": r[arm],
        "F1-Score": f1[arm],
        "AUPR": aupr[arm],
        "AUROC": auc[arm],
        "Thresh_star": thresholds[arm],
        "Quantile_star": qis[arm]
    }

def compute_metrics_(params):
    params.model.eval()
    N = 0
    mu_sum = 0
    cov_sum = 0
    gamma_sum = 0

    for it, (input_data, labels) in enumerate(self.data_loader):
        input_data = self.to_var(input_data)
        enc, dec, z, gamma = params.model.dagmm(input_data)
        phi, mu, cov = params.model.compute_gmm_params(z, gamma)

        batch_gamma_sum = torch.sum(gamma, dim=0)

        gamma_sum += batch_gamma_sum
        mu_sum += mu * batch_gamma_sum.unsqueeze(-1)  # keep sums of the numerator only
        cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)  # keep sums of the numerator only

        N += input_data.size(0)

    train_phi = gamma_sum / N
    train_mu = mu_sum / gamma_sum.unsqueeze(-1)
    train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

    print("N:", N)
    print("phi :\n", train_phi)
    print("mu :\n", train_mu)
    print("cov :\n", train_cov)

    train_energy = []
    train_labels = []
    train_z = []
    for it, (input_data, labels) in enumerate(self.data_loader):
        input_data = self.to_var(input_data)
        enc, dec, z, gamma = params.model(input_data)
        sample_energy, cov_diag = params.model.compute_energy(z, phi=train_phi, mu=train_mu, cov=train_cov,
                                                            size_average=False)

        train_energy.append(sample_energy.data.cpu().numpy())
        train_z.append(z.data.cpu().numpy())
        train_labels.append(labels.numpy())

    train_energy = np.concatenate(train_energy, axis=0)
    train_z = np.concatenate(train_z, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)


    test_energy = []
    test_labels = []
    test_z = []
    for it, (input_data, labels) in enumerate(self.data_loader):
        input_data = self.to_var(input_data)
        enc, dec, z, gamma = params.model.dagmm(input_data)
        sample_energy, cov_diag = params.model.compute_energy(z, size_average=False)
        test_energy.append(sample_energy.data.cpu().numpy())
        test_z.append(z.data.cpu().numpy())
        test_labels.append(labels.numpy())

    test_energy = np.concatenate(test_energy, axis=0)
    test_z = np.concatenate(test_z, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    combined_labels = np.concatenate([train_labels, test_labels], axis=0)

    thresh = np.percentile(combined_energy, 100 - 20)

    pred = (test_energy > thresh).astype(int)
    gt = test_labels.astype(int)

    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = prf(gt, pred, average='binary')
    return accuracy, precision, recall, f_score








