import numpy as np

class Metrics:

    def __init__(self, dynamics):
        """

        Parameters
        ----------
        dynamics of type array (number of data points, number of iterations)
        """
        self.dynamics = dynamics[:, :-1]

    def mac(self):
        """
        Mean Absolute Change (MAC)
        Returns
        -------

        """
        return np.mean(np.abs(np.diff(self.dynamics, axis=1)), axis=1)

    def msc(self):
        """
        Mean Squared Change (MSC)
        Returns
        -------

        """
        return np.mean(np.diff(self.dynamics, axis=1) ** 2, axis=1)

    def sdc(self):
        """
        Standard Deviation of Changes
        Returns
        -------

        """
        return np.std(np.diff(self.dynamics, axis=1), axis=1)

    def std(self):
        """
        Standard Deviation of Changes
        Returns
        -------

        """
        return np.std(self.dynamics, axis=1)

    def _pc(self):
        """
        Percentage Change (PC)
        Returns
        -------

        """
        return np.std(np.diff(self.dynamics, axis=1) / self.dynamics[:, :-1] * 100, axis=1)

    def _rmac(self):
        """
        Relative Mean Absolute Change (RMAC)
        Returns
        -------

        """
        mac = self.mac()
        return mac / np.mean(self.dynamics, axis=1)

    def cv(self):
        """
        Coefficient of variation (CV).

        Returns:
        float: Coefficient of variation.
        """
        mean_value = np.mean(self.dynamics, axis=1)
        std_dev = np.std(self.dynamics, axis=1)

        return std_dev / mean_value

    def _iqr(self):
        """
        Interquartile range (IQR).

        Returns:
        float: Interquartile range.
        """
        q1 = np.percentile(self.dynamics, 25, axis=1)
        q3 = np.percentile(self.dynamics, 75, axis=1)
        return q3 - q1

def contamination(data):
    in_dist = len(data[data[:, -1] == 0])
    oo_dist = len(data[data[:, -1] == 1])
    n = in_dist + oo_dist
    return n, oo_dist/n