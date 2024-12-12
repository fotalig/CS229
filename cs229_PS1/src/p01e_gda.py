import numpy as np
from numpy.ma import transpose

import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    clf = GDA()
    clf.fit(x_train, y_train)

    util.plot(x_train, y_train, clf.theta, 'output/p01e_{}.png'.format(pred_path[-5]))


    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    y_pred = clf.predict(x_eval)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        [m, n] = x.shape
        phi = np.sum(y == 1) / m
        u_0 = np.zeros([n,1])
        u_1 = np.zeros([n,1])
        for i in range(m):
            u_0 += (y[i] == 0) * np.transpose(x[i,None])
            u_1 += (y[i] == 1) * np.transpose(x[i,None])
        u_0 = u_0 / np.sum(y == 0)
        u_1 = u_1 / np.sum(y == 1)
        sigma = np.zeros([n, n])
        for i in range(m):
            u_y = u_0 * (y[i] == 0) + u_1 * (y[i] == 1)
            sigma += np.matmul(np.transpose(x[i,None]) - u_y, np.transpose(np.transpose(x[i,None]) - u_y))

        sigma = sigma / m
        sigma_inv = np.linalg.inv(sigma)
        theta = np.matmul(sigma_inv,(u_1 - u_0))
        theta_0 = np.matmul(np.transpose(u_0 + u_1), np.matmul(sigma_inv, (u_0 - u_1))) / 2 - np.log((1-phi)/phi)
        self.theta = np.vstack((theta_0,theta))

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-(self.theta[0] + x.dot(self.theta[1:3]))))
        # *** END CODE HERE
