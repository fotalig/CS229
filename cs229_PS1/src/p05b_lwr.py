import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    clf = LocallyWeightedLinearRegression(tau)
    clf.fit(x_train, y_train)
    # Get MSE value on the validation set
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = clf.predict(x_eval)
    MSE = (y_eval - y_pred).dot((y_eval - y_pred).T) / y_pred.size

    # Plot validation predictions on top of training set
    plt.figure()
    plt.plot(x_eval,y_eval,'bx', linewidth=2)
    plt.plot(x_eval, y_pred, 'ro', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('output/p05b.png')
    #plt.show()
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m_eval, n_eval = x.shape
        m_train, n_train = self.x.shape
        y_pred = np.zeros([m_eval, 1])
        for i in range(m_eval):
            x_pred = x[i,:]
            x_pred_vec = np.tile(x_pred, (m_train, 1))
            W = np.diag(np.exp( - np.power(np.linalg.norm(self.x - x_pred_vec,axis=1)[:,None],2).T[0,:] / (2 * self.tau**2)))
            xT_W = self.x.T.dot(W)
            theta = np.matmul(np.linalg.inv(xT_W.dot(self.x)),xT_W.dot(self.y))
            y_pred[i] = theta.T.dot(x[i])
        return y_pred.T[0,:]
        # *** END CODE HERE ***
