import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    util.plot(x_train, y_train, clf.theta, 'output/p01b_{}.png'.format(pred_path[-5]))

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept = True)
    y_pred = clf.predict(x_eval)
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """


    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        [m, n] = x.shape
        theta = np.zeros([n, 1])

        e = 1
        fit_iter = 0

        while e > self.eps and fit_iter < self.max_iter:

            g_z = 1 / (1 + np.exp(-np.matmul(x, theta)))
            j_dot = np.matmul(np.transpose(x), g_z - y[:, None]) / m

            g_z_diag = np.zeros([m,m])
            for i in range(m):
                g_z_diag[i,i] = g_z[i,0] * (1 - g_z[i,0])
            hess = np.matmul(np.transpose(x),np.matmul(g_z_diag,x)) / m
            theta_old = theta
            theta = theta -  np.matmul(np.linalg.inv(hess), j_dot)
            e = np.linalg.norm(theta - theta_old,1)
            fit_iter += 1

        self.theta = theta

        return self.theta
        # *** END CODE HERE ***


    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x.dot(self.theta)))
        # *** END CODE HERE ***
