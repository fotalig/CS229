import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    MSE_min = 1e5
    for tau in tau_values:
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_eval)
        MSE = (y_eval - y_pred).dot((y_eval - y_pred).T) / y_pred.size
        if MSE < MSE_min:
            tau_min, MSE_min = tau, MSE
        plt.figure()
        plt.plot(x_eval, y_eval, 'bx', linewidth=2)
        plt.plot(x_eval, y_pred, 'ro', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('output/p05c_tau={}.png'.format(tau))
        plt.show()
    # Fit a LWR model with the best tau value
        model_final = LocallyWeightedLinearRegression(tau_min)
        model_final.fit(x_train,y_train)
    # Run on the test set to get the MSE value
        y_pred = model_final.predict(x_test)
    # Save predictions to pred_path
        np.savetxt(pred_path, y_pred)
    # Plot data
    # *** END CODE HERE ***
