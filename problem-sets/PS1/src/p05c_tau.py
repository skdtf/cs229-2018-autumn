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
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    # tau_values = np.linspace(0.01, 0.1, 10)
    # mse_map = {}
    # for tau in tau_values:
    #     clf = LocallyWeightedLinearRegression(tau=tau)
    #     clf.fit(x_train, y_train)
    #     y_pred = clf.predict(x_eval)
    #     mse = np.mean((y_pred - y_eval)**2)
    #     mse_map[tau] = mse
    # tau, mse = min(mse_map.items(), key=lambda x: x[1])
    # print(f"Best tau: {tau}, Best MSE: {mse}")
    # *** START CODE HERE ***
    tau = 0.05

    clf = LocallyWeightedLinearRegression(tau=tau)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    mse = np.mean((y_pred - y_test)**2)
    print(f"MSE: {mse}")
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    # *** END CODE HERE ***
