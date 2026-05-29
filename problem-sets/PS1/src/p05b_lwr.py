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
    clf = LocallyWeightedLinearRegression(tau=tau)
    clf.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = clf.predict(x_eval)
    mse = np.mean((y_pred - y_eval)**2)
    print(f"MSE: {mse}")
    plot_regression(
        x_train, y_train, x_eval, y_pred,
        'output/p05b_{}_lwr.png'.format(tau))
    # *** START CODE HERE ***
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
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
        self.x = x
        self.y = y

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        y_pred = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            W = create_weight_diagonal_matrix(self.x, x[i], self.tau)
            theta = np.linalg.solve(
                self.x.T.dot(W).dot(self.x),
                self.x.T.dot(W).dot(self.y)
            )
            y_pred[i] = x[i].dot(theta)
        return y_pred

def create_weight_diagonal_matrix(x, x0, tau):
    w = np.exp(
    -np.sum((x - x0)**2, axis=1) / (2 * tau**2)
)   
    return np.diag(w)


def plot_regression(x_train, y_train, x_eval, y_pred, save_path):
    plt.figure()
    plt.plot(x_train[:, -1], y_train, 'bx', label='label')
    plt.plot(x_eval[:, -1], y_pred, 'ro', label='prediction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.savefig(save_path)
