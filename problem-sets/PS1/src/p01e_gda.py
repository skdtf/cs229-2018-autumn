import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    print(f"Loading training data from {train_path}")
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    print(f"Training set: {x_train.shape[0]} examples, {x_train.shape[1]} features")

    print("Fitting GDA model...")
    clf = GDA()
    clf.fit(x_train, y_train)

    print(f"Loading evaluation data from {eval_path}")
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    print(f"Evaluation set: {x_eval.shape[0]} examples")

    print("Predicting on evaluation set...")
    y_pred = clf.predict(x_eval)
    acc = np.mean(y_pred == y_eval)
    print(f"Theta: {clf.theta}")
    print(f"Accuracy: {acc}")
    print(f"Saving predictions to {pred_path}")
    util.plot(x_train, y_train, clf.theta, 'output/p01e_{}_gda.png'.format(pred_path[-5]))
    np.savetxt(pred_path, y_pred, fmt='%d')


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
        m, n = x.shape
        if self.verbose:
            print(f"[GDA fit] Step 1/6: dataset shape = ({m}, {n})")

        phi = np.sum(y == 1) / m
        if self.verbose:
            print(f"[GDA fit] Step 2/6: phi (P(y=1)) = {phi}")

        mu_0 = np.sum(x[y == 0], axis=0) / np.sum(y == 0)
        mu_1 = np.sum(x[y == 1], axis=0) / np.sum(y == 1)
        if self.verbose:
            print(f"[GDA fit] Step 3/6: mu_0 = {mu_0}")
            print(f"[GDA fit] Step 3/6: mu_1 = {mu_1}")

        sigma = ((x[y == 0] - mu_0).T.dot(x[y == 0] - mu_0)
                 + (x[y == 1] - mu_1).T.dot(x[y == 1] - mu_1)) / m
        if self.verbose:
            print(f"[GDA fit] Step 4/6: Sigma =\n{sigma}")

        sigma_inv = np.linalg.inv(sigma)
        if self.verbose:
            print("[GDA fit] Step 5/6: computed Sigma inverse")

        theta = sigma_inv.dot(mu_1 - mu_0)
        theta_0 = (0.5 * (mu_0 + mu_1).T.dot(sigma_inv).dot(mu_0 - mu_1)
                   - np.log((1 - phi) / phi))
        if self.verbose:
            print(f"[GDA fit] Step 6/6: theta (weights) = {theta}")
            print(f"[GDA fit] Step 6/6: theta_0 (intercept) = {theta_0}")

        self.theta = np.insert(theta, 0, theta_0)
        if self.verbose:
            print(f"[GDA fit] Done. self.theta = {self.theta}")

        return self.theta

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        return util.add_intercept(x) @ self.theta >= 0
