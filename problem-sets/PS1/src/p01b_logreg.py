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
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = clf.predict(x_eval)
    util.plot(x_train, y_train, clf.theta, 'output/p01b_{}_logreg.png'.format(pred_path[-5]))
    acc = accuracy(y_pred, y_eval)
    print(f"Accuracy: {acc}")
    np.savetxt(pred_path, y_pred, fmt='%d')

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
        theta_old = self.theta
        if theta_old is None:
            theta_old = np.zeros(x.shape[1])
        theta_new = newton_method(theta_old, x, y, self.step_size)
        iter = 1
        while np.linalg.norm(theta_new - theta_old, ord=1) > self.eps and iter < self.max_iter:
            theta_old = theta_new
            theta_new = newton_method(theta_old, x, y, self.step_size)
            iter += 1
        self.theta = theta_new
        if self.verbose:
            print(f"Converged in {iter} iterations")
            print(f"Theta: {theta_new}")
            print(f"Cost: {cost_function(theta_new, x, y)}")
            print(f"Gradient: {cost_function_derivative(theta_new, x, y)}")
            print(f"Hessian: {cost_function_hessian(theta_new, x, y)}")
            print(f"Step size: {self.step_size}")
            print(f"Max iterations: {self.max_iter}")
            print(f"Eps: {self.eps}")



    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        return x @ self.theta >= 0


def h(theta, x):
    return 1 / (1 + np.exp(-np.dot(x, theta)))

def cost_function_derivative(theta, x, y):
    return 1/len(y) * np.dot(x.T, h(theta, x) - y)

def cost_function_hessian(theta, x, y):
    m = len(y)
    h_vals = h(theta, x)               
    d = h_vals * (1 - h_vals)    
    return (1 / m) * (x.T @ (d[:, np.newaxis] * x)) 

def newton_method(theta, x, y, step_size):
    H = cost_function_hessian(theta, x, y)
    grad = cost_function_derivative(theta, x, y)
    return theta - step_size * np.linalg.solve(H, grad)

def cost_function(theta, x, y):
    return -1/len(y) * np.sum(y * np.log(h(theta, x)) + (1 - y) * np.log(1 - h(theta, x)))

def accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)
