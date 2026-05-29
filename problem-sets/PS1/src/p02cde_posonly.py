import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    model_t = LogisticRegression()
    model_t.fit(x_train, t_train)
    t_pred_c = model_t.predict(x_test)
    np.savetxt(pred_path_c, t_pred_c > 0.5, fmt='%d')
    util.plot(x_test, t_test, model_t.theta, 'output/p02c_{}_posonly.png'.format(pred_path_c[-5]))
    accuracy_val = accuracy(t_pred_c, t_test)
    print(f"Accuracy: {accuracy_val}")
    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    x_train_d, y_train_d = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test_d, y_test_d = util.load_dataset(test_path, label_col='y', add_intercept=True)
    model_y = LogisticRegression()
    model_y.fit(x_train_d, y_train_d)
    y_pred_d = model_y.predict(x_test_d)
    np.savetxt(pred_path_d, y_pred_d > 0.5, fmt='%d')
    util.plot(x_test_d, y_test_d, model_y.theta, 'output/p02d_{}_posonly.png'.format(pred_path_d[-5]))
    accuracy_val = accuracy(y_pred_d, y_test_d)
    print(f"Accuracy: {accuracy_val}")
    # Part (e): Apply correction factor using validation set and test on true labels
    x_valid_e, y_valid_e = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    

    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE
def accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)