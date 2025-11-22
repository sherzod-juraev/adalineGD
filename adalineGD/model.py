from numpy import where, dot, ndarray
from numpy.random import uniform

class AdalineGD:
    """ Adaptive Linear Neuron classifier.

    Parameters
    ------------

    eta : float
        Learning rate (between 0.0 and 1.0)

    n_iter : int
        Passes over the training dataset.
    """

    def __init__(self, eta: float = .01, n_iter: int = 100):

        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None
        self.b = 1e-1
        self.eps = 1e-5

    def fit(self, X: ndarray, y: ndarray, /) -> bool:
        """ Fit training data.

        Parameters
        ----------

        X : {array - like}, shape = [n_examples, n_features]
            Training vectors, where n_examples
            is the number of examples and
            n_features is the number of features.

        y : array - like, shape = [n_examples]
            Target values.
        """

        self.w_ = uniform(-.01, .01, size=X.shape[1])
        J_last, J_old = None, None
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            errors = y - net_input
            # update weights
            self.w_ += self.eta * X.T.dot(errors)
            # update bias
            self.b += self.eta * errors.sum()
            J_last = ((errors ** 2).sum()) / 2
            if i != 0:
                if abs(J_last - J_old) < self.eps:
                    return True
            J_old = J_last
        return False

    def net_input(self, X):
        """Calculate net input"""

        return dot(X, self.w_) + self.b

    def predict(self, X):
        """Return class label after unit step"""
        net_input = self.net_input(X)
        return where(net_input >= 0, 1, -1)