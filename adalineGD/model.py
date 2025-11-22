from numpy import where, dot
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

    def fit(self, X, y):
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
        pass

    def net_input(self, X):
        """Calculate net input"""
