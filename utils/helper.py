import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

#--- 1. Set a global theme with a style sheet ---
plt.style.use('seaborn')

# --- 2. Make fine-grained global changes using plt.rcParams ---
# Note: We are now using 'plt.rcParams' directly.
plt.rcParams['font.family'] = 'serif'
plt.rcParams['lines.linewidth'] = 2.5 # Make lines thicker by default


def plot_decision_regions(X, y, classifier, ax=ax, test_idx=None, resolution=0.02):
    """
    Plot decision boundaries for a classifier using two features.

    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)
        Feature matrix containing exactly two features.

    y : ndarray of shape (n_samples,)
        Target class labels corresponding to each sample in X.

    classifier : object
        Trained classifier with a `.predict` method that takes a 2D array.

    ax : matplotlib.axes.Axes
        Matplotlib Axes object on which to draw the decision surface and points.

    test_idx : array-like of shape (n_test_samples,), optional
        Indices of test examples within X. These points are highlighted.

    resolution : float, default=0.02
        Step size used to create the meshgrid for plotting the decision surface.

    Notes
    -----
    - This function assumes `X` contains exactly two features.
    - Useful for visualizing classifiers like SVM, LogisticRegression, or KNN.
    - Colors and markers are automatically assigned to unique classes in `y`.
    """
    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    ax.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y == cl, 0],
                   y=X[y == cl, 1],
                   alpha=0.8,
                   c=colors[idx],
                   marker=markers[idx],
                   label=f'Class {cl}',
                   edgecolor='black')

    # highlight test examples
    if test_idx is not None:
        X_test, y_test = X[test_idx, :], y[test_idx]
        ax.scatter(X_test[:, 0], X_test[:, 1],
                   c='none', edgecolor='black', alpha=1.0,
                   linewidth=1, marker='o',
                   s=100, label='Test set')



class LogisticRegressionGD:
    """Gradient descent-based logistic regression classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after training.
    b_ : Scalar
      Bias unit after fitting.
    losses_ : list
       Log loss function values in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : Instance of LogisticRegressionGD

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * errors.mean()
            loss = (-y.dot(np.log(output)) - (1 - y).dot(np.log(1 - output))) / X.shape[0]
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)