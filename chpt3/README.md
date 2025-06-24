# Chapter 3: Utility Functions
I made some changes to how I am organising the codes and I walk thru this book from chapter 3

The README will be used to explain some changes or how the work is organized if there are changes from the previous chapters.

I will also use the readme to explain some of the code that I have written in the chapter.

As our project progresses into Chapter 3 and beyond, the complexity will naturally increase. To maintain a clean, organized, and efficient workflow, it's crucial to separate our primary analysis from reusable, generic code. For this reason, I have created a `utils` directory containing a `helper.py` file. This module will act as a central repository for all helper functions, such as the `plot_decision_regions` function we've already used. This approach embodies the "Don't Repeat Yourself" (DRY) principle, making our main notebooks leaner and more focused on the specific logic of each chapter, while ensuring that our utility functions are easily maintainable and accessible for reuse throughout the rest of this walkthrough. From now on, any new helper function that we build will be added to this file.

In addition to the `helper.py` file, I have also created a `__init__.py` file in the `utils` directory. This file is essential for Python to recognize the directory as a package, allowing us to import functions from `helper.py` seamlessly. The `__init__.py` file can be empty, but it serves as a marker for the Python interpreter.

I have also chosen to include some Jupyter notebook commands for automatic code reloading:

**`%load_ext autoreload`** - Loads the autoreload extension that monitors Python files for changes.

**`%autoreload 2`** - Automatically reloads all modified modules before executing code in each cell.

This eliminates the need to restart your kernel when you modify imported Python files during development - changes are automatically picked up the next time you run a cell.


## Class Definition & Documentation for the Logistic Regression Classifier
```python
class LogisticRegressionGD:
    """Gradient descent-based logistic regression classifier..."""
```
Defines a logistic regression classifier that uses gradient descent for training. The docstring documents the hyperparameters (learning rate, iterations, random seed) and what attributes will be available after training (weights, bias, loss history).

## Constructor (`__init__`)
```python
def __init__(self, eta=0.01, n_iter=50, random_state=1):
```
Initializes the classifier with hyperparameters:
- `eta`: Learning rate controlling step size during gradient descent
- `n_iter`: Number of training epochs 
- `random_state`: Seed for reproducible weight initialization

## Training Method (`fit`)
```python
def fit(self, X, y):
```
**Weight Initialization:**
```python
rgen = np.random.RandomState(self.random_state)
self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
self.b_ = np.float_(0.)
```
Initializes weights from a normal distribution and bias to zero.

**Training Loop:**
```python
for i in range(self.n_iter):
    net_input = self.net_input(X)
    output = self.activation(net_input)
    errors = (y - output)
```
For each epoch: computes predictions, calculates prediction errors.

**Parameter Updates:**
```python
self.w_ += self.eta * X.T.dot(errors) / X.shape[0]
self.b_ += self.eta * errors.mean()
```
Updates weights and bias using gradient descent rules.

**Loss Tracking:**
```python
loss = (-y.dot(np.log(output)) - (1 - y).dot(np.log(1 - output))) / X.shape[0]
```
Computes and stores the logistic loss (cross-entropy) for monitoring training progress.

## Utility Methods

**`net_input`:** Computes the linear combination (z = Xw + b) before applying activation.

**`activation`:** Applies sigmoid function with clipping to prevent numerical overflow.

**`predict`:** Makes binary predictions by thresholding sigmoid output at 0.5.