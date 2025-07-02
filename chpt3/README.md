# 📘 Project Structure – Chapter 3 Onward

As we start Chapter 3, the project is evolving in complexity. To keep things clean, reusable, and maintainable, here’s how the code is now organized:

---

## 🔧 `utils` Package for Helper Functions

* **`utils/helper.py`** — holds reusable utility functions (e.g., `plot_decision_regions`), following the DRY (“Don’t Repeat Yourself”) principle.
* **`utils/__init__.py`** — creates a package namespace so you can import helpers straightforwardly (e.g., `from utils.helper import ...`).

---

## 🔁 Automatic Module Reloading in Notebooks

To streamline development and auto-pick up code changes without restarting your kernel, include these lines in your Jupyter cells:

```python
%load_ext autoreload
%autoreload 2
```

This ensures any modifications to `helper.py` (or other modules) are loaded before each cell runs.

---

## 🤖 `LogisticRegressionGD` Class Overview

### **Purpose**

A gradient-descent-based logistic regression classifier, with built-in training tracking (weights, bias, loss history).

### **Key Methods**

#### `__init__(eta=0.01, n_iter=50, random_state=1)`

* `eta`: Learning rate — how big each update step is.
* `n_iter`: Number of iterations (epochs).
* `random_state`: Seed for repeatable weight initialization.

#### `fit(self, X, y)`

1. **Initialize**

   * Weights (`w_`) from a small normal distribution.
   * Bias (`b_`) set to zero.
2. **Train (per epoch)**

   * Compute linear combination: `net_input = X·w + b`
   * Apply sigmoid activation: `output = sigmoid(net_input)`
   * Calculate prediction errors: `errors = y – output`
3. **Update parameters**

   ```python
   w_ += eta * (X.T · errors) / n_samples
   b_ += eta * mean(errors)
   ```
4. **Track training loss** (cross-entropy):

   ```python
   loss = (−y·log(output) − (1−y)·log(1−output)) / n_samples
   ```

   * This is stored in `self.loss_` for later analysis.

---

### 🛠 Utility Methods

* **`net_input(X)`**
  Computes the linear output (`Xw + b`) before applying activation.

* **`activation(z)`**
  Applies the sigmoid function to `z`, with clipping to avoid overflow.

* **`predict(X)`**
  Applies the sigmoid output and thresholds at 0.5 to return class labels {0,1}.

---

## ✅ Notes on the README

* **Main notebooks** will focus only on the core analysis—helper functions live in `utils/helper.py`.
* **`utils/__init__.py`** lets Python recognize `utils` as a module directory.
* **Auto-reloading** keeps the development loop smooth—no manual restarts needed.

