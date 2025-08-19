

# 📊 Regression and Data Analysis

This project explores **regression analysis, correlation methods, and implementing linear regression from scratch** using modern Python tools.

We use the **Ames Housing Dataset** as an example to predict house prices (`SalePrice`) from features like living area, quality, and year built.

---

## 🚀 Project Setup

We use **PowerShell** + **uv** (a modern Python package manager) to create and manage the environment.

### 1. Create Project Structure

```powershell
# Create a folder for the project
mkdir chpt9

# Create an empty README.md file
New-Item chpt9\README.md

# Or create a file with initial content
echo "# My Project Title" > chpt9\README.md
```

### 2. Run a marimo Notebook

```powershell
uv run python -m marimo edit chpt9\chpt9.py
```

> 💡 `uv run` automatically creates a virtual environment if none exists.

### 3. Add Dependencies

```powershell
# Add mlxtend for plotting
uv add mlxtend

# Add Jinja2 (dependency for other packages)
uv add jinja2
```

---

## 📈 Regression Analysis

**Regression analysis** models the relationship between:

* **Dependent variable** (outcome we want to predict, e.g., `SalePrice`)
* **Independent variables** (predictors/features, e.g., `GrLivArea`, `OverallQual`, `YearBuilt`)

General equation:

$$
SalePrice = (w_1 \times GrLivArea) + (w_2 \times OverallQual) + \dots + b
$$

Where:

* $w_i$ = weight/importance of feature
* $b$ = intercept

---

## 🧑‍💻 Linear Regression from Scratch

We implemented a gradient descent–based linear regression model:

```python
import numpy as np

class LinearRegressionGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.array([0.])
        self.losses_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return self.net_input(X)
```

### 🔍 How it Works

* **Initialize** weights & bias
* **Iterate**: predict → compute error → update weights & bias
* **Track** mean squared error (`losses_`) over epochs

---

## 🔗 Correlation Analysis

Before regression, we study feature relationships.

| Method       | Measures                            | Best For                                     |
| ------------ | ----------------------------------- | -------------------------------------------- |
| **Pearson**  | Linear relationship                 | Data with straight-line trend & few outliers |
| **Spearman** | Monotonic relationship (rank-based) | Non-linear trends, data with outliers        |
| **Kendall**  | Pairwise agreement                  | Small datasets, robustness                   |

Example with **pandas**:

```python
# Spearman correlation matrix
corr_matrix = df.corr(method='spearman')

# Visualize as heatmap
corr_matrix.style.background_gradient(cmap='viridis')
```

---

## 📚 Tools & Libraries

* [uv](https://github.com/astral-sh/uv) → fast package manager & venvs
* [marimo](https://marimo.io/) → interactive Python notebooks
* [mlxtend](http://rasbt.github.io/mlxtend/) → plotting & ML utilities
* [pandas](https://pandas.pydata.org/) → data analysis
* [numpy](https://numpy.org/) → numerical computing

---



