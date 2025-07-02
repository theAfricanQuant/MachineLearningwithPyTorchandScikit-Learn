

### 1. 📦 Imports

```python
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
```

* `clone`: makes a copy of your model so we can reuse it.
* `combinations`: helps us try different sets of features.
* `numpy`: for math operations.
* `accuracy_score`: to measure how good our model is.
* `train_test_split`: splits the data into training and testing parts.

---

### 2. 🧱 `__init__`: Setting up the class

```python
def __init__(self, estimator, k_features, scoring=accuracy_score,
             test_size=0.25, random_state=1):
```

This function runs when you first create an `SBS` object.

* `estimator`: the model (like a decision tree or logistic regression) you want to use.
* `k_features`: how many features you want to keep at the end.
* `scoring`: how we measure model performance (by default: accuracy).
* `test_size`: how much data to use for testing (25% here).
* `random_state`: ensures results are reproducible by fixing the random split.

🧠 Think of this like setting up the rules before a chess game.

---

### 3. 🏋️ `fit`: Training SBS

```python
def fit(self, X, y):
```

This is the heart of SBS. It figures out which features to keep.

#### a. Split into training and testing data

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                    random_state=self.random_state)
```

We divide the data so we can train on one part and test on another. This helps avoid overfitting.

#### b. Start with all features

```python
dim = X_train.shape[1]
self.indices_ = tuple(range(dim))
self.subsets_ = [self.indices_]
```

* `dim` is how many columns (features) we have.
* `self.indices_` keeps track of which features we’re currently using.
* `self.subsets_` stores all the steps of feature removal.

#### c. Score the full model

```python
score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
self.scores_ = [score]
```

We run the model with all features and save how well it performs.

---

### 4. 🔁 The loop: Remove features one by one

```python
while dim > self.k_features:
```

We keep removing one feature until we’re left with the desired number (`k_features`).

#### a. Try removing each feature

```python
for p in combinations(self.indices_, r=dim - 1):
    score = self._calc_score(X_train, y_train, X_test, y_test, p)
    scores.append(score)
    subsets.append(p)
```

If we have 5 features, we try all combinations of 4 features (removing 1 at a time). For each combo, we:

* Run the model.
* Get the score.
* Save the combo.

Mathematically: You are testing all ${n \choose n-1} = n$ ways of removing one feature.

#### b. Pick the best one

```python
best = np.argmax(scores)
self.indices_ = subsets[best]
self.subsets_.append(self.indices_)
dim -= 1
self.scores_.append(scores[best])
```

* `np.argmax(scores)` finds the best-scoring feature set.
* We update `indices_` to this best set.
* We save it and repeat.

---

### 5. ✅ Final best score

```python
self.k_score_ = self.scores_[-1]
```

After the loop finishes, we save the final score using `k_features`.

---

### 6. 🔄 `transform`: Reduce a dataset to the selected features

```python
def transform(self, X):
    return X[:, self.indices_]
```

This takes in new data and gives back only the selected features.

---

### 7. 🧠 `_calc_score`: Train and evaluate the model

```python
def _calc_score(self, X_train, y_train, X_test, y_test, indices):
    self.estimator.fit(X_train[:, indices], y_train)
    y_pred = self.estimator.predict(X_test[:, indices])
    score = self.scoring(y_test, y_pred)
    return score
```

* Use only selected columns (features) from training and test sets.
* Train the model and make predictions.
* Compare predictions to actual values using `accuracy_score`.

