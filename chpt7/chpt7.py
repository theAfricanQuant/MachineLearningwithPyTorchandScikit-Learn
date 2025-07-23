import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    sys.path.insert(0, '..')

    from python_environment_check import check_packages


    d = {
        'numpy': '1.21.2',
        'scipy': '1.7.0',
        'matplotlib': '3.4.3',
        'sklearn': '1.0',
        'pandas': '1.3.2',
        'xgboost': '1.5.0',
    }
    check_packages(d)
    return


@app.cell
def _():
    from scipy.special import comb
    import math


    def ensemble_error(n_classifier, error):
        k_start = int(math.ceil(n_classifier / 2.))
        probs = [comb(n_classifier, k) * error**k * (1-error)**(n_classifier - k)
                 for k in range(k_start, n_classifier + 1)]
        return sum(probs)

    ensemble_error(n_classifier=11, error=0.25)
    return (ensemble_error,)


@app.cell
def _(ensemble_error):
    import numpy as np


    error_range = np.arange(0.0, 1.01, 0.01)
    ens_errors = [ensemble_error(n_classifier=11, error=error)
                  for error in error_range]
    return ens_errors, error_range, np


@app.cell
def _():
    return


@app.cell
def _(ens_errors, error_range):
    import matplotlib.pyplot as plt


    plt.plot(error_range, 
             ens_errors, 
             label='Ensemble error', 
             linewidth=2)

    plt.plot(error_range, 
             error_range, 
             linestyle='--',
             label='Base error',
             linewidth=2)

    plt.xlabel('Base error')
    plt.ylabel('Base/Ensemble error')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.5)
    plt.savefig('chpt7/figures/07_03.png', dpi=300)
    plt.show()
    return (plt,)


@app.cell
def _(np):
    np.argmax(np.bincount([0, 0, 1], 
                          weights=[0.2, 0.2, 0.6]))
    return


@app.cell
def _(np):
    ex = np.array([[0.9, 0.1],
                   [0.8, 0.2],
                   [0.4, 0.6]])

    p = np.average(ex, 
                   axis=0, 
                   weights=[0.2, 0.2, 0.6])
    p
    return (p,)


@app.cell
def _(np, p):
    np.argmax(p)
    return


@app.cell
def _(np):
    from sklearn.base import BaseEstimator
    from sklearn.base import ClassifierMixin
    from sklearn.preprocessing import LabelEncoder
    from sklearn.base import clone
    from sklearn.pipeline import _name_estimators
    # import numpy as np
    import operator



    # Scikit-learn 0.16 and newer requires reversing the parent classes
    # See https://github.com/rasbt/machine-learning-book/discussions/205 for more details
    import sklearn
    base_classes = (ClassifierMixin, BaseEstimator) if sklearn.__version__ >= "0.16" else (BaseEstimator, ClassifierMixin)

    # class MajorityVoteClassifier(BaseEstimator, 
    #                             ClassifierMixin):

    class MajorityVoteClassifier(*base_classes):

        """ A majority vote ensemble classifier

        Parameters
        ----------
        classifiers : array-like, shape = [n_classifiers]
          Different classifiers for the ensemble

        vote : str, {'classlabel', 'probability'} (default='classlabel')
          If 'classlabel' the prediction is based on the argmax of
            class labels. Else if 'probability', the argmax of
            the sum of probabilities is used to predict the class label
            (recommended for calibrated classifiers).

        weights : array-like, shape = [n_classifiers], optional (default=None)
          If a list of `int` or `float` values are provided, the classifiers
          are weighted by importance; Uses uniform weights if `weights=None`.

        """
        def __init__(self, classifiers, vote='classlabel', weights=None):

            self.classifiers = classifiers
            self.named_classifiers = {key: value for key, value
                                      in _name_estimators(classifiers)}
            self.vote = vote
            self.weights = weights

        def fit(self, X, y):
            """ Fit classifiers.

            Parameters
            ----------
            X : {array-like, sparse matrix}, shape = [n_examples, n_features]
                Matrix of training examples.

            y : array-like, shape = [n_examples]
                Vector of target class labels.

            Returns
            -------
            self : object

            """
            if self.vote not in ('probability', 'classlabel'):
                raise ValueError(f"vote must be 'probability' or 'classlabel'"
                                 f"; got (vote={self.vote})")

            if self.weights and len(self.weights) != len(self.classifiers):
                raise ValueError(f'Number of classifiers and weights must be equal'
                                 f'; got {len(self.weights)} weights,'
                                 f' {len(self.classifiers)} classifiers')

            # Use LabelEncoder to ensure class labels start with 0, which
            # is important for np.argmax call in self.predict
            self.lablenc_ = LabelEncoder()
            self.lablenc_.fit(y)
            self.classes_ = self.lablenc_.classes_
            self.classifiers_ = []
            for clf in self.classifiers:
                fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
                self.classifiers_.append(fitted_clf)
            return self

        def predict(self, X):
            """ Predict class labels for X.

            Parameters
            ----------
            X : {array-like, sparse matrix}, shape = [n_examples, n_features]
                Matrix of training examples.

            Returns
            ----------
            maj_vote : array-like, shape = [n_examples]
                Predicted class labels.
            
            """
            if self.vote == 'probability':
                maj_vote = np.argmax(self.predict_proba(X), axis=1)
            else:  # 'classlabel' vote

                #  Collect results from clf.predict calls
                predictions = np.asarray([clf.predict(X)
                                          for clf in self.classifiers_]).T

                maj_vote = np.apply_along_axis(
                                          lambda x:
                                          np.argmax(np.bincount(x,
                                                    weights=self.weights)),
                                          axis=1,
                                          arr=predictions)
            maj_vote = self.lablenc_.inverse_transform(maj_vote)
            return maj_vote

        def predict_proba(self, X):
            """ Predict class probabilities for X.

            Parameters
            ----------
            X : {array-like, sparse matrix}, shape = [n_examples, n_features]
                Training vectors, where n_examples is the number of examples and
                n_features is the number of features.

            Returns
            ----------
            avg_proba : array-like, shape = [n_examples, n_classes]
                Weighted average probability for each class per example.

            """
            probas = np.asarray([clf.predict_proba(X)
                                 for clf in self.classifiers_])
            avg_proba = np.average(probas, axis=0, weights=self.weights)
            return avg_proba

        def get_params(self, deep=True):
            """ Get classifier parameter names for GridSearch"""
            if not deep:
                return super().get_params(deep=False)
            else:
                out = self.named_classifiers.copy()
                for name, step in self.named_classifiers.items():
                    for key, value in step.get_params(deep=True).items():
                        out[f'{name}__{key}'] = value
                return out
    return LabelEncoder, MajorityVoteClassifier


@app.cell
def _(LabelEncoder):
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    # from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split


    iris = datasets.load_iris()
    X, y = iris.data[50:, [1, 2]], iris.target[50:]
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test =\
           train_test_split(X, y, 
                            test_size=0.5, 
                            random_state=1,
                            stratify=y)
    return StandardScaler, X_test, X_train, y_test, y_train


@app.cell
def _(StandardScaler, X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier 
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score


    clf1 = LogisticRegression(penalty='l2', 
                              C=0.001,
                              solver='lbfgs',
                              random_state=1)

    clf2 = DecisionTreeClassifier(max_depth=1,
                                  criterion='entropy',
                                  random_state=0)

    clf3 = KNeighborsClassifier(n_neighbors=1,
                                p=2,
                                metric='minkowski')

    pipe1 = Pipeline([['sc', StandardScaler()],
                      ['clf', clf1]])
    pipe3 = Pipeline([['sc', StandardScaler()],
                      ['clf', clf3]])

    clf_labels = ['Logistic regression', 'Decision tree', 'KNN']

    print('10-fold cross validation:\n')
    for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
        scores = cross_val_score(estimator=clf,
                                 X=X_train,
                                 y=y_train,
                                 cv=10,
                                 scoring='roc_auc')
        print(f'ROC AUC: {scores.mean():.2f} '
              f'(+/- {scores.std():.2f}) [{label}]')
    return clf2, clf_labels, cross_val_score, pipe1, pipe3


@app.cell
def _(
    MajorityVoteClassifier,
    X_train,
    clf2,
    clf_labels,
    cross_val_score,
    pipe1,
    pipe3,
    y_train,
):
    # Majority Rule (hard) Voting

    mv_clf1 = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])

    # Define clf_labels1 in one go instead of using +=
    clf_labels1 = clf_labels + ['Majority voting']  # This replaces line 5
    all_clf1 = [pipe1, clf2, pipe3, mv_clf1]

    for classifier, classifier_label in zip(all_clf1, clf_labels1):
        cv_scores = cross_val_score(estimator=classifier,
                                   X=X_train,
                                   y=y_train,
                                   cv=10,
                                   scoring='roc_auc')
        print(f'ROC AUC: {cv_scores.mean():.2f} '
              f'(+/- {cv_scores.std():.2f}) [{classifier_label}]')

    # Alternative approach - define everything at once:
    # clf_labels1, all_clf1 = clf_labels + ['Majority voting'], [pipe1, clf2, pipe3, mv_clf1]

    return (all_clf1,)


@app.cell
def _(
    MajorityVoteClassifier,
    X_train,
    all_clf1,
    clf2,
    clf_labels,
    cross_val_score,
    pipe1,
    pipe3,
    y_train,
):
    # Majority Rule (hard) Voting

    mv_clf2 = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])

    clf_labels2 = clf_labels + ['Majority voting']


    for classifier2, classifier_label2 in zip(all_clf1, clf_labels2):
        scores1 = cross_val_score(estimator=classifier2,
                                 X=X_train,
                                 y=y_train,
                                 cv=10,
                                 scoring='roc_auc')
        print(f'ROC AUC: {scores1.mean():.2f} '
              f'(+/- {scores1.std():.2f}) [{classifier_label2}]')
    return clf_labels2, mv_clf2


@app.cell
def _(X_test, X_train, all_clf1, clf_labels2, plt, y_test, y_train):
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc


    colors = ['black', 'orange', 'blue', 'green']
    linestyles = [':', '--', '-.', '-']
    for clf4, label4, clr, ls \
            in zip(all_clf1,
                   clf_labels2, colors, linestyles):

        # assuming the label of the positive class is 1
        y_pred = clf4.fit(X_train,
                         y_train).predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_test,
                                         y_score=y_pred)
        roc_auc = auc(x=fpr, y=tpr)
        plt.plot(fpr, tpr,
                 color=clr,
                 linestyle=ls,
                 label=f'{label4} (auc = {roc_auc:.2f})')

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1],
             linestyle='--',
             color='gray',
             linewidth=2)

    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid(alpha=0.5)
    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('True positive rate (TPR)')


    plt.savefig('chpt7/figures/07_04', dpi=300)
    plt.show()
    return


@app.cell
def _(StandardScaler, X_train):
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    return (X_train_std,)


@app.cell
def _(X_train_std, clf2, clf_labels2, mv_clf2, np, pipe1, pipe3, plt, y_train):
    def _():
        from itertools import product


        all_clf = [pipe1, clf2, pipe3, mv_clf2]

        x_min = X_train_std[:, 0].min() - 1
        x_max = X_train_std[:, 0].max() + 1
        y_min = X_train_std[:, 1].min() - 1
        y_max = X_train_std[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        f, axarr = plt.subplots(nrows=2, ncols=2, 
                                sharex='col', 
                                sharey='row', 
                                figsize=(7, 5))

        for idx, clf, tt in zip(product([0, 1], [0, 1]),
                                all_clf, clf_labels2):
            clf.fit(X_train_std, y_train)
        
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
        
            axarr[idx[0], idx[1]].scatter(X_train_std[y_train==0, 0], 
                                          X_train_std[y_train==0, 1], 
                                          c='blue', 
                                          marker='^',
                                          s=50)
        
            axarr[idx[0], idx[1]].scatter(X_train_std[y_train==1, 0], 
                                          X_train_std[y_train==1, 1], 
                                          c='green', 
                                          marker='o',
                                          s=50)
        
            axarr[idx[0], idx[1]].set_title(tt)

        plt.text(-3.5, -5., 
                 s='Sepal width [standardized]', 
                 ha='center', va='center', fontsize=12)
        plt.text(-12.5, 4.5, 
                 s='Petal length [standardized]', 
                 ha='center', va='center', 
                 fontsize=12, rotation=90)

        plt.savefig('chpt7/figures/07_05', dpi=300)
        return plt.show()


    _()
    return


@app.cell
def _(mv_clf2):
    print(mv_clf2.get_params())
    return


@app.cell
def _(mv_clf2):
    # Store the parameters in a variable
    params = mv_clf2.get_params()

    # You can now inspect the dictionary, for example, by printing its keys
    print(params.keys())
    return


@app.cell
def _():
    from sklearn.model_selection import GridSearchCV

    def perform_grid_search(estimator, X_train, y_train):
        """
        Performs a GridSearchCV and returns the fitted grid object.

        Args:
            estimator: The scikit-learn estimator or pipeline to tune.
            X_train: The training feature data.
            y_train: The training target labels.

        Returns:
            The fitted GridSearchCV object.
        """
        params = {
            'decisiontreeclassifier__max_depth': [1, 2],
            'pipeline-1__clf__C': [0.001, 0.1, 100.0]
        }

        grid = GridSearchCV(estimator=estimator,
                          param_grid=params,
                          cv=10,
                          scoring='roc_auc')
    
        grid.fit(X_train, y_train)

        # Optional: Print the results for logging
        print("Grid Search Results:")
        for r, _ in enumerate(grid.cv_results_['mean_test_score']):
            mean_score = grid.cv_results_['mean_test_score'][r]
            std_dev = grid.cv_results_['std_test_score'][r]
            current_params = grid.cv_results_['params'][r]
            print(f'{mean_score:.3f} +/- {std_dev:.2f} {current_params}')
    
        print("=" * 100)
        print(f'Best parameters: {grid.best_params_}')
        print(f'ROC AUC: {grid.best_score_:.2f}')
        print("=" * 100)

        # Return the entire fitted grid object
        return grid
    return (perform_grid_search,)


@app.cell
def _(X_train, mv_clf2, perform_grid_search, y_train):
    # Assuming mv_clf2, X_train, and y_train are already defined

    # Call the function and capture the result
    fitted_grid = perform_grid_search(estimator=mv_clf2,
                                      X_train=X_train, 
                                      y_train=y_train)

    # Now you can easily access any attribute from the result
    best_model = fitted_grid.best_estimator_
    print("\nAccessing the best model after the function call:")
    print(best_model)
    return best_model, fitted_grid


@app.cell
def _(best_model):
    print(best_model.classifiers)
    return


@app.cell
def _(best_model, fitted_grid):
    print(best_model.set_params(**fitted_grid.best_estimator_.get_params()))
    return


@app.cell
def _():
    def compare_decision_tree_and_bagging():
        """
        Loads the Wine dataset, compares a single Decision Tree against a 
        Bagging Classifier, prints their accuracies, and plots their 
        decision boundaries.
        """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import BaggingClassifier
        from sklearn.metrics import accuracy_score

        # 1. Load and Preprocess Data
        df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                              'machine-learning-databases/wine/wine.data',
                              header=None)
    
        df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                           'Alcalinity of ash', 'Magnesium', 'Total phenols',
                           'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                           'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                           'Proline']

        # Filter for two classes and select features
        df_wine = df_wine[df_wine['Class label'] != 1]
        y = df_wine['Class label'].values
        X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values

        # Encode labels and split data
        le = LabelEncoder()
        y = le.fit_transform(y)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y,
                             test_size=0.2,
                             random_state=1,
                             stratify=y)

        # 2. Define, Train, and Evaluate Models
        tree = DecisionTreeClassifier(criterion='entropy',
                                      max_depth=None,
                                      random_state=1)
    
        bag = BaggingClassifier(base_estimator=tree,
                                n_estimators=500,
                                max_samples=1.0,
                                max_features=1.0,
                                bootstrap=True,
                                bootstrap_features=False,
                                n_jobs=1,
                                random_state=1)

        # Evaluate single decision tree
        tree = tree.fit(X_train, y_train)
        y_train_pred = tree.predict(X_train)
        y_test_pred = tree.predict(X_test)
        tree_train = accuracy_score(y_train, y_train_pred)
        tree_test = accuracy_score(y_test, y_test_pred)
        print(f'Decision tree train/test accuracies '
              f'{tree_train:.3f}/{tree_test:.3f}')

        # Evaluate bagging classifier
        bag = bag.fit(X_train, y_train)
        y_train_pred = bag.predict(X_train)
        y_test_pred = bag.predict(X_test)
        bag_train = accuracy_score(y_train, y_train_pred)
        bag_test = accuracy_score(y_test, y_test_pred)
        print(f'Bagging train/test accuracies '
              f'{bag_train:.3f}/{bag_test:.3f}')

        # 3. Visualize Decision Boundaries
        x_min = X_train[:, 0].min() - 1
        x_max = X_train[:, 0].max() + 1
        y_min = X_train[:, 1].min() - 1
        y_max = X_train[:, 1].max() + 1
    
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
    
        f, axarr = plt.subplots(nrows=1, ncols=2,
                                sharex='col',
                                sharey='row',
                                figsize=(8, 3))

        for idx, clf, tt in zip([0, 1],
                                [tree, bag],
                                ['Decision tree', 'Bagging']):
            clf.fit(X_train, y_train)

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            axarr[idx].contourf(xx, yy, Z, alpha=0.3)
            axarr[idx].scatter(X_train[y_train == 0, 0],
                               X_train[y_train == 0, 1],
                               c='blue', marker='^')
            axarr[idx].scatter(X_train[y_train == 1, 0],
                               X_train[y_train == 1, 1],
                               c='green', marker='o')
            axarr[idx].set_title(tt)

        axarr[0].set_ylabel('OD280/OD315 of diluted wines', fontsize=12)
        plt.tight_layout()
        plt.text(0, -0.2,
                 s='Alcohol',
                 ha='center',
                 va='center',
                 fontsize=12,
                 transform=axarr[1].transAxes)
    
        plt.show()


    # Run the complete analysis and plotting
    compare_decision_tree_and_bagging()
    return


@app.cell
def _(X_test, X_train, y_test, y_train):
    def compare_tree_and_adaboost(X_train, y_train, X_test, y_test):
        """
        Compares a single Decision Tree stump against an AdaBoost Classifier,
        prints their accuracies, and plots their decision boundaries.

        Args:
            X_train, y_train: Training data and labels.
            X_test, y_test: Testing data and labels.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.metrics import accuracy_score

        # 1. Define Models
        tree = DecisionTreeClassifier(criterion='entropy',
                                      max_depth=1,
                                      random_state=1)

        ada = AdaBoostClassifier(base_estimator=tree,
                                 n_estimators=500,
                                 learning_rate=0.1,
                                 random_state=1)

        # 2. Train and Evaluate Models
        # Evaluate single decision tree stump
        tree = tree.fit(X_train, y_train)
        y_train_pred = tree.predict(X_train)
        y_test_pred = tree.predict(X_test)
        tree_train = accuracy_score(y_train, y_train_pred)
        tree_test = accuracy_score(y_test, y_test_pred)
        print(f'Decision tree train/test accuracies '
              f'{tree_train:.3f}/{tree_test:.3f}')

        # Evaluate AdaBoost classifier
        ada = ada.fit(X_train, y_train)
        y_train_pred = ada.predict(X_train)
        y_test_pred = ada.predict(X_test)
        ada_train = accuracy_score(y_train, y_train_pred)
        ada_test = accuracy_score(y_test, y_test_pred)
        print(f'AdaBoost train/test accuracies '
              f'{ada_train:.3f}/{ada_test:.3f}')

        # 3. Visualize Decision Boundaries
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        f, axarr = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(8, 3))

        for idx, clf, tt in zip([0, 1],
                                [tree, ada],
                                ['Decision tree', 'AdaBoost']):
            clf.fit(X_train, y_train)
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            axarr[idx].contourf(xx, yy, Z, alpha=0.3)
            axarr[idx].scatter(X_train[y_train == 0, 0],
                               X_train[y_train == 0, 1],
                               c='blue', marker='^')
            axarr[idx].scatter(X_train[y_train == 1, 0],
                               X_train[y_train == 1, 1],
                               c='green', marker='o')
            axarr[idx].set_title(tt)

        axarr[0].set_ylabel('OD280/OD315 of diluted wines', fontsize=12)
        plt.tight_layout()
        plt.text(0, -0.2,
                 s='Alcohol',
                 ha='center',
                 va='center',
                 fontsize=12,
                 transform=axarr[1].transAxes)

        plt.show()


    # Run the complete analysis and plotting
    compare_tree_and_adaboost(X_train, y_train, X_test, y_test)
    return


@app.cell
def _(X_test, X_train, y_test, y_train):
    def train_and_evaluate_xgboost(X_train, y_train, X_test, y_test):
        """
        Initializes, trains, and evaluates an XGBoost classifier.

        Args:
            X_train, y_train: Training data and labels.
            X_test, y_test: Testing data and labels.

        Returns:
            A tuple containing the trained model, training accuracy, and test accuracy.
        """
        import xgboost as xgb
        from sklearn.metrics import accuracy_score

        # Initialize the XGBoost classifier with specified parameters
        model = xgb.XGBClassifier(n_estimators=1000, 
                                  learning_rate=0.01, 
                                  max_depth=4, 
                                  random_state=1, 
                                  use_label_encoder=False)

        # Fit the model to the training data
        gbm = model.fit(X_train, y_train)

        # Make predictions on training and test sets
        y_train_pred = gbm.predict(X_train)
        y_test_pred = gbm.predict(X_test)

        # Calculate accuracies
        gbm_train_acc = accuracy_score(y_train, y_train_pred)
        gbm_test_acc = accuracy_score(y_test, y_test_pred)

        # Print the results
        print(f'XGBoost train/test accuracies: '
              f'{gbm_train_acc:.3f}/{gbm_test_acc:.3f}')
    
        return gbm, gbm_train_acc, gbm_test_acc


    # Assuming X_train, y_train, X_test, y_test are already defined

    # Call the function and capture the returned values
    trained_gbm, train_accuracy, test_accuracy = \
        train_and_evaluate_xgboost(X_train, y_train, X_test, y_test)

    # You can now use the trained model for other tasks
    # print(trained_gbm)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
