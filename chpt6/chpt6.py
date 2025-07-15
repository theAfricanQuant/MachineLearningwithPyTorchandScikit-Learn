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
        'pandas': '1.3.2'
    }
    check_packages(d)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Model Evaluation and Hyperparameter Tuning

    Hyperparameter tuning is a crucial aspect of building effective machine learning models, as highlighted in the provided text. After preparing data and selecting algorithms, the next step involves **fine-tuning** these algorithms to optimize their performance. This process is essential for overcoming common problems encountered in machine learning and ensuring the predictive models are robust.

    One extremely useful tool for streamlining this process in scikit-learn is the **`Pipeline`** class. A pipeline allows you to chain together multiple data transformation steps (like standardization or dimensionality reduction) with an estimator (your machine learning model) into a single object. This ensures that parameters learned during training for transformations (e.g., the mean and standard deviation for standardization, or the principal components for PCA) are consistently applied to new, unseen data, such as a test dataset. This prevents data leakage and ensures a consistent workflow from preprocessing to prediction.

    While scikit-learn offers convenience functions like `make_pipeline`, the text explicitly suggests using the `Pipeline` class directly. The key difference is that `Pipeline` requires you to explicitly name each step, providing more control and clarity, especially in more complex workflows or when you need to access specific steps by name for hyperparameter tuning. For example, you can define a pipeline as `Pipeline([('scaler', StandardScaler()), ('pca', PCA()), ('svc', SVC())])`, allowing you to later tune hyperparameters for the 'scaler', 'pca', or 'svc' steps. `make_pipeline` automatically names the steps based on their class names (e.g., `standardscaler`, `pca`), which might be less explicit when referencing them.

    In this chapter, we will be working with the **Breast Cancer Wisconsin dataset**. This dataset comprises 569 instances, each representing either a malignant or benign tumor. It includes 30 real-valued features derived from digitized images of cell nuclei, along with a unique ID and a diagnosis (M for malignant, B for benign). The goal is to build a model that can accurately predict whether a tumor is benign or malignant based on these features.
    """
    )
    return


@app.cell
def _():
    import pandas as pd

    df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                     'machine-learning-databases'
                     '/breast-cancer-wisconsin/wdbc.data', header=None)

    df.shape
    return (df,)


@app.cell
def _(df):
    from sklearn.preprocessing import LabelEncoder

    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    le.classes_
    return X, le, y


@app.cell
def _(le):
    le.transform(['M', 'B'])
    return


@app.cell
def _(X, y):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, 
                         test_size=0.20,
                         stratify=y,
                         random_state=1)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Many machine learning algorithms perform best with scaled input features. For the Breast Cancer Wisconsin dataset, which has features on different scales, we'll use **StandardScaler** to normalize them. Additionally, to reduce the 30-dimensional data to a 2-dimensional subspace, we'll employ **Principal Component Analysis (PCA)**, a feature extraction technique.

    Instead of applying these steps (scaling, PCA, then training a Logistic Regression classifier) separately to training and test data, we can chain them together using scikit-learn's **`Pipeline`** class. This automates the sequence, ensuring consistent transformations and preventing data leakage, ultimately simplifying and robustifying the machine learning workflow.
    """
    )
    return


@app.cell
def _(X_test, X_train, y_test, y_train):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline # Changed from make_pipeline

    pipe_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('log_reg', LogisticRegression())
    ])

    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
    test_acc = pipe_lr.score(X_test, y_test)
    print(f'Test accuracy: {test_acc:.3f}')
    return LogisticRegression, PCA, Pipeline, StandardScaler, pipe_lr, y_pred


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Cross-Validation for Model Performance Assessment

        Cross-validation techniques are crucial for obtaining reliable estimates of a model's generalization performance—how well it performs on unseen data.

        ## The Holdout Method

        The **holdout method** is a classic approach where the dataset is split into a **training set** and a **test set**. The training set is used to train the model, and the test set is used to estimate its generalization performance.

        However, for **model selection** (tuning hyperparameters), repeatedly using the same test set can lead to **overfitting** the test data, making the performance estimate biased.

        A better practice involves splitting the data into three parts:
        1.  **Training dataset**: Used to fit different models.
        2.  **Validation dataset**: Used for model selection (evaluating hyperparameter settings).
        3.  **Test dataset**: Held aside and used *only once* at the very end to obtain an unbiased estimate of the final model's generalization performance.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("Here's a flowchart illustrating the holdout method:")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
        graph TD
            A[Initial Dataset] --> B{Split Data};
            B -- Training Data --> C[Train Models with Different Hyperparameters];
            B -- Validation Data --> D[Evaluate Models on Validation Data];
            D --> E{Select Best Hyperparameters};
            E --> F[Train Final Model on Training Data with Best Hyperparameters];
            B -- Test Data --> G[Evaluate Final Model on Test Data];
            G --> H[Obtain Unbiased Performance Estimate];
        """
    )

    return


@app.cell(hide_code=True)
def _(mo):

    mo.md(
        """
        A disadvantage of the holdout method is its sensitivity to the specific data partition; the performance estimate can vary significantly with different splits.

        ## K-Fold Cross-Validation

        **K-fold cross-validation** offers a more robust performance estimation. It randomly splits the training dataset into *k* equal-sized "folds" without replacement.

        The procedure is as follows:
        1.  **Iterate *k* times**:
            * In each iteration, *k-1* folds are used as the **training folds**.
            * The remaining one fold is used as the **test fold** (or validation fold for hyperparameter tuning).
        2.  This yields *k* models and *k* performance estimates ($E_1, E_2, ..., E_k$).
        3.  The **average performance** (E) is calculated from these *k* estimates:
            $$E = \\frac{1}{k} \\sum_{i=1}^{k} E_i$$

        This average provides a performance estimate that is less sensitive to the data partitioning than the single holdout method. K-fold cross-validation is primarily used for **model tuning** (finding optimal hyperparameters).

        Once satisfactory hyperparameters are found, the model is typically retrained on the **complete training dataset** (all *k* folds) to obtain a single, final, and more robust model. The final generalization performance is then estimated using an **independent test dataset** that was never part of the cross-validation process.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("Here's a flowchart for k-fold cross-validation:")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
        graph TD
            A[Training Dataset] --> B{Split into k Folds};
            subgraph Iteration 1
                B1_Train[Fold 2, ..., Fold k] --> C1[Train Model 1];
                B1_Test[Fold 1] --> D1[Evaluate Model 1 (E1)];
            end
            subgraph Iteration 2
                B2_Train[Fold 1, Fold 3, ..., Fold k] --> C2[Train Model 2];
                B2_Test[Fold 2] --> D2[Evaluate Model 2 (E2)];
            end
            subgraph ...
                BX_Train[...] --> CX[Train Model X];
                BX_Test[...] --> DX[Evaluate Model X (EX)];
            end
            subgraph Iteration k
                BK_Train[Fold 1, ..., Fold k-1] --> CK[Train Model k];
                BK_Test[Fold k] --> DK[Evaluate Model k (Ek)];
            end
            DK --> E[Calculate Average Performance E = (E1 + ... + Ek) / k
            $$E = \\frac{1}{k} \\sum_{i=1}^{k} E_i$$
            ];
            E --> F[Select Best Hyperparameters];
            F --> G[Retrain Final Model on Full Training Dataset];
            G --> H[Evaluate Final Model on Independent Test Dataset];
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
        graph TD
            A[Training Dataset] --> B{Split into k Folds};
            subgraph Iteration 1
                B1_Train[Fold 2, ..., Fold k] --> C1[Train Model 1];
                B1_Test[Fold 1] --> D1[Evaluate Model 1];
            end
            DK --> E[Calculate Average Performance];
            E --> F[Select Best Hyperparameters];
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        A common standard value for *k* is **10**, offering a good balance between bias and variance. For small datasets, increasing *k* can be beneficial (e.g., **Leave-One-Out Cross-Validation (LOOCV)** where *k = n*, the number of training examples). For large datasets, a smaller *k* (e.g., 5) can reduce computational cost while still providing an accurate estimate.

        ### Stratified K-Fold Cross-Validation

        A key improvement is **stratified k-fold cross-validation**. This method ensures that the **class label proportions** are preserved in each fold, making each fold representative of the overall class distribution in the training dataset. This is particularly beneficial for datasets with **unequal class proportions**, leading to more reliable bias and variance estimates.
        """
    )
    return


@app.cell
def _(X_train, pipe_lr, y_train):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    

    kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)

    scores = []
    for k, (train, test) in enumerate(kfold):
        pipe_lr.fit(X_train[train], y_train[train])
        score = pipe_lr.score(X_train[test], y_train[test])
        scores.append(score)

        print(f'Fold: {k+1:02d}, '
              f'Class distr.: {np.bincount(y_train[train])}, '
              f'Acc.: {score:.3f}')
    
    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    print(f'\nCV accuracy: {mean_acc:.3f} +/- {std_acc:.3f}')
    return StratifiedKFold, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        Using `StratifiedKFold` from `sklearn.model_selection`, we initialized an iterator with `y_train` and `n_splits` to generate indices for K-fold cross-validation. In each fold, `train` indices were used to fit our `pipe_lr` (which handles scaling and logistic regression), and `test` indices were used to calculate the accuracy. These accuracy scores were collected to compute the average accuracy and its standard deviation, illustrating the K-fold process. Scikit-learn offers a more concise built-in cross-validation scorer for this purpose.
        """
    )
    return


@app.cell
def _(X_train, pipe_lr, y_train):
    from sklearn.model_selection import cross_val_score

    cv_scores = cross_val_score(estimator=pipe_lr,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             n_jobs=1)

    return cross_val_score, cv_scores


@app.cell
def _(cv_scores):
    print(f'CV accuracy scores: {cv_scores}')

    return


@app.cell
def _(cv_scores, np):
    print(f'CV accuracy: {np.mean(cv_scores):.3f} '
          f'+/- {np.std(cv_scores):.3f}')
    return


@app.cell
def _(LogisticRegression, StandardScaler, X_train, make_pipeline, np, y_train):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve


    pipe_lr = make_pipeline(StandardScaler(),
                            LogisticRegression(penalty='l2', max_iter=10000))

    train_sizes, train_scores, test_scores =\
                    learning_curve(estimator=pipe_lr,
                                   X=X_train,
                                   y=y_train,
                                   train_sizes=np.linspace(0.1, 1.0, 10),
                                   cv=10,
                                   n_jobs=1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='Training accuracy')

    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='Validation accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training examples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.03])
    plt.tight_layout()
    # plt.savefig('figures/06_05.png', dpi=300)
    plt.show()
    return (
        pipe_lr,
        plt,
        test_mean,
        test_std,
        train_mean,
        train_sizes,
        train_std,
    )


@app.cell
def _(LogisticRegression, Pipeline, StandardScaler, X_train, np, y_train):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve

    def calculate_learning_curve_data(X_train, y_train, cv=10, n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 10)):
        """
        Calculates the data needed to plot a learning curve for a Logistic Regression model
        with a specific pipeline configuration.

        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training target labels.
            cv (int, cross-validation generator or an iterable, optional):
                Determines the cross-validation splitting strategy. Defaults to 10.
            n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 1.
            train_sizes (array-like, optional): Relative or absolute numbers of training
                examples that will be used to generate the learning curve.
                Defaults to np.linspace(0.1, 1.0, 10).

        Returns:
            tuple: A tuple containing (train_sizes, train_mean, train_std, test_mean, test_std).
        """
        # Redefine pipe_lr *within* this function's scope for this specific learning curve
        # This ensures it doesn't interfere with a globally defined pipe_lr for other tasks.
        pipe_lr_for_curve = Pipeline([
            ('scaler', StandardScaler()),
            ('log_reg', LogisticRegression(penalty='l2', max_iter=10000, random_state=1)) # Added random_state for reproducibility
        ])

        train_sizes, train_scores, test_scores = learning_curve(
            estimator=pipe_lr_for_curve, # Use the local pipeline
            X=X_train,
            y=y_train,
            train_sizes=train_sizes,
            cv=cv,
            n_jobs=n_jobs
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        return train_sizes, train_mean, train_std, test_mean, test_std



    # Call the function to get the data for plotting
    train_sizes, train_mean, train_std, test_mean, test_std = \
        calculate_learning_curve_data(X_train, y_train)


    return plt, test_mean, test_std, train_mean, train_sizes, train_std


@app.cell
def _(plt, test_mean, test_std, train_mean, train_sizes, train_std):
    # Now, use your existing plotting code in a separate Marimo cell:
    fig, ax = plt.subplots(figsize=(8, 6)) # Create a figure and an axes object

    ax.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='Training accuracy')

    ax.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    ax.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='Validation accuracy')

    ax.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    ax.grid()
    ax.set_xlabel('Number of training examples')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='lower right')
    ax.set_ylim([0.8, 1.03])
    fig.tight_layout() # Use fig.tight_layout() for the figure

    # Marimo will automatically render the figure object returned from a cell.
    fig
    return


@app.cell
def _(LogisticRegression, Pipeline, StandardScaler, np):
    from sklearn.model_selection import validation_curve

    def calculate_validation_curve_data(X_train, y_train, param_range, cv=10):
        """
        Calculates the data needed to plot a validation curve for the 'C' parameter
        of a Logistic Regression model within a specific pipeline configuration.

        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training target labels.
            param_range (array-like): The range of parameter values to explore.
            cv (int, cross-validation generator or an iterable, optional):
                Determines the cross-validation splitting strategy. Defaults to 10.

        Returns:
            tuple: A tuple containing (param_range, train_mean, train_std, test_mean, test_std).
        """
        # Define pipe_lr *within* this function's scope for this specific validation curve
        # This ensures it uses the intended configuration and doesn't conflict with
        # other pipe_lr definitions outside this function.
        pipe_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('log_reg', LogisticRegression(penalty='l2', max_iter=10000, random_state=1))
        ])

        train_scores, test_scores = validation_curve(
            estimator=pipe_lr,
            X=X_train,
            y=y_train,
            param_name='log_reg__C', # Note: 'logisticregression' is the default name for LogisticRegression in a Pipeline
            param_range=param_range,
            cv=cv,
            n_jobs=-1 # Use all available CPU cores for faster computation
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        return param_range, train_mean, train_std, test_mean, test_std

    return (calculate_validation_curve_data,)


@app.cell
def _(X_train, calculate_validation_curve_data, plt, y_train):
    def _():
        # Define the parameter range
        param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

        # Call the function to get the data for plotting
        param_range_plot, train_mean, train_std, test_mean, test_std = \
            calculate_validation_curve_data(X_train, y_train, param_range=param_range)

        # Now, use your existing plotting code in a separate Marimo cell:
        fig, ax = plt.subplots(figsize=(8, 6)) # Create a figure and an axes object

        ax.plot(param_range_plot, train_mean,
                 color='blue', marker='o',
                 markersize=5, label='Training accuracy')

        ax.fill_between(param_range_plot, train_mean + train_std,
                         train_mean - train_std, alpha=0.15,
                         color='blue')

        ax.plot(param_range_plot, test_mean,
                 color='green', linestyle='--',
                 marker='s', markersize=5,
                 label='Validation accuracy')

        ax.fill_between(param_range_plot,
                         test_mean + test_std,
                         test_mean - test_std,
                         alpha=0.15, color='green')

        ax.grid()
        ax.set_xscale('log') # Set x-axis to log scale
        ax.legend(loc='lower right')
        ax.set_xlabel('Parameter C')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0.8, 1.0])
        fig.tight_layout() # Use fig.tight_layout() for the figure

        # Marimo will automatically render the figure object returned from a cell.
        return fig


    _()
    return


@app.cell
def _(Pipeline, StandardScaler, X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    pipe_svc = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(random_state=1)) # Named the SVC step 'svc'
    ])

    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    param_grid = [{'svc__C': param_range,
                   'svc__kernel': ['linear']},
                  {'svc__C': param_range,
                   'svc__gamma': param_range,
                   'svc__kernel': ['rbf']}]

    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      scoring='accuracy',
                      refit=True,
                      cv=10)

    gs = gs.fit(X_train, y_train)

    print(gs.best_score_)
    print(gs.best_params_)
    return GridSearchCV, SVC, gs, param_range, pipe_svc


@app.cell
def _(X_test, gs, y_test):
    clf = gs.best_estimator_

    print(f'Test accuracy: {clf.score(X_test, y_test):.3f}')
    return


@app.cell
def _(Pipeline, SVC, StandardScaler):
    from sklearn.model_selection import RandomizedSearchCV

    # This cell defines the function. The function itself does not output anything directly to Marimo.
    def get_randomized_search_results(X_train, y_train, param_range):
        """
        Performs RandomizedSearchCV and returns the fitted RandomizedSearchCV object.
        This function encapsulates the internal definitions (pipe_svc, param_grid, rs)
        to avoid redefinition errors with global variables in Marimo.
        """
        pipe_svc = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(random_state=1))
        ])

        param_grid = [{'svc__C': param_range,
                       'svc__kernel': ['linear']},
                      {'svc__C': param_range,
                       'svc__gamma': param_range,
                       'svc__kernel': ['rbf']}]

        rs = RandomizedSearchCV(estimator=pipe_svc,
                                param_distributions=param_grid,
                                scoring='accuracy',
                                refit=True,
                                n_iter=20,
                                cv=10,
                                random_state=1,
                                n_jobs=-1)

        # Fit the model here, as this function is designed to *perform* the search
        rs.fit(X_train, y_train)

        # Return the fitted RandomizedSearchCV object
        return rs
    return (get_randomized_search_results,)


@app.cell
def _(X_train, get_randomized_search_results, mo, param_range, y_train):
    # This cell calls the function and makes its return value the cell's output.
    # The variable 'fitted_rs_model' will now be accessible in subsequent cells.
    fitted_rs_model = get_randomized_search_results(X_train, y_train, param_range)

    # You can also display properties of the returned object directly from this cell
    mo.md(f"Best score found: **{fitted_rs_model.best_score_:.4f}**")
    return (fitted_rs_model,)


@app.cell
def _(fitted_rs_model, mo):
    mo.md(f"Best parameters: `{fitted_rs_model.best_params_}`")
    return


@app.cell
def _():
    import scipy.stats

    scipy.stats.loguniform(0.0001, 1000.0)
    return


@app.cell
def _():
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingRandomSearchCV
    return (HalvingRandomSearchCV,)


@app.cell
def _(HalvingRandomSearchCV, X_train, param_range, pipe_svc, y_train):
    param_grid_ = [{'svc__C': param_range,
                   'svc__kernel': ['linear']},
                  {'svc__C': param_range,
                   'svc__gamma': param_range,
                   'svc__kernel': ['rbf']}]

    hs = HalvingRandomSearchCV(
        pipe_svc,
        param_distributions=param_grid_,
        n_candidates='exhaust',
        resource='n_samples',
        factor=1.5,
        random_state=1,
        n_jobs=-1)

    hs = hs.fit(X_train, y_train)
    print(hs.best_score_)
    print(hs.best_params_)
    return (hs,)


@app.cell
def _(X_test, hs, y_test):
    clf1 = hs.best_estimator_
    print(f'Test accuracy: {hs.score(X_test, y_test):.3f}')
    return


@app.cell
def _(GridSearchCV, Pipeline, SVC, StandardScaler, cross_val_score, np):
    def perform_nested_cv_grid_search(X_train, y_train, param_range, inner_cv_folds=2, outer_cv_folds=5):
        """
        Performs nested cross-validation with GridSearchCV for an SVC model.

        The inner loop (GridSearchCV) tunes hyperparameters.
        The outer loop (cross_val_score) evaluates the generalization performance
        of the entire tuning process.

        Args:
            X_train (array-like): Training features for the outer CV.
            y_train (array-like): Training target labels for the outer CV.
            param_range (list): The range of parameter values for C and gamma.
            inner_cv_folds (int): Number of folds for the inner cross-validation loop (GridSearchCV).
            outer_cv_folds (int): Number of folds for the outer cross-validation loop (cross_val_score).

        Returns:
            tuple: A tuple containing (mean_accuracy, std_accuracy) from the outer CV scores.
        """

        # 1. Define the Pipeline for the estimator
        # This pipeline will be used by GridSearchCV
        pipe_svc = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(random_state=1)) # Naming the SVC step 'svc' is crucial
        ])

        # 2. Define the parameter grid for the inner GridSearchCV
        param_grid = [{'svc__C': param_range,
                       'svc__kernel': ['linear']},
                      {'svc__C': param_range,
                       'svc__gamma': param_range,
                       'svc__kernel': ['rbf']}]

        # 3. Create the GridSearchCV object (inner cross-validation loop)
        # This 'gs' object is itself an estimator that performs hyperparameter tuning
        gs_inner_loop = GridSearchCV(estimator=pipe_svc,
                                     param_grid=param_grid,
                                     scoring='accuracy',
                                     cv=inner_cv_folds, # Inner CV folds
                                     n_jobs=-1, # Use all available cores for the inner loop
                                     refit=True) # Refit=True is default and important for cross_val_score

        # 4. Perform the outer cross-validation using cross_val_score
        # Pass the GridSearchCV object as the estimator
        scores = cross_val_score(estimator=gs_inner_loop, # GridSearchCV object is the estimator here
                                 X=X_train,
                                 y=y_train,
                                 scoring='accuracy',
                                 cv=outer_cv_folds, # Outer CV folds
                                 n_jobs=-1) # Use all available cores for the outer loop's fits

        mean_accuracy = np.mean(scores)
        std_accuracy = np.std(scores)

        return mean_accuracy, std_accuracy

    return (perform_nested_cv_grid_search,)


@app.cell
def _(X_train, param_range, perform_nested_cv_grid_search, y_train):
    # Example function call:
    nested_mean_acc, nested_std_acc = perform_nested_cv_grid_search(
        X_train, y_train, param_range, inner_cv_folds=2, outer_cv_folds=5
    )
    return nested_mean_acc, nested_std_acc


@app.cell
def _(mo, nested_mean_acc, nested_std_acc):
    mo.md(f'Nested CV accuracy: {nested_mean_acc:.3f} +/- {nested_std_acc:.3f}')
    return


@app.cell
def _(GridSearchCV, X_train, cross_val_score, np, y_train):
    from sklearn.tree import DecisionTreeClassifier

    dtcgs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                      param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                      scoring='accuracy',
                      cv=2)

    scores_dtc = cross_val_score(dtcgs, X_train, y_train, 
                             scoring='accuracy', cv=5)
    print(f'CV accuracy: {np.mean(scores_dtc):.3f} '
          f'+/- {np.std(scores_dtc):.3f}')
    return


@app.cell(hide_code=True)
def _(mo):

    mo.md(
        """
        ## The Confusion Matrix: Unpacking Classifier Performance

        A **confusion matrix** is a tabular summary of the prediction results on a classification problem. It allows visualization of the performance of an algorithm, and it's particularly useful when the classes are imbalanced or when the costs of different types of errors are not equal.

        It's a square matrix where rows represent the **actual classes** and columns represent the **predicted classes**. For a binary classification problem (like predicting "positive" or "negative"), it typically looks like this:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
        graph TD
            subgraph Confusion Matrix
                A[Actual: Positive] --> B[Predicted: Positive];
                A --> C[Predicted: Negative];
                D[Actual: Negative] --> E[Predicted: Positive];
                D --> F[Predicted: Negative];
            end
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("Let's clarify the four key terms within the confusion matrix:")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("### 1. True Positives (TP)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        This occurs when the model **correctly predicts the positive class**. The actual class was positive, and the model predicted positive.

        * **Example:** A patient *actually has* cancer, and the model *correctly predicts* they have cancer.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
        %%{init: {'theme': 'default'}}%%
        graph TD
            subgraph Confusion_Matrix_Highlighting_TP
                A[Actual: Positive] --> B[Predicted: Positive]
                A --> C[Predicted: Negative]
                D[Actual: Negative] --> E[Predicted: Positive]
                D --> F[Predicted: Negative]
            end

            class A,B tpNode;

            classDef tpNode fill:#D4EDDA,stroke:#28A745,stroke-width:2px;
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("### 2. True Negatives (TN)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        This occurs when the model **correctly predicts the negative class**. The actual class was negative, and the model predicted negative.

        * **Example:** A patient *does not have* cancer, and the model *correctly predicts* they do not have cancer.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
        graph TD
            subgraph Confusion Matrix (Highlighting TN)
                style D fill:#D4EDDA,stroke:#28A745,stroke-width:2px
                style F fill:#D4EDDA,stroke:#28A745,stroke-width:2px
                A[Actual: Positive] --> B[Predicted: Positive];
                A --> C[Predicted: Negative];
                D[Actual: Negative] --> E[Predicted: Positive];
                D --> F[Predicted: Negative];
            end
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("### 3. False Positives (FP) - Type I Error")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        This occurs when the model **incorrectly predicts the positive class**. The actual class was negative, but the model predicted positive. This is often referred to as a "Type I error."

        * **Example:** A patient *does not have* cancer, but the model *incorrectly predicts* they have cancer (a "false alarm").
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
        %%{init: {'theme': 'default'}}%%
        graph TD
            subgraph Confusion_Matrix_Highlighting_FP
                A[Actual: Positive] --> B[Predicted: Positive]
                A --> C[Predicted: Negative]
                D[Actual: Negative] --> E[Predicted: Positive]
                D --> F[Predicted: Negative]
            end

            class D,E fpNode;

            classDef fpNode fill:#F8D7DA,stroke:#DC3545,stroke-width:2px;
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("### 4. False Negatives (FN) - Type II Error")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        This occurs when the model **incorrectly predicts the negative class**. The actual class was positive, but the model predicted negative. This is often referred to as a "Type II error."

        * **Example:** A patient *actually has* cancer, but the model *incorrectly predicts* they do not have cancer (a "missed detection" or "miss").
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
        %%{init: {'theme': 'default'}}%%
        graph TD
            subgraph Confusion_Matrix_Highlighting_FN
                A[Actual: Positive] --> B[Predicted: Positive]
                A --> C[Predicted: Negative]
                D[Actual: Negative] --> E[Predicted: Positive]
                D --> F[Predicted: Negative]
            end

            class A,C fnNode;

            classDef fnNode fill:#F8D7DA,stroke:#DC3545,stroke-width:2px;
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        By examining the counts in each of these four cells, a confusion matrix provides a much richer understanding of a classifier's strengths and weaknesses than simple overall accuracy alone. For instance, in medical diagnoses, false negatives (missing a disease) are often far more critical than false positives (a false alarm).
        """
    )
    return


@app.cell
def _(X_test, X_train, np, pipe_svc, plt, y_test, y_train):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    pipe_svc.fit(X_train, y_train)
    # y_pred = pipe_svc.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=pipe_svc.predict(X_test))

    # Labels
    group_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    group_counts = [f"{value}" for value in confmat.flatten()]
    labels = [f"{name}\n{count}" for name, count in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)

    # Plot
    plt.figure(figsize=(6, 4))
    sns.heatmap(confmat, annot=labels, fmt='', cmap='YlGnBu', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # 📊 Accuracy and Error

    - **Accuracy (ACC)**: Proportion of correctly predicted samples.
    - **Error (ERR)**: Proportion of incorrectly predicted samples.

    \\[
    \\text{ERR} = \\frac{FP + FN}{TP + TN + FP + FN} \\\\
    \\text{ACC} = \\frac{TP + TN}{TP + TN + FP + FN} \\\\
    \\text{ACC} = 1 - \\text{ERR}
    \\]
    """
    )

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 🧪 True Positive Rate (TPR) and False Positive Rate (FPR)

    Useful for **imbalanced class problems**:

    - **True Positive Rate (TPR)** = Recall = Sensitivity

    \[
    \text{TPR} = \frac{TP}{TP + FN}
    \]

    - **False Positive Rate (FPR)**:

    \[
    \text{FPR} = \frac{FP}{FP + TN}
    \]

    **Use case:** In tumor diagnosis:

    - High **TPR** ensures malignant tumors are detected.

    - Low **FPR** avoids alarming healthy patients.

    """
    )

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 🎯 Precision and Recall

    - **Recall (REC)**: Fraction of true positives among all actual positives.
    - **Precision (PRE)**: Fraction of true positives among all predicted positives.

    \[
    \text{REC} = \frac{TP}{TP + FN}
    \quad
    \text{PRE} = \frac{TP}{TP + FP}
    \]

    **Trade-off:**

    - High **Recall** reduces missed detections (malignant tumors).

    - High **Precision** reduces false alarms (wrongly predicted malignancy).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(text=r"""
    # ⚖️ F1 Score: Balancing Precision and Recall

    The **F1 Score** is the harmonic mean of precision and recall.

    \[
    F1 = 2 \cdot \frac{PRE \cdot REC}{PRE + REC}
    \]

    - High F1 means a good balance between precision and recall.
    - Useful when you need a **single metric** that considers both FP and FN.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 🧪 Matthews Correlation Coefficient (MCC)

    A robust metric that considers **all confusion matrix elements**:

    \[
    \text{MCC} = \frac{(TP \cdot TN) - (FP \cdot FN)}
    {\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
    \]

    - Ranges from **–1** (total disagreement) to **+1** (perfect prediction).
    - Unlike F1, MCC includes **True Negatives (TN)**.
    - Favored in **biological and medical classification** tasks.
    """

    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 📚 Further Reading

    - **Precision, Recall, F-Factor, and Beyond**:  
      David M. W. Powers (2011),  
      [arXiv:2010.16061](https://arxiv.org/abs/2010.16061)

    - **MCC vs. F1 Score**:  
      D. Chicco & G. Jurman (2019),  
      [BMC Genomics Article](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7)
    """

         )
    return


@app.cell
def _(y_pred, y_test):
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.metrics import matthews_corrcoef

    pre_val = precision_score(y_true=y_test, y_pred=y_pred)
    print(f'Precision: {pre_val:.3f}')

    rec_val = recall_score(y_true=y_test, y_pred=y_pred)
    print(f'Recall: {rec_val:.3f}')

    f1_val = f1_score(y_true=y_test, y_pred=y_pred)
    print(f'F1: {f1_val:.3f}')

    mcc_val = matthews_corrcoef(y_true=y_test, y_pred=y_pred)
    print(f'MCC: {mcc_val:.3f}')
    return f1_score, precision_score


@app.cell
def _(GridSearchCV, X_train, f1_score, pipe_svc, y_train):
    from sklearn.metrics import make_scorer

    scorer = make_scorer(f1_score, pos_label=0)

    c_gamma_range = [0.01, 0.1, 1.0, 10.0]

    param_grid_cv = [{'svc__C': c_gamma_range,
                   'svc__kernel': ['linear']},
                  {'svc__C': c_gamma_range,
                   'svc__gamma': c_gamma_range,
                   'svc__kernel': ['rbf']}]

    gscv = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid_cv,
                      scoring=scorer,
                      cv=10,
                      n_jobs=-1)
    gscv = gscv.fit(X_train, y_train)
    print(gscv.best_score_)
    print(gscv.best_params_)
    return (make_scorer,)


@app.cell
def _(
    LogisticRegression,
    PCA,
    Pipeline,
    StandardScaler,
    StratifiedKFold,
    X_train,
    np,
    plt,
    y_train,
):
    def _():
        from sklearn.metrics import roc_curve, auc
        from numpy import interp


        pipe_lr = Pipeline([
        ('sc', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('clf', LogisticRegression(penalty='l2', 
                                   random_state=1,
                                   solver='lbfgs',
                                   C=100.0))
         ])


        X_train2 = X_train[:, [4, 14]]
        

        cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))

        fig = plt.figure(figsize=(7, 5))

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []

        for i, (train, test) in enumerate(cv):
            probas = pipe_lr.fit(X_train2[train],
                                 y_train[train]).predict_proba(X_train2[test])

            fpr, tpr, thresholds = roc_curve(y_train[test],
                                             probas[:, 1],
                                             pos_label=1)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr,
                     tpr,
                     label=f'ROC fold {i+1} (area = {roc_auc:.2f})')

        plt.plot([0, 1],
                 [0, 1],
                 linestyle='--',
                 color=(0.6, 0.6, 0.6),
                 label='Random guessing (area = 0.5)')

        mean_tpr /= len(cv)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k--',
                 label=f'Mean ROC (area = {mean_auc:.2f})', lw=2)
        plt.plot([0, 0, 1],
                 [0, 1, 1],
                 linestyle=':',
                 color='black',
                 label='Perfect performance (area = 1.0)')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(loc='lower right')

        plt.tight_layout()
        # plt.savefig('figures/06_10.png', dpi=300)
        return plt.show()


    _()
    return


@app.cell
def _(make_scorer, precision_score):
    pre_scorer = make_scorer(score_func=precision_score, 
                             pos_label=1, 
                             greater_is_better=True, 
                             average='micro')
    pre_scorer
    return


@app.cell
def _(X, np, y):
    X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
    y_imb = np.hstack((y[y == 0], y[y == 1][:40]))

    y_pred1 = np.zeros(y_imb.shape[0])
    np.mean(y_pred1 == y_imb) * 100
    return X_imb, y_imb


@app.cell
def _(X_imb, y_imb):
    from sklearn.utils import resample

    print('Number of class 1 examples before:', X_imb[y_imb == 1].shape[0])

    X_upsampled, y_upsampled = resample(X_imb[y_imb == 1],
                                        y_imb[y_imb == 1],
                                        replace=True,
                                        n_samples=X_imb[y_imb == 0].shape[0],
                                        random_state=123)

    print('Number of class 1 examples after:', X_upsampled.shape[0])
    return X_upsampled, y_upsampled


@app.cell
def _(X, X_upsampled, np, y, y_upsampled):
    X_bal = np.vstack((X[y == 0], X_upsampled))
    y_bal = np.hstack((y[y == 0], y_upsampled))

    y_pred2 = np.zeros(y_bal.shape[0])
    np.mean(y_pred2== y_bal) * 100
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
