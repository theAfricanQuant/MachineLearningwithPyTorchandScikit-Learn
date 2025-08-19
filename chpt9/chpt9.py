import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    sys.path.insert(0, '..')

    from python_environment_check import check_packages


    d = {
        'numpy': '1.21.2',
        'mlxtend': '0.19.0',
        'matplotlib': '3.4.3',
        'sklearn': '1.0',
        'pandas': '1.3.2',
    }
    check_packages(d)

    return


@app.cell
def _():
    import pandas as pd


    columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
               'Central Air', 'Total Bsmt SF', 'SalePrice']



    df = (
        pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt', 
                     sep='\t',
                     usecols=columns)
        # The chained version using .assign()
        .assign(**{'Central Air': lambda d: d['Central Air'].map({'N': 0, 'Y': 1})})
        
        # remove rows that contain missing values
        .dropna(axis=0)
    )

    df.isnull().sum()
    return (df,)


@app.cell
def _(df):
    import matplotlib.pyplot as plt
    from mlxtend.plotting import scatterplotmatrix

    scatterplotmatrix(df.values, figsize=(12, 10), 
                      names=df.columns, alpha=0.5)
    plt.tight_layout()
    #plt.savefig('figures/09_04.png', dpi=300)
    plt.show()
    return (plt,)


@app.cell
def _(df, plt):
    import numpy as np
    from mlxtend.plotting import heatmap


    cm = np.corrcoef(df.values.T)
    hm = heatmap(cm, row_names=df.columns, column_names=df.columns)

    plt.tight_layout()
    #plt.savefig('figures/09_05.png', dpi=300)
    plt.show()
    return (np,)


@app.cell
def _(df):
    (
        df
        .corr()
        .style
        .background_gradient(cmap="RdBu", vmax=1, vmin=-1)
        .set_sticky(axis="")
    )
    return


@app.cell
def _(df):
    (
        df
        .corr(method="spearman")
        .style
        .background_gradient(cmap="RdBu", vmax=1, vmin=-1)
        .set_sticky(axis="index")
    )
    return


@app.cell
def _(np):
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
    return (LinearRegressionGD,)


@app.cell
def _(df):
    X = df[['Gr Liv Area']].values
    y = df['SalePrice'].values
    return X, y


@app.cell
def _(X, np, y):
    from sklearn.preprocessing import StandardScaler


    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_std = sc_x.fit_transform(X)
    y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
    return X_std, sc_x, sc_y, y_std


@app.cell
def _(LinearRegressionGD, X_std, y_std):
    lr = LinearRegressionGD(eta=0.1)
    lr.fit(X_std, y_std)
    return (lr,)


@app.cell
def _(lr, plt):
    plt.plot(range(1, lr.n_iter+1), lr.losses_)
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.tight_layout()
    #plt.savefig('figures/09_06.png', dpi=300)
    plt.show()
    return


@app.cell
def _(plt):
    def lin_regplot(X, y, model):
        plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
        plt.plot(X, model.predict(X), color='black', lw=2)    
        return 
    return (lin_regplot,)


@app.cell
def _(X_std, lin_regplot, lr, plt, y_std):
    lin_regplot(X_std, y_std, lr)
    plt.xlabel('Living area above ground (standardized)')
    plt.ylabel('Sale price (standardized)')

    plt.show()
    return


@app.cell
def _(lr, np, sc_x, sc_y):
    feature_std = sc_x.transform(np.array([[2500]]))
    target_std = lr.predict(feature_std)
    target_reverted = sc_y.inverse_transform(target_std.reshape(-1, 1))
    print(f'Sale price: ${target_reverted.flatten()[0]:.2f}')
    return


@app.cell
def _(lr):
    print(f'Slope: {lr.w_[0]:.3f}')
    print(f'Intercept: {lr.b_[0]:.3f}')
    return


@app.cell
def _(X, y):
    from sklearn.linear_model import LinearRegression


    slr = LinearRegression()
    slr.fit(X, y)
    y_pred = slr.predict(X)
    print(f'Slope: {slr.coef_[0]:.3f}')
    print(f'Intercept: {slr.intercept_:.3f}')
    return LinearRegression, slr


@app.cell
def _(X, lin_regplot, plt, slr, y):
    lin_regplot(X, y, slr)
    plt.xlabel('Living area above ground in square feet')
    plt.ylabel('Sale price in U.S. dollars')

    plt.tight_layout()

    plt.show()
    return


@app.cell
def _(X, np, y):
    # adding a column vector of "ones"
    Xb = np.hstack((np.ones((X.shape[0], 1)), X))
    w = np.zeros(X.shape[1])
    z = np.linalg.inv(np.dot(Xb.T, Xb))
    w = np.dot(z, np.dot(Xb.T, y))

    print(f'Slope: {w[1]:.3f}')
    print(f'Intercept: {w[0]:.3f}')
    return


@app.cell
def _(LinearRegression, X, np, plt, y):
    from sklearn.linear_model import RANSACRegressor


    ransac = RANSACRegressor(LinearRegression(), 
                             max_trials=100, # default
                             min_samples=0.95, 
                             loss='absolute_error', # default
                             residual_threshold=None, # default 
                             random_state=123)


    ransac.fit(X, y)

    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    line_X = np.arange(3, 10, 1)
    line_y_ransac = ransac.predict(line_X[:, np.newaxis])
    plt.scatter(X[inlier_mask], y[inlier_mask],
                c='steelblue', edgecolor='white', 
                marker='o', label='Inliers')
    plt.scatter(X[outlier_mask], y[outlier_mask],
                c='limegreen', edgecolor='white', 
                marker='s', label='Outliers')
    plt.plot(line_X, line_y_ransac, color='black', lw=2)   
    plt.xlabel('Living area above ground in square feet')
    plt.ylabel('Sale price in U.S. dollars')
    plt.legend(loc='upper left')

    plt.tight_layout()

    plt.show()
    return RANSACRegressor, inlier_mask, line_X, ransac


@app.cell
def _(ransac):
    print(f'Slope: {ransac.estimator_.coef_[0]:.3f}')
    print(f'Intercept: {ransac.estimator_.intercept_:.3f}')
    return


@app.cell
def _(np, y):
    def median_absolute_deviation(data):
        return np.median(np.abs(data - np.median(data)))
    
    median_absolute_deviation(y)
    return


@app.cell
def _(LinearRegression, RANSACRegressor, X, inlier_mask, line_X, np, plt, y):
    ransac2 = RANSACRegressor(LinearRegression(), 
                             max_trials=100, # default
                             min_samples=0.95, 
                             loss='absolute_error', # default
                             residual_threshold=65000, # default 
                             random_state=123)

    ransac2.fit(X, y)

    inlier_mask2 = ransac2.inlier_mask_
    outlier_mask2 = np.logical_not(inlier_mask)

    line_X2 = np.arange(3, 10, 1)
    line_y_ransac2 = ransac2.predict(line_X[:, np.newaxis])
    plt.scatter(X[inlier_mask2], y[inlier_mask2],
                c='steelblue', edgecolor='white', 
                marker='o', label='Inliers')
    plt.scatter(X[outlier_mask2], y[outlier_mask2],
                c='limegreen', edgecolor='white', 
                marker='s', label='Outliers')
    plt.plot(line_X2, line_y_ransac2, color='black', lw=2)   
    plt.xlabel('Living area above ground in square feet')
    plt.ylabel('Sale price in U.S. dollars')
    plt.legend(loc='upper left')

    plt.tight_layout()

    plt.show()
    return (ransac2,)


@app.cell
def _(ransac2):
    print(f'Slope: {ransac2.estimator_.coef_[0]:.3f}')
    print(f'Intercept: {ransac2.estimator_.intercept_:.3f}')
    return


@app.cell
def _(df):
    from sklearn.model_selection import train_test_split

    target = 'SalePrice'
    features = df.columns[df.columns != target]


    X_nu = df[features].values
    y_nu = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_nu, y_nu, test_size=0.3, random_state=123)
    return X_test, X_train, train_test_split, y_test, y_train


@app.cell
def _(LinearRegression, X_test, X_train, y_train):

    slr_nu = LinearRegression()

    slr_nu.fit(X_train, y_train)
    y_train_pred = slr_nu.predict(X_train)
    y_test_pred = slr_nu.predict(X_test)
    return y_test_pred, y_train_pred


@app.cell
def _(np, plt, y_test, y_test_pred, y_train, y_train_pred):
    x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
    x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

    ax1.scatter(y_test_pred, y_test_pred - y_test,
                c='limegreen', marker='s', edgecolor='white',
                label='Test data')
    ax2.scatter(y_train_pred, y_train_pred - y_train,
                c='steelblue', marker='o', edgecolor='white',
                label='Training data')
    ax1.set_ylabel('Residuals')

    for ax in (ax1, ax2):
        ax.set_xlabel('Predicted values')
        ax.legend(loc='upper left')
        ax.hlines(y=0, xmin=x_min-100, xmax=x_max+100, color='black', lw=2)

    plt.tight_layout()


    plt.show()
    return


@app.cell
def _(y_test, y_test_pred, y_train, y_train_pred):
    from sklearn.metrics import mean_squared_error


    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    print(f'MSE train: {mse_train:.2f}')
    print(f'MSE test: {mse_test:.2f}')
    return (mean_squared_error,)


@app.cell
def _(y_test, y_test_pred, y_train, y_train_pred):
    from sklearn.metrics import mean_absolute_error


    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    print(f'MAE train: {mae_train:.2f}')
    print(f'MAE test: {mae_test:.2f}')
    return (mean_absolute_error,)


@app.cell
def _(y_test, y_test_pred, y_train, y_train_pred):
    from sklearn.metrics import r2_score


    r2_train = r2_score(y_train, y_train_pred)
    r2_test =r2_score(y_test, y_test_pred)
    print(f'R^2 train: {r2_train:.2f}')
    print(f'R^2 test: {r2_test:.2f}')
    return (r2_score,)


@app.cell
def _(X_test, X_train, y_train):
    from sklearn.linear_model import Lasso


    lasso = Lasso(alpha=1.0)
    lasso.fit(X_train, y_train)
    y_train_pred_lasso = lasso.predict(X_train)
    y_test_pred_lasso = lasso.predict(X_test)
    print(lasso.coef_)
    return y_test_pred_lasso, y_train_pred_lasso


@app.cell
def _(
    mean_squared_error,
    r2_score,
    y_test,
    y_test_pred_lasso,
    y_train,
    y_train_pred_lasso,
):
    train_mse = mean_squared_error(y_train, y_train_pred_lasso)
    test_mse = mean_squared_error(y_test, y_test_pred_lasso)
    print(f'MSE train: {train_mse:.3f}, test: {test_mse:.3f}')

    train_r2 = r2_score(y_train, y_train_pred_lasso)
    test_r2 = r2_score(y_test, y_test_pred_lasso)
    print(f'R^2 train: {train_r2:.3f}, {test_r2:.3f}')
    return


@app.cell
def _(LinearRegression, mean_squared_error, np, plt, r2_score):


    from sklearn.preprocessing import PolynomialFeatures


    def compare_linear_quadratic_fits(X, y):
        """
        Fits, plots, and evaluates linear and quadratic regression models.

        Args:
            X (np.ndarray): The feature data.
            y (np.ndarray): The target data.

        Returns:
            dict: A dictionary containing the MSE and R-squared scores
                  for both models.
        """
        # 1. Initialize models and polynomial features
        lr = LinearRegression()
        pr = LinearRegression()
        quadratic = PolynomialFeatures(degree=2)
        X_quad = quadratic.fit_transform(X)

        # 2. Fit linear model and generate prediction line
        lr.fit(X, y)
        X_fit = np.arange(X.min() - 10, X.max() + 10, 10)[:, np.newaxis]
        y_lin_fit = lr.predict(X_fit)

        # 3. Fit quadratic model and generate prediction curve
        pr.fit(X_quad, y)
        y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

        # 4. Generate the plot
        plt.figure() # Create a new figure to avoid overplotting
        plt.scatter(X, y, label='Training points')
        plt.plot(X_fit, y_lin_fit, label='Linear fit', linestyle='--')
        plt.plot(X_fit, y_quad_fit, label='Quadratic fit')
        plt.xlabel('Explanatory variable')
        plt.ylabel('Predicted or known target values')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

        # 5. Calculate predictions on original training data
        y_lin_pred = lr.predict(X)
        y_quad_pred = pr.predict(X_quad)

        # 6. Calculate and return metrics
        results = {
            'mse_lin': mean_squared_error(y, y_lin_pred),
            'mse_quad': mean_squared_error(y, y_quad_pred),
            'r2_lin': r2_score(y, y_lin_pred),
            'r2_quad': r2_score(y, y_quad_pred)
        }
    
        return results

    # --- Example Usage ---
    # You would have this data defined in another cell in marimo

    X_data = np.array([258.0, 270.0, 294.0, 320.0, 342.0, 368.0, 
                       396.0, 446.0, 480.0, 586.0])[:, np.newaxis]

    y_data = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 
                       360.8, 368.0, 391.2, 390.8])

    # Call the function in its own cell to get the metrics
    metrics = compare_linear_quadratic_fits(X_data, y_data)
    print(metrics)
    return (PolynomialFeatures,)


@app.cell
def _(LinearRegression, PolynomialFeatures, df, np, plt, r2_score):
    def _():
        X = df[['Gr Liv Area']].values
        y = df['SalePrice'].values

        X = X[(df['Gr Liv Area'] < 4000)]
        y = y[(df['Gr Liv Area'] < 4000)]


        regr = LinearRegression()

        # create quadratic features
        quadratic = PolynomialFeatures(degree=2)
        cubic = PolynomialFeatures(degree=3)
        X_quad = quadratic.fit_transform(X)
        X_cubic = cubic.fit_transform(X)

        # fit features
        X_fit = np.arange(X.min()-1, X.max()+2, 1)[:, np.newaxis]

        regr = regr.fit(X, y)
        y_lin_fit = regr.predict(X_fit)
        linear_r2 = r2_score(y, regr.predict(X))

        regr = regr.fit(X_quad, y)
        y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
        quadratic_r2 = r2_score(y, regr.predict(X_quad))

        regr = regr.fit(X_cubic, y)
        y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
        cubic_r2 = r2_score(y, regr.predict(X_cubic))


        # plot results
        plt.scatter(X, y, label='Training points', color='lightgray')

        plt.plot(X_fit, y_lin_fit, 
                 label=f'Linear (d=1), $R^2$={linear_r2:.2f}',
                 color='blue', 
                 lw=2, 
                 linestyle=':')

        plt.plot(X_fit, y_quad_fit, 
                 label=f'Quadratic (d=2), $R^2$={quadratic_r2:.2f}',
                 color='red', 
                 lw=2,
                 linestyle='-')

        plt.plot(X_fit, y_cubic_fit, 
                 label=f'Cubic (d=3), $R^2$={cubic_r2:.2f}',
                 color='green', 
                 lw=2,
                 linestyle='--')


        plt.xlabel('Living area above ground in square feet')
        plt.ylabel('Sale price in U.S. dollars')
        plt.legend(loc='upper left')

        plt.tight_layout()
   
        return plt.show()


    _()
    return


@app.cell
def _(LinearRegression, PolynomialFeatures, df, np, plt, r2_score):
    def _():
        X = df[['Overall Qual']].values
        y = df['SalePrice'].values


        regr = LinearRegression()

        # create quadratic features
        quadratic = PolynomialFeatures(degree=2)
        cubic = PolynomialFeatures(degree=3)
        X_quad = quadratic.fit_transform(X)
        X_cubic = cubic.fit_transform(X)

        # fit features
        X_fit = np.arange(X.min()-1, X.max()+2, 1)[:, np.newaxis]

        regr = regr.fit(X, y)
        y_lin_fit = regr.predict(X_fit)
        linear_r2 = r2_score(y, regr.predict(X))

        regr = regr.fit(X_quad, y)
        y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
        quadratic_r2 = r2_score(y, regr.predict(X_quad))

        regr = regr.fit(X_cubic, y)
        y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
        cubic_r2 = r2_score(y, regr.predict(X_cubic))


        # plot results
        plt.scatter(X, y, label='Training points', color='lightgray')

        plt.plot(X_fit, y_lin_fit, 
                 label=f'Linear (d=1), $R^2$={linear_r2:.2f}',
                 color='blue', 
                 lw=2, 
                 linestyle=':')

        plt.plot(X_fit, y_quad_fit, 
                 label=f'Quadratic (d=2), $R^2$={quadratic_r2:.2f}',
                 color='red', 
                 lw=2,
                 linestyle='-')

        plt.plot(X_fit, y_cubic_fit, 
                 label=f'Cubic (d=3), $R^2$={cubic_r2:.2f}',
                 color='green', 
                 lw=2,
                 linestyle='--')


        plt.xlabel('Overall quality of the house')
        plt.ylabel('Sale price in U.S. dollars')
        plt.legend(loc='upper left')

        plt.tight_layout()
        return plt.show()


    _()
    return


@app.cell
def _(df, lin_regplot, plt, r2_score):
    def _():
        from sklearn.tree import DecisionTreeRegressor


        X = df[['Gr Liv Area']].values
        y = df['SalePrice'].values



        tree = DecisionTreeRegressor(max_depth=3)
        tree.fit(X, y)
        sort_idx = X.flatten().argsort()

        tree_r2 = r2_score(y, tree.predict(X))
        print(tree_r2)

        lin_regplot(X[sort_idx], y[sort_idx], tree)
        plt.xlabel('Living area above ground in square feet')
        plt.ylabel('Sale price in U.S. dollars')

        plt.tight_layout()
        #plt.savefig('figures/09_15.png', dpi=300)
        return plt.show()


    _()
    return


@app.cell
def _(df, mean_absolute_error, np, plt, r2_score, train_test_split):
    def _():
        target = 'SalePrice'
        features = df.columns[df.columns != target]

        X = df[features].values
        y = df[target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=123)

        from sklearn.ensemble import RandomForestRegressor


        forest = RandomForestRegressor(n_estimators=1000, 
                                       criterion='squared_error', 
                                       random_state=1, 
                                       n_jobs=-1)
        forest.fit(X_train, y_train)
        y_train_pred = forest.predict(X_train)
        y_test_pred = forest.predict(X_test)


        mae_train = mean_absolute_error(y_train, y_train_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        print(f'MAE train: {mae_train:.2f}')
        print(f'MAE test: {mae_test:.2f}')


        r2_train = r2_score(y_train, y_train_pred)
        r2_test =r2_score(y_test, y_test_pred)
        print(f'R^2 train: {r2_train:.2f}')
        print(f'R^2 test: {r2_test:.2f}')


        x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
        x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

        ax1.scatter(y_test_pred, y_test_pred - y_test,
                    c='limegreen', marker='s', edgecolor='white',
                    label='Test data')
        ax2.scatter(y_train_pred, y_train_pred - y_train,
                    c='steelblue', marker='o', edgecolor='white',
                    label='Training data')
        ax1.set_ylabel('Residuals')

        for ax in (ax1, ax2):
            ax.set_xlabel('Predicted values')
            ax.legend(loc='upper left')
            ax.hlines(y=0, xmin=x_min-100, xmax=x_max+100, color='black', lw=2)

        plt.tight_layout()
        return plt.show()


    _()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
