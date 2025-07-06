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
    ### **Making Big Data Smaller**

    Imagine you have a giant spreadsheet with hundreds of columns describing something, like all the stats for every Pokémon. It's too much information to look at\! The goal is to shrink this data down so it's easier to work with, without losing the most important parts. This is called **dimensionality reduction**.

    -----

    ### **How to Shrink Data**

    There are two main ways to do this:

      * **Feature Selection:** This is like deciding which columns in your spreadsheet are the most important and just deleting the rest. You stick with the original features.
      * **Feature Extraction:** This is a bit more clever. Instead of just deleting columns, you **create new summary columns** that combine information from the original ones. This is what **Principal Component Analysis (PCA)** does.

    -----

    ### **What is Principal Component Analysis (PCA)?**

    PCA is a popular technique for feature extraction. Think of your data as a big cloud of dots.

      * **Goal of PCA:** PCA finds the direction where the cloud of dots is most spread out. This direction captures the most important information (the most **variance**) about the data. This main direction is called the **Principal Component 1 (PC1)**.
      * **Next Steps:** Then, it finds the next most-spread-out direction that is at a right angle to the first one. This is **PC2**.

    By using these new "Principal Component" directions instead of your original data columns, you can describe your data with fewer features while keeping most of the interesting information. It’s like squishing a 3D object's shadow onto a 2D wall – you lose a dimension, but you can still tell what the object is.

    -----

    ### **The Steps of PCA**

    Here’s the basic recipe for how PCA works:

    1.  **Standardize the Data:** Make sure all your features are on the same scale (for example, everything is measured from 1 to 100). This prevents one feature from being treated as more important just because it has bigger numbers.
    2.  **Find Relationships:** PCA looks at how the original features relate to each other.
    3.  **Find the New Directions:** It uses some math (called *eigendecomposition*) to find the new "principal component" directions and figures out how much information each one holds.
    4.  **Pick the Best Ones:** It ranks the new directions and you decide to keep the top few (e.g., the top 2 or 3) that capture the most information.
    5.  **Transform the Data:** Use the top new directions you picked to create your new, smaller, and simpler dataset.


    ### Extracting the principal components step by step

    The firstfour steps of a PCA:

    * Standardizing the data
    * Constructing the covariance matrix
    * Obtaining the eigenvalues and eigenvectors of the covariance matrix
    * Sorting the eigenvalues by decreasing order to rank the eigenvectors
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import pandas as pd

    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                          'machine-learning-databases/wine/wine.data',
                          header=None)

    # if the Wine dataset is temporarily unavailable from the
    # UCI machine learning repository, un-comment the following line
    # of code to load the dataset from a local path:

    # df_wine = pd.read_csv('wine.data', header=None)

    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                       'Alcalinity of ash', 'Magnesium', 'Total phenols',
                       'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                       'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines', 'Proline']

    df_wine.head()
    return (df_wine,)


@app.cell
def _(df_wine):
    # Splitting the data into 70% training and 30% test subsets.
    from sklearn.model_selection import train_test_split

    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, 
                         stratify=y,
                         random_state=0)
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_test, X_train):
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_test_std, X_train_std


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    When preparing data for a machine learning model, a common mistake is to treat the test dataset as a completely separate world, rather than as new data being judged by the standards of the training data. This explanation clarifies why it's crucial to reuse the transformation parameters learned from the training set.

    ***

    ### The Golden Rule: Learn Once, Apply to All

    In machine learning, the **training data** is the "textbook" the model studies to learn the rules of the world. Any transformations, like standardizing data, are part of this learning process. The model learns the *mean* (average) and *standard deviation* (how spread out the data is) from this training data and only this data.

    The **test data** is like a final exam with questions the model has never seen before. To properly evaluate the model's knowledge, this exam must be graded using the same scale and rules learned from the textbook. Applying a new and separate standardization to the test data is like grading the final exam on a completely different curve—it makes the results meaningless.

    ***

    ### An Analogy: Grading on a Curve 🧑‍🏫

    A 9th grader would understand the concept of a teacher grading on a curve.

    Imagine a science class takes a big midterm exam. This is the **training set**. The teacher calculates the class average (the mean) and the spread of scores (the standard deviation). Let's say the average was a 75%. This "curve" is now the official standard for judging performance in this class.

    Now, a student who was sick, let's call him Alex, takes a makeup exam a week later. This is the **test set**.

    * **The WRONG Way ❌:** If the teacher took only Alex's score, calculated a "new average" and "new spread" based on just him, it would be absurd. Alex would be perfectly average *in a class of one*, and his score would tell you nothing about how he compares to his classmates. This is what happens when someone mistakenly uses `fit_transform` on the test data—they are creating a new, separate curve.

    * **The RIGHT Way ✅:** The teacher must take Alex's score and compare it to the *original* class curve from the midterm. If Alex got a 60%, the teacher uses the original average of 75% to see that he performed below average *compared to the rest of the class*. This is the correct approach. It uses the parameters from the training set (`transform`) to evaluate new data fairly.

    ***

    ### Applying it to the Example

    The text describes a model learning to classify objects based on their length.

    1.  **Learning the "Ruler" (Training):** The model is shown three objects with lengths **10 cm, 20 cm, and 30 cm**. It learns that the average length is **20 cm** and the standard deviation is about **8.2 cm**. This becomes its permanent "ruler" for judging length. Based on this ruler, it learns a rule: "if a standardized length is below 0.6, it's class 2."

    2.  **The Mistake (Incorrectly Testing):** Now, it sees new objects with lengths **5 cm, 6 cm, and 7 cm**. If a person makes the mistake of standardizing this new set by itself, the model creates a *new, temporary ruler*. The new average is 6 cm. On this new, flawed ruler, the 7 cm object now looks "long" and gets incorrectly classified as class 1. The model has lost all context.

    3.  **The Correct Method (Correctly Testing):** The model *must* use its original ruler where the average is 20 cm. When it measures the 5, 6, and 7 cm objects with this original ruler, it correctly sees that they are all *much shorter* than the average it learned about. All three get very low standardized scores and are correctly classified as class 2, which makes intuitive sense.

    By reusing the training parameters, the model maintains a consistent frame of reference, allowing it to judge new, unseen data accurately.
    """
    )
    return


@app.cell
def _(X_train_std):
    import numpy as np
    cov_mat = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

    print('\nEigenvalues \n', eigen_vals)
    return eigen_vals, eigen_vecs, np


@app.cell
def _(eigen_vals, np):
    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    return cum_var_exp, var_exp


@app.cell
def _():
    import matplotlib.pyplot as plt

    #--- 1. Set a global theme with a style sheet ---
    plt.style.use('seaborn')

    # --- 2. Make fine-grained global changes using plt.rcParams ---
    # Note: We are now using 'plt.rcParams' directly.
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['lines.linewidth'] = 2.5 # Make lines thicker by default
    return (plt,)


@app.cell
def _(cum_var_exp, plt, var_exp):
    plt.bar(range(1, 14), var_exp, align='center',
            label='Individual explained variance')
    plt.step(range(1, 14), cum_var_exp, where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('./chpt5/figures/05_02.png', dpi=300)
    plt.show()
    return


@app.cell
def _(eigen_vals, eigen_vecs, np):
    # Make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
                   for i in range(len(eigen_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs.sort(key=lambda k: k[0], reverse=True)
    return (eigen_pairs,)


@app.cell
def _(eigen_pairs, np):
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
                   eigen_pairs[1][1][:, np.newaxis]))
    print('Matrix W:\n', w)
    return (w,)


@app.cell
def _(X_train_std, w):
    X_train_std[0].dot(w)
    return


@app.cell
def _(X_train_std, np, plt, w, y_train):
    X_train_pca = X_train_std.dot(w)
    colors = ['r', 'b', 'g']
    markers = ['o', 's', '^']

    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_pca[y_train == l, 0], 
                    X_train_pca[y_train == l, 1], 
                    c=c, label=f'Class {l}', marker=m)

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('./chpt5/figures/05_03.png', dpi=300)
    plt.show()
    return X_train_pca, l


@app.cell
def _(X_train_std):
    from sklearn.decomposition import PCA

    pca = PCA()
    X_train_pca2 = pca.fit_transform(X_train_std)
    pca.explained_variance_ratio_
    return PCA, pca


@app.cell
def _(np, pca, plt):
    plt.bar(range(1, 14), pca.explained_variance_ratio_, align='center')
    plt.step(range(1, 14), np.cumsum(pca.explained_variance_ratio_), where='mid')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')

    plt.show()
    return


@app.cell
def _(PCA, X_test_std, X_train_std):
    pca3 = PCA(n_components=2)
    X_train_pca3 = pca3.fit_transform(X_train_std)
    X_test_pca3 = pca3.transform(X_test_std)
    return (X_train_pca3,)


@app.cell
def _(X_train_pca3, plt):
    plt.scatter(X_train_pca3[:, 0], X_train_pca3[:, 1])
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Principal component analysis in scikit-learn

    The `PCA` class in scikit-learn provides a straightforward way to perform principal component analysis without manually doing all the steps like creating a covariance matrix.

    It works like other scikit-learn "transformer" tools, using a simple two-step process.

    ***

    ### The Two-Step Process ⚙️

    1.  **Fit the Model**: First, you use the `.fit()` method on your **training data**. During this step, the `PCA` model learns the essential patterns from the data, which are the principal components (the directions of maximum variance). It learns these rules *only* from the training set.

    2.  **Transform the Data**: After the model has been fitted, you use the `.transform()` method to convert your data into the new, lower-dimensional feature space. It's crucial to use this same fitted model to transform *both* the **training data** and the **test data**. This ensures that both datasets are transformed using the exact same rules, maintaining consistency.
    """
    )
    return


@app.cell
def _(np, plt):
    from matplotlib.colors import ListedColormap

    def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

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
        plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # plot class examples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], 
                        y=X[y == cl, 1],
                        alpha=0.8, 
                        c=colors[idx],
                        marker=markers[idx], 
                        label=f'Class {cl}', 
                        edgecolor='black')
    return (plot_decision_regions,)


@app.cell
def _(PCA, X_test_std, X_train_pca, X_train_std, y_train):
    from sklearn.linear_model import LogisticRegression

    pca4 = PCA(n_components=2)
    X_train_pca4 = pca4.fit_transform(X_train_std)
    X_test_pca4 = pca4.transform(X_test_std)

    lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
    lr = lr.fit(X_train_pca, y_train)
    return LogisticRegression, X_test_pca4, X_train_pca4, lr


@app.cell
def _(X_train_pca4, lr, plot_decision_regions, plt, y_train):
    plot_decision_regions(X_train_pca4, y_train, classifier=lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('./chpt5/figures/05_04.png', dpi=300)
    plt.show()
    return


@app.cell
def _(X_test_pca4, lr, plot_decision_regions, plt, y_test):
    plot_decision_regions(X_test_pca4, y_test, classifier=lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('./chpt5/figures/05_05.png', dpi=300)
    plt.show()
    return


@app.cell
def _(PCA, X_train_std):
    pca5 = PCA(n_components=None)
    X_train_pca5 = pca5.fit_transform(X_train_std)
    pca5.explained_variance_ratio_
    return


@app.cell
def _(df_wine, eigen_vals, eigen_vecs, np, plt):
    loadings = eigen_vecs * np.sqrt(eigen_vals)

    fig, ax = plt.subplots()

    ax.bar(range(13), loadings[:, 0], align='center')
    ax.set_ylabel('Loadings for PC 1')
    ax.set_xticks(range(13))
    ax.set_xticklabels(df_wine.columns[1:], rotation=90)

    plt.ylim([-1, 1])
    plt.tight_layout()
    plt.savefig('chpt5/figures/05_05_02.png', dpi=300)
    plt.show()
    return (loadings,)


@app.cell
def _(loadings):
    loadings[:, 0]
    return


@app.cell
def _(df_wine, np, pca, plt):
    sklearn_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    fig1, ax1 = plt.subplots()

    ax1.bar(range(13), sklearn_loadings[:, 0], align='center')
    ax1.set_ylabel('Loadings for PC 1')
    ax1.set_xticks(range(13))
    ax1.set_xticklabels(df_wine.columns[1:], rotation=90)

    plt.ylim([-1, 1])
    plt.tight_layout()
    plt.savefig('chpt5/figures/05_05_03.png', dpi=300)
    plt.show()
    return


@app.cell
def _(X_train_std, np, y_train):
    np.set_printoptions(precision=4)

    mean_vecs = []
    for label in range(1, 4):
        mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
        print(f'MV {label}: {mean_vecs[label - 1]}\n')
    return (mean_vecs,)


@app.cell
def _(X_train_std, mean_vecs, np, y_train):
    dd = 13 # number of features
    S_W = np.zeros((dd, dd))
    for label1, mv1 in zip(range(1, 4), mean_vecs):
        class_scatter = np.zeros((dd, dd))  # scatter matrix for each class
        for row in X_train_std[y_train == label1]:
            row1, mv1 = row.reshape(dd, 1), mv1.reshape(dd, 1)  # make column vectors
            class_scatter += (row1 - mv1).dot((row1 - mv1).T)
        S_W += class_scatter                          # sum class scatter matrices

    print('Within-class scatter matrix: '
          f'{S_W.shape[0]}x{S_W.shape[1]}')
    return S_W, dd


@app.cell
def _(np, y_train):
    print('Class label distribution:',  
          np.bincount(y_train)[1:])
    return


@app.cell
def _(X_train_std, mean_vecs, np, y_train):
    ddd = 13  # number of features
    S_W2 = np.zeros((ddd, ddd))
    for label2, mv2 in zip(range(1, 4), mean_vecs):
        class_scatter2 = np.cov(X_train_std[y_train == label2].T)
        S_W2 += class_scatter2
    
    print('Scaled within-class scatter matrix: '
          f'{S_W2.shape[0]}x{S_W2.shape[1]}')
    return


@app.cell
def _(X_train_std, dd, mean_vecs, np, y_train):
    mean_overall3 = np.mean(X_train_std, axis=0)
    mean_overall4 = mean_overall3.reshape(dd, 1)  # make column vector

    d4 = 13  # number of features
    S_B4 = np.zeros((d4, d4))

    for i, mean_vec3 in enumerate(mean_vecs):
        n = X_train_std[y_train == i + 1, :].shape[0]
        mean_vec4 = mean_vec3.reshape(d4, 1)  # make column vector
        S_B4 += n * (mean_vec4 - mean_overall4).dot((mean_vec4 - mean_overall4).T)

    print('Between-class scatter matrix: '
          f'{S_B4.shape[0]}x{S_B4.shape[1]}')
    return (S_B4,)


@app.cell
def _(S_B4, S_W, np):
    eigen_vals5, eigen_vecs5 = np.linalg.eig(np.linalg.inv(S_W).dot(S_B4))
    return eigen_vals5, eigen_vecs5


@app.cell
def _(eigen_vals5, eigen_vecs5, np):
    # Make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs6 = [(np.abs(eigen_vals5[i]), eigen_vecs5[:, i])
                   for i in range(len(eigen_vals5))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs7 = sorted(eigen_pairs6, key=lambda k: k[0], reverse=True)

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues

    print('Eigenvalues in descending order:\n')
    for eigen_val6 in eigen_pairs7:
        print(eigen_val6[0])
    return (eigen_pairs7,)


@app.cell
def _(eigen_vals5, np, plt):
    tot1 = sum(eigen_vals5.real)
    discr1 = [(i / tot1) for i in sorted(eigen_vals5.real, reverse=True)]
    cum_discr1 = np.cumsum(discr1)

    plt.bar(range(1, 14), discr1, align='center',
            label='Individual discriminability')
    plt.step(range(1, 14), cum_discr1, where='mid',
             label='Cumulative discriminability')
    plt.ylabel('Discriminability ratio')
    plt.xlabel('Linear discriminants')
    plt.ylim([-0.1, 1.1])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('chpt5/figures/05_07.png', dpi=300)
    plt.show()
    return


@app.cell
def _(eigen_pairs7, np):
    w1 = np.hstack((eigen_pairs7[0][1][:, np.newaxis].real,
                  eigen_pairs7[1][1][:, np.newaxis].real))
    print('Matrix W:\n', w1)
    return


@app.cell
def _(X_train_std, l, np, plt, w, y_train):
    X_train_lda9 = X_train_std.dot(w)
    colors9 = ['r', 'b', 'g']
    markers9 = ['o', 's', '^']

    for l9, c9, m9 in zip(np.unique(y_train), colors9, markers9):
        plt.scatter(X_train_lda9[y_train == l9, 0],
                    X_train_lda9[y_train == l9, 1] * (-1),
                    c=c9, label=f'Class {l}', marker=m9)

    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('chpt5/figures/05_08.png', dpi=300)
    plt.show()
    return


@app.cell
def _(X_train_std, y_train):
    # LDA via scikit-learn
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    lda11 = LDA(n_components=2)
    X_train_lda11 = lda11.fit_transform(X_train_std, y_train)
    return X_train_lda11, lda11


@app.cell
def _(LogisticRegression, X_train_lda11, plot_decision_regions, plt, y_train):
    lr11 = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
    lr12 = lr11.fit(X_train_lda11, y_train)

    plot_decision_regions(X_train_lda11, y_train, classifier=lr12)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('chpt5/figures/05_09.png', dpi=300)
    plt.show()
    return (lr12,)


@app.cell
def _(X_test_std, lda11, lr12, plot_decision_regions, plt, y_test):
    X_test_lda13 = lda11.transform(X_test_std)

    plot_decision_regions(X_test_lda13, y_test, classifier=lr12)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('chpt5/figures/05_10.png', dpi=300)
    plt.show()
    return


@app.cell
def _(plt):
    from sklearn.datasets import load_digits

    digits = load_digits()

    fig22, ax22 = plt.subplots(1, 4)

    for i2 in range(4):
        ax22[i2].imshow(digits.images[i2], cmap='Greys')
    
    plt.savefig('chpt5/figures/05_12.png', dpi=300)
    plt.show() 
    return (digits,)


@app.cell
def _(digits):
    digits.data.shape
    return


@app.cell
def _(digits):
    y_digits = digits.target
    X_digits = digits.data
    return X_digits, y_digits


@app.cell
def _(X_digits):
    from sklearn.manifold import TSNE


    tsne = TSNE(n_components=2,
                init='pca', learning_rate='auto',
                random_state=123)
    X_digits_tsne = tsne.fit_transform(X_digits)
    return (X_digits_tsne,)


@app.cell
def _(X_digits_tsne, np, plt, y_digits):
    import matplotlib.patheffects as PathEffects


    def plot_projection(x, colors):
    
        f = plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')
        for i in range(10):
            plt.scatter(x[colors == i, 0],
                        x[colors == i, 1])

        for i in range(10):

            xtext, ytext = np.median(x[colors == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
        
    plot_projection(X_digits_tsne, y_digits)
    plt.savefig('chpt5/figures/05_13.png', dpi=300)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
