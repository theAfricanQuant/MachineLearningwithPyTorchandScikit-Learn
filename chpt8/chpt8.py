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
        'pandas': '1.3.2',
        'sklearn': '1.0',
        'pyprind': '2.11.3',
        'nltk': '3.6',
    }
    check_packages(d)
    return (sys,)


@app.cell
def _(sys):
    # import sys
    import tarfile
    import time
    import urllib.request
    from pathlib import Path

    def download_and_extract_imdb(url: str, download_dir: str = "."):
        """
        Downloads and extracts the ACL IMDb dataset with error handling.
        """
        data_path = Path(download_dir)
        file_name = Path(url).name
        target_path = data_path / file_name
        extract_dir = data_path / "aclImdb"

        if extract_dir.is_dir():
            print(f"'{extract_dir}' directory already exists. All good!")
            return

        if not target_path.is_file():
            print(f"Downloading {url}...")
            # (Progress bar logic remains the same)
            start_time = 0
            def reporthook(count, block_size, total_size):
                nonlocal start_time
                if count == 0:
                    start_time = time.time()
                    return
                duration = time.time() - start_time
                progress_size = int(count * block_size)
                speed = progress_size / (1024.**2 * (duration + 1e-9))
                percent = progress_size * 100. / total_size
                sys.stdout.write(
                    f"\r{percent:.1f}% | {progress_size / (1024.**2):.2f} MB | {speed:.2f} MB/s"
                )
                sys.stdout.flush()

            try:
                urllib.request.urlretrieve(url, target_path, reporthook)
                print("\nDownload complete.")
            except Exception as e:
                print(f"\nDownload failed: {e}")
                if target_path.exists():
                    target_path.unlink() # Clean up partial download
                return # Stop execution

        # --- NEW: Error handling for extraction ---
        try:
            print(f"Extracting '{target_path}'...")
            with tarfile.open(target_path, "r:gz") as tar:
                tar.extractall(path=data_path)
            print(f"Extraction complete. Data is in '{extract_dir}'.")

            print(f"Removing archive '{target_path}'...")
            target_path.unlink()
            print("Cleanup complete.")

        except tarfile.ReadError:
            print("\n--- ERROR ---")
            print(f"Could not extract '{target_path}'. The file may be corrupted.")
            print("Please delete the file manually and run the script again.")
        except Exception as e:
            print(f"\nAn unexpected error occurred during extraction: {e}")

    # --- Main execution block ---
    if __name__ == "__main__":
        SOURCE_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        download_and_extract_imdb(url=SOURCE_URL)
    return (Path,)


@app.cell
def _(sys):
    import pyprind
    import pandas as pd
    import os
    # import sys
    from packaging import version


    # change the `basepath` to the directory of the
    # unzipped movie dataset

    basepath = 'aclImdb'

    labels = {'pos': 1, 'neg': 0}

    # if the progress bar does not show, change stream=sys.stdout to stream=2
    pbar = pyprind.ProgBar(50000, stream=sys.stdout)

    df = pd.DataFrame()
    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = os.path.join(basepath, s, l)
            for file in sorted(os.listdir(path)):
                with open(os.path.join(path, file), 
                          'r', encoding='utf-8') as infile:
                    txt = infile.read()

                if version.parse(pd.__version__) >= version.parse("1.3.2"):
                    x = pd.DataFrame([[txt, labels[l]]], columns=['review', 'sentiment'])
                    df = pd.concat([df, x], ignore_index=False)

                else:
                    df = df.append([[txt, labels[l]]], 
                                   ignore_index=True)
                pbar.update()
    df.columns = ['review', 'sentiment']
    return df, pd, pyprind


@app.cell
def _(df, pd):
    # Assume 'df' is your existing DataFrame with 'review' and 'sentiment' columns

    # 1. Shuffle the DataFrame using the modern and recommended method.
    #    This replaces the older, version-dependent logic.
    df_shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)

    # 2. Save the shuffled DataFrame to a Parquet file.
    #    Note: 'encoding' is not needed for the binary Parquet format.
    df_shuffled.to_parquet('movie_data.parquet', index=False)

    # 3. Read the data back from the Parquet file.
    df_loaded = pd.read_parquet('movie_data.parquet')

    # Display the first few rows to verify it worked.
    df_loaded.head(3)
    return (df_loaded,)


@app.cell
def _(df_loaded):
    df_loaded.shape
    return


@app.cell
def _():
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer

    count = CountVectorizer()
    docs = np.array([
            'The sun is shining',
            'The weather is sweet',
            'The sun is shining, the weather is sweet, and one and one is two'])
    bag = count.fit_transform(docs)

    print(count.vocabulary_)

    print(bag.toarray())
    return CountVectorizer, count, docs, np


@app.cell
def _(np):
    np.set_printoptions(precision=2)
    return


@app.cell
def _(count, docs):
    from sklearn.feature_extraction.text import TfidfTransformer

    tfidf1 = TfidfTransformer(use_idf=True, 
                             norm='l2', 
                             smooth_idf=True)
    print(tfidf1.fit_transform(count.fit_transform(docs))
          .toarray())
    return (TfidfTransformer,)


@app.cell
def _(np):
    tf_is = 3
    n_docs = 3
    idf_is = np.log((n_docs+1) / (3+1))
    tfidf_is = tf_is * (idf_is + 1)
    print(f'tf-idf of term "is" = {tfidf_is:.2f}')
    return


@app.cell
def _(TfidfTransformer, count, docs):
    tfidf2 = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
    raw_tfidf = tfidf2.fit_transform(count.fit_transform(docs)).toarray()[-1]
    raw_tfidf 
    return (raw_tfidf,)


@app.cell
def _(np, raw_tfidf):
    l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf**2))
    l2_tfidf
    return


@app.cell
def _(df_loaded):
    df_loaded.loc[0, 'review'][-50:]
    return


@app.cell
def _(df_loaded):
    import re
    def preprocessor(text):
        text = re.sub('<[^>]*>', '', text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                               text)
        text = (re.sub('[\W]+', ' ', text.lower()) +
                ' '.join(emoticons).replace('-', ''))
        return text

    preprocessor(df_loaded.loc[0, 'review'][-50:])
    return preprocessor, re


@app.cell
def _(preprocessor):
    preprocessor("</a>This :) is :( a test :-)!")
    return


@app.cell
def _(df, preprocessor):
    df['review'] = df['review'].apply(preprocessor)
    return


@app.cell
def _():
    from nltk.stem.porter import PorterStemmer

    porter = PorterStemmer()

    def tokenizer1(text):
        return text.split()


    def tokenizer_porter(text):
        return [porter.stem(word) for word in text.split()]
    return tokenizer1, tokenizer_porter


@app.cell
def _(tokenizer1):
    tokenizer1('runners like running and thus they run')
    return


@app.cell
def _(tokenizer_porter):
    tokenizer_porter('runners like running and thus they run')
    return


@app.cell
def _():
    import nltk

    nltk.download('stopwords')
    return


@app.cell
def _(tokenizer_porter):
    from nltk.corpus import stopwords

    stop = stopwords.words('english')
    [w for w in tokenizer_porter('a runner likes running and runs a lot')
     if w not in stop]
    return (stop,)


@app.cell
def _(df_loaded):
    X_train = df_loaded.loc[:25000, 'review'].values
    y_train = df_loaded.loc[:25000, 'sentiment'].values
    X_test = df_loaded.loc[25000:, 'review'].values
    y_test = df_loaded.loc[25000:, 'sentiment'].values
    return X_test, X_train, y_test, y_train


@app.cell
def _(stop, tokenizer, tokenizer_porter):
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import GridSearchCV

    tfidf = TfidfVectorizer(strip_accents=None,
                            lowercase=False,
                            preprocessor=None)

    """
    param_grid = [{'vect__ngram_range': [(1, 1)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer, tokenizer_porter],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]},
                  {'vect__ngram_range': [(1, 1)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer, tokenizer_porter],
                   'vect__use_idf':[False],
                   'vect__norm':[None],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]},
                  ]
    """

    small_param_grid = [{'vect__ngram_range': [(1, 1)],
                         'vect__stop_words': [None],
                         'vect__tokenizer': [tokenizer, tokenizer_porter],
                         'clf__penalty': ['l2'],
                         'clf__C': [1.0, 10.0]},
                        {'vect__ngram_range': [(1, 1)],
                         'vect__stop_words': [stop, None],
                         'vect__tokenizer': [tokenizer],
                         'vect__use_idf':[False],
                         'vect__norm':[None],
                         'clf__penalty': ['l2'],
                      'clf__C': [1.0, 10.0]},
                  ]

    lr_tfidf = Pipeline([('vect', tfidf),
                         ('clf', LogisticRegression(solver='liblinear'))])

    gs_lr_tfidf = GridSearchCV(lr_tfidf, small_param_grid,
                               scoring='accuracy',
                               cv=5,
                               verbose=1,
                               n_jobs=-1)
    return GridSearchCV, LogisticRegression, gs_lr_tfidf


@app.cell
def _(X_train, gs_lr_tfidf, y_train):
    gs_lr_tfidf.fit(X_train, y_train)
    return


app._unparsable_cell(
    r"""
    print(f'Best parameter set: {gs_lr_tfidf.best_params_}')
    print(f'CV Accuracy: {gs_lr_tfidf.best_score_:.3f}')|
    """,
    name="_"
)


@app.cell
def _(X_test, gs_lr_tfidf, y_test):
    clf = gs_lr_tfidf.best_estimator_
    print(f'Test Accuracy: {clf.score(X_test, y_test):.3f}')
    return (clf,)


@app.cell
def _(LogisticRegression, np):
    # from sklearn.linear_model import LogisticRegression
    # import numpy as np

    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score

    np.random.seed(0)
    np.set_printoptions(precision=6)
    y = [np.random.randint(3) for i in range(25)]
    X = (y + np.random.randn(25)).reshape(-1, 1)

    cv5_idx = list(StratifiedKFold(n_splits=5, shuffle=False).split(X, y))
    
    lr = LogisticRegression()
    cross_val_score(lr, X, y, cv=cv5_idx)
    return X, cross_val_score, cv5_idx, lr, y


@app.cell
def _(GridSearchCV, X, cv5_idx, lr, y):
    # from sklearn.model_selection import GridSearchCV

    # lr = LogisticRegression()
    gs = GridSearchCV(lr, {}, cv=cv5_idx, verbose=3).fit(X, y) 
    return (gs,)


@app.cell
def _(gs):
    gs.best_score_
    return


@app.cell
def _(X, cross_val_score, cv5_idx, lr, y):
    cross_val_score(lr, X, y, cv=cv5_idx).mean()
    return


@app.cell
def _(Path):
    # from pathlib import Path

    # Define the path to your Parquet file
    parquet_file = Path('movie_data.parquet')

    # Check if the Parquet file exists in the current directory
    if not parquet_file.is_file():
        print('The file "movie_data.parquet" was not found.')
        print('Please make sure it is in this directory. You can create it by:')
        print('a) executing the previous code cells in this notebook, or')
        print('b) downloading it from the source repository if available.')

    # Now you can proceed to load the data, for example:
    # df = pd.read_parquet(parquet_file)
    return


@app.cell
def _(re, stop):
    # import re
    import pyarrow.parquet as pq
    # from nltk.corpus import stopwords

    # This assumes 'pyarrow' and 'pandas' are installed.
    # If not: uv pip install pyarrow pandas

    # The tokenizer function does not need to change.
    # stop = stopwords.words('english')

    def tokenizer(text):
        text = str(text) # Handles non-string inputs like NaN
        text = re.sub('<[^>]*>', '', text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        text = re.sub('[\W]+', ' ', text.lower()) + \
            ' '.join(emoticons).replace('-', '')
        tokenized = [w for w in text.split() if w not in stop]
        return tokenized

    # This function is rewritten for the Parquet format.
    def stream_docs(path):
        """
        Reads a Parquet file chunk by chunk and yields documents.
        Assumes the Parquet file has 'review' and 'sentiment' columns.
        """
        parquet_file = pq.ParquetFile(path)
        # Iterate over row groups (chunks) in the Parquet file
        for i in range(parquet_file.num_row_groups):
            # Read one chunk into a pandas DataFrame
            table_chunk = parquet_file.read_row_group(i).to_pandas()
            # Iterate over the rows of the current chunk
            for row in table_chunk.itertuples():
                text = row.review
                label = row.sentiment
                yield text, label
    return stream_docs, tokenizer


@app.cell
def _(stream_docs):
    next(stream_docs(path='movie_data.parquet'))
    return


@app.function
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


@app.cell
def _(tokenizer):
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import SGDClassifier


    vect = HashingVectorizer(decode_error='ignore', 
                             n_features=2**21,
                             preprocessor=None, 
                             tokenizer=tokenizer)
    return SGDClassifier, vect


@app.cell
def _(SGDClassifier, stream_docs):
    from distutils.version import LooseVersion as Version
    from sklearn import __version__ as sklearn_version

    clf1 = SGDClassifier(loss='log', random_state=1)


    doc_stream = stream_docs(path='movie_data.parquet')
    return clf1, doc_stream


@app.cell
def _(clf1, doc_stream, np, pyprind, vect):
    # from distutils.version import LooseVersion as Version
    # from sklearn import __version__ as sklearn_version

    # clf = SGDClassifier(loss='log', random_state=1)


    # doc_stream = stream_docs(path='movie_data.csv')
    # import pyprind
    pbar1 = pyprind.ProgBar(45)

    classes = np.array([0, 1])
    for _ in range(45):
        X_train1, y_train1 = get_minibatch(doc_stream, size=1000)
        if not X_train1:
            break
        X_train1 = vect.transform(X_train1)
        clf1.partial_fit(X_train1, y_train1, classes=classes)
        pbar1.update()
    return


@app.cell
def _(clf, doc_stream, vect):

    X_test1, y_test1 = get_minibatch(doc_stream, size=5000)
    X_test2 = vect.transform(X_test1)
    print(f'Accuracy: {clf.score(X_test2, y_test1):.3f}')
    return X_test2, y_test1


@app.cell
def _(X_test2, clf1, y_test1):
    clf2 = clf1.partial_fit(X_test2, y_test1)
    return


@app.cell
def _(pd):
    df2 = pd.read_parquet("movie_data.parquet")

    # the following is necessary on some computers:
    df3 = df2.rename(columns={"0": "review", "1": "sentiment"})

    df3.head(3)

    return (df3,)


@app.cell
def _(CountVectorizer, df):
    # from sklearn.feature_extraction.text import CountVectorizer

    count1 = CountVectorizer(stop_words='english',
                            max_df=.1,
                            max_features=5000)
    X1 = count1.fit_transform(df['review'].values)
    return X1, count1


@app.cell
def _(X1):
    from sklearn.decomposition import LatentDirichletAllocation

    lda = LatentDirichletAllocation(n_components=10,
                                    random_state=123,
                                    learning_method='batch')
    X_topics = lda.fit_transform(X1)
    return X_topics, lda


@app.cell
def _(lda):
    lda.components_.shape
    return


@app.cell
def _(count1, lda):
    n_top_words = 5
    feature_names = count1.get_feature_names_out()

    for topic_idx, topic in enumerate(lda.components_):
        print(f'Topic {(topic_idx + 1)}:')
        print(' '.join([feature_names[i]
                        for i in topic.argsort()\
                            [:-n_top_words - 1:-1]]))
    return


@app.cell
def _(X_topics, df3):
    horror = X_topics[:, 5].argsort()[::-1]

    for iter_idx, movie_idx in enumerate(horror[:3]):
        print(f'\nHorror movie #{(iter_idx + 1)}:')
        print(df3['review'][movie_idx][:300], '...')
    return


if __name__ == "__main__":
    app.run()
