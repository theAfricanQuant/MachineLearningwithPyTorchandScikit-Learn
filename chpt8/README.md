
# Modernizing a Machine Learning Workflow for Sentiment Analysis

## Project Summary

This project follows a comprehensive chapter on sentiment analysis that builds a classifier to distinguish between positive and negative IMDb movie reviews. The workflow starts with raw data and moves through feature engineering to model training and evaluation, covering several key topics in Natural Language Processing (NLP).

The core stages of the project are:

1.  **Data Acquisition:** It begins by downloading and extracting the 50,000 IMDb movie review dataset.
2.  **Preprocessing:** The raw text files are cleaned to remove HTML tags and other noise, then compiled into a single, organized dataset.
3.  **Feature Engineering:** Text is converted into numerical vectors using fundamental NLP techniques, including the **bag-of-words** model, raw term frequency (**tf**), and the more advanced term frequency-inverse document frequency (**tf-idf**) to weigh word importance.
4.  **Model Training:** A **Logistic Regression** classifier is trained on the feature vectors. The process includes using `GridSearchCV` to systematically find the optimal model hyperparameters.
5.  **Out-of-Core Learning:** To handle datasets that are too large to fit into memory, the chapter demonstrates **out-of-core learning**. This involves using `SGDClassifier` to train the model by streaming data in small, manageable mini-batches with a `HashingVectorizer`.
6.  **Topic Modeling:** Finally, it introduces **Latent Dirichlet Allocation (LDA)**, an unsupervised learning technique, to discover abstract topics (like "horror," "action," or "comedy") from the text corpus without pre-existing labels.

-----

## The Modernization Effort: What This Work Was All About

The original code from the text, while functional, uses older Python patterns and file formats. Our collaborative effort focused on **modernizing and improving this workflow** to make it more efficient, robust, and readable for today's standards.

The key goals of this refactoring were:

  * **Performance:** Replacing inefficient data structures and file formats with high-performance alternatives.
  * **Robustness:** Adding error handling and using tools that are less prone to data integrity issues.
  * **Readability & Modernity:** Updating outdated libraries (`os`) with modern, object-oriented APIs (`pathlib`).

Specifically, we made the following key upgrades:

  * ✅ Switched from `os.path` to **`pathlib`** for clean, object-oriented filesystem operations.
  * ✅ Replaced the slow, inefficient DataFrame creation loop with a high-performance pattern.
  * ✅ Migrated the dataset from CSV to **Apache Parquet** for massive speed and storage benefits.
  * ✅ Refactored the out-of-core learning stream to work with the binary Parquet format.
  * ✅ Added robust error handling to the data download and extraction process.

-----

## Step-by-Step Code Refactoring

Here we break down the original code and show our improved, modernized versions.

### 1\. Data Acquisition and Extraction

The first step is to get the data. Our refactored function automates the download and extraction, uses `pathlib`, and includes error handling to prevent issues with corrupt files.

#### Modernized Code 🚀

```python
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

def download_and_extract_imdb(url: str, download_dir: str = "."):
    """
    Downloads and extracts the ACL IMDb dataset with a progress bar
    and robust error handling.
    """
    data_path = Path(download_dir)
    file_name = Path(url).name
    target_path = data_path / file_name
    extract_dir = data_path / "aclImdb"

    # If final directory already exists, we're done.
    if extract_dir.is_dir():
        print(f"'{extract_dir}' directory already exists. All good!")
        return

    # Download the file if it doesn't exist
    if not target_path.is_file():
        print(f"Downloading {url}...")
        # (Progress bar logic is included here)
        try:
            # reporthook defined as in previous examples
            urllib.request.urlretrieve(url, target_path, reporthook=...)
            print("\nDownload complete.")
        except Exception as e:
            print(f"\nDownload failed: {e}")
            return

    # Extract the archive with error handling
    try:
        print(f"Extracting '{target_path}'...")
        with tarfile.open(target_path, "r:gz") as tar:
            tar.extractall(path=data_path)
        print(f"Extraction complete. Data is in '{extract_dir}'.")
        
        print(f"Removing archive '{target_path}'...")
        target_path.unlink() # Clean up the .tar.gz file
        print("Cleanup complete.")
    except tarfile.ReadError:
        print(f"\n--- ERROR: Could not extract '{target_path}'. The file may be corrupted.")
        print("Please delete the file manually and run the script again.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during extraction: {e}")
```

#### Breakdown of Improvements

1.  **Using `pathlib`**: Instead of `os.path.join` and `os.path.isfile`, we use `pathlib.Path` objects. This makes path manipulations like `data_path / file_name` cleaner and more intuitive.
2.  **Robust Checks**: The function first checks if the final `aclImdb` directory already exists. If so, it skips all work. This makes the script safely re-runnable.
3.  **Error Handling**: The `try...except` block around the extraction (`tarfile.open`) is crucial. It catches errors if the download was corrupted, preventing the script from crashing silently.
4.  **Automatic Cleanup**: After a successful extraction, `target_path.unlink()` deletes the large `.tar.gz` archive, saving disk space.

-----

### 2\. Preprocessing Data into a DataFrame

The original text uses a `for` loop to append rows to a pandas DataFrame one by one. This is **extremely inefficient**. Our refactored code uses a standard, high-performance pattern.

#### Modernized Code ⚡

```python
import pandas as pd
import pyprind
from pathlib import Path

base_path = Path('aclImdb')
labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)

# 1. Collect all data into a Python list first. This is much faster.
data = []
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        dir_path = base_path / s / l
        for file_path in sorted(dir_path.iterdir()):
            txt = file_path.read_text(encoding='utf-8')
            data.append([txt, labels[l]])
            pbar.update()

# 2. Create the DataFrame once from the list. This is much faster.
df = pd.DataFrame(data, columns=['review', 'sentiment'])
```

#### Breakdown of Improvements

1.  **Efficient Data Collection**: Instead of appending to a DataFrame, we append each `[text, label]` pair to a simple Python `list`. This operation is memory-efficient and very fast.
2.  **Single DataFrame Creation**: After the loop finishes, we create the DataFrame **once** from the list of all data. This single operation is orders of magnitude faster than building the DataFrame iteratively.
3.  **Modern File Iteration**: We again use `pathlib`'s `iterdir()` method to loop through the files, which is a clean and modern way to handle directory traversal.

-----

### 3\. Saving and Loading Data: From CSV to Parquet

We upgraded the data storage from CSV to **Apache Parquet**, a modern columnar format designed for high-performance analytics.

#### Modernized Code 💾

```python
# Assume 'df_shuffled' is our final, processed DataFrame

# --- SAVING ---
# No 'encoding' needed. index=False prevents writing the redundant index column.
df_shuffled.to_parquet('movie_data.parquet', index=False)


# --- LOADING ---
# No 'encoding' needed. Column names and types are preserved automatically.
df_loaded = pd.read_parquet('movie_data.parquet')
```

#### Breakdown of Improvements

1.  **Parquet \> CSV**: Parquet is a binary, columnar format that provides:
      * **Smaller Files:** Superior compression results in less disk usage.
      * **Faster I/O:** Reading and writing Parquet files is significantly faster.
      * **Schema Preservation:** Parquet stores column names and data types (like `int`, `string`, etc.) within the file itself. This completely eliminates the need for the `df.rename(columns={"0": "review", ...})` hack mentioned in the original text.
2.  **Simpler Code**: The `to_parquet` and `read_parquet` calls are clean and simple. The `encoding` parameter is **not needed** because Parquet's specification mandates UTF-8 for string data internally.

**❗️ Important Dependency:** To use Parquet, the `pyarrow` library is required.

```bash
# Using pip
pip install pyarrow

# Or, if using uv
uv pip install pyarrow
```

-----

### 4\. Streaming for Out-of-Core Learning

The original text shows how to stream a CSV file line-by-line. This approach **does not work for Parquet**. Our refactored `stream_docs` function achieves the same memory-efficient goal by reading the Parquet file in large chunks (called "row groups").

#### Modernized Code 🌊

```python
import pyarrow.parquet as pq

def stream_docs(path):
    """
    Reads a Parquet file chunk by chunk (row group by row group)
    and yields documents.
    """
    parquet_file = pq.ParquetFile(path)
    # Iterate over the large chunks (row groups) within the Parquet file
    for i in range(parquet_file.num_row_groups):
        # Read one chunk into a small, in-memory pandas DataFrame
        table_chunk = parquet_file.read_row_group(i).to_pandas()
        
        # Now, iterate over the rows of this small chunk
        for row in table_chunk.itertuples():
            yield row.review, row.sentiment
```

#### Breakdown of Improvements

1.  **Correct Format Handling**: This function correctly handles the binary nature of Parquet by using the `pyarrow` library, the standard tool for this task.
2.  **Preserves Streaming Principle**: By iterating through `row_groups`, we still only load a fraction of the total dataset into memory at any given time, achieving the goal of out-of-core learning.
3.  **Robust Data Access**: Instead of brittle string slicing like `line[:-3]`, we access data by column name (`row.review`, `row.sentiment`). This is far more reliable and readable.

---

Of course. Here is a summary of the provided text in Markdown format.

---

# Summary of Sentiment Analysis Chapter

This chapter provides a comprehensive, hands-on guide to **Sentiment Analysis**, a subfield of Natural Language Processing (NLP). The primary goal is to build a machine learning model capable of classifying 50,000 IMDb movie reviews as either positive or negative.

The project follows a complete end-to-end machine learning workflow, starting from raw, unstructured text data and ending with trained models and topic analysis.

---

## Core Workflow

The chapter is structured around these key stages:

1.  **Data Acquisition and Preparation**:
    * The raw IMDb dataset, consisting of individual text files in a directory structure, is downloaded.
    * The text files are read, cleaned, and compiled into a single, organized CSV file for easier access.

2.  **Feature Engineering**:
    * The core challenge of converting text to numbers is addressed using the **Bag-of-Words** model.
    * This model is enhanced by calculating **Term Frequency-Inverse Document Frequency (TF-IDF)**, a technique that weighs words by their importance and down-weights overly common words.

3.  **Text Preprocessing**:
    * A custom `preprocessor` is built to clean text by removing HTML markup and punctuation using regular expressions.
    * Text is then **tokenized** (split into individual words), and techniques like **stemming** (reducing words to their root form with NLTK's `PorterStemmer`) and **stop-word removal** are applied to refine the tokens.

4.  **In-Memory Model Training**:
    * A `LogisticRegression` classifier is trained on the preprocessed data.
    * A `Pipeline` is used to chain the vectorization and classification steps together.
    * `GridSearchCV` is employed to systematically search for the best model hyperparameters, achieving a final test accuracy of approximately 90%.

5.  **Out-of-Core (Streaming) Learning**:
    * To address the challenge of datasets too large for memory, the chapter demonstrates **out-of-core learning**.
    * This involves streaming the data from the disk in small mini-batches and training an `SGDClassifier` incrementally using its `partial_fit` method.
    * A `HashingVectorizer` is used for this task because it's memory-efficient and doesn't need to see all the data at once.

6.  **Unsupervised Topic Modeling**:
    * Finally, the chapter explores an unsupervised task by applying **Latent Dirichlet Allocation (LDA)** to the dataset.
    * LDA is used to discover 10 abstract topics from the reviews without using their labels, identifying clusters of words related to genres like horror, action, and comedy.
