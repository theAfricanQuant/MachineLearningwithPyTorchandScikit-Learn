# Project Setup with uv

This document outlines the steps taken to initialize this project's Python environment and install its dependencies using the `uv` package manager.

## Introduction

[uv](https://github.com/astral-sh/uv) is an extremely fast Python package installer and resolver, written in Rust. It's used here to create a virtual environment and manage project dependencies, ensuring a consistent and reproducible setup.

## Installation Steps

The entire setup process was completed with three main commands.

### 1. Project Initialization

First, the project and its virtual environment were initialized. The `-p 3.9` flag specifies that the environment should use a Python 3.9 interpreter.

```bash
uv init -p 3.9
```
This command:
* Creates a virtual environment in the `.venv` directory.
* Generates a `pyproject.toml` file to manage project metadata and dependencies.

### 2. Installing `ipykernel`

To enable the use of Jupyter notebooks within this environment, `ipykernel` and its related dependencies were added.

```bash
uv add ipykernel
```
This command installs `ipykernel` and other essential packages for an interactive development experience.

### 3. Installing Core Machine Learning Libraries

Finally, the specific versions of the core data science and machine learning libraries required for this project were added. Pinning the versions ensures that the project will work consistently for anyone who sets it up.

```bash
uv add numpy==1.21.2 scipy==1.7.0 scikit-learn==1.0 matplotlib==3.4.3 pandas==1.3.2
```

This command installs the following key packages:
* `numpy==1.21.2`
* `scipy==1.7.0`
* `scikit-learn==1.0`
* `matplotlib==3.4.3`
* `pandas==1.3.2`
* ...along with their respective dependencies like `joblib`, `kiwisolver`, `pytz`, etc.

## How to Use This Environment

To activate the virtual environment and start working, run the following command from the project root:

**On Windows:**
```powershell
.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

Once activated, you can run Python scripts or start a Jupyter notebook, and they will use the packages installed in this environment.