# Python Environment Checker

This script is a command-line tool designed to verify that a Python environment meets the specific dependency requirements for a project. It checks two key things:

1.  The version of the Python interpreter.
2.  The versions of several critical scientific computing and data science packages.

This is useful for ensuring that code will run as expected and for guiding users in setting up their environment correctly.

---

## How It Works

The script executes its checks in a clear, sequential order.

### 1. Python Version Validation

First, it checks the system's current Python version against a recommended minimum.

-   **Requirement:** Python `3.8` or newer.
-   **Method:** It uses `sys.version` to get the current version and `distutils.version.LooseVersion` to perform a reliable comparison.
-   **Output:** It prints an `[OK]` or `[FAIL]` message indicating if the Python version is suitable.

### 2. Package Dependency Verification

Next, it verifies a predefined list of required Python packages and their minimum versions.

#### Required Packages

The script checks for the following packages and versions:

| Package | Required Version |
| :--- | :--- |
| `numpy` | `>= 1.21.2` |
| `scipy` | `>= 1.7.0` |
| `matplotlib`| `>= 3.4.3` |
| `sklearn` | `>= 1.0` |
| `pandas` | `>= 1.3.2` |

#### Verification Process

For each package in the list, the script:
1.  **Attempts to import it**. If the import fails, it reports that the package is `[FAIL]` not installed.
2.  **Finds the version number**. It intelligently searches for the version in common attributes like `.__version__`, `.version`, or `.version_info`.
3.  **Compares versions**. It compares the installed version against the required version.
4.  **Prints the status**. It prints `[OK]` if the version is sufficient and `[FAIL]` if it is outdated, prompting the user to upgrade.

---

## Full Script

Below is the complete source code for the environment checker.

```python
import sys
from distutils.version import LooseVersion

# 1. Check Python version
if LooseVersion(sys.version) < LooseVersion('3.8'):
    print('[FAIL] We recommend Python 3.8 or newer but'
          ' found version %s' % (sys.version))
else:
    print('[OK] Your Python version is %s' % (sys.version))


def get_packages(pkgs):
    """
    Attempts to import packages and retrieve their version numbers.
    """
    versions = []
    for p in pkgs:
        try:
            imported = __import__(p)
            try:
                versions.append(imported.__version__)
            except AttributeError:
                try:
                    versions.append(imported.version)
                except AttributeError:
                    try:
                        versions.append(imported.version_info)
                    except AttributeError:
                        versions.append('0.0')
        except ImportError:
            print(f'[FAIL]: {p} is not installed and/or cannot be imported.')
            versions.append('N/A')
    return versions


def check_packages(d):
    """
    Compares installed package versions against a dictionary of required versions.
    """
    versions = get_packages(d.keys())

    for (pkg_name, suggested_ver), actual_ver in zip(d.items(), versions):
        if actual_ver == 'N/A':
            continue
        
        actual_ver, suggested_ver = LooseVersion(actual_ver), LooseVersion(suggested_ver)
        
        # Special case check for matplotlib
        if pkg_name == "matplotlib" and actual_ver == LooseVersion("3.8"):
            print(f'[FAIL] {pkg_name} {actual_ver}, please upgrade to {suggested_ver} >= matplotlib > 3.8')
        elif actual_ver < suggested_ver:
            print(f'[FAIL] {pkg_name} {actual_ver}, please upgrade to >= {suggested_ver}')
        else:
            print(f'[OK] {pkg_name} {actual_ver}')


if __name__ == '__main__':
    # Dictionary of required packages and their minimum versions
    required_packages = {
        'numpy': '1.21.2',
        'scipy': '1.7.0',
        'matplotlib': '3.4.3',
        'sklearn': '1.0',
        'pandas': '1.3.2'
    }
    
    # Run the check
    check_packages(required_packages)
```

---

## Usage

To use the script:

1.  Save the code as a Python file (e.g., `python_environment_check.py`).
2.  Open a terminal or command prompt.
3.  Navigate to the directory where you saved the file.
4.  Run the script using the following command:

```bash
python python_environment_check.py
```

The script will then print the status of your environment directly to the console.