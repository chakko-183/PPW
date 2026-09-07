# AI Coding Agent Instructions for Proyek Sains Data

## Project Overview

This repository, **Proyek Sains Data**, is a data science project structured as a Jupyter Book. It includes notebooks for data exploration, preprocessing, and modeling, along with supporting datasets and configurations.

### Key Components

- **Jupyter Book Configuration**:
  - `_config.yml`: Defines metadata (e.g., title, author) and execution settings.
  - `_toc.yml`: Specifies the table of contents and notebook structure.
- **Notebooks**:
  - `car-test.ipynb`, `dataExploration.ipynb`, `timeSeries.ipynb`, etc.: Contain analysis and modeling workflows.
- **Data**:
  - Located in `data/` and its subdirectories, including CSV, ARFF, and NetCDF files.
- **Dependencies**:
  - Managed via `requirements.txt`.

## Developer Workflows

### Setting Up the Environment

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure Jupyter Book is installed for building the documentation.

### Running Notebooks

- Notebooks are configured to execute on each build (`execute_notebooks: force` in `_config.yml`).
- Use Jupyter Notebook or JupyterLab to interactively run and debug notebooks.

### Building the Jupyter Book

1. Build the book locally:
   ```bash
   jupyter-book build proyek-sains-data
   ```
2. Deploy to GitHub Pages:
   ```bash
   ghp-import -n -p -f proyek-sains-data/_build/html
   ```

### Testing and Debugging

- Ensure all notebooks execute without errors before building the book.
- Use `nbconvert` to convert notebooks to other formats for testing:
  ```bash
  jupyter nbconvert --execute --to html proyek-sains-data/*.ipynb
  ```

## Project-Specific Conventions

- **Notebook Structure**:
  - Follow the order defined in `_toc.yml`.
  - Use markdown cells for explanations and code cells for implementation.
- **Data Handling**:
  - Store raw data in `data/`.
  - Use relative paths to access datasets within notebooks.
- **Version Control**:
  - Commit changes to notebooks and data files.
  - Use meaningful commit messages (e.g., `Add preprocessing for voice detection`).

## External Dependencies

- **Libraries**:
  - Core: `numpy`, `scikit-learn`, `librosa`
  - Audio processing: `soundfile`, `sounddevice`, `pydub`
  - Jupyter Book: `jupyter-book`, `ghp-import`
- **Data Formats**:
  - CSV, ARFF, NetCDF

## Examples

### Accessing a Dataset

```python
import pandas as pd

data = pd.read_csv('data/NO2_Kwanyar.csv')
print(data.head())
```

### Training a Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

---

For further details, refer to the Jupyter Book documentation: [https://jupyterbook.org](https://jupyterbook.org).
