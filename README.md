# predictive-modeling-py
**Predictive modeling in Python.**

This code in this repository attempts to build a predictive model on a subset of data collected by Steinmetz et al. for their 2019 publication. Final project for STA 141 @ UC Davis.

# Directory

The contents of this repository are described below.

## Folders

- `scripts/` contains the files used to load in the `.rds` session files into Python and store them.
  - `scripts/open_rds.py` contains a script used to open the `.rds` files and store them as `ListVector` objects. See the module docstring for details.
  - `scripts/mouse.py` creates a class to contain the information on a given session. Data are transformed to have uniform data types across sessions (mainly `numpy.ndarray` and `pandas.DataFrame` objects). This class also transforms some of these raw data into features that will be used for the predictive models. These features are **not present** in the orginal `.rds` files and are used extensively in the report.
- `data/` contains the original `.rds` files.
- `test_data/` contains the test `.rds` files used to evaluate the model.
- `figures/` contains all of the plots used in the report. The report does not pull from this folder; the plots are created dynamically. This folder serves as a separate repository of images.

## Notebooks

- `data_structure.ipynb` contains scripts to test the pipelines between R and Python. It serves as validation effort to ensure the `open_rds.py` file is working well and that the data transformation applied by `mouse.py` are giving accurate results when compared to the raw data.
- `exploratory_data_analysis.ipynb` contains the basic data analysis conducted on the data for the purpose of this report.
- `model_training.ipynb` conducts the actual traning of models and the selection of a "best" model.
- `model_test.ipynb` contains the scripts that test the full model on the test sets in `test_data/` and an analysis on the effectiveness of that model.

## The Model

- `random_forest_model.pkl` is the final model selected from `model_training.ipynb`. Use `joblib.load('random_forest_model.pkl')` to load the model.

## The Report

- `report.ipynb` contains the written report.
- `report.html` contains the HTML compiled report, generated via `jupyter nbconvert report.ipynb --to html --no-input` to omit the code cells. This is the file submitted for the assignment.
