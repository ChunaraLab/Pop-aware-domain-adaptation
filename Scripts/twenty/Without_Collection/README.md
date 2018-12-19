## FRED : Frustratingly easy domain adaptation
* `/Only_Symptoms` - Models that only use the symptoms while ignoring the demographic information
* `/With_demographics` - Models that use the symptoms along with the demographic information

## Hierarchical model
* `/Hyperparameters` - Hyperparameters for the model; only make changes to these files for altering the hierarchical model
* `data_preprocessing.py` - Data Preprocessing; sampling code to be included
* `split_data.py` - For splitting the data, never really used
* `objective_function.py` - Has the objective function to be optimized; only change this file to change the objective function.
* `util.py` - Some functions that aid the optimization
* `main.py` - Driver script
* `testing.ipynb` - Code for testing the model



## `Logistic Regression All data.ipynb`
* Simple logistic regression trained and tested on the same dataset. The model performance achieved via any method should be as close to this as possible.


## `Combine demographics with symptoms.ipynb`
* Combines the demographic information with the symptoms to create new features

## `Distance metric.ipynb`
* Distance metrics for finding the closest Distance
* Currently only includes the l2 distance between the feature variance

## Datasets

* NYUMC Data
* GoViral Data
* Fluwatch Data
* HongKong Data
* Hutterite Data


### Resources
