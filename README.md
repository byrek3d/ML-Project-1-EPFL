
# First project of the Machine Learning Master's course at EPFL. 
The goal of the project is to explore and compare different supervised learning algorithms and how they deal with a data-set from  CERN in the field of physics, in order to best predict the presence of the Higgs Boson.
## How to run the solution 

Run the command: "python run.py" 

The train and test csv files should be located in a folder named "data" one directory above the location of the run.py file. The predictions will be saved in a file named 'result_LS.csv'
## Code Organization

We organized our codebase using several files:

 - `implementations.py` contains the mandatory functions together with some helper functions used in the implementation of the 6 mandatory ones.

 - `run.py` is the execution script.

### Notes on some of the helper functions:
In the beginning of the run.py file we have placed some useful functions used throughout the code.
```python
def  number_to_nan(tX)
def  nan_to_median(tX)
def  number_to_other_number(tX, new_value)
```
These functions are used in the data cleaning process, replacing the value -999 with other more useful information
```python
def  accuracy(y_true, y_pred)
```
We found it more useful to calculate the accuracy of the predictions, instead of the Mean Squared Errors in some case. We used this function to do this
```python
def  build_poly(x, degree)
```
We use this function for expanding our features to a higher dimension. For a detailed explanation on how it works and why we decided on doing the expansion this way, see the report





