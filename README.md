# CollaborativeFiltering
Comparison of 7 collaborative filtering models and nuclear norm optimization

## Quick Start:
run  `example.py`  with all the following files in the same directory:
- `example.py`
- `generateData.py`
- `gradientDescent.py`
- `LeastSquares.py`
- `misc.py`
- `regularizedLeastSquares.py`
- `semidef_prog.py`


## Notes:

Run either `example.py` or  `Example_to_excel.py`.

`Example_to_excel.py` works like `example.py` except that it will automatically write the data into an excel file. Make sure to indicate how many iterations and the dimensions of the data (`n` & `d`) you want to perform before running the file. (more iterations or larger matrix size will take longer to run)

Other notes:
`generateData.py` is the file that creates the data for all the methods to use.

`gradientDescent.py`  is the file that contains the gradient descent algorithm

`LeastSquares.py`  contains the code for all least squares objective functions that don’t involve regularization

`regularizedLeastSquares.py` contains the code for all least squares objective functions that do involve regularization

`misc.py` contains a helper function used for calculating the error only for the given set of samples. (ie it sets all other values to be the “predicted” value so that it has a zero error and thus doesn’t contribute to the objective function).
