import sys
import numpy as np
import pandas as pd
import scipy
import matplotlib
import sklearn
from sklearn.linear_model import LinearRegression

x = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([3.0, 5.0, 7.0, 9.0])
model = LinearRegression().fit(x, y)
assert np.isclose(model.predict([[5.0]])[0], 11.0)
print('Python:', sys.version.split()[0])
print('NumPy:', np.__version__)
print('Pandas:', pd.__version__)
print('SciPy:', scipy.__version__)
print('Matplotlib:', matplotlib.__version__)
print('Scikit-learn:', sklearn.__version__)
print('Linear regression test: PASSED')
