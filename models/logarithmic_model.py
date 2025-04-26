import numpy as np
from scipy.optimize import curve_fit

def logarithmic(x, a, b):
    # ensure argument to log is ≥ 1e-3
    return a * np.log(np.maximum(b * x + 1, 1e-3))

def fit_logarithmic_model(x, y):
    popt, _ = curve_fit(logarithmic, x, y, maxfev=10000)
    return popt
