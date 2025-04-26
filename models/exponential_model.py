import numpy as np
from scipy.optimize import curve_fit

def exponential(x, a, b):
    return a * np.exp(b * x)

def fit_exponential_model(x, y):
    popt, _ = curve_fit(
        exponential, x, y,
        maxfev=10000,
        bounds=([0, 0], [1e5, 0.1])   # a ≥ 0, b ∈ [0, 0.1]
    )
    return popt
