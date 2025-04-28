import numpy as np
from scipy.optimize import curve_fit

def power_law(x, a, c):
    return a * np.power(x, c)

def fit_power_law_model(x, y):
    # Bound a ≥ 0, c free
    popt, _ = curve_fit(
        power_law, x, y,
        maxfev=10000,
        bounds=([y[0], -np.inf], [np.inf, np.inf])  # Set lower bound of a to initial price
    )
    return popt
