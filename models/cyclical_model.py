
import numpy as np
from numpy.polynomial.polynomial import Polynomial

def fit_cyclical_model(x, y, degree=4):
    # Ensure the fit starts from the first actual price
    coefs = Polynomial.fit(x, y, degree).convert().coef
    coefs[0] = y[0]  # Set constant term to initial price
    return coefs
