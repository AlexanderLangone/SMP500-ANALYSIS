
import numpy as np
from numpy.polynomial.polynomial import Polynomial

def fit_cyclical_model(x, y, degree=4):
    coefs = Polynomial.fit(x, y, degree).convert().coef
    return coefs
