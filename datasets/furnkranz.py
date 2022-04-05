from ast import arg
from typing import Tuple
import numpy as np
import random


def login_function(
    a,
    b,
    c,
    d,
    e,
    f,
    h,
    i,
    j,
) -> bool:
    return (
        ((a and i) or c or j) and
        ((not b and i) or not d or h)) or \
        (b and d and f and h)

def get_dataset(N) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    for i in range(0, N):
        args = [random.choice([True, False]) for i in range(0, 9)]
        args_with_negations = []
        for arg in args:
            args_with_negations.append(arg)
            args_with_negations.append(not arg)
        X.append(args_with_negations)
        y.append(login_function(*args))
    return np.array(X, dtype=int), np.array(y, dtype=int)
