import numpy as np

def diskrand(*args):
    result = 2j * np.pi * np.random.rand(*args)
    np.exp(result, out = result)
    magnitude = np.random.rand(*args)
    np.sqrt(magnitude, out = magnitude)
    result *= magnitude
    return result
