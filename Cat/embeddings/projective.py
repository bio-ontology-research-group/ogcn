import numpy as np

def homogenize(v):
    return np.append(v, 1)

def transform(mat, v):
    return mat.dot(v)

def dehomogenize(v):
    num, den = tuple(v)
    return num/den

def

#continue once the graph part is fixed
