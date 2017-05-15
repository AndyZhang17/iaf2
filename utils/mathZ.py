__author__ = 'andy17'

import numpy as np
import numpy.random as npr

def wrap(x,dtype=None):
    if dtype:
        return np.asarray(x,dtype=dtype)
    else:
        return np.asarray(x)


def weightsInit( dimin, dimout,scale=1.,normalise=True, dtype=None):
    out = npr.randn(dimin,dimout)
    if normalise:
        out = out *(float(scale)/(dimin+dimout))
    return wrap(out,dtype=dtype)


def biasInit( dim, mean, scale, dtype=None ):
    out = npr.randn(dim)*float(scale) + mean
    return wrap(out,dtype=dtype)


def permutMat(dim,enforcing=True,dtype=None):
    mat = npr.permutation(np.eye(dim))
    while enforcing and np.sum(np.diag(mat))!=0:
        mat = npr.permutation(np.eye(dim))
    return wrap(mat,dtype=dtype)