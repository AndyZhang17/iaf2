__author__ = 'andy17'

import numpy as np
import numpy.random as npr

def weightsInit( dimin, dimout,scale=1.,normalise=True):
    if normalise:
        return npr.randn(dimin,dimout)*(float(scale)/(dimin+dimout))
    else:
        return npr.randn(dimin,dimout)

def permutMat(dim,enforcing=True):
    mat = npr.permutation(np.eye(dim))
    while enforcing and np.sum(np.diag(mat))!=0:
        mat = npr.permutation(np.eye(dim))
    return mat