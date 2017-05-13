__author__ = 'andy17'

import utils
import theano
import numpy as np
import theano.tensor as T
import theano.tensor.nlinalg as tlin
'''
by default, all input/output variables in this part are theano symbolic vars

'''

PI = utils.PI


def multiNormInit(mean,varmat):
    d = T.sum( T.ones_like(mean) )
    const = - d/2.*np.log(2*PI) - 0.5*T.log( T.abs_(tlin.det(varmat)) )
    varinv = tlin.matrix_inverse(varmat)
    def loglik(x):
        subx = x - mean
        subxcvt = T.dot(subx,varinv)   # Nxd
        subxsqr = subx*subxcvt.T       # Nxd
        return - T.sum( subxsqr, axis=1 )/2. + const
    return loglik




