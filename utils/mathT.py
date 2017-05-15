__author__ = 'andy17'

import utils
import utils.theanoGeneral as utilsT
import theano
import numpy as np
import numpy.linalg as nlin
import theano.tensor as T
import theano.tensor.nlinalg as tlin
from theano.tensor.shared_randomstreams import RandomStreams as trands

'''

'''

PI = utils.PI

# def multiNormInit(mean,varmat):
#     d = T.sum( T.ones_like(mean) )
#     const = - d/2.*np.log(2*PI) - 0.5*T.log( T.abs_(tlin.det(varmat)) )
#     varinv = tlin.matrix_inverse(varmat)
#     def loglik(x):
#         subx = x - mean
#         subxcvt = T.dot(subx,varinv)   # Nxd
#         subxsqr = subx*subxcvt       # Nxd
#         return - T.sum( subxsqr, axis=1 )/2. + const
#     return loglik

def multiNormInit(mean,varmat):
    '''
    :param mean: numpy.ndarray, (d,)
    :param varmat: numpy.ndarray, (d,d)
    :return: log-pdf function, linking theano.tensors
    '''
    d = mean.shape[0]
    const  =  - d/2.*np.log(2*PI) - 0.5*np.log( np.abs(nlin.det(varmat)) )
    varinv = nlin.inv(varmat)

    mean_   = utilsT.sharedf( mean )
    const_  = utilsT.sharedf( const  )
    varinv_ = utilsT.sharedf( varinv )

    def loglik(x):
        subx = x-mean_
        subxcvt = T.dot(subx,varinv_)
        subxsqr = subx*subxcvt
        return - T.sum(subxsqr, axis=1)/2. + const_
    return loglik

def sharedNormVar(num,dim,seed=None):
    trng = trands(seed=seed)
    return trng.normal((num,dim))

def multiGmm(means,varmats,weights):
    '''
    :param means:   numpy.ndarray, (N,d)
    :param varmats: numpy.ndarray, (N,d,d)
    :param weights: numpy.ndarray, (N,)
    :return: log-pdf function, linking theano.tensors
    '''
    logliks = list()
    for i,weight in enumerate(weights):
        logliks.append( multiNormInit(means[i,:], varmats[i,:,:]) )
    def loglik(x):
        out = T.zeros_like( T.sum(x,axis=1) )
        for i,f in enumerate(logliks):
            out = out + T.exp(f(x))*weights[i]
        return T.log(out)
    return loglik



