__author__ = 'andy17'

import utils
import utils.theanoGeneral as utilsT
import theano
import numpy as np
import numpy.linalg as nlin
import theano.tensor as T
import theano.tensor.nlinalg as tlin
from theano.tensor.shared_randomstreams import RandomStreams as trands
PI = utils.PI

'''

'''



'''
functions of random streams
'''
def sharedNormVar(num,dim,seed=None):
    trng = trands(seed=seed)
    return trng.normal((num,dim))


'''
uni-variate and multi-variate normal distributions
'''

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


def multiNormInit_sharedParams(mean,varmat,dim):
    '''
    :param mean:  theano.tensor.TensorVaraible
    :param varmat: theano.tensor.TensorVaraible
    :param dim: number
    :return:
    '''
    d      = dim
    const  = - d/2.*np.log(2*PI) - 0.5*T.log( T.abs_(tlin.det(varmat)) )
    varinv = tlin.matrix_inverse(varmat)
    def loglik(x):
        subx = x - mean
        subxcvt = T.dot(subx,varinv)   # Nxd
        subxsqr = subx*subxcvt       # Nxd
        return - T.sum( subxsqr, axis=1 )/2. + const
    return loglik



def normInit(mean,var):
    const = -0.5*np.log(2*PI) - 0.5*np.log(var)
    def loglik(x):
        subx = x-mean
        subxsqr = subx*subx/var
        return -T.sum(subxsqr,axis=1)/2. + const
    return loglik


def normInit_sharedParams(mean,var,offset=None):
    const = -0.5*np.log(2*PI) - 0.5*T.log(var)
    def loglik(x):
        subx = x-mean
        subxsqr = subx*subx/var
        out = -subxsqr/2. + const
        if offset:
            return T.switch( out>offset, out, offset )
        return out
    return loglik


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



