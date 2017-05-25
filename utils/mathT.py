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

def sharedUnifVar(num,dim,seed=None):
    trng = trands(seed=seed)
    return trng.uniform((num,dim))

def sharedNormVar(num,dim,seed=None):
    trng = trands(seed=seed)
    return trng.normal((num,dim))

def sharedNormVar2(shapein,seed=None):
    trng = trands(seed=seed)
    return trng.normal(shapein)



'''
uni-variate and multi-variate normal distributions
'''

def uniformInit(volume=1.):
    def loglik(x):
        return T.ones_like(T.sum(x,axis=1))/volume
    return loglik


def multiNormInit(mean,varmat):
    '''
    multi-variate normal distribution
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


def indepNormInit(meanlst,varlst):
    ###
    ### Untested, note the const
    mu  = np.asarray(meanlst)       # ( K, )
    var = np.asarray(varlst)        # ( K, )
    const = -0.5*np.log(2*PI) - 0.5*np.log(var)   # ( K, )

    mut  = utils.theanoGeneral.sharedf(mu)   # ( K, )
    vart = utils.theanoGeneral.sharedf(var)
    cstt = utils.theanoGeneral.sharedf(const)

    mu_    =  mut.dimshuffle(['x',0])
    var_   = vart.dimshuffle(['x',0])
    const_ = cstt.dimshuffle(['x',0])

    def loglik(x):
        # x : ( N, K )
        # return : ( N, )
        subx2 = T.sqr( x - mu_ )/var_/2.
        logs = -subx2 + const_
        return T.sum(logs,axis=1)
    return loglik

def indepNormInit_sharedParams(means,varlst):
    const = -.5*np.log(2*PI) - .5*T.log(varlst) # ( K, )
    const = const.dimshuffle(['x',0])  # ( 1,K )
    def loglik(x):  # ( N, K )
        subx2 = T.sqr( x-means )/varlst/2
        logs = -subx2 + const
        return T.sum(logs,axis=1)
    return loglik




def normInit(mean,var):
    '''
    single variate normal distribution
    '''
    const = -0.5*np.log(2*PI) - 0.5*np.log(var)
    def loglik(x):
        subx = x-mean
        subxsqr = subx*subx/var
        return -subxsqr/2. + const
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



'''
multi-variate GMM
'''

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



