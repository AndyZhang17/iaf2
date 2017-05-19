__author__ = 'andy17'

import math
import numpy as np
import numpy.random as npr
import numpy.linalg as nlin


PI = math.pi

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


def normInit(mean,var):
    const = -.5*np.log(2*PI) - .5*np.log(var)
    def loglik(x):
        subx = x-mean
        subxsqr = subx*subx/var
        return -subxsqr/2. + const
    return loglik


def multiNormInit(mean,varmat):
    d = mean.shape[0]
    const  =  - d/2.*np.log(2*PI) - 0.5*np.log( np.abs(nlin.det(varmat)) )
    varinv = nlin.inv(varmat)
    def loglik(x):
        subx = x-mean
        subxcvt = np.dot(subx,varinv)
        subxsqr = subx*subxcvt
        return - np.sum(subxsqr, axis=1)/2. + const
    return loglik



'''
multi-variate GMM
'''
def multiGmmInit(means,varmats,weights):
    logliks = list()
    for i,weight in enumerate(weights):
        logliks.append(multiNormInit(means[i],varmats[i]))
    def loglik(x):
        out = np.zeros(x.shape[0])
        for i,f in enumerate(logliks):
            out = out + np.exp(f(x))*weights[i]
        return np.log(out)
    return loglik


'''
General
'''
def gridPoints(xparams,yparams):
    x = np.arange(xparams[0],xparams[1],xparams[2])
    y = np.arange(yparams[0],yparams[1],yparams[2])
    xx, yy = np.meshgrid(x,y)
    # xax, yax = np.meshgrid(x,y,sparse=True)
    points = np.vstack( [ xx.ravel(),yy.ravel() ] )
    return points.T, xx, yy

def sigmoid(x):
    return 1./(1.+np.exp(-x))


