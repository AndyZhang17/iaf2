__author__ = 'andy17'

import theano
import theano.tensor as T
import numpy as np
import numpy.random as npr
import sys
import utils





def sharedf(x, target=None, name=None, borrow=False):
    if target:
        return theano.shared(np.asarray(x,dtype=floatX),target=target,name=name,borrow=borrow)
    else:
        return theano.shared(np.asarray(x,dtype=floatX),name=name,borrow=borrow)



