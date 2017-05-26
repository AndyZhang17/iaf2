__author__ = 'andy17'
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import theano.tensor.nlinalg as tlin
import utils
import utils.theanoGeneral as utilsT
import utils.mathT as mathT
import utils.mathZ as mathZ
import numpy as np
import numpy.linalg as nlin
floatX = utils.floatX



class MeanField(object):
    def __init__(self,dim,name=None):
        self.name = name
        self.dim = dim

        self.means = utilsT.sharedf( np.zeros(dim) )
        self.vars  = utilsT.sharedf( np.ones(dim) )

        self.varmat = tlin.diag(self.vars)
        self.rmat  = tlin.diag(T.sqrt(self.vars))
        self.means_ = self.means.dimshuffle(['x',0])
        self.qzft = mathT.multiNormInit_sharedParams(self.means,self.varmat,self.dim)
        self.qzfn = None
        self.params = [self.means,self.vars]

    def getParams(self):
        return self.params

    def getZ(self,samplingsize,seed=None):
        e = mathT.sharedNormVar(samplingsize,self.dim,seed=seed)
        z = T.dot(e,self.rmat) + self.means_
        return z, self.qzft(z)

    def nlogqz(self,z,renew=True):
        if renew or ( self.qzfn is None ):
            self.qzfn = mathZ.multiNormInit(self.means.eval(),self.varmat.eval())
        return self.qzfn(z)

    def getParamValues(self):
        values = {}
        values['mus']  = self.means.eval()
        values['vars'] = self.vars.eval()
        return values

    def setParamValues(self,values):
        self.means.set_value( np.asarray(values['mus'],  dtype=floatX))
        self.vars.set_value(  np.asarray(values['vars'], dtype=floatX))




