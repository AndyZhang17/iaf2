__author__ = 'andy17'

import theano
import theano.tensor as T
import models as M
import utils
import utils.theanoGeneral as utilsT
import utils.mathT as mathT
import numpy as np
import numpy.random as npr


class Banana(M.GraphModel):
    def __init__(self,name=None):
        super(Banana,self).__init__(name)
        self.dimx, self.dimz = 1, 2

        self.stdx_ztrue = 0.7

        self.stdx_z = utilsT.sharedf(1.)
        self.params = [self.stdx_z]

        self.logP_z       = mathT.multiNormInit( mean=np.zeros(self.dimz), varmat=np.eye(self.dimz) )  # wont change
        self.logP_uninorm = mathT.normInit_sharedParams(mean=utilsT.sharedf(0), var=T.sqr(self.stdx_z), offset=None)

    def setX(self,x):
        self.x = x

    def setStd(self,std):
        self.stdx_z.set_value(np.asarray(std,dtype=utils.floatX))

    def setTrueStd(self,std):
        self.stdx_ztrue = std

    def logPxz(self,z):                      # z : N x dimZ
        zprods = T.prod(z,axis=1)
        subs = zprods - self.x
        logpx_z = self.logP_uninorm(subs)    # N
        logpz = self.logP_z(z)               # N
        return logpz+logpx_z, logpx_z, logpz

    def getParams(self):
        return self.params

    def generate(self,num,savepath):
        zs = npr.randn(num,self.dimz)
        xs = npr.randn(num)*self.stdx_ztrue + np.prod(zs,axis=1)
        np.savez(savepath,x=xs,z=zs,std=self.stdx_ztrue)




