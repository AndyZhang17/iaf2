__author__ = 'andy17'

import theano
import theano.tensor as T
import models as M
import utils
import utils.theanoGeneral as utilsT
import utils.mathT as mathT
import numpy as np


class Banana(M.GraphModel):
    def __init__(self,name=None):
        super(Banana,self).__init__(name)
        self.dimx, self.dimz = 1, 2

        self.stdx_z = utilsT.sharedf(1.)
        self.params = []

        self.logP_z       = mathT.multiNormInit(mean=np.zeros(self.dimz), varmat=np.eye(self.dimz))  # wont change

        self.logP_uninorm = mathT.normInit_sharedParams(mean=utilsT.sharedf(0), var=T.sqr(self.stdx_z))

    def setX(self,x):
        self.x = x

    def setStd(self,std):
        self.stdx_z.set_value(np.asarray(std,dtype=utils.floatX))

    def logPrior(self,z):
        return self.logP_z(z)

    def logPx_z(self,z):
        zprods = T.prod(z,axis=1)
        subs = zprods - self.x
        return self.logP_uninorm(subs)

    def logPxz(self,z):
        return self.logPrior(z) + self.logPx_z(z)

    def getParams(self):
        return self.params


