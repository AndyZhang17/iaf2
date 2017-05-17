__author__ = 'andy17'

import theano
import theano.tensor as T
import models as M
import utils
import utils.theanoGeneral as utilsT
import utils.mathT as mathT
import numpy as np
import numpy.random as npr

'''
models, with batch evaluation
'''

class Banana(M.GraphModel):
    '''
    Graphical model

        (z1)    (z2)
          \     /
           \   /
           ( x )
                   p(z) = N( z ; 0, I )
                   p(x|z) = N( x ; z1*z2, std^2 )
    '''

    def __init__(self, name=None):
        super(Banana,self).__init__(name)

        self.dimx, self.dimz = 1, 2

        # true parameter of the function
        self.stdx_ztrue = 0.7

        # shared params, to be updated (learnt)
        self.stdx_z = utilsT.sharedf(1.)
        self.params = [ self.stdx_z ]

        # prior of z : wont change
        self.logP_z       = mathT.multiNormInit( mean=np.zeros(self.dimz), varmat=np.eye(self.dimz) )
        # ( x-z1*z2 ) ~ N( 0, sigma2 )
        self.logP_uninorm = mathT.normInit_sharedParams(mean=utilsT.sharedf(0), var=T.sqr(self.stdx_z), offset=None)

    def setX(self,x):  # x : B x dimX
        # x is a TheanoVariable
        self.x = x


    def setParamValues(self,values):
        self.stdx_z.set_value( np.asarray(values['std'],dtype=utils.floatX) )

    def setTrueParamValues(self,values):
        self.stdx_ztrue = values['std']

    def logPxz(self, x, z):
        # z : B x N x dimZ
        # x : B
        (B,N,dimZ) = z.shape.eval()

        zprods = T.prod( z,axis=2 )
        subs = zprods - x.dimshuffle(0,'x')
        logpx_zs = self.logP_uninorm(subs)   # p(x|z), B x N

        logpzlst = list()   # p(z), B x N
        for id in range(B):
            logpz = self.logP_z(z[id])      # N
            logpzlst.append(logpz)
        logpzs = T.concatenate(logpzlst).reshape( (B,N) )  # B x N

        return logpzs+logpx_zs, logpx_zs, logpzs

    def getParams(self):
        return self.params

    def generate(self,num,savepath):
        zs = npr.randn(num,self.dimz)
        xs = npr.randn(num)*self.stdx_ztrue + np.prod(zs,axis=1)
        np.savez(savepath,x=xs,z=zs,std=self.stdx_ztrue,size=num)




