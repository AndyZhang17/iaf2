__author__ = 'andy17'

import theano
import theano.tensor as T
import models as M
import utils
import utils.theanoGeneral as utilsT
import utils.mathT as mathT
import utils.mathZ as mathZ
import numpy as np
import numpy.random as npr

PI = utils.PI

class Banana(M.GraphModel):
    def __init__(self,name=None):
        super(Banana,self).__init__(name)
        self.dimx, self.dimz = 1, 2

        self.stdx_ztrue = 0.7

        self.var_xz = utilsT.sharedf(1.)
        self.params = []

        self.logP_z       = mathT.multiNormInit( mean=np.zeros(self.dimz), varmat=np.eye(self.dimz) )  # wont change
        self.logP_uninorm = mathT.normInit_sharedParams(mean=utilsT.sharedf(0), var=self.var_xz, offset=None)

        self.normal = mathZ.normInit(0,self.var_xz.get_value())
        self.nprior = mathZ.multiNormInit(np.zeros(self.dimz),np.eye(self.dimz))

    def setX(self,x):
        self.x = x

    def setVar(self,var):
        self.var_xz.set_value(np.asarray(var,dtype=utils.floatX))
        self.normal = mathZ.normInit(0,var)

    def logPxz(self,z):                      # z : N x dimZ
        # p( x,z )
        zprods = T.prod(z,axis=1)
        subs = zprods - self.x
        logpx_z = self.logP_uninorm(subs)    # N
        logpz = self.logP_z(z)               # N
        return logpz+logpx_z, logpx_z, logpz

    def getParams(self):
        return self.params

    def nlogPxz(self,valx,z):
        zprods = np.prod(z,axis=1)
        subs =zprods - valx
        px_z = self.normal(subs)
        return px_z+self.nprior(z), px_z





class Loquat(M.GraphModel):
    def __init__(self,name=None):
        super(Loquat,self).__init__(name)
        self.dimz = 2

    def logPxz(self,z):                      # z : N x dimZ
        z1, z2  = z[:,0], z[:,1]
        zmag = T.sqrt( T.sum( T.sqr(z),axis=1 ) )
        out = .5*T.sqr( (zmag-2.)/0.4 ) - T.log( T.exp( -.5*T.sqr( (z1-2.)/.6) ) + T.exp( -.5*T.sqr((z1+2.)/.6)) )
        return -out, -out, None

    def nlogPz(self,z):
        z1, z2  = z[:,0], z[:,1]
        zmag = np.sqrt( np.sum( np.square(z),axis=1 ) )
        out = .5*np.square( (zmag-2.)/.4 ) \
              - np.log( np.exp( -.5*np.square((z1-2)/.6) ) + np.exp( -.5*np.square((z1+2)/.6)))
        return -out


class Apple(M.GraphModel):
    def __init__(self,name=None):
        super(Apple,self).__init__(name)
        self.dimz = 2

    def logPxz(self,z):                      # z : N x dimZ
        z2  = z[:,1]
        z1  = T.sin(PI/2.*z[:,0])
        out = -T.sqr((z2 - z1)/0.4)/2.        # N
        return out, out, None

    def getParams(self):
        return []

    def nlogPz(self,z):
        z2  = z[:,1]
        z1  = np.sin( PI/2.*z[:,0] )
        out = -np.square((z2 - z1)/0.4)/2.        # N
        return out



class Lemon(M.GraphModel):
    def __init__(self,name=None):
        super(Lemon,self).__init__(name)
        self.dimz = 2

    def logPxz(self,z):                      # z : N x dimZ
        z1,z2  = z[:,0], z[:,1]
        w1z  = T.sin( PI/2.*z1 )
        w2z = 3 * T.exp( -.5*T.sqr( (z1-1)/0.6 ) )
        out = -T.log(
            T.exp( -.5*T.sqr( (z2-w1z)/.35 )) + T.exp( -.5*T.sqr( (z2-w1z+w2z)/.35 ) )
        )        # N
        return -out, -out, None


    def nlogPz(self,z):
        # z = z*2.
        z1, z2  = z[:,0], z[:,1]
        w1z  = np.sin( PI/2.*z1 )
        w2z = 3 * np.exp( -.5*np.square( (z1-1)/0.6 ) )
        out = -np.log(
            np.exp( -.5*np.square( (z2-w1z)/.35 ) ) + np.exp( -.5*np.square( (z2-w1z+w2z)/.35) )
            )        # N
        return -out







class Orange(M.GraphModel):
    def __init__(self,name=None):
        super(Orange,self).__init__(name)
        self.dimz = 2

    def logPxz(self,z):                      # z : N x dimZ
        z1,z2  = z[:,0], z[:,1]
        w1z  = T.sin( PI/2.*z1 )
        # w2z = 3 * T.exp( -.5*T.sqr( (z1-1)/0.6 ) )
        w3z = 3 * T.nnet.sigmoid( (z1-1)/.3 )
        out = -T.log(
            T.exp( -.5*T.sqr( (z2-w1z)/.4 )) + T.exp( -.5*T.sqr( (z2-w1z+w3z)/.35 ) )
        )        # N
        return -out, -out, None


    def nlogPz(self,z):
        # z = z*2.
        z1, z2  = z[:,0], z[:,1]
        w1z  = np.sin( PI/2.*z1 )
        # w2z = 3 * T.exp( -.5*T.sqr( (z1-1)/0.6 ) )
        w3z = 3 * mathZ.sigmoid( (z1-1)/.3 )
        out = -np.log(
            np.exp( -.5*np.square( (z2-w1z)/.4 ) ) + np.exp( -.5*np.square((z2-w1z+w3z)/.35) )
            )        # N
        return -out

