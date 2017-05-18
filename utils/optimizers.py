__author__ = 'andy17'

import utils.theanoGeneral as utilsT
import theano.tensor as T
import numpy as np



def sharedConst(v,offset=0.):
    shp = v.get_value().shape
    return utilsT.sharedf( np.ones(shp)*offset )


class Adagrad(object):
    def __init__(self,params, lr=0.01, epsilon=1e-8):
        self.lr = lr
        self.epsilon = epsilon
        self.Gs = [ sharedConst(p,offset=epsilon) for p in params]

    def getUpdates(self,params,grads):
        newGs = [ T.sqr(g)+G for g,G in zip(grads,self.Gs) ]
        vs = [ self.lr/T.sqrt(G)*g for g,G in zip(grads,self.Gs)]
        newps = [ p-v for p,v in zip(params,vs) ]
        return zip(params,newps) + zip(self.Gs,newGs)


class Adadelta(object):
    def __init__(self,params,lam=0.8,epsilon=1e-8):
        self.epsilon = epsilon
        self.lam = lam
        self.g2s = [ sharedConst(p,offset=epsilon) for p in params ]
        self.v2s = [ sharedConst(p,offset=epsilon) for p in params ]


    def getUpdates(self,params,grads):
        newg2s = [ self.lam*g2 + (1-self.lam)*T.sqr(g) for g2,g in zip(self.g2s,grads) ]
        vs     = [ T.sqrt(v2/g2)*g for v2,g2,g in zip(self.v2s,newg2s,grads) ]
        newv2s = [ self.lam*v2 + (1-self.lam)*T.sqr(v) for v2,v in zip(self.v2s,vs) ]
        newps  = [ p-v for p,v in zip(params,vs) ]

        updates = zip(params,newps) + zip(self.v2s,newv2s) + zip(self.g2s,newg2s)
        return updates





class SGD(object):
    '''
    SGD with momentum
    Tested
    '''
    def __init__(self, params, lr=0.01, momentum=0.0, decay=0.0):
        self.LR0 = lr
        self.mom = momentum
        self.dcy = decay

        # updated variable
        self.lr  = utilsT.sharedf(lr)
        self.it  = utilsT.sharedf(0)



    def getUpdates(self, params, grads):
        oldvs = [ sharedConst(p) for p in params ]
        newvs = [ self.mom*oldv +self.lr*g for oldv,g in zip(oldvs,grads) ]
        newps = [ p-v for p,v in zip(params,newvs) ]
        newit = self.it + 1
        newlr = self.LR0/(1+self.dcy*self.it)

        updates = zip(params,newps) + zip(oldvs,newvs) + [ (self.lr,newlr), (self.it,newit) ]
        return updates




