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


class Logistic(object):
    def __init__(self, dimx):
        self.dimx = dimx

        self.wPrior = mathT.multiNormInit( mean=np.zeros(dimx),varmat=np.eye(dimx) )

        self.wn = npr.randn(dimx,1)
        self.bn = npr.randn(1)

    def setParamValues(self, w=None,b=None):
        if np.any(w) is not None:
            self.wn = w
        if np.any(b) is not None:
            self.bn = b


    def genData(self,fromx,savefile=None,verbose=False):
        # generate numpy data
        probs = mathZ.sigmoid( np.dot(fromx,self.wn)+self.bn )
        probs = np.squeeze(probs)
        y = ( npr.rand(fromx.shape[0]) < probs ) + 0   # y = {0,1}, binary
        outd = {'x':fromx,'y':y,'w':self.wn,'b':self.bn}
        
        if verbose:
            print 'Data generation, logistic\n\t' \
                  'input x  : %s\n\tlabel y  : %s, E(y) : %0.2f\n\t' \
                  'weight.T : %s\n\tbias  b  : %s' %(fromx.shape,y.shape,np.mean(y),
                                                   self.wn.T,self.bn)
            print 'Data saving path : %s\nkeys : %s' %(savefile,outd.keys())
        if savefile:
            np.savez(savefile,x=fromx,y=y,w=self.wn,b=self.bn)
        return outd



