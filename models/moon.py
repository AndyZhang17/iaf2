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

floatX = utils.floatX

class Logistic(object):
    def __init__(self,dimx,debug=False):
        self.dimx = dimx
        self.debug = debug

        self._wPrior  = mathT.multiNormInit(mean=np.zeros(self.dimx),varmat=np.eye(self.dimx))
        self._wPriorn = mathZ.multiNormInit(mean=np.zeros(self.dimx),varmat=np.eye(self.dimx))

        self._unitNorm  = mathT.normInit(0,1)
        self._unitNormn = mathZ.normInit(0,1)

        self.priorcenter = 1.

    def setPriorCenter(self,num):
        self.priorcenter = num

    def logPy_xw(self,x,y,ws,bs):
        '''
         log( p( y | x,w ) ), inputs are all TensorVariables
         N : number of data points, L : number of w samples
         :param x:  ( N , dimx )
         :param y:  ( N , )
         :param ws: ( L , dimx )
         :param bs: ( L , )
         :return: L, L x N, logP(y|x,w)
        '''
        y_ = y.dimshuffle(['x',0])
        presig = T.dot( ws, x.T ) + bs.dimshuffle([0,'x'])
        sigall = T.nnet.sigmoid( presig )   # (L, N) + (L, )
        proball = sigall * y_ + ( 1.-sigall ) * (1-y_) + 1e-9
        logall = T.log( proball )
        if self.debug:
            return T.sum(logall,axis=1), logall, y_, sigall, proball
        return T.mean(logall,axis=1), logall

    def logPw(self,ws):
        '''
        :param ws: TensorVaraiables, ( L, dimx )
        :return:
        '''
        return self._wPrior(ws)

    def logPw2(self,ws):
        wmag = T.sqrt( T.sum( T.sqr(ws),axis=1 )/self.dimx )
        return self._unitNorm( wmag - self.priorcenter )

    def nlogPy_xw(self,x,y,ws,bs):
        #  x : ( N, Dx ),    y : ( N, )
        # ws : ( L, Dx ),   bs : ( L, )
        wn, bn = np.asarray(ws),np.asarray(bs)
        y_ = y.reshape((1,y.shape[0]))  # ( 1, N )
        sigall = mathZ.sigmoid( np.dot( wn, x.T ) + bn.reshape( (bn.shape[0],1) ) )   # ( L, N ) + ( L, )
        sigall_neg = 1. - sigall
        proball = sigall*y_ + sigall_neg*(1-y_)
        logall = np.log( proball )
        return np.mean( logall, axis=1), logall

    def nlogPw(self,ws):
        wn = np.asarray(ws)
        return self._wPriorn(wn)

    def nlogPw2(self,ws):
        ws = np.asarray(ws)
        wmag = np.sqrt( np.sum( np.square(ws), axis=1 )/self.dimx )
        return self._unitNormn( wmag - self.priorcenter )













class Multiclass(object):
    def __init__(self, dimx, dimy):
        # y: one-of-k labeling
        self.dimx = dimx
        self.dimy = dimy # >= 2
        self.dimwflat = dimx*dimy

        # p(w), tensor function
        self._wPrior  = mathT.multiNormInit( mean=np.zeros(self.dimwflat),varmat=np.eye(self.dimwflat) )
        self._wPriorn = mathZ.multiNormInit( mean=np.zeros(self.dimwflat),varmat=np.eye(self.dimwflat) )

        # true params
        self.wn_true = npr.randn(dimx,dimy)   # dimx * dimy, dimy = K
        self.bn_true = npr.randn(dimy)        # dimy


    # Numpy.ndarray funcitons

    def setTrueParamValues(self, w=None,b=None):
        w = np.asarray(w,dtype=floatX)
        b = np.asarray(b,dtype=floatX)
        if np.any(w) is not None:
            self.wn_true = w.reshape( (self.dimx,self.dimy) )
        if np.any(b) is not None:
            self.bn_true = b.reshape((self.dimy,))

    def genData(self,fromx,savefile=None,verbose=False):
        # generate numpy data
        probs = mathZ.softmax( np.dot(fromx,self.wn_true) + self.bn_true )
        y, label = mathZ.randFromSftmx(probs)  # y = {0,1}, binary
        outd = {'x':fromx,'y':y,'label':label,'w':self.wn_true,'b':self.bn_true}
        if verbose:
            print 'Data generation, logistic\n\t' \
                  'input x  : %s\n\t1-of-K y  : %s\n\t' \
                  'weight.T : %s\n\tbias  b  : %s' %( fromx.shape,y.shape,
                                                   self.wn_true.T,self.bn_true)
            print 'Data saving path : %s\nkeys : %s' %( savefile,outd.keys() )
        if savefile:
            np.savez(savefile,x=fromx,y=y,label=label,w=self.wn_true,b=self.bn_true)
        return outd

    def nlogPw(self,ws):
        wn = np.asarray(ws)
        L  = wn.shape[0]
        return self._wPriorn( wn.reshape( (L,-1) ) )

    def nlogPy_xw(self,x,y,ws,bs):
        #  x : ( N, Dx ),       y : ( N, K )
        # ws : ( L, Dx, K ),   bs : ( L, K )
        wn, bn = np.asarray(ws), np.asarray(bs)
        L, DX, K = wn.shape
        wn_ = np.transpose(wn, axes=[0,2,1])             # ( L, K, Dx )
        presig = np.dot( wn_, x.T ) + bn.reshape( (L,K,1) )   # ( L, K, N )

        presig = np.transpose( presig, axes=[0,2,1] )       # ( L, N, K )
        probs_ = mathZ.softmax( presig.reshape( (-1,K) ) ).reshape( presig.shape )     # ( L*N,  K )
        probs = np.sum( probs_ * y.reshape( (1,-1,K) ), axis=2 )       # ( L, N, K )
        logprobs = np.log( probs )                        # ( L, N )
        return np.mean( logprobs, axis=1 ), logprobs   # ( L, ), ( L, N )

    def nPredict(self,x,ws,bs):
        wn, bn = np.asarray(ws), np.asarray(bs)
        L, DX, K = wn.shape
        N = x.shape[0]
        wn_ = np.transpose(wn, axes=[0,2,1])             # ( L, K, Dx )
        presig = np.dot( wn_, x.T ) + bn.reshape( (L,K,1) )   # ( L, K, N )
        presig = np.transpose( presig, axes=[0,2,1] )       # ( L, N, K )
        probs_ = mathZ.softmax( presig.reshape( (-1,K) ) ).reshape( presig.shape )     # ( L, N,  K )
        labelout = np.argmax( probs_, axis=2 )
        # yout = np.zeros( (L,N,K) )
        # yout[np.arange(L),np.arange(N),labelout] = 1
        return labelout

    def nAcc(self,truelabel,labels):
        # (N,), (L,N)
        truth = truelabel.reshape((1,-1))
        flags = ( np.abs( labels - truth ) == 0 )+0
        return np.mean(flags,axis=1), flags



    # theano TensorVariable funcitons

    def logPy_xw(self,x,y,ws,bs):
        '''
         log( p(y|x,w) ), inputs are all TensorVariables
         N : number of data points, L : number of w samples
         :param x:  N x dimx
         :param y:  N x dimy-K
         :param ws: L x dimx x dimy
         :param bs: L x dimy
         :return: L, L x N, logP(y|x,w)
        '''
        ws_ = ws.dimshuffle( [0,2,1] )
        presig_ = T.dot( ws_, x.T ) + bs.dimshuffle( [0,1,'x'] )

        presig = presig_.dimshuffle( [0,2,1] )
        shps = presig.shape.eval()
        probs_ = T.nnet.softmax( presig.reshape((-1,self.dimy) ) ).reshape( shps )
        probs = T.sum( probs_ * y.dimshuffle(['x',0,1]), axis=2 )
        logprobs = T.log(probs)
        return T.mean( logprobs, axis=1 ), logprobs

    def logPw(self,ws,bs):
        '''
        :param ws: L x dimx x dimy
        :param bs: L x K
        :return:   L
        '''
        ws_ = ws.reshape((-1,self.dimwflat))
        return self._wPrior(ws_)








