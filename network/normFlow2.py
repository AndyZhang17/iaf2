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
import numpy.random as npr

floatX = utils.floatX

'''
normalisation flow, with batch inference enabled
'''


class NormFlowLayer(object):
    def __init__( self, dim, samplingsize, batchsize, name ):
        self.name = name
        self.dim = dim
        self.splsize = samplingsize
        self.batchsize = batchsize

    def forward(self,x):
        # return ( output, log|jacobite| )
        pass
    def getParams(self):
        pass

    def reInit(self):
        pass

    def getParamValues(self):
        return {}

    def setParamValues(self,values):
        pass



class PermuteLayer(NormFlowLayer):
    def __init__(self,dim, samplingsize, batchsize,name=None):
        super(PermuteLayer,self).__init__(dim, samplingsize, batchsize,name)

        # all batch members share the same permutation matrix
        permmat = mathZ.permutMat(dim,enforcing=True,dtype=floatX)
        jacon    = np.zeros(self.batchsize,self.splsize,dtype=floatX)

        self.w       = utilsT.sharedf( permmat )    # d x d
        self.logjaco = utilsT.sharedf( jacon )      # B x N

    def forward(self,x):                # x: B x N x d
        ylst = list()
        for id in range(self.batchsize):
            y = T.dot(x[id], self.w)
        outy = T.concatenate(ylst).reshape((self.batchsize,self.splsize,self.dim))
        return outy, self.logjaco     # B x N x d, B x N

    def getParams(self):
        return []

    def setParamValues(self,values):
        self.w.set_value( np.asarray(values['w'],dtype=floatX) )

    def getParamValues(self):
        return { 'w':self.w.get_value() }



class LinLayer(NormFlowLayer):
    def __init__( self, dim, samplingsize, batchsize, name=None ):
        super(LinLayer,self).__init__(dim, samplingsize,batchsize,name)

        # define weight mask and weight
        self.scale = (.0002/self.dim)**.5

        # values setups
        mask = np.triu( np.ones((dim,dim)) )
        wn = npr.randn(batchsize,dim,dim) * self.scale/(dim+dim)
        bn = np.zeros( batchsize,dim)
        un = npr.randn(batchsize,dim) * self.scale

        self.mask = utilsT.sharedf( mask )
        self.w = utilsT.sharedf( wn*mask )
        self.b = utilsT.sharedf( bn )
        self.u = utilsT.sharedf( un )

        self.wmked  = self.w * self.mask               # masked weight
        self.iwdiag = theano.shared(np.arange(dim))
        self.wdiag = self.wmked[:,self.iwdiag,self.iwdiag]

        self.params = [ self.w, self.b, self.u ]
        self.paramshapes = [ (batchsize,dim,dim), (batchsize,dim), (batchsize,dim) ]


    def reInit(self):
        dim  = self.dim
        mask = np.triu( np.ones((dim,dim)) )
        wn = npr.randn(self.batchsize,dim,dim) * self.scale/(dim+dim)
        bn = np.zeros( self.batchsize,dim)
        un = npr.randn(self.batchsize,dim)
        self.w.set_value( np.asarray( wn*mask, dtype=floatX ) )
        self.b.set_value( np.asarray( bn, dtype=floatX ) )
        self.u.set_value( np.asarray( un, dtype=floatX ) )

    def setParamValues(self,values):
        self.w.set_value( np.asarray(values['w'],dtype=floatX) )
        self.b.set_value( np.asarray(values['b'],dtype=floatX) )
        self.u.set_value( np.asarray(values['u'],dtype=floatX) )

    def getParamValues(self):
        return {'w':self.w.get_value(),'b':self.b.get_value(),'u': self.b.get_value()}

    def forward(self,x):  # x: B x N x d
        ys = list()
        logjacos = list()
        for id in self.batchsize:
            pretanh = T.dot( x[id], self.wmked[id] ) + self.b[id]
            coshsqr = T.sqr( T.cosh( pretanh ) )
            y = x[id] + self.u[id] * T.tanh( pretanh )                  # N x d
            logjaco = T.sum( T.log( T.abs_( 1.+self.u[id]/coshsqr*self.wdiag[id] ) ), axis=1 )

            ys.append(y)
            logjacos.append(logjaco)
        outy       = T.concatenate(ys).reshape( (self.batchsize,self.splsize,self.dim) )   # B x N x d
        outlogjaco = T.concatenate(logjacos).reshape( (self.batchsize,self.splsize) )      # B x N
        return outy, outlogjaco

    def getParams(self):
        return self.params



class NormFlowModel(object):

    def __init__(self,dim,numlayers,noisestd=1.,name=None):
        self.dim = dim
        self.name = name
        self.layers = []
        for i in range(numlayers):
            self.layers.append( LinLayer(    dim,'linear-%d'%(2*i) )  )
            self.layers.append( PermuteLayer(dim,'perm-%d'%(2*i+1) )  )
        # top layer noise
        self.noisestd = utilsT.sharedf(noisestd)
        self.logPrior = mathT.multiNormInit( np.zeros(dim), np.eye(dim)*(noisestd**2)  )

    def getNoiseVar(self,samplingsize,seed=None):
        e = mathT.sharedNormVar(samplingsize,self.dim,seed=seed)
        return e*self.noisestd

    def reparam(self,e):
        outputs  = [e] + [None]*len(self.layers)
        logjacos = [self.logPrior(e)] + [None]*len(self.layers)
        for k,layer in enumerate(self.layers):
            out, jaco = layer.forward(outputs[k])
            outputs[k+1] = out
            logjacos[k+1] = logjacos[k] - jaco
        return outputs[-1], logjacos[-1]

    def getParams(self):
        params = []
        for layer in self.layers:
            params.extend(layer.getParams())
        return params

    def reInit(self):
        for layer in self.layers:
            layer.reInit()

    def getParamValues(self):
        paramValues = list()
        for layer in self.layers:
            paramValues.append( layer.getParamValues() )
        return paramValues

    def setParamValues(self,paramValues):
        for k, layer in enumerate( self.layers ):
            layer.setParamValues(paramValues[k])




