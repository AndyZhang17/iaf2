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


def initValues( dim, batchsize, type, scale=None ):
    if type == 'LIN':
        wn = npr.randn( batchsize, dim, dim) * scale/(dim+dim) * np.triu(np.ones((dim,dim)))
        bn = np.zeros((batchsize,dim),dtype=floatX)
        un = npr.randn(batchsize,dim) * scale
        wn, un = np.asarray(wn,dtype=floatX), np.asarray(un,dtype=floatX)
        return {'w':wn,'b':bn,'u':un}
    else:
        wn = list()
        for i in range(batchsize):
            wn.append( mathZ.permutMat(dim,enforcing=True,dtype=floatX) )
        return {'w': np.asarray(wn,dtype=floatX)}


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
        weights = list()
        for i in range(batchsize):
            weights.append( mathZ.permutMat(dim,enforcing=True,dtype=floatX) )
        weights = np.asarray(weights,dtype=floatX)
        jacon    = np.zeros( (self.batchsize,self.splsize), dtype=floatX)

        self.w       = utilsT.sharedf( weights )    # d x d
        self.logjaco = utilsT.sharedf( jacon )      # B x N

    def forward(self,x):                # x: B x N x d
        ylst = list()
        for id in range(self.batchsize):
            ylst.append( T.dot(x[id], self.w[id]) )
        outy = T.concatenate(ylst).reshape( (self.batchsize,self.splsize,self.dim) )
        return outy, self.logjaco     # B x N x d, B x N

    def getParams(self):
        return []

    def setParamValues(self,values):
        self.w.set_value( np.asarray(values['w'], dtype=floatX) )

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
        bn = np.zeros( ( batchsize,dim ) )
        un = npr.randn(batchsize,dim) * self.scale

        self.mask = utilsT.sharedf( mask )
        self.w = utilsT.sharedf( wn*mask )       # B x d x d
        self.b = utilsT.sharedf( bn )            # B x d
        self.u = utilsT.sharedf( un )            # B x d

        self.wmked  = self.w * self.mask               # masked weight
        self.iwdiag = theano.shared(np.arange(dim))
        self.wdiag = self.wmked[:,self.iwdiag,self.iwdiag]

        self.params = [ self.w, self.b, self.u ]
        self.paramshapes = [ (batchsize,dim,dim), (batchsize,dim), (batchsize,dim) ]


    def reInit(self):
        dim  = self.dim
        mask = np.triu( np.ones((dim,dim)) )
        wn = npr.randn(self.batchsize,dim,dim) * self.scale/(dim+dim)
        bn = np.zeros( ( self.batchsize,dim ) )
        un = npr.randn( self.batchsize,dim )
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
        for id in range(self.batchsize):
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

    def __init__(self,dim, samplingsize, batchsize, numlayers, noisestd=1.,name=None):
        self.name = name

        self.batchsize, self.splsize, self.dim = samplingsize, dim, batchsize
        self.shape0   = ( batchsize, samplingsize, dim )
        self.shape1   = ( batchsize, samplingsize  )
        self.shapeflt = ( samplingsize * batchsize,dim )

        self.layers = []
        for i in range(numlayers):
            self.layers.append( LinLayer(    dim,samplingsize,batchsize, 'linear-%d'%(2*i) )  )
            self.layers.append( PermuteLayer(dim,samplingsize,batchsize, 'perm-%d'%(2*i+1) )  )
        self.numlayers = len( self.layers )

        # top layer noise
        self.noisestd = noisestd
        self.logPrior = mathT.multiNormInit( np.zeros(dim), np.eye(dim)*(noisestd**2) ) # N(0,std2)

    def getNoiseVar(self,seed=None):
        e = mathT.sharedNormVar2( (self.batchsize, self.splsize,self.dim), seed=seed )
        return e*self.noisestd

    def reparam(self,e):   # e : B x N x d, shape0

        loglst = list()
        for i in range(self.batchsize):
            loglst.append( self.logPrior(e[i]) )
        log0 = T.concatenate(loglst).reshape( self.shape1 )

        outys    = [e]      + [ None ]*self.numlayers
        logjacos = [ log0 ] + [ None ]*self.numlayers
        # e_flat = e.flatten(ndim=2).reshape( self.shapeflt )
        # logjacos = [ self.logPrior(e_flat).reshpae(self.shape1) ] + [None]*self.numlayers

        for k,layer in enumerate(self.layers):
            outy,  jaco   = layer.forward(outys[k])
            outys[k+1]    = outy
            logjacos[k+1] = logjacos[k] - jaco

        return outys[-1], logjacos[-1]

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

    def setParamValues(self,valuelst):
        '''

        :param paramValues: [ dict() ]
        Linear:
            # w : B x d x d
            # b : B x d
            # u : B x d
        Permutation :
            # w : d x d
        :return:
        '''
        for k, layer in enumerate( self.layers ):
            layer.setParamValues( paramValues[k] )



class ModelParamSet(object):
    def __init__(self, dim, datasize, numlayers ):
        self.datasize, self.dim = datasize, dim
        self.values = list()
        for i in range(numlayers):
            self.values.append(initValues(dim,datasize,type='LIN',scale=(.0001)**.5))
            self.values.append(initValues(dim,datasize,type='PERM') )

    def getModelValue(self,indices):
        valuelst = list()


        for nl, value_layer in enumerate(self.values):
            value = {}
            for key in value_layer.keys():
                value[key] = value_layer[key][indices]
            valuelst.append(value)

        return valuelst


    def setModelValue(self,ids,newvalues):
        for i, vl in enumerate(self.values):
            keys = vl.keys()
            for key in keys:
                self.values[i][key][ids] = np.copy( newvalues[i][key] )