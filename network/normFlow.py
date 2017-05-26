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


class NormFlowLayer(object):
    def __init__(self,dim,name):
        self.dim=dim
        self.name=name

    def forward(self,x):
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
    def __init__(self,dim,name=None):
        super(PermuteLayer,self).__init__(dim,name)
        self.w = utilsT.sharedf( mathZ.permutMat(dim,enforcing=True) )
        self.logjaco = utilsT.sharedf(0.)

    def forward(self,x):
        return T.dot(x,self.w), self.logjaco

    def getParams(self):
        return []

    def setParamValues(self,values):
        self.w.set_value( np.asarray(values['w'],dtype=floatX) )

    def getParamValues(self):
        return { 'w':self.w.get_value() }




class LinLayer(NormFlowLayer):
    def __init__(self,dim,name=None,scale=None):
        super(LinLayer,self).__init__(dim,name)

        # define weight mask and weight
        self.scale = (.0002/self.dim)**.5
        if scale:
            self.scale = scale
        mask = np.triu( np.ones((dim,dim)) )
        weight = mathZ.weightsInit(dim,dim,scale=self.scale,normalise=True)      # TODO scaling

        self.mask = utilsT.sharedf( mask )
        self.w = utilsT.sharedf( weight*mask )
        self.b = utilsT.sharedf( np.zeros(dim) )
        self.u = utilsT.sharedf( mathZ.biasInit(dim,mean=0,scale=self.scale)/2 )

        self.wmked  = self.mask*self.w            # masked weight
        self.wdiag  = tlin.extract_diag(self.wmked)
        self.params = [ self.w, self.b, self.u ]
        self.paramshapes = [ (dim,dim), (dim,), (dim,) ]


    def reInit(self):
        dim  = self.dim
        mask = np.triu( np.ones((dim,dim)) )
        weight = mathZ.weightsInit(dim,dim,scale=self.scale,normalise=True,dtype=floatX)       # TODO scaling
        self.w.set_value( np.asarray( weight*mask, dtype=floatX ) )
        self.b.set_value( np.zeros(dim, dtype=floatX) )
        self.u.set_value( mathZ.biasInit(dim,mean=0,scale=self.scale,dtype=floatX)  )

    def setParamValues(self,values):
        self.w.set_value( np.asarray(values['w'],dtype=floatX) )
        self.b.set_value( np.asarray(values['b'],dtype=floatX) )
        self.u.set_value( np.asarray(values['u'],dtype=floatX) )

    def getParamValues(self):
        return {'w':self.w.get_value(),'b':self.b.get_value(),'u': self.u.get_value()}

    def forward(self,x):
        # x: Nxd
        pretanh = T.dot(x, self.wmked ) +self.b   # N x d
        coshsqr = T.sqr( T.cosh( pretanh ) )      # N x d
        y = x + self.u * T.tanh( pretanh )
        logjaco = T.sum( T.log( T.abs_( 1.+self.u/coshsqr*self.wdiag ) ), axis=1 )
        return y, logjaco   # N x d,  N

    def getParams(self):
        return self.params



class NormFlowModel(object):

    def __init__(self,dim,numlayers,noisestd=1.,z0type='normal',name=None, scalelst=[]):
        self.dim = dim
        self.name = name
        self.layers = []
        for i in range(numlayers):
            scale = None
            if len(scalelst)==numlayers:
                scale = scalelst[i]
            elif scalelst:
                scale = scalelst[0]
            self.layers.append( LinLayer( dim, name='linear-%d'%(2*i), scale=scale )  )
            self.layers.append( PermuteLayer( dim,'perm-%d'%(2*i+1) )  )
        # top layer noise
        self.noisestd = noisestd


        self.z0type=z0type.lower()
        if self.z0type=='normal':
            self.logPrior = mathT.multiNormInit( np.zeros(dim), np.eye(dim)*(noisestd**2)  )
        elif self.z0type=='uniform':
            self.logPrior = mathT.uniformInit(volume=noisestd**dim)

    def getNoiseVar(self,samplingsize,seed=None):
        if self.z0type=='normal':
            e = mathT.sharedNormVar(samplingsize,self.dim,seed=seed)
            return e*self.noisestd
        elif self.z0type=='uniform':
            e = mathT.sharedUnifVar(samplingsize,self.dim,seed=seed)
            return (e-0.5)*self.noisestd

    def reparam(self,e,inter=False):
        outputs  = [e] + [None]*len(self.layers)
        logjacos = [self.logPrior(e)] + [None]*len(self.layers)
        for k,layer in enumerate(self.layers):
            out, jaco = layer.forward(outputs[k])
            outputs[k+1] = out
            logjacos[k+1] = logjacos[k] - jaco
        if inter:
            return outputs, logjacos
        else:
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




