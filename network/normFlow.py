__author__ = 'andy17'
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import theano.tensor.nlinalg as tlin
import utils
import utils.theanoGeneral as utilsT
import utils.mathT as mathT
import utils.mathZ as mathZ


class NormFlowLayer(object):
    def __init__(self,dim,name):
        self.dim=dim
        self.name=name
    def forward(self,x):
        # return ( output, log|jacobite| )
        pass
    def getParams(self):
        pass

class PermuteLayer(NormFlowLayer):
    def __init__(self,dim,name=None):
        super(PermuteLayer,self).__init__(dim,name)
        self.w = utilsT.sharedf( mathZ.permutMat(dim,enforcing=True) )
        self.logjaco = utilsT.sharedf(0.)
    def forward(self,x):
        return T.dot(x,self.w),self.logjaco
    def getParams(self):
        return []


class LinLayer(NormFlowLayer):
    def __init__(self,dim,name=None):
        super(LinLayer,self).__init__(dim,name)

        # define weight mask and weight
        mask = np.triu( np.ones((dim,dim)) )
        weight = mathZ.weightsInit(dim,dim,scale=1.,normalise=True)
        self.mask = utilsT.sharedf( mask )
        self.w = utilsT.sharedf( weight*mask )
        self.b = np.zeros(dim)
        self.u = np.ones(dim)

        self.wmked = self.mask*self.w
        self.params = [ self.w, self.b, self.u ]
        self.paramshapes = [ (dim,dim), (dim,), (dim,) ]

        self.wdiag = tlin.extract_diag(self.w)

    def forward(self,x):
        # x: Nxd
        pretanh = T.dot(x, self.wmked ) +self.b   # N x d
        coshsqr = T.sqr( T.cosh( pretanh ) )      # N x d

        logjaco = T.sum( T.log( T.abs_( 1.+self.u*coshsqr*self.wdiag ) ), axis=1 )
        y = x + self.u * T.tanh( pretanh )
        return y, logjaco

    def getParams(self):
        return self.params



class NormFlowModel(object):
    def __init__(self,dim,numlayers,name=None):
        self.dim=dim
        self.name=name
        self.layers = []
        for i in range(numlayers):
            self.layers.append( LinLayer(dim,'linear-%d'%(i))  )
            self.layers.append( PermuteLayer(dim,'perm-%d'%(i)) )

    def logPrior(self,e):
        # TODO gaussian noise prior function
        pass

    def reparam(self,e):
        outputs  = [e] + [None]*len(self.layers)
        logjacos = [self.logPrior(e)] + [None]*len(self.layers)
        for k,layer in enumerate(self.layers):
            out, jaco = layer.forward(outputs[k])
            outputs[k+1] = out
            logjacos[k+1] = logjacos[k] - jaco
        return outputs[-1], logjacos[-1]




