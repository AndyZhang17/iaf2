__author__ = 'andy17'

import utils.theanoGeneral as utilsT
import theano.tensor as T



class sgd_nesterov(object):
    def __init__(self, params):
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, params, grads, learning_rate, momentum):
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            update = momentum * memory - learning_rate * grad
            update2 = momentum * momentum * memory - (
                1 + momentum) * learning_rate * grad
            updates.append((memory, update))
            updates.append((param, param + update2))
        return updates



class SGD(object):
    def __init__(self,lr=0.01,momentum=0.0,decay=0.0):
        self.lr = utilsT.sharedf(lr)
        self.iter = utilsT.sharedf(0)
        self.mom = utilsT.sharedf(momentum)
        self.dcy = utilsT.sharedf(decay)

    def getUpdates(self, params, grads):
        def sharedZero(v):
            sp = v.get_value().shape
            return utilsT.sharedf( np.zeros(sp) )
        olddeltas = [ sharedZero(grad) for grad in grads ]
        newdeltas = [ self.mom*oldd + self.lr*g for oldd,g in zip(olddeltas,grads) ]
        newparams = [ p-dlta for p,delta in zip(params,newdeltas) ]

        newiter = self.iter + 1
        newlr   = self.lr/(1+self.dcy*self.iter)

        updates   = zip(olddeltas,newdeltas) + zip(params,newparams) + [(self.lr,newlr),(self.iter,newiter)]
        return updates


