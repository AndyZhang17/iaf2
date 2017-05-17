__author__ = 'andy17'

import numpy as np
import numpy.random as npr

class Dataset0(object):
    def __init__(self,data,autoreshuffle=True,verbose=False):
        self.keys = data.keys()
        self.data = data
        self.size = data[self.keys[0]].shape[0]
        self.idxs = np.arange(self.size)

        self.batchsize = None
        self.startidx  = 0
        self.out = dict()
        self.autoreshuffle = autoreshuffle

        for key in self.keys:
            self.out[key] = None

        self.shuffle()

        if verbose:
            shps = list()
            for key in self.keys:
                shps.append(self.data[key].shape)
            kmsg ='\n\t'.join( [ str(key)+' : '+str(shp) for key,shp in zip(self.keys,shps) ] )
            message = 'Data ready, size : %d \n keys : %s : ' %(self.size,kmsg)
            print( message )

    def shuffle(self):
        self.idxs = npr.permutation(self.size)
        self.startidx = 0

    def setBatchSize(self,size):
        self.batchsize = size

    def getSize(self):
        return self.size

    def getMiniBatch(self):
        end = self.startidx + self.batchsize
        if end>self.size:
            if not self.autoreshuffle:
                return None
            self.shuffle()
            end = self.batchsize

        ids = self.idxs[self.startidx:end]
        for key in self.keys:
            self.out[key] = self.data[key][ids]
        self.startidx = end
        return self.out, ids




