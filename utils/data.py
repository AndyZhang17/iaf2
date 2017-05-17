__author__ = 'andy17'

import numpy as np
import numpy.random as npr

class Dataset0(object):
    def __init__(self,data,outkeys,autoreshuffle=True,verbose=False):
        self.keys    = data.keys()
        self.outkeys = outkeys

        self.data = data
        self.size = int( data['size'] )
        self.idxs = np.arange(self.size)

        self.startidx  = 0
        self.batchsize = None
        self.autoreshuffle = autoreshuffle
        self.shuffle()

        if verbose:
            shps = [ self.data[key].shape for key in self.outkeys ]
            kmsg = '\n\t'.join( [key+' : '+str(shp) for key,shp in zip(self.outkeys,shps)] )
            message = 'Data ready\n size : %d \n keys : %s : ' %(self.size,kmsg)
            print( message )

    def shuffle(self):
        self.idxs = npr.permutation(self.size)
        self.startidx = 0

    def setBatchSize(self,size):
        self.batchsize = size

    def getAllData(self):
        return self.data

    def getMiniBatch(self):
        end = self.startidx + self.batchsize
        if end>self.size:
            if not self.autoreshuffle:  # reaching the end of the dataset
                return None
            self.shuffle()
            end = self.batchsize

        ids = self.idxs[ self.startidx:end ]
        out = dict()
        for key in self.outkeys:
            out[key] = self.data[key][ids]
        self.startidx = end
        return out, ids




