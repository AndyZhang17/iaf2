__author__ = 'andy17'


import theano
import network
import network.normFlow as nf
import config
import utils
import utils.mathT as mathT
import utils.theanoGeneral as utilsT
import numpy as np

DIM = 2
SAMPLINGNUM = 10

# defining target model
mean = utilsT.sharedf([2.0,1.0])
varmat = utilsT.sharedf( np.eye(DIM)/2 )
logTarget = mathT.multiNormInit(mean,varmat)


