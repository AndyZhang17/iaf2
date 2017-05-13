__author__ = 'andy17'

import os
os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=cpu,floatX=float32"

import theano


floatX = theano.config.floatX