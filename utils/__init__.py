__author__ = 'andy17'

import numpy as np
import config
floatX = config.floatX

import math
PI = np.asarray( math.pi, dtype=floatX )



import theanoGeneral
import optimizers
import mathT
import mathZ
import plotZ
import data


def cpDict(dicin):
    keys = dicin.keys()
    newdic = dict()
    for k in keys:
        newdic[k] = dicin[k]
    return newdic