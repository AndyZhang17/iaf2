__author__ = 'andy17'
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

def hist2d(x,y,bins=50):
    heatmap, xedges, yedges = np.histogram2d(x,y,bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.figure()
    plt.clf()
    plt.imshow(heatmap.T,extent=extent,origin='lower')
    plt.show()

def category2d(points,cat,area=10,colorlims=None):
    x,y = points[:,0],points[:,1]
    area = area
    cs = cat
    plt.scatter( x,y, s=area, c=cs, alpha=1.0)
    plt.show()