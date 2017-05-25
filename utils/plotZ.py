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


def hist2d_II(points,bins=50,axes=(0,1)):
    x = points[:,axes[0]]
    y = points[:,axes[1]]
    heatmap, xedges, yedges = np.histogram2d(x,y,bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.figure()
    plt.clf()
    plt.imshow(heatmap.T,extent=extent,origin='lower')
    plt.show()


def category2d(points,cat,dotarea=10,axes=(0,1)):
    fig = plt.figure()
    x,y = points[:,axes[0]],points[:,axes[1]]
    area = dotarea
    cs = cat
    plt.scatter( x,y, s=area, c=cs, alpha=1.0)
    return fig

