__author__ = 'andy17'
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def category2d(points,cat,dotarea=10,axes=(0,1),cmap=None):
    fig = plt.figure()
    x,y = points[:,axes[0]],points[:,axes[1]]
    area = dotarea
    cs = cat
    plt.scatter( x,y, s=area, c=cs, alpha=1.0, cmap=cmap)
    return fig

def category3d(points,cat,dotarea=10,axes=(0,1,2)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x,y,z = points[:,axes[0]],points[:,axes[1]],points[:,axes[2]]
    cs  = cat
    marker = 'o'
    ax.scatter(x,y,z,c = cs,marker=marker,s=dotarea)
    return fig
