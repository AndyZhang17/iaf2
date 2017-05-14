__author__ = 'andy17'


def hist2d(x,y,bins=50):
    heatmap, xedges, yedges = np.histogram2d(x,y,bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.figure()
    plt.clf()
    plt.imshow(heatmap.T,extent=extent,origin='lower')
    plt.show()

