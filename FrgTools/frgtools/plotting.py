import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

def CrossoutHeatmap(points, ax=None, scale=1, **kwargs):
    ax = ax or plt.gca()
    l = np.array([[[1,1],[-1,-1]]])*scale/2.
    r = np.array([[[-1,1],[1,-1]]])*scale/2.
    p = np.atleast_3d(points).transpose(0,2,1)
    c = LineCollection(np.concatenate((l+p,r+p), axis=0), **kwargs)
    ax.add_collection(c)
    return c

def CategoricalHeatmap(x, y, z, ax = None, xlabel = '', ylabel = '', zlabel = '', title = '', fillvalue = np.nan):
	"""
	Takes three 1-d inputs (x, y, z) and constructs an x by y heatmap with values z.
	"""
	if ax == None:
		fig, ax = plt.subplots()

	x = np.array(x)
	y = np.array(y)
	z = np.array(z)

	uX = np.unique(x)
	uXt = []
	for i in range(uX.shape[0]):
	    uXt.append(i)

	uY = np.unique(y)
	uYt = []
	for i in range(uY.shape[0]):
	    uYt.append(i)
	    
	zmat = np.full((uY.shape[0], uX.shape[0]), fill_value = fillvalue)
	for i in range(z.shape[0]):
	    m = np.where(uY == y[i])[0]
	    n = np.where(uX == x[i])[0]
	#     print('({0}, {1})'.format(m,n))
	    if m.shape[0] > 0 and n.shape[0] > 0:
	        zmat[m,n] = z[i]


	znan = np.isnan(zmat)

	im = ax.imshow(
		zmat,
		origin = 'lower',
		cmap = plt.cm.inferno,
		vmin = z.min(),
		vmax = z.max()
		)

	CrossoutHeatmap(znan, ax = ax, scale = 0.8, color = "black")

	ax.set_xticks(uXt)
	ax.set_xticklabels(uX)
	ax.set_xlabel(xlabel)
	ax.set_yticks(uYt)
	ax.set_yticklabels(uY)
	ax.set_ylabel(ylabel)
	cb = plt.colorbar(im, ax = ax, fraction = 0.046)
	cb.set_label(zlabel)
	ax.set_title(title)

	return ax, cb