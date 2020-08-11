import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib_scalebar.scalebar import ScaleBar as mplsb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
# from matplotlib import lines

def hline(y, ax = None, **kwargs):

	lineArguments = {
		'color': 'k',
		'linestyle': ':'
	}

	for k,v in kwargs.items():
		lineArguments[k] = v

	if ax is None:
		ax = plt.gca()

	xlim0 = ax.get_xlim()
	ax.plot(xlim0, [y, y], **lineArguments)
	ax.set_xlim(xlim0)

def vline(x, ax = None, **kwargs):

	lineArguments = {
		'color': 'k',
		'linestyle': ':'
	}

	for k,v in kwargs.items():
		lineArguments[k] = v

	if ax is None:
		ax = plt.gca()

	ylim0 = ax.get_ylim()
	ax.plot([x, x], ylim0, **lineArguments)
	ax.set_ylim(ylim0)

def scalebar(ax = None, scale = 1, **kwargs):
	"""
	Lightweight wrapper around matplotlib_scalebar.scalebar.ScaleBar
	Default positioning and text color (white text on low opacity black background in lower right corner)

	Two typically used parameters
		ax: defaults to current matplotlib axes
		scale: defaults to 1. Should hold the SI unit scale (ie for units in microns, scale = 1e-6, for units in km, scale = 1e3, etc.)
	All other parameters for ScaleBar can be passed as keyword arguments

	"""
	scalebarArguments = {
		'location': 'lower right',
		'box_color': [0,0,0],
		'box_alpha': 0.2,
		'color': [1,1,1],
		'pad': 0.05
	}

	for k, v in kwargs.items():
		scalebarArguments[k] = v

	if ax is None:
		ax = plt.gca()
	sb = mplsb(
			dx = scale,
			**scalebarArguments
		)
	ax.add_artist(sb)
	return sb
	
def colorbar(im, ax = None, orientation = 'vertical', **kwargs):
	if ax is None:
		ax = plt.gca()


	colorbarArguments = {
		'position': 'right',
		'size': '5%',
		'pad': 0.05
	}
	for k, v in kwargs.items():
		colorbarArguments[k] = v

	divider = make_axes_locatable(ax)
	cax = divider.append_axes(**colorbarArguments)
	cb = plt.colorbar(im, cax = cax, orientation = orientation)

	return cb

### Plot Builders

def waterfall(data, ax = None, lognorm = False, ticks = {}, tickoffset = 0, tickwidth = 1, **kwargs):
	if ax == None:
		fig, ax = plt.subplots(figsize = (8, 4))


	defaultArgs = {
		'cmap': plt.cm.inferno
	}

	for k, v in defaultArgs.items():
		if k not in kwargs.keys():
			kwargs[k] = v

	if lognorm:
		data[data <= 0] = np.nan
		kwargs['norm'] = LogNorm(1, np.nanmax(data))


	im = ax.imshow(
		data, 
		**kwargs
		) 

	ax.set_aspect('auto')
	
	cb = plt.colorbar(im, ax = ax, fraction = 0.03)
	
	if lognorm:
		addonstr = ' (log)'
	else:
		addonstr = ''

	cb.set_label('Counts{}'.format(addonstr))
	

	xleft, xright = ax.get_xlim()
	ybot, ytop = ax.get_ylim()
	ticksize = (ytop - ybot)/20

	for idx, t in enumerate(ticks.items()):
		label = t[0]
		position = t[1]

		c = plt.cm.tab10(idx)
		ax.text(1.0, 0.6 - idx*0.05, label, color = c, transform = ax.figure.transFigure)        
		for p in position:
			if p <= xright and p >= xleft:
				ax.plot([p, p], [ytop + (tickoffset*idx)*ticksize, ytop + (tickoffset*(idx) + 0.8) * ticksize], color = c, linewidth = tickwidth, clip_on = False)

	ax.set_clip_on(False)
	ax.set_ylim((0, ytop))
	# plt.show()

def categorical_heatmap(x, y, z, ax = None, xlabel = '', ylabel = '', zlabel = '', title = '', fillvalue = np.nan, multiplevaluehandling = 'mean'):
	"""
	Takes three 1-d inputs (x, y, z) and constructs an x by y heatmap with values z.
	"""
	def _CrossoutHeatmap(points, ax=None, scale=1, **kwargs):
		"""
		Helper function to draw x's on unfilled points in the heatmap
		"""

		ax = ax or plt.gca()
		l = np.array([[[1,1],[-1,-1]]])*scale/2.
		r = np.array([[[-1,1],[1,-1]]])*scale/2.
		p = np.atleast_3d(points).transpose(0,2,1)
		c = LineCollection(np.concatenate((l+p,r+p), axis=0), **kwargs)
		ax.add_collection(c)
		return c

	if ax == None:
		fig, ax = plt.subplots()

	if multiplevaluehandling == 'mean':
		MultipleValueFunction = np.mean
	elif multiplevaluehandling == 'min':
		MultipleValueFunction = np.min
	elif multiplevaluehandling == 'max':
		MultipleValueFunction = np.max
	elif multiplevaluehandling == 'std':
		MultipleValueFunction = np.std
	else:
		print('Invalid value "{0}" passed to multiplevaluehandling: Only mean, min, max, and std are valid.'.format(multiplevaluehandling))
		return


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
	zlist = [[[] for m in range(uX.shape[0])] for n in range(uY.shape[0])]

	for i in range(z.shape[0]):
		m = np.where(uY == y[i])[0]
		n = np.where(uX == x[i])[0]
	#     print('({0}, {1})'.format(m,n))
		if m.shape[0] > 0 and n.shape[0] > 0:
			try:
				zlist[m[0]][n[0]].append(z[i])
			except:
				print('--')
				print(z[i])
				print(m[0])
				print(n[0])

	for m, mlist in enumerate(zlist):
		for n, vals in enumerate(mlist):
			if len(vals) > 0:
				zmat[m,n] = MultipleValueFunction(vals)

	# zmat[m,n] = z[i]

	znan = np.isnan(zmat)

	im = ax.imshow(
		zmat,
		origin = 'lower',
		cmap = plt.cm.inferno,
		vmin = np.nanmin(zmat),
		vmax = np.nanmax(zmat)
		)


	ax.set_xticks(uXt)
	ax.set_xticklabels(uX)
	ax.set_xlabel(xlabel)
	ax.set_yticks(uYt)
	ax.set_yticklabels(uY)
	ax.set_ylabel(ylabel)
	cb = plt.colorbar(im, ax = ax, fraction = 0.046)
	cb.set_label(zlabel)
	ax.set_title(title)

	# _CrossoutHeatmap(znan, ax = ax, scale = 0.8, color = "black")

	return ax, cb