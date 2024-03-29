import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
import affine6p
from skimage.transform import resize as skim_resize
import cv2

def adjust_brightness(img, value):
	'''
	adjust image brightness on a 0-255 scale

	example:
		img_brighter = adjust_brightness(img, 25)
		img_darker = adjust_brightness(img, -20)
		img_completelyblownout = adjust_brightness(img, 255)
	'''
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)

	v += value
	v[v > 255] = 255
	v[v < 0] = 0

	final_hsv = cv2.merge((h, s, v))
	img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
	return img

## Affine transformation scripts

def affine_transform(img, T, resample = Image.NEAREST, plot = False, adjustcenterofrotation = False, **kwargs):
	"""
	Performs an affine transformation on an array image, returns the transformed array

	img: Takes an input image in numpy array format
	T: 3x3 Affine transformation matrix in numpy array format
	resample: PIL resampling method
	plot: if True, displays original + transformed images side by side
	adjustcenterofrotation: Certain transformation matrices assume rotation from 0,0, others assume center of image. If 
		transformation matrix assumes rotation axis in center of image, set adjustcenterofrotation = True
	"""
	img_ = Image.fromarray(img)


	if adjustcenterofrotation:
		t_translate = np.array([		#move (0,0) to center of image instead of corner
			[1, 0, -img_.size[0]/2],
			[0, 1, -img_.size[1]/2],
			[0,0,1]
		])

		t_translate_back = np.array([	#move (0,0) to back to corner
			[1, 0, img_.size[0]/2],
			[0, 1, img_.size[1]/2],
			[0,0,1]
		])
	else:
		t_translate = np.array([
			[1,0,0],
			[0,1,0],
			[0,0,1]
			])

		t_translate_back = np.array([
			[1,0,0],
			[0,1,0],
			[0,0,1]
			])

	T_composite = t_translate_back @ T @ t_translate
	T_inv = np.linalg.inv(T_composite)

	img_t = img_.transform(
		img_.size,
		Image.AFFINE,
		data = T_inv.flatten()[:6],
		resample = resample,
	)
	img_t = np.array(img_t)
	
	if plot:
		fig, ax = plt.subplots(1,2)
		ax[0].imshow(img, **kwargs)
		ax[0].set_title('Original')
		ax[1].imshow(img_t, **kwargs)
		ax[1].set_title('Transformed')
		plt.show()

	return img_t

def affine_calculate(p, p0):
	"""
	Takes two m x 2 lists or numpy arrays of points, calculated the affine transformation matrix to move p -> p0
	"""
	p1 = np.array(p)
	p2 = np.array(p0)

	T = affine6p.estimate(p, p0).get_matrix()

	return T


## image imputing

def impute_nearest(data, invalid = None):
	"""
	Replace the value of invalid 'data' cells (indicated by 'invalid') 
	by the value of the nearest valid data cell

	Input:
		data:    numpy array of any dimension
		invalid: a binary array of same shape as 'data'. True cells set where data
				 value should be replaced.
				 If None (default), use: invalid  = np.isnan(data)

	Output: 
		Return a filled array. 
	"""
	#import numpy as np
	#import scipy.ndimage as nd

	if invalid is None: invalid = np.isnan(data)

	ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True) #return indices of nearest coordinates to each invalid point
	return data[tuple(ind)]	


## pick points on image

class __ImgPicker():
	def __init__(self, img, pts, markersize = 0.3, **kwargs):
		self.numPoints = pts
		self.currentPoint = 0
		self.repeating = False
		self.finished = False
		self.markersize = markersize

		self.fig, self.ax = plt.subplots()
		self.ax.imshow(img, picker = True, **kwargs)
		self.fig.canvas.mpl_connect('pick_event', self.onpick)

		self.buttonAx = plt.axes([0.4, 0, 0.1, 0.075])
		self.stopButton = Button(self.buttonAx, 'Done')
		self.stopButton.on_clicked(self.setFinished)

		self.pickedPoints = []
		self.pointArtists = []
		self.pointText = []

		plt.show(block = True)        
	
	def setFinished(self, event):
		self.finished = True
		if self.numPoints != np.inf:
			while len(self.pickedPoints) < self.numPoints:
				self.pickedPoints.append([np.nan, np.nan])
		plt.close(self.fig)
	
	def onpick(self, event):
		if not self.finished:
			mevt = event.mouseevent
			idx = self.currentPoint
			if not self.repeating: #if this is our first pass through the points, add a slot in the list for the point
				self.pickedPoints.append(None)
				self.pointArtists.append(None)
				self.pointText.append(None)

			x = mevt.xdata
			y = mevt.ydata
			self.pickedPoints[idx] = [x,y]

			if self.pointArtists[idx] is not None:
				self.pointArtists[idx].remove()
			self.pointArtists[idx] = plt.Circle((x,y), self.markersize, color = [1,1,1])
			self.ax.add_patch(self.pointArtists[idx])

			if self.pointText[idx] is not None:
				self.pointText[idx].set_position((x,y))
			else:
				self.pointText[idx] = self.ax.text(x,y, '{0}'.format(idx), color = [0,0,0], ha = 'center', va = 'center')
				self.ax.add_artist(self.pointText[idx])

			self.fig.canvas.draw()
			self.fig.canvas.flush_events()

			self.currentPoint += 1
			if self.currentPoint >= self.numPoints: 
				self.currentPoint = 0
				self.repeating = True


class AffineTransformer:
	'''
	Object to aid in manual image registration using affine transformations (rotation, translation, and rescaling). 

	Usage:
		- initialize the object by inputting a reference image and the number of registration points. kwargs can be passed to 
				the plt.plot() command to improve image plotting to aid in placement of registration points
			- a new reference image can be used with the .set_reference() command
		- fit the transform between a new image and the reference image by .fit(img = new_image)
		- the affine transformation matrix moving any image such that the new image would match the reference image 
			can be applied by .apply(img = moving_image)

	NOTE: this object relies of interactive matplotlib widgets. Jupyter lab plot backends will not play nicely with this tool.
			Jupyter notebook, however, will work with the "%matplotlib notebook" magic command enabled.
	'''
	def __init__(self, img, pts, **kwargs):
		self.set_reference(img, pts, **kwargs)

	def set_reference(self, img, pts, **kwargs):
		self.num_pts = pts
		self.reference_pts = pick_points(img, pts)
		self.reference_shape = img.shape

	def fit(self, img):
		if img.shape != self.reference_shape:
			print('Warning: moving image and reference image have different dimensions - look out for funny business')
			img_t = self._resize(img, order = 0)
			self.resize_default = True
		else:
			img_t = img 
			self.resize_default = False

		self.moving_pts = pick_points(img_t, pts = self.num_pts)

	def apply(self, img, resample = Image.NEAREST, plot = False, fill = np.nan, adjustcenterofrotation = False, resize = None, order = 0, **kwargs):
		# Note: the affine_calculate() call would ideally be in .fit(), but this is a silly workaround that
		# 		makes the helper play nice with Jupyter notebook. Issue is that the plot is nonblocking in notebook,
		#		so affine_calculate() gets called before the user has a chance to select points on the moving image.

		if resize is None:
			resize = self.resize_default

		self.T = affine_calculate(self.moving_pts, self.reference_pts)

		if resize:
			img_t = self._resize(img, order = order, cval = fill) 
		else:
			img_t = img
		img_t = affine_transform(img_t, self.T, resample = resample, plot = plot, adjustcenterofrotation = adjustcenterofrotation, **kwargs)

		return img_t[:self.reference_shape[0], :self.reference_shape[1]]

	def _resize(self, img, order = 0, **kwargs):
		xratio = img.shape[1] / self.reference_shape[1]
		yratio = img.shape[0] / self.reference_shape[0]

		target_shape = np.round(img.shape / np.min([xratio, yratio])).astype(int)
		img_t = skim_resize(img, target_shape, order = order, **kwargs)
		return img_t


def pick_points(img, pts = 4, **kwargs):
	"""
	Given an image and a number of points, allows the user to interactively select points on the image.
	These points are returned when the "Done" button is pressed. Useful to generate inputs for AffineCalculate.
	If <pts are clicked, remaining points will be filled with [np.nan, np.nan]
	
	Note - for infinite points, you can set pts = np.inf, and only the number of clicked points will be returned
	"""
	imgpicker = __ImgPicker(img, pts, **kwargs)
	return imgpicker.pickedPoints