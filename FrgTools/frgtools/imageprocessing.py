import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image

## Affine transformation scripts

def AffineTransform(img, T, resample = Image.NEAREST, plot = False):
	"""
	Performs an affine transformation on an array image, returns the transformed array

	img: Takes an input image in numpy array format
	T: 3x3 Affine transformation matrix in numpy array format
	resample: PIL resampling method
	"""
	img_ = Image.fromarray(img)

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

	T_composite = t_translate_back @ T @ t_translate
	T_inv = np.linalg.inv(T_composite)

	img_t = img_.transform(
	    img_.size,
	    Image.AFFINE,
	    data = T_inv.flatten()[:6],
	    resample = resample
	)

	if plot:
		fig, ax = plt.subplots(1,2)
		ax[0].imshow(img)
		ax[0].set_title('Original')
		ax[1].imshow(img_t)
		ax[1].set_title('Transformed')
		plt.show()

	return img_t

def AffineCalculate(p1, p2):
	"""
	Takes two m x 2 lists or numpy arrays of points, calculated the affine transformation matrix to move p1 -> p2
	"""
	p1 = np.array(p1)
	p2 = np.array(p2)

	# if p1.shape[0] < p1.shape[1]:
	# 	p1 = np.transpose(p1)
	# if p2.shape[0] < p2.shape[1]:
	# 	p2 = np.transpose(p2)

	p1 = np.hstack((p1, np.ones((p1.shape[0], 1))))
	p2 = np.hstack((p2, np.ones((p2.shape[0], 1))))

	T, _, _, _ = np.linalg.lstsq(p1, p2)
	T[2, :2] = [0,0]

	return T

class __ImgPicker():
		def __init__(self, img, pts, markersize = 0.3):
			self.numPoints = pts
			self.currentPoint = 0
			self.finished = False
			self.markersize = markersize

			self.fig, self.ax = plt.subplots()
			self.ax.imshow(img, picker = True)
			self.fig.canvas.mpl_connect('pick_event', self.onpick)

			self.buttonAx = plt.axes([0.4, 0, 0.1, 0.075])
			self.stopButton = Button(self.buttonAx, 'Done')
			self.stopButton.on_clicked(self.setFinished)

			self.pickedPoints = [None for x in range(self.numPoints)]
			self.pointArtists = [None for x in range(self.numPoints)]
			self.pointText = [None for x in range(self.numPoints)]

			plt.show(block = True)        
		
		def setFinished(self, event):
			self.finished = True
			plt.close(self.fig)
		
		def onpick(self, event):
			if not self.finished:
				mevt = event.mouseevent
				idx = self.currentPoint % self.numPoints
				self.currentPoint += 1

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

def ImagePointPicker(img, pts = 4):
	imgpicker = _ImgPicker(img, pts)
	return imgpicker.pickedPoints