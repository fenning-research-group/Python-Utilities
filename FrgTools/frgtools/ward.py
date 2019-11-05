import numpy as np
from scipy.signal import savgol_filter
from math import ceil
import matplotlib.pyplot as plt
from scipy import ndimage as nd

def FullSpectrumFit(wavelengths, reflectance, plot = False):
	eva_peak = 1730
	eva_tolerance = 5
	h2o_peak = 1902
	h20_tolerance = 5

	if np.mean(reflectance) > 1:
		reflectance = reflectance / 100

	absSpectrum = -np.log(reflectance)
	absPeaks, absBaseline = _RemoveBaseline(absSpectrum)

	eva_idx = np.argmin(np.abs(wavelengths - eva_peak))
	eva_abs = np.max(absPeaks[eva_idx-5 : eva_idx+5])
	eva_idx_used = np.where(absPeaks == eva_abs)[0][0]

	h2o_idx = np.argmin(np.abs(wavelengths - h2o_peak))
	h2o_abs = np.max(absPeaks[h2o_idx-5 : h2o_idx+5])
	h2o_idx_used = np.where(absPeaks == h2o_abs)[0][0]

	h2o_ratio = h2o_abs/eva_abs
	h2o_meas = (h2o_ratio - 0.002153)/.03491 #from mini module calibration curve 2019-04-09, no data with condensation risk

	if plot:
		fig, ax = plt.subplots(1,2, figsize = (8,3))

		ax[0].plot(wavelengths, absSpectrum, label = 'Raw')
		ax[0].set_xlabel('Wavelengths (nm)')
		ax[0].set_ylabel('Absorbance (AU)')

		ax[1].plot(wavelengths, absPeaks, label = 'Corrected')
		ax[1].plot(np.ones((2,)) * wavelengths[eva_idx_used], [0, eva_abs], label = 'EVA Peak', linestyle = '--')
		ax[1].plot(np.ones((2,)) * wavelengths[h2o_idx_used], [0, h2o_abs], label = 'Water Peak', linestyle = '--')
		ax[1].legend()
		ax[1].set_xlabel('Wavelengths (nm)')
		ax[1].set_ylabel('Baseline-Removed Absorbance (AU)')

		plt.tight_layout()
		plt.show()

	return h2o_meas

def ThreePointFit(wavelengths, reflectance, celltype, plot = False):
	if str.lower(celltype) in ['albsf2', 'al-bsf2']:
		wl_eva = 1730
		wl_h2o = 1902
		wl_ref = 1872

		p1 = 24.75
		p2 = 0.3461
		celltype = 'albsf2'
	elif str.lower(celltype) in ['albsf', 'al-bsf']:
		wl_eva = 1730
		wl_h2o = 1902
		wl_ref = 1942

		p1 = 34.53
		p2 = 1.545
		celltype = 'albsf'
	elif str.lower(celltype) in ['perc']:
		wl_eva = 1730
		wl_h2o = 1902
		wl_ref = 1942

		p1 = 29.75
		p2 = 1.367
		celltype = 'perc'
	else:
		print('Celltype Error: valid types are "albsf" or "perc"')
		return

	if np.mean(reflectance) > 1:
		reflectance = reflectance / 100
	ab = -np.log(reflectance)	#convert reflectance values to absorbance

	allWavelengthsPresent = True
	missingWavelength = None
	for each in [wl_eva, wl_h2o, wl_ref]:
		if each not in wavelengths:
			allWavelengthsPresent = False
			missingWavelength = each
			break

	if not allWavelengthsPresent:
		print('Wavelength Error: Necessary wavelength {0} missing from dataset - cannot fit.'.format(missingWavelength))
		return

	evaIdx = np.where(wavelengths == wl_eva)[0][0]
	h2oIdx = np.where(wavelengths == wl_h2o)[0][0]
	refIdx = np.where(wavelengths == wl_ref)[0][0]
	
	ratio = np.divide(ab[h2oIdx]-ab[refIdx], ab[evaIdx]-ab[refIdx])
	h2o_meas = ratio*p1 + p2
	# h2o[h2o < 0] = 0	


	## Avg Reflectance Fitting
	# avgRef = np.mean(ref, axis = 2)
	return h2o_meas

def ImputeWater(data, invalid = None):
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

	ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
	return data[tuple(ind)]	

def RegisterToDummy(start, start_ref = None):
	#scale dummy mask with busbar oriented upper horizontal.

	dummy = np.zeros(start.shape)
	border = [np.round(x*2/53) for x in start.shape]
	busbar = np.round(start.shape[0]*23/53)
	dummy[border:-border, border:-border] = 0.05
	dummy[busbar:busbar+border,:] = 1

	if start_ref is None:
		start_ref = start
		
	start_gauss = gaussian(start_ref, sigma = 0.5)
	result = ird.similarity(
		dummy,
		start_gauss,
		numiter = 30,
		constraints = {
			'angle': [0, 360],
			'tx': [0, 2],
			'ty': [0, 2],
			'scale': [1, 0.02]
		}
	)
	
	start_reg = ird.transform_img(
		start,
		tvec = result['tvec'].round(4),
		angle = result['angle'],
		scale = result['scale']
	)
	
	return start_reg

def _RemoveBaseline(spectrum, sensitivity = 5):
	def _PeakStripping(spectrum, window):
		spectrum_smoothed = savgol_filter(spectrum, window, 0)
		baseline = []

		for s, ss in zip(spectrum, spectrum_smoothed):
			baseline.append(min([s, ss]))

		return np.array(baseline)

	if sensitivity < 3:
		sensitivity = 3

	lp = ceil(0.5 * len(spectrum))
	#pad spectrum at either end
	spectrum_0 = np.hstack([
			np.full((lp,), spectrum[0]),
			spectrum,
			np.full((lp,), spectrum[-1])
		])
	l2 = len(spectrum_0)

	n = 1
	nmax = len(spectrum)*0.9
	foundMin = False
	S = spectrum_0
	A = []
	baselines = []
	while not foundMin:
		n = n + 2
		i = (n-1)/2
		baseline = _PeakStripping(S, n)
		A.append(np.trapz(S - baseline))
		S = baseline
		baselines.append(baseline)

		if i > sensitivity:
			if (A[-2] < A[-3]) and (A[-2] < A[-1]):
				foundMin = True

		if n > nmax:
			foundMin = True

	minIdx = np.argmin(A[sensitivity + 1:]) + sensitivity
	baseline = baselines[minIdx][lp:-lp]
	spectrum_corrected = spectrum - baseline

	return spectrum_corrected, baseline

