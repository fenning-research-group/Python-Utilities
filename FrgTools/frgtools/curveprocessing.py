import numpy as np
from scipy.signal import savgol_filter
from math import ceil
from functools import partial

def remove_baseline(spectrum, sensitivity = 5):
	"""
	Given a series of values and a somewhat arbitrary sensitivity value, approximates
	a baseline by iterative savitsky-golay smoothing with increasing window size. The baseline
	is not allowed to rise above the values at any point.

	Returns the spectrum with baseline removed, as well as the baseline itself.
	"""
	def _PeakStripping(spectrum, window):
		spectrum_smoothed = savgol_filter(spectrum, window, 1)
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

	minIdx = np.argmin(A[sensitivity:]) + sensitivity
	baseline = baselines[minIdx][lp:-lp]
	spectrum_corrected = spectrum - baseline

	return spectrum_corrected, baseline


### general functional forms for curve fitting

def lorentzian(x, amplitude, center, width):
	return (amplitude*width**2) / ((x-center)**2 + width**2)	

def __n_lorentzian_generator(n, x, *args):
	'''
	sum of n lorentzian curves. arguments should be passed in order amplitude_1, center_1, width_1, amplitude_2...etc
	'''
	if len(args) != n*3:
		print('Error: must be three arguments (amplitude, center, and width) per lorentzian!')
		return

	y = 0
	for idx in range(n):
		amplitude = args[3*idx]
		center = args[3*idx + 1]
		width = args[3*idx + 2]

		y += lorentzian(x,amplitude,center,width)

	return y

def multiple_lorentzian(n):
	'''
	sum of n gaussian curves. arguments should be passed in order amplitude_1, center_1, width_1, amplitude_2...etc
	'''
	return partial(__n_lorentzian_generator, n)


def gaussian(x, amplitude, center, sigma):
	return amplitude * np.exp(-(x-center)**2 / (2*sigma**2))

def __n_gaussian_generator(n, x, *args):	
	'''
	sum of n gaussian curves. arguments should be passed in order amplitude_1, center_1, width_1, amplitude_2...etc
	'''
	if len(args) != n*3:
		print('Error: must be three arguments (amplitude, center, and standard deviation) per gaussian!')
		return

	y = 0
	for idx in range(n):
		amplitude = args[3*idx]
		center = args[3*idx + 1]
		sigma = args[3*idx + 2]

		y += gaussian(x,amplitude,center,sigma)

	return y

def multiple_gaussian(n):
	'''
	sum of n gaussian curves. arguments should be passed in order amplitude_1, center_1, width_1, amplitude_2...etc
	'''
	return partial(__n_gaussian_generator, n)



def voigt(x, gamplitude, gcenter, gsigma, lamplitude,lcenter,lwidth):
	return (gamplitude*(1/(gsigma*(np.sqrt(2*np.pi))))*(np.exp(-((x-gcenter)**2)/((2*gsigma)**2)))) + ( lamplitude*lwidth**2/((x-lcenter)**2 + lwidth**2) )

def __n_voigt_generator(n, x, *args):
	'''
	generates function as sum of n voigt curves. arguments should be passed in order amplitude_gauss_1, center_gauss_1, width_gauss_1,amplitude_lorentz_1, center_lorentz_1, width_lorentz_1, amplitude_gauss_2...etc
	'''

	if len(args) != n*6:
		print('Error: must be six arguments (gaussian amplitude, center, and standard deviation, lorentzian amplitude,center,andwidth) per voigt!')
		return

	y = 0
	for idx in range(n):
		gamplitude = args[3*idx]
		gcenter = args[3*idx + 1]
		gsigma = args[3*idx + 2]
		lamplitude = args[3*idx + 3]
		lcenter = args[3*idx + 4]
		lwidth = args[3*idx + 5]


		y += voigt(x, gamplitude, gcenter, gsigma, lamplitude, lcenter, lwidth)

	return y


def multiple_voigt(n):
	'''
	sum of n voigt curves. arguments should be passed in order amplitude_gauss_1, center_gauss_1, width_gauss_1,amplitude_lorentz_1, center_lorentz_1, width_lorentz_1, amplitude_gauss_2...etc
	'''
	return partial(__n_voigt_generator, n)
