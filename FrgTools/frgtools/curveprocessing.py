import numpy as np
from scipy.signal import savgol_filter
from math import ceil

def RemoveBaseline(spectrum, sensitivity = 5):
	"""
	Given a series of values and a somewhat arbitrary sensitivity value, approximates
	a baseline by iterative savitsky-golay smoothing with increasing window size. The baseline
	is not allowed to rise above the values at any point.

	Returns the spectrum with baseline removed, as well as the baseline itself.
	"""
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

