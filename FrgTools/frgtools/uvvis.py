import os
import csv
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt


def load_lambda(fpath):
	'''
	Loads an entire folder of output files from Lambda 1050 
	specrophotometer in the basement of SME (NE-MRC)

	input:
		fpath: filepath to directory in which uvvis data is stored

	output:
		dictionary of uvvis data. keys are the sample names, values are:
			wavelength: wavelengths scanned (nm)
			signal: values measured, units depend on tool setting
			type: signal type. again, depends on tool setting
			filepath: path to file
	'''
	def readLambdaCSV(fpath):
		wl = []
		signal = []
		with open(fpath, 'r') as d:
			d_reader = csv.reader(d, delimiter = ',')
			header = d_reader.__next__()	#get header line

			if '%R' in header[1]:
				signalType = 'Reflectance'
			elif '%T' in header[1]:
				signalType = 'Transmittance'
			elif 'A' in header[1]:
				signalType = 'Absorbance'
			else:
				signalType = 'Unknown'

			for row in d_reader:
				wl.append(float(row[0]))
				signal.append(float(row[1]))

		return np.array(wl), np.array(signal), signalType

	readMe = '.Sample.'
	ignoreMe = '.sp'
	rawFids = os.listdir(fpath)

	data = {}

	for f in rawFids:
		if readMe in f:
			if f.endswith(ignoreMe):
				continue
			path = os.path.join(fpath, f)
			name = os.path.basename(f).split(readMe)[0]
			if '.Cycle' in f:
				cycle = os.path.basename(f).split('.Cycle')[-1].split('.Raw')[0]
				name += '_{}'.format(cycle)
			wl, signal, signalType = readLambdaCSV(path)

			data[name] = {
				'wavelength': wl,
				'signal':	signal,
				'type':	signalType,
				'filepath': path
			}
			
	return data

def calc_absorbance(r, t, t_substrate = None):
	'''
	given reflectance and transmission measurements, approximates absorbance.
	Note that this is the most basic approach, different samples may 
	require different treatment to obtain accurate absorbance values.

	If a substrate transmission is measured, that can be removed to better
	isolate the absorbance of the sample itself
	'''
	r = np.asarray(r)
	t = np.asarray(t)
	if t_substrate is None:
		t_substrate = np.ones(r.shape)
	else:
		t_substrate = np.asarray(t_substrate)

	if r.max() > 1:
		r /= 100 #convert percents to fractional values
	if t.max() > 1:
		t /= 100
	if t_substrate.max() > 1:
		t_substrate /= 100


	t /= t_substrate #ignore transmission losses in the substrate
	A = -np.log(t/(1-r)) #absorbance = -log(I/I0). 
	
	return A

def beers(a, pathlength, concentration = 1):
	'''
	Uses Beer-Lambert's law to convert absorbance values to
	absorption coefficients (alpha)

	A = alpha * path_length * concentration

	inputs:
		a: absorbance
		pathlength: optical path length (cm). often this will be film thickness, cuvette width, etc.
		concentration: relevant for solutions. defaults to one, aka no effect. (g/cm3)
	
	returns:
		alpha: absorption coefficient (cm^-1)
	'''

	return a/(pathlenght*concentration)

def kubelka_munk(r):
	'''
	Kubelka-Munk transform to analyze diffuse reflectance data.

	Note that this approach assumes that the data is capturing DIFFUSE reflectance,
	ie from measurement of a powder sample or scattering particulates embedded in a
	non-absorbing medium. 

	Also note that the calculated value is the ratio alpha/S, where alpha = absorption
	coefficient and S = the scattering coefficient, which varies with particle size and
	packing. This value is proportional to alpha, and is often used in its place, but be
	aware of the difference - if S is significantly large, or worse variable across
	the wavelengths measured, analysis using this value in place of alpha will be impacted.
	
	See Paper 3 shared by A. El-Denglawey at the following link:
	  https://www.researchgate.net/post/Which-form-of-Kubelka-Munk-function-should-I-used-to-calculate-band-gap-of-powder-sample
	'''
	r = np.asarray(r)
	return (1-r)**2 / (2*r)

def tauc(wl, a, thickness, bandgap_type, wlmin = None, wlmax = None, fit_width = None, fit_threshold = 0.1, plot = False, verbose = False):
	'''
	Performs Tauc plotting analysis to determine optical bandgap from absorbance data
	Plots data in tauc units, then performs linear fits in a moving window to find the
	best linear region. this best fit line is extrapolated to the x-axis, which corresponds
	to the bandgap.

	inputs

		wl: array of wavelengths (nm)
		a: absorbance values (A). if you already have the absorption coefficient, set thickness = 1
		thickness: sample thickness (cm)
		bandgap_type: ['direct', 'indirect']. determines coefficient on tauc value

		wlmin: minimum wavelength (nm) to include in plot
		wlmax: maximum wavelenght (nm) to include in plot
		fit_width: width of linear fit window, in units of wl, a vector indices
		fit_threshold: window values must be above this fraction of maximum tauc value.
						prevents fitting region before the absorption onset
		plot: boolean flag to generate plot of fit
		verbose: boolean flag to (True) generate detailed output or (False) just output Eg

	output (verbose = False)
		bandgap: optical band gap (eV)
	
	output (verbose = True)
		dictionary with values:
			bandgap: optical band gap (eV)
			r2: r-squared value of linear fit
			bandgap_min: minimum bandgap within 95% confidence interval
			bandgap_max: maximum bandgap within 95% confidence interval
	'''
	wl = np.array(wl)
	if wlmin is None:
		wlmin = wl.min()
	if wlmax is None:
		wlmax = wl.max()
	wlmask = np.where((wl >= wlmin) & (wl <= wlmax))

	wl = wl[wlmask]
	a = np.array(a)[wlmask]
	alpha = a/thickness #A = alpha*t, calculate absorption coefficient

	if fit_width is None:
		fit_width = len(wl)//20  #default to 5% of data width

	fit_pad = fit_width//2

	if str.lower(bandgap_type) == 'direct':
		n = 0.5
	elif str.lower(bandgap_type) == 'indirect':
		n = 2
	else:
		raise ValueError('argument "bandgap_type" must be provided as either "direct" or "indirect"')


	c = 3e8                    	#speed of light, m/s
	h = 4.13567e-15            	#planck's constant, eV
	nu = c/(wl*1e-9)  			#convert nm to hz
	ev = 1240/wl     			#convert nm to ev

	taucvalue = (alpha*h*nu) ** (1/n)
	taucvalue_threshold = taucvalue.max() * fit_threshold
	best_slope = None
	best_intercept = None
	best_r2 = 0

	for idx in range(fit_pad, len(wl) - fit_pad):
		if taucvalue[idx] >= taucvalue_threshold:
			fit_window = slice(idx-fit_pad, idx+fit_pad)
			slope, intercept, rval, _, stderr = linregress(ev[fit_window], taucvalue[fit_window])
			r2 = rval**2
			if r2 > best_r2 and slope > 0:
				best_r2 = r2
				best_slope = slope
				best_intercept = intercept

	Eg = -best_intercept / best_slope #x intercept
	
	if plot:
		fig, ax = plt.subplots()
		ax.plot(ev, taucvalue, 'k')
		ylim0 = ax.get_ylim()
		ax.plot(ev, ev*best_slope + best_intercept, color = plt.cm.tab10(3), linestyle = ':')
		ax.set_ylim(*ylim0)
		ax.set_xlabel('Photon Energy (eV)')
		if n == 0.5:
			ax.set_ylabel(r'$({\alpha}h{\nu})^2$');   
		else:
			ax.set_ylabel(r'$({\alpha}h{\nu})^{1/2}$');
		plt.show()

	if not verbose:
		return Eg
	else:
		### calculate 95% CI of Eg
		mx = ev.mean()
		sx2 = ((ev-mx)**2).sum()
		sd_intercept = stderr * np.sqrt(1./len(ev) + mx*mx/sx2)
		sd_slope = stderr * np.sqrt(1./sx2)

		Eg_min = -(best_intercept - 1.96*sd_intercept) / (best_slope + 1.96*sd_slope)
		Eg_max = -(best_intercept + 1.96*sd_intercept) / (best_slope - 1.96*sd_slope)
		
		output = {
			'bandgap': Eg,
			'r2': best_r2,
			'bandgap_min': Eg_min,
			'bandgap_max': Eg_max
		}
		return output
	