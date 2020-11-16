import os
import csv
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt


def load_lambda(fpath):
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

def tauc(wl, a, bandgap_type, fit_width = None, fit_threshold = 0.1, plot = False):
	wl = np.array(wl)
	a = np.array(a)

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
		  
	taucvalue = (a*h*nu) ** (1/n)
	taucvalue_threshold = taucvalue.max() * fit_threshold
	best_slope = None
	best_intercept = None
	best_r2 = 0

	for idx in range(fit_pad, len(wl) - fit_pad):
		if taucvalue[idx] >= taucvalue_threshold:
			fit_window = slice(idx-fit_pad, idx+fit_pad)
			slope, intercept, rval, _, _ = linregress(ev[fit_window], taucvalue[fit_window])
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

	return Eg