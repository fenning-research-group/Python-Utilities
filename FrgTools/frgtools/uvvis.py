import os
import csv
import numpy as np

def LoadLambda(fpath):
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
	rawFids = os.listdir(fpath)

	data = {}

	for f in rawFids:
		if readMe in f:
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