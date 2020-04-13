import os
import csv
import numpy as np

def load_pyris(fpath):
	"""
	Loads data exported by Lambda Pyris FTIR tool in UCSD Materials Research Center basement.
	"""
	def readPyrisCSV(fpath):
		wl = []
		signal = []
		with open(fpath, 'r') as d:
			d_reader = csv.reader(d, delimiter = ',')
			garbage = d_reader.__next__()
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

	readMe = '.csv'
	rawFids = os.listdir(fpath)

	data = {}

	for f in rawFids:
		if readMe in f:
			path = os.path.join(fpath, f)
			try:	
				name = os.path.basename(f).split(readMe)[0]
				wl, signal, signalType = readPyrisCSV(path)
				data[name] = {
					'wavenumber': wl,
					'signal':	signal,
					'type':	signalType,
					'filepath': path
				}
			except:
				print('Error processing file {0}'.format(path))
			
	return data