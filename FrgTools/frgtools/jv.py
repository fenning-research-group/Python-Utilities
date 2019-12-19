import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def LoadTracer(fpath):
	def skipLines(f, numLines):
		for i in range(numLines):
			f.readline()
			
	def parseLine(line, leadingTabs, trailingTabs):
		contents = []
		counter = 0
		totalPer = leadingTabs + trailingTabs
		lineparts = line.split('\t')
		for part in lineparts:
			if counter == leadingTabs:
				if part != '\n':
					contents.append(part)
			counter = counter + 1
			if counter > totalPer:
				counter = 0

		return contents
	
	data = {}
	with open(fpath, 'r') as f:
		data['ID'] = parseLine(f.readline(), 1, 1)
		data['Device'] = parseLine(f.readline(), 1 ,1)
		data['Curve'] = parseLine(f.readline(), 1 ,1)
		for idx, each in enumerate(data['Curve']):
			if 'Ill' in each:
				data['Curve'][idx] = 'Illuminated'
			else:
				data['Curve'][idx] = 'Dark'
		data['Area'] = [float(x) for x in parseLine(f.readline(), 1 ,1)]
		skipLines(f, 2)
		data['Date'] = parseLine(f.readline(), 1 ,1)        
		data['Time'] = parseLine(f.readline(), 1 ,1)
		skipLines(f, 4)
		
		data['V'] = [[] for x in range(len(data['Curve']))]
		data['I'] = [[] for x in range(len(data['Curve']))]
		data['P'] = [[] for x in range(len(data['Curve']))]
		
		for line in f:
			raw = parseLine(line, 0, 0)
			for i in range(len(data['Curve'])):
				try:
					data['V'][i].append(float(raw[i*3 + 0]))
					data['I'][i].append(float(raw[i*3 + 1]))
					data['P'][i].append(float(raw[i*3 + 2]))
				except:
					pass
	
	return data

def LoadFRG(fpath):

	def readFRGFile(fpath):
		data = {}

		with open(fpath, 'r') as f:
			ReadingHeader = True 	#check when header has been fully parsed
			BothDirections = False	#sweep forward + reverse

			while ReadingHeader:	
				line = f.readline()
				lineparts = line.split(':')

				if 'Data' in lineparts[0]:
					ReadingHeader = False
					f.readline()
					f.readline()
				else:
					try:
						data[lineparts[0]] = float(lineparts[1])
					except:
						data[lineparts[0]] = lineparts[1][1:].replace('\n', '')


			vforward = []
			iforward = []
			timeforward = []			
			if data['sweepDir'] == 'Forward + Reverse':
				BothDirections = True
				vreverse = []
				ireverse = []
				timereverse = []

			for line in f:
				lineparts = f.readline().split('\t')
				if len(lineparts) == 1:
					break
				vforward.append(lineparts[0])	
				iforward.append(lineparts[1])
				timeforward.append(lineparts[2])
				if BothDirections:
					vreverse.append(lineparts[0])	
					ireverse.append(lineparts[1])
					timereverse.append(lineparts[2])

			data['V'] = np.array(vforward).astype(float)
			data['I'] = np.array(iforward).astype(float)
			data['J'] = data['I']/data['area_cm2']
			data['delay'] = np.array(timeforward).astype(float)

			if BothDirections:
				data2 = data.copy()
				data2['sampleName'] = data['sampleName'] + '_Reverse'
				data['sampleName'] = data['sampleName'] + '_Forward'
				data2['V'] = np.array(vreverse).astype(float)
				data2['I'] = np.array(ireverse).astype(float)
				data2['J'] = data2['I']/data2['area_cm2']
				data2['delay'] = np.array(timereverse).astype(float)
				output = [data, data2]
			else:
				output = data

		return output



	fids = [os.path.join(fpath, x) for x in os.listdir(fpath)]
	
	alldata = {}
	for f in fids:
		output = readFRGFile(f)
		if type(output) == list:
			for each in output:
				alldata[each['sampleName']] = each
		else:
			alldata[output['sampleName']] = output
	return alldata

def FitDark(v, i, area, plot = False, init_guess = {}, bounds = {}, maxfev = 5000):
	"""
	Takes inputs of voltage (V), measured current (A), and cell area (cm2)

	Fitting by 2-diode model provides parameters:
	Diode saturation currents: Jo1, (Jo2 if 2-diode model) (A/cm2)
	Series resistance: Rs (ohms cm2)
	Shunt resistance: Rsh (ohms)
	"""

	def _Dark2Diode(x, jo1, jo2, rs, rsh):
		v = x[0]
		j_meas = x[1]
		
		#constants
		q = 1.60217662e-19 #coulombs
		k = 1.380649e-23 #J/K
		T = 298.15 #assume room temperature

		#calculation
		d1 = jo1 * np.exp((q*(v-(j_meas*rs)))/(k*T))
		d2 = jo2 * np.exp((q*(v-(j_meas*rs)))/(2*k*T))
		j = d1 + d2 + (v-(j_meas*rs))/rsh
		
		return j

	j = [i[n]/area for n in range(len(v))]
	
	v = np.asarray(v)
	j = np.asarray(j)
	x = np.vstack((v,j))
	   
	init_guess_ = [1e-12, 1e-12, 1, 5e2] #jo1, jo2, rs, rsh, jl
	for key, val in init_guess.items():
		if key == 'jo1':
			init_guess_[0] = val
		elif key == 'jo2':
			init_guess_[1] = val
		elif key == 'rs':
			init_guess_[2] = val
		elif key == 'rsh':
			init_guess_[3] = val

	bounds_=[[0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]]

	for key, vals in bounds.items():
		if key == 'jo1':
			bounds_[0][0] = vals[0]
			bounds_[1][0] = vals[1]
		elif key == 'jo2':
			bounds_[0][1] = vals[0]
			bounds_[1][1] = vals[1]
		elif key == 'rs':
			bounds_[0][2] = vals[0]
			bounds_[1][2] = vals[1]
		elif key == 'rsh':
			bounds_[0][3] = vals[0]
			bounds_[1][3] = vals[1]

	best_vals, covar = curve_fit(_Dark2Diode, x, x[1,:], p0 = init_guess_, bounds = bounds_, maxfev = maxfev)
	
	results = {
		'jo1': best_vals[0],
		'jo2': best_vals[1],
		'rs': best_vals[2],
		'rsh': best_vals[3],
		'covar': covar,
		'jfit': _Dark2Diode(x, best_vals[0], best_vals[1], best_vals[2], best_vals[3])
	}

	
	if plot:
		fig, ax = plt.subplots()
		ax.plot(v, j*1000, label = 'Measured')
		ax.plot(v, results['jfit']*1000, linestyle = '--', label = 'Fit')
		ax.set_xlabel('Voltage (V)')
		ax.set_ylabel('Current Density (mA/cm2)')
		plt.show()
	
	return results

def FitLight(v, i, area, diodes = 2, plot = False, init_guess = {}, bounds = {}, maxfev = 5000, type = None):
	"""
	Takes inputs of voltage (V), measured current (A), and cell area (cm2)

	Fits an illuminated JV curve to find at least the basic JV parameters:
	Open-circuit voltage: Voc (V)
	Short-circuit current: Jsc (mA/cm2)
	Max power point voltage: Vmpp (V)

	Fitting by 2-diode (default) or 1-diode model as specified by diodes argument provides additional parameters:
	Diode saturation currents: Jo1, (Jo2 if 2-diode model) (A/cm2)
	Series resistance: Rs (ohms cm2)
	Shunt resistance: Rsh (ohms)
	Photogenerated current: Jl (A/cm2)
	"""

	def _Light2Diode(x, jo1, jo2, rs, rsh, jl):
		v = x[0]
		j_meas = x[1]
		
		#constants
		q = 1.60217662e-19 #coulombs
		k = 1.380649e-23 #J/K
		T = 298.15 #assume room temperature

		#calculation
		d1 = jo1 * np.exp((q*(v+(j_meas*rs)))/(k*T))
		d2 = jo2 * np.exp((q*(v+(j_meas*rs)))/(2*k*T))
		j = jl - d1 - d2 - (v+j_meas*rs)/rsh
		
		return j

	def _Light1Diode(x, jo1, rs, rsh, jl):
		v = x[0]
		j_meas = x[1]
		
		#constants
		q = 1.60217662e-19 #coulombs
		k = 1.380649e-23 #J/K
		T = 298.15 #assume room temperature

		#calculation
		d1 = jo1 * np.exp((q*(v+(j_meas*rs)))/(k*T))
		j = jl - d1 - (v+j_meas*rs)/rsh
		
		return j

	j = [i_/area for i_ in i]	#convert A to mA
	
	if max(j) > .05:
		print('Current seems too high (max = {0} mA/cm2). Please double check that your area (cm2) and measured current (A) are correct.'.format(max(j*1000)))

	v = np.asarray(v)
	j = np.asarray(j)
	p = np.multiply(v, j)
	x = np.vstack((v,j))
	
	jsc = j[np.argmin(np.abs(v))]

	if diodes == 2:
		init_guess_ = [1e-12, 1e-12, 2, 1e3, jsc] #jo1, jo2, rs, rsh, jl
		for key, val in init_guess.items():
			for idx, choices in enumerate([['jo1', 'j01'], ['jo2', 'j02'], ['rs', 'rseries'], ['rsh', 'rshunt'], ['jl', 'jill', 'jilluminated']]):
				if str.lower(key) in choices:
					init_guess_[idx] = val
					break

		bounds_=[[0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf]]
		for key, vals in bounds.items():
			for idx, choices in enumerate([['jo1', 'j01'], ['jo2', 'j02'], ['rs', 'rseries'], ['rsh', 'rshunt'], ['jl', 'jill', 'jilluminated']]):
				if str.lower(key) in choices:
					bounds_[0][idx] = vals[0]
					bounds_[1][idx] = vals[1]
					break

		print(init_guess_)
		print(bounds_)
		best_vals, covar = curve_fit(_Light2Diode, x, x[1,:], p0 = init_guess_, maxfev = maxfev, bounds = bounds_)
		
		results = {
			'jo1': best_vals[0],
			'jo2': best_vals[1],
			'rs': best_vals[2],
			'rsh': best_vals[3],
			'jl': best_vals[4],
			'covar': covar
		}
		results['jfit'] = _Light2Diode(x, results['jo1'], results['jo2'], results['rs'], results['rsh'], results['jl'])
	
	elif diodes == 1:
		init_guess_ = [1e-12, 2, 1e3, jsc] #jo1, rs, rsh, jl
		for key, val in init_guess.items():
			for idx, choices in enumerate([['jo1', 'j01'], ['rs', 'rseries'], ['rsh', 'rshunt'], ['jl', 'jill', 'jilluminated']]):
				if str.lower(key) in choices:
					init_guess_[idx] = val
					break
					
		bounds_=[[0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]]
		for key, vals in bounds.items():
			for idx, choices in enumerate([['jo1', 'j01'], ['rs', 'rseries'], ['rsh', 'rshunt'], ['jl', 'jill', 'jilluminated']]):
				if str.lower(key) in choices:
					bounds_[0][idx] = vals[0]
					bounds_[1][idx] = vals[1]
					break
	
		print(init_guess_)
		print(bounds_)
		best_vals, covar = curve_fit(_Light1Diode, x, x[1,:], p0 = init_guess_, maxfev = maxfev, bounds = bounds_)
		
		results = {
			'jo1': best_vals[0],
			'rs': best_vals[1],
			'rsh': best_vals[2],
			'jl': best_vals[3],
			'covar': covar
		}
		results['jfit'] = _Light1Diode(x, results['jo1'], results['rs'], results['rsh'], results['jl'])
	else:
		print('Error: Invalid number of diodes requested for fitting. Diode must equal 1 or 2. User provided {0}. Diode equation not fit.'.format(diodes))
		results = {}



	if plot and len(results) > 0:
		fig, ax = plt.subplots()
		ax.plot(v, j*1000, label = 'Measured')
		xlim0 = ax.get_xlim()
		ylim0 = ax.get_ylim()
		ax.plot(v, results['jfit']*1000, linestyle = '--', label = 'Fit')
		ax.set_xlim(xlim0)
		ax.set_ylim(ylim0)
		ax.set_xlabel('Voltage (V)')
		ax.set_ylabel('Current Density (mA/cm2)')
		plt.show()
	
	results['voc'] = v[np.argmin(np.abs(j))]
	results['jsc'] = j[np.argmin(np.abs(v))]
	results['vmpp'] = v[np.argmax(p)]
	results['pce'] = p.max()/100
	results['ff'] = p.max() / (results['voc']*results['jsc'])	

	return results