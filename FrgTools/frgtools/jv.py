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

def FitLight(v, i, area, diodes = 2, plot = False, init_guess = {}, bounds = {}, maxfev = 5000):
	"""
	Takes inputs of voltage (V), measured current (A), and cell area (cm2)

	Fits an illuminated JV curve to find at least the basic JV parameters:
	Open-circuit voltage: Voc (V)
	Short-circuit current: Jsc (A/cm2)
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

	j = [i_/area for i_ in i]
	
	if max(j) > 0.05:
		print('Current seems too high (max = {0} mA/cm2). Please double check that your area (cm2) and measured current (A) are correct.'.format(max(j)))

	v = np.asarray(v)
	j = np.asarray(j)
	p = np.multiply(v, j)
	x = np.vstack((v,j))
	
	init_guess_ = [1e-12, 1e-12, 2, 1e3, 35e-3] #jo1, jo2, rs, rsh, jl
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

	if diodes == 2:
		best_vals, covar = curve_fit(_Light2Diode, x, x[1,:], p0 = init_guess_, maxfev = maxfev, bounds = bounds)
		
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
		best_vals, covar = curve_fit(light1diode, x, x[1,:], p0 = init_guess_, maxfev=5000, bounds = bounds)
		
		results = {
			'jo1': best_vals[0],
			'rs': best_vals[1],
			'rsh': best_vals[2],
			'jl': best_vals[3],
			'covar': covar
		}
		results['jfit'] = _Light2Diode(x, results['jo1'], results['rs'], results['rsh'], results['jl'])
	else:
		print('Error: Invalid number of diodes requested for fitting. Diode must equal 1 or 2. User provided {0}. Diode equation not fit.'.format(diodes))
		results = {}

	if plot and len(results) > 0:
		fig, ax = plt.subplots()
		ax.plot(v, j*1000, label = 'Measured')
		ax.plot(v, results['jfit']*1000, linestyle = '--', label = 'Fit')
		ax.set_xlabel('Voltage (V)')
		ax.set_ylabel('Current Density (mA/cm2)')
		plt.show()
	
	results['voc'] = v[np.argmin(np.abs(j))]
	results['jsc'] = j[np.argmin(np.abs(v))]
	results['vmpp'] = v[np.argmax(p)]
	

	return results