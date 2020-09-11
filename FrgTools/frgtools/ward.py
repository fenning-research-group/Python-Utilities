import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
import imreg_dft as ird
from .curveprocessing import remove_baseline

def fit_fullspectrum(wavelengths, reflectance, plot = False):
	eva_peak = 1730
	eva_tolerance = 5
	h2o_peak = 1902
	h20_tolerance = 5

	if np.mean(reflectance) > 1:
		reflectance = reflectance / 100

	absSpectrum = -np.log(reflectance)
	absPeaks, absBaseline = RemoveBaseline(absSpectrum)

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
		ax[0].plot(wavelengths, absBaseline, label = 'Baseline')
		ax[0].legend()
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

def fit_threepoint(wavelengths, reflectance, celltype, plot = False):
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

def expectedwater(temperature, relhum, material = 'EVA9100'):
	"""
	Given a temperature (C) and relative humidity (%), calculates the saturation water content in solar encapsulants
	based on literature data or experiments carried out at Arizona State University as part of the PVRD2 project.
	"""
	material = str.lower(material)
	if 'eva' in material:
		if 'hum' in material:
			"""
			ASU EVA water saturation fit on 2019-02-27, using damp heat data only
			Not very confident in this fit
			"""
			a = 0.7949
			b = -1968		
		elif 'kempe' in material:
			"""
			Kempe, M. D. (2005). Control of moisture ingress into photovoltaic modules. 
			31st IEEE Photovoltaic Specialists Conference, (February), 503–506. 
			https://doi.org/10.1109/PVSC.2005.1488180
			"""
			a = 2.677
			b = -2141		
		else:
			"""
			ASU EVA water saturation fit on 2019-01-29, using submerged samples
			"""
			a = 3.612
			b = -2414
	elif 'poe' in material:
		a = 0.04898
		b = -1415
	else:
		raise Exception('{} not a valid material.'.format(material))

	tempArrh = 1/(273+temperature) #arrhenius temperature, 1/K
	waterConcentration = a * np.exp(b * tempArrh) #water content, g/cm3
	waterConcentration *= relhum/100 #scale by relative humidity, assuming Henrian behavior

	return waterConcentration

def expecteddiffusivity(temperature, material = 'EVA9100'):
	"""
	Given a temperature, calculates the diffusivity of water in EVA9100
	"""

	diffusivity = 0.001353*np.exp(-0.191*(1/(273.15 + temperature))/(8.617e-5))  #ASU uptake fit 2018-11-27     
	return diffusivity