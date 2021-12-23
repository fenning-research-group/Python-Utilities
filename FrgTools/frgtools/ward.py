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
	absPeaks, absBaseline = remove_baseline(absSpectrum)

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



#current version of bifacial_fitfull is still very specific, requiring the polymer type (EVA or POE)
#and sampleName is used when plot = True to distinguish which plots are being shown
#note, used to require export_folder, but was removed when export = False
#bifacial fitfull uses rubberband method to remove baseline, then fits gaussian to the baseline removed spectra
#returns the baseline corrected absorbance, the baseline, and gaussian fitted curves

from scipy.spatial import ConvexHull
from lmfit.models import GaussianModel
from scipy import stats
import pandas as pd
from .curveprocessing import rubberband

def bifacial_fitfull(wavelengths, reflectance, polymer_type, sampleName, plot = False):
    
    ####step 1, remove baseline using rubber band method, added to frgtools.curveprocessing
    wl = np.array(wavelengths)
    raw_absorbance = np.log(1/np.array(reflectance))
    baseline = rubberband(wl,raw_absorbance)

    corrected_abs = raw_absorbance - baseline
    
    ##currently uses these fixed peaks for the gaussian fitting
    ##may change that later to automatically find the local peaks of the curve (especially for POE)
    
    peak1 = wl[15] #1730
    peak2 = wl[31] #1762
    peak3 = wl[51] #1802
    peak4 = wl[102] #1904
    peak5 = wl[124] #1948
    
    if polymer_type == 'EVA':
        peaks_list = [peak1,peak2,peak3,peak4,peak5]
    elif polymer_type == 'POE':
        peaks_list = [peak1,peak2,peak3,peak4]
    
    if plot == True:
        fig, ax = plt.subplots(1,2, figsize = (8,3))

        ax[0].plot(wl, raw_absorbance, label = 'Raw')
        ax[0].plot(wl, baseline, label = 'Baseline')
        ax[0].legend()
        ax[0].set_xlabel('Wavelengths (nm)')
        ax[0].set_ylabel('Absorbance (AU)')

        ax[1].plot(wl, corrected_abs, label = 'Corrected')
        ax[1].legend()
#         ax[1].set_ylim(top = 0.35)
        ax[1].set_xlabel('Wavelengths (nm)')
        ax[1].set_ylabel('Baseline-Removed Absorbance (AU)')
        fig.suptitle(sampleName, y = 1.05)
        
        for i in peaks_list:
            ypeak = corrected_abs[np.where(wl == i)][0]
            ax[1].plot(np.ones((2,))*i,[0,ypeak], linestyle = '--', color = 'g')

        plt.tight_layout()
        plt.show()
        
    ####step 2, use corrected absorbance to fit gaussian model (jack's code)
    
    ##############################################################################
    ################################ USER INPUTS #################################
    ##############################################################################
    
    ##dealing with peaks_list
    
    peaks_list = list(map(int, peaks_list))
    
    first_guess = []

    for i in peaks_list:
        first_guess.append([1,i,10])
    
    first_guess.append([1,1700,10])

    # Define your intial parameters here.  
    initial_vals = {
#                     'export_folder' : export_folder,
                    'start' : 1, # Which run do you want to start the fit? 1 is the first. 
                                   # This parameter matters when fitting a select number of spectra
                    'step' : 1, # At what interval should spectra be selected? 
                                # 1 is recommended because of peak shifting
                    'amount' : 'all', # How many total spectra do you want to fit? 
                    # 'all' or 'All' will fit all starting at the first spectrum, 
                    # while entering an integer will fit that many spectra 
                    # starting at the specified spectrum 
                    'lower_bound' : 1700, # Lowest wavelength of fitting domain
                    'upper_bound' : 2000, # Highest wavelength of fitting domain
                    'Fixed_Peaks' : [1700], # Peaks not allowed to move during the fit
                    'Mobile_Peaks' : peaks_list,# Peaks allowed to move during the fit
                    'tolerance' : 4.0, # Amount allowed to deviate from peaks defined above 
                    'min_amplitude' : 0.01, # Define minimum amplitude. Get this value from SingleSpecFit
                    'max_amplitude' : 10, # Define maximum amplitude. Get this value from SingleSpecFit
                    'min_sigma' : 1, # Define minimum sigma. Get this value from SingleSpecFit
                    'max_sigma' : 50, # Define maximum sigma. Get this value from SingleSpecFit
                    'first_vals' : pd.DataFrame(np.array(first_guess
                                                        ),
                                               columns=['amplitude', 'center', 'sigma'])
                    # 'first_vals' is the model's first guess at parameters 
                   }
    
    # Add 'x_peaks' to initial_vals: all peaks added to one list. Sort it.
    initial_vals['x_peaks'] = initial_vals['Mobile_Peaks']+initial_vals['Fixed_Peaks']
    initial_vals['x_peaks'].sort()
    # change the index of the df to the initial centers
    initial_vals['first_vals']['centers'] = initial_vals['first_vals']['center']
    initial_vals['first_vals'] = initial_vals['first_vals'].set_index('centers')
    # store the name of the run for plotting purposes
    run_name = sampleName
    
    
    def initial_fit(initial_vals):
    
        """
        "get_fit_parameters" is where the magic happens. The lmfit Gaussian
        model is used to fit the data. The first guess at parameters comes from 
        first_vals and y_fit is used as the data to be fit. This function outputs
        the best fit parameters as "best_vals" and the component names.

        """
        x_fit = wl

        y_fit = corrected_abs


        def get_fit_parameters(x_fit, y_fit, x_peaks, first_vals):

            # Initiate the model by adding the first fixed component
            # Define the model parameters using first_vals
            sigma = first_vals.loc[initial_vals['Fixed_Peaks'][0],'sigma']
            center=first_vals.loc[initial_vals['Fixed_Peaks'][0],'center']
            A = first_vals.loc[initial_vals['Fixed_Peaks'][0],'amplitude']
            # Initiate the dict to store the model components
            components = {}
            # Initiate a list to store the component names
            component_names = []
            # Name the component
            prefix = 'Component' + '_' + str(initial_vals['Fixed_Peaks'][0])
            # Call the GaussianModel
            peak = GaussianModel(prefix=prefix)
            # Set the initial parameter guesses
            pars = peak.make_params(center=center, sigma=sigma, amplitude=A)
            # This peak will not move
            pars[prefix+'center'].set(vary = False) 
            # All amplitudes must be positive
            pars[prefix+'amplitude'].set(min=initial_vals['min_amplitude'], max = initial_vals['max_amplitude'])
            pars[prefix+'sigma'].set(min=initial_vals['min_sigma'],max=initial_vals['max_sigma'])
            # Add the component and its name to the respective dict and list
            components[prefix] = peak
            component_names.append(prefix)
            # Assign this peak to "mod". This variable will be appended iteratively
            # to create the overall model
            mod = components[component_names[0]]

            # If there is more than one fixed peak, the following for loop will exectute
            if len(initial_vals['Fixed_Peaks']) > 1:
                # This for loop is identical to the process for defining and adding
                # fixed components outlined above. It is now iterative.
                for i in np.arange(1 , len(initial_vals['Fixed_Peaks'])):

                    sigma = first_vals.loc[initial_vals['Fixed_Peaks'][i],'sigma']
                    center=first_vals.loc[initial_vals['Fixed_Peaks'][i],'center']
                    A = first_vals.loc[initial_vals['Fixed_Peaks'][i],'amplitude']
                    prefix = 'Component' + '_' + str(initial_vals['Fixed_Peaks'][i])

                    peak = GaussianModel(prefix=prefix)
                    pars.update(peak.make_params(center=center, sigma=sigma, amplitude=A))
                    pars[prefix+'center'].set(vary = False) 
                    pars[prefix+'amplitude'].set(min=initial_vals['min_amplitude'], max = initial_vals['max_amplitude'])
                    pars[prefix+'sigma'].set(min=initial_vals['min_sigma'],max=initial_vals['max_sigma'])
                    components[prefix] = peak
                    component_names.append(prefix)
                    mod += components[component_names[i]]
            # Add the mobile components to the model
            if len(initial_vals['Mobile_Peaks']) > 0:
                # This for loop is identical to the process for defining and adding
                # components outlined above. It is now iterative.
                for i in np.arange(0 , len(initial_vals['Mobile_Peaks'])):

                    sigma = first_vals.loc[initial_vals['Mobile_Peaks'][i],'sigma']
                    center=first_vals.loc[initial_vals['Mobile_Peaks'][i],'center']
                    A = first_vals.loc[initial_vals['Mobile_Peaks'][i],'amplitude']
                    prefix = 'Component' + '_' + str(initial_vals['Mobile_Peaks'][i])

                    peak = GaussianModel(prefix=prefix)
                    pars.update(peak.make_params(center=center, sigma=sigma, amplitude=A))
                    pars[prefix+'center'].set(min=center-initial_vals['tolerance'],
                                              max=center+initial_vals['tolerance']) 
                    pars[prefix+'amplitude'].set(min=initial_vals['min_amplitude'], max = initial_vals['max_amplitude'])
                    pars[prefix+'sigma'].set(min=initial_vals['min_sigma'],max=initial_vals['max_sigma'])
                    components[prefix] = peak
                    component_names.append(prefix)
                    mod += components[component_names[i+len(initial_vals['Fixed_Peaks'])]]

            # Exectute the fitting operation and store to "out"
            out = mod.fit(y_fit, pars, x=x_fit, method = 'lbfgsb')
            # Plot the fit using lmfit's built-in plotting function, includes fit
            # residuals
            # out.plot(fig=1)
            # Create an array of zeros to populate a dataframe
            d = np.zeros((len(x_peaks),3))
            # Create a dataframe to store the best fit parameter values
            best_vals = pd.DataFrame(d ,columns = ['amplitude',
                                                 'center', 
                                                 'sigma'])
            # Populate the dataframe with the best fit values
            for i in range(len(x_peaks)):
                best_vals.loc[i,'amplitude'] = out.best_values[component_names[i] + 'amplitude']
                best_vals.loc[i,'center'] = out.best_values[component_names[i] + 'center']
                best_vals.loc[i,'sigma'] = out.best_values[component_names[i] + 'sigma']
            # set index based on component wavelengths
            best_vals = best_vals.set_index(np.array(initial_vals['Fixed_Peaks']+initial_vals['Mobile_Peaks']))

            return best_vals, component_names

        """
        "plot_components" plots the following onto a single plot: each component,
        the best-fit line, and the raw data. It also stores each component's
        amplitude and area to separate dataframes.

        """
        def plot_components(x_fit, y_fit, best_vals, x_peaks, component_names):

            # GM is the equation representing the gaussian Model. Given a set 
            # of parameters and x-values, the y-vals are output as "data"
            def GM(amp, mu, sigma):
                data = []
                for x in x_fit:
                    y = ((amp)/(sigma*np.sqrt(2*np.pi)))*(np.e**((-(x-mu)**2)/(2*sigma**2)))
                    data.append(y)
                return data

            # generateY uses GM to output dataframes containing the wavelengths
            # and absorbances for each component as well as the sum of all
            # components (best-fit line) and stores them to a dictionary "curves"
            def generateY(x_fit, best_vals):
                # initiate the curves dict
                curves = {}
                # prepare data to initiate a dataframe
                d = {'Wavelength':x_fit,
                     'Abs':0}
                # within the dict "curves", initiate the best_fit df. Each
                # component's absorbance will be added to this df, forming the best
                # fit line.
                curves['Best_Fit'] = pd.DataFrame(d , 
                                                  index = range(len(x_fit)), 
                                                  columns = ['Wavelength', 'Abs'])
                  # Store the raw data in the curves data frame
                d = {'Wavelength' : x_fit,
                     'Abs' : y_fit}
                curves['Raw_Data'] = pd.DataFrame(d,
                                                  index = range(len(x_fit)),
                                                  columns = ['Wavelength','Abs'])
                inds = initial_vals['Fixed_Peaks']+initial_vals['Mobile_Peaks']
                # iteratively add each component to the dict "curves"
                for i in range(len(x_peaks)):
                    amp = best_vals.loc[inds[i],'amplitude']
                    mu = best_vals.loc[inds[i],'center']
                    sigma = best_vals.loc[inds[i],'sigma']  
                    # add the component to curves using GM and best-fit parameters
                    # to produce the absorbance values
                    curves[component_names[i]] = pd.DataFrame(list(zip(x_fit,GM(amp, mu, sigma))),
                                                             columns = ['Wavelength', 'Abs'])
                    # add the component to the best fit dataframe 
                    curves['Best_Fit']['Abs'] = curves['Best_Fit']['Abs'].add(curves[component_names[i]]['Abs'], fill_value = 0)
                return curves

            # Define a function to calculate MSE, RMSE and nRMSE (normalized by the 
            # interquartile range)
            def MSE_RMSE(y_fit, curves):

                y_true = list(y_fit)
                y_pred = list(curves['Best_Fit']['Abs'])
                MSE = np.square(np.subtract(y_true,y_pred)).mean()
                RMSE = np.sqrt(MSE)
                IQR = stats.iqr(y_true, interpolation = 'midpoint')
                nRMSE = RMSE/IQR

                return [['MSE', 'RMSE', 'nRMSE'],[MSE, RMSE, nRMSE]]

            # call generateY to produce the dict. "curves"
            curves = generateY(x_fit, best_vals)
            # Call MSE_RMSE to generate fit scores
            errors = MSE_RMSE(y_fit, curves)

            ######################plotting
            # initiate a figure to plot all the components onto
            
            if plot == True:
                plt.figure(figsize=(4.5,4)) 
                plt.figure(dpi = 200)
                plt.xlabel("Wavelength (nm)", fontsize=12)
                plt.ylabel("Absorbance (AU)", fontsize=12)
                # create a color scheme
                colors = plt.cm.jet(np.linspace(0,1,len(initial_vals['x_peaks'])))
                sorted_names = []
                for i in component_names:
                    sorted_names.append(i)
                sorted_names.sort()
                # iteratively add all components to the plot      
                for i in range(len(x_peaks)):
                    plt.plot(curves[sorted_names[i]].loc[:,'Wavelength'], 
                             curves[sorted_names[i]].loc[:,'Abs'], 
                             label = sorted_names[i],
                             color=colors[i])
                    # shade the area under the curve
                    plt.fill_between(curves[sorted_names[i]].loc[:,'Wavelength'], 
                                     0,
                                     curves[sorted_names[i]].loc[:,'Abs'],
                                     alpha=0.3,
                                     color=colors[i])        
                # add the raw data to the plot
                plt.plot(x_fit, y_fit, linewidth=2, label='Raw Data', color = 'hotpink', alpha = 1)
                # add the best fit to the plot
                plt.plot(curves['Best_Fit']['Wavelength'],curves['Best_Fit']['Abs'], '--', label='Best Fit', color = 'black', alpha=0.5)
                plt.xlim(initial_vals['lower_bound'],initial_vals['upper_bound'])
                plt.legend(fontsize=5)
                plt.title(run_name)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
            
                plt.show()
                
            best_vals = best_vals.reset_index()
            # create a dataframe and populate it with each component's amplitude
            amplitudes = pd.DataFrame(component_names ,columns = ['Components'])
            amplitudes[run_name] = best_vals['amplitude']
            # create a dataframe and populate it with each component's center
            centers = pd.DataFrame(component_names ,columns = ['Components'])
            centers[run_name] = best_vals['center']
            # create a dataframe and populate it with each component's sigma
            sigmas = pd.DataFrame(component_names ,columns = ['Components'])
            sigmas[run_name] = best_vals['sigma']
            # create a dataframe and populate it with each component's area
            areas = pd.DataFrame(component_names ,columns = ['Components'])
            # create a dataframe and populate it with each component's maximum
            maxima = pd.DataFrame(component_names ,columns = ['Components'])
            temp_areas = []
            temp_maxima = []
            for name in component_names:
                temp_areas.append(np.trapz(y = curves[name]['Abs'], 
                                     x = curves[name]['Wavelength']))
                temp_maxima.append(max(curves[name]['Abs']))
            areas[run_name] = temp_areas
            maxima[run_name] = temp_maxima

            return curves, amplitudes, centers, areas, sigmas, maxima, errors
        
        
        # call the functions defined above and store their outputs
        # fit the desired data and return the best fit parameter values
        best_vals, component_names = get_fit_parameters(x_fit, y_fit, initial_vals['x_peaks'], initial_vals['first_vals'])
        # plot the fitting result and return the curves, areas, and amplitudes
        curves, amplitudes, centers, areas, sigmas, maxima, errors = plot_components(x_fit, y_fit, best_vals, initial_vals['x_peaks'], component_names)

        return best_vals, curves, amplitudes, centers, areas, sigmas, maxima, run_name, errors
    
    """
    "data_export" takes a user defined directory, creates a new folder within that
    directory, and populates that folder with the amplitudes, areas, centers,
    insitu dataset, and curves for the fit.

    """
            
    # Initiate a dict to store curves and errors
    # Run the initial fit         
    curves = {} 
    errors = {}
    best_vals, temp_curves, amplitudes, centers, areas, sigmas, maxima, run_name, temp_errors = initial_fit(initial_vals)
    curves[run_name] = temp_curves
    errors[run_name] = temp_errors

#     # call the iterative fitting function
#     fit_all(best_vals, curves, amplitudes, areas, initial_vals, errors)

    # call the function. Unless the second arg is 'True', this function will not execute
    
    
    return corrected_abs, baseline, curves
