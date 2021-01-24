import os
import numpy as np
from tqdm import tqdm
from renishawWiRE import WDFReader
from .curveprocessing import gaussian
from PIL import Image
from scipy.optimize import curve_fit

def shifttonm(shift, incident_wl):
    '''
    converts a raman shift (cm-1) to wavelength (nm)

    inputs
        shift: raman shift, cm-1
        incident_wl: excitation source wavelength, nm

    output
        emitted wavelength, nm
    '''
    incident_wn = 1/(incident_wl * 1e-7)
    exit_wn = incident_wn - shift
    return 1e7/exit_wn 

def nmtoshift(wl, incident_wl):
    '''
    converts an emitted wavelength (nm) to raman shift (cm-1)

    inputs
        wl: emitted wavelength, nm
        incident_wl: excitation source wavelength, nm

    output
        shift: raman shift, cm-1
    '''
    incident_wn = 1/(incident_wl * 1e-7)
    exit_wn = 1/(wl * 1e-7)
    return incident_wn - exit_wn


### Renishaw data processing

def _fit_gauss(x,y):
    #remove points with 0 counts - this often happens if the counts are too high for the detector,
    # renishaw defaults these to 0 (big peak turns into two tails with 0s in between). will ruin 
    # gaussian fit otherwise.
    zeromask = y == 0
    x = x[~zeromask]
    y = y[~zeromask]


    p0 = [y.max(), x[np.argmax(y)], 20]
    bounds = [[0, 500, 10], [1e6, 700, 100]]
    try:
        (A,mu,sigma), _ = curve_fit(gaussian, x, y, p0 = p0, bounds = bounds, maxfev = 1500)
    except:
        A,mu,sigma = np.nan,np.nan,np.nan #error fitting, return nan for gaussian params
    
    fwhm = sigma*2.35482
    return A,mu,fwhm

def _fit_max(x,y):
    maxidx = np.argmax(y)
    A = y[maxidx]
    mu = x[maxidx]
    
    idx = np.where(y > A/2)[0] #indices >= half max
    fwhm_min_idx, fwhm_max_idx = idx[0], idx[-1]
    fwhm_min = x[fwhm_min_idx]
    fwhm_max = x[fwhm_max_idx]
    fwhm = np.abs(fwhm_max - fwhm_min)    

    return A,mu,fwhm

fitmethods_key = {
    'gaussian': _fit_gauss,
    'max': _fit_max
} 


def load_renishaw(fid, method = 'gaussian', threshold = 0, longpass_cutoff = 0):
    '''
    processes a PL map from the renishaw microraman tool
    
    fid: path to .wdf file 
    threshold: minimum counts below which map pixels are considered background, set to nan
    method: how to fit peak counts + wavelength ['gaussian', 'max']
    longpass_cutoff: wavelength below which laser longpass is blocking counts. prevents gaussian fitting from
                        considering wavelengths blocked by filter.
    '''

    if method not in fitmethods_key:
        raise ValueError(f'Invalid fitting method - must be one of {list(fitmethods_key.items())}')
    fitmethod = fitmethods_key[method]
    
    
    d = WDFReader(fid)
    d.img = np.asarray(Image.open(d.img))
    if d.xlist_unit.name == 'Nanometre':
        d.wl = d.xdata
        d.shift = nmtoshift(d.wl, d.laser_length)
    else:
        d.wl = shifttonm(d.xdata, d.laser_length)
        d.shift = d.xdata

    longpass_mask = d.wl <= longpass_cutoff
    
    d.maxcts = np.zeros(d.spectra.shape[:-1])
    d.fwhm = np.zeros(d.spectra.shape[:-1])
    d.peakwl = np.zeros(d.spectra.shape[:-1])
    
    
    for m,n in tqdm(np.ndindex(d.maxcts.shape), total = np.product(d.maxcts.shape), desc = 'Fitting Spectra', leave = False):
        d.maxcts[m,n], d.peakwl[m,n], d.fwhm[m,n] = fitmethod(d.wl[~longpass_mask], d.spectra[m,n, ~longpass_mask])
        
    mask = d.maxcts < threshold
    d.maxcts[mask] = np.nan
    d.fwhm[mask] = np.nan
    d.peakwl[mask] = np.nan
    
    d.extent = [d.xpos[0], d.xpos[-1], d.ypos[0], d.ypos[-1]] #map dimensions

    return d

def load_renishaw_textfiles(path, incident_wl, method = 'gaussian', threshold = 0, longpass_cutoff = 0, nm = True):
    '''
    Loads raw spectra files output when exporting renishaw area scans to xy data.
    Assumes input data is in raman shift (cm-1)

    inputs:
        path: path to folder holding all raw xy files
        incident_wl: excitation source wavelength, nm
        method: method to fit peaks of individual spectra. must be 'gaussian' or 'max'
        threshold: points with peak counts below this value will be set to nan. useful to remove background.
        longpass_cutoff: wavelength below which laser longpass is blocking counts. prevents gaussian fitting from
                    considering wavelengths blocked by filter.
        nm: True if data is exported in nanometers, False if in raman shifts
    
    output:
        data object with consolidated area scan data

    '''
    class RenishawData:
        def __init__(self, r):
            self.shift = r['shift']
            self.wl = r['wavelengths']
            self.laser_length = r['incidentWavelength']
            self.spectra = r['spectra']
            self.maxcts = r['intensity']
            self.fwhm = r['fwhm']
            self.peakwl = r['peakWavelength']
            self.extent = r['extent']
            self.xpos = r['x'],
            self.ypos = r['y']


    if method not in fitmethods_key:
        raise ValueError(f'Invalid fitting method - must be one of {list(fitmethods_key.items())}')
    fitmethod = fitmethods_key[method]

    # parse filenames to get x/y coordinates
    fids = [os.path.join(path, x) for x in os.listdir(path)]
    allx = []
    ally = []
    for f in fids:
        allx.append(float(f.split('X')[-1].split('_')[1]))
        ally.append(float(f.split('Y')[-1].split('_')[1]))
    allx = np.array(allx)
    ally = np.array(ally)
    x = np.unique(allx)
    y = np.unique(ally)
    x.sort()
    y.sort()
    
    # get wavelengths/shifts
    temp = np.loadtxt(fids[0])
    if nm:
        wl = temp[:,0]
        shift = nmtoshift(wl, incident_wl = incident_wl)
    else:
        shift = temp[:,0]
        wl = shifttonm(shift, incident_wl = incident_wl)
    longpass_mask = wl <= longpass_cutoff

    #read data
    spectra = np.zeros((len(y), len(x), len(wl)))
    maxcts = np.zeros(spectra.shape[:-1])
    fwhm = np.zeros(spectra.shape[:-1])
    peakwl = np.zeros(spectra.shape[:-1])

    for f, x_, y_ in tqdm(zip(fids, allx, ally), total = len(fids), desc = 'Loading Renishaw Scans', leave = False):
        m = np.where(y == y_)
        n = np.where(x == x_)
        newdata = np.loadtxt(f)[:,1] #directly loading data into spectra[m,n] causes errors, idk why. whatever.
        spectra[m,n] = newdata       
        maxcts[m,n], peakwl[m,n], fwhm[m,n] = fitmethod(wl, newdata)
    
    #remove data points with max counts below threshold
    mask = maxcts < threshold
    maxcts[mask] = np.nan
    fwhm[mask] = np.nan
    peakwl[mask] = np.nan      
    
    extent = [x[0], x[-1], y[0], y[-1]]

    results = {
        'shift': temp[:,0],
        'incidentWavelength': incident_wl,
        'wavelengths': wl,
        'spectra': spectra,
        'intensity': maxcts,
        'peakWavelength': peakwl,
        'fwhm':fwhm,
        'x': x,
        'y': y,
        'extent': extent
    }

    return RenishawData(results)