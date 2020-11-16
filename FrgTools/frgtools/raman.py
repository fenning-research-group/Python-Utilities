import os
import numpy as np
from tqdm import tqdm

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

def load_renishaw(path, incident_wl):
    '''
    Loads raw spectra files output when exporting renishaw area scans to xy data.
    Assumes input data is in raman shift (cm-1)

    inputs:
        path: path to folder holding all raw xy files
        incident_wl: excitation source wavelength, nm

    output:
        dictionary with consolidated area scan data

    '''
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
    
    temp = np.loadtxt(fids[0])
    shift = temp[:,0]
    wl = shifttonm(shift, incident_wl = incident_wl)
    
    cts = np.zeros((len(y), len(x), len(wl)))
    for f, x_, y_ in tqdm(zip(fids, allx, ally), total = len(fids), desc = 'Loading Renishaw Scans', leave = False):
        m = np.where(y == y_)
        n = np.where(x == x_)
        cts[m,n] = np.loadtxt(f)[:,1]    
    
    extent = [x[0], x[-1], y[0], y[-1]]

    results = {
        'shift': temp[:,0],
        'incidentWavelength': incident_wl,
        'wavelengths': wl,
        'counts': cts,
        'intensity': cts.max(axis = 2),
        'peakWavelength': wl[np.argmax(cts, axis = 2)],
        'x': x,
        'y': y,
        'extent': extent
    }

    return results