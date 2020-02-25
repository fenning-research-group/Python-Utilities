import os
import numpy as np
from tqdm import tqdm

def ShiftToNm(shift, incident_wl):
    incident_wn = 1/(incident_wl * 1e-7)
    exit_wn = incident_wn - shift
    return 1e7/exit_wn 

def LoadRenishaw(path, incident_wl):
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
    wl = shift_to_nm(shift, incident_wl = incident_wl)
    
    cts = np.zeros((len(y), len(x), len(wl)))
    for f, x_, y_ in tqdm(zip(fids, allx, ally), total = len(fids)):
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