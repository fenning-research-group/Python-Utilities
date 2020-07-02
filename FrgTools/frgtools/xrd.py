import numpy as np
import os
from scipy.signal import find_peaks_cwt, savgol_filter
import re

def load_smartlab(fpath):
    """
    Function to load XRD data from Rigaku Smartlab .RAS files.

    heavy inspiration from https://github.com/Traecp/Rigaku-SmartLab - thanks!
    """
    RAS_HEADER_START = "*RAS_HEADER_START"
    RAS_HEADER_END   = "*RAS_HEADER_END"
    RAS_INT_START    = "*RAS_INT_START"
    RAS_INT_END      = "*RAS_INT_END"
    HEADER_SPLIT     = "\""
    DATA_SPLIT       = " "
    
    data = {}
    internal = {}
    data['name'] = os.path.basename(fpath)[:-4] #remove .ras
    data['header'] = dict()
    data['counts'] = []
    internal['MEAS_COND_AXIS_NAME'] = dict() #Array
    internal['MEAS_COND_AXIS_NAME_INTERNAL'] = dict() #Array
    internal['MEAS_COND_AXIS_OFFSET'] = dict() #Array
    internal['MEAS_COND_AXIS_POSITION'] = dict() #Array
    internal['MEAS_COND_AXIS_UNIT'] = dict() #Once
    
    data['scanaxis'] = ""
    internal['scan_axis_internal'] = ""
    data['angles'] = []
    data['numscans'] = 0
    internal['points_per_scan'] = 0

    with open(fpath, encoding="Latin-1", mode="r") as f:
        scan_start   = False
        scan_end     = False
        header_start = False
        scan_data    = []
        scan_angle   = []
        header_initialized = False
        scan_is_3d = False

        for line in f:
            if line.strip():
                line = line.strip()
                # print("Scan start: ", scan_start)
                if line.startswith(RAS_HEADER_START):
                    header_start = True
                    # print(line)
                    continue
                if line.startswith(RAS_HEADER_END):
                    header_start = False
                    header_initialized = True
                    # print(line)
                    continue
                if line.startswith(RAS_INT_START):
                    scan_start = True
                    continue
                if line.startswith(RAS_INT_END):
                    scan_start = False
                    pad_points = internal['points_per_scan'] - len(scan_data)
                    if pad_points > 0:
                        print("Data not complete. Number of data point missing for this scan: ", pad_points)
                        pad_data   = [0]*pad_points
                        scan_data.extend(pad_data)
                    data['angles'] = scan_angle
                    data['counts'].append(scan_data)
                    data['numscans'] +=1
                    scan_data = []
                    scan_angle= []
                    # continue
                    
                if scan_start:
                    ls = line.split(DATA_SPLIT)
                    # print(ls)
                elif header_start:
                    ls = line.split(HEADER_SPLIT)
                else:
                    continue
                    
                    
                if header_start:
                    key = ls[0][1:].strip()
                    val = ls[1].strip()
                    if not header_initialized: #If the header is read for the first time, we need to fill different metadata information (basically all)
                        data['header'][key] = val #We collect all metadata in the header - done only Once.
                        if "MEAS_COND_AXIS_NAME-" in key:
                            tmp = key.split("-")
                            order = int(tmp[1].strip())
                            internal['MEAS_COND_AXIS_NAME'][order] = val
                        if "MEAS_COND_AXIS_NAME_INTERNAL-" in key:
                            tmp = key.split("-")
                            order = int(tmp[1].strip())
                            internal['MEAS_COND_AXIS_NAME_INTERNAL'][order] = val
                        if "MEAS_COND_AXIS_OFFSET-" in key:
                            tmp = key.split("-")
                            order = int(tmp[1].strip())
                            try:
                                val = float(val)
                            except:
                                val = 0
                            internal['MEAS_COND_AXIS_OFFSET'][order] = val
                        if "MEAS_COND_AXIS_POSITION-" in key:
                            tmp = key.split("-")
                            order = int(tmp[1].strip())
                            try:
                                val = float(val)
                                internal['MEAS_COND_AXIS_POSITION'][order] = [val]
                            except:
                                internal['MEAS_COND_AXIS_POSITION'][order] = val
                        if "MEAS_COND_AXIS_UNIT-" in key:
                            tmp = key.split("-")
                            order = int(tmp[1].strip())
                            internal['MEAS_COND_AXIS_UNIT'][order] = val
                        if "MEAS_DATA_COUNT" in key:
                            internal['points_per_scan'] = int(float(val))
                        if key == "MEAS_SCAN_AXIS_X":
                            data['scanaxis'] = val
                        if key == "MEAS_SCAN_AXIS_X_INTERNAL":
                            internal['scan_axis_internal'] = val
                        if key == "MEAS_SCAN_START":
                            internal['scan_angle_start'] = float(val)
                        if key == "MEAS_SCAN_STEP":
                            internal['scan_angle_step'] = float(val)
                        if key == "MEAS_SCAN_STOP":
                            internal['scan_angle_stop'] = float(val)
                        if key == "MEAS_SCAN_START_TIME":
                            data['date'], data['time'] = val.split(' ')
                        if key == "MEAS_SCAN_MODE":
                            data['scanmode'] = val
                        if key == "MEAS_SCAN_SPEED":
                            data['scanspeed'] = float(val)
                        if key == "MEAS_3DE_STEP_AXIS_INTERNAL":
                            scan_is_3d = True
                            internal['MEAS_3DE_STEP_AXIS_INTERNAL'] = val.strip()

                            
                    else: #Header already initialized, we add new position to the axis, if they are number and not string.
                        if "MEAS_COND_AXIS_POSITION-" in key:
                            tmp = key.split("-")
                            order = int(tmp[1].strip())
                            try:
                                val = float(val)
                                internal['MEAS_COND_AXIS_POSITION'][order].append(val)
                            except:
                                continue
                                    
                if scan_start:
                    a = float(ls[0].strip())
                    v = float(ls[1].strip())
                    scan_angle.append(a)
                    scan_data.append(v)
                    # print("Angle {:.2f} Intensity: {:.2f}".format(a,v))
                
        data['counts'] = np.asarray(data['counts'])
        if data['numscans'] == 1:
            data['counts'] = data['counts'][0]
        # data['angles'] = np.linspace(internal['scan_angle_start'], internal['scan_angle_stop'], internal['points_per_scan'])

        if scan_is_3d:
            for k, v in internal['MEAS_COND_AXIS_NAME'].items():
                if v == internal['MEAS_3DE_STEP_AXIS_INTERNAL']:
                    axis2_idx = k
                    data['angles2'] = internal['MEAS_COND_AXIS_POSITION'][axis2_idx]


        return data

def remove_background(x, y, peaks = None, peakwidth = None, window = None, plot = False, verbose = False):
    x = np.array(x)
    y = np.array(y)
    if peaks is None:
        peaks = find_peaks_cwt(y, widths = np.arange(10,15), noise_perc = 0.02)
    if peakwidth is None:
        peakwidth = 1.5
    if window is None:
        window = int(len(x)/20)
        if window%2 == 0:
            window += 1
    
    rmidx = []
    for p in peaks:
        for rmidx_ in np.where(np.abs(x - x[p]) <= peakwidth/2)[0]:
            if rmidx_ not in rmidx:
                rmidx.append(rmidx_)
    
    xbl = np.delete(x, rmidx)
    ybl = np.delete(y, rmidx)
    ybl = np.interp(x, xbl, ybl)
    ybl = savgol_filter(ybl, window, 1)
    
    y_corrected = y - ybl
    y_corrected[y_corrected < 0] = 0
    
    if plot:
        fig, ax = plt.subplots(figsize = (4,3))
        ax.plot(x,y, 'k', alpha = 0.4, label = 'Raw', zorder = 3)
        ax.plot(x,ybl, color = 'k', label = 'Baseline', zorder = 2)
        ax.plot(x,y_corrected, color = plt.cm.tab10(0), label = 'Corrected', zorder = 1)
        ylim0 = ax.get_ylim()
        for p in x[p]:
            ax.plot([p,p], ylim0, color = plt.cm.tab10(3), alpha = 0.2, zorder = 0)
        plt.legend()
        plt.show()
        
    if not verbose:
        return y_corrected
    else:
        return {
            'peaks': x[p],
            'peaksidx': p,
            'background': ybl,
            'corrected': y_corrected,
        }

# def stackplot(df, )

def bgRemove_deprecated(xdata,ydata,tol): # thank you: https://github.com/andrewrgarcia/XRDpy
    'approx. # points for half width of peaks'
    L=len(ydata)
    lmda = int(0.50*L/(xdata[0]-xdata[L-1]))
    newdat=np.zeros(L)
    for i in range(L):
        if ydata[(i+lmda)%L] > tol*ydata[i]:          #tolerance 'tol'
            newdat[(i+lmda)%L] = ydata[(i+lmda)%L] - ydata[i]
        else:
            if ydata[(i+lmda)%L] < ydata[i]:
                newdat[(i+lmda)%L] = 0
    return newdat