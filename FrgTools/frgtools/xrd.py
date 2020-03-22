import numpy as np
import os

def LoadSmartlab(fpath):
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
        data['angles'] = np.linspace(internal['scan_angle_start'], internal['scan_angle_stop'], internal['points_per_scan'])


        return data