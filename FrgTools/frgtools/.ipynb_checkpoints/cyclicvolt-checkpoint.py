# getCV
"""
Loads CV data from a text file. Pulls the last curve out of the
repeats (red/ox cycle).
Returns structure. 

Example: mydata = getCV(use UI to select file)

		 mydata = getCV ('C:\Myname\CV\File.ras')   (directly offer fpath)

Output: Structure with J and V from the final run, along with all other associated data in text file.
    mydata['v'] --> voltage
    mydata['j'] --> current density
        ['header'] --> all info from the header
        ['area']
        ['date']
        ['time']
        ['num_cycles']
        ['cathodic_peak'] --> onset potential method 1
        ['anodic_peak'] --> anodic peak current (at the right most end (highest positive potential))
        ['inflection_onset'] --> onset method 2
        ['intersect_onset'] --> onset method 3
        ['scans'] --> 
            [j][#]
            [v][#]
            
            
Onset potential methods: 
	1: Direct minimum
	2: Inflection point
	3: Intersection at baseline

"""
import os
import numpy as np
import csv
# from .plotting import directionalplot

def getCV(fpath, area =1):
    cvdata = load_cv(fpath)
    cvdata = getOnsetPotentials(cvdata)
    return cvdata

def load_cv(fpath, area = 1):
	# headerkey = {
	#     'Run on channel': 'channel',
	#     'Electrode connection': 'electrode_connection',
	#     'Ewe,I filtering': 'ewe_ifilter',
	#     'Channel': 'channel',
	#     'Acquisition started on': 'datetime',
	#     'Reference electrode': 'ref_electrode',
	# }


	DATA_START = 'mode\tox/red\terror'

	data = {
		'v': None,
		'j': None,
		'numscans': None,
		'area': area,
		'scans': {},
		'header': {}
	}

	key = None
	previouskey = None
	with open(fpath, 'r', errors = 'ignore', encoding = 'utf-8') as f:
		headerlines = 54 #initial guess, will get picked up in code at 
		line = f.readline()#.decode('utf-8')
		while not line.startswith(DATA_START):
			if line.startswith('Acquisition started on'):
				data['date'], data['time'] = line.split(' : ')[1].split(' ')
			else:
				if ' : ' in line:
					key, value = line.split(' : ')
					data['header'][key.strip()] = value.strip()
				elif '  ' in line:
					parts = line.strip().split('  ')
					key = parts[0]
					value = parts[-1]
					if key == 'vs.':
						data['header'][previouskey] += ' vs. {}'.format(value.strip())
					else:
						data['header'][key.strip()] = value.strip()
				previouskey = key
			line = f.readline()#.decode('utf-8')
		
		colnames = []
		for c in line.strip().split('\t'):
			if c == 'ox/red':
				colnames.append(c)
			elif c.startswith('<I>'):
				colnames.append('i')
			elif c.startswith('Ewe'):
				colnames.append('v')
			elif c.startswith('cycle'):
				colnames.append('cycle')
			else:
				colnames.append(c.split('/')[0])
		f_reader = csv.DictReader(f, fieldnames = colnames, delimiter = '\t')  
		data['scans'] = {k:[[]] for k in colnames}
		currentcycle = 1
		for row in f_reader:
			if int(row['cycle']) > currentcycle:
				currentcycle = int(row['cycle'])
				for k in data['scans'].keys():
					data['scans'][k].append([])
			for k, val in row.items():
				data['scans'][k][currentcycle-1].append(float(val))

		data['scans'] = {k:[np.array(cycledata) for cycledata in v] for k,v in data['scans'].items()}
		data['scans']['j'] = [i/area for i in data['scans']['i']]
		data['num_cycles'] = currentcycle
		data['v'] = data['scans']['v'][-1]
		data['j'] = data['scans']['j'][-1]

		return data

def getOnsetPotentials(cvdata,fitwidth = 40):
    # Finding cathodic peak voltage
    X = cvdata['v']
    Y = cvdata['j']
    idx0 = np.argmin(Y)
    cvdata['cathodic_peak'] = X[idx0] 
    
    # Finding anodic peak current at the end of the voltammogram
    idx1 = np.argmax(X)
    cvdata['anodic_peak'] = Y[idx1] 

    # Inflection point onset
    idx2 = np.argmin(X)
    XX = X[idx1+500 : idx2 - 20]
    YY = Y[idx1+500 : idx2 - 20]
    
    dydx = np.gradient(YY)/np.gradient(XX) # Finds first derivative, delta y/ delta x
    dydx = smooth(dydx)
    
    idx3 = np.argmax(dydx) # Inflection point on the cathodic curve
    idx4 = np.argmin(abs(YY)) # Getting closest to the baseline center as possible
    cvdata['inflection_onset'] = XX[idx3]
    
    X_ = XX[idx3 - fitwidth//2: idx3 + fitwidth//2]
    Y_ = YY[idx3 - fitwidth//2: idx3 + fitwidth//2]
    linfit_3 = np.polyfit(X_,Y_,1)
    
    X_ = XX[idx4 - fitwidth//2: idx4 + fitwidth//2]
    Y_ = YY[idx4 - fitwidth//2: idx4 + fitwidth//2]
    linfit_4 = np.polyfit(X_,Y_,1)
    
    intersect_x = (linfit_4[1] - linfit_3[1]) / (linfit_3[0] - linfit_4[1])
    intersect_y = linfit_3[0] * intersect_x + linfit_3[1]
    cvdata['intersect_onset'] = [intersect_x, intersect_y]
    return cvdata

def smooth(a,WSZ=5):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))
