# getCV
"""
Loads CV data from a text file. Pulls the last curve out of the
repeats (red/ox cycle).
Returns structure. 

Example: mydata = getCV     (use UI to select file)

		 mydata = getCV ('C:\Myname\CV\File.ras')   (directly offer fpath)

Onset potential methods: 
	1: Direct minimum
	2: Inflection point
	3: Intersection at baseline
	4: Fixed
"""
import os
import numpy as np
import csv
from .plotting import directionalplot

# def getCV(fpath, area = 1):
# 	#getFileList(fpath)
# 	txt = open(fpath,'r')
# 	if area == 1: 
# 		#Make title say Current (mA)

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
			elif c.startswith('control'):
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
		data['v'] = data['scans']['v'][currentcycle-1]
		data['j'] = data['scans']['j'][currentcycle-1]

		return data
