import json
import matplotlib.pyplot as plt
import numpy as np
import os

rootDir = os.path.dirname(__file__)
radiiFilepath = os.path.join(rootDir, 'Include', 'atomicradii.json')
with open(radiiFilepath, 'r') as f:
	radii = json.load(f)


### Tolerance factor calculations ###

def goldschmidt(a, b, x):
	"""
	a,b,x can either be a string ('Cs', 'Br', etc) for a single site species, or a
	dictionary of species:fraction for mixed sites. ({'Cs':0.5, 'FA':0.5}, etc)

	example:

		tf = tolerancefactor.goldschmidt(
			a = 'Cs',
			b = {'Pb': 0.5, 'Sn': 0.5},
			x = 'Br'
		)	
	"""
	r_a = _parse_radius(a)
	r_b = _parse_radius(b)
	r_x = _parse_radius(x)

	return (r_a+r_x) / (np.sqrt(2)*(r_b+r_x))

def bartel(a, b, x, oxidationstate_a = 1):
	"""
	returns tau and the associated probability of perovskite formation, based on:
		https://advances.sciencemag.org/content/5/2/eaav0693
		https://advances.sciencemag.org/content/suppl/2019/02/04/5.2.eaav0693.DC1

	
	a,b,x can either be a string ('Cs', 'Br', etc) for a single site species, or a
	dictionary of species:fraction for mixed sites. ({'Cs':0.5, 'FA':0.5}, etc)

	example:

		tau, p = tolerancefactor.bartel(
			a = 'Cs',
			b = {'Pb': 0.5, 'Sn': 0.5},
			x = 'Br'
		)	
	"""
	def fsigmoid(x, a, b):
		return 1.0 / (1.0 + np.exp(-a*(x-b)))

	r_a = _parse_radius(a)
	r_b = _parse_radius(b)
	r_x = _parse_radius(x)

	tau = (r_x/r_b) - oxidationstate_a*(oxidationstate_a - (r_a/r_b)/np.log(r_a/r_b))

	p_opt = [-1.73194355,  4.26437976] #found by fitting sigmoid to webplotdigitized data from paper
	probability_of_formation = fsigmoid(tau, *p_opt)

	return tau, probability_of_formation

### Atomic radius handling ###

def add_radius(name, radius):
	if radius < 0.5 or radius > 5:
		response = input('{0} provided as radius for {1} - is this the correct value in Angstroms? (y/n):'.format(radius, name))
		if str.lower(response) != 'y':
			return

	if name in radii.keys():
		currentRadius = _lookup_radius(name)
		response = input('{0} already in database with radius {1} A. Overwrite with radius {2} A? (y/n): '.format(name, currentRadius, radius))
		if str.lower(response) != 'y':
			return

	radii[name] = radius
	with open(radiiFilepath, 'w') as f:
		json.dump(radii, f)

def _lookup_radius(key):
	try:
		radius = radii[key]
	except:
		print('Error: {0} not included in atomic radii database. Please add using frgtools.tolerancefactor.AddAtomicRadius(name, radius).'.format(key))
		radius = False
	return radius

def _parse_radius(raw):
	if type(raw) is dict:
		radius = 0
		for k, v in raw.items():
			if type(k) is str:
				thisRadius = _lookup_radius(k)
			else:
				thisRadius = k
			radius = radius + v*thisRadius
	elif type(raw) is str:
		radius = _lookup_radius(raw)
	else:
		radius = raw

	return radius
