import json
import matplotlib.pyplot as plt
import numpy as np

_radiiDatabaseFilepath = './Include/atomicradii.json'
with open(_radiiDatabaseFilepath, 'r') as f:
	_radiiDatabase = json.load(f)

def GoldschmidtTF(a, b, x):
	r_a = _ParseAtomicRadius(a)
	r_b = _ParseAtomicRadius(b)
	r_x = _ParseAtomicRadius(x)

    return (r_a+r_x) / (np.sqrt(2)*(r_b+r_x))

def ChathamFormability(a, b, x, oxidationstate_a = 1):
	"""
	https://advances.sciencemag.org/content/suppl/2019/02/04/5.2.eaav0693.DC1
	"""
	def fsigmoid(x, a, b):
    	return 1.0 / (1.0 + np.exp(-a*(x-b)))

	r_a = _ParseAtomicRadius(a)
	r_b = _ParseAtomicRadius(b)
	r_x = _ParseAtomicRadius(x)

    p_opt = [-1.73194355,  4.26437976] #found by fitting sigmoid to webplotdigitized data from paper
    tau = (r_x/r_b) - oxidationstate_a*(ox_state_a - (r_a/r_b)/np.log(r_a/r_b))
    
    return fsigmoid(tau, *p_opt)

def _LookUpAtomicRadius(key):
	try:
		radius = _radiiDatabase[key]
	except:
		print('Error: {0} not included in atomic radii database.'.format(key))
		radius = False
	return radius

def _ParseAtomicRadius(raw):
	if type(raw) is dict:
		radius = 0
		for k, v in raw:
			if type(k) is str:
				thisRadius = _LookUpAtomicRadius(k)
			else:
				thisRadius = k
			radius = radius + v*thisRadius
	else:
		radius = raw

	return radius

# def AddAtomicRadius(name, radius):
