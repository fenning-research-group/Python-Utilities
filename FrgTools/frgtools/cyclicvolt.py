# getCV
"""
Loads CV data from a text file. Pulls the last curve out of the
repeats (red/ox cycle).
Returns structure. 

Example: mydata = getCV     (use UI to select file)

         mydata = getCV ('C:\Myname\CV\File.ras')   (directly offer filepath)

Onset potential methods: 
    1: Direct minimum
    2: Inflection point
    3: Intersection at baseline
    4: Fixed
"""
import os

def getFileList(directory = os.path.dirname(file_name)):#os.path.realpath(__file__)))
	return os.listdir(directory)

def getCV(filepath, area=1):
	#getFileList(filepath)
	txt = open(filepath,'r')
	if area == 1: 
		#Make title say Current (mA)

print(getCV("C:\Users\Skaggs\Desktop\Perovskites\2019_08_09 CV SnOx\10nm\180_11.4_FTO_02_CV_C02"))