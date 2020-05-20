import os
from tqdm import tqdm
import re

def listdir(path = '.', display = True):
	exclude = ['desktop.ini']
	fids = [os.path.abspath(os.path.join(path, x)) for x in os.listdir(path) if not any([exclude_ in x for exclude_ in exclude])]
	if display:
		print('Files in \'{}\':'.format(path))
		for i, f in enumerate(fids):
			print('{}:{}'.format(i, os.path.basename(f)))
	return fids

def searchdir(path = '.', find = [], ignore = ['desktop.ini'], fids = [], match_directories = False):
	"""
	path: path to directory to be searched
	find: list of substrings used to identify files of interest.
	ignore: list of substrings used to identify files/folders to ignore
	fids: empty list to be populated with filepaths. should probably leave empty to start
	match_directories: boolean flag, determines whether or not to consider a filepath a match if the find substring is present in the directory name.

	Traverses all subdirectories of a given path and return files which contain strings included in "find".
	Ignores files and folders that contain strings included in "ignore"
	"""
	f1s = [os.path.abspath(os.path.join(path, x)) for x in os.listdir(path) if not any([y in x for y in ignore])]
	for f1 in tqdm(f1s, leave = False, desc = path):
		if match_directories:
			comparestring = f1
		else:
			comparestring = os.path.basename(f1)

		if os.path.isdir(f1):
			try:
				fids = searchdir(path = f1, find = find, ignore = ignore, fids = fids, match_directories = match_directories)
			except:
				tqdm.write('Error searching {}'.format(f1))
		elif any([x in comparestring for x in find]):
			fids.append(f1)
	temp = fids.copy()
	fids = []
	return temp

def natsort(l):
    """
    Input a list (os.listdir(path))
    Natsort will sort list 'l' as windows does. 

    Example:
    List: [PL1, PL10, PL11, PL2, PL3]
	natsort(List) -> [PL1,PL2,PL3,PL10,PL11]
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l

    
# load video numpy array
import cv2
import numpy as np

def load_video(fpath):
	"""
	Loads a video, returns frame by frame as numpy array

	fpath: string, filepath to video file.
	"""
	cap = cv2.VideoCapture(filename = fpath)
	frames = []
	cap.open(filename = fpath)
	while (cap.isOpened()):
		ret,frame = cap.read()
		if not ret:
			break
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frames.append(gray)
	cap.release()
	return np.array(frames)


### script to send email from generic FRG alert address
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

senderUsername = 'frgalerts'
senderAddress = 'frgalerts@gmail.com'
senderPassword = 'sggdoxnrywcjucaj'


def SendEmail(recipient, subject = '', body = ''):
	fromaddr = senderAddress
	server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
	server.login(senderUsername, senderPassword)
	
	msg = MIMEMultipart()
	msg['From'] = senderAddress
	msg['To'] = recipient
	msg['Subject'] = '[FRG-alert] ' + subject
	body = body
	msg.attach(MIMEText(body, 'plain'))

	text = msg.as_string()
	try:
		server.sendmail(fromaddr, recipient, text)
	except:
		print('Error encountered when sending email "{0}" to {1}'.format(subject, recipient))


### general functional forms

def gaussian(x, a, b, c):
	"""
	a = magnitude
	b = center value
	c = standard deviation
	"""
	return a * np.exp( -(x-b)**2 / (2*c**2))

