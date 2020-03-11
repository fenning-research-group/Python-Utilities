import os

def listdir(path, display = True):
	exclude = ['desktop.ini']
	fids = [os.path.abspath(os.path.join(path, x)) for x in os.listdir(path) if not any([exclude_ in x for exclude_ in exclude])]
	if display:
		print('Files in \'{}\':'.format(path))
		for i, f in enumerate(fids):
			print('{}:{}'.format(i, os.path.basename(f)))
	return fids

def searchdir(path, find = [], ignore = ['desktop.ini'], fids = []):
	f1s = [os.path.join(path, x) for x in os.listdir(path) if not any([y in x for y in ignore])]
	for f1 in f1s:
		if os.path.isdir(f1):
			fids = searchdir(f1, fids)
		elif any([x in f1 for x in find]):
			fids.append(f1)
	return fids

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
	msg['Subject'] = '[frgMapper] ' + subject
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

