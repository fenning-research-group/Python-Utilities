import os
from tqdm import tqdm

#search directories

def listdir(path = '.', find = None, ignore = ['desktop.ini', '.DS_Store'], list_directories = True, display = False):
    """
    path: path to directory whose contents are to be listed
    find: list of substrings used to identify files/folders of interest.
    ignore: list of substrings used to identify files/folders to ignore
    list_directories: boolean flag, determines whether or not to list directories.
    display: boolean flag. prints out a list of filenames/indices to console to aid indexing.

    Traverses all subdirectories of a given path and return files which contain strings included in "find".
    Ignores files and folders that contain strings included in "ignore"
    """
    if type(find) is str:
        find = [find]
    if type(ignore) is str:
        ignore = [ignore]
    
    fids = [os.path.abspath(os.path.join(path, x)) for x in os.listdir(path) if not any([exclude_ in x for exclude_ in ignore])]
    if find is not None:
        fids = [f for f in fids if any([include_ in f for include_ in find])]
    if not list_directories:
        fids = [f for f in fids if not os.path.isdir(f)]
    if display:
        print('Files in \'{}\':'.format(path))
        for i, f in enumerate(fids):
            print('{}:{}'.format(i, os.path.basename(f)))
    return fids

def searchdir(path = '.', find = [], ignore = ['desktop.ini', '.DS_Store'], fids = [], match_directories = False):
    """
    NOTE - this function is buggy sometimes if ran more than once - fids will carry over across runs.
            quick fix - explicitly pass "fids = []", will clear old hits.

    path: path to directory to be searched
    find: list of substrings used to identify files of interest.
    ignore: list of substrings used to identify files/folders to ignore
    fids: empty list to be populated with filepaths. should probably leave empty to start
    match_directories: boolean flag, determines whether or not to consider a filepath a match if the find substring is present in the directory name.

    Traverses all subdirectories of a given path and return files which contain strings included in "find".
    Ignores files and folders that contain strings included in "ignore"
    """
    if type(find) is str:
        find = [find]
    if type(ignore) is str:
        ignore = [ignore]

    f1s = [os.path.abspath(os.path.join(path, x)) for x in os.listdir(path) if not any([y in x for y in ignore])]
    for f1 in f1s:
        if match_directories:
            comparestring = f1
        else:
            comparestring = os.path.basename(f1)
        if os.path.isdir(f1):
            try:
                fids = searchdir(path = f1, find = find, ignore = ignore, fids = fids, match_directories = match_directories)
            except:
                print('Error searching {}'.format(f1))
        elif any([x in comparestring for x in find]):
            fids.append(f1)
    temp = fids.copy()
    fids = []
    return temp

def treeview(d, tabs = 0, print_last_branch = False):
    '''
    prints the hierarchy of a dictionary of h5 file
    '''
    def print_with_tabs(printme, tabs):
        printstr = ''
        for t in range(tabs):
            printstr += '\t'
        printstr += f'{printme}'
        print(printstr)
        
    try:
        keys = d.keys()
        for k in keys:
            print_with_tabs(k, tabs)
            print_tree(d[k], tabs = tabs+1)
    except:
        if print_last_branch:
            print_with_tabs(d, tabs)

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


def sendemail(recipient, subject = '', body = ''):
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

