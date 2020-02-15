import os

def listdir(path, display = True):
    exclude = ['desktop.ini']
    fids = [os.path.join(path, x) for x in os.listdir(path) if not any([exclude_ in x for exclude_ in exclude])]
    if display:
        print('Files in \'{}\':'.format(path))
        for i, f in enumerate(fids):
            print('{}:{}'.format(i, os.path.basename(f)))
    return fids