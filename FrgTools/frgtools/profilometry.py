import numpy as np

def load_filmetrics(fid):
    '''
    reads output text file from filmetrics tool in Nano3

    all units in microns
    '''
    data = {}
    
    with open(fid, 'r') as f:
        data['units'] = f.readline().split('(')[1][:-2]
        data['scale'] = float(f.readline().split('\t')[2])

    data['z'] = np.genfromtxt(
        fid,
        skip_header = 3
    )
    data['x'] = [data['scale']*i for i in range(data['z'].shape[1])]
    data['y'] = [data['scale']*i for i in range(data['z'].shape[0])]
    data['extent'] = [data['x'][0], data['x'][-1], data['y'][0], data['y'][-1]]
    data['filepath'] = fid

    return data