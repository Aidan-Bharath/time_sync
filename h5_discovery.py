
import sync
import h5py as h5
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from glob import glob 

def pl(*args,**kwds):
    plt.figure()
    for i,arg in enumerate(args):
        if 'label' in kwds:
            label = kwds['label'][i]
        plt.plot(arg[0],arg[1],label=label)
    plt.legend()
    plt.show()


def main(*args,**kwds):

    norm = np.linalg.norm
    mean = np.mean 

    dfiles = args[0]

    start, stop = 0,1000000

    for df in dfiles:
        with h5.File(df,'r') as hdf:
            if 'ADV' in df:
                base = np.vstack([hdf['mpltime'][:],
                                  norm(hdf['vel'][:],axis=0)])
            elif 'SBM' in df:
                mod = np.vstack([hdf['mpltime'][:],
                                 norm(
                                     mean(hdf['vel'][:],axis=1)
                                    ,axis=0)])

    sts = sync.sync_timeseries(base,mod,*args,**kwds)
    itr, corr = sts[0], np.array(sts[1])

    pl((itr,corr[:,0]),(itr,corr[:,1]),
        label=('Correction','Correlation'))

    pl(base,mod,label=('ADV','SMB'))

if __name__ == "__main__":

    bfile = "../data_for_Michelle_timesync"
    dfiles = [i for i in glob(f"{bfile}/*.h5") 
                if 'unix' not in i]
    

    args = [dfiles]
    kwds = {
        'window':0.01,
        'shift':50,
        'plot':False
    }

    main(*args,**kwds)
