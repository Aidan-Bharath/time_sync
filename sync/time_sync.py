import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate as interp


def sync_timeseries(*args,**kwds):

    """
    Base function to synchornize timeseries data
    using and auto correlation

    args:   
        args[0]: Basis Timeseries
                array(X,2) - [:,0] timestamp
                           - [:,1] data

        args[1]: Timesseries to be Sync'd
                array(X,2) - [:,0] timestamp
                           - [:,1] data
    """

    assert kwds['window'], f"Provide a time window"
    wdw = kwds['window']

    base, mod = args[0], args[1]
    bTDiff, mTDiff = base[0,1]-base[0,0], mod[0,1]-mod[0,0]
    
    bWdw, mWdw = int(np.floor(wdw/bTDiff)), int(np.floor(wdw/mTDiff))
    
    loops = int(np.floor(base.shape[-1]/bWdw))
    ts_shift = []
    for i in range(loops):
        start, stop = base[0,i*bWdw], base[0,i*bWdw+bWdw]
        if any(mod[0,:] <= start) and any(mod[0,:] >= stop):
            mSlc = np.argwhere(
                     ((mod[0,:] >= start) == (mod[0,:] <= stop)) == True
                    ).flatten()
            b, m = base[:,i*bWdw:i*bWdw+bWdw], mod[:,mSlc[0]:mSlc[-1]]
            ts_shift.append(auto_correlate(b,m,*args,**kwds))


    return [i for i in range(len(ts_shift))], ts_shift


def auto_correlate(*args, shift = 50, direction = 'both', **kwds):

    """
    compute the auto correlation between two timeseries

    args:   
        args[0]: Basis Timeseries
                array(X,2) - [:,0] timestamp
                           - [:,1] data

        args[1]: Timesseries to be Sync'd
                array(X,2) - [:,0] timestamp
                           - [:,1] data
    """

    base, mod = args[0], args[1]

    if direction == 'both': 
        itr = np.arange(-shift,shift,1)
        return shift_interp(base,mod,itr,*args, **kwds)
    elif direction == 'lower': 
        itr = np.arange(-shift,0,1)
        return shift_interp(base,mod,itr,*args, **kwds)
    elif direction == 'higher': 
        itr = np.arange(0,shift,1)
        return shift_interp(base,mod,itr,*args, **kwds)
    else: 
        assert True, f"Correlation shift not specified"



def shift_interp(*args, **kwds):
    """
    compute the cross correlation between two timeseries

    args:   
        args[0]: Basis Timeseries
                array(X,2) - [:,0] timestamp
                           - [:,1] data

        args[1]: Timesseries to be Sync'd
                array(X,2) - [:,0] timestamp
                           - [:,1] data
        
        args[2]: Time shift iterator array
                array(X)
    """

    base, mod, itr = args[0], args[1], args[2]
    corr = []
    for i,j in enumerate(itr): 
        if j < 0:
            time, data = mod[0,abs(j):], mod[1,:-abs(j)]
        elif j > 0:
            time, data = mod[0,:-abs(j)], mod[1,abs(j):]
        else:
            time, data = mod[0,:], mod[1,:]

        bSlc = np.argwhere(
                ((base[0,:] > time[0]) == (base[0,:] < time[-1])) == True
                ).flatten()


        bd = base[-1,bSlc]
        dt = base[0,bSlc][1]-base[0,bSlc][0]

        ts_mod = interp.interp1d(time,data)
        corr.append(ac(bd,ts_mod(base[0,bSlc]),dt))
        if 'plot' in kwds and kwds['plot']:
            if j == -50:
                plt.plot(base[0,bSlc],ts_mod(base[0,bSlc]),label='interp')
                plt.plot(base[0,bSlc],bd,label='base')
            elif j == 49:
                plt.plot(base[0,bSlc],ts_mod(base[0,bSlc]),label='interp1')
                plt.plot(base[0,bSlc],bd,label='base1')

    td = mod[0,1]-mod[0,0]
    corr = np.array(corr)

    if 'plot' in kwds and kwds['plot']:
        plt.legend()
        plt.show()

        td = mod[0,1]-mod[0,0]
        plt.plot(itr*td,corr)
        plt.show()
    
    return itr[corr.argmax()]*td, corr.max()

def ac(*args,**kwds):

    """
    Calculate the auto correlation

    args:   
        args[0]: Basis Timeseries
                array(X) - base dataset

        args[1]: comparison Timesseries 
                array(X) - comparison dataset
    """

    return np.sum(args[0]*args[1]*args[2])

if __name__ == "__main__":

    pass
