#!/usr/bin/python

# from bmi3d import *
import argparse
import tables
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, splrep, splev
import pickle

from ismore.common_state_lists import *
from utils.constants import *

from db import dbfunctions as dbfn

class IBMITaskEntry(dbfn.TaskEntry):
    pass

class IBMIPassiveTaskEntry(IBMITaskEntry):
    trial_end_states = ['rest']


##### Trial filter functions #####
def trial_has_movement(te, trial_msgs):
    return 'trial' in trial_msgs['msg']


##### Trial proc functions #####
def _interpolate_table(state_data, ts_data, ts_interp):
    '''
    Interpolate a record array of data to new time samples

    Parameters
    ----------
    state_data : np.recarray 
    '''
    dtype = state_data.dtype
    states = dtype.names
    n_states = len(states)
    n_ts = len(ts_interp)
    interp_data = np.zeros([n_ts, n_states + 1])
    interp_data[:,0] = ts_interp
    for k, state in enumerate(states):
        # interp_fn = interp1d(ts_data, state_data[state])
        # interp_data[:,k+1] = interp_fn(ts_interp)

        from scipy.interpolate import interp1d, splrep, splev
        tck = splrep(ts_data, state_data[state], s=3)
        interp_data[:,k+1] = splev(ts_interp, tck)
        # import pdb; pdb.set_trace()
        
    return pd.DataFrame(interp_data, columns=('ts',) + states)

def _get_inds(ts_arrival, ts_start, ts_end, interpolate=False):
    idxs = [i for (i, ts) in enumerate(ts_arrival) if ts_start <= ts <= ts_end]
        
    if interpolate:
        # add one more idx to the beginning and end, if possible
        if idxs[0] != 0:
            idxs = [idxs[0]-1] + idxs
        if idxs[-1] != len(ts_arrival)-1:
            idxs = idxs + [idxs[-1]+1]
    return idxs

def extract_data(tbl, ts_start, ts_end, ts_interp=None, interpolate=False):
    '''
    Get trajectories from table, interpolating if necessary

    Parameters
    ----------
    tbl : HDF table
    '''
    tbl_ts = tbl[:]['ts_arrival']
    idxs = _get_inds(tbl_ts, ts_start, ts_end, interpolate=interpolate)
    data = tbl[idxs]['data']

    if interpolate:
        return _interpolate_table(data, tbl_ts[idxs], ts_interp)
    else:
        df_data = pd.DataFrame(data, columns=data.dtype.names)
        df_ts = pd.DataFrame(tbl_ts, columns=['ts'])
        return pd.concat([df_ts, df_data], axis=1)    

def extract_trajectories(te, trial_msgs, ts_step=0.010):
    '''
    Get trajectories from table, interpolating if necessary

    Parameters
    ----------
    ts_step : float
        TODO
    '''

    hdf = te.hdf
    task = te.hdf.root.task
    aa_flag = 'armassist' in hdf.root
    rh_flag = 'rehand' in hdf.root
    if aa_flag:
        armassist = hdf.root.armassist
    if rh_flag:
        rehand = hdf.root.rehand    


    # determine type of task (record vs. playback)
    if 'command_vel' in hdf.root.task.colnames:  # was a playback trajectories task
        INTERPOLATE_TRAJ = False
    else:                                        # was a record trajectories task
        INTERPOLATE_TRAJ = True                  

    start_msg_ind = np.nonzero(trial_msgs['msg'] == 'trial')[0][0]
    idx_start = trial_msgs['time'][start_msg_ind]
    idx_end = trial_msgs['time'][start_msg_ind+1]

    traj_trial_type = dict()

    # actual start and end times of this trial 
    ts_start = task[idx_start]['ts']  # secs
    ts_end   = task[idx_end]['ts']    # secs

    if INTERPOLATE_TRAJ:
        # finely-spaced vector of time-stamps onto which we will interpolate armassist and rehand data
        ts_interp = np.arange(ts_start, ts_end, ts_step)
    else:
        ts_interp = None

    extraction_kwargs = dict(ts_interp=ts_interp, interpolate=INTERPOLATE_TRAJ)
    if aa_flag:
        df_aa = extract_data(armassist, ts_start, ts_end, **extraction_kwargs)
        aa_inds = _get_inds(armassist[:]['ts_arrival'], ts_start, ts_end, interpolate=INTERPOLATE_TRAJ)
        traj_trial_type['armassist_raw'] = armassist[aa_inds]

    if rh_flag:
        df_rh = extract_data(rehand, ts_start, ts_end, **extraction_kwargs)
        rh_inds = _get_inds(rehand[:]['ts_arrival'], ts_start, ts_end, interpolate=INTERPOLATE_TRAJ)
        traj_trial_type['rehand_raw'] = rehand[rh_inds]

    if INTERPOLATE_TRAJ:
        df_traj = pd.concat([df_aa, df_rh[rh_pos_states + rh_vel_states]], axis=1)
    else:
        df_traj = None


    traj_trial_type['ts_start'] = ts_start
    traj_trial_type['ts_end']   = ts_end
    traj_trial_type['task'] = task[idx_start:idx_end]

    traj_trial_type['armassist'] = df_aa
    traj_trial_type['rehand'] = df_rh
    traj_trial_type['traj'] = df_traj


    return traj_trial_type

from scipy.interpolate import interp1d
from math import factorial
from scipy.interpolate import interp1d

def savgol_filter(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')



def pos_to_vel(data, window_size=101, **kwargs):
    # from scipy.signal import savgol_filter
    return savgol_filter(data, window_size, 3, deriv=1, **kwargs)

