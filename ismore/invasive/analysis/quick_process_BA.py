from ismore import brainamp_channel_lists
from db import dbfunctions as dbfn
import numpy as np
import scipy.signal
import tables

def get_filtered_data(te = 11198, chan_names = brainamp_channel_lists.sleep_mont_new):

    tsk = dbfn.TaskEntry(te)
    supp = '/storage/supp_hdf/'+tsk.name+'.supp.hdf'
    supp_hdf = tables.openFile(supp)
    fs = 1000.


    band  = [10, 450]  # Hz
    nyq   = 0.5 * fs
    low   = band[0] / nyq
    high  = band[1] / nyq
    bpf_coeffs = scipy.signal.butter(4, [low, high], btype='band')
    
    band  = [.1, 30]  # Hz
    nyq   = 0.5 * fs
    low   = band[0] / nyq
    high  = band[1] / nyq
    bpf_coeffs_eeg = scipy.signal.butter(2, [low, high], btype='band')
    # calculate coefficients for multiple 2nd-order notch filers
    notchf_coeffs = []
    for freq in [50, 150, 250, 350]:
        band  = [freq - 2, freq + 2]  # Hz
        nyq   = 0.5 * fs
        low   = band[0] / nyq
        high  = band[1] / nyq
        notchf_coeffs.append(scipy.signal.butter(2, [low, high], btype='stop'))

    # get channel names: 
    C = []
    C_unfilt = []
    for channel in chan_names:
        # Apply notch filter: 
        signal = supp_hdf.root.brainamp[:]['chan'+channel]['data']
        C_unfilt.append(signal)

        if 'EEG' in channel:
            b, a = bpf_coeffs_eeg
            filt_signal = scipy.signal.filtfilt(b, a, signal)
            print channel, 'applying EEG'

        else:
            b, a = bpf_coeffs
            filt_signal = scipy.signal.filtfilt(b, a, signal)

        for n, (b, a) in enumerate(notchf_coeffs):
            filt_signal = scipy.signal.filtfilt(b, a, filt_signal)

        C.append(filt_signal)

    C = np.vstack((C)).T
    C_unfilt = np.vstack((C_unfilt)).T
    return C, C_unfilt

def fft_signal(signal):
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft
    T = 1./ 1000.
    N = signal.shape[0]
    yf = scipy.fftpack.fft(signal)
    xf = np.linspace(0, 1./(2*T), N/2.)
    plt.plot(xf, np.log10(2./N*np.abs(yf[:N/2])))