'''
Code for feature extraction methods/classes from EMG, to be used with a 
decoder (similar to other types of feature extractors in riglib.bmi.extractor)
'''

from collections import OrderedDict
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
#import math
# from riglib.filter import Filter
from ismore.filter import Filter
from riglib.bmi.extractor import FeatureExtractor
from utils.ringbuffer import RingBuffer
from ismore import brainamp_channel_lists

import time
def extract_MAV(samples):
    '''
    Calculate the mean absolute value (MAV) for multiple channels.

    Parameters
    ----------
    samples : np.ndarray of shape (n_channels, n_time_points)
        Observed EMG voltages in microvolts

    Returns
    -------
    np.ndarray of shape (n_channels, 1)
    '''

    return np.mean(abs(samples), axis=1, keepdims=True)

def extract_WAMP(samples, threshold=0):
    '''
    Calculate the Willison amplitude (WAMP) for multiple channels.

    Parameters
    ----------
    samples : np.ndarray of shape (n_channels, n_time_points)
        Observed EMG voltages in microvolts
    threshold : np.float
        threshold in uV below which WAMP isn't counted

    Returns
    -------
    np.ndarray of shape (n_channels, 1)
    '''

    condition = abs(np.diff(samples)) >= threshold
    return np.sum(condition, axis=1, keepdims=True)

def extract_VAR(samples):
    '''
    Calculate the variance (VAR) for multiple channels.

    Parameters
    ----------
    samples : np.ndarray of shape (n_channels, n_time_points)
        Observed EMG voltages in microvolts

    Returns
    -------
    np.ndarray of shape (n_channels, 1)
    '''

    N = samples.shape[1]
    return (1./(N-1)) * np.sum(samples**2, axis=1, keepdims=True)

def extract_WL(samples):
    '''
    Calculate the waveform length (WL) for multiple channels.

    Parameters
    ----------
    samples : np.ndarray of shape (n_channels, n_time_points)
        Observed EMG voltages in microvolts

    Returns
    -------
    np.ndarray of shape (n_channels, 1)
    '''

    return np.sum(abs(np.diff(samples)), axis=1, keepdims=True)

def extract_RMS(samples):
    '''
    Calculate the root mean square (RMS) value for multiple channels.

    Parameters
    ----------
    samples : np.ndarray of shape (n_channels, n_time_points)
        Observed EMG voltages in microvolts

    Returns
    -------
    np.ndarray of shape (n_channels, 1)
    '''
    
    N = samples.shape[1]
    # try:
    #     np.sqrt((1./N) * np.sum(samples**2, axis=1, keepdims=True))

    # except ZeroDivisionError:

    #     import pdb; pdb.set_trace()

    return np.sqrt((1./N) * np.sum(samples**2, axis=1, keepdims=True))

def extract_ZC(samples, threshold=0):
    '''
    Compute the number of zero crossings (ZC) for multiple channels.

    Parameters
    ----------
    samples : np.ndarray of shape (n_channels, n_time_points)
        Observed EMG voltages in microvolts
    threshold : np.float
        threshold in uV below which zero crossings aren't counted

    Returns
    -------
    np.ndarray of shape (n_channels, 1)
    '''

    zero_crossing = np.sign(samples[:, 1:] * samples[:, :-1]) == -1
    greater_than_threshold = abs(np.diff(samples)) >= threshold
    condition = np.logical_and(zero_crossing, greater_than_threshold)

    return np.sum(condition, axis=1, keepdims=True)

def extract_SSC(samples, threshold=0):
    '''
    Compute the number of slope-sign changes (SSC) for multiple channels.

    Parameters
    ----------
    samples : np.ndarray of shape (n_channels, n_time_points)
        Observed EMG voltages in microvolts
    threshold : np.float
        threshold in uV below which SSCs aren't counted

    Returns
    -------
    np.ndarray of shape (n_channels, 1)
    '''

    diff = np.diff(samples)
    condition = diff[:, 1:] * diff[:, :-1] >= threshold
    return np.sum(condition, axis=1, keepdims=True)

def extract_LOGVAR(samples):
    '''
    Calculate the log variance (LOGVAR) for multiple channels.

    Parameters
    ----------
    samples : np.ndarray of shape (n_channels, n_time_points)
        Observed EMG voltages in microvolts

    Returns
    -------
    np.ndarray of shape (n_channels, 1)
    '''

    N = samples.shape[1]
    
    # print 'np.log(1./(N-1))', np.log((1./(N-1)) * np.sum(samples**2, axis=1, keepdims=True))[0]
    # if np.any(np.isnan(np.log((1./(N-1)) * np.sum(samples**2, axis=1, keepdims=True)))):
    #     print 'isnan'
    #     return np.zeros_like(np.log((1./(N-1)) * np.sum(samples**2, axis=1, keepdims=True)).shape)
    
    # else:
    return np.log((1./(N-1)) * np.sum(samples**2, axis=1, keepdims=True))


# dictionary mapping feature names to the corresponding functions defined above 
FEATURE_FUNCTIONS_DICT = {
    'MAV':  extract_MAV,
    'WAMP': extract_WAMP,
    'VAR':  extract_VAR,
    'WL':   extract_WL,
    'RMS':  extract_RMS,
    'ZC':   extract_ZC,
    'SSC':  extract_SSC,
    'LOGVAR':  extract_LOGVAR,
}


class EMGMultiFeatureExtractor(FeatureExtractor):
    '''
    Extract many different types of EMG features from raw EMG voltages
    '''
    feature_type = 'emg_multi_features'

    def __init__(self, source=None, emg_channels=[], feature_names=FEATURE_FUNCTIONS_DICT.keys(), feature_fn_kwargs={}, win_len=0.2, fs=1000):#, brainamp_channels = []):  
        '''
        Constructor for EMGMultiFeatureExtractor

        Parameters
        ----------
        source : MultiChanDataSource instance, optional, default=None
            DataSource interface to separate process responsible for collecting data from the EMG recording system
        channels : iterable of strings, optional, default=[]
            Names of channels from which to extract data
        feature_names : iterable, optional, default=[]
            Types of features to include in the extractor's output. See FEATURE_FUNCTIONS_DICT for available options
        feature_fn_kwargs : dict, optional, default={}
            Optional kwargs to pass to the individual feature extractors
        win_len : float, optional, default=0.2
            Length of time (in seconds) of raw EMG data to use for feature extraction 
        fs : float, optional, default=1000
            Sampling rate for the EMG data

        Returns
        -------
        EMGMultiFeatureExtractor instance
        '''

        self.source            = source
        # if not channels_filt:
        #     self.channels =  channels_filt
        #     print 'channels_filt'
        # else:
        #     self.channels = channels 
        #self.channels = self.extractor_kwargs['channels'] 
        #self.channels = brainamp_channel_lists.emg14_filt
        self.emg_channels = emg_channels
        #self.brainamp_channels = brainamp_channels
        self.feature_names     = feature_names
        self.feature_fn_kwargs = feature_fn_kwargs
        self.win_len           = win_len
        
        # for i in range(len(self.channels)-1):
        #     self.channels[i] = self.channels[i] + "_filt"

        if source is not None:
            self.fs = source.source.update_freq
        else:
            self.fs = fs
        # if self.channels == brainamp_channel_lists.emg_48hd_6mono_raw_filt:
        #     self.n_features = 46* len(feature_names) # 46 = 20 channels/high-density emg array + 6 bipolar emg channels
        # else:
        #     self.n_features = len(channels) * len(feature_names)

        self.n_features = len(emg_channels) * len(feature_names)
        self.feature_dtype = ('emg_multi_features', 'u4', self.n_features, 1)
        self.n_win_pts = int(self.win_len * self.fs)
        
        # Test if I can access all the data without loosing packets from here.
        # self.samples_all = [] #testing
        # self.t0 = time.time()
        # self.saved = True
        # self.timestamp = []

        # calculate coefficients for a 4th-order Butterworth BPF from 10-450 Hz
        if self.fs >= 1000: 
            band  = [10, 450]  # Hz
        else:
            band = [10, 200]
        nyq   = 0.5 * self.fs
        low   = band[0] / nyq
        high  = band[1] / nyq
        self.bpf_coeffs = butter(4, [low, high], btype='band')
        # self.band_pass_filter = Filter(self.bpf_coeffs[0], self.bpf_coeffs[1])

        # calculate coefficients for multiple 2nd-order notch filters
        

        if self.fs >= 1000: 
            notch_freqs  = [50, 150, 250, 350]  # Hz
        else:
            notch_freqs = [50, 150]

        self.notchf_coeffs = []
        for freq in notch_freqs:
            band  = [freq - 1, freq + 1]  # Hz
            nyq   = 0.5 * self.fs
            low   = band[0] / nyq
            high  = band[1] / nyq
            self.notchf_coeffs.append(butter(2, [low, high], btype='bandstop'))

        self.notch_filters = []
        for b, a in self.notchf_coeffs:
            self.notch_filters.append(Filter(b=b, a=a))

        n_channels = len(self.emg_channels)
        self.channel_filterbank = [None]*n_channels
        for k in range(n_channels):
            filts = [Filter(self.bpf_coeffs[0], self.bpf_coeffs[1])]
            for b, a in self.notchf_coeffs:
                filts.append(Filter(b=b, a=a))
            self.channel_filterbank[k] = filts


        # buffer to keep the last block of EMG activity. The filters will be applied to two consecutive blocks of EMG to avoid introducing artifacts due to the filter.

        # self.samples_buffer = RingBuffer(
        #     item_len=len(self.emg_channels),
        #     capacity=self.n_win_pts*2,
        # )
        
        
    def get_samples(self):
        '''
        Get samples from this extractor's MultiChanDataSource.

        Parameters
        ----------
        None 

        Returns
        -------
        Voltage samples of shape (n_channels, n_time_points)
        '''

        #import pdb; pdb.set_trace
        # self.new_samples = self.source.get(self.n_win_pts, self.emg_channels)['data']
        
        # #self.new_samples = self.source.get(self.n_win_pts, self.emg_channels)['data']
        # #self.samples_buffer.add_multiple_values(new_samples)
        # #return self.samples_buffer

        # return self.new_samples    
        #print self.channels
        #print self.emg_channels
        return self.source.get(self.n_win_pts, self.emg_channels)['data']


    def extract_filtered_samples(self, samples):
        '''
        Parameters
        ----------
        samples : np.ndarray of shape (n_channels, n_time_points)
            Raw EMG voltages from which to extract features

        Returns
        -------
        features : np.ndarray of shape (n_features, 1)
        '''
        
        # apply band-pass and notch filters separately to each channel
        
        for k in range(samples.shape[0]): #for loop on number of electrodes
            samples[k] = lfilter(self.bpf_coeffs[0],self.bpf_coeffs[1], samples[k]) 
            for b, a in self.notchf_coeffs:
                samples[k] = lfilter(b = b, a = a, x = samples[k]) 

        return samples


    def extract_features(self, samples):
        '''
        Parameters
        ----------
        samples : np.ndarray of shape (n_channels, n_time_points)
            Raw EMG voltages from which to extract features

        Returns
        -------
        features : np.ndarray of shape (n_features, 1)
        '''
        

        # apply band-pass and notch filters separately to each channel
        
        # for k in range(samples.shape[0]): #for loop on number of electrodes
        #     samples[k] = lfilter(self.bpf_coeffs[0],self.bpf_coeffs[1], samples[k]) 
        #     for b, a in self.notchf_coeffs:
        #         samples[k] = lfilter(b = b, a = a, x = samples[k]) 
        
        # extract actual features
        features = np.zeros((0, 1))
        for name in self.feature_names:
            fn = FEATURE_FUNCTIONS_DICT[name]
            if name in self.feature_fn_kwargs:
                kwargs = self.feature_fn_kwargs[name]
            else:
                kwargs = {}  # empty dictionary of kwargs
            new_features = fn(samples, **kwargs)
            features = np.vstack([features, new_features])
            
        
        return features.reshape(-1)

    def __call__(self):
        '''
        Get samples from this extractor's data source, filter them and extract features.
        '''
        # samples_buffer  = self.get_samples()
        # samples_emg  = samples_buffer.get_all()
        
        # samples_emg = self.get_samples()
        # if samples_emg.shape[1] > self.n_win_pts:

        #     samples_emg = samples_emg[:,self.n_win_pts:]
        
        # samples = self.extract_filtered_samples(samples_emg)
        #print self.source
        #from ismore.brainamp import rda
        #samples = self.source.get_data(self.source)
        #samples = rda.SimEMGData.get_data()
        samples = self.get_samples()
        if np.all(samples == 0):
            print 'all emg samples = 0'
        #print 'samples', samples

        # Test if I can access all the data without loosing packets from here.
        # if time.time() - self.t0 < 120:
        #     self.samples_all.append(samples) #testing 
        #     self.timestamp.append(time.time())
        #     #print 'time', time.time() - self.t0
        # elif self.saved == True:
        #     print "saving data"
        #     import scipy.io
        #     scipy.io.savemat('/home/tecnalia/test_48HD_6mono', mdict={'data': self.samples_all, 'timestamps': self.timestamp})   
        #     self.saved = False

   

        # if self.brainamp_channels == brainamp_channel_lists.emg_48hd_6mono_raw_filt:
        #     channels_str_2discard = [5,11,17,23,29,35,41,47,49,51,53,55,57]
        #     channels_str_2keep = [i for i in range(59) if i not in channels_str_2discard]
        #     samples_diff = np.diff(samples, axis = 0)
        #     samples = samples_diff[channels_str_2keep,:].copy
        
        features = self.extract_features(samples)
        # print 'features', features[-10:]
        # print 'isinf', np.any(np.isinf(features))
        if np.any(np.isinf(features)):
            # print 'features', features[-1]
            return dict(emg_multi_features=np.zeros_like(features.shape))
            
        else:
            return dict(emg_multi_features=features)


class ReplayEMGMultiFeatureExtractor(EMGMultiFeatureExtractor):
    '''
    Extract EMG features from EMG data stored in a file (instead of reading from the streaming input source)
    '''
    def __init__(self, hdf_table=None, cycle_rate=60., **kwargs):
        '''
        Parameters
        ----------
        hdf_table : HDF table
            Data table to replay, e.g., hdf.root.brainamp
        cycle_rate : float, optional, default=60.0
            Rate at which the task FSM "cycles", i.e., the rate at which the task will ask for new observations
        '''
        kwargs.pop('source', None)
        super(ReplayEMGMultiFeatureExtractor, self).__init__(source=None, **kwargs)
        self.hdf_table = hdf_table
        self.n_calls = 0
        self.cycle_rate = cycle_rate

    def get_samples(self):
        self.n_calls += 1
        table_idx = int(1./self.cycle_rate * self.n_calls * self.fs)
        table_idx = max(table_idx, 1)
        # import pdb; pdb.set_trace()
        start_idx = max(table_idx - self.n_win_pts, 0)
        if 0:
            print "self.emg_channels"
            print self.emg_channels
            for ch in self.emg_channels:
                print ch
                print self.hdf_table[:table_idx][ch]['data']
        samples = np.vstack([self.hdf_table[:table_idx][ch]['data'] for ch in self.emg_channels])
        return samples
