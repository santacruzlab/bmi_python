'''
Code for feature extraction methods/classes from EEG, to be used with a 
decoder (similar to other types of feature extractors in riglib.bmi.extractor)
'''

from collections import OrderedDict
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
import spectrum 
import time
import math
from riglib.filter import Filter
from copy import deepcopy
from riglib.bmi.extractor import FeatureExtractor
from utils.ringbuffer import RingBuffer
#from scipy.ndimage.filters import laplace
import nitime.algorithms as tsa
#from tools import nextpow2
#import statsmodels


# def levinson_durbin(s, nlags=10, isacov=False):
#     '''Levinson-Durbin recursion for autoregressive processes

#     Parameters
#     ----------
#     s : array_like
#         If isacov is False, then this is the time series. If iasacov is true
#         then this is interpreted as autocovariance starting with lag 0
#     nlags : integer
#         largest lag to include in recursion or order of the autoregressive
#         process
#     isacov : boolean
#         flag to indicate whether the first argument, s, contains the
#         autocovariances or the data series.

#     Returns
#     -------
#     sigma_v : float
#         estimate of the error variance ?
#     arcoefs : ndarray
#         estimate of the autoregressive coefficients
#     pacf : ndarray
#         partial autocorrelation function
#     sigma : ndarray
#         entire sigma array from intermediate result, last value is sigma_v
#     phi : ndarray
#         entire phi array from intermediate result, last column contains
#         autoregressive coefficients for AR(nlags) with a leading 1

#     Notes
#     -----
#     This function returns currently all results, but maybe we drop sigma and
#     phi from the returns.

#     If this function is called with the time series (isacov=False), then the
#     sample autocovariance function is calculated with the default options
#     (biased, no fft).
#     '''
#     s = np.asarray(s)
#     order = nlags  # rename compared to nitime
#     #from nitime

#     ##if sxx is not None and type(sxx) == np.ndarray:
#     ##    sxx_m = sxx[:order+1]
#     ##else:
#     ##    sxx_m = ut.autocov(s)[:order+1]
#     if isacov:
#         sxx_m = s
#     else:
#         sxx_m = acovf(s)[:order + 1]  # not tested

#     phi = np.zeros((order + 1, order + 1), 'd')
#     sig = np.zeros(order + 1)
#     # initial points for the recursion
#     phi[1, 1] = sxx_m[1] / sxx_m[0]
#     sig[1] = sxx_m[0] - phi[1, 1] * sxx_m[1]
#     for k in range(2, order + 1):
#         phi[k, k] = (sxx_m[k] - np.dot(phi[1:k, k-1],
#                                        sxx_m[1:k][::-1])) / sig[k-1]
#         for j in range(1, k):
#             phi[j, k] = phi[j, k-1] - phi[k, k] * phi[k-j, k-1]
#         sig[k] = sig[k-1] * (1 - phi[k, k]**2)

#     sigma_v = sig[-1]
#     arcoefs = phi[1:, -1]
#     pacf_ = np.diag(phi).copy()
#     pacf_[0] = 1.
#     return sigma_v, arcoefs, pacf_, sig, phi  # return everything


def extract_AR_psd(samples, freq_bands = []): #freq_bands (in Hz)
    '''
    Calculate the psd using the AR model coefficients for multiple channels. The psd is computed with a freq resolution = 1Hz. 
    For the channel selection we may want to keep this freq resolution but for the online procedure we may want to average certain frequency bands.

    Parameters
    ----------
    samples : np.ndarray of shape (n_time_points,) 
        Observed EEG voltages in microvolts

    Returns
    -------
    np.ndarray of shape (nfft) or if freq_bands != [] then PSD shape = (len(freq_bands))
    '''
    #compute the psd with a freq_resolution = 1Hz /frequency_resolution
    #a) aryule + arma2psd (spectrum library)
    #b) correlation + levinson (spectrum)
    order = 20
    fs = 1000 #sampling freq in Hz
    #print time.time()

    # # spectrum with levinson
    # t0 = time.time()
    # acf = np.correlate(samples, samples, 'full')
    # AC = acf[len(samples)-1:] #check length of the autoccoralation cuz levinson asks for a n+1 array being n the length of the signal.
    # AR = spectrum.LEVINSON(AC,order)
    # PSD = spectrum.arma2psd(AR[0], NFFT = fs)
    # #import pdb; pdb.set_trace()
    # PSD = PSD[len(PSD):len(PSD)/2-1:-1]

    # periodogram
    # _, PSD = tsa.periodogram(samples, fs)

    #print time.time() - t0

    #c) nitime.algorithms.autoregressive.AR_est_YW + nitime.algorithms.autoregressive.AR_psd (nitime)
    #t0 = time.time()
    # AR_coeffs, sigma_v = tsa.autoregressive.AR_est_YW(samples, order)
    # # sigma_v, AR_coeffs, _, _, _ = statsmodels.tsa.stattools.levinson_durbin(samples, order= order, isacov=False)
    # n_freqs = fs    
    # norm_freqs, AR_psd = tsa.autoregressive.AR_psd (AR_coeffs, sigma_v, n_freqs, sides = 'onesided')
    
    # # n_freqs = tools.nextpow2(len(samples))
    # n_freqs = fs
    # norm_freqs, AR_psd = tsa.autoregressive.AR_psd (AR_coeffs, sigma_v, n_freqs, sides = 'onesided')
    
    # #d) statsmodels?
    # AR_coeffs, sigma_v = levinson_durbin(samples, order, isacov = False)

    # nitime using levinson durbin method
    AR_coeffs_LD, sigma_v = tsa.autoregressive.AR_est_LD(samples, order)
    n_freqs = fs    
    norm_freqs, PSD = tsa.autoregressive.AR_psd (AR_coeffs_LD, sigma_v, n_freqs, sides = 'onesided')
    PSD = PSD[1:] #get rid of PSD in freq = 0
    PSD = np.log(PSD) #compute the Log of the PSD
    #print 'time taken', time.time() - t0
    #import pdb; pdb.set_trace()
    if freq_bands != []:
        for low, high in freq_bands:
            try:
                PSD_mean = np.hstack([PSD_mean, np.mean(PSD[low-1:high-1])])
            except NameError:
                PSD_mean = np.mean(PSD[low-1:high-1])
        PSD = PSD_mean
         #to do, compute psd of frequency_resolution Hz freq bins
    # if the input argument frequency_resolution != 1, then compute the average of the frequency bins determined

    #compute the power in the desired freq bands and return these values
    #import pdb; pdb.set_trace()
    #return PSD
    return PSD

def extract_MTM_psd(samples, NW = 3): #frequency_resoltuions (in Hz)
    '''
    Extract spectral features from a block of time series samples

        Parameters
        ----------
        cont_samples : np.ndarray of shape (n_channels, n_samples)
            Raw voltage time series (one per channel) from which to extract spectral features 

        Returns
        -------
        EEG_power : np.ndarray of shape (n_channels * n_features, 1)
            Multi-band power estimates for each channel, for each band specified when the feature extractor was instantiated.
    '''
    pass
    
    #compute the psd with a freq_resolution = 1Hz /frequency_resolution
   
    # multitaper psd (already implemented in extractor.py)
    
    # if the input argument frequency_resolution != 1, then compute the average of the frequency bins determined


     


# dictionary mapping feature names to the corresponding functions defined above 
FEATURE_FUNCTIONS_DICT = {
    'AR':  extract_AR_psd,
    'MTM': extract_MTM_psd,
}

# NEIGHBOUR_CHANNELS_DICT = { #define the neighbour channels for each channel (for the Laplacian filter)
#     '1':  [2,3,4],
#     '2':  [5,6],
#     '3':  [4,5],
# }
NEIGHBOUR_CHANNELS_DICT = {}

class EEGMultiFeatureExtractor(FeatureExtractor):
    '''
    Extract different types of EEG features from raw EEG voltages
    '''
    feature_type = 'eeg_multi_features'

    def __init__(self, source=None, channels_2train = [], channels=[], feature_names=FEATURE_FUNCTIONS_DICT.keys(), feature_fn_kwargs={}, win_len=0.5, fs=1000, neighbour_channels=NEIGHBOUR_CHANNELS_DICT, brainamp_channels = []):  
        '''
        Constructor for EEGMultiFeatureExtractor

        Parameters
        ----------
        source : MultiChanDataSource instance, optional, default=None
            DataSource interface to separate process responsible for collecting data from the EEG recording system
        channels : iterable of strings, optional, default=[]
            Names of channels from which to extract data
        feature_names : iterable, optional, default=[]
            Types of features to include in the extractor's output. See FEATURE_FUNCTIONS_DICT for available options
        feature_fn_kwargs : dict, optional, default={}
            Optional kwargs to pass to the individual feature extractors
        win_len : float, optional, default=0.2
            Length of time (in seconds) of raw EEG data to use for feature extraction 
        fs : float, optional, default=1000
            Sampling rate for the EEG data

        Returns
        -------
        EEGMultiFeatureExtractor instance

        '''

        self.source             = source
        self.channels           = channels #channels to use for the deocding online
        self.brainamp_channels  = brainamp_channels # all the channels being recorded from the brainamp source and stored in the hdf file (raw+filt)
        self.feature_names      = feature_names
        self.feature_fn_kwargs  = feature_fn_kwargs
        self.win_len            = win_len
        self.neighbour_channels = neighbour_channels
        self.channels_2train    = channels_2train


        if source is not None:
            self.fs = source.source.update_freq
        else:
            self.fs = fs

        #import pdb; pdb.set_trace()
        if channels_2train != []:
            if 'chan' in self.channels_2train[0]:
                self.n_features = np.sum([len(self.feature_fn_kwargs[self.feature_names[0]]['freq_bands'][i[4:]]) for i in self.channels_2train])
            else:
                self.n_features = np.sum([len(self.feature_fn_kwargs[self.feature_names[0]]['freq_bands'][i]) for i in self.channels_2train])
        #self.n_features = len(self.channels_2train) * len(feature_names) #* len(self.feature_fn_kwargs['self.feature_names'])
        #self.feature_dtype = ('eeg_multi_features', 'u4', self.n_features, 1)

        self.n_win_pts = int(self.win_len * self.fs)

        # calculate coefficients for a 4th-order Butterworth BPF from 1-50 Hz
        # band  = [1, 100]  # Hz
        # nyq   = 0.5 * self.fs
        # low   = band[0] / nyq
        # high  = band[1] / nyq
        # self.bpf_coeffs = butter(4, [low, high], btype='band')

        # band  = [49,51]  # Hz
        # nyq   = 0.5 * self.fs
        # low   = band[0] / nyq
        # high  = band[1] / nyq
        # self.notchf_coeffs = butter(2, [low, high], btype='bandstop')
        # self.band_pass_filter = Filter(self.bpf_coeffs[0], self.bpf_coeffs[1])

        

        n_channels = len(self.channels)
        # self.channel_filterbank = [None]*n_channels
        # for k in range(n_channels):
        #     filts = [Filter(self.bpf_coeffs[0], self.bpf_coeffs[1])]
        #     self.channel_filterbank[k] = filts

        # buffer to keep the last block of EEG activity. The filters will be applied to two consecutive blocks of EEG to avoid introducing artifacts due to the filter.

        # self.samples_buffer = RingBuffer(
        #     item_len=len(self.channels),
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
        # samples = dict()
        # #samples[self.channels] = []
        # for name in self.channels:
        #     print 'name'
        #     print name
        #     samples[name] = self.source.get(self.n_win_pts, name)['data']
        
        # return samples
        return self.source.get(self.n_win_pts, self.channels)['data']

        
    def Laplacian_filter(self, samples):
        '''
        Parameters
        ----------
        samples : np.ndarray of shape (n_channels, n_time_points)
            Filtered (with BPfilter) EEG voltages. Laplacian filter will be applied to those

        Returns
        -------
        samples : np.ndarray of shape (n_channels, n_time_points)
        '''       
        
        #samples_copy = deepcopy(samples)
        samples_copy = samples.copy()
        # apply Laplacian spatial filter to each channel separately
        for k, neighbours in enumerate(self.neighbour_channels): #for loop on number of electrodes
            samples_laplace = samples_copy[neighbours].copy()
            for n in range(len(self.neighbour_channels[neighbours])):
                samples_laplace = np.vstack([samples_laplace, samples_copy[self.neighbour_channels[neighbours][n]]]) 
            samples[neighbours]['data'] = samples_laplace[0,:]['data'] - np.mean(samples_laplace[1:,:]['data'], axis = 0)

        return samples

    def Laplacian_filter_online(self, samples):
        '''
        Parameters
        ----------
        samples : np.ndarray of shape (n_channels, n_time_points)
            Filtered (with BPfilter) EEG voltages. Laplacian filter will be applied to those

        Returns
        -------
        samples : np.ndarray of shape (n_channels, n_time_points)
        '''       

        #samples_copy = deepcopy(samples)
        samples_copy = samples.copy()
        # apply Laplacian spatial filter to each channel separately
        for k, neighbours in enumerate(self.neighbour_channels): #for loop on number of electrodes
            #import pdb; pdb.set_trace()
            #index = neighbours.index('_')
            channel_index_in_samples = np.where(self.channels == neighbours)
            samples_laplace = samples_copy[channel_index_in_samples,:].copy()
            #samples_laplace = samples_copy[int(neighbours[:index])-1,:].copy()
            for n in range(len(self.neighbour_channels[neighbours])):
                channel_index_in_samples2 = np.where(self.channels == self.neighbour_channels[neighbours][n])
                #index = self.neighbour_channels[neighbours][n].index('_')
                # samples_laplace = np.vstack([samples_laplace, samples_copy[int(self.neighbour_channels[neighbours][n][:index])-1,:]]) 
                samples_laplace = np.vstack([samples_laplace, samples_copy[channel_index_in_samples2,:]]) 
            # samples[int(neighbours[:index])-1,:] = samples_laplace[0,:] - np.mean(samples_laplace[1:,:], axis = 0)
            samples[channel_index_in_samples,:] = samples_laplace[0,:] - np.mean(samples_laplace[1:,:], axis = 0)
        
        return samples


    def extract_features(self, samples,chan_freq):
        '''
        Parameters
        ----------
        samples : np.ndarray of shape (n_channels, n_time_points)
            Raw EEG voltages from which to extract features

        Returns
        -------
        features : np.ndarray of shape (n_features, 1)
        '''
        
        # apply band-pass separately to each channel
        
        # for k in range(samples.shape[0]): #for loop on number of electrodes
        #     samples[k] = lfilter(self.bpf_coeffs[0],self.bpf_coeffs[1], samples[k]) 
        
        # extract actual features
        features = None
        for name in self.feature_names:
            fn = FEATURE_FUNCTIONS_DICT[name]
            if name in self.feature_fn_kwargs:
                kwargs = self.feature_fn_kwargs[name]
                #import pdb; pdb.set_trace()
                # try:
                #import pdb; pdb.set_trace()
                # except KeyError
                if kwargs['freq_bands'] != dict():
                    #import pdb; pdb.set_trace()
                    freq_band = kwargs['freq_bands'][chan_freq]#changed
                    #freq_band = []
                else:
                    freq_band = []
            else:
                kwargs = {}  # empty dictionary of kwargs
            #new_features = fn(samples, **kwargs)
            new_features = fn(samples, freq_band)#changed
            
            if features == None:
                features = new_features
            else:
                features = np.vstack([features, new_features])
            
        return features
        #return features.reshape(-1)

    def __call__(self):
        '''
        Get samples from this extractor's data source, filter them and extract features. Used for the online decoding
        '''
        samples = self.get_samples()
        # print 'samples'
        #print samples
        
        # if np.all(samples == 0):
        #     print 'all samples = 0'
        #     samples = np.random.rand(samples.shape[0],samples.shape[1])

        samples = self.Laplacian_filter_online(samples)
        features = None

        for k, chan_freq in enumerate(self.channels_2train):# loop on channels
            #import pdb; pdb.set_trace()
            features_chan = self.extract_features(samples[k],chan_freq)
            if features == None:
                features = features_chan 
            else:
                features = np.hstack([features, features_chan])
        #return dict(eeg_multi_features=features)
        return features

    def sim_call_rest(self):

        # Gererate artificial rest data
        fsample = 1000.00 #Sample frequency in Hz
        f = 10 # in Hz
        rest_amp = 20
        t = np.arange(0, self.n_win_pts/fsample , 1/fsample)
        cnt = 1
        cnt_noise = 1
        samples = []
        #rest_signal_2test = []
        #print self.channels
        for k in self.channels: #for loop on number of electrodes
            #print 'channel in sim rest', k
            if k in ['8_filt', '9_filt', '13_filt', '14_filt', '18_filt', '19_filt']:
            #if k in ['13_filt', '14_filt', '8_filt', '19_filt']:
                rest_noise = rest_amp*0.1*np.random.randn(self.n_win_pts) #10% of signal amplitude
                rest_signal = np.zeros(self.n_win_pts)
                #rest_signal_2test = np.zeros(self.n_win_pts)
                
                for i in np.arange(self.n_win_pts):
                    rest_signal[i] = (rest_amp + cnt) * math.sin((f+cnt-1)*2*math.pi*t[i]) + rest_noise[i] #rest sinusoidal signal
                cnt += 1
                # rest_signal_2test = rest_signal.copy()

            else:
                #rest_signal_2test = np.zeros(self.n_win_pts)
                rest_signal = (rest_amp + cnt_noise)*0.1*np.random.randn(self.n_win_pts) #10% of signal amplitude. only noise 
                cnt_noise += 1
               
            if samples == []:
                samples = rest_signal.copy()
            else:
                samples = np.vstack([samples, rest_signal.copy()])   
            #samples = np.vstack([samples, rest_signal])   
            #samples[k] = rest_signal
            
            # if k in ['12_filt','14_filt', '18_filt','19_filt', '23_filt']:
            #     if rest_signal_2test == []:
            #         # print k
            #         # print 'len1', len(rest_signal)
                    
            #         rest_signal_2test = rest_signal.copy()
            #     else:
            #         # print k
            #         # #print rest_signal
            #         # print 'len2', rest_signal_2test.shape
            #         #print 'len3'. len(rest_signal)
            #         rest_signal_2test = np.vstack([rest_signal_2test, rest_signal.copy()])  
        # print 'samples'
        #print samples
        #import pdb; pdb.set_trace()
        # if np.all(samples == 0):
        #     print 'all samples = 0'
        #     samples = np.random.rand(samples.shape[0],samples.shape[1])
        
        samples = self.Laplacian_filter_online(samples)
        features = None

        for k, chan_freq in enumerate(self.channels_2train):# loop on channels
            features_chan = self.extract_features(samples[k],chan_freq)
            if features == None:
                features = features_chan 
            else:
                features = np.hstack([features, features_chan])
        #import pdb; pdb.set_trace()
        #return dict(eeg_multi_features=features)
        #return features, rest_signal_2test
        return features

    def sim_call_mov(self):

        # Gererate artificial mov data

        fsample = 1000.00 #Sample frequency in Hz
        f = 10 # in Hz
        rest_amp = 20
        move_amp = 5; #mov state amplitude 
        t = np.arange(0, self.n_win_pts/fsample , 1/fsample)
        cnt = 1
        cnt_noise = 1
        #samples = dict()
        samples= []
        #move_signal_2test = []
        for k in self.channels: #for loop on number of electrodes
            if k in ['8_filt', '9_filt', '13_filt', '14_filt', '18_filt', '19_filt']:
            #if k in ['13_filt', '14_filt', '8_filt', '19_filt']:
                
                move_noise = move_amp*0.1*np.random.randn(self.n_win_pts) #10% of signal amplitude
                move_signal = np.zeros(self.n_win_pts)
                #move_signal_2test = np.zeros(self.n_win_pts)
                for i in np.arange(self.n_win_pts):
                    
                    move_signal[i] = (move_amp + cnt) * math.sin((f+cnt-1)*2*math.pi*t[i]) + move_noise[i]
                cnt += 1
                #move_signal_2test = move_signal
            else:
                #move_signal_2test = np.zeros(self.n_win_pts)
                move_signal = (rest_amp + cnt_noise)*0.1*np.random.randn(self.n_win_pts) #10% of signal amplitude
                cnt_noise += 1
               
            if samples == []:
                samples = move_signal
            else:
                samples = np.vstack([samples, move_signal]) 

            # if k in ['14_filt','15_filt', '17_filt','18_filt', '19_filt']:
            #     if move_signal_2test == []:
            #         move_signal_2test = move_signal.copy()
            #     else:
            #         move_signal_2test = np.vstack([move_signal_2test, move_signal])        
            #samples[k] = mov_signal
            #move_signal_2test = samples.copy()
        samples = self.Laplacian_filter_online(samples)
        features = None

        for k, chan_freq in enumerate(self.channels_2train):# loop on channels
            features_chan = self.extract_features(samples[k],chan_freq)
            if features == None:
                features = features_chan 
            else:
                features = np.hstack([features, features_chan])
        #import pdb; pdb.set_trace()
        #return dict(eeg_multi_features=features)
        #return features, move_signal_2test
        return features

    def extract_features_2retrain(self, rest_data, mov_data):
        '''
        Filter samples and extract features. Used for the online retraining of the decoder 
        '''

        rest_data = self.Laplacian_filter_online(rest_data)
        mov_data = self.Laplacian_filter_online(mov_data)


        rest_features = None
        mov_features = None
        min_len = min(rest_data.shape[1],mov_data.shape[1])
        rest_data = rest_data[:,rest_data.shape[1]-min_len:]
        mov_data = mov_data[:,mov_data.shape[1]-min_len:]
        t0 = time.time()
        for k, chan_freq in enumerate(self.channels_2train):# for loop on channels #THIS HAS TO BE FIXED!!!! THE DATA FROM BRAINAMP MIGHT NOT BE IN ORDER ACCORDING TO OUR LIST OF CHANNELS
            n = 0
            r_features_ch = None
            m_features_ch = None
            while n <= (rest_data.shape[1] - 500):
                r_feats = self.extract_features(rest_data[k,n:n+500],chan_freq)                        
                m_feats = self.extract_features(mov_data[k,n:n+500],chan_freq)
                if r_features_ch == None:
                    r_features_ch = r_feats
                    m_features_ch = m_feats
                else:
                    r_features_ch = np.vstack([r_features_ch, r_feats])
                    m_features_ch = np.vstack([m_features_ch, m_feats])
                n +=50
            #rest_features_chan = self.extract_features(rest_data[k])
            #mov_features_chan = self.extract_features(mov_data[k])

            if rest_features == None:
                rest_features = r_features_ch
                mov_features = m_features_ch
            else:
                rest_features = np.hstack([rest_features, r_features_ch])
                mov_features = np.hstack([mov_features, m_features_ch])
        # print 'features'
        # print time.time() - t0
        #return dict(rest_features=rest_features, mov_features = mov_features)
        return rest_features, mov_features


