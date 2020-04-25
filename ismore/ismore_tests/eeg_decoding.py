import tables
import os
import numpy as np
import pandas as pd
import math
import pdb # pdb.set_trace() Kind of the same thing as keyboard function in matlab
import sklearn
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.signal import butter, lfilter, filtfilt
from ismore.common_state_lists import *
from ismore import ismore_bmi_lib
from riglib.filter import Filter
from utils.ringbuffer import RingBuffer
try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
except:
    from sklearn.lda import LDA


class EEGDecoderBase(object):
    '''
    Abstract base class for all concrete EEG decoder classes
    '''
    pass


class LinearEEGDecoder(EEGDecoderBase):
    '''Concrete base class for a linear EEG decoder.'''

    def __init__(self, channels_2train, plant_type, fs, win_len, buffer_len, filt_training_data, extractor_cls, extractor_kwargs, trial_hand_side):
        self.channels_2train= channels_2train #Channels that will be used to train the LDA (without the ones used for the Laplacian filter)
        self.channels_2train_buffers = channels_2train
        self.plant_type     = plant_type
        self.fs             = fs
        self.win_len        = win_len
        self.buffer_len     = buffer_len
        self.filt_training_data = filt_training_data
        
        self.feature_names = extractor_kwargs['feature_names']
        self.trial_hand_side = trial_hand_side
        self.feature_fn_kwargs = extractor_kwargs['feature_fn_kwargs']
        self.brainamp_channels = extractor_kwargs['brainamp_channels']
        neighbours = extractor_kwargs['neighbour_channels']
 
        self.channel_names = extractor_kwargs['channels']
        self.neighbour_channels = dict()
        for chan_neighbour in neighbours:#Add the ones used for the Laplacian filter here
            self.neighbour_channels['chan' + chan_neighbour] = []
            for k,chans in enumerate(neighbours[chan_neighbour]):
                if chans not in self.channel_names:
                    self.channel_names.append(chans)
                new_channel = 'chan' + chans
                self.neighbour_channels['chan' + chan_neighbour].append(new_channel)
             

        # channel names in HDF file have 'chan' in front (e.g., 'chanBiceps'
        self.channel_names = ['chan' + name for name in self.channel_names]
        self.channels_2train = ['chan' + name for name in self.channels_2train]

        #self.n_features = len(self.channels_2train) * len(self.feature_fn_kwargs[self.feature_names[0]]['freq_bands']) # len(self.eeg_decoder.freq_bands)
        # # even if it is called states_to_decode in this case only MOV vs REST will be decoded and then a predefined movement will be triggered
        # ssm = ismore_bmi_lib.SSM_CLS_DICT[plant_type]()
        # self.states_to_decode = [s.name for s in ssm.states if s.order == 1]

        self.extractor_cls = extractor_cls
        self.extractor_kwargs = extractor_kwargs

        # Old way of buffering data to retrain decoder
        self.rest_data_buffer = RingBuffer(
            item_len=len(self.channel_names),
            capacity=self.buffer_len * self.fs,
        )
        
        self.mov_data_buffer = RingBuffer(
            item_len=len(self.channel_names),
            capacity=self.buffer_len * self.fs,
        )

        # New way of buffering data to retrain decoder
        # self.rest_data_buffer = RingBuffer(
        #     item_len=len(self.channel_names),
        #     capacity=self.buffer_len * self.fs,
        # )
        
        # self.mov_data_buffer = RingBuffer(
        #     item_len=len(self.channel_names),
        #     capacity=self.buffer_len * self.fs,
        # )
        #For the simulation we keep the old way of buffering data but these two need to be defined to avoid errors due to inheritances even if they are empty
        self.rest_feature_buffer = RingBuffer(
            item_len=len(self.channels_2train),
            capacity=2390# 120(2min) * 1000Hz = self.win_len + n*50 (50ms if task runs at 20Hz and features are computed offline with a sliding window of 50ms)# self.buffer_len * self.fs,
        )#n = 2390

        self.mov_feature_buffer = RingBuffer(
            item_len=len(self.channels_2train),
            capacity=2390# same as above. self.buffer_len * self.fs,
        )

    def __call__(self, features):
        #TO DO! I might need to define extra functions if we wanna call this script from ismorestasks.py to predict mov/rest
        #decoder_output = pd.Series(0.0, 1)
        decoder_output = self.decoder.predict(features)
        return decoder_output

    def train_LDA(self, train_hdf_names, test_hdf_names):
        '''Use LDA to train this decoder from data from multiple .hdf files.'''

        # save this info as part of the decoder object
        #self.K               = K # include here the params for the SVM
        self.train_hdf_names = train_hdf_names
        self.test_hdf_names  = test_hdf_names

        # calculate coefficients for a 4th-order Butterworth BPF from 10-450 Hz
        band  = [0.5, 80]  # Hz
        nyq   = 0.5 * self.fs
        low   = band[0] / nyq
        high  = band[1] / nyq
        self.bpf_coeffs = butter(4, [low, high], btype='band')


        band  = [48,52]  # Hz
        nyq   = 0.5 * self.fs
        low   = band[0] / nyq
        high  = band[1] / nyq
        self.notchf_coeffs = butter(2, [low, high], btype='bandstop')

        f_extractor = self.extractor_cls(None, channels_2train = self.channels_2train, channels = self.channel_names, feature_names = self.feature_names, feature_fn_kwargs = self.feature_fn_kwargs, win_len=self.win_len, fs=self.fs, neighbour_channels = self.neighbour_channels, brainamp_channels = self.brainamp_channels)
        #f_extractor = self.extractor_cls(None, **self.extractor_kwargs)
        
        all_hdf_names = train_hdf_names + [name for name in test_hdf_names if name not in train_hdf_names]
        
        self.hdf_data_4buffer = train_hdf_names#train_hdf_names[-2:]

        self.features_train = None
        self.labels_train = None
        self.features_test = None
        self.labels_test = None

        for hdf_name in all_hdf_names:
            # load EMG data from HDF file
            hdf = tables.openFile(hdf_name)

            store_dir_supp = '/storage/supp_hdf/'
            index_slash = hdf_name.encode('ascii','ignore').rfind('/')            
            hdf_supp_name = store_dir_supp + hdf_name.encode('ascii','ignore')[index_slash + 1:-3] + 'supp.hdf'
            
            
            try:
                hdf_supp = tables.open_file(hdf_supp_name)
                eeg = hdf_supp.root.brainamp[:][self.channel_names]
            except:
                eeg = hdf.root.brainamp[:][self.channel_names]
            
            
            #eeg = hdf.root.brainamp[:][self.channel_names]
            len_file = len(hdf.root.brainamp[:][self.channel_names[0]])
            # Parameters to generate artificial EEG data
            fsample = 1000.00 #Sample frequency in Hz
            t = np.arange(0, 10 , 1/fsample)#[0 :1/fsample:10-1/fsample]; #time frames of 10seg for each state
            time = np.arange(0, len_file/fsample, 1/fsample) #time vector for 5min
            f = 10 # in Hz
            rest_amp = 20
            move_amp = 5; #mov state amplitude 

            # cues = hdf.root.task_msgs[:]['msg']
            # cues_trial_type = hdf.root.task[:]['trial_type']
            # cues_events = hdf.root.task_msgs[:]['time']
            # cues_times = hdf.root.task[:]['ts']

            # n_win_pts = int(self.win_len * self.fs)
            
            # step_pts = 50  # TODO -- don't hardcode This is in sample points! Be careful if fps is changed!!

            # original_ts = eeg[self.channel_names[0]]['ts_arrival']
            # idx = 1            
            # while original_ts[idx] == original_ts[0]:
            #     idx = idx + 1
  
            # ts_step = 1./self.fs
            # ts_eeg = np.arange(original_ts[0]-(ts_step*(idx-1)),original_ts[len(original_ts)-1],ts_step)

            # Generate artificial eeg data
            cnt = 1
            cnt_noise = 1
            for k in range(len(self.channel_names)): #for loop on number of electrodes
                if self.channel_names[k] in ['chan8_filt', 'chan9_filt', 'chan13_filt', 'chan14_filt', 'chan18_filt', 'chan19_filt']:
                    rest_noise = rest_amp*0.1*np.random.randn(len(t)) #10% of signal amplitude
                    rest_signal = np.zeros(len(t))
                    move_noise = move_amp*0.1*np.random.randn(len(t)) #10% of signal amplitude
                    move_signal = np.zeros(len(t))
                    for i in np.arange(len(t)):
                        rest_signal[i] = (rest_amp+cnt) * math.sin((f+cnt-1)*2*math.pi*t[i]) + rest_noise[i] #rest sinusoidal signal
                        move_signal[i] = (move_amp+cnt) * math.sin((f+cnt-1)*2*math.pi*t[i]) + move_noise[i]
                    cnt += 1
                    signal = []
                    # label = []
                    for i in np.arange(30):
                        signal = np.hstack([signal, rest_signal, move_signal])
                        # label = np.hstack([label, np.ones([len(rest_signal)]), np.zeros([len(move_signal)])])
            
                else:
                    rest_signal = (rest_amp + cnt_noise)*0.1*np.random.randn(len(t)) #10% of signal amplitude. only noise 
                    move_signal = (rest_amp + cnt_noise)*0.1*np.random.randn(len(t)) #10% of signal amplitude
                    cnt_noise += 1
                    signal = []
                    # label = []
                    for i in np.arange(30):
                        signal = np.hstack([signal, rest_signal, move_signal])
        
        
                eeg[self.channel_names[k]]['data'] = signal[:len_file].copy()    
                eeg[self.channel_names[k]]['data'] = lfilter(self.bpf_coeffs[0],self.bpf_coeffs[1], eeg[self.channel_names[k]]['data']) 
                eeg[self.channel_names[k]]['data'] = lfilter(self.notchf_coeffs[0],self.notchf_coeffs[1], eeg[self.channel_names[k]]['data']) 
            
            #steps to follow to train the decoder


            #2.- Break done the signal into trial intervals (concatenate relax and right trials as they happen in the exp)
            
            # rest_start_events = cues_events[np.where(cues == 'trial')][np.where(cues_trial_type[cues_events[np.where(cues == 'trial')]] == 'relax')]
            # rest_start_times = cues_times[rest_start_events]
            # mov_start_events = cues_events[np.where(cues == 'trial')][np.where(cues_trial_type[cues_events[np.where(cues == 'trial')]] == 'right')]
            # mov_start_times = cues_times[mov_start_events]
            # rest_end_events = cues_events[np.where(cues == 'wait')][1:][np.where(cues_trial_type[cues_events[np.where(cues ==  'trial')]]== 'relax')]
            # if  rest_end_events[-1] == len(cues_times):
            #     rest_end_events[-1] = rest_end_events[-1]-1
            # rest_end_times = cues_times[rest_end_events]
            # mov_end_events = cues_events[np.where(cues == 'wait')][1:][np.where(cues_trial_type[cues_events[np.where(cues ==  'trial')]]== 'right')]
            # if  mov_end_events[-1] == len(cues_times):
            #     mov_end_events[-1] = mov_end_events[-1]-1
            # mov_end_times = cues_times[mov_end_events]
            
            # trial_type_labels = np.vstack([np.zeros(len(rest_start_times))
            # trial_type_labels[where(cues_trial_type[trial_start_events] == 'Right')] = 1

            # build time vectors relative to first time stamp
            # t_min = min(ts_eeg[0], trial_start_times[0])
            # ts_eeg = ts_eeg - t_min
            # trial_start_times = trial_start_times - t_min
            # trial_end_times = trial_end_times - t_min

            # The amount and length of REST intervals and MOVEMENT intervals with the paretic hand should be the same
            # rest_start_idxs_eeg = np.zeros(len(rest_start_times))
            # mov_start_idxs_eeg = np.zeros(len(mov_start_times))
            # rest_end_idxs_eeg = np.zeros(len(rest_end_times))
            # mov_end_idxs_eeg = np.zeros(len(mov_end_times))
            # for idx in range(len(rest_start_times)):
            #     # find indexes of emg signal corresponding to start and end of rest/mov periods
            #     time_dif = [ts_eeg - rest_start_times[idx]]
            #     bool_idx = time_dif == min(abs(time_dif[0]))
            #     if np.all(bool_idx == False):
            #         bool_idx = time_dif == -1*min(abs(time_dif[0]))
            #     rest_start_idxs_eeg[idx] = np.where(bool_idx[0])[0]

            #     time_dif = [ts_eeg - rest_end_times[idx]]
            #     bool_idx = time_dif == min(abs(time_dif[0]))
            #     if np.all(bool_idx == False):
            #         bool_idx = time_dif == -1*min(abs(time_dif[0]))
            #     rest_end_idxs_eeg[idx] = np.where(bool_idx[0])[0]

            #     time_dif = [ts_eeg - mov_start_times[idx]]
            #     bool_idx = time_dif == min(abs(time_dif[0]))
            #     if np.all(bool_idx == False):
            #         bool_idx = time_dif == -1*min(abs(time_dif[0]))
            #     mov_start_idxs_eeg[idx] = np.where(bool_idx[0])[0]

            #     time_dif = [ts_eeg - mov_end_times[idx]]
            #     bool_idx = time_dif == min(abs(time_dif[0]))
            #     if np.all(bool_idx == False):
            #         bool_idx = time_dif == -1*min(abs(time_dif[0]))
            #     mov_end_idxs_eeg[idx] = np.where(bool_idx[0])[0]
            rest_start_idxs_eeg = np.arange(0,len(time), fsample *20)
            mov_start_idxs_eeg = np.arange(10*1000,len(time), fsample *20)
            rest_end_idxs_eeg = np.arange(10*1000-1,len(time), fsample *20) 
            mov_end_idxs_eeg = np.arange(20*1000-1,len(time), fsample *20) #+1
            #import pdb; pdb.set_trace()
            if len(mov_end_idxs_eeg) < len(rest_end_idxs_eeg) or len(mov_start_idxs_eeg) < len(rest_start_idxs_eeg):
                rest_end_idxs_eeg = rest_end_idxs_eeg[:len(mov_end_idxs_eeg)]
                rest_start_idxs_eeg = rest_start_idxs_eeg[:len(mov_end_idxs_eeg)]
                mov_start_idxs_eeg = mov_start_idxs_eeg[:len(mov_end_idxs_eeg)]
            #import pdb; pdb.set_trace()
            if hdf_name in self.hdf_data_4buffer:
                n_trials = len(rest_end_idxs_eeg) # number of trials to store in the data buffers
                eeg_data_buffer = None
                for chan in self.channel_names:
                    if eeg_data_buffer == None:
                        eeg_data_buffer = eeg[chan]['data']
                    else:
                        eeg_data_buffer = np.vstack([eeg_data_buffer, eeg[chan]['data']])
                    #import pdb; pdb.set_trace()
                for kk in range(n_trials):
                    self.rest_data_buffer.add_multiple_values(eeg_data_buffer[:,rest_start_idxs_eeg[kk]:rest_end_idxs_eeg[kk]])
                    self.mov_data_buffer.add_multiple_values(eeg_data_buffer[:,mov_start_idxs_eeg[kk]:mov_end_idxs_eeg[kk]])
            # Laplacian filter -  this has to be applied always, independently of using raw or filt data
            eeg = f_extractor.Laplacian_filter(eeg)

            self.features_train_file = None
            self.features_test_file = None
            for k in self.channels_2train: 
                r_features_ch = None
                m_features_ch = None
                #k = 'chan13_filt'
                # chan_n = 7
                for idx in range(len(rest_start_idxs_eeg)):
                    rest_window = eeg[k]['data'][rest_start_idxs_eeg[idx]:rest_end_idxs_eeg[idx]]
                    mov_window = eeg[k]['data'][mov_start_idxs_eeg[idx]:mov_end_idxs_eeg[idx]]     
                    found_index = k.find('n') + 1
                    chan_freq = k[found_index:]
                    #3.- Take windows of 500ms every 50ms (i.e. overlap of 450ms)
                    #4.- Extract features (AR-psd) of each of these windows 
                    n = 0
                    while n <= (len(rest_window) - 500) and n <= (len(mov_window) - 500):
                        r_feats = f_extractor.extract_features(rest_window[n:n+500],chan_freq)                        
                        m_feats = f_extractor.extract_features(mov_window[n:n+500],chan_freq)
                        if r_features_ch == None:
                            r_features_ch = r_feats
                            m_features_ch = m_feats
                        else:
                            r_features_ch = np.vstack([r_features_ch, r_feats])
                            m_features_ch = np.vstack([m_features_ch, m_feats])
                        n +=50
                if hdf_name in train_hdf_names:
                    if  self.features_train_file == None:
                        self.features_train_file = np.vstack([r_features_ch, m_features_ch])
                    else:
                        self.features_train_file = np.hstack([self.features_train_file, np.vstack([r_features_ch, m_features_ch])])
                        
                else:
                    if  self.features_test_file == None:
                        self.features_test_file = np.vstack([r_features_ch, m_features_ch])
                    else:
                        self.features_test_file = np.hstack([self.features_test_file, np.vstack([r_features_ch, m_features_ch])])
            
            if hdf_name not in train_hdf_names:
                self.labels_test_file = np.vstack([np.zeros([r_features_ch.shape[0],1]), np.ones([m_features_ch.shape[0],1])])
                if self.features_test == None:
                    self.features_test = self.features_test_file
                    self.labels_test = self.labels_test_file
                else:
                    self.features_test = np.vstack([self.features_test, self.features_test_file])
                    self.labels_test = np.vstack([self.labels_test, self.labels_test_file])
            else:
                self.labels_train_file = np.vstack([np.zeros([r_features_ch.shape[0],1]), np.ones([m_features_ch.shape[0],1])])
                if self.features_train == None:
                    self.features_train = self.features_train_file
                    self.labels_train = self.labels_train_file
                else:
                    self.features_train = np.vstack([self.features_train, self.features_train_file])
                    self.labels_train = np.vstack([self.labels_train, self.labels_train_file])
        #import pdb; pdb.set_trace()    
        # create EEG decoder object and build the model with the training data
        self.decoder = LDA()#ADD HERE PARAMS!!!!)
        
        self.decoder.fit(self.features_train, self.labels_train.ravel())
        #y_predicted = eeg.decoder.predict(self.features_test)
        
        if test_hdf_names != []:
            accuracy = self.decoder.score(self.features_test, self.labels_test.ravel()) #Check if this score function exists for LDA too 
            print "accuracy of the decoder"
            print accuracy  

        #Compute features that will be stored in the feature buffers for the online retraining
        mov_data = self.mov_data_buffer.get_all()
        rest_data = self.rest_data_buffer.get_all()
        self.channels_2train = self.channels_2train_buffers
        f_extractor = self.extractor_cls(None, channels_2train = self.channels_2train, channels = self.channel_names, feature_names = self.feature_names, feature_fn_kwargs = self.feature_fn_kwargs, win_len=self.win_len, fs=self.fs, neighbour_channels = self.neighbour_channels, brainamp_channels = self.brainamp_channels)
        rest_features, mov_features = f_extractor.extract_features_2retrain(rest_data, mov_data)
        # mov_data = f_extractor.Laplacian_filter(mov_data)
        # rest_data = f_extractor.Laplacian_filter(rest_data)
        # rest_features, mov_features = f_extractor.extract_features(rest_data, mov_data)
        #import pdb; pdb.set_trace()
        self.rest_feature_buffer.add_multiple_values(rest_features.T)
        self.mov_feature_buffer.add_multiple_values(mov_features.T)
        #import pdb; pdb.set_trace()