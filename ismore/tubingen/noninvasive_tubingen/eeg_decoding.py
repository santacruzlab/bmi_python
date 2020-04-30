import tables
import os
import numpy as np
import pandas as pd
import math
import sklearn
import copy
import matplotlib.pyplot as plt
from numpy import linalg
from numpy.linalg import solve
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from scipy.signal import butter, lfilter, filtfilt
from ismore.common_state_lists import *
from ismore import ismore_bmi_lib
from ismore.tubingen import brainamp_channel_lists
from ismore.tubingen.noninvasive_tubingen.eeg_feature_extraction import extract_AR_psd
from ismore.filter import Filter
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
        self.calibration_data = extractor_kwargs['calibration_data']
        self.artifact_rejection = extractor_kwargs['artifact_rejection']
        self.bipolar_EOG = extractor_kwargs['bipolar_EOG']
        self.fs_down = 100
        self.feature_names = extractor_kwargs['feature_names']
        self.trial_hand_side = trial_hand_side
        self.feature_fn_kwargs = extractor_kwargs['feature_fn_kwargs']
        #self.brainamp_channels = extractor_kwargs['brainamp_channels']
        neighbours = extractor_kwargs['neighbour_channels']
        
        if self.filt_training_data == True:
            self.channel_names = extractor_kwargs['eeg_channels']
            self.channel_names = [chan[:-5] for chan in self.channel_names]
            
        else:
            self.channel_names = extractor_kwargs['eeg_channels']
        self.neighbour_channels = dict()
        for chan_neighbour in neighbours:#Add the ones used for the Laplacian filter here
            self.neighbour_channels['chan' + chan_neighbour] = []
            for k,chans in enumerate(neighbours[chan_neighbour]):
                if chans not in self.channel_names:
                    self.channel_names.append(chans)
                new_channel = 'chan' + chans
                self.neighbour_channels['chan' + chan_neighbour].append(new_channel)
        
        self.n_features = 0
        for k,c in enumerate(self.channels_2train):
            self.n_features = self.n_features + len(self.feature_names)*len(extractor_kwargs['feature_fn_kwargs'][self.feature_names[0]]['freq_bands'][c])
        
        
        # channel names in HDF file have 'chan' in front (e.g., 'chanBiceps'
        self.channel_names = ['chan' + name for name in self.channel_names]
        self.channels_2train = ['chan' + name for name in self.channels_2train]
        
        #self.n_features = 0
        #for k,c in enumerate(self.channels_2train):
            #print c
            #self.n_features = self.n_features + len(self.feature_names)*len(extractor_kwargs['feature_fn_kwargs'][self.feature_names]['freq_bands'][c].keys())

        #self.n_features = len(self.channels_2train) * len(self.feature_fn_kwargs[self.feature_names[0]]['freq_bands']) # len(self.eeg_decoder.freq_bands)
        # # even if it is called states_to_decode in this case only MOV vs REST will be decoded and then a predefined movement will be triggered
        # ssm = ismore_bmi_lib.SSM_CLS_DICT[plant_type]()
        # self.states_to_decode = [s.name for s in ssm.states if s.order == 1]
        self.psd_step = 0.05*self.fs_down # step size for the eeg window for the psd
        self.extractor_cls = extractor_cls
        self.extractor_kwargs = extractor_kwargs
        # Old way of buffering data to retrain decoder
        # self.rest_data_buffer = RingBuffer(
        #     item_len=len(self.channel_names),
        #     capacity=self.buffer_len * self.fs,
        # )
        
        # self.mov_data_buffer = RingBuffer(
        #     item_len=len(self.channel_names),
        #     capacity=self.buffer_len * self.fs,
        # )

        # New way of buffering data to retrain decoder
        # self.rest_data_buffer = RingBuffer(
        #     item_len=len(self.channel_names),
        #     capacity=self.buffer_len * self.fs_down,
        # )
        
        # self.mov_data_buffer = RingBuffer(
        #     item_len=len(self.channel_names),
        #     capacity=self.buffer_len * self.fs_down,
        # )

        self.rest_feature_buffer = RingBuffer(
            item_len=self.n_features,
            capacity= int((self.buffer_len - self.win_len)* self.fs_down / self.psd_step) #2390# 120(2min) * 1000Hz = self.win_len + n*50 (50ms if task runs at 20Hz and features are computed offline with a sliding window of 50ms)# self.buffer_len * self.fs,
        )#n = 2390

        self.mov_feature_buffer = RingBuffer(
            item_len=self.n_features,
            capacity= int((self.buffer_len - self.win_len)* self.fs_down / self.psd_step) # 2390# same as above. self.buffer_len * self.fs,
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
        self.fs_down = 100
        # calculate coefficients for a 4th-order Butterworth BPF from 5-80 Hz
        band  = [1, 48]  # Hz

        #######################################
        ### only for the sleep study!!!!!! ####
        band  = [1, 15]  # Hz
        ################################

        nyq   = 0.5 * self.fs
        low   = band[0] / nyq
        high  = band[1] / nyq
        self.bpf_coeffs = butter(2, [low, high], btype='band')


        band  = [48,52]  # Hz
        nyq   = 0.5 * self.fs
        low   = band[0] / nyq
        high  = band[1] / nyq
        self.notchf_coeffs = butter(2, [low, high], btype='bandstop')

        psd_points = self.win_len * self.fs_down # length of the window of eeg in which the psd is computed
        # psd_step = 50 # step size for the eeg window for the psd

        f_extractor = self.extractor_cls(None, channels_2train = self.channels_2train, eeg_channels = self.channel_names, feature_names = self.feature_names, feature_fn_kwargs = self.feature_fn_kwargs, win_len=self.win_len, fs=self.fs, neighbour_channels = self.neighbour_channels, artifact_rejection = self.artifact_rejection, calibration_data = self.calibration_data, eog_coeffs = None, TH_lowF = None, TH_highF = None, bipolar_EOG = self.bipolar_EOG)#, brainamp_channels = self.brainamp_channels)
        #f_extractor = self.extractor_cls(None, **self.extractor_kwargs)
        
        all_hdf_names = train_hdf_names + [name for name in test_hdf_names if name not in train_hdf_names]
        
        self.hdf_data_4buffer = train_hdf_names#train_hdf_names[-2:]

        self.features_train = None
        self.labels_train = None
        self.features_test = None
        self.labels_test = None

        self.rest_lowF_feature_train = None
        self.mov_lowF_feature_train = None
        self.rest_highF_feature_train = None
        self.mov_highF_feature_train = None
        self.rest_lowF_feature_test = None
        self.mov_lowF_feature_test = None
        self.rest_highF_feature_test = None
        self.mov_highF_feature_test = None

        self.TH_lowF = None
        self.TH_highF = None
        self.eog_coeffs = None

        if self.artifact_rejection == True:

            eog_all = None
            eeg_all = None
            for hdf_name in all_hdf_names:
                # load EMG data from HDF file
                hdf = tables.openFile(hdf_name)

                store_dir_supp = '/storage/supp_hdf/'
                index_slash = hdf_name.encode('ascii','ignore').rfind('/')            
                hdf_supp_name = store_dir_supp + hdf_name.encode('ascii','ignore')[index_slash + 1:-3] + 'supp.hdf'
                
                cues = hdf.root.task_msgs[:]['msg']
                cues_trial_type = hdf.root.task[:]['trial_type']
                cues_events = hdf.root.task_msgs[:]['time']
                cues_times = hdf.root.task[:]['ts']

                n_win_pts = int(self.win_len * self.fs)
            
                #step_pts = 50  # TODO -- don't hardcode This is in sample points! Be careful if fps is changed!!
    
                try:
                    hdf_supp = tables.open_file(hdf_supp_name)
                    eeg = hdf_supp.root.brainamp[:][self.channel_names]
                except:
                    eeg = hdf.root.brainamp[:][self.channel_names]

                original_ts = eeg[self.channel_names[0]]['ts_arrival']
            
                idx = 1            
                while original_ts[idx] == original_ts[0]:
                    idx = idx + 1
  
                ts_step = 1./self.fs
                ts_eeg = np.arange(original_ts[0]-(ts_step*(idx-1)),original_ts[len(original_ts)-1],ts_step)

                
                if self.filt_training_data == True:
                    self.dtype_eeg = np.dtype([('data', np.float64),
                                       ('ts_arrival', np.float64)])

                    for k in range(len(self.channel_names)): #for loop on number of electrodes
                        eeg[self.channel_names[k]]['data'] = lfilter(self.bpf_coeffs[0],self.bpf_coeffs[1], eeg[self.channel_names[k]]['data']) 
                
                if hdf_name in train_hdf_names:
                    # Use training data to compute coefficients of the regression to remove the EOG
                    # THINK IF WE WANNA BIPOLARIZE FIRST AND THEN FILTER!!!
                    # EOG channels are the last ones in the list
                    if self.bipolar_EOG == True:
                        eog_v = eeg[self.channel_names[-1]]['data'] 
                        eog_h = eeg[self.channel_names[-2]]['data']
                        neog_channs = 2
                    else:
                        eog_v = eeg[self.channel_names[-2]]['data'] - eeg[self.channel_names[-1]]['data']
                        eog_h = eeg[self.channel_names[-4]]['data'] - eeg[self.channel_names[-3]]['data']
                        neog_channs = 4

                    eog_v_h = np.vstack([eog_v,eog_h])

                    # Put all the training data together to reject EOG artifacts and estimate the coefficients.
                    if eog_all is None:
                        eog_all = eog_v_h
                        eeg_all = np.array([eeg[self.channel_names[i]]['data'] for i in np.arange(len(self.channel_names)-neog_channs)])
                    else:
                        eog_all = np.hstack([eog_all,eog_v_h])
                        eeg_all = np.hstack([eeg_all,np.array([eeg[self.channel_names[i]]['data'] for i in np.arange(len(self.channel_names)-neog_channs)])])

                if self.calibration_data == 'screening':
                    trial_events = cues_events[np.where(cues == 'trial')] #just in case the block was stopped in the middle of a trial
                    if len(trial_events) > len(cues_events[np.where(cues == 'wait')[0][1:]]):
                        trial_events = trial_events[:-1]
                    rest_start_events = trial_events[np.where(cues_trial_type[trial_events] == 'relax')]
                    # rest_start_times = cues_times[rest_start_events]
                    mov_start_events = trial_events[np.where(cues_trial_type[trial_events] == self.trial_hand_side)]
                    # mov_start_times = cues_times[mov_start_events]
                    rest_end_events = cues_events[np.where(cues == 'wait')[0][1:]][np.where(cues_trial_type[trial_events]== 'relax')]
                    # if  rest_end_events[-1] == len(cues_times):
                    #     rest_end_events[-1] = rest_end_events[-1]-1
                    # rest_end_times = cues_times[rest_end_events]
                    mov_end_events = cues_events[np.where(cues == 'wait')[0][1:]][np.where(cues_trial_type[trial_events]== self.trial_hand_side)]

                    # if  mov_end_events[-1] == len(cues_times):
                    #     mov_end_events[-1] = mov_end_events[-1]-1
                    # mov_end_times = cues_times[mov_end_events]

                elif self.calibration_data == 'compliant':
                    
                    mov_start_events = cues_events[np.hstack([np.where(cues == 'trial'),np.where(cues == 'trial_return')])][0] #just in case the block was stopped in the middle of a trial
                    
                    # if len(mov_start_events) > len(cues_events[np.where(cues == 'wait')[0][1:]]):
                    #     mov_start_events = mov_start_events[:-1]
                    #
                    mov_end_events = cues_events[np.hstack([np.where(cues == 'instruct_rest_return'),(np.where(cues == 'wait')[0][1:],)])][0]
                    rest_start_events = cues_events[np.hstack([np.where(cues == 'rest'),np.where(cues == 'rest_return')])][0]
                    rest_end_events = cues_events[np.hstack([np.where(cues == 'instruct_trial_type'),np.where(cues == 'instruct_trial_return')])][0]
                
                elif self.calibration_data == 'compliant_testing':
                    trial_cues = np.where(cues == 'trial')[0]
                    mov_start_events = cues_events[trial_cues]#just in case the block was stopped in the middle of a trial
                    # if len(mov_start_events) > len(cues_events[np.where(cues == 'wait')[0][1:]]):
                    #     mov_start_events = mov_start_events[:-1]
                    wait_cues = np.where(cues == 'wait')[0][1:]
                    instruct_trial_type_cues = np.where(cues == 'instruct_trial_type')[0]
                    mov_end_cues = [num for num in instruct_trial_type_cues if num - 1 in trial_cues]       
                    mov_end_events = np.sort(cues_events[np.hstack([mov_end_cues,wait_cues])])
                    rest_end_cues = [num for num in instruct_trial_type_cues if num not in mov_end_cues]
                    rest_start_events = cues_events[np.where(cues == 'rest')]
                    rest_end_events = cues_events[rest_end_cues]

                elif self.calibration_data == 'active':

                    mov_start_events = cues_events[np.where(cues == 'trial')]
                    mov_end_events = cues_events[np.where(cues == 'rest')[0][1:]]
                    rest_start_events =cues_events[np.where(cues == 'rest')[0][:-1]]
                    rest_end_events = cues_events[np.where(cues == 'ready')]



                if  mov_end_events[-1] == len(cues_times):
                    mov_end_events[-1] = mov_end_events[-1]-1
                mov_end_times = cues_times[mov_end_events]
                mov_start_times = cues_times[mov_start_events]
                rest_start_times = cues_times[rest_start_events]       
                if  rest_end_events[-1] == len(cues_times):
                    rest_end_events[-1] = rest_end_events[-1]-1
                rest_end_times = cues_times[rest_end_events]
                
                # trial_type_labels = np.vstack([np.zeros(len(rest_start_times))
                # trial_type_labels[where(cues_trial_type[trial_start_events] == 'Right')] = 1

                # build time vectors relative to first time stamp
                # t_min = min(ts_eeg[0], trial_start_times[0])
                # ts_eeg = ts_eeg - t_min
                # trial_start_times = trial_start_times - t_min
                # trial_end_times = trial_end_times - t_min

                #keep same number of trials of each class
                if len(rest_start_times) > len(rest_end_times):
                    rest_start_times = rest_start_times[:len(rest_end_times)]
                if len(mov_start_times) > len(mov_end_times):
                    mov_start_times = mov_start_times[:len(mov_end_times)]
                if len(rest_start_times) > len(mov_start_times):
                    rest_start_times = rest_start_times[:len(mov_start_times)]
                    rest_end_times =  rest_end_times[:len(mov_end_times)]
                elif len(mov_start_times) > len(rest_start_times):
                    mov_start_times = mov_start_times[:len(rest_start_times)]
                    mov_end_times =  mov_end_times[:len(rest_end_times)]
                # The amount and length of REST intervals and MOVEMENT intervals with the paretic hand should be the same

                rest_start_idxs_eeg = np.zeros(len(rest_start_times))
                mov_start_idxs_eeg = np.zeros(len(mov_start_times))
                rest_end_idxs_eeg = np.zeros(len(rest_end_times))
                mov_end_idxs_eeg = np.zeros(len(mov_end_times))
                for idx in range(len(rest_start_times)):
                    # find indexes of eeg signal corresponding to start and end of rest/mov periods
                    time_dif = [ts_eeg - rest_start_times[idx]]
                    
                    bool_idx = time_dif == min(abs(time_dif[0]))
                    if np.all(bool_idx == False):
                        bool_idx = time_dif == -1*min(abs(time_dif[0]))
                    rest_start_idxs_eeg[idx] = np.where(bool_idx[0])[0]
                    time_dif = [ts_eeg - rest_end_times[idx]]
                    bool_idx = time_dif == min(abs(time_dif[0]))
                    if np.all(bool_idx == False):
                        bool_idx = time_dif == -1*min(abs(time_dif[0]))
                    rest_end_idxs_eeg[idx] = np.where(bool_idx[0])[0]

                    time_dif = [ts_eeg - mov_start_times[idx]]
                    bool_idx = time_dif == min(abs(time_dif[0]))
                    if np.all(bool_idx == False):
                        bool_idx = time_dif == -1*min(abs(time_dif[0]))
                    mov_start_idxs_eeg[idx] = np.where(bool_idx[0])[0]

                    time_dif = [ts_eeg - mov_end_times[idx]]
                    bool_idx = time_dif == min(abs(time_dif[0]))
                    if np.all(bool_idx == False):
                        bool_idx = time_dif == -1*min(abs(time_dif[0]))
                    mov_end_idxs_eeg[idx] = np.where(bool_idx[0])[0]
            
                # get rid of first rest and first trials segments to avoid using data with initial artifact from the filter
                rest_start_idxs_eeg = rest_start_idxs_eeg[1:]
                mov_start_idxs_eeg = mov_start_idxs_eeg[1:]
                rest_end_idxs_eeg = rest_end_idxs_eeg[1:]
                mov_end_idxs_eeg = mov_end_idxs_eeg[1:]

                r_lowF_feature_file = None
                m_lowF_feature_file = None
                r_highF_feature_file = None
                m_highF_feature_file = None
                # Compute power in delta and gamma bands for movement (lowF) and muscle (highF) artifact rejection
                for k in self.channel_names:  
                     
                    if 'EOG' not in k:#[4:] not in brainamp_channel_lists.eog4 + brainamp_channel_lists.eog4_filt:
                        r_lowF_feature_ch = None
                        m_lowF_feature_ch = None
                        r_highF_feature_ch = None
                        m_highF_feature_ch = None

                        for idx in range(len(rest_start_idxs_eeg)):
                            rest_window_all = eeg[k]['data'][rest_start_idxs_eeg[idx]:rest_end_idxs_eeg[idx]]
                            mov_window_all = eeg[k]['data'][mov_start_idxs_eeg[idx]:mov_end_idxs_eeg[idx]] 
                   
                            rest_window = rest_window_all[np.arange(0,len(rest_window_all),self.fs/self.fs_down)]
                            mov_window = mov_window_all[np.arange(0,len(mov_window_all),self.fs/self.fs_down)]
                            found_index = k.find('n') + 1
                            chan_freq = k[found_index:]
                            #3.- Take windows of 500ms every 50ms (i.e. overlap of 450ms)
                            #4.- Extract features (AR-psd) of each of these windows 
                            n = 0
                            while n <= (len(rest_window) - psd_points) and n <= (len(mov_window) - psd_points):

                                if self.artifact_rejection == True:
                                    r_lowF_feature = extract_AR_psd(rest_window[n:n+psd_points],[1,4])
                                    m_lowF_feature = extract_AR_psd(mov_window[n:n+psd_points],[1,4])

                                    r_highF_feature = extract_AR_psd(rest_window[n:n+psd_points],[30,48])
                                    m_highF_feature = extract_AR_psd(mov_window[n:n+psd_points],[30,48])

                                    if r_lowF_feature_ch is None:
                                        r_lowF_feature_ch = r_lowF_feature
                                        m_lowF_feature_ch = m_lowF_feature
                                        r_highF_feature_ch = r_highF_feature
                                        m_highF_feature_ch = m_highF_feature
                                    else:
                                        r_lowF_feature_ch = np.vstack([r_lowF_feature_ch, r_lowF_feature])
                                        m_lowF_feature_ch = np.vstack([m_lowF_feature_ch, m_lowF_feature])
                                        r_highF_feature_ch = np.vstack([r_highF_feature_ch, r_highF_feature])
                                        m_highF_feature_ch = np.vstack([m_highF_feature_ch, m_highF_feature])
                        
                                n += self.psd_step
                        
                        # Build feature array with columns being the channels and rows the features computed in each window
                        if r_lowF_feature_file is None: 
                            r_lowF_feature_file = r_lowF_feature_ch
                            m_lowF_feature_file = m_lowF_feature_ch
                            r_highF_feature_file = r_highF_feature_ch
                            m_highF_feature_file = m_highF_feature_ch
                                        
                        else:
                            r_lowF_feature_file = np.hstack([r_lowF_feature_file, r_lowF_feature_ch])
                            m_lowF_feature_file = np.hstack([m_lowF_feature_file, m_lowF_feature_ch])
                            r_highF_feature_file = np.hstack([r_highF_feature_file, r_highF_feature_ch])
                            m_highF_feature_file = np.hstack([m_highF_feature_file, m_highF_feature_ch])
                        
                  
                if hdf_name not in train_hdf_names:
                    if self.artifact_rejection == True:
                        if self.rest_lowF_feature_test is None:
                            self.rest_lowF_feature_test = r_lowF_feature_file
                            self.mov_lowF_feature_test = m_lowF_feature_file
                            self.rest_highF_feature_test = r_highF_feature_file
                            self.mov_highF_feature_test = m_highF_feature_file

                        else:
                            self.rest_lowF_feature_test = np.vstack([self.rest_lowF_feature_test,r_lowF_feature_file])
                            self.mov_lowF_feature_test = np.vstack([self.mov_lowF_feature_test,m_lowF_feature_file])
                            self.rest_highF_feature_test = np.vstack([self.rest_highF_feature_test,r_highF_feature_file])
                            self.mov_highF_feature_test = np.vstack([self.mov_highF_feature_test,m_highF_feature_file])                
                else:
                    if self.artifact_rejection == True:
                        if self.rest_lowF_feature_train is None:
                            self.rest_lowF_feature_train = r_lowF_feature_file
                            self.mov_lowF_feature_train = m_lowF_feature_file
                            self.rest_highF_feature_train = r_highF_feature_file
                            self.mov_highF_feature_train = m_highF_feature_file

                        else:
                            self.rest_lowF_feature_train = np.vstack([self.rest_lowF_feature_train,r_lowF_feature_file])
                            self.mov_lowF_feature_train = np.vstack([self.mov_lowF_feature_train,m_lowF_feature_file])
                            self.rest_highF_feature_train = np.vstack([self.rest_highF_feature_train,r_highF_feature_file])
                            self.mov_highF_feature_train = np.vstack([self.mov_highF_feature_train,m_highF_feature_file])

                hdf.close()
                hdf_supp.close()
            # Compute the coefficients of the regression for the EOG artifact rejection
            # UNCOMMENT
            
            # Curve fit
            #coeff, _ = curve_fit(self.func, eog_all, eeg_all)
            #b0, b1 = coeff[0], coeff[1]
            import scipy.io
            scipy.io.savemat('/home/tecnalia/code/ismore/noninvasive/eog_eeg.mat', mdict = {'eog': eog_all, 'eeg': eeg_all})
           
            covariance = np.cov(np.vstack([eog_all, eeg_all]))
            autoCovariance_eog = covariance[:eog_all.shape[0],:eog_all.shape[0]]
            crossCovariance = covariance[:eog_all.shape[0],eog_all.shape[0]:]
            self.eog_coeffs = np.linalg.solve(autoCovariance_eog, crossCovariance)
            
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
            
            
            cues = hdf.root.task_msgs[:]['msg']
            cues_trial_type = hdf.root.task[:]['trial_type']
            cues_events = hdf.root.task_msgs[:]['time']
            cues_times = hdf.root.task[:]['ts']

            n_win_pts = int(self.win_len * self.fs)
            
            #step_pts = 50  # TODO -- don't hardcode This is in sample points! Be careful if fps is changed!!

            original_ts = eeg[self.channel_names[0]]['ts_arrival']
            
            idx = 1            
            while original_ts[idx] == original_ts[0]:
                idx = idx + 1
  
            ts_step = 1./self.fs
            ts_eeg = np.arange(original_ts[0]-(ts_step*(idx-1)),original_ts[len(original_ts)-1],ts_step)

            #steps to follow to train the decoder
            #1.- Filter the whole signal - only if non-filtered data is extracted from the source/HDF file.
            # BP and NOTCH-filters might or might not be applied here, depending on what data (raw or filt) we are reading from the hdf file
            #self.filt_training_data = True
            if self.filt_training_data == True:
                self.dtype_eeg = np.dtype([('data', np.float64),
                                       ('ts_arrival', np.float64)])
                
                for k in range(len(self.channel_names)): #for loop on number of electrodes
                    eeg[self.channel_names[k]]['data'] = lfilter(self.bpf_coeffs[0],self.bpf_coeffs[1], eeg[self.channel_names[k]]['data']) 
                    #eeg[self.channel_names[k]]['data'] = lfilter(self.notchf_coeffs[0],self.notchf_coeffs[1], eeg[self.channel_names[k]]['data']) 

            # Apply artifact rejection (optional)
            if self.artifact_rejection == True:
                # EOG removal
                if self.bipolar_EOG == True:
                    eog_v = eeg[self.channel_names[-1]]['data'] 
                    eog_h = eeg[self.channel_names[-2]]['data']
                        
                else:
                    eog_v = eeg[self.channel_names[-2]]['data'] - eeg[self.channel_names[-1]]['data']
                    eog_h = eeg[self.channel_names[-4]]['data'] - eeg[self.channel_names[-3]]['data']
                #eeg_mat = np.empty([len(self.channel_names),len(eeg[self.channel_names[k]]['data'])])      
                #eog_v_h = np.vstack([eog_v,eog_h])
                for k in range(len(self.channel_names)): #for loop on number of electrodes
                   # eeg_mat[k] = eeg[self.channel_names[k]]['data']
                    if 'EOG' not in self.channel_names[k]:#[4:] not in brainamp_channel_lists.eog4 + brainamp_channel_lists.eog4_filt:
                        eeg[self.channel_names[k]]['data'] = eeg[self.channel_names[k]]['data'] - self.eog_coeffs[0,k]*eog_v - self.eog_coeffs[1,k]*eog_h
               # import scipy.io
               # scipy.io.savemat('/home/tecnalia/code/ismore/noninvasive/eog_eeg_data_for_artifacting.mat', mdict = {'eog_coeffs': self.eog_coeffs, 'covariance': covariance, 'eeg': eeg_mat, 'channels': self.channel_names})

            # Laplacian filter -  this has to be applied always, independently of using raw or filt data
            eeg = f_extractor.Laplacian_filter(eeg)
            #2.- Break done the signal into trial intervals (concatenate relax and right trials as they happen in the exp)            
            if self.calibration_data == 'screening':
                trial_events = cues_events[np.where(cues == 'trial')] #just in case the block was stopped in the middle of a trial
                if len(trial_events) > len(cues_events[np.where(cues == 'wait')][1:]):
                    trial_events = trial_events[:-1]
                rest_start_events = trial_events[np.where(cues_trial_type[trial_events] == 'relax')]
                # rest_start_times = cues_times[rest_start_events]
                mov_start_events = trial_events[np.where(cues_trial_type[trial_events] == self.trial_hand_side)]
                # mov_start_times = cues_times[mov_start_events]
                rest_end_events = cues_events[np.where(cues == 'wait')][1:][np.where(cues_trial_type[trial_events]== 'relax')]
                # if  rest_end_events[-1] == len(cues_times):
                #     rest_end_events[-1] = rest_end_events[-1]-1
                # rest_end_times = cues_times[rest_end_events]
                mov_end_events = cues_events[np.where(cues == 'wait')][1:][np.where(cues_trial_type[trial_events]== self.trial_hand_side)]

                # if  mov_end_events[-1] == len(cues_times):
                #     mov_end_events[-1] = mov_end_events[-1]-1
                # mov_end_times = cues_times[mov_end_events]

            elif self.calibration_data == 'compliant':
                mov_start_events = cues_events[np.hstack([np.where(cues == 'trial'),np.where(cues == 'trial_return')])][0] #just in case the block was stopped in the middle of a trial
                # if len(mov_start_events) > len(cues_events[np.where(cues == 'wait')[0][1:]]):
                #     mov_start_events = mov_start_events[:-1]
                mov_end_events = cues_events[np.hstack([np.where(cues == 'instruct_rest_return'),(np.where(cues == 'wait')[0][1:],)])][0]
                rest_start_events = cues_events[np.hstack([np.where(cues == 'rest'),np.where(cues == 'rest_return')])][0]
                rest_end_events = cues_events[np.hstack([np.where(cues == 'instruct_trial_type'),np.where(cues == 'instruct_trial_return')])][0]
            
            elif self.calibration_data == 'compliant_testing':
                trial_cues = np.where(cues == 'trial')[0]
                mov_start_events = cues_events[trial_cues]#just in case the block was stopped in the middle of a trial
                # if len(mov_start_events) > len(cues_events[np.where(cues == 'wait')[0][1:]]):
                #     mov_start_events = mov_start_events[:-1]
                wait_cues = np.where(cues == 'wait')[0][1:]
                instruct_trial_type_cues = np.where(cues == 'instruct_trial_type')[0]
                mov_end_cues = [num for num in instruct_trial_type_cues if num - 1 in trial_cues]       
                mov_end_events = np.sort(cues_events[np.hstack([mov_end_cues,wait_cues])])
                rest_end_cues = [num for num in instruct_trial_type_cues if num not in mov_end_cues]
                rest_start_events = cues_events[np.where(cues == 'rest')]
                rest_end_events = cues_events[rest_end_cues]

            elif self.calibration_data == 'active':

                mov_start_events = cues_events[np.where(cues == 'trial')]
                mov_end_events = cues_events[np.where(cues == 'rest')[0][1:]]
                rest_start_events =cues_events[np.where(cues == 'rest')[0][:-1]]
                rest_end_events = cues_events[np.where(cues == 'ready')]
   
            if  mov_end_events[-1] == len(cues_times):
                mov_end_events[-1] = mov_end_events[-1]-1
            mov_end_times = cues_times[mov_end_events]
            mov_start_times = cues_times[mov_start_events]
            rest_start_times = cues_times[rest_start_events]       
            if  rest_end_events[-1] == len(cues_times):
                rest_end_events[-1] = rest_end_events[-1]-1
            rest_end_times = cues_times[rest_end_events]
             
            # trial_type_labels = np.vstack([np.zeros(len(rest_start_times))
            # trial_type_labels[where(cues_trial_type[trial_start_events] == 'Right')] = 1

            # build time vectors relative to first time stamp
            # t_min = min(ts_eeg[0], trial_start_times[0])
            # ts_eeg = ts_eeg - t_min
            # trial_start_times = trial_start_times - t_min
            # trial_end_times = trial_end_times - t_min

            #keep same number of trials of each class
            if len(rest_start_times) > len(rest_end_times):
                rest_start_times = rest_start_times[:len(rest_end_times)]
            if len(mov_start_times) > len(mov_end_times):
                mov_start_times = mov_start_times[:len(mov_end_times)]
            if len(rest_start_times) > len(mov_start_times):
                rest_start_times = rest_start_times[:len(mov_start_times)]
                rest_end_times =  rest_end_times[:len(mov_end_times)]
            elif len(mov_start_times) > len(rest_start_times):
                mov_start_times = mov_start_times[:len(rest_start_times)]
                mov_end_times =  mov_end_times[:len(rest_end_times)]
            # The amount and length of REST intervals and MOVEMENT intervals with the paretic hand should be the same

            rest_start_idxs_eeg = np.zeros(len(rest_start_times))
            mov_start_idxs_eeg = np.zeros(len(mov_start_times))
            rest_end_idxs_eeg = np.zeros(len(rest_end_times))
            mov_end_idxs_eeg = np.zeros(len(mov_end_times))
            for idx in range(len(rest_start_times)):
                # find indexes of eeg signal corresponding to start and end of rest/mov periods
                time_dif = [ts_eeg - rest_start_times[idx]]
                bool_idx = time_dif == min(abs(time_dif[0]))
                if np.all(bool_idx == False):
                    bool_idx = time_dif == -1*min(abs(time_dif[0]))
                rest_start_idxs_eeg[idx] = np.where(bool_idx[0])[0]

                time_dif = [ts_eeg - rest_end_times[idx]]
                bool_idx = time_dif == min(abs(time_dif[0]))
                if np.all(bool_idx == False):
                    bool_idx = time_dif == -1*min(abs(time_dif[0]))
                rest_end_idxs_eeg[idx] = np.where(bool_idx[0])[0]

                time_dif = [ts_eeg - mov_start_times[idx]]
                bool_idx = time_dif == min(abs(time_dif[0]))
                if np.all(bool_idx == False):
                    bool_idx = time_dif == -1*min(abs(time_dif[0]))
                mov_start_idxs_eeg[idx] = np.where(bool_idx[0])[0]

                time_dif = [ts_eeg - mov_end_times[idx]]
                bool_idx = time_dif == min(abs(time_dif[0]))
                if np.all(bool_idx == False):
                    bool_idx = time_dif == -1*min(abs(time_dif[0]))
                mov_end_idxs_eeg[idx] = np.where(bool_idx[0])[0]
            
            # get rid of first rest and first trials segments to avoid using data with initial artifact from the filter
            rest_start_idxs_eeg = rest_start_idxs_eeg[1:]
            mov_start_idxs_eeg = mov_start_idxs_eeg[1:]
            rest_end_idxs_eeg = rest_end_idxs_eeg[1:]
            mov_end_idxs_eeg = mov_end_idxs_eeg[1:]
            self.features_train_file = None
            self.features_test_file = None
            for k in self.channels_2train: 
                r_features_ch = None
                m_features_ch = None
               
                for idx in range(len(rest_start_idxs_eeg)):
                    rest_window_all = eeg[k]['data'][rest_start_idxs_eeg[idx]:rest_end_idxs_eeg[idx]]
                    mov_window_all = eeg[k]['data'][mov_start_idxs_eeg[idx]:mov_end_idxs_eeg[idx]] 
                    
                    rest_window = rest_window_all[np.arange(0,len(rest_window_all),self.fs/self.fs_down)]
                    mov_window = mov_window_all[np.arange(0,len(mov_window_all),self.fs/self.fs_down)]
                    found_index = k.find('n') + 1
                    chan_freq = k[found_index:]
                    #3.- Take windows of 500ms every 50ms (i.e. overlap of 450ms)
                    #4.- Extract features (AR-psd) of each of these windows 
                    n = 0
                    while n <= (len(rest_window) - psd_points) and n <= (len(mov_window) - psd_points):
                        
                        r_feats = f_extractor.extract_features(rest_window[n:n+psd_points],chan_freq)                        
                        m_feats = f_extractor.extract_features(mov_window[n:n+psd_points],chan_freq)             

                        if r_features_ch is None:
                            r_features_ch = r_feats
                            m_features_ch = m_feats
                        else:
                            r_features_ch = np.vstack([r_features_ch, r_feats])
                            m_features_ch = np.vstack([m_features_ch, m_feats])
                        n += self.psd_step
                if hdf_name in train_hdf_names:
                    if  self.features_train_file is None:
                        self.features_train_file = np.vstack([r_features_ch,m_features_ch])
                        #self.labels_train_file2 = np.vstack([np.zeros([r_features_ch.shape[0],1]), np.ones([m_features_ch.shape[0],1])])
                    else:
                        self.features_train_file = np.hstack([self.features_train_file, np.vstack([r_features_ch,m_features_ch])])
                        #self.labels_train_file2 = np.hstack([self.labels_train_file2, np.vstack([np.zeros([r_features_ch.shape[0],1]), np.ones([m_features_ch.shape[0],1])])])

                else:
                    if  self.features_test_file is None:
                        self.features_test_file = np.vstack([r_features_ch, m_features_ch])
                        
                    else:
                        self.features_test_file = np.hstack([self.features_test_file, np.vstack([r_features_ch, m_features_ch])])
                        
            if hdf_name not in train_hdf_names:
                
                self.labels_test_file = np.vstack([np.zeros([r_features_ch.shape[0],1]), np.ones([m_features_ch.shape[0],1])])
                
                if self.features_test is None:
                    self.features_test = self.features_test_file
                    self.labels_test = self.labels_test_file
                else:
                    self.features_test = np.vstack([self.features_test, self.features_test_file])
                    self.labels_test = np.vstack([self.labels_test, self.labels_test_file])
            else:
                
                self.labels_train_file = np.vstack([np.zeros([r_features_ch.shape[0],1]),np.ones([m_features_ch.shape[0],1])])
                if self.features_train is None:
                    self.features_train = self.features_train_file
                    self.labels_train = self.labels_train_file
                else:
                    self.features_train = np.vstack([self.features_train, self.features_train_file])
                    self.labels_train = np.vstack([self.labels_train, self.labels_train_file])
        # Reject the trials that are not below the computed TH.
        if self.artifact_rejection == True:
            # First iteration
            # Compute thresholds for low and high frequencies artifact rejection during the rest period.
 
            self.TH_lowF = np.mean(self.rest_lowF_feature_train, axis = 0) + 3*np.std(self.rest_lowF_feature_train, axis = 0)
            self.TH_highF = np.mean(self.rest_highF_feature_train, axis = 0) + 3*np.std(self.rest_highF_feature_train, axis = 0)
            
       
            # Separate features into rest and mov features and apply artifact rejection to each of them separately
            rest_features_artifacted = self.features_train[np.where([self.labels_train == 0])[1],:]
            mov_features_artifacted = self.features_train[np.where([self.labels_train == 1])[1],:]#If any channel is contaminated, remove that point in all channels.
           
            # If we force to reject same amount of rest and mov data
            # index_keep_rest = np.all(self.rest_lowF_feature_train < self.TH_lowF, axis = 1) & np.all(self.rest_highF_feature_train < self.TH_highF, axis = 1)
            # index_keep_mov = np.all(self.mov_lowF_feature_train < self.TH_lowF, axis = 1) & np.all(self.mov_highF_feature_train < self.TH_highF, axis = 1)
            #index_keep_common = np.where([index_keep_mov & index_keep_rest])[1]
            # rest_features_clean1 = rest_features_artifacted[index_keep_common,:]
            # mov_features_clean1 = mov_features_artifacted[index_keep_common,:]
            # rest_lowF_feature_clean = self.rest_lowF_feature_train[index_keep_common,:]
            # mov_lowF_feature_clean = self.mov_lowF_feature_train[index_keep_common,:]
            # rest_highF_feature_clean = self.rest_highF_feature_train[index_keep_common,:]
            # mov_highF_feature_clean = self.mov_highF_feature_train[index_keep_common,:]
            # windows_rejected_train_it1 = self.rest_lowF_feature_train.shape[0] - len(index_keep_common)
            # total_windows_train_it1 = self.rest_lowF_feature_train.shape[0] 
            #print 'THS', self.TH_lowF, self.TH_highF
          
            
            # We don't force to reject same amount of rest and mov windows
            index_keep_rest = np.where([np.all(self.rest_lowF_feature_train < self.TH_lowF, axis = 1) & np.all(self.rest_highF_feature_train < self.TH_highF, axis = 1)])[1]
            index_keep_mov = np.where([np.all(self.mov_lowF_feature_train < self.TH_lowF, axis = 1) & np.all(self.mov_highF_feature_train < self.TH_highF, axis = 1)])[1]
            rest_features_clean1 = rest_features_artifacted[index_keep_rest,:]
            mov_features_clean1 = mov_features_artifacted[index_keep_mov,:]
            rest_lowF_feature_clean = self.rest_lowF_feature_train[index_keep_rest,:]
            mov_lowF_feature_clean = self.mov_lowF_feature_train[index_keep_mov,:]
            rest_highF_feature_clean = self.rest_highF_feature_train[index_keep_rest,:]
            mov_highF_feature_clean = self.mov_highF_feature_train[index_keep_mov,:]
            windows_rest_rejected_train_it1 = self.rest_lowF_feature_train.shape[0] - len(index_keep_rest)
            windows_mov_rejected_train_it1 = self.mov_lowF_feature_train.shape[0] - len(index_keep_mov)
            total_windows_train_it1 = self.rest_lowF_feature_train.shape[0] 
            
            # Second iteration
            # Recompute new TH in cleaned rest periods
            self.TH_lowF = np.mean(rest_lowF_feature_clean, axis = 0) + 3*np.std(rest_lowF_feature_clean, axis = 0)
            self.TH_highF = np.mean(rest_highF_feature_clean, axis = 0) + 3*np.std(rest_highF_feature_clean, axis = 0)

            print self.TH_lowF, self.TH_highF
            # If we force to reject same amount of rest and mov data
            # index_keep_rest = np.all(rest_lowF_feature_clean < self.TH_lowF, axis = 1) & np.all(rest_highF_feature_clean < self.TH_highF, axis = 1)
            # index_keep_mov = np.all(mov_lowF_feature_clean < self.TH_lowF, axis = 1) & np.all(mov_highF_feature_clean < self.TH_highF, axis = 1)    
            # index_keep_common = np.where([index_keep_mov & index_keep_rest])[1]
            # windows_rejected_train_it2 = rest_lowF_feature_clean.shape[0] - len(index_keep_common)
            # total_windows_train_it2 = rest_lowF_feature_clean.shape[0] 
            # self.rest_features_train_clean = rest_features_clean1[index_keep_common,:]
            # self.mov_features_train_clean = mov_features_clean1[index_keep_common,:]

            # We don't force to reject same amount of rest and mov windows
            index_keep_rest = np.where([np.all(rest_lowF_feature_clean < self.TH_lowF, axis = 1) & np.all(rest_highF_feature_clean < self.TH_highF, axis = 1)])[1]
            index_keep_mov = np.where([np.all(mov_lowF_feature_clean < self.TH_lowF, axis = 1) & np.all(mov_highF_feature_clean < self.TH_highF, axis = 1)])[1]
            windows_rest_rejected_train_it2 = rest_lowF_feature_clean.shape[0] - len(index_keep_rest)
            windows_mov_rejected_train_it2 = mov_lowF_feature_clean.shape[0] - len(index_keep_mov)
            total_windows_rest_train_it2 = rest_lowF_feature_clean.shape[0] 
            total_windows_mov_train_it2 = mov_lowF_feature_clean.shape[0] 
            self.rest_features_train_clean = rest_features_clean1[index_keep_rest,:]
            self.mov_features_train_clean = mov_features_clean1[index_keep_mov,:]

     
            # Replicate data to balance both rest and mov datasets in case windows were not forced to be rejected in pairs
            nrest_clean_windows = self.rest_features_train_clean.shape[0]
            nmov_clean_windows = self.mov_features_train_clean.shape[0]
            if nrest_clean_windows > nmov_clean_windows:
                dif_nwindows = nrest_clean_windows - nmov_clean_windows
                self.mov_features_train_clean = np.vstack([self.mov_features_train_clean,self.mov_features_train_clean[-dif_nwindows:,:]])
            elif nmov_clean_windows > nrest_clean_windows:
                dif_nwindows = nmov_clean_windows - nrest_clean_windows
                self.rest_features_train_clean = np.vstack([self.rest_features_train_clean, self.rest_features_train_clean[-dif_nwindows:,:]])
            
            # Build training feature and label matrix with clean data
            self.features_train = np.vstack([self.rest_features_train_clean,self.mov_features_train_clean])
            self.labels_train = np.vstack([np.zeros([self.rest_features_train_clean.shape[0],1]), np.ones([self.mov_features_train_clean.shape[0],1])])
        
        # normalize features_train
        mean_feat = np.mean(self.features_train, axis = 0)
        std_feat = np.std(self.features_train, axis = 0)
        self.features_train = (self.features_train - mean_feat) / std_feat

        self.decoder = LDA()#ADD HERE PARAMS!!!!)
        self.decoder.fit(self.features_train, self.labels_train.ravel())
        #y_predicted = eeg.decoder.predict(self.features_test)       

        if test_hdf_names != []:
            if self.artifact_rejection == True:
                # Separate features into rest and mov features and apply artifact rejection to each of them separately
                features_test_rest = self.features_test[np.where([self.labels_test == 0])[1],:]
                features_test_mov = self.features_test[np.where([self.labels_test == 1])[1],:]
                
                 # If we force to reject same amount of rest and mov data
                # index_keep_rest = np.all(self.rest_lowF_feature_test < self.TH_lowF, axis = 1) & np.all(self.rest_highF_feature_test < self.TH_highF, axis = 1)
                # index_keep_mov = np.all(self.mov_lowF_feature_test < self.TH_lowF, axis = 1) & np.all(self.mov_highF_feature_test < self.TH_highF, axis = 1)
                # index_keep_common = np.where([index_keep_mov & index_keep_rest])[1]
                # windows_rejected_test = self.rest_lowF_feature_test.shape[0] - len(index_keep_common)
                # total_windows_test = self.rest_lowF_feature_test.shape[0] 
                
                 # We don't force to reject same amount of rest and mov windows
                index_keep_rest = np.where([np.all(self.rest_lowF_feature_test < self.TH_lowF, axis = 1) & np.all(self.rest_highF_feature_test < self.TH_highF, axis = 1)])[1]
                index_keep_mov = np.where([np.all(self.mov_lowF_feature_test < self.TH_lowF, axis = 1) & np.all(self.mov_highF_feature_test < self.TH_highF, axis = 1)])[1]
                windows_rest_rejected_test = self.rest_lowF_feature_test.shape[0] - len(index_keep_rest)
                windows_mov_rejected_test = self.mov_lowF_feature_test.shape[0] - len(index_keep_mov)
                total_windows_test = self.rest_lowF_feature_test.shape[0] 
                # Build testing feature and label matrix with clean data
                self.features_test = np.vstack([features_test_rest[index_keep_rest,:],features_test_mov[index_keep_mov,:]])
                self.labels_test = np.vstack([np.zeros([features_test_rest[index_keep_rest,:].shape[0],1]), np.ones([features_test_mov[index_keep_mov,:].shape[0],1])])
            
            # normalize features_test
            mean_feat = np.mean(self.features_test, axis = 0)
            std_feat = np.std(self.features_test, axis = 0)
            self.features_test = (self.features_test - mean_feat) / std_feat
            accuracy = self.decoder.score(self.features_test, self.labels_test.ravel()) #Check if this score function exists for LDA too 
            print "accuracy of the decoder"
            print accuracy  
            # probability = self.decoder.predict_proba(self.features_test)
            # plt.figure()
            # plt.plot(self.decoder.predict(self.features_test))
            # plt.plot(self.labels_test,'r')
            # plt.show(block = False)
        if self.artifact_rejection == True:
            try:
                print str(windows_rest_rejected_train_it1), " out of ", str(total_windows_train_it1), " rest windows rejected in the first iteration of the training set"
                print str(windows_mov_rejected_train_it1), " out of ", str(total_windows_train_it1), " mov windows rejected in the first iteration of the training set"
                print str(windows_rest_rejected_train_it2), " out of ", str(total_windows_rest_train_it2), " rest windows rejected in the second iteration of the training set"
                print str(windows_mov_rejected_train_it2), " out of ", str(total_windows_mov_train_it2), " mov windows rejected in the second iteration of the training set"
                print str(windows_rest_rejected_test), " out of ", str(total_windows_test), " rest windows rejected in the test set"
                print str(windows_mov_rejected_test), " out of ", str(total_windows_test), " mov windows rejected in the test set"
            except:
                pass
        #Compute features that will be stored in the feature buffers for the online retraining
        # mov_data = self.mov_data_buffer.get_all()
        # rest_data = self.rest_data_buffer.get_all()
        self.channels_2train = self.channels_2train_buffers
        f_extractor = self.extractor_cls(None, channels_2train = self.channels_2train, eeg_channels = self.channel_names, feature_names = self.feature_names, feature_fn_kwargs = self.feature_fn_kwargs, win_len=self.win_len, fs=self.fs, neighbour_channels = self.neighbour_channels, artifact_rejection = self.artifact_rejection, calibration_data = self.calibration_data, eog_coeffs = self.eog_coeffs, TH_lowF = self.TH_lowF, TH_highF = self.TH_highF)#, brainamp_channels = self.brainamp_channels)
        # CHANGE THIS!!!! put directly the rest and mov features from the training set by looking at the labels and separating them
        #rest_features, mov_features = f_extractor.extract_features_2retrain(rest_data, mov_data)
        # mov_data = f_extractor.Laplacian_filter(mov_data)
        # rest_data = f_extractor.Laplacian_filter(rest_data)
        # rest_features, mov_features = f_extractor.extract_features(rest_data, mov_data)
        # Save features from clean windows in the buffer for retraining the decoder online
        try:
            self.rest_feature_buffer.add_multiple_values(self.rest_features_train_clean.T)
            self.mov_feature_buffer.add_multiple_values(self.mov_features_train_clean.T)
        except:
            self.rest_features_train_clean = self.features_train[np.where([self.labels_train == 0])[1],:]
            self.mov_features_train_clean = self.features_train[np.where([self.labels_train == 1])[1],:]
            self.rest_feature_buffer.add_multiple_values(self.rest_features_train_clean.T[:,:self.rest_feature_buffer.capacity-1])
            self.mov_feature_buffer.add_multiple_values(self.mov_features_train_clean.T[:,:self.rest_feature_buffer.capacity-1])