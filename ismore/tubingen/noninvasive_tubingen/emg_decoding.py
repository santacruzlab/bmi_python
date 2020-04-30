import tables
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from ismore.tubingen import brainamp_channel_lists
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.signal import butter, lfilter, filtfilt

from ismore.common_state_lists import *
from ismore import ismore_bmi_lib

# from riglib.filter import Filter
from ismore.filter import Filter
from scipy.signal import filtfilt



class EMGDecoderBase(object):
    '''
    Abstract base class for all concrete EMG decoder classes
    '''
    pass


class LinearEMGDecoder(EMGDecoderBase):
    '''Concrete base class for a linear EMG decoder.'''

    def __init__(self, channels_2train, plant_type, fs, win_len, filt_training_data, extractor_cls, extractor_kwargs, opt_channels_2train_dict):
        
        if channels_2train == brainamp_channel_lists.emg_48hd_6mono_filt:
            self.recorded_channels = brainamp_channel_lists.emg_48hd_6mono
            self.recorded_channels = ['chan' + name for name in self.recorded_channels]
            self.emg_channels = extractor_kwargs["emg_channels"]
            self.HD_EMG_diag = True
        else:
            self.emg_channels = channels_2train
            self.HD_EMG_diag = False

        self.plant_type     = plant_type
        self.fs             = fs
        self.win_len        = win_len
        #self.filt_training_data = extractor_kwargs.pop('filt_training_data', False)
        self.filt_training_data = filt_training_data
        #self.channels_filt  = extractor_kwargs['channels_filt']
        self.dtype = np.dtype([('data',       np.float64),
                               ('ts_arrival', np.float64)])
        self.feature_names = extractor_kwargs['feature_names']
        self.feature_fn_kwargs = extractor_kwargs['feature_fn_kwargs']
        #self.brainamp_channels = extractor_kwargs['brainamp_channels']
        # channel names in HDF file have 'chan' in front (e.g., 'chanBiceps'
        self.channel_names = ['chan' + name[:-5] for name in self.emg_channels]
        
        ssm = ismore_bmi_lib.SSM_CLS_DICT[plant_type]()
        self.states_to_decode = [s.name for s in ssm.states if s.order == 1]
        self.opt_channels_2train_dict = opt_channels_2train_dict
        self.extractor_cls = extractor_cls
        self.extractor_kwargs = extractor_kwargs
        self.fixed_var_scalar = extractor_kwargs['fixed_var_scalar']
        self.subset_muscles = self.extractor_kwargs['subset_muscles']
        self.index_all_features = np.arange(0,len(self.emg_channels)*len(self.feature_names),len(self.emg_channels))
        
        if not self.subset_muscles:
            # Use all the muscles for the decoding of all DoFs.
            self.subset_features = dict()
        else:
            # Use a subset of muscles for the decoding of each DoF. 
            self.subset_features = dict()
            for state in self.states_to_decode:
                self.subset_features[state] = np.int()
                for index in np.arange(len(self.subset_muscles[state])):
                    self.subset_features[state] = np.hstack([self.subset_features[state], np.array(self.index_all_features + self.subset_muscles[state][index])]) 
                self.subset_features[state].sort()
  

    def __call__(self, features):
        decoder_output = pd.Series(0.0, self.states_to_decode)
     
        for state in self.states_to_decode:

            if not self.subset_features:
                # Use all the muscles for the decoding of all DoFs.
                decoder_output[state] = self.beta[state].T.dot(features.reshape(-1,1))
            else:
                # Use a subset of muscles for the decoding of each DoF 
                decoder_output[state] = self.beta[state].T.dot(features[self.subset_features[state]].reshape(-1,1))
            

        return decoder_output

    def train_ridge(self, K, train_hdf_names, test_hdf_names, states_to_flip):
        '''Use ridge regression to train this decoder from data from multiple .hdf files.'''

        # save this info as part of the decoder object
        self.K               = K
        self.train_hdf_names = train_hdf_names
        self.test_hdf_names  = test_hdf_names
        self.states_to_flip  = states_to_flip


        # will be 2-D arrays, each with shape (N, n_features)
        # e.g., if extracting 7 features from each of 14 channels, then the
        #   shape might be (10000, 98)
        feature_data_train = None
        feature_data_test  = None
        #emg_filtfilt = None
        #emg_filt = None
        #emg_raw = None
        
        #emg_signal = None
        
        # calculate coefficients for a 4th-order Butterworth LPF at 1.5 Hz
        fs_synch = 20 #Frequency at which emg and kin data are synchronized
        nyq   = 0.5 * fs_synch
        cuttoff_freq  = 1.5 / nyq
        bpf_kin_coeffs = butter(4, cuttoff_freq, btype='low')
        lpf = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1])   
        
        cuttoff_freq_test  = 5 / nyq
        bpf_kin_coeffs_test = butter(4, cuttoff_freq_test, btype='low')
        lpf_test = Filter(bpf_kin_coeffs_test[0], bpf_kin_coeffs_test[1])  
        
        # each will be a dictionary where:
        # key: a kinematic state (e.g., 'aa_px')
        # value: kinematic data for that state, interpolated to correspond to
        #   the same times as the rows of feature_data
        #   e.g., if feature_data_train has N rows, then each value in
        #   kin_data_train will be an array of length N
        kin_data_train = dict()
        kin_data_train_lpf = dict()
        kin_data_test  = dict()
        kin_data_test_lpf  = dict()
        kin_data_filts = dict()
        
        
        window = int(120 * fs_synch) # sliding window (buffer) used to normalize the test signal
        
        ## ---- calculate coefficients for a 4th-order Butterworth BPF from 10-450 Hz
        if self.fs >= 1000: 
            band  = [10, 450]  # Hz
        else:
            band = [10, 200]
        nyq   = 0.5 * self.fs
        low   = band[0] / nyq
        high  = band[1] / nyq
        bpf_coeffs = butter(4, [low, high], btype='band')
        # self.band_pass_filter = Filter(self.bpf_coeffs[0], self.bpf_coeffs[1])


        ## ---- calculate coefficients for multiple 2nd-order notch filers
        notchf_coeffs = []

        if self.fs >= 1000: 
            notch_freqs  = [50, 150, 250, 350]  # Hz
        else:
            notch_freqs = [50, 150]

        for freq in notch_freqs:
            band  = [freq - 1, freq + 1]  # Hz
            nyq   = 0.5 * self.fs
            low   = band[0] / nyq
            high  = band[1] / nyq
            notchf_coeffs.append(butter(2, [low, high], btype='bandstop'))

        notch_filters = []
        for b, a in notchf_coeffs:
            notch_filters.append(Filter(b=b, a=a))

        n_channels = len(self.channel_names)
        channel_filterbank = [None]*n_channels
        for k in range(n_channels):
            filts = [Filter(bpf_coeffs[0], bpf_coeffs[1])]
            for b, a in notchf_coeffs:
                filts.append(Filter(b=b, a=a))
            channel_filterbank[k] = filts
        #andrea

        f_extractor = self.extractor_cls(None, emg_channels = self.emg_channels, feature_names = self.feature_names, feature_fn_kwargs = self.feature_fn_kwargs, win_len=self.win_len, fs=self.fs)#, brainamp_channels = self.brainamp_channels)
        # f_extractor = self.extractor_cls(None, **self.extractor_kwargs)


        all_hdf_names = train_hdf_names + [name for name in test_hdf_names if name not in train_hdf_names]
        for hdf_name in all_hdf_names:
            # load EMG data from HDF file
            hdf = tables.open_file(hdf_name)
            
            store_dir_supp = '/storage/supp_hdf/'
            index_slash = hdf_name.encode('ascii','ignore').rfind('/')            
            hdf_supp_name = store_dir_supp + hdf_name.encode('ascii','ignore')[index_slash + 1:-3] + 'supp.hdf'
            
            hdf_supp = tables.open_file(hdf_supp_name)
            try:                
                emg = hdf_supp.root.brainamp[:][self.recorded_channels]
                original_ts = emg[self.recorded_channels[0]]['ts_arrival']
            except:
                emg = hdf_supp.root.brainamp[:][self.channel_names]
                #emg = hdf.root.brainamp[:][self.channel_names]
                original_ts = emg[self.channel_names[0]]['ts_arrival']
                #emg = hdf.root.brainamp[:][self.channel_names]

            # try:
            #     emg = hdf.root.brainamp[:][self.channel_names]
            # except:  # in older HDF files, brainamp data was stored under table 'emg'
            #     emg = hdf.root.emg[:][self.channel_names]

            # "correct" the saved vector of timestamps by assuming that the
            #   last occurrence of the first EMG timestamp is correct
            #   e.g., if fs = 1000, EMG data arrives in blocks of 4 points, and
            #      the saved timestamps are:
            #        [5.103, 5.103, 5.103, 5.103, 5.107, 5.107, ...]
            #      then the "corrected" timestamps (saved in ts_vec) would be:
            #        [5.100, 5.101, 5.102, 5.103, 5.104, 5.105, ...]

           
                
            idx = 1            
            while original_ts[idx] == original_ts[0]:
                idx = idx + 1
            # idx = idx - 1
            ts_step = 1./self.fs
            # ts_before = original_ts[idx] + (ts_step * np.arange(-idx, 0))
            # ts_after = original_ts[idx] + (ts_step * np.arange(1, len(original_ts)))
            # ts_vec = np.hstack([ts_before, original_ts[idx], ts_after])
            ts_emg = np.arange(original_ts[0]-(ts_step*(idx-1)),original_ts[len(original_ts)-1],ts_step)
            if self.plant_type in ['ArmAssist', 'IsMore']:
                if 'armassist' not in hdf.root:
                    raise Exception('Invalid plant_type %s: no ArmAssist data saved in HDF file.' % self.plant_type)

                else:
                    ts_aa = hdf.root.armassist[1:]['ts_arrival']

            if self.plant_type in ['ReHand', 'IsMore']:
                if 'rehand' not in hdf.root:
                    raise Exception('Invalid plant_type %s: no ReHand data saved in HDF file.' % self.plant_type)
                
                else:
                    ts_rh = hdf.root.rehand[:]['ts_arrival']

            if 'ts_rh' in locals() and 'ts_aa' in locals():
                ts_max = min(ts_emg[len(ts_emg)-1], ts_aa[len(ts_aa)-1], ts_rh[len(ts_rh)-1])
                ts_min = max(ts_emg[0], ts_aa[0], ts_rh[0])
            elif 'ts_rh' not in locals() and 'ts_aa' in locals():
                ts_max = min(ts_emg[len(ts_emg)-1], ts_aa[len(ts_aa)-1])
                ts_min = max(ts_emg[0], ts_aa[0])
            elif 'ts_rh' in locals() and 'ts_aa' not in locals():
                ts_max = min(ts_emg[len(ts_emg)-1], ts_rh[len(ts_rh)-1])
                ts_min = max(ts_emg[0], ts_rh[0])
            else:
                ts_max = ts_emg[len(ts_emg)-1]
                ts_min = ts_emg[0]

            ts_vec = ts_emg[(ts_emg < ts_max) * (ts_emg > ts_min)]

            # cut off small amount of data from start and end of emg
            cutoff_time = 0#.5  # secs
            #cutoff_pts = int(cutoff_time * self.fs)
            #
            #emg = emg[cutoff_pts:-cutoff_pts]
            #ts_vec = ts_vec[cutoff_pts:-cutoff_pts]

            

            n_win_pts = int(self.win_len * self.fs)
            
            step_pts = 50  # TODO -- don't hardcode This is in sample points! Be careful if fs is changed!!
            #step_pts_dist = 20  # TODO -- don't hardcode
            start_idxs = np.arange(0, len(ts_vec) - n_win_pts + 1, step_pts) #andrea
            #start_idxs = np.arange(n_win_pts - 1, len(emg), step_pts)
            

            features = np.zeros((len(start_idxs), f_extractor.n_features))
            #emg_samples = np.zeros((len(self.channel_names), n_win_pts*len(start_idxs)))
            
            #Using causal filters
            # for k in range(n_channels): #for loop on number of electrodes
            #     for filt in channel_filterbank[k]:
            #         emg[self.channel_names[k]]['data'] = (filt(emg[self.channel_names[k]]['data']))

            # if hdf_name in train_hdf_names:# testing data is filtered with a causal filter while training data can be filtered with filtfilt
            #     for k in range(n_channels): #for loop on number of electrodes
            #         emg[self.channel_names[k]]['data']  = filtfilt(bpf_coeffs[0],bpf_coeffs[1], emg[self.channel_names[k]]['data'] ) 
                    
            #         for b, a in notchf_coeffs:
            #             emg[self.channel_names[k]]['data']  = filtfilt(b = b, a = a, x = emg[self.channel_names[k]]['data'] ) 
            # else: 

            
            # Diagonalization
            if self.HD_EMG_diag:
                #diag_emg = np.array()
                diag_emg = dict()#np.ndarray(emg.shape)
                if hdf_name in train_hdf_names:
                    chan_2keep = self.opt_channels_2train_dict['channels_str_2keep']
                    chan_2discard = self.opt_channels_2train_dict['channels_str_2discard']
                    chan_diag1_1 = self.opt_channels_2train_dict['channels_diag1_1']
                    chan_diag1_2 = self.opt_channels_2train_dict['channels_diag1_2']
                    chan_diag2_1 = self.opt_channels_2train_dict['channels_diag2_1']
                    chan_diag2_2 = self.opt_channels_2train_dict['channels_diag2_2']


                else:
                    chan_2keep = self.extractor_kwargs['channels_str_2keep']
                    chan_2discard = self.extractor_kwargs['channels_str_2discard']
                    chan_diag1_1 = self.extractor_kwargs['channels_diag1_1']
                    chan_diag1_2 = self.extractor_kwargs['channels_diag1_2']
                    chan_diag2_1 = self.extractor_kwargs['channels_diag2_1']
                    chan_diag2_2 = self.extractor_kwargs['channels_diag2_2']
                data = np.zeros([len(self.recorded_channels),len(emg[self.recorded_channels[0]]['data'])])
                
                for k in range(len(self.recorded_channels)):
                    data[k,:] = emg[self.recorded_channels[k]]['data']
                data_diff = np.diff(data, axis = 0)
                
                #diag_emg = np.zeros((1, len(emg[self.recorded_channels[0]]['data'])), dtype=self.dtype)
                for i in range(n_channels):
                    if i < len(chan_2keep):
                        diag_emg[self.channel_names[i]] = np.zeros((len(emg[self.recorded_channels[0]]['data'])), dtype=self.dtype) 
                        diag_emg[self.channel_names[i]]['data'] = data_diff[chan_2keep[i],:]
                        #filtered_data[i,:] = data_diff[chan_2keep[i],:]
                    elif i < (len(chan_2keep) + len(chan_diag1_1)):
                        diag_emg[self.channel_names[i]] = np.zeros((len(emg[self.recorded_channels[0]]['data'])), dtype=self.dtype) 
                        diag_emg[self.channel_names[i]]['data'] = emg[self.recorded_channels[chan_diag1_1[i-len(chan_2keep)]]]['data'] - emg[self.recorded_channels[chan_diag1_2[i-len(chan_2keep)]]]['data']
                        #filtered_data[i,:] = data[chan_diag1_1[i-len(chan_2keep)]]['data'] - data[chan_diag1_2[i-len(chan_2keep)]]['data']
                    else:
                        diag_emg[self.channel_names[i]] = np.zeros((len(emg[self.recorded_channels[0]]['data'])), dtype=self.dtype) 
                        diag_emg[self.channel_names[i]]['data'] = emg[self.recorded_channels[chan_diag2_1[i-len(chan_2keep)-len(chan_diag1_1)]]]['data'] - emg[self.recorded_channels[chan_diag2_2[i-len(chan_2keep)-len(chan_diag1_1)]]]['data']
                        #filtered_data[i,:] = data[chan_diag2_1[i-len(chan_2keep)-len(chan_diag1_1)]]['data'] - data[chan_diag2_2[i-len(chan_2keep)-len(chan_diag1_1)]]['data']        
                    # for filt in channel_filterbank[i]: #channel_filterbank_emg has filters for the amount of raw channels, which is larger than the number of filtered channels. That's why this is not a problem
                    #     diag_emg[self.channel_names[i]]['data'] =  filt(diag_emg[self.channel_names[i]]['data']) 
                        #filtered_data[i]['data'] =  filt(filtered_data[i]['data'] ) 
                emg = diag_emg.copy()    

                          

            if self.filt_training_data:
                for k in range(n_channels): #for loop on number of electrodes
                    for filt in channel_filterbank[k]: #channel_filterbank_emg has filters for the amount of raw channels, which is larger than the number of filtered channels. That's why this is not a problem
                        emg[self.channel_names[k]]['data'] =  filt(emg[self.channel_names[k]]['data']) 
                        
            import matplotlib.pyplot as plt
            # plt.figure()
            # for key in emg.keys():
            #     plt.plot(emg[key]['data'])
            # plt.legend(emg.keys())
            # plt.show(block = False)

            # EMG artifact rejection
            # m = np.zeros([len(emg.keys())])
            # std = np.zeros([len(emg.keys())])
            # for count, key in enumerate(emg.keys()):
            #     m[count] = np.mean(emg[key]['data'])
            #     std[count] = np.std(emg[key]['data'])
            #     if std[count] == 0:
            #         std[count] = 1
            # for count, key in enumerate(emg.keys()):
            #     indpos = np.where(emg[key]['data'] > m[count]+std[count]*10)  
            #     indneg = np.where(emg[key]['data'] < m[count]-std[count]*10) 
            #     ind = np.sort(np.hstack([indpos, indneg]))

            #     if ind.size != 0:

            #         clean_idxs = [idx for idx in np.arange(0,len(emg[key]['data'])) if idx not in ind]    
            #         if np.where(ind == 0) != []:
            #             emg[key]['data'][0] = emg[key]['data'][clean_idxs[0]]
            #         if np.where(ind == len(emg[key]['data'])) != []:
            #             emg[key]['data'][-1] = emg[key]['data'][clean_idxs[-1]]

            #         ind = ind[np.where(ind !=0)]
            #         ind = ind[np.where(ind != len(emg[key]['data']))]
            #         clean_idxs = [idx for idx in np.arange(0,len(emg[key]['data'])) if idx not in ind]                    
            #         clean_data = emg[key]['data'][clean_idxs].copy()
            #         interp_fn = interp1d(clean_idxs, clean_data)
                    
                    
            #         interp_state_data = interp_fn(ind)
            #         # plt.figure(); plt.plot(emg[key]['data'])
            #         emg[key]['data'][ind] = interp_state_data.copy()

                    # plt.plot(emg[key]['data'])
                    # plt.show(block = False)
                    # emg[self.channel_names[k]]['data']  = lfilter(bpf_coeffs[0],bpf_coeffs[1], emg[self.channel_names[k]]['data'] ) 
                    
                    # for b, a in notchf_coeffs:
                    #     emg[self.channel_names[k]]['data']  = lfilter(b = b, a = a, x = emg[self.channel_names[k]]['data'] ) 
                        
            
            # from scipy.io import savemat
            # savemat(os.path.expandvars('$HOME/code/ismore/test_filter.mat'), dict(filtered_data = data_filt, raw_data = emg[self.channel_names[5]]['data']))
            
            

            for i, start_idx in enumerate(start_idxs):
                end_idx = start_idx + n_win_pts 
                
                # samples has shape (n_chan, n_win_pts) 
                samples = np.vstack([emg[chan]['data'][start_idx:end_idx] for chan in self.channel_names]) 
                #if we wanna filter only the segments of data for each output use these lines below and comment lines 210-213 using causal filters
                # for k in range(samples.shape[0]): #for loop on number of electrodes
                #     samples[k] = filtfilt(bpf_coeffs[0],bpf_coeffs[1], samples[k]) 
                #     # plt.figure()
                #     # plt.plot(samples[k], color = 'blue')
                #     for b, a in notchf_coeffs:
                #         samples[k] = filtfilt(b = b, a = a, x = samples[k]) 
                #     # plt.plot(samples[k], color = 'red')
                #     # plt.show()
                
                
                features[i, :] = f_extractor.extract_features(samples).T
                #emg_samples[:,i*n_win_pts:(i+1)*n_win_pts] = f_extractor.extract_filtered_samples(samples)
            # emg_together = np.empty([0, len(emg[self.channel_names[0]]['data'])],emg[self.channel_names[0]]['data'].dtype)
            # #andrea begin
            # if hdf_name in train_hdf_names:
            # #     for k in range(n_channels):
            # #         emg_together = np.vstack([emg_together,emg[:][self.channel_names[k]]['data'] ])
                
            #     if emg_raw is None:                   
            #         emg_raw = emg.copy()
            #     else:
            #         emg_raw = np.hstack([emg_raw, emg])
                 
            # savemat(os.path.expandvars('$HOME/code/ismore/emg_raw.mat'), dict(emg = emg))
            # print "saved emg_raw"
            #emg2 = emg.copy()
            #print len(emg[self.channel_names[0]]['data'])
            # emg_signal_filt = np.empty([0, len(emg[self.channel_names[0]]['data'])],emg[self.channel_names[0]]['data'].dtype)
            # #emg_signal_filtfilt = np.empty([0, len(emg[self.channel_names[0]]['data'])],emg[self.channel_names[0]]['data'].dtype)
            # for k in range(n_channels): #for loop on number of electrodes
            #     for filt in channel_filterbank[k]:
            #         emg[self.channel_names[k]]['data'] = (filt(emg[self.channel_names[k]]['data']))
            #     emg_signal_filt = np.vstack([emg_signal_filt, emg[self.channel_names[k]]['data']])
                

            # for k in range(n_channels): #for loop on number of electrodes
            #     emg2[self.channel_names[k]]['data'] = filtfilt(bpf_coeffs[0],bpf_coeffs[1], emg[self.channel_names[k]]['data']) 
            #     for b, a in notchf_coeffs:
            #         emg2[self.channel_names[k]]['data'] = filtfilt(b = b, a = a, x = emg2[self.channel_names[k]]['data']) 
            #     emg_signal_filtfilt = np.vstack([emg_signal_filtfilt, emg2[self.channel_names[k]]['data']])
            
            
            #andrea end
            
            if hdf_name in train_hdf_names:
                if feature_data_train is None:
                    feature_data_train = features.copy()
                    #emg_filtfilt = emg_signal_filtfilt.copy()
                    #emg_raw = emg.copy()
                    #emg_filt = emg_signal_filt.copy()
                    
                else:
                    feature_data_train = np.vstack([feature_data_train, features])               
                    #emg_filtfilt = np.hstack([emg_filtfilt, emg_signal_filtfilt])
                    #emg_raw = np.hstack([emg_raw, emg])
                    #emg_filt = np.hstack([emg_filt, emg_signal_filt])
                    
                    
            if hdf_name in test_hdf_names:
                if feature_data_test is None:
                    feature_data_test = features.copy()
                else:
                    feature_data_test = np.vstack([feature_data_test, features])
                # plt.figure()
                # plt.plot(feature_data_test[:,3])
                # plt.show()
                # plt.figure()
                # plt.plot(feature_data_test[:,10])
                # plt.show()
                # plt.figure()
                # plt.plot(feature_data_test[:,13])
                # plt.show()
                
            # we will interpolate ArmAssist and/or ReHand data at the times in ts_features
            ts_features = ts_vec[start_idxs + n_win_pts - 1]#[:-1]
            
            # TODO -- a lot of code is repeated below, find way to reduce
            if self.plant_type in ['ArmAssist', 'IsMore']:
                if 'armassist' not in hdf.root:
                    raise Exception('Invalid plant_type %s: no ArmAssist data saved in HDF file.' % self.plant_type)

                for (pos_state, vel_state) in zip(aa_pos_states, aa_vel_states):
                    # differentiate ArmAssist position data to get velocity; 
                    # the ArmAssist application doesn't send velocity 
                    #   feedback data, so it is not saved in the HDF file

                    delta_pos_raw = np.diff(hdf.root.armassist[:]['data'][pos_state])
                    # Use lfilter to filter kinematics
                    lpf = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1]) 
                    delta_pos = lpf(delta_pos_raw)
                    # Use zero phase filter to filter kinematics
                    #delta_pos = filtfilt(bpf_kin_coeffs[0], bpf_kin_coeffs[1], delta_pos_raw)
                    
                    delta_ts  = np.diff(hdf.root.armassist[:]['ts'])
                    vel_state_data = delta_pos / delta_ts
                    ts_data = hdf.root.armassist[1:]['ts_arrival']
                    interp_fn = interp1d(ts_data, vel_state_data)
                    
                    interp_state_data = interp_fn(ts_features)
      
                    
                    mdiff = np.mean(abs(np.diff(interp_state_data)))
                    stddiff = np.std(abs(np.diff(interp_state_data)))
                    if stddiff == 0:
                        stddiff = 1
                    inddiff = np.array(np.where(abs(np.diff(interp_state_data)>(mdiff+stddiff*10))))
                    inddiff = np.sort(np.hstack([inddiff, inddiff +1]))
                    if inddiff.size != 0:
                        clean_idxs = [idx for idx in np.arange(0,len(interp_state_data)) if idx not in inddiff]    
                        if np.where(inddiff == 0) != []:
                            interp_state_data[0] = interp_state_data[clean_idxs[0]]
                        if np.where(inddiff == len(interp_state_data)) != []:
                            interp_state_data[-1] = interp_state_data[clean_idxs[-1]]

                        inddiff = inddiff[np.where(inddiff !=0)]
                        inddiff = inddiff[np.where(inddiff != len(interp_state_data))]
                        clean_idxs = [idx for idx in np.arange(0,len(interp_state_data)) if idx not in inddiff]                    
                        clean_data = interp_state_data[clean_idxs].copy()
                        interp_fn = interp1d(clean_idxs, clean_data)
                    
                        interp_data = interp_fn(inddiff)
                        
                        #plt.figure(); plt.plot(interp_state_data.T)
                        interp_state_data[inddiff] = interp_data.copy()
                        #plt.plot(interp_state_data.T)
                        #plt.show(block = False)
                        #interp_state_data[inddiff] = interp_data.copy()
                    
                      
                               

                    if hdf_name in train_hdf_names:
                        lpf = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1])                
                        kin_data_lpf = lpf(interp_state_data) 
                        #USe zero phase filter
                        #kin_data_lpf = filtfilt(bpf_kin_coeffs[0], bpf_kin_coeffs[1], interp_state_data) 
                        try:
                            kin_data_train[vel_state] = np.concatenate([kin_data_train[vel_state], interp_state_data])
                            kin_data_train_lpf[vel_state] = np.concatenate([kin_data_train_lpf[vel_state], kin_data_lpf])
                        except KeyError:
                            kin_data_train[vel_state] = interp_state_data.copy()
                            kin_data_train_lpf[vel_state] = kin_data_lpf.copy()
                        

                    if hdf_name in test_hdf_names:
                        lpf_test = Filter(bpf_kin_coeffs_test[0], bpf_kin_coeffs_test[1])                
                        kin_data_lpf = lpf_test(interp_state_data) 
                        try:
                            kin_data_test[vel_state] = np.concatenate([kin_data_test[vel_state], interp_state_data])
                            kin_data_test_lpf[vel_state] = np.concatenate([kin_data_test_lpf[vel_state], kin_data_lpf])
                        except KeyError:
                            kin_data_test[vel_state] = interp_state_data.copy()
                            kin_data_test_lpf[vel_state] = kin_data_lpf.copy()
                
                

            if self.plant_type in ['ReHand', 'IsMore']:
                if 'rehand' not in hdf.root:
                    raise Exception('Invalid plant_type %s: no ReHand data saved in HDF file.' % self.plant_type)

                for vel_state in rh_vel_states:
                    ts_data    = hdf.root.rehand[:]['ts_arrival']
                    state_data = hdf.root.rehand[:]['data'][vel_state]
                    interp_fn = interp1d(ts_data, state_data)
                    interp_state_data = interp_fn(ts_features)

                    mdiff = np.mean(abs(np.diff(interp_state_data)))
                    stddiff = np.std(abs(np.diff(interp_state_data)))
                    if stddiff == 0:
                        stddiff = 1
                    inddiff = np.array(np.where(abs(np.diff(interp_state_data)>(mdiff+stddiff*10))))
                    inddiff = np.sort(np.hstack([inddiff, inddiff +1]))
                    if inddiff.size != 0:
                        clean_idxs = [idx for idx in np.arange(0,len(interp_state_data)) if idx not in inddiff]    
                        if np.where(inddiff == 0) != []:
                            interp_state_data[0] = interp_state_data[clean_idxs[0]]
                        if np.where(inddiff == len(interp_state_data)) != []:
                            interp_state_data[-1] = interp_state_data[clean_idxs[-1]]
                            
                        inddiff = inddiff[np.where(inddiff !=0)]
                        inddiff = inddiff[np.where(inddiff != len(interp_state_data))]
                        clean_idxs = [idx for idx in np.arange(0,len(interp_state_data)) if idx not in inddiff]                    
                        clean_data = interp_state_data[clean_idxs].copy()
                        interp_fn = interp1d(clean_idxs, clean_data)
                    
                        interp_data = interp_fn(inddiff)
                        #plt.figure(); plt.plot(interp_state_data.T)
                        interp_state_data[inddiff] = interp_data.copy()
                        #plt.plot(interp_state_data.T)
                        #plt.show(block = False)

                        #interp_state_data[inddiff] = interp_data.copy()

                                                 
                    if hdf_name in train_hdf_names:
                        # lpf = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1]) 
                        # kin_data_lpf = lpf(interp_state_data) 
                        #USe zero phase filter
                        kin_data_lpf = filtfilt(bpf_kin_coeffs[0], bpf_kin_coeffs[1], interp_state_data) 
                        try:
                            kin_data_train[vel_state] = np.concatenate([kin_data_train[vel_state], interp_state_data])
                            kin_data_train_lpf[vel_state] = np.concatenate([kin_data_train_lpf[vel_state], kin_data_lpf])
                        except KeyError:
                            kin_data_train[vel_state] = interp_state_data.copy()
                            kin_data_train_lpf[vel_state] = kin_data_lpf.copy()
                        

                    if hdf_name in test_hdf_names:
                        lpf_test = Filter(bpf_kin_coeffs_test[0], bpf_kin_coeffs_test[1]) 
                        kin_data_lpf = lpf_test(interp_state_data) 
                        try:
                            kin_data_test[vel_state] = np.concatenate([kin_data_test[vel_state], interp_state_data])
                            kin_data_test_lpf[vel_state] = np.concatenate([kin_data_test_lpf[vel_state], kin_data_lpf])
                        except KeyError:
                            kin_data_test[vel_state] = interp_state_data.copy()
                            kin_data_test_lpf[vel_state] = kin_data_lpf.copy()
                
        self.features_mean = np.mean(feature_data_train, axis=0)
        
        if self.fixed_var_scalar:
            self.features_std = np.zeros_like(self.features_mean)
            Z_features_train = np.zeros_like(feature_data_train)

            self.fixed_var_ft_ix = {}
            for ft in self.extractor_kwargs['feature_names']:
                ix = np.array([i for i, j in enumerate(self.extractor_kwargs['emg_feature_name_list']) if ft in j])
                self.fixed_var_ft_ix[ft] = ix
                self.fixed_var_ft_ix[ft, 'std_scalar'] = np.mean(np.std(feature_data_train[:, ix], axis=0))
                Z_features_train[:, ix] = (feature_data_train[:, ix] - self.features_mean[ix][np.newaxis, :] ) / self.fixed_var_ft_ix[ft, 'std_scalar']
                self.features_std[ix] = self.fixed_var_ft_ix[ft, 'std_scalar']
        
        else:
            self.features_std = np.std(feature_data_train, axis=0)
            Z_features_train = (feature_data_train - self.features_mean) / self.features_std
        
        # Use these features to start the task:
        self.recent_features_std = self.features_std
        self.recent_features_mean = self.features_mean

        # from scipy.io import savemat
        # savemat(os.path.expandvars('$HOME/code/ismore/features_corrected.mat'), dict(train_features = feature_data_train, test_features = feature_data_test, train_features_norm = Z_features_train))
        # print "saved emg_filt"

      
        # To concatenate the kinematics from all the runs
        # kin_data_vel = np.vstack([kin_data_train[key] for key in ismore_vel_states])
        # kin_data_vel_lpf = np.vstack([kin_data_train_lpf[key] for key in ismore_vel_states])
        # kin_data_vel_test = np.vstack([kin_data_test[key] for key in ismore_vel_states])
        # kin_data_vel_lpf_test = np.vstack([kin_data_test_lpf[key] for key in ismore_vel_states])
        

        # plt.figure()
        # plt.plot(feature_data_train[:,6])
        # plt.plot(kin_data_train_lpf['aa_vy'], color = 'red')
        # plt.show()

        # train vector of coefficients for each DoF using ridge regression
        self.beta = dict()
        for state in kin_data_train_lpf:
            
            if not self.subset_features:
                # Use all the muscles for the decoding of all DoFs.
                self.beta[state] = ridge(kin_data_train_lpf[state].reshape(-1,1), Z_features_train, K, zscore=False)
            else: 
                # Use a subset of muscles for each DoF decoding 
                self.beta[state] = ridge(kin_data_train_lpf[state].reshape(-1,1), Z_features_train[:,self.subset_features[state]], K, zscore=False)
        for state in states_to_flip:
            if state in self.beta:
                self.beta[state] *= -1.0
        #savemat("/home/tecnalia/code/ismore/python_train_feats.mat", dict(beta = self.beta, emg = emg_signal, train_features=feature_data_train, kin_data=kin_data_vel,kin_data_lpf=kin_data_vel_lpf, train_features_norm=Z_features_train, features_mean = self.features_mean))
        #print "saved data"
        # test coefficients for each DoF on testing data
        #self.features_mean = np.mean(feature_data_test, axis=0)#andrea
        #self.features_std = np.std(feature_data_test, axis=0)#andrea
        # plt.figure()
        # plt.plot(feature_data_test[:,3])
        # plt.show()
        # plt.figure()
        # plt.plot(feature_data_test[:,10])
        # plt.show()
        # plt.figure()
        # plt.plot(feature_data_test[:,13])
        # plt.show()
        
        if test_hdf_names != []:

            self.features_mean_test = np.zeros((feature_data_test.shape[0] - window, feature_data_test.shape[1]))
            self.features_std_test = np.zeros((feature_data_test.shape[0] - window, feature_data_test.shape[1])) 
            
            for n in range(0,feature_data_test.shape[0] - window):

                self.features_mean_test[n,:] = np.mean(feature_data_test[n+1:n+window], axis = 0)
                self.features_std_test[n,:] = np.std(feature_data_test[n+1:n+window], axis = 0)

                #self.features_mean_test[n,:] = np.mean(feature_data_test[n:n+window -1], axis = 0)
                #self.features_std_test[n,:] = np.std(feature_data_test[n:n+window -1], axis = 0)


            self.features_std_test[self.features_std_test == 0] = 1
            feature_data_test = feature_data_test [window:]
            Z_features_test = (feature_data_test - self.features_mean_test) / self.features_std_test
        
        
            cc_values_raw = dict()
            cc_values_lpf = dict()
            nrmse_raw = dict()
            nrmse_lpf = dict()
            for state in kin_data_test:
                
                if not self.subset_features:
                    # Use all the muscles for the decoding of all DoFs.
                    pred_kin_data_raw = np.dot(Z_features_test, self.beta[state])
                    pred_kin_data = np.dot(Z_features_test, self.beta[state])
                    pred_kin_data_lpf = np.dot(Z_features_test, self.beta[state])
                else: 
                    # Use a subset of muscles for each DoF decoding 
                    pred_kin_data_raw = np.dot(Z_features_test[:,self.subset_features[state]], self.beta[state])
                    pred_kin_data = np.dot(Z_features_test[:,self.subset_features[state]], self.beta[state])
                    pred_kin_data_lpf = np.dot(Z_features_test[:,self.subset_features[state]], self.beta[state])
            
                for index in range(len(pred_kin_data)):  #andrea - weighted mov avge             
                    win = min(9,index)
                    weights = np.arange(1./(win+1), 1 + 1./(win+1), 1./(win+1))
                    pred_kin_data_lpf[index] = np.sum(weights*pred_kin_data[index-win:index+1].ravel())/np.sum(weights)#len(pred_kin_data[index-win:index+1])
                        
                # for index in range(len(pred_kin_data)):  #andrea - non-weighted mov avge             
                #     win = min(9,index)
                #     pred_kin_data_lpf[index] = np.mean(pred_kin_data[index-win:index+1].ravel())/len(pred_kin_data[index-win:index+1])
                #     if np.isnan(pred_kin_data_lpf[index]):

                
                
                kin_data_test[state] = kin_data_test[state][window:].reshape(-1,1)
                kin_data_test_lpf[state] = kin_data_test_lpf[state][window:].reshape(-1,1)
                cc_values_lpf[state] = pearsonr(kin_data_test_lpf[state], pred_kin_data_lpf)[0]
                #cc_values_lpf[state] = pearsonr(kin_data_test_lpf[state], pred_kin_data)[0]
                #cc_values_2[state] = pearsonr(kin_data_test[state], pred_kin_data_2)[0]
                cc_values_raw[state] = pearsonr(kin_data_test[state], pred_kin_data_raw)[0]
                
                nrmse_lpf[state] = math.sqrt(math.fsum(np.square(kin_data_test_lpf[state] - pred_kin_data_lpf))/len(kin_data_test_lpf[state]))/(np.amax(kin_data_test_lpf[state]) - np.amin(kin_data_test_lpf[state]))
                
                nrmse_raw[state] = math.sqrt(math.fsum(np.square(kin_data_test[state] - pred_kin_data_raw))/len(kin_data_test[state]))/(np.amax(kin_data_test[state]) - np.amin(kin_data_test[state]))
            
                plt.figure()
                plt.plot(kin_data_test[state], color = 'blue')
                plt.plot(kin_data_test_lpf[state], color = 'brown')
                plt.plot(pred_kin_data_raw, color = 'black')
                plt.plot(pred_kin_data_lpf, color = 'green')
                plt.title(state)
                plt.legend(['original_raw', 'original_lpf','predicted_raw','predicted_lpf'])
                plt.show(block = False)
            #savemat(os.path.expandvars('$HOME/code/ismore/python_train_feats.mat'), dict(beta = self.beta, train_features=feature_data_train, test_features = feature_data_test,  kin_data=kin_data_vel,kin_data_lpf=kin_data_vel_lpf,kin_data_test=kin_data_vel_test,kin_data_lpf_test=kin_data_vel_lpf_test,  train_features_norm=Z_features_train, test_features_norm=Z_features_test, features_mean_test = self.features_mean_test, pred_kin_data = pred_kin_data, pred_kin_data_raw = pred_kin_data_raw))
            #print "saved data"
            #savemat(os.path.expandvars('$HOME/code/ismore/predicted_vindex.mat'), dict(pred = pred_kin_data_raw, pred_lpf = pred_kin_data_lpf, kin_data_test = kin_data_test[state], kin_data_test_lpf =kin_data_test_lpf[state]))

            print cc_values_raw
            print cc_values_lpf
            print nrmse_raw
            print nrmse_lpf
        #print cc_values_lpf
        #plt.figure(); plt.plot(kin_data_test_lpf['aa_vx']); plt.plot(pred_kin_data_lpf);  plt.show(block = False)
        # TODO -- set gamma_coeffs manually for now
        # self.gamma_coeffs = pd.Series(0.0, self.states_to_decode)
        # if self.plant_type in ['ArmAssist', 'IsMore']:
        #     self.gamma_coeffs['aa_vx']     = 0.9
        #     self.gamma_coeffs['aa_vy']     = 0.9
        #     self.gamma_coeffs['aa_vpsi']   = 0.9
        # if self.plant_type in ['ReHand', 'IsMore']:
        #     self.gamma_coeffs['rh_vthumb'] = 0.9
        #     self.gamma_coeffs['rh_vindex'] = 0.9
        #     self.gamma_coeffs['rh_vfing3'] = 0.9
        #     self.gamma_coeffs['rh_vprono'] = 0.9
       


def ridge(Y, X, K, zscore=True):
    '''
    Same as MATLAB's ridge regression function.
    '''

    p = X.shape[1]
    if zscore:
        Z = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    else:
        Z = X

    Z = np.mat(Z)
    Y = np.mat(Y)
    W = np.array(np.linalg.pinv(Z.T * Z + K*np.mat(np.eye(p))) * Z.T*Y)

    return W
    #return np.linalg.pinv(Z.T.dot(Z) + K*np.identity(p)).dot(Z.T).dot(Y)