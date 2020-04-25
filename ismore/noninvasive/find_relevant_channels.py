import tables
import numpy as np
import math

from ismore import brainamp_channel_lists
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.signal import butter, lfilter, filtfilt
from ismore.noninvasive.emg_feature_extraction import EMGMultiFeatureExtractor

from ismore.common_state_lists import *
from ismore import ismore_bmi_lib

# from riglib.filter import Filter
from ismore.filter import Filter

channels_str_2discard = [5,11,17,23,29,35,41,47,49,51,53,55,57]
channels_str_2keep = [i for i in range(59) if i not in channels_str_2discard]
channels_diag1_1 = [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46]
channels_diag1_2 = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41]
channels_diag2_1 = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40]
channels_diag2_2 = [7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47]

channels_str_2keep_mirrored = [18,19,20,21,22,12,13,14,15,16,6,7,8,9,10,0,1,2,3,4,42,43,44,45,46,36,37,38,39,40,30,31,32,33,34,24,25,26,27,28,48,50,52,54,56,58]
channels_diag1_1_mirrored = [12,13,14,15,16,6,7,8,9,10,0,1,2,3,4,36,37,38,39,40,30,31,32,33,34,24,25,26,27,28]
channels_diag1_2_mirrored = [19,20,21,22,23,13,14,15,16,17,7,8,9,10,11,43,44,45,46,47,37,38,39,40,41,31,32,33,34,35]
channels_diag2_1_mirrored = [18,19,20,21,22,12,13,14,15,16,6,7,8,9,10,42,43,44,45,46,36,37,38,39,40,30,31,32,33,34]
channels_diag2_2_mirrored = [13,14,15,16,17,7,8,9,10,11,1,2,3,4,5,37,38,39,40,41,31,32,33,34,35,25,26,27,28,29]

# class RelevantChannels(object):
#     '''
#     Abstract base class for all relevant channel finders classes
#     '''
#     pass

class FindRelevantChannels(object):
    def __init__(self, train_hdf_names, test_relevant_channels_hdf_names, channels_2train, plant_type, filt_training_data, extractor_kwargs, nchannels_2select, relevant_dofs, min_HD_nchans, mirrored):

        # self.recorded_channels = channels_2train
        self.plant_type     = plant_type
        self.fs             = extractor_kwargs['fs']
        self.win_len        = extractor_kwargs['win_len']
        self.filt_training_data = filt_training_data
        self.K = extractor_kwargs['K']
        self.relevant_dofs =  relevant_dofs
        self.nrelevant_dofs = len(relevant_dofs)
        self.train_hdf_names = train_hdf_names
        self.test_relevant_channels_hdf_names = test_relevant_channels_hdf_names

        self.opt_channels = list()
        self.opt_channels_dict = dict()
        self.feature_names = extractor_kwargs['feature_names']
        self.feature_fn_kwargs = extractor_kwargs['feature_fn_kwargs']
        self.n_feats = len(self.feature_names)
        #self.brainamp_channels = extractor_kwargs['brainamp_channels']
        # channel names in HDF file have 'chan' in front (e.g., 'chanBiceps'
        self.recorded_channels = brainamp_channel_lists.emg_48hd_6mono
        self.recorded_channels = ['chan' + name for name in self.recorded_channels]
        #self.channel_names = ['chan' + name for name in self.channels]
        
        ssm = ismore_bmi_lib.SSM_CLS_DICT[plant_type]()
        self.states_to_decode = [s.name for s in ssm.states if s.order == 1]
        self.ndofs = len(self.states_to_decode)
        self.extractor_chanfinder = EMGMultiFeatureExtractor
        self.extractor_kwargs = extractor_kwargs

        # self.chan_2keep = channels_dict['channels_str_2keep']
        # #self.chan_2discard = channels_dict['channels_str_2discard']
        # self.chan_diag1_1 = channels_dict['channels_diag1_1']
        # self.chan_diag1_2 = channels_dict['channels_diag1_2']
        # self.chan_diag2_1 = channels_dict['channels_diag2_1']
        # self.chan_diag2_2 = channels_dict['channels_diag2_2']

        self.channel_names = [str(i) + 'str_filt' for i in range(len(channels_str_2keep)-6)] + brainamp_channel_lists.emg_6bip_hd_filt + [str(j) + 'diag1_filt' for j in range(len(channels_diag1_1))] + [str(k) + 'diag2_filt' for k in range(len(channels_diag2_1))]
        
        self.bip_idxs = [ind for ind, c in enumerate(self.channel_names) if c in brainamp_channel_lists.emg_6bip_hd_filt]
        self.nchans_hd = len(self.channel_names) - len(self.bip_idxs)
        self.nchans_bip = len(self.bip_idxs)
        self.mirrored = mirrored

        self.nchannels_2select = nchannels_2select
        self.min_HD_nchans = min_HD_nchans
        self.min_nfeats = self.min_HD_nchans*self.n_feats #minimum number of features from the HD-EMG that will be kept
        
    def __call__(self):
        
        # from scipy import io
        # import pdb; pdb.set_trace()
        # #data = io.loadmat('code/ismore/rejected_channels.mat')
        # import os
        # data = io.loadmat(os.path.expandvars('$HOME/code/ismore/rejected_channels.mat'))

        # rejected_channels = data['rej_channels']
        # cc_lpf = data['cc_lpf']


        self.n_iter = len(self.channel_names) - self.min_HD_nchans + 1 - 6
        rejected_channels, CC = self.ridge_chan_iterations(self.train_hdf_names, self.test_relevant_channels_hdf_names)

        import pickle
        # data = pickle.load(open('/storage/results_relevant_channels_AM_6142_6146_relevant_DOFs.pkl'))

        #data = pickle.load(open('/storage/results_relevant_channels_' + self.train_hdf_names[0][21:23] + '_' + self.train_hdf_names[0][-8:-4] + '_' + self.train_hdf_names[-1][-8:-4] + '.pkl'))
        #rejected_channels = data['rej_channels']
        #CC = data['cc_lpf']

        #import time
        # t0 = time.time()
        #rejected_channels, CC = self.ridge_chan_iterations(self.train_hdf_names, self.test_relevant_channels_hdf_names)
        # print "ridge_chan_iterations", (time.time() - t0)
        

    
        self.opt_channels, self.opt_channels_dict, self.opt_channels_2train_dict = self.relevant_channels(rejected_channels, CC)
        

        return self.opt_channels, self.opt_channels_dict, self.opt_channels_2train_dict

    def ridge_chan_iterations(self, train_hdf_names, test_relevant_channels_hdf_names):
        
        train_hdf_names = [name for name in train_hdf_names if name not in test_relevant_channels_hdf_names]
        
        #self.train_hdf_names = train_hdf_names
        #self.test_relevant_channels_hdf_names  = test_relevant_channels_hdf_names

        n_features_hd = self.n_feats * self.nchans_hd
        n_features_bip = self.n_feats * self.nchans_bip

        feature_data_train_hd = None
        feature_data_train_bip = None
        feature_data_test_hd  = None
        feature_data_test_bip = None

        fs_synch = 20 #Frequency at which emg and kin data are synchronized
        nyq   = 0.5 * fs_synch
        cuttoff_freq  = 1.5 / nyq
        bpf_kin_coeffs = butter(4, cuttoff_freq, btype='low')
        lpf = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1])   
        
        kin_data_train = dict()
        kin_data_train_lpf = dict()
        kin_data_test  = dict()
        kin_data_test_lpf  = dict()
        kin_data_filts = dict()

        window = int(60 * fs_synch) # sliding window used to normalize the test signal
        #andrea
        # calculate coefficients for a 4th-order Butterworth BPF from 10-450 Hz
        band  = [10, 450]  # Hz
        nyq   = 0.5 * self.fs
        low   = band[0] / nyq
        high  = band[1] / nyq
        bpf_coeffs = butter(4, [low, high], btype='band')
        # self.band_pass_filter = Filter(self.bpf_coeffs[0], self.bpf_coeffs[1])

        # calculate coefficients for multiple 2nd-order notch filters
        notchf_coeffs = []
        for freq in [50, 150, 250, 350]:
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

        f_extractor = self.extractor_chanfinder(None, emg_channels = [], feature_names = self.feature_names, feature_fn_kwargs = self.feature_fn_kwargs, win_len=self.win_len, fs=self.fs)#, brainamp_channels = self.brainamp_channels)
        # f_extractor = self.extractor_cls(None, **self.extractor_kwargs)


        all_hdf_names = train_hdf_names + [name for name in test_relevant_channels_hdf_names if name not in train_hdf_names]

        for hdf_name in all_hdf_names:
            # load EMG data from HDF file
            hdf = tables.openFile(hdf_name)
            
            store_dir_supp = '/storage/supp_hdf/'
            index_slash = hdf_name.encode('ascii','ignore').rfind('/')            
            hdf_supp_name = store_dir_supp + hdf_name.encode('ascii','ignore')[index_slash + 1:-3] + 'supp.hdf'
            
            hdf_supp = tables.open_file(hdf_supp_name)
                           
            emg = hdf_supp.root.brainamp[:][self.recorded_channels]
            original_ts = emg[self.recorded_channels[0]]['ts_arrival']

            idx = 1            
            while original_ts[idx] == original_ts[0]:
                idx = idx + 1
            # idx = idx - 1
            ts_step = 1./self.fs
            # ts_before = original_ts[idx] + (ts_step * np.arange(-idx, 0))
            # ts_after = original_ts[idx] + (ts_step * np.arange(1, len(original_ts)))
            # ts_vec = np.hstack([ts_before, original_ts[idx], ts_after])
            ts_emg = np.arange(original_ts[0]-(ts_step*(idx-1)),original_ts[-1],ts_step)
            #import pdb; pdb.set_trace() 
            if self.plant_type in ['ArmAssist', 'IsMore']:
                if 'armassist' not in hdf.root:
                    raise Exception('Invalid plant_type %s: no ArmAssist data saved in HDF file.' % self.plant_type)

                else:
                    ts_aa = hdf.root.armassist[1:]['ts_arrival'] #get rid of first time stamp because we are gonna compute the velocity from the position--> one sample point less

            if self.plant_type in ['ReHand', 'IsMore']:
                if 'rehand' not in hdf.root:
                    raise Exception('Invalid plant_type %s: no ReHand data saved in HDF file.' % self.plant_type)
                
                else:
                    ts_rh = hdf.root.rehand[:]['ts_arrival']

            if 'ts_rh' in locals() and 'ts_aa' in locals():
                ts_max = min(ts_emg[-1], ts_aa[-1], ts_rh[-1])
                ts_min = max(ts_emg[0], ts_aa[0], ts_rh[0])
            elif 'ts_rh' not in locals() and 'ts_aa' in locals():
                ts_max = min(ts_emg[-1], ts_aa[-1])
                ts_min = max(ts_emg[0], ts_aa[0])
            elif 'ts_rh' in locals() and 'ts_aa' not in locals():
                ts_max = min(ts_emg[-1], ts_rh[-1])
                ts_min = max(ts_emg[0], ts_rh[0])
            else:
                ts_max = ts_emg[-1]
                ts_min = ts_emg[0]

            ts_vec = ts_emg[(ts_emg < ts_max) * (ts_emg > ts_min)]

            n_win_pts = int(self.win_len * self.fs)
            
            step_pts = 50  # TODO -- don't hardcode This is in sample points! Be careful if fs is changed!!
            start_idxs = np.arange(0, len(ts_vec) - n_win_pts + 1, step_pts)
            features_hd = np.zeros((len(start_idxs), n_features_hd))
            features_bip = np.zeros((len(start_idxs), n_features_bip))
            
            diag_emg = np.ndarray([len(self.channel_names),len(emg[self.recorded_channels[0]]['data'])])

            data = np.zeros([len(self.recorded_channels),len(emg[self.recorded_channels[0]]['data'])])
                
            for k in range(len(self.recorded_channels)):
                data[k,:] = emg[self.recorded_channels[k]]['data']
            data_diff = np.diff(data, axis = 0)


            for i in range(n_channels):
                if i < len(channels_str_2keep):
                    #import pdb; pdb.set_trace()
                    diag_emg[i,:] = data_diff[channels_str_2keep[i],:]
                elif i < (len(channels_str_2keep) + len(channels_diag1_1)):
                    diag_emg[i,:] = emg[self.recorded_channels[channels_diag1_1[i-len(channels_str_2keep)]]]['data'] - emg[self.recorded_channels[channels_diag1_2[i-len(channels_str_2keep)]]]['data']
                else:
                    diag_emg[i,:] = emg[self.recorded_channels[channels_diag2_1[i-len(channels_str_2keep)-len(channels_diag1_1)]]]['data'] - emg[self.recorded_channels[channels_diag2_2[i-len(channels_str_2keep)-len(channels_diag1_1)]]]['data']
                for filt in channel_filterbank[i]: #channel_filterbank_emg has filters for the amount of raw channels, which is larger than the number of filtered channels. That's why this is not a problem
                    diag_emg[i,:] =  filt(diag_emg[i,:]) 
            
                # EMG artifact rejection
                # m = np.mean(diag_emg[i,:])
                # std = np.std(diag_emg[i,:])
                # try:
                #     std[np.where(std == 0)] = 1
                # except:
                #     pass
                # #import pdb; pdb.set_trace()
                # indpos = np.where(diag_emg[i,:] > m+std*10)  
                # indneg = np.where(diag_emg[i,:] < m-std*10) 
                # ind = np.sort(np.hstack([indpos, indneg]))

                # if ind.size != 0:
                    
                #     clean_idxs = [idx for idx in np.arange(0,len(diag_emg[i,:])) if idx not in ind]    
                #     if np.where(ind == 0) != []:
                #         diag_emg[i,0] = diag_emg[i,clean_idxs[0]]
                #     if np.where(ind == len(diag_emg[i,:])) != []:
                #         diag_emg[i,-1] = diag_emg[i,clean_idxs[-1]]

                #     ind = ind[np.where(ind !=0)]
                #     ind = ind[np.where(ind != len(diag_emg[i,:]))]
                #     clean_idxs = [idx for idx in np.arange(0,len(diag_emg[i,:])) if idx not in ind]                     
                #     clean_data = diag_emg[i, clean_idxs].copy()
                #     interp_fn = interp1d(clean_idxs, clean_data)

                #     #import pdb; pdb.set_trace()
                #     interp_state_data = interp_fn(ind)
                    
                #     diag_emg[i, ind] = interp_state_data.copy()

            for i, start_idx in enumerate(start_idxs):
                end_idx = start_idx + n_win_pts
                # samples has shape (n_chan, n_win_pts) 
                samples_bip = np.vstack([diag_emg[ind,start_idx:end_idx] for ind in self.bip_idxs])
                samples_hd = np.vstack([diag_emg[ind,start_idx:end_idx] for ind in range(len(self.channel_names)) if ind not in self.bip_idxs]) 
                #samples = np.vstack([samples, samples_bip])
                features_hd[i, :] = f_extractor.extract_features(samples_hd).T
                features_bip[i, :] = f_extractor.extract_features(samples_bip).T

            if hdf_name in train_hdf_names:
                if feature_data_train_hd is None:
                    feature_data_train_hd = features_hd.copy()
                    feature_data_train_bip = features_bip.copy()

                else:
                    feature_data_train_hd = np.vstack([feature_data_train_hd, features_hd])  
                    feature_data_train_bip = np.vstack([feature_data_train_bip, features_bip])               
                    #emg_filtfilt = np.hstack([emg_filtfilt, emg_signal_filtfilt])
                    #emg_raw = np.hstack([emg_raw, emg])
                    #emg_filt = np.hstack([emg_filt, emg_signal_filt])
                    
                    
            if hdf_name in test_relevant_channels_hdf_names:
                if feature_data_test_hd is None:
                    feature_data_test_hd = features_hd.copy()
                    feature_data_test_bip = features_bip.copy()

                else:
                    feature_data_test_hd = np.vstack([feature_data_test_hd, features_hd])  
                    feature_data_test_bip = np.vstack([feature_data_test_bip, features_bip])  
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
            #import pdb; pdb.set_trace()
            ts_features = ts_vec[start_idxs + n_win_pts - 1]#[:-1] 

            if self.plant_type in ['ArmAssist', 'IsMore']:
                if 'armassist' not in hdf.root:
                    raise Exception('Invalid plant_type %s: no ArmAssist data saved in HDF file.' % self.plant_type)

                for (pos_state, vel_state) in zip(aa_pos_states, aa_vel_states):
                    # differentiate ArmAssist position data to get velocity; 
                    # the ArmAssist application doesn't send velocity 
                    #   feedback data, so it is not saved in the HDF file
                    delta_pos_raw = np.diff(hdf.root.armassist[:]['data'][pos_state])
                    lpf = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1])   
                    delta_pos = lpf(delta_pos_raw)
                    delta_ts  = np.diff(hdf.root.armassist[:]['ts'])
                    vel_state_data = delta_pos / delta_ts
                    ts_data = hdf.root.armassist[1:]['ts_arrival']#we need one ts point less because we have computed the velocity
                    interp_fn = interp1d(ts_data, vel_state_data)
                    if ts_data[0] > ts_features[0] or ts_data[-1] < ts_features[-1]:
                        import pdb; pdb.set_trace()
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
                        interp_state_data[inddiff] = interp_data.copy()

                    lpf = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1])   
                    kin_data_lpf = lpf(interp_state_data)                    

                    if hdf_name in train_hdf_names:
                        try:
                            kin_data_train[vel_state] = np.concatenate([kin_data_train[vel_state], interp_state_data])
                            kin_data_train_lpf[vel_state] = np.concatenate([kin_data_train_lpf[vel_state], kin_data_lpf])
                        except KeyError:
                            kin_data_train[vel_state] = interp_state_data.copy()
                            kin_data_train_lpf[vel_state] = kin_data_lpf.copy()
                        

                    if hdf_name in test_relevant_channels_hdf_names:
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
                        interp_state_data[inddiff] = interp_data.copy()

                    lpf = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1])   
                    kin_data_lpf = lpf(interp_state_data)                              
                    if hdf_name in train_hdf_names:
                        try:
                            kin_data_train[vel_state] = np.concatenate([kin_data_train[vel_state], interp_state_data])
                            kin_data_train_lpf[vel_state] = np.concatenate([kin_data_train_lpf[vel_state], kin_data_lpf])
                        except KeyError:
                            kin_data_train[vel_state] = interp_state_data.copy()
                            kin_data_train_lpf[vel_state] = kin_data_lpf.copy()
                        

                    if hdf_name in test_relevant_channels_hdf_names:
                        try:
                            kin_data_test[vel_state] = np.concatenate([kin_data_test[vel_state], interp_state_data])
                            kin_data_test_lpf[vel_state] = np.concatenate([kin_data_test_lpf[vel_state], kin_data_lpf])
                        except KeyError:
                            kin_data_test[vel_state] = interp_state_data.copy()
                            kin_data_test_lpf[vel_state] = kin_data_lpf.copy()
            hdf.close()
            hdf_supp.close()

        for dof in range(self.ndofs):
            kin_data_test[self.states_to_decode[dof]] = kin_data_test[self.states_to_decode[dof]][window:].reshape(-1,1)
            kin_data_test_lpf[self.states_to_decode[dof]] = kin_data_test_lpf[self.states_to_decode[dof]][window:].reshape(-1,1)
        
        #start loop on channels here
        self.n_iter = int(round((feature_data_train_hd.shape[1] - self.min_nfeats)/self.n_feats) + 1)
        #import pdb; pdb.set_trace()
        Beta = dict()
        cc_raw = dict()
        cc_lpf = dict()
        Yout = dict()
        Ytest = dict()
        channels_in = np.arange(0,feature_data_train_hd.shape[1])
        rej_channels = []
        rej_idxs = []
        for it in range(self.n_iter):
            #it += 1
            if it > 0:
                chans_left = Beta[str(it-1)].shape[1]/self.n_feats -self.nchans_bip
                mean_Beta_DOFs = np.ndarray([chans_left,self.nrelevant_dofs])
                for dof,rel_dof in enumerate(self.relevant_dofs):
                    mean_Beta_DOFs[:,dof] = np.mean(abs(Beta[str(it-1)][int(rel_dof),:-self.n_feats*self.nchans_bip].reshape([self.n_feats,chans_left]).T), axis = 1) 
                mean_Beta = np.mean(mean_Beta_DOFs, axis = 1)
                idx_min_weight = np.where(mean_Beta == min(mean_Beta))
                if rej_idxs == []:
                    rej_idxs = idx_min_weight[0][0] + self.nchans_hd*np.arange(0,self.n_feats)
                else:
                    rej_idxs = np.vstack([rej_idxs, idx_min_weight[0][0] + self.nchans_hd*np.arange(0,self.n_feats)])
                new_rej_channels = channels_in[idx_min_weight[0][0]] + self.nchans_hd*np.arange(0,self.n_feats)
                if rej_channels == []:
                    rej_channels = new_rej_channels
                else:
                    rej_channels = np.vstack([rej_channels, new_rej_channels])
                channels_in = np.array([chan_in for chan_in in channels_in if chan_in not in new_rej_channels])
            
            for dof in range(self.ndofs):
                feature_data_train = np.hstack([feature_data_train_hd[:,channels_in], feature_data_train_bip])
                feature_data_test = np.hstack([feature_data_test_hd[:,channels_in], feature_data_test_bip])
                self.features_mean = np.mean(feature_data_train, axis=0)
                self.features_std = np.std(feature_data_train, axis=0)
                Z_features_train = (feature_data_train - self.features_mean) / self.features_std

                self.features_mean_test = np.zeros((feature_data_test.shape[0] - window, feature_data_test.shape[1]))
                self.features_std_test = np.zeros((feature_data_test.shape[0] - window, feature_data_test.shape[1])) 


                for n in range(0,feature_data_test.shape[0] - window):
                    self.features_mean_test[n,:] = np.mean(feature_data_test[n+1:n+window], axis = 0)
                    self.features_std_test[n,:] = np.std(feature_data_test[n+1:n+window], axis = 0)
        
                self.features_std_test[self.features_std_test == 0] = 1

                feature_data_test = feature_data_test[window:]
                Z_features_test = (feature_data_test - self.features_mean_test) / self.features_std_test
                
                beta = ridge(kin_data_train_lpf[self.states_to_decode[dof]].reshape(-1,1), Z_features_train, self.K, zscore=False)
                #import pdb; pdb.set_trace()
                if len(Beta) < it + 1:
                    Beta[str(it)] = beta.T
                else:
                    Beta[str(it)] = np.vstack([Beta[str(it)], beta.T])
       
                
                pred_kin_data_raw = np.dot(Z_features_test, beta)
                pred_kin_data = np.dot(Z_features_test, beta)
                pred_kin_data_lpf = np.dot(Z_features_test, beta)

                for index in range(len(pred_kin_data)):  #andrea - weighted mov avge             
                    win = min(9,index)
                    weights = np.arange(1./(win+1), 1 + 1./(win+1), 1./(win+1))
                    pred_kin_data_lpf[index] = np.sum(weights*pred_kin_data[index-win:index+1].ravel())/np.sum(weights)#len(pred_kin_data[index-win:index+1])
                    if np.isnan(pred_kin_data_lpf[index]):
                        import pdb; pdb.set_trace()
           
            
                
                # if len(cc_lpf) < it + 1:
                #     cc_lpf[str(it)] = pearsonr(kin_data_test_lpf[self.states_to_decode[dof]], pred_kin_data_lpf)[0]
                #     cc_raw[str(it)] = pearsonr(kin_data_test[self.states_to_decode[dof]], pred_kin_data_raw)[0]
                #     Yout[str(it)] = pred_kin_data_lpf
                #     Ytest[str(it)] = kin_data_test_lpf[self.states_to_decode[dof]]
                # else:
                #     cc_lpf[str(it)] = np.vstack([cc_lpf[str(it)], pearsonr(kin_data_test_lpf[self.states_to_decode[dof]], pred_kin_data_lpf)[0]])
                #     cc_raw[str(it)] = np.vstack([cc_raw[str(it)], pearsonr(kin_data_test[self.states_to_decode[dof]], pred_kin_data_raw)[0]])
                #     Yout[str(it)] = np.vstack([Yout[str(it)], pred_kin_data_lpf])
                #     Ytest[str(it)] = np.vstack([Ytest[str(it)], kin_data_test_lpf[self.states_to_decode[dof]]])
        
                if it < 1:
                    cc_lpf[str(dof)] = pearsonr(kin_data_test_lpf[self.states_to_decode[dof]], pred_kin_data_lpf)[0]
                    cc_raw[str(dof)] = pearsonr(kin_data_test[self.states_to_decode[dof]], pred_kin_data_raw)[0]
                else:
                    cc_lpf[str(dof)] = np.hstack([cc_lpf[str(dof)], pearsonr(kin_data_test_lpf[self.states_to_decode[dof]], pred_kin_data_lpf)[0]])
                    cc_raw[str(dof)] = np.hstack([cc_raw[str(dof)], pearsonr(kin_data_test[self.states_to_decode[dof]], pred_kin_data_raw)[0]])
        
        import os
        # from scipy.io import savemat
        import pickle
        # pickle.dump(rej_channels, open(os.path.join('/storage/rej_channels2.pkl'), 'wb'))
        # pickle.dump(cc_lpf, open(os.path.join('/storage/cc_lpf2.pkl'), 'wb'))
        # pickle.dump(Beta, open(os.path.join('/storage/Beta2.pkl'), 'wb'))

        pickle.dump(dict(rej_channels = rej_channels, cc_lpf = cc_lpf, cc_raw = cc_raw, channels_in = channels_in, Beta = Beta, kin_data_test_lpf = kin_data_test_lpf, pred_kin_data_lpf = pred_kin_data_lpf, feature_data_test = feature_data_test, Z_features_test = Z_features_test), open(os.path.join('/storage/results_relevant_channels_' + self.train_hdf_names[0][21:23] + '_' + self.train_hdf_names[0][-8:-4] + '_' + self.train_hdf_names[-1][-8:-4] + '.pkl'), 'wb'))


        # pickle.dump(dict(rej_channels = rej_channels, cc_lpf = cc_lpf, cc_raw = cc_raw, channels_in = channels_in, Beta = Beta, kin_data_test_lpf = kin_data_test_lpf, pred_kin_data_lpf = pred_kin_data_lpf, feature_data_test = feature_data_test, Z_features_test = Z_features_test), open(os.path.join('/storage/results_relevant_channels_6103_6105_relevant_DOFs.pkl'), 'wb'))

        #pickle.dump(dict(rej_channels = rej_channels, cc_lpf = cc_lpf, cc_raw = cc_raw, channels_in = channels_in, Beta = Beta, kin_data_test_lpf = kin_data_test_lpf, pred_kin_data_lpf = pred_kin_data_lpf, feature_data_test = feature_data_test, Z_features_test = Z_features_test), open(os.path.join('/storage/results_relevant_channels_6103_6105_relevant_DOFs_2.pkl'), 'wb'))

        import matplotlib.pyplot as plt
        #n_features = np.arange(0,self.n_feats*self.n_iter,self.n_feats) + self.min_nfeats + self.nchans_bip*self.n_feats
        n_features = np.arange(self.n_feats*self.n_iter-self.n_feats,0-self.n_feats,-self.n_feats) + self.min_nfeats + self.nchans_bip*self.n_feats

        plt.figure()   
        for dof in range(self.ndofs):
            plt.plot(n_features, cc_lpf[str(dof)])
        plt.legend(['Vx','Vy','Vpsi','Prosup','Thumb','Index','3Fing'])
        plt.show(block = False)
        #import pdb; pdb.set_trace()
        #savemat(os.path.expandvars('$HOME/code/ismore/rejected_channels.mat'), dict(rej_channels = rej_channels, cc_lpf = cc_lpf, cc_raw = cc_raw, channels_in = channels_in, Beta = Beta, kin_data_test_lpf = kin_data_test_lpf, pred_kin_data_lpf = pred_kin_data_lpf, feature_data_test = feature_data_test, Z_features_test = Z_features_test))
        #savemat(os.path.expandvars('$HOME/code/ismore/rejected_channels.mat'), dict(rej_channels = rej_channels, cc_lpf = cc_lpf, cc_raw = cc_raw, channels_in = channels_in, kin_data_test_lpf = kin_data_test_lpf, pred_kin_data_lpf = pred_kin_data_lpf, feature_data_test = feature_data_test, Z_features_test = Z_features_test))
        
        return rej_channels, cc_lpf
       
    def relevant_channels(self,rejected_channels, CC):
        
        #n_features = np.arange(0,self.n_feats*self.n_iter,self.n_feats) + self.min_nfeats + self.nchans_bip*self.n_feats
        n_features = np.arange(self.n_feats*self.n_iter-self.n_feats,0-self.n_feats,-self.n_feats) + self.min_nfeats + self.nchans_bip*self.n_feats

        import matplotlib.pyplot as plt
        # plt.figure()   
        # for dof in range(self.ndofs):
        #     plt.plot(n_features, CC[str(dof)])
        # plt.legend(['Vx','Vy','Vpsi','Prosup','Thumb','Index','3Fing'])
        # plt.show(block = False)
        #chan_out = rejected_channels[np.arange(0,len(rejected_channels),self.n_feats)].sort()
        chan_out = rejected_channels[:,0]
        #nchan_out = self.nchans_bip + self.nchans_hd - self.nchannels_2select
       
        #max_idx = self.nchannels_2select * len(CC['0'])/ (self.nchans_bip + self.nchans_hd)
        max_idx = np.where(n_features == self.nchannels_2select*self.n_feats)
        cc = []
        for dof in range(len(self.relevant_dofs)):#self.ndofs):
            if cc == []:
                #cc = CC[str(dof)][-max_idx:]
                #cc = CC[self.relevant_dofs[dof]][-max_idx:]
                cc = CC[self.relevant_dofs[dof]][max_idx[0]:]
            else:
                #cc = np.vstack([cc, CC[str(dof)][-max_idx:]])
                #cc = np.vstack([cc, CC[self.relevant_dofs[dof]][-max_idx:]])
                cc = np.vstack([cc, CC[self.relevant_dofs[dof]][max_idx[0]:]])
        cc_mean = np.mean(cc, axis = 0)

        cc_range = np.max(cc_mean) - np.min(cc_mean)
        cc_min_lim = np.max(cc_mean) - 5*cc_range/100
        chans_above_th = np.where(cc_mean >= cc_min_lim)[0]
        selected_chan_num = np.max(chans_above_th)# the bigger the selected_chan_num value, the lower number of channels are being selected
        
        # selected_chan_num = 1

        # select the number of channels giving the maximum performance
        # selected_chan_num = np.where(cc_mean == np.max(cc_mean))
        # nchan_out = self.nchans_bip + self.nchans_hd - n_features[max_idx[0] + selected_chan_num[0]]/self.n_feats
        # print n_features[max_idx[0] + selected_chan_num[0]]/self.n_feats, " channels were selected"

        # select the minimum number of channels that gives a minimum of a 5% of the range of the performance (max(ccmean)-min(cc_mean))
        nchan_out = self.nchans_bip + self.nchans_hd - n_features[max_idx[0] + selected_chan_num]/self.n_feats
        print n_features[max_idx[0] + selected_chan_num]/self.n_feats, " channels were selected"

        chan_out_final = chan_out[:nchan_out]
        chan_in = np.array([chan for chan in np.arange(self.nchans_bip + self.nchans_hd) if chan not in chan_out_final])

        if self.mirrored == True:
            self.opt_channels = []
            self.opt_channels_dict = dict()
            self.opt_channels_dict['channels_str_2discard'] = channels_str_2discard
            self.opt_channels_dict['channels_str_2keep'] = [channels_str_2keep_mirrored[i] for i in np.arange(len(channels_str_2keep_mirrored[:-6])) if i in chan_in] + channels_str_2keep_mirrored[-6:]
            self.opt_channels = self.opt_channels + [self.channel_names[i] for i in np.arange(len(channels_str_2keep_mirrored[:-6])) if i in chan_in] + brainamp_channel_lists.emg_6bip_hd_filt
            self.opt_channels_dict['channels_diag1_1'] = [channels_diag1_1_mirrored[i] for i in np.arange(len(channels_diag1_1_mirrored)) if i in chan_in - 40] # 40 = 20ext + 20flex
            self.opt_channels_dict['channels_diag1_2'] = [channels_diag1_2_mirrored[i] for i in np.arange(len(channels_diag1_2_mirrored)) if i in chan_in - 40] # 46 = 20ext + 20flex + 6bipolar
            self.opt_channels = self.opt_channels + [self.channel_names[i+46] for i in np.arange(len(channels_diag1_1_mirrored)) if i in chan_in - 40]
            self.opt_channels_dict['channels_diag2_1'] = [channels_diag2_1_mirrored[i] for i in np.arange(len(channels_diag2_1_mirrored)) if i in chan_in - 70] # 70 = 20ext + 20flex + 30diag1
            self.opt_channels_dict['channels_diag2_2'] = [channels_diag2_2_mirrored[i] for i in np.arange(len(channels_diag2_2_mirrored)) if i in chan_in - 70] # 76 = 20ext + 20flex + 6bipolar + 30diag1
            self.opt_channels = self.opt_channels + [self.channel_names[i+76] for i in np.arange(len(channels_diag2_1_mirrored)) if i in chan_in - 70] 
            #Warning!!: chan_in channels include the bipolar ones at the end while self.channel_names include them after the straight hd channels.
            self.opt_channels_2train_dict = dict()
            self.opt_channels_2train_dict['channels_str_2discard'] = channels_str_2discard
            self.opt_channels_2train_dict['channels_str_2keep'] = [channels_str_2keep[i] for i in np.arange(len(channels_str_2keep[:-6])) if i in chan_in] + channels_str_2keep[-6:]
            self.opt_channels_2train_dict['channels_diag1_1'] = [channels_diag1_1[i] for i in np.arange(len(channels_diag1_1)) if i in chan_in - 40] # 40 = 20ext + 20flex
            self.opt_channels_2train_dict['channels_diag1_2'] = [channels_diag1_2[i] for i in np.arange(len(channels_diag1_2)) if i in chan_in - 40] # 46 = 20ext + 20flex + 6bipolar
            self.opt_channels_2train_dict['channels_diag2_1'] = [channels_diag2_1[i] for i in np.arange(len(channels_diag2_1)) if i in chan_in - 70] # 70 = 20ext + 20flex + 30diag1
            self.opt_channels_2train_dict['channels_diag2_2'] = [channels_diag2_2[i] for i in np.arange(len(channels_diag2_2)) if i in chan_in - 70] # 76 = 20ext + 20flex + 6bipolar + 30diag1
            # import pdb; pdb.set_trace()
        else:
            self.opt_channels = []
            self.opt_channels_dict = dict()
            self.opt_channels_dict['channels_str_2discard'] = channels_str_2discard
            self.opt_channels_dict['channels_str_2keep'] = [channels_str_2keep[i] for i in np.arange(len(channels_str_2keep[:-6])) if i in chan_in] + channels_str_2keep[-6:]
            self.opt_channels = self.opt_channels + [self.channel_names[i] for i in np.arange(len(channels_str_2keep[:-6])) if i in chan_in] + brainamp_channel_lists.emg_6bip_hd_filt
            self.opt_channels_dict['channels_diag1_1'] = [channels_diag1_1[i] for i in np.arange(len(channels_diag1_1)) if i in chan_in - 40] # 40 = 20ext + 20flex
            self.opt_channels_dict['channels_diag1_2'] = [channels_diag1_2[i] for i in np.arange(len(channels_diag1_2)) if i in chan_in - 40] # 46 = 20ext + 20flex + 6bipolar
            self.opt_channels = self.opt_channels + [self.channel_names[i+46] for i in np.arange(len(channels_diag1_1)) if i in chan_in - 40]
            self.opt_channels_dict['channels_diag2_1'] = [channels_diag2_1[i] for i in np.arange(len(channels_diag2_1)) if i in chan_in - 70] # 70 = 20ext + 20flex + 30diag1
            self.opt_channels_dict['channels_diag2_2'] = [channels_diag2_2[i] for i in np.arange(len(channels_diag2_2)) if i in chan_in - 70] # 76 = 20ext + 20flex + 6bipolar + 30diag1
            self.opt_channels = self.opt_channels + [self.channel_names[i+76] for i in np.arange(len(channels_diag2_1)) if i in chan_in - 70] 
            #Warning!!: chan_in channels include the bipolar ones at the end while self.channel_names include them after the straight hd channels.
            self.opt_channels_2train_dict = self.opt_channels_dict
        
        import os
        from scipy.io import savemat
        
        if self.mirrored:
            savemat(os.path.expandvars('$HOME/code/ismore/noninvasive/opt_channels_' + self.train_hdf_names[0][21:23] + '_' + self.train_hdf_names[0][-8:-4] + '_' + self.train_hdf_names[-1][-8:-4] + '_mirrored.mat'), dict(opt_channels = self.opt_channels, channels_str_2discard = self.opt_channels_dict['channels_str_2discard'], channels_str_2keep = self.opt_channels_dict['channels_str_2keep'], channels_diag1_1 = self.opt_channels_dict['channels_diag1_1'], channels_diag1_2 = self.opt_channels_dict['channels_diag1_2'], channels_diag2_1 = self.opt_channels_dict['channels_diag2_1'], channels_diag2_2 = self.opt_channels_dict['channels_diag2_2'],channels_2train_str_2discard = self.opt_channels_2train_dict['channels_str_2discard'], channels_2train_str_2keep = self.opt_channels_2train_dict['channels_str_2keep'], channels_2train_diag1_1 = self.opt_channels_2train_dict['channels_diag1_1'], channels_2train_diag1_2 = self.opt_channels_2train_dict['channels_diag1_2'], channels_2train_diag2_1 = self.opt_channels_2train_dict['channels_diag2_1'], channels_2train_diag2_2 = self.opt_channels_2train_dict['channels_diag2_2']))    
        else:
            savemat(os.path.expandvars('$HOME/code/ismore/noninvasive/opt_channels_' + self.train_hdf_names[0][21:23] + '_' + self.train_hdf_names[0][-8:-4] + '_' + self.train_hdf_names[-1][-8:-4] + '.mat'), dict(opt_channels = self.opt_channels, channels_str_2discard = self.opt_channels_dict['channels_str_2discard'], channels_str_2keep = self.opt_channels_dict['channels_str_2keep'], channels_diag1_1 = self.opt_channels_dict['channels_diag1_1'], channels_diag1_2 = self.opt_channels_dict['channels_diag1_2'], channels_diag2_1 = self.opt_channels_dict['channels_diag2_1'], channels_diag2_2 = self.opt_channels_dict['channels_diag2_2']))    
        import pdb; pdb.set_trace()

        return self.opt_channels, self.opt_channels_dict, self.opt_channels_2train_dict

def ridge(Y, X, K, zscore):
    '''
    Same as MATLAB's ridge regression function.
    '''
    #import pdb; pdb.set_trace()
    p = X.shape[1]
    if zscore:
        Z = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    else:
        Z = X

    Z = np.mat(Z)
    Y = np.mat(Y)
    #import pdb; pdb.set_trace() 
    W = np.array(np.linalg.pinv(Z.T * Z + K*np.mat(np.eye(p))) * Z.T*Y)

    return W
    #return np.linalg.pinv(Z.T.dot(Z) + K*np.identity(p)).dot(Z.T).dot(Y)
