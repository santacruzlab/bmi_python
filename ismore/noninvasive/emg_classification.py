import tables
import os
import numpy as np
import pandas as pd
import math
import pdb # pdb.set_trace() Kind of the same thing as keyboard function in matlab
import matplotlib.pyplot as plt
from ismore import brainamp_channel_lists
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.signal import butter, lfilter, filtfilt

from ismore.common_state_lists import *
from ismore import ismore_bmi_lib

# from riglib.filter import Filter
from ismore.filter import Filter
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import svm
from scipy.stats import mode




class EMGClassifierBase(object):
    '''
    Abstract base class for all concrete EMG decoder classes
    '''
    pass


class SVM_EMGClassifier(EMGClassifierBase):
    '''Concrete base class for an SVM classifier.'''

    def __init__(self, channels_2train, fs, win_len, filt_training_data, bip_and_filt_needed, extractor_cls, extractor_kwargs, classifier_type):
        
        self.recorded_channels = channels_2train
        self.recorded_channels = ['chan' + name for name in self.recorded_channels]
        self.emg_channels = extractor_kwargs["emg_channels"]
        
        #self.emg_channels = channels_2train
        # self.opt_channels_2train_dict = opt_channels_2train_dict

        self.fs             = fs
        self.win_len        = win_len
        self.filt_training_data = filt_training_data
        self.bip_and_filt_needed = bip_and_filt_needed
        self.classifier_type  = classifier_type

        self.dtype = np.dtype([('data',       np.float64),
                               ('ts_arrival', np.float64)])
        self.feature_names = extractor_kwargs['feature_names']
        self.feature_fn_kwargs = extractor_kwargs['feature_fn_kwargs']

        # channel names in HDF file have 'chan' in front (e.g., 'chanBiceps'
        self.channel_names = ['chan' + name[:] for name in self.emg_channels]
        self.channel_names_original = self.channel_names
        self.extractor_cls = extractor_cls
        self.extractor_kwargs = extractor_kwargs

    #nerea -- check how it would be this to use the classifier online when we input the features of the emg already
    def __call__(self, features):
        classifier_output = []
        
        classifier_output = self.classifier.predict(features)

        return classifier_output

    def train_svm(self, C, gamma, train_hdf_names, test_hdf_names):
        '''Use SVM to train this decoder from data from multiple .hdf files.'''
        self.C               = C
        self.gamma           = gamma
        self.train_hdf_names = train_hdf_names
        self.test_hdf_names  = test_hdf_names

        # will be 2-D arrays, each with shape (N, n_features)
        # e.g., if extracting 7 features from each of 14 channels, then the
        #   shape might be (10000, 98)
        feature_data_train = None
        feature_data_test  = None

        feature_label_train = None
        feature_label_test  = None

        
        #nerea -- needed?
        fs_synch = 20
        window = int(60 * fs_synch) # sliding window used to normalize the test signal

        # calculate coefficients for a 4th-order Butterworth BPF from 10-450 Hz
        band  = [10, 450]  # Hz
        nyq   = 0.5 * self.fs
        low   = band[0] / nyq
        high  = band[1] / nyq
        bpf_coeffs = butter(4, [low, high], btype='band')


        # calculate coefficients for multiple 2nd-order notch filers
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

        n_channels = len(self.channel_names_original)
        channel_filterbank = [None]*n_channels
        for k in range(n_channels):
            filts = [Filter(bpf_coeffs[0], bpf_coeffs[1])]
            for b, a in notchf_coeffs:
                filts.append(Filter(b=b, a=a))
            channel_filterbank[k] = filts


       # f_extractor = self.extractor_cls(None, **self.extractor_kwargs)


        all_hdf_names = train_hdf_names + [name for name in test_hdf_names if name not in train_hdf_names]
        
        for hdf_name in all_hdf_names:
            # load task & EMG data from HDF file
            
            # hdf = tables.openFile(hdf_name)
            hdf = tables.open_file(hdf_name)
            
            store_dir_supp = '/storage/supp_hdf/'
            index_slash = hdf_name.encode('ascii','ignore').rfind('/')            
            hdf_supp_name = store_dir_supp + hdf_name.encode('ascii','ignore')[index_slash + 1:-3] + 'supp.hdf'
            
  
            hdf_supp = tables.open_file(hdf_supp_name) 
            #import pdb; pdb.set_trace()
            #take the channels that will be used for the classification
            emg = hdf_supp.root.brainamp[:][self.channel_names_original]
            emg_original = emg
            emg_msgs = hdf_supp.root.brainamp[:][self.channel_names_original]
            
            # try:
            #     emg = hdf.root.brainamp[:][self.channel_names]
            # except:  # in older HDF files, brainamp data was stored under table 'emg'
            #     emg = hdf.root.emg[:][self.channel_names]
            # import pdb; pdb.set_trace()
            original_ts = emg_original[self.channel_names_original[0]]['ts_arrival']
            task = hdf.root.task

            #nerea
            #self.plant_type = task[0]['plant_type']
            self.plant_type = 'IsMore'

            task_msgs = hdf.root.task_msgs
            trial_types = task[:]['trial_type']
            state_types = set(task_msgs[:]['msg'])
            task_state = task_msgs[:]['msg']
            task_state_idx = task_msgs[:]['time']
            ts_emg = emg_original[self.channel_names_original[0]]['ts_arrival']

            #check if the EMG channels are monopolar and if they need to be bipolarized
            emg_chan_recorded = hdf_supp.root.brainamp.colnames
           
            
            if self.bip_and_filt_needed:
                
                print "Data needs to be bipolarized and filtered!"
                for ind_ch in range(len(emg_chan_recorded)):
                    #first remove 'chan' from the beginning                    
                    emg_chan_recorded[ind_ch] = emg_chan_recorded[ind_ch].replace('chan','')
                
                
                emg_bipolar_ch_names = []
                # take name of bipolar channels (wihtout ind)
                import re
                for ind_ch in range(len(self.channel_names_original)):
                    emg_bipolar_ch_names.append(" ".join(re.findall("[a-zA-Z]+", self.channel_names_original[ind_ch])))
                    # emg_bipolar_ch_names[ind_ch] = 'chan' + emg_bipolar_ch_names[ind_ch] + '_filt'
                
                emg_bipolar_ch_names = set(emg_bipolar_ch_names)
                
                
                # Bipolarize channels 2 with 1
                # dtype = np.dtype([('data', np.float64), ('ts_arrival', np.float64)])
                bipolarized_EMG = dict()
                # bipolarized_EMG = []
                bipolar_ch_name_filt = []
                # data = np.zeros([len(emg_bipolar_ch_names),len(emg[self.channel_names[0]]['data'])])
                for ch_ind,bipolar_ch_name in enumerate(emg_bipolar_ch_names):
                    bip_ch_1 = bipolar_ch_name + '1'
                    bip_ch_2 =  bipolar_ch_name + '2'
                    
                    # data[ch_ind,:]=  emg[:][bip_ch_2]['data'] - emg[:][bip_ch_1]['data']
                    bipolar_ch_name_filt.append(bipolar_ch_name + '_filt')
                  
                    bipolarized_EMG[bipolar_ch_name_filt[ch_ind]] = np.zeros(len(emg_original[bip_ch_1]['data']), dtype=self.dtype)
                           
                    # bipolarized_EMG[bipolar_ch_name_filt[ch_ind]]['data'] =  data
                    bipolarized_EMG[bipolar_ch_name_filt[ch_ind]]['data'] =  emg_original[bip_ch_2]['data'] - emg_original[bip_ch_1]['data']
                import pdb; pdb.set_trace() 

                n_channels_2filt = len(emg_bipolar_ch_names)
                channel_filterbank_2filt  = [None]*n_channels_2filt
                for k in range(n_channels_2filt):
                    filts_2filt = [Filter(bpf_coeffs[0], bpf_coeffs[1])]
                    for b, a in notchf_coeffs:
                        filts_2filt.append(Filter(b=b, a=a))
                
                channel_filterbank_2filt [k] = filts_2filt
                for ch_ind,bipolar_ch_name in enumerate(bipolar_ch_name_filt): #for loop on number of electrodes
                    
                    for filt in channel_filterbank_2filt [k]: #channel_filterbank_emg has filters for the amount of raw channels, which is larger than the number of filtered channels. That's why this is not a problem
                        bipolarized_EMG[bipolar_ch_name]['data']  =  filt(bipolarized_EMG[bipolar_ch_name]['data'] )
            
            
         
            # binary classifier that classifies between rest state and movement state of muscles (in general, for any kind of task)
            
            mov_states = []
            no_mov_states = []
            #import pdb; pdb.set_trace() 

            # #nerea changed - if there is preparation during instruc trial return there is no movement   
            # for state in state_types:
            #    if state.startswith('trial') or state.startswith('instruct_trial_return'): #states where the subject was moving (we also consider that during the return instruction the subject was moving since tehre was not a rest period in between)
            #        mov_states.append(state)

            for state in state_types:
                if state.startswith('trial'): #states where the subject was moving (we also consider that during the return instruction the subject was moving since tehre was not a rest period in between)
                    mov_states.append(state)        
                        
            no_mov_states = state_types-set(mov_states)
            class_label = np.zeros(len(emg_original))

            # no_mov states are set with label 0

            if self.classifier_type == 'Mov-NoMov':
                
                self.output_classes = ['Mov', 'NoMov']
                self.num_output_classes = len(self.output_classes)
                self.mov_class_labels = np.arange(1,self.num_output_classes+1) #not taking rest into account

                for state in mov_states:
                    print 'state : ', state
                    #look for indices in emg data                    
                    state_idx = [idx for (idx, msg) in enumerate(task_msgs[:]['msg']) if msg == state]
                    print 'state_idx ', state_idx
                    for trial_idx in state_idx:
                        # import pdb; pdb.set_trace() 

                        trial_idx_start = task_state_idx[trial_idx]
                        trial_ts_start = task[trial_idx_start]['ts']
                        trial_emg_idx_start = np.where(abs(ts_emg- trial_ts_start) == abs(ts_emg- trial_ts_start).min())[0][0]
                        # import pdb; pdb.set_trace() 
                        trial_idx_end = task_state_idx[trial_idx+1]
                        trial_ts_end = task[trial_idx_end]['ts']
                        trial_emg_idx_end = np.where(abs(ts_emg- trial_ts_end) == abs(ts_emg- trial_ts_end).min())[0][0]
                          
                        #set datapoint ind of movement to 1
                        class_label[trial_emg_idx_start:trial_emg_idx_end] = 1
                
            elif self.classifier_type == 'MultiClass':

                movs2classify = 'go&return_different'

                if movs2classify == 'go&return_same':

                    self.output_classes = list(set(trial_types)) # nerea: if we wanna split go-return phases we need to do it in a different way
                    self.num_output_classes = len(self.output_classes) #not taking rest and return into account
                    self.mov_class_labels = np.arange(1,self.num_output_classes+1) #not taking rest and return into account
                
                    for state in mov_states:
                        print 'state : ', state
                        #look for indices in emg data                    
                        state_idx = [idx for (idx, msg) in enumerate(task_msgs[:]['msg']) if msg == state]
                        print 'state_idx ', state_idx

                        for trial_idx in state_idx:

                            trial_idx_start = task_state_idx[trial_idx]
                            trial_ts_start = task[trial_idx_start]['ts']
                            trial_emg_idx_start = np.where(abs(ts_emg- trial_ts_start) == abs(ts_emg- trial_ts_start).min())[0][0]

                            trial_idx_end = task_state_idx[trial_idx+1]
                            trial_ts_end = task[trial_idx_end]['ts']
                            trial_emg_idx_end = np.where(abs(ts_emg- trial_ts_end) == abs(ts_emg- trial_ts_end).min())[0][0]
                        
                            # set movment label to the corresponding class
                            trial_type = task[:]['trial_type'][trial_idx_start]
                            [trial_mov_label]= [i for i, j in enumerate(self.output_classes) if j == trial_type]
                            class_label[trial_emg_idx_start:trial_emg_idx_end] = self.mov_class_labels[trial_mov_label]

                elif movs2classify == 'go&return_different':
                    # we add an extra class for the returning phase
                    self.output_classes = list(set(trial_types)) # nerea: if we wanna split go-return phases we need to do it in a different way
                    self.output_classes.append('rest') # for the trial type in which the subject is returning to the rest position, we call it 'rest' so that we send the exo towards that recorded 'rest' postion in the targets_matrix
                    self.num_output_classes = len(self.output_classes) 
                    self.mov_class_labels = np.arange(1,self.num_output_classes+1) 
                
                    for state in mov_states:
                        print 'state : ', state
                        #look for indices in emg data                    
                        state_idx = [idx for (idx, msg) in enumerate(task_msgs[:]['msg']) if msg == state]
                        print 'state_idx ', state_idx

                        for trial_idx in state_idx:

                            trial_idx_start = task_state_idx[trial_idx]
                            trial_ts_start = task[trial_idx_start]['ts']
                            trial_emg_idx_start = np.where(abs(ts_emg- trial_ts_start) == abs(ts_emg- trial_ts_start).min())[0][0]

                            trial_idx_end = task_state_idx[trial_idx+1]
                            trial_ts_end = task[trial_idx_end]['ts']
                            trial_emg_idx_end = np.where(abs(ts_emg- trial_ts_end) == abs(ts_emg- trial_ts_end).min())[0][0]
                            
                            # set movment label to the corresponding class
                            if state == 'trial':
                                trial_type = task[:]['trial_type'][trial_idx_start]
                            elif state in ['trial_return', 'instruct_trial_return']:
                                trial_type = 'rest'
                            [trial_mov_label]= [i for i, j in enumerate(self.output_classes) if j == trial_type]
                            class_label[trial_emg_idx_start:trial_emg_idx_end] = self.mov_class_labels[trial_mov_label]

                #plt.figure()
                #plt.plot(emg[:]['chanBiceps']['data'], color='blue')
                #plt.plot(class_label,color='red')
                #plt.show()

            #nerea - check if it is necessary to cut last part of the signal or not!!!    

            self.HD_EMG_diag = 0
            # Diagonalization
            if self.HD_EMG_diag:
                #diag_emg = np.array()
                #import pdb; pdb.set_trace()
                diag_emg = dict()#np.ndarray(emg.shape)
                if hdf_name in train_hdf_names:
                    chan_2keep = self.opt_channels_2train_dict['channels_str_2keep']
                    # import pdb; pdb.set_trace()
                    chan_2discard = self.opt_channels_2train_dict['channels_str_2discard']
                    chan_diag1_1 = self.opt_channels_2train_dict['channels_diag1_1']
                    chan_diag1_2 = self.opt_channels_2train_dict['channels_diag1_2']
                    chan_diag2_1 = self.opt_channels_2train_dict['channels_diag2_1']
                    chan_diag2_2 = self.opt_channels_2train_dict['channels_diag2_2']


                else:
                    chan_2keep = self.extractor_kwargs['channels_str_2keep']
                    # import pdb; pdb.set_trace()
                    chan_2discard = self.extractor_kwargs['channels_str_2discard']
                    chan_diag1_1 = self.extractor_kwargs['channels_diag1_1']
                    chan_diag1_2 = self.extractor_kwargs['channels_diag1_2']
                    chan_diag2_1 = self.extractor_kwargs['channels_diag2_1']
                    chan_diag2_2 = self.extractor_kwargs['channels_diag2_2']
                data = np.zeros([len(self.recorded_channels),len(emg[self.recorded_channels[0]]['data'])])
                
                for k in range(len(self.recorded_channels)):
                    data[k,:] = emg[self.recorded_channels[k]]['data']
                data_diff = np.diff(data, axis = 0)
                
                # diag_emg = np.zeros((1, len(emg[self.recorded_channels[0]]['data'])), dtype=self.dtype)
                for i in range(n_channels):
                    if i < len(chan_2keep):
                        diag_emg[self.channel_names[i]] = np.zeros((len(emg[self.recorded_channels[0]]['data'])), dtype=self.dtype) 
                        #import pdb; pdb.set_trace()
                        diag_emg[self.channel_names[i]]['data'] = data_diff[chan_2keep[i],:]
                        #import pdb; pdb.set_trace()
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


            # EMG filtering
            if self.bip_and_filt_needed:
                emg = bipolarized_EMG.copy()
                self.channel_names = bipolar_ch_name_filt
                self.emg_channels = bipolar_ch_name_filt
                self.channels_to_process = bipolar_ch_name_filt
                n_channels= len(self.channels_to_process )

            else:
                self.emg_channels_2process = self.channel_names
                if self.filt_training_data:
                    for k in range(n_channels): #for loop on number of electrodes
                        
                        for filt in channel_filterbank[k]: #channel_filterbank_emg has filters for the amount of raw channels, which is larger than the number of filtered channels. That's why this is not a problem
                            emg[self.emg_channels_2process[k]]['data'] =  filt(emg[self.emg_channels_2process[k]]['data']) 
            



            # Take only 70% of the rest periods with less variability in the train data to train the classifier
            if self.classifier_type == 'Mov-NoMov' and hdf_name in train_hdf_names:
                idx_rest = np.where(class_label == 0)
                idx_rest_ends = np.where(np.diff(class_label)==1)
                idx_rest_ends = np.append(idx_rest_ends,len(class_label))
                idx_rest_starts = np.where(np.diff(class_label)==-1)
                idx_rest_starts = np.append(0,idx_rest_starts) #append 0 since the first trial of rest starts already at the beginning
                num_trials= len(idx_rest_starts)
                
                mean_rest_interval_ch = [None]*n_channels
                std_rest_interval_ch = [None]*n_channels
                mean_rest_interval_trial = [None]*num_trials
                mean_std_rest_interval_trial = [None]*num_trials
                for idx_trial in range(num_trials):
                    idx_start = idx_rest_starts[idx_trial]
                    idx_end = idx_rest_ends[idx_trial]
                    
                    #calculate mean and std of each rest interval, average all channels
                    for idx_chan in range(n_channels): #for loop on number of electrodes
                        emg_rest_interval_ch = emg[self.channel_names[idx_chan]]['data'][idx_start:idx_end]
                        mean_rest_interval_ch[idx_chan] = np.mean(emg_rest_interval_ch)
                        std_rest_interval_ch[idx_chan] = np.std(emg_rest_interval_ch)
                    
                    mean_rest_interval_trial[idx_trial] = np.mean(mean_rest_interval_ch)
                    mean_std_rest_interval_trial[idx_trial] = np.mean(std_rest_interval_ch)

                perc_trials = 0.7
                num_selected_trials = int(perc_trials* num_trials)
                sort_std_val = np.sort(mean_std_rest_interval_trial)
                sort_std_idx = np.argsort(mean_std_rest_interval_trial)

                selected_trials_idx = sort_std_idx[0:num_selected_trials]

                idx_mov = np.where(class_label == 1)

                idx_rest_selected = []
                for idx_trial, idx_sel_trial in enumerate(selected_trials_idx):
                    
                    idx_start = idx_rest_starts[idx_sel_trial]
                    idx_end = idx_rest_ends[idx_sel_trial]
                    idx_rest_selected = idx_rest_selected + range(idx_start,idx_end)
                   
                idx_mov_rest = np.append(idx_rest_selected,idx_mov)              
                idx_mov_rest_sort = np.sort(idx_mov_rest)
                           
                
                # take only the emg data of mov class and the selected trials of rest class
                try:
                    emg_to_feat = emg[:][:][idx_mov_rest_sort]
                except:
                    
                    emg_to_feat = dict()
                    # emg_to_feat[bipolar_ch_name] =np.vstack(emg[bipolar_ch_name]['data'][idx_mov_rest_sort] for bipolar_ch_name in bipolar_ch_name_filt)
                    for ch_ind,bipolar_ch_name in enumerate(bipolar_ch_name_filt):
                        emg_to_feat[bipolar_ch_name] = np.zeros(len(emg[bipolar_ch_name]['data'][idx_mov_rest_sort]), dtype=self.dtype)
                        emg_to_feat[bipolar_ch_name]['data']=emg[bipolar_ch_name]['data'][idx_mov_rest_sort]
                
                class_label = class_label[idx_mov_rest_sort]
                ts_emg = ts_emg[idx_mov_rest_sort]
                
            else:
    
                emg_to_feat = emg
           
            ##EMG feature extraction 
            n_win_pts = int(self.win_len * self.fs)
            step_pts = 50  # TODO -- don't hardcode This is in sample points! Be careful if fs is changed!!
            #step_pts_dist = 20  # TODO -- don't hardcode
            


            start_idxs = np.arange(0, len(ts_emg) - n_win_pts + 1, step_pts) #andrea -- galdetu andrea zeatik erabiltzeun hau ta ez hurrengua
            #start_idxs = np.arange(n_win_pts - 1, len(emg), step_pts) # ????
            import pdb; pdb.set_trace()
            f_extractor = self.extractor_cls(None, emg_channels = self.emg_channels, feature_names = self.feature_names, feature_fn_kwargs = self.feature_fn_kwargs, win_len=self.win_len, fs=self.fs)#, brainamp_channels = self.brainamp_channels)
            features = np.zeros((len(start_idxs), f_extractor.n_features))
            class_label_features = np.zeros((len(start_idxs)))
            #emg_samples = np.zeros((len(self.channel_names), n_win_pts*len(start_idxs)))


            for i, start_idx in enumerate(start_idxs):
                end_idx = start_idx + n_win_pts 
                
                # samples has shape (n_chan, n_win_pts) 
                #samples = np.vstack([emg[chan]['data'][start_idx:end_idx] for chan in self.channel_names]) 
                samples = np.vstack([emg_to_feat[chan]['data'][start_idx:end_idx] for chan in self.channel_names])

                #predominant_class = int(max(Counter(class_label[start_idx:end_idx])))
                predominant_class = int(mode(class_label[start_idx:end_idx])[0]) #should we take the predominant class of just look at the last label

                if samples.shape[1] == 0:                    
                    import pdb; pdb.set_trace()
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
                class_label_features[i] = predominant_class
                #emg_samples[:,i*n_win_pts:(i+1)*n_win_pts] = f_extractor.extract_filtered_samples(samples)
                
            
            if self.classifier_type == 'MultiClass':
            # once the features are extracted for the whole run, we take only the data that corresponds to moving_states (rest data is excluded) 
                idx_mov = np.where(class_label_features != 0)
                features = features[idx_mov[0],:]
                class_label_features = class_label_features[idx_mov[0]]
    
            


            if hdf_name in train_hdf_names:
                if feature_data_train is None: #first run
                    feature_data_train = features.copy()
                    feature_label_train = class_label_features.copy()
                    #emg_filtfilt = emg_signal_filtfilt.copy()
                    #emg_raw = emg.copy()
                    #emg_filt = emg_signal_filt.copy()
                    
                else:
                    feature_data_train = np.vstack([feature_data_train, features])    
                    feature_label_train = np.hstack([feature_label_train, class_label_features]) 
                    #emg_filtfilt = np.hstack([emg_filtfilt, emg_signal_filtfilt])
                    #emg_raw = np.hstack([emg_raw, emg])
                    #emg_filt = np.hstack([emg_filt, emg_signal_filt])
            
            
                    
            if hdf_name in test_hdf_names:
                if feature_data_test is None:
                    feature_data_test = features.copy()
                    feature_label_test = class_label_features.copy()
                else:
                    feature_data_test = np.vstack([feature_data_test, features])
                    feature_label_test = np.hstack([feature_label_test, class_label_features])

            hdf.close()
            hdf_supp.close()        
        
        # Z-score normalization of train and test data
        self.features_mean_train = np.mean(feature_data_train, axis=0)
        self.features_std_train = np.std(feature_data_train, axis=0)
        Z_features_train = (feature_data_train - self.features_mean_train) / self.features_std_train
        
        self.features_mean_test = np.mean(feature_data_test, axis=0)
        self.features_std_test = np.std(feature_data_test, axis=0)
        Z_features_test = (feature_data_test - self.features_mean_train) / self.features_std_train

        
        #5-CV
        # from sklearn.model_selection import cross_val_score
        # print '5-CV'
        # clf = svm.SVC(gamma=self.gamma, C=self.C)
        # scores = cross_val_score(clf, Z_features_train, feature_label_train, cv=5)
        # print 'CV scores ', scores
      
        import pdb; pdb.set_trace()
        # build svm classifier with the input parameters
        clf = svm.SVC(gamma=self.gamma, C=self.C,probability=True)

        # train classifier
        print "training classifier ..."
        clf.fit(Z_features_train, feature_label_train)
        print "classifier is trained"
     
        self.classifier = clf
        self.classifier.plant_type = self.plant_type
        self.classifier.output_classes = self.output_classes
        self.classifier.num_output_classes  = self.num_output_classes 
        self.classifier.mov_class_labels = self.mov_class_labels


        #test classfier
        print 'predicting test set'
        predicted_label= self.classifier.predict(Z_features_test)
        predicted_prob= self.classifier.predict_proba(Z_features_test)
        print 'testing finished'

        from sklearn.metrics import accuracy_score
        acc_score = accuracy_score(feature_label_test, predicted_label)
        print 'acc_score : ', acc_score


        self.classifier.acc_score = acc_score
        self.classifier.feature_label_test = feature_label_test
        self.classifier.predicted_label = predicted_label

        plt.figure()
        plt.plot(Z_features_test[:,:], color='blue')
        plt.plot(feature_label_test, color='green')
        plt.plot(predicted_label, color='red')
        plt.plot(predicted_prob, color='yellow')
        plt.legend(['EMG feature','true class', 'predicted class'])

        print 'Close the figure in order to continue '
        plt.show(block=False)

        #import pdb; pdb.set_trace()




  




        


