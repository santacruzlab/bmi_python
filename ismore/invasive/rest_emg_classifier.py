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
# from ismore import ismore_bmi_lib

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

class SVM_rest_EMGClassifier(EMGClassifierBase):
    '''Binary EMG SVM classifier for detecting movement intention or rest during rest periods.'''

    def __init__(self, channels_2train, fs, win_len, step_len, filt_training_data, extractor_cls, extractor_kwargs):
        self.recorded_channels = channels_2train
        self.recorded_channels = ['chan' + name for name in self.recorded_channels]
        self.emg_channels = extractor_kwargs["emg_channels"]

        self.fs             = fs
        self.win_len        = win_len
        self.step_len       = step_len
        self.filt_training_data = filt_training_data

        self.dtype = np.dtype([('data',       np.float64),
        ('ts_arrival', np.float64)])
        self.feature_names = extractor_kwargs['feature_names']

        # channel names in HDF file have 'chan' in front (e.g., 'chanBiceps'
        self.channel_names = ['chan' + name[:] for name in self.emg_channels]
        self.channel_names_original = self.channel_names
        self.extractor_cls = extractor_cls
        self.extractor_kwargs = extractor_kwargs

    #nerea -- check how it would be this to use the classifier online when we input the features of the emg already
    def __call__(self, features):
        classifier_output = []
        #classifier_output = self.classifier.predict(features)
        classifier_output = self.classifier.predict_proba(features)[:,1]
        return classifier_output

    def train_svm(self, C, gamma, train_hdf_names, test_hdf_names):
        '''Train binary SVM for EMG classification rest vs move'''

        self.C = C
        self.gamma = gamma
        self.train_hdf_names = train_hdf_names
        self.test_hdf_names = test_hdf_names

        feature_data_train = None
        feature_data_test  = None

        feature_label_train = None
        feature_label_test  = None

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


        all_hdf_names = train_hdf_names + [name for name in test_hdf_names if name not in train_hdf_names]
        
        for hdf_name in all_hdf_names:
            # load task & EMG data from HDF file

            hdf = tables.open_file(hdf_name)
            
            store_dir_supp = '/storage/supp_hdf/'
            index_slash = hdf_name.encode('ascii','ignore').rfind('/')            
            hdf_supp_name = store_dir_supp + hdf_name.encode('ascii','ignore')[index_slash + 1:-3] + 'supp.hdf'
            hdf_supp = tables.open_file(hdf_supp_name) 

            emg = hdf_supp.root.brainamp[:][self.channel_names_original]
            emg_original = emg
            #emg_msgs = hdf_supp.root.brainamp[:][self.channel_names_original]
            #emg_msgs = hdf.root.brainamp_msgs[:][self.channel_names_original]
            
            # try:
            #     emg = hdf.root.brainamp[:][self.channel_names]
            # except:  # in older HDF files, brainamp data was stored under table 'emg'
            #     emg = hdf.root.emg[:][self.channel_names]

            original_ts = emg_original[self.channel_names_original[0]]['ts_arrival']
            task = hdf.root.task

            task_msgs = hdf.root.task_msgs
            trial_types = task[:]['trial_type']
            state_types = set(task_msgs[:]['msg'])
            task_state = task_msgs[:]['msg']
            task_state_idx = task_msgs[:]['time']
            ts_emg = emg_original[self.channel_names_original[0]]['ts_arrival']


            #we did not record the ts of the task in some files, let's use the ts of the rehand instead
            rehand = hdf.root.rehand
            rehand_msgs = hdf.root.rehand_msgs
            rehand_ts = hdf.root.rehand[:]['ts']     

            #check if the EMG channels are monopolar and if they need to be bipolarized
            emg_chan_recorded = hdf_supp.root.brainamp.colnames
            self.emg_channels_2process = self.channel_names

            # filter emg data
            if self.filt_training_data:
                for k in range(n_channels): #for loop on number of electrodes
                    
                    for filt in channel_filterbank[k]: #channel_filterbank_emg has filters for the amount of raw channels, which is larger than the number of filtered channels. That's why this is not a problem
                        emg[self.emg_channels_2process[k]]['data'] =  filt(emg[self.emg_channels_2process[k]]['data']) 

                # change channels names to _filt


            # binary classifier that classifies between rest state and movement state of muscles (in general, for any kind of task)
            mov_states = []
            no_mov_states = []
         
            for state in state_types:
                if state.startswith('target'): #states where the subject was moving (we also consider that during the return instruction the subject was moving since tehre was not a rest period in between)
                    mov_states.append(state)      

            for state in state_types:
                if state.startswith('rest'): #states where the subject was moving (we also consider that during the return instruction the subject was moving since tehre was not a rest period in between)
                    no_mov_states.append(state)    
                        
            # no_mov states are set with label 0                
            self.output_classes = ['Mov', 'NoMov']
            self.num_output_classes = len(self.output_classes)
            self.mov_class_labels = np.arange(1,self.num_output_classes+1) #not taking rest into account

            mov_state_idx = []
            for state in mov_states:
                print 'mov state : ', state
                #look for indices in emg data                    
                #state_idx = [idx for (idx, msg) in enumerate(task_msgs[:]['msg']) if msg == state] #using task msgs
                state_idx = [idx for (idx, msg) in enumerate(rehand_msgs[:]['msg']) if msg == state] # using rehand msgs
                print 'state_idx ', state_idx
                for trial_idx in state_idx:
                   
                    # # using task ts
                    # trial_idx_start = task_state_idx[trial_idx]
                    # trial_ts_start = task[trial_idx_start]['ts']  

                    # trial_idx_end = task_state_idx[trial_idx+1]
                    # trial_ts_end = task[trial_idx_end]['ts']

                    #using rehand ts
                    task_state_idx = rehand_msgs[:]['time'] #overwritten
                    trial_idx_start = task_state_idx[trial_idx]
                    trial_ts_start = rehand[trial_idx_start]['ts']  

                    trial_idx_end = task_state_idx[trial_idx+1]
                    try:
                        trial_ts_end = rehand[trial_idx_end]['ts']
                    except:
                        trial_ts_end = rehand[trial_idx_end-1]['ts']
                    trial_emg_idx_start = np.where(abs(ts_emg- trial_ts_start) == abs(ts_emg- trial_ts_start).min())[0][0]
                    trial_emg_idx_end = np.where(abs(ts_emg- trial_ts_end) == abs(ts_emg- trial_ts_end).min())[0][0]

                    mov_state_idx = mov_state_idx + range(trial_emg_idx_start,trial_emg_idx_end)

            
            no_mov_state_idx = []
            for state in no_mov_states:
                print 'rest state : ', state
                #look for indices in emg data                    
                #state_idx = [idx for (idx, msg) in enumerate(task_msgs[:]['msg']) if msg == state] #using task msgs
                state_idx = [idx for (idx, msg) in enumerate(rehand_msgs[:]['msg']) if msg == state] # using rehand msgs
                print 'state_idx ', state_idx
                for trial_idx in state_idx:
                   
                    # # using task ts
                    # trial_idx_start = task_state_idx[trial_idx]
                    # trial_ts_start = task[trial_idx_start]['ts']  

                    # trial_idx_end = task_state_idx[trial_idx+1]
                    # trial_ts_end = task[trial_idx_end]['ts']

                    #using rehand ts                    
                    task_state_idx = rehand_msgs[:]['time'] #overwritten
                    trial_idx_start = task_state_idx[trial_idx]
                    trial_ts_start = rehand[trial_idx_start]['ts']  

                    trial_idx_end = task_state_idx[trial_idx+1]
                    trial_ts_end = rehand[trial_idx_end]['ts']

                    trial_emg_idx_start = np.where(abs(ts_emg- trial_ts_start) == abs(ts_emg- trial_ts_start).min())[0][0]
                    trial_emg_idx_end = np.where(abs(ts_emg- trial_ts_end) == abs(ts_emg- trial_ts_end).min())[0][0]

                    no_mov_state_idx = no_mov_state_idx + range(trial_emg_idx_start,trial_emg_idx_end)
                 
        
            emg_to_feat = emg[:][:][no_mov_state_idx + mov_state_idx]

            # no_mov_states = state_types-set(mov_states)
            class_label = np.zeros(len(emg_to_feat))
             #set datapoint ind of movement to 1
            class_label[0:len(no_mov_state_idx)-1] = 0
            class_label[len(no_mov_state_idx):] = 1

            
            # # Take only 70% of the rest periods with less variability in the train data to train the classifier
            # if hdf_name in train_hdf_names:
            #     idx_rest = np.where(class_label == 0)
            #     idx_rest_ends = np.where(np.diff(class_label)==1)
            #     idx_rest_ends = np.append(idx_rest_ends,len(class_label))
            #     idx_rest_starts = np.where(np.diff(class_label)==-1)
            #     idx_rest_starts = np.append(0,idx_rest_starts) #append 0 since the first trial of rest starts already at the beginning
            #     num_trials= len(idx_rest_starts)
                
            #     mean_rest_interval_ch = [None]*n_channels
            #     std_rest_interval_ch = [None]*n_channels
            #     mean_rest_interval_trial = [None]*num_trials
            #     mean_std_rest_interval_trial = [None]*num_trials
            #     for idx_trial in range(num_trials):
            #         idx_start = idx_rest_starts[idx_trial]
            #         idx_end = idx_rest_ends[idx_trial]
                    
            #         #calculate mean and std of each rest interval, average all channels
            #         for idx_chan in range(n_channels): #for loop on number of electrodes
            #             emg_rest_interval_ch = emg[self.channel_names[idx_chan]]['data'][idx_start:idx_end]
            #             mean_rest_interval_ch[idx_chan] = np.mean(emg_rest_interval_ch)
            #             std_rest_interval_ch[idx_chan] = np.std(emg_rest_interval_ch)
                    
            #         mean_rest_interval_trial[idx_trial] = np.mean(mean_rest_interval_ch)
            #         mean_std_rest_interval_trial[idx_trial] = np.mean(std_rest_interval_ch)

            #     perc_trials = 0.7
            #     num_selected_trials = int(perc_trials* num_trials)
            #     sort_std_val = np.sort(mean_std_rest_interval_trial)
            #     sort_std_idx = np.argsort(mean_std_rest_interval_trial)

            #     selected_trials_idx = sort_std_idx[0:num_selected_trials]

            #     idx_mov = np.where(class_label == 1)

            #     idx_rest_selected = []
            #     for idx_trial, idx_sel_trial in enumerate(selected_trials_idx):
                    
            #         idx_start = idx_rest_starts[idx_sel_trial]
            #         idx_end = idx_rest_ends[idx_sel_trial]
            #         idx_rest_selected = idx_rest_selected + range(idx_start,idx_end)
                   
            #     idx_mov_rest = np.append(idx_rest_selected,idx_mov)              
            #     idx_mov_rest_sort = np.sort(idx_mov_rest)
                           
                
            #     # take only the emg data of mov class and the selected trials of rest class
            #     try:
            #         emg_to_feat = emg[:][:][idx_mov_rest_sort]
            #     except:
                    
            #         emg_to_feat = dict()
            #         # emg_to_feat[bipolar_ch_name] =np.vstack(emg[bipolar_ch_name]['data'][idx_mov_rest_sort] for bipolar_ch_name in bipolar_ch_name_filt)
            #         for ch_ind,bipolar_ch_name in enumerate(bipolar_ch_name_filt):
            #             emg_to_feat[bipolar_ch_name] = np.zeros(len(emg[bipolar_ch_name]['data'][idx_mov_rest_sort]), dtype=self.dtype)
            #             emg_to_feat[bipolar_ch_name]['data']=emg[bipolar_ch_name]['data'][idx_mov_rest_sort]
                
            #     class_label = class_label[idx_mov_rest_sort]
            #     ts_emg = ts_emg[idx_mov_rest_sort]
                
            # else:
    
            #     emg_to_feat = emg

            ##EMG feature extraction 
            n_win_pts = int(self.win_len * self.fs)
            step_pts = int(1./self.step_len)
            # TODO -- don't hardcode This is in sample points! Be careful if fs is changed!!
            #step_pts_dist = 20  # TODO -- don't hardcode           


            #start_idxs = np.arange(0, len(ts_emg) - n_win_pts + 1, step_pts) #andrea -- galdetu andrea zeatik erabiltzeun hau ta ez hurrengua
            #start_idxs = np.arange(n_win_pts - 1, len(emg), step_pts) # ????
            start_idxs = np.arange(0, len(emg_to_feat) - n_win_pts + 1, step_pts) #andrea -- galdetu andrea zeatik erabiltzeun hau ta ez hurrengua
            
                        
            f_extractor = self.extractor_cls(None, emg_channels = self.emg_channels, feature_names = self.feature_names, win_len=self.win_len, fs=self.fs)#, brainamp_channels = self.brainamp_channels)
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
        
        # # Z-score normalization of train and test data
        self.features_mean_train = np.mean(feature_data_train, axis=0)
        self.features_mean_test = np.mean(feature_data_test, axis=0)

        if self.extractor_kwargs['use_scalar_fixed_var']:
            # This is a scalar value: 
            self.features_std_train = np.mean(np.std(feature_data_train, axis=0))
            self.features_std_test = np.mean(np.std(feature_data_test, axis=0))
            self.scalar_fixed_var = True
        else:
            self.features_std_train = np.std(feature_data_train, axis=0)
            self.features_std_test = np.std(feature_data_test, axis=0)
            self.scalar_fixed_var = False
        
        Z_features_train = (feature_data_train - self.features_mean_train) / self.features_std_train
        Z_features_test = (feature_data_test - self.features_mean_train) / self.features_std_train

        if self.filt_training_data:
            self.extractor_kwargs["emg_channels"] = [name + '_filt' for name in self.extractor_kwargs["emg_channels"]]

        #5-CV
        # from sklearn.model_selection import cross_val_score
        # print '5-CV'
        # clf = svm.SVC(gamma=self.gamma, C=self.C)
        # scores = cross_val_score(clf, Z_features_train, feature_label_train, cv=5)
        # print 'CV scores ', scores
      
        # build svm classifier with the input parameters
        clf = svm.SVC(gamma=self.gamma, C=self.C, probability= True)

        # train classifier
        print "training classifier ..."
        clf.fit(Z_features_train, feature_label_train)
        print "classifier is trained"
     
        self.classifier = clf
        self.classifier.output_classes = self.output_classes
        self.classifier.num_output_classes  = self.num_output_classes 
        self.classifier.mov_class_labels = self.mov_class_labels

        # save mean and std of training dataset to normalize testing dataset
        self.classifier.features_std_train = self.features_std_train
        self.classifier.features_mean_train = self.features_mean_train

        print "shape Z_features_test ", np.shape(Z_features_test)
        #test classfier
        print 'predicting test set'
        predicted_label= self.classifier.predict(Z_features_test)
        predicted_prob = self.classifier.predict_proba(Z_features_test)[:,1]
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
        plt.plot(predicted_prob, color='black')
        plt.legend(['EMG feature','true class', 'predicted class'])

        
        print 'Close the figure in order to continue '
        plt.show()

