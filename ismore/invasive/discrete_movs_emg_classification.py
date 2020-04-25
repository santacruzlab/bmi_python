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

import unicodedata
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.stats import linregress



class EMGClassifierBase(object):
    '''
    Abstract base class for all concrete EMG decoder classes
    '''
    pass


class SVM_mov_EMGClassifier(EMGClassifierBase):
    '''Concrete base class for an SVM classifier.'''

    def __init__(self, channels_2train, filt_training_data, extractor_cls, extractor_kwargs):
        
        self.channels_2train = ['chan' + name for name in channels_2train]
        self.emg_channels = extractor_kwargs["emg_channels"]
        self.fs             = extractor_kwargs['fs']
        self.win_len        = extractor_kwargs['win_len']
        self.step_len       = extractor_kwargs['step_len']
        self.filt_training_data = filt_training_data
        self.dtype = np.dtype([('data',       np.float64),
                               ('ts_arrival', np.float64)])
        self.feature_names = extractor_kwargs['feature_names']
        self.feature_fn_kwargs = extractor_kwargs['feature_fn_kwargs']

        # channel names in HDF file have 'chan' in front (e.g., 'chanBiceps'
        self.channel_names = ['chan' + name[:] for name in self.emg_channels]
        self.channel_names_original = self.channel_names
        self.extractor_cls = extractor_cls
        self.extractor_kwargs = extractor_kwargs
        self.n_features = len(self.channels_2train) * len(self.feature_names)

    def __call__(self, features, output_type):
        classifier_output = []

        if output_type == 'label':
            classifier_output = self.classifier.predict(features)
        elif output_type == 'prob':
            classifier_output = self.classifier.predict_proba(features)[:,1]
        return classifier_output


    def process_data(self,train_hdf_ids,test_hdf_ids, normalize_data,tt2classify,movs2classify, movs_labels,dbname,class_mode):
        '''Process trainind and testing datasets for classification'''

        self.train_hdf_ids = train_hdf_ids
        self.test_hdf_ids  = test_hdf_ids
        self.normalize_data = normalize_data
        self.tt2classify    = tt2classify
        self.movs_labels     = movs_labels
        self.dbname         = dbname

        train_features = []
        train_labels = []

        # will be 2-D arrays, each with shape (N, n_features)
        # e.g., if extracting 7 features from each of 14 channels, then the  shape might be (10000, 98)
        feature_data_train = None
        feature_data_test  = None

        feature_label_train = None
        feature_label_test  = None

        ts_features_test = None
        vel_filt_train = None
        vel_filt_test = None


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


        # all_hdf_ids = train_hdf_ids + [name for name in test_hdf_ids if name not in train_hdf_ids]
        from db import dbfunctions as dbfn 
        from db.tracker import models

        all_hdf_ids = train_hdf_ids + test_hdf_ids
        for hdf_id in all_hdf_ids:         

            print 'hdf_id ------ ', hdf_id
            te = dbfn.TaskEntry(hdf_id, dbname = self.dbname)
            hdf = te.hdf
            hdf_supp = tables.open_file('/storage/supp_hdf/' + te.name + '.supp.hdf')
            
            # hdf = tables.open_file(hdf_name)            
            # store_dir_supp = '/storage/supp_hdf/'
            # index_slash = hdf_name.encode('ascii','ignore').rfind('/')            
            # hdf_supp_name = store_dir_supp + hdf_name.encode('ascii','ignore')[index_slash + 1:-3] + 'supp.hdf'
            # hdf_supp = tables.open_file(hdf_supp_name) 

            #take the channels that will be used for the classification
            emg = hdf_supp.root.brainamp[:][self.channel_names_original]
            emg_original = emg
            emg_msgs = hdf_supp.root.brainamp[:][self.channel_names_original]
            
            original_ts = emg_original[self.channel_names_original[0]]['ts_arrival']
            ts_emg = emg[self.channel_names[0]]['ts_arrival']
            task = hdf.root.task

            task_msgs = hdf.root.task_msgs
            trial_types = task[:]['trial_type']
            trial_types_set = set(trial_types)
            state_types = set(task_msgs[:]['msg'])
            task_state = task_msgs[:]['msg']
            task_state_idx = task_msgs[:]['time']
            ts_emg = emg_original[self.channel_names_original[0]]['ts_arrival']
            task_name= unicodedata.normalize('NFKD', te.task.name).encode('ascii','ignore')

            trial_emg_idx_start_all = []
            trial_emg_idx_end_all = []


            try:
                goal_idx = np.hstack(task[:]['goal_idx'])       # non-invasive phase data
            except:
                goal_idx = np.hstack(task[:]['target_index'])   # invasive phase data

            tm_te = te.targets_matrix
            te_tm = models.DataFile.objects.using(dbname).get(pk=te.targets_matrix)
            tm_path = unicodedata.normalize('NFKD', te_tm.path).encode('ascii','ignore')
            # tm = pickle.load(open('/storage/target_matrices/' + tm_path)) # pickle not working

            try:
                tm = pd.read_pickle('/storage/target_matrices/' + tm_path)
            except:
                tm = pd.read_pickle('/storage/target_matrices/' + 'targets_HUD1_7727_7865_8164_None_HUD1_20171122_1502_fixed_thumb_point_all_targs_blue_mod_fix_cha_cha_cha_fix_B3_fix_rest.pkl')
                #tm = pd.read_pickle('/storage/misc/' + tm_path)

            subgoal_names = tm['subgoal_names']

            class_label = np.zeros(len(emg_original)) # create class label vector for the entire emg dataset

            for trial_type in tt2classify:
                if trial_type in set(trial_types): #only if that trial type is present in this dataset

                    subtrials_num = len(subgoal_names[trial_type])
                                           
                    for ind_subtrial in range(subtrials_num):
                                                    
                        subgoal_name = ''.join(subgoal_names[trial_type][ind_subtrial])
                        
                        if ind_subtrial == 0:
                            mov_label_subtrial = 'rest-' + subgoal_name
                        else:
                            prev_subgoal_name = ''.join(subgoal_names[trial_type][ind_subtrial-1])
                            mov_label_subtrial = prev_subgoal_name + '-' + subgoal_name
            
                        if mov_label_subtrial in movs2classify:
                            trial_type_T = np.where(trial_types == trial_type)
                            subtrial_type_T = np.where(goal_idx ==ind_subtrial)

                            mov_idx = sorted(set(trial_type_T[0]) & set(subtrial_type_T[0]))

                            try:
                                trials_trans = np.where(np.diff(mov_idx)>1)[0][0] +1 #if several subtrials
                            except:
                                trials_trans = None #only one subtrial

                            trials_trans_all = np.sort(np.hstack([0,trials_trans,len(mov_idx)-1]))
                            trials_trans_all = [x for x in trials_trans_all if x is not None]

                            mov_label= movs_labels[movs2classify.index(mov_label_subtrial)]


                            if task_name in ['ismore_EXGEndPointMovement_testing']: #non-invasive phase task


                                ts_task = task[:]['ts']  #non-invasive / ts variable not recorded in invasive
                                mov_idx_ts = ts_task[mov_idx]
                                
                                for ind_trans, idx_trans in enumerate(trials_trans_all[:-1]):
                                    trans_start = mov_idx[idx_trans]
                                    trans_end = mov_idx[trials_trans_all[ind_trans+1]-1]

                                    trial_ts_start = ts_task[trans_start]
                                    trial_emg_idx_start = np.where(abs(ts_emg- trial_ts_start) == abs(ts_emg- trial_ts_start).min())[0][0]
                                    trial_emg_idx_start_all.append(trial_emg_idx_start)

                                    trial_ts_end = ts_task[trans_end]
                                    trial_emg_idx_end = np.where(abs(ts_emg- trial_ts_end) == abs(ts_emg- trial_ts_end).min())[0][0]
                                    trial_emg_idx_end_all.append(trial_emg_idx_end)

                                    class_label[trial_emg_idx_start:trial_emg_idx_end] = mov_label


                            elif task_name in ['compliant_move', 'ismore_bmi', 'ismore_bmi_w_emg_rest', 'ismore_hybrid_bmi_w_emg_rest_gotostart_prp']: #invasive phase task

                                rehand_state_idx = te.hdf.root.rehand_msgs[:]['time']  #non-invasive / ts variable not recorded in invasive
                                rehand_ts = te.hdf.root.rehand[:]['ts']
                                rehand_state = te.hdf.root.rehand_msgs[:]['msg']
                                task_state = te.hdf.root.task_msgs[:]['msg']
                                task_state_idx = te.hdf.root.task_msgs[:]['time']

                                for ind_trans, idx_trans in enumerate(trials_trans_all[:-1]):
                                    trans_start = mov_idx[idx_trans]

                                    # indexes might not be exactly the same as they are saved at different times.
                                    try:
                                        trans_start_idx = np.where(task_state_idx == trans_start)[0][0]
                                    except:
                                            idx_target = task_state_idx[task_state=='target']
                                            dist_to_target = np.abs(idx_target-trans_start)
                                            trans_start = idx_target[dist_to_target == np.min(dist_to_target)][0]
                                            trans_start_idx = np.where(task_state_idx == trans_start)[0][0]
                                            
                                    trans_start_idx_rh = rehand_state_idx[trans_start_idx]
                                    trial_ts_start = rehand_ts[trans_start_idx_rh]

                                    trial_emg_idx_start = np.where(abs(ts_emg- trial_ts_start) == abs(ts_emg- trial_ts_start).min())[0][0]
                                    trial_emg_idx_start_all.append(trial_emg_idx_start)
                                  
                                    if task_state[trans_start_idx] != 'target':
                                        print "The data is not from a TARGET state -- check data!!"
                                        import pdb; pdb.set_trace()

                                    trans_end_idx_rh = rehand_state_idx[trans_start_idx+1]-1
                                    trial_ts_end = rehand_ts[trans_end_idx_rh]
                                    trial_emg_idx_end = np.where(abs(ts_emg- trial_ts_end) == abs(ts_emg- trial_ts_end).min())[0][0]
                                    trial_emg_idx_end_all.append(trial_emg_idx_end)
                                  
                                    class_label[trial_emg_idx_start:trial_emg_idx_end] = mov_label

            # filter emg data
            self.emg_channels_2process = self.channel_names
            if self.filt_training_data:
                for k in range(n_channels): #for loop on number of electrodes
                    
                    for filt in channel_filterbank[k]: #channel_filterbank_emg has filters for the amount of raw channels, which is larger than the number of filtered channels. That's why this is not a problem
                        emg[self.emg_channels_2process[k]]['data'] =  filt(emg[self.emg_channels_2process[k]]['data']) 
        
            
            emg_filt = np.vstack([emg[chan]['data'][:] for chan in self.channels_2train])
            emg_rect = np.abs(emg_filt)  # Rectification
            
            ##EMG feature extraction 
            n_win_pts = int(self.win_len * self.fs)
            step_pts = int(self.step_len * self.fs)
            # emg_rect_t is a temporal data matrix that includes an extra piece of data at the beginning of the run (window length) 
            # repeating the initial value of each channel in order to have an output of features for the whole dataset.
            emg_rect_t = np.zeros((len(self.channels_2train), n_win_pts+len(emg_rect[0,:])))
            
            for ind_ch in range(len(self.channels_2train)):
                emg_rect_t[ind_ch,:] = np.hstack([np.repeat(emg_rect[ind_ch,0],n_win_pts) ,emg_rect[ind_ch, :]])

            if class_mode == 'windows':
                # start_idxs = np.arange(0, len(ts_emg) - n_win_pts + 1, step_pts) 
                start_idxs = np.arange(0, len(ts_emg) , step_pts) 

                # start_idxs = np.arange(0,emg_rect_t.shape[1],step_pts)
                # start_idxs = np.arange(0,emg_rect_t.shape[1]- n_win_pts +1, step_pts)
                # this features extractor is not 100% correct - the first feature point is the optuput of the first window of data
                f_extractor = self.extractor_cls(None, emg_channels = self.channels_2train, feature_names = self.feature_names, feature_fn_kwargs = self.feature_fn_kwargs, win_len=self.win_len, fs=self.fs)
                features = np.zeros((len(start_idxs), f_extractor.n_features))
                class_label_features = np.zeros((len(start_idxs)))
                ts_feat= []
                # import pdb;pdb.set_trace()

                for i, start_idx in enumerate(start_idxs):
                    end_idx = start_idx + n_win_pts 

                    # samples has shape (n_chan, n_win_pts) 
                    # samples = emg_rect[:,start_idx:end_idx]
                    samples = emg_rect_t[:,start_idx:end_idx]

                    #predominant_class = int(max(Counter(class_label[start_idx:end_idx])))
                    predominant_class = int(mode(class_label[start_idx:end_idx])[0]) #should we take the predominant class of just look at the last label


                    if samples.shape[1] == 0:
                        import pdb; pdb.set_trace()

                    features[i, :] = f_extractor.extract_features(samples).T
                    ts_feat.append(np.mean(ts_emg[start_idx:end_idx]))
                    class_label_features[i] = predominant_class


            elif class_mode == 'trials':

                features = np.zeros((len(trial_emg_idx_start_all),self.n_features))
                class_label_features = np.zeros(len(trial_emg_idx_start_all))

                for trial_idx in range(len(trial_emg_idx_start_all)):
                    win_len = trial_emg_idx_end_all[trial_idx] - trial_emg_idx_start_all[trial_idx]

                    f_extractor = self.extractor_cls(None, emg_channels = self.channels_2train, feature_names = self.feature_names, feature_fn_kwargs = self.feature_fn_kwargs, win_len=self.win_len, fs=self.fs)

                    samples = emg_rect_t[:,trial_emg_idx_start_all[trial_idx]:trial_emg_idx_end_all[trial_idx]]
                    predominant_class = int(mode(class_label[trial_emg_idx_start_all[trial_idx]:trial_emg_idx_end_all[trial_idx]])[0]) #should we take the predominant class of just look at the last label
                    features[trial_idx, :] = f_extractor.extract_features(samples).T
                    class_label_features[trial_idx] = predominant_class
            
            task_len = len(task[:]['ts'])
            feat_len = features.shape[0]

            if task_len < feat_len :
                features = features[0:task_len,:]
                class_label_features = class_label_features[0:task_len]

            ts_feat = np.hstack((ts_feat))
            # normalize data of one same run
            # Z-score normalization of train and test data
            self.features_mean= np.mean(features, axis=0)

            if self.extractor_kwargs['use_scalar_fixed_var']:
                print 'using scalar var'
                self.features_std = np.mean(np.std(features, axis=0))
                self.scalar_fixed_var = True
            else:
                self.features_std = np.std(features, axis=0)
                self.scalar_fixed_var = False

            if self.normalize_data is True:
                features = (features - self.features_mean) / self.features_std
    
            self.features_std_train = self.features_std
            self.features_mean_train = self.features_mean

            self.recent_features_mean_train = self.features_mean
            self.recent_features_std_train = self.features_std
            
            # once the features are extracted for the whole run, we take only the data 
            # that corresponds to moving_states (rest data is excluded) for training
            # import pdb; pdb.set_trace()
            idx_mov = np.where(class_label_features != 0)
            features = features[idx_mov[0],:]
            class_label_features = class_label_features[idx_mov[0]]
            try:
                vel_filt = task[:]['plant_vel_filt'][idx_mov[0],:]
            except:
                fs_synch = 20 #Frequency at which emg and kin data are synchronized
                nyq   = 0.5 * fs_synch
                cuttoff_freq  = 1.5 / nyq
                bpf_kin_coeffs = butter(4, cuttoff_freq, btype='low')
                lpf = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1])
                vel_filt = np.zeros(task[:]['plant_vel'].shape)
                for ind_dof in range(task[:]['plant_vel'].shape[1]):

                    vel_filt = lpf(task[:]['plant_vel'][:,ind_dof])
                    vel_filt = vel_filt[idx_mov[0],:]

            # vel_filt_Z = (vel_filt - np.mean(vel_filt, axis=0)) / np.std(vel_filt,axis=0)
            vel_filt_Z = vel_filt
            if hdf_id in train_hdf_ids:
                if feature_data_train is None: #first run
                    feature_data_train = features.copy()
                    feature_label_train = class_label_features.copy()
                    vel_filt_train = vel_filt_Z.copy()                  
                else:
                    feature_data_train = np.vstack([feature_data_train, features])    
                    feature_label_train = np.hstack([feature_label_train, class_label_features]) 
                    vel_filt_train = np.vstack([vel_filt_train, vel_filt_Z]) 
       
            # we take all the data for testing -- then calculate accuracy only in points where class_label !=0            
            if hdf_id in test_hdf_ids:
                if feature_data_test is None:
                    feature_data_test = features.copy()
                    ts_features_test = ts_feat.copy()
                    feature_label_test = class_label_features.copy()
                    vel_filt_test = vel_filt_Z.copy() 
                else:
                    feature_data_test = np.vstack([feature_data_test, features])
                    feature_label_test = np.hstack([feature_label_test, class_label_features])
                    ts_features_test = np.hstack([ts_features_test, ts_feat])
                    vel_filt_test= np.vstack([vel_filt_test, vel_filt_Z]) 

            hdf.close()
            hdf_supp.close()        
        
        # #Z-score normalization of train and test data
        # self.features_mean_train = np.mean(feature_data_train, axis=0)
        # self.features_std_train = np.std(feature_data_train, axis=0)
        # self.features_mean_test = np.mean(feature_data_test, axis=0)
        # self.features_std_test = np.std(feature_data_test, axis=0)

        # if self.normalize_data is True:
        #     emg_features_train = (feature_data_train - self.features_mean_train) / self.features_std_train
        #     # emg_features_test = (feature_data_test - self.features_mean_train) / self.features_std_train
        #     emg_features_test = (feature_data_test - self.features_mean_test) / self.features_std_test
        # else:
        #     emg_features_train = feature_data_train
        #     emg_features_test = feature_data_test
            
        # return emg_features_train, feature_label_train, emg_features_test, feature_label_test
        return feature_data_train, feature_label_train, vel_filt_train, feature_data_test, feature_label_test, ts_features_test,vel_filt_test

    def train_svm(self, C, gamma, train_data, train_label):
        '''Use SVM to train this decoder from data from multiple .hdf files.'''
        self.gamma = gamma
        self.C = C

        clf = svm.SVC(gamma=self.gamma, C=self.C,probability=True) # build svm classifier with the input parameters

        print "training classifier ..."
        clf.fit(train_data, train_label)
        print "classifier is trained"
     
        self.classifier = clf

    def test_svm(self, test_data, test_label):

        Z_features_test = test_data
        feature_label_test = test_label

        #test classfier
        print 'predicting test set'
        predicted_label= self.classifier.predict(Z_features_test)
        predicted_prob= self.classifier.predict_proba(Z_features_test)

        #evaluate only in states used for trainind (class_label >0)
        acc_score = accuracy_score(feature_label_test[feature_label_test>0], predicted_label[feature_label_test>0])


        print 'acc_score : ', acc_score


        self.classifier.acc_score = acc_score
        self.classifier.feature_label_test = feature_label_test
        self.classifier.predicted_label = predicted_label

        plt.figure()     
        plt.plot(Z_features_test[:,:])   
        plt.plot(feature_label_test, color='red', linewidth=3.)
        plt.plot(predicted_label, '*g')
        
        # plt.plot(predicted_prob, color='yellow')
        plt.legend(['true class', 'predicted class','EMG feat'])

        # print 'Close the figure in order to continue '
        plt.show()

        return predicted_label, predicted_prob

    def get_LM_from_train(self,prob_train, kin_train, dofs):
        self.m = []
        self.b = []
        for ind_dof in dofs:
            slope, intercept, r_value, p_value, std_err = linregress(prob_train,kin_train[:,ind_dof])
            self.m.append(slope)
            self.b.append(intercept)


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

    def __call__(self, features, output_type):
        classifier_output = []

        if output_type == 'label':
            classifier_output = self.classifier.predict(features)
        elif output_type == 'prob':
            classifier_output = self.classifier.predict_proba(features)[:,1]
        return classifier_output

    
    def process_data(self,train_hdf_names,test_hdf_names, normalize_data):
        '''Process train and testing data for classification'''

        self.train_hdf_names = train_hdf_names
        self.test_hdf_names = test_hdf_names
        self.normalize_data = normalize_data

        train_features = []
        train_labels = []

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
            emg_msgs = hdf.root.brainamp_msgs[:][self.channel_names_original]
            
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


            #we did not record the ts of the task in some files, let's use the ts of the 'rehand 'instead
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
                    trial_ts_end = rehand[trial_idx_end]['ts']

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
            step_pts = int(self.step_len * self.fs)
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
            import pdb;pbd.set_trace()
                        
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

        self.features_mean_train = np.mean(feature_data_train, axis=0)
        self.features_std_train = np.std(feature_data_train, axis=0)

        self.features_mean_test = np.mean(feature_data_test, axis=0)
        self.features_std_test = np.std(feature_data_test, axis=0)

        if self.normalize_data == True:

            # # Z-score normalization of train and test data
            train_features = (feature_data_train - self.features_mean_train) / self.features_std_train
            test_features = (feature_data_test - self.features_mean_train) / self.features_std_train

        else:

            train_features = feature_data_train
            test_features = feature_data_test  

        # save mean and std of training dataset to normalize testing dataset
        self.classifier.features_std_train = self.features_std_train
        self.classifier.features_mean_train = self.features_mean_train
        


        return train_features, feature_label_train, test_features, feature_label_test

    def train_svm(self, C, gamma, Z_features_train, feature_label_train):
        # build svm classifier with the input parameters
        clf = svm.SVC(gamma=self.gamma, C=self.C, probability= True)

        # train classifier
        print "training classifier ..."
        clf.fit(Z_features_train, feature_label_train)
        print "classifier is trained"
     
        #check if these parameters can be obtained inside this function
        self.classifier = clf
        self.classifier.output_classes = self.output_classes
        self.classifier.num_output_classes  = self.num_output_classes 
        self.classifier.mov_class_labels = self.mov_class_labels


    def test_svm(self, Z_features_test, feature_label_test):

        print "shape Z_features_test ", np.shape(Z_features_test)
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

        import pdb; pdb.set_trace()
        print 'Close the figure in order to continue '
        plt.show()



        


