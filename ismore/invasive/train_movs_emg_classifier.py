import tables
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
import pickle
import time

from ismore import brainamp_channel_lists

from ismore.invasive import discrete_movs_emg_classification

from ismore.noninvasive.emg_feature_extraction import EMGMultiFeatureExtractor
from ismore.common_state_lists import *
from utils.constants import *
from db import dbfunctions as dbfn
from db.tracker import models
from matplotlib import pyplot as plt

saveClassifier = True
use_scalar_fixed_var = True

dbname = 'default'
# dbname = 'tecnalia'

emg_channels = brainamp_channel_lists.emg14_bip #list of recorded channels

filt_training_data = True

## Feature Extraction
feature_names = ['WL']
# feature_names = ['MAV', 'VAR', 'WL', 'RMS', 'LOGVAR']#,'WAMP','ZC','SSC']
win_len  = 0.500 # secs
# win_len  = 1. # secs
# win_len  = 2. # secs
step_len = 0.050 # secs
# step_len = 0.001 #secs
fs = 1000  # Hz


feature_fn_kwargs = {
    'WAMP': {'threshold': 30},  
    'ZC':   {'threshold': 30},
    'SSC':  {'threshold': 700},
}


extractor_cls = EMGMultiFeatureExtractor

#set svm classifier parameters
C=1.0
gamma=0.01

# ---------------------------------- Multi-movement classification ------------------------ #

# Calibration H - 2017.07.11
calibration_H_pre1 = [4737,4738,4742,4743,4746,4747] #R

# Calibration P - 2017.07.12s
calibration_P_pre1 = [4769,4770,4773,4774,4777,4778] #R


# Calibration P - 2017.07.14
calibration_P_pre2 = [4795,4796,4807,4811,4813] #R #4797 not saved
[4767,4771,4802,4809] #GT

# Calibration H - 2017.09.20
calibration_H_post1 = [6967,6968,6971,6973,6974,6979,6980,6987,6988]
# 6976,6982,6984 --> USB connection lost with exo at some point, inomplete runs.

# Calibration P - 2017.09.18
calibration_P_post1 = [6937,6938,6946,6949,6950,6953,6954] #R
[6935,6947,6951,] #GT

# Calibration P
calibration_P_post2 = [9426,9627,9429,9430,9431,9432] #R -- 2017.12.06 -- neural data also recorded, spikes and raw data

# Calibration H
calibration_H_post2 = [9690,9691,9692,9693, 9694, 9695, 9696, 9697]

#######-------------------------------------------------------- MEASUREMENTS ----------------------------------------------------########


train_te_list = calibration_H_pre1 + calibration_H_post1 + calibration_H_post2 # all movements

test_te_list = calibration_P_post1

# #Hand movements
# select list of channels used for training
channels_2train = [
    'InterFirst',
    'AbdPolLo',
    'ExtCU',
    'ExtCarp',
    'ExtDig',
    'FlexDig',
    'FlexCarp',]
    # 'PronTer',
    # 'Biceps',
    # 'Triceps',
    # 'FrontDelt',
    # 'MidDelt']

for subset_muscles in [channels_2train]:
    subset_muscles_ix = [emg_channels.index(subset_muscles[i]) for i in range(len(subset_muscles))]

# channels_2train = emg_channels

tt2classify = ['grasp'] # trial_types to classify
movs2classify = ['rest-grasp', 'grasp-back']
movs_labels = [1,2]

tt2classify = ['grasp', 'blue_grasp', 'grasp_down', 'grasp_up'] # trial_types to classify
movs2classify = ['rest-grasp', 'grasp-back', 'rest-blue_grasp',  'blue_grasp-back' , 'rest-grasp_down',  'grasp_down-back', 'rest-grasp_up' , 'grasp_up-back' ]
movs_labels = [1,2,1,2,1,2,1,2]

# tt2classify = ['grasp', 'point'] # trial_types to classify
# movs2classify = ['rest-grasp', 'grasp-back', 'rest-point', 'point-back']
# movs_labels = [1,2,3,4]


# # tt2classify = ['grasp','point','up'] 
# # movs2classify = ['rest-grasp', 'grasp-back', 'rest-point', 'point-back', 'rest-up', 'up-back', 'rest-down', 'down-back']
# # movs_labels = [1,2,3,4,5,6,7,8] # if label =0, we do not consider that data for testing, only for plotting

# tt2classify = ['grasp','point','up','down'] 
# movs2classify = ['rest-' + tt for tt in tt2classify ]
# movs_labels = [1,2,3,4]

# tt2classify = ['red_up' , 'red_down' , 'green_point', 'blue_grasp']
# movs2classify = ['rest-' + tt for tt in tt2classify ]
# movs_labels = [1,2,3,4]



# tt2classify = ['up','down'] 
# movs2classify = [ 'rest-up', 'rest-down', 'down-back', 'up-back', ]
# movs_labels = [1,2,1,2]

# # movs2classify = [ 'rest-up', 'rest-down']
# # movs_labels = [1,2]

# tt2classify = ['grasp','point'] 
# movs2classify = ['rest-grasp','rest-point']
# movs_labels = [1,2]

# --- Arm movements
# channels_2train = [
#     'Biceps',
#     'Triceps',
#     'FrontDelt',
#     'MidDelt',
#     'TeresMajor',
#     'PectMajor']

# channels_2train = emg_channels 

# tt2classify = ['red','green','blue','red to blue', 'red to green','blue to red', 'blue to green']
# tt2classify = ['red','green','blue']

# movs2classify = ['rest-' + tt for tt in B1_targets ]
# # movs2classify = [tt + '-back' for tt in B1_targets ]
# movs_labels = [1,2,3]

# # ## --------------
# B2_targets = ['grasp','point','up','down'] 
# B3_targets = ['grasp_up', 'grasp_down', 'point_up', 'point_down']


# # Invasive - compliant blocks
# blk1_targets = ['red', 'green', 'blue', 'red_to_blue', 'red_to_green','blue_to_red', 'blue_to_green']
# blk2_targets = B2_targets + B3_targets
# blk3_targets = ['red_up' , 'red_down' , 'green_point', 'blue_grasp']
# blk4_targets = ['red_grasp_up', 'red_point_down','green_grasp_down','blue_grasp_up']


# tt2classify = blk3_targets
# movs2classify = ['rest-' + tt for tt in tt2classify ]
# channels_2train = emg_channels
# movs_labels = [1,2,3,4]

# ## --- Combined arm-hand movements

# channels_2train = emg_channels
# tt2classify = ['red_up','green_point','blue_grasp' ] 
# movs2classify = ['rest-red_up','rest-green_point', 'rest-blue_grasp']
# movs_labels = [1,2,3]

### see differences between healthy and paretic
# train_te_list = calibration_H_pre1 
# train_te_list = calibration_P_pre1

# channels_2train = emg_channels
# tt2classify = ['red' ] 
# movs2classify = ['rest-red']
# movs_labels = [1]


# ## --------------


extractor_kwargs = {
    'emg_channels':      emg_channels,
    'feature_names':     feature_names,
    'feature_fn_kwargs': feature_fn_kwargs,
    'win_len':           win_len,
    'step_len':          step_len,
    'fs':                fs,
    'channels_2train':   channels_2train,
    'subset_muscles_ix':  subset_muscles_ix,
    'use_scalar_fixed_var': use_scalar_fixed_var,
}

# Task types - data
from db import dbfunctions as dbfn
import numpy as np
import unicodedata


def get_trial_type_te_list(te_list , mov_list, dbname):
    mov_te_list = []

    for idx_te, te_id in enumerate(te_list):
        print 'checking te : ', te_id
        try:
            te = dbfn.TaskEntry(te_id)
            task_name= unicodedata.normalize('NFKD', te.task.name).encode('ascii','ignore')
            if task_name not in ['ismore_disable_system', 'ismore_recordGoalTargets']:
                trial_types_te = np.unique(te.hdf.root.task[:]['trial_type'])
                for idx_tt, tt in enumerate(trial_types_te):
                    print 'trial type : ', tt
                    if tt in mov_list:
                        mov_te_list.append(te_id)
            te.close()
            te.close_hdf()

        except:
            print 'data not found in storage'
            pass

    mov_te_list = np.unique(mov_te_list).tolist()
    # mov_te_list = mov_te_list.tolist()
    print 'Final mov_te_list is : ', mov_te_list
    return mov_te_list 


# get task entries with specific type of trial to be classified
train_hdf_ids = get_trial_type_te_list(train_te_list, tt2classify, dbname)
test_hdf_ids = get_trial_type_te_list(test_te_list, tt2classify, dbname)

# test_hdf_ids = test_hdf_ids[0:len(test_hdf_ids)/3]
# test_hdf_ids = test_hdf_ids[(len(test_hdf_ids)/3)+1:np.int(len(test_hdf_ids)*(2./3.))]
# test_hdf_ids = test_hdf_ids[np.int(len(test_hdf_ids)*(2./3.))+1:-1]


normalize_data = True
mov_classifier = discrete_movs_emg_classification.SVM_mov_EMGClassifier(channels_2train, filt_training_data, 
    extractor_cls, extractor_kwargs)

class_mode = 'trials'
class_mode = 'windows'

[train_data, train_label, vel_filt_train, _, _, _, _] = mov_classifier.process_data(train_hdf_ids, [], normalize_data,tt2classify,movs2classify, movs_labels,dbname, class_mode)
# [_ , _ , _ , test_data, test_label, ts_features_test, vel_filt_test] = mov_classifier.process_data([], test_hdf_ids, normalize_data,tt2classify,movs2classify, movs_labels,dbname,class_mode)


mov_classifier.train_svm(C, gamma, train_data, train_label)

predicted_label_train, predicted_prob_train = mov_classifier.test_svm(train_data, train_label)
# predicted_label, predicted_prob = mov_classifier.test_svm(test_data, test_label)

# training data
x_train = predicted_prob_train[:,1]
# y_train = vel_filt_train[:,3]
# slope, intercept, r_value, p_value, std_err = linregress(x_train,y_train)

rh_dof = [3,4,5]
mov_classifier.get_LM_from_train(x_train,vel_filt_train,rh_dof)


# mov_classifier.m = m
# mov_classifier.b = b

# save grasp_emg_classifier
mov_classifier.training_ids = train_hdf_ids
classifier_name = 'grasp_emg_classifier_scalarvar_%s_%s' %(str(use_scalar_fixed_var), time.strftime('%Y%m%d_%H%M'))
pkl_name = classifier_name + '.pkl'
storage_dir = '/storage/decoders'
mov_classifier.path = os.path.join(storage_dir, pkl_name)
pickle.dump(mov_classifier, open(os.path.join(storage_dir, pkl_name), 'wb'))


# #testing data
# pred_kin_test = np.zeros([len(predicted_prob),len(rh_dof)])
# for idx_dof, ind_dof in enumerate(rh_dof):
#     pred_kin_test[:,idx_dof] = m[idx_dof]* predicted_prob[:,1] + b[idx_dof]



# plt.figure()
# plt.plot(pred_kin_test)
# plt.plot(vel_filt_test[:,rh_dof])
# plt.plot(test_label)


# # check rest_emg_classifier output for bmi sessions using rest_emg_classifier
# for te_id in bmi_invasive_tes:
#     print te_id
#     te = dbfn.TaskEntry(te_id, dbname = dbname)
#     rest_emg_output = te.hdf.root.task[:]['rest_emg_output']
#     print str(te.date.month) + '_' + str(te.date.day)
#     plt.plot(rest_emg_output)
#     plt.show()



# ### see differences between healthy and paretic
# channels_2train = emg_channels
# tt2classify = ['red' ] 
# movs2classify = ['rest-red']


# normalize_data = True
# mov_classifier = discrete_movs_emg_classification.SVM_mov_EMGClassifier(channels_2train, filt_training_data, extractor_cls, extractor_kwargs)

# train_te_list = calibration_H_pre1 
# movs_labels = [1]
# train_hdf_ids = get_trial_type_te_list(train_te_list, tt2classify, dbname)
# [train_data1, train_label1, test_data, test_label] = mov_classifier.process_data(train_hdf_ids, [], normalize_data,tt2classify,movs2classify, movs_labels,dbname)

# train_te_list = calibration_P_pre1 
# movs_labels = [2]
# train_hdf_ids = get_trial_type_te_list(train_te_list, tt2classify, dbname)
# [train_data2, train_label2, test_data, test_label] = mov_classifier.process_data(train_hdf_ids, [], normalize_data,tt2classify,movs2classify, movs_labels,dbname)

# train_data = np.vstack([train_data1, train_data2])
# train_label = np.hstack([train_label1, train_label2])

# mov_classifier.train_svm(C, gamma, train_data, train_label)



# train_te_list = calibration_P_pre1






# # # ---------------------------------- Rest vs mov classification ------------------------ #

# # states2classify = ['rest', 'trial', 'trial_return']

# # rest_classifier = discrete_movs_emg_classification.SVM_rest_EMGClassifier(channels_2train, fs, win_len, filt_training_data, extractor_cls, extractor_kwargs, classifier_type)
# # classifier_MovNoMov.train_svm(C, gamma, train_hdf_names, test_hdf_names)
# # classifier.classifier_MovNoMov = classifier_MovNoMov


# print 'rest classifier trained'

# rest_classifier.training_ids = train_hdf_ids

# train_ids_str = str(min(train_hdf_ids)) + '_' + str(max(train_hdf_ids))

# subject_name  = models.TaskEntry.objects.using(dbname).get(id=train_hdf_ids[0]).subject.name

# classifier_name = 'emg_classifier_%s_%s_%s' % (subject_name,train_ids_str, time.strftime('%Y%m%d_%H%M'))
# pkl_name = classifier_name + '.pkl'
# rest_classifier.classifier_name = classifier_name


# # --------------------
if saveClassifier:
    ## Store a record of the data file in the database
    storage_dir = '/storage/decoders'
    if not os.path.exists(storage_dir):
        os.popen('mkdir -p %s' % storage_dir)

    #pickle.dump(mov_classifier, open(os.path.join(storage_dir, pkl_name), 'wb'))


    # Create a new database record for the decoder object if it doesn't already exist
    dfs = models.Decoder.objects.filter(name=classifier_name)


    if len(dfs) == 0:
        df = models.Decoder()
        df.path = pkl_name
        df.name = classifier_name
        df.entry = models.TaskEntry.objects.using(dbname).get(id=min(train_hdf_ids))
        # # if you recorded hdf files in another machine and you want to read them in a new machine and save the classfier in this new machine:
        # #df.entry = models.TaskEntry.objects.using(dbname).get(id=an_id_in_our_current_db_where_we_used_a_decoder)
        # dbname = 'default'
        # df.entry = models.TaskEntry.objects.using(dbname).get(id=3578)
        df.save()
    elif len(dfs) == 1:
        pass # no new data base record needed
    elif len(dfs) > 1:
        print "More than one classifier with the same name! fix manually!"

# # --------------------



