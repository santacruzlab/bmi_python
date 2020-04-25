import tables
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
import pickle
import time


from ismore import brainamp_channel_lists
from ismore.noninvasive import emg_classification


from ismore.noninvasive.emg_feature_extraction import EMGMultiFeatureExtractor
from ismore.common_state_lists import *
from utils.constants import *
from db import dbfunctions as dbfn
from db.tracker import models

##svm
from sklearn import svm

saveClassifier = True

db_name = 'tubingen'
db_name = 'default'

#db_name = 'tubingen'
#Noise signal with the simulator
train_hdf_ids = [8076, 8077, 8078, 8079]
test_hdf_ids = [8080]
channels_2train = brainamp_channel_lists.emg14


#db_name = 'tubingen'
train_hdf_ids = [8164, 8165, 8166, 8167]
test_hdf_ids = [8168]

#db_name = 'tecnalia'
train_hdf_ids = [4071]
test_hdf_ids = [4072]

#db_name = 'tecnalia' 
#data recorded by nerea herself in tecnalia. compliant. 
#1. red + pointing+ pronation. 2. green + supination + opne hand. 3. blue + pinch (inde-thumb)
train_hdf_ids = [4113, 4114, 4115, 4116,4117]
test_hdf_ids = [4117]
emg_channels = brainamp_channel_lists.emg12_bip
channels_filt = brainamp_channel_lists.emg12_bip_filt


db_name = 'default'
# DK
train_hdf_ids = [8280, 8281, 8282,8283]
test_hdf_ids = [8218]
# CB
# train_hdf_ids = [7482, 7493]
# test_hdf_ids = [8218,8221,8222,8223,8224]

# -------------------------------------#

#data from calibration session of EMG learning protocol, right arm compliant, raghing+supination+open-hand.


#CB
db_name = 'tubingen'
emg_channels = brainamp_channel_lists.emg_6bip_hd_filt#emg_48hd_6mono
channels_filt = brainamp_channel_lists.emg_6bip_hd_filt#emg_48hd_6mono_filt
channels_2train = brainamp_channel_lists.emg_6bip_hd_filt#emg_48hd_6mono_filt

# DK
# db_name = 'tubingen'
# emg_channels = brainamp_channel_lists.emg_6mono_hd
# channels_filt = brainamp_channel_lists.emg_6mono_hd
# channels_2train = brainamp_channel_lists.emg_6mono_hd


# emg_channels = brainamp_channel_lists.emg_upper_arm_hd_filt#emg_48hd_6mono
# channels_filt = brainamp_channel_lists.emg_upper_arm_hd_filt#emg_48hd_6mono_filt
channels_2train = brainamp_channel_lists.emg_upper_arm_hd_filt#emg_48hd_6mono_filt

# db_name = 'tubingen'
# emg_channels = brainamp_channel_lists.emg_48hd_6mono
# channels_filt = brainamp_channel_lists.emg_48hd_6mono_filt
# channels_2train = brainamp_channel_lists.emg_48hd_6mono_filt


#AS
# train_hdf_ids = [7652,7653,7654,7655]
# test_hdf_ids = [7656]

# #MW
# train_hdf_ids = [7811,7812,7813,7814]
# test_hdf_ids = [7815]

# #NM
# train_hdf_ids = [7821,7822,7823,7824]
# test_hdf_ids = [7825]

# #PZ
# train_hdf_ids = [7547,7548,7549,7550]
# test_hdf_ids = [7551]


# #RE patient
#db_name = 'tecnalia'
#train_hdf_ids = [2651,2652]
#test_hdf_ids = [2653]

# db_name = 'tecnalia'
# train_hdf_ids = [2651,2652]
# test_hdf_ids = [2653]


all_hdf_ids = np.append(train_hdf_ids, test_hdf_ids)


dec_acc_MultiClass = []
dec_acc_MovNoMov = []

# no CV
doCV = False

if doCV: #within runs of same session
    nCVfolds = len(all_hdf_ids)
elif doCV == False:
    nCVfolds =1

for indCV in range(nCVfolds):

    test_hdf_ids = [all_hdf_ids[indCV]]
    train_hdf_ids = [x for x in all_hdf_ids if x != test_hdf_ids]

    # -------------------------------------#
    # test_relevant_channels_hdf_ids = test_hdf_ids
    # test_relevant_channels_hdf_names = []
    # for id in test_relevant_channels_hdf_ids:
    #     te = dbfn.TaskEntry(id, dbname= db_name)
    #     test_relevant_channels_hdf_names.append(te.hdf_filename)
    #     te.close_hdf()

    # no CV

    #CB
    train_hdf_ids = [7482, 7493] # S1 compliant session Left Arm (Motor Learning study)
    train_hdf_ids = [7500,7511] # S2 compliant session Left Arm (Motor Learning study)
    train_hdf_ids = [7515,7525] # S3 compliant session Left Arm (Motor Learning study)
    train_hdf_ids = [7530,7541] # S4 compliant session Left Arm (Motor Learning study)    
    train_hdf_ids = [7482, 7493, 7500,7511,7515,7525,7530,7541] #S1, S2, S3, S4
    test_hdf_ids = [8221]#,8222,8223,8224] #hybrid left arm
    
    # # # DK
    train_hdf_ids = [8281, 8282,8283] # healthy complian runs
    test_hdf_ids = [8292,8294,8295] # hybrid paretic arm
    test_hdf_ids = [8294] 
    train_hdf_ids = [8294] 

    train_hdf_names = []

    for id in train_hdf_ids:
        te = dbfn.TaskEntry(id, dbname= db_name)
        train_hdf_names.append(te.hdf_filename)
        te.close_hdf()

    test_hdf_names = []
    for id in test_hdf_ids:
        te = dbfn.TaskEntry(id, dbname=db_name)
        test_hdf_names.append(te.hdf_filename)
        te.close_hdf()

    import pdb; pdb.set_trace()
    filt_training_data = True
    # filt_training_data = False
    bip_and_filt_needed = True
    bip_and_filt_needed = False

    ## Feature Extraction
    feature_names = ['WL']
    #feature_names = ['MAV', 'VAR', 'WL', 'RMS', 'LOGVAR']
    win_len = 1  # secs
    fs = 1000  # Hz

    #not using them 
    feature_fn_kwargs = {
        'WAMP': {'threshold': 30},  
        'ZC':   {'threshold': 30},
        'SSC':  {'threshold': 700},
    }

    extractor_kwargs = {
        'emg_channels':      emg_channels,
        'feature_names':     feature_names,
        'feature_fn_kwargs': feature_fn_kwargs,
        'win_len':           win_len,
        'fs':                fs,
        'channels_2train':   channels_2train,

    }
            
    # channels_str_2discard = [5,11,17,23,29,35,41,47,49,51,53,55,57]
    # channels_str_2keep = [i for i in range(59) if i not in channels_str_2discard]
    # channels_diag1_1 = [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46]
    # channels_diag1_2 = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41]
    # channels_diag2_1 = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40]
    # channels_diag2_2 = [7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47]
    # channel_names = [str(i) + 'str_filt' for i in range(len(channels_str_2keep)-6)] + brainamp_channel_lists.emg_6bip_hd_filt + [str(j) + 'diag1_filt' for j in range(len(channels_diag1_1))] + [str(k) + 'diag2_filt' for k in range(len(channels_diag2_1))]


    # bip_idxs = [ind for ind, c in enumerate(channel_names) if c in brainamp_channel_lists.emg_6bip_hd_filt]
    # nchans_hd = len(channel_names) - len(bip_idxs)
    # nchans_bip = len(bip_idxs)

    # chan_out_final = chan_out[:nchan_out]
    # chan_in = np.array([chan for chan in np.arange(nchans_bip + nchans_hd) if chan not in chan_out_final])


    # opt_channels = []
    # opt_channels_dict = dict()


    # opt_channels_dict['channels_str_2discard'] = channels_str_2discard
    # opt_channels_dict['channels_str_2keep'] = [channels_str_2keep[i] for i in np.arange(len(channels_str_2keep[:-6])) if i in chan_in] + channels_str_2keep[-6:]
    # opt_channels = opt_channels + [channel_names[i] for i in np.arange(len(channels_str_2keep[:-6])) if i in chan_in] + brainamp_channel_lists.emg_6bip_hd_filt
    # opt_channels_dict['channels_diag1_1'] = [channels_diag1_1[i] for i in np.arange(len(channels_diag1_1)) if i in chan_in - 40] # 40 = 20ext + 20flex
    # opt_channels_dict['channels_diag1_2'] = [channels_diag1_2[i] for i in np.arange(len(channels_diag1_2)) if i in chan_in - 40] # 46 = 20ext + 20flex + 6bipolar
    # opt_channels = opt_channels + [channel_names[i+46] for i in np.arange(len(channels_diag1_1)) if i in chan_in - 40]
    # opt_channels_dict['channels_diag2_1'] = [channels_diag2_1[i] for i in np.arange(len(channels_diag2_1)) if i in chan_in - 70] # 70 = 20ext + 20flex + 30diag1
    # opt_channels_dict['channels_diag2_2'] = [channels_diag2_2[i] for i in np.arange(len(channels_diag2_2)) if i in chan_in - 70] # 76 = 20ext + 20flex + 6bipolar + 30diag1
    # opt_channels = opt_channels + [channel_names[i+76] for i in np.arange(len(channels_diag2_1)) if i in chan_in - 70] 
    # #Warning!!: chan_in channels include the bipolar ones at the end while self.channel_names include them after the straight hd channels.
    # opt_channels_2train_dict = opt_channels_dict

    #set svm classifier parameters
    C=1.0
    gamma=0.01

    # we set the following parameters to default values -- no need to declare them
    # cache_size=200
    # class_weight=None
    # coef0=0.0
    # decision_function_shape=None
    # degree=3
    # kernel='rbf'
    # max_iter=-1
    # probability=False
    # random_state=None
    # shrinking=True
    # tol=0.001
    # verbose=False


    extractor_cls = EMGMultiFeatureExtractor
    classifier = emg_classification.SVM_EMGClassifier(channels_2train, fs, win_len, filt_training_data, bip_and_filt_needed, extractor_cls, extractor_kwargs, 'Mov-NoMov&MultiClass')

    #create classifier and train
    classifier_type = 'Mov-NoMov' # -- binary classifier that classifies between rest state and movement state of muscles (in general, for any kind of task)
    classifier_MovNoMov = emg_classification.SVM_EMGClassifier(channels_2train, fs, win_len, filt_training_data, bip_and_filt_needed, extractor_cls, extractor_kwargs, classifier_type)
    classifier_MovNoMov.train_svm(C, gamma, train_hdf_names, test_hdf_names)
    classifier.classifier_MovNoMov = classifier_MovNoMov
    print 'Mov-NoMov classifier trained'
   
    dec_acc_MovNoMov.append(classifier.classifier_MovNoMov.classifier.acc_score)




    classifier_type = 'MultiClass' # -- multiclass classifier that classifies between the differents trial types that we have recorded
    classifier_MultiClass= emg_classification.SVM_EMGClassifier(channels_2train, fs, win_len, filt_training_data, bip_and_filt_needed, extractor_cls, extractor_kwargs, classifier_type)
    classifier_MultiClass.train_svm(C, gamma, train_hdf_names, test_hdf_names)
    classifier.classifier_MultiClass = classifier_MultiClass
    print 'MultiClass classifier trained'

    dec_acc_MultiClass.append(classifier.classifier_MultiClass.classifier.acc_score)




    classifier.training_ids = train_hdf_ids

    train_ids_str = str(min(train_hdf_ids)) + '_' + str(max(train_hdf_ids))

    subject_name  = models.TaskEntry.objects.using(db_name).get(id=train_hdf_ids[0]).subject.name

    subject_name  = models.TaskEntry.objects.using("tubingen").get(id=train_hdf_ids[0]).subject.name

    classifier_name = 'emg_classifier_%s_%s_%s' % (subject_name,train_ids_str, time.strftime('%Y%m%d_%H%M'))
    pkl_name = classifier_name + '.pkl'
    classifier.classifier_name = classifier_name


# import pdb; pdb.set_trace()


# # --------------------
if saveClassifier:
    ## Store a record of the data file in the database
    storage_dir = '/storage/decoders'
    if not os.path.exists(storage_dir):
        os.popen('mkdir -p %s' % storage_dir)

    pickle.dump(classifier, open(os.path.join(storage_dir, pkl_name), 'wb'))


    # Create a new database record for the decoder object if it doesn't already exist
    dfs = models.Decoder.objects.filter(name=classifier_name)


    if len(dfs) == 0:
        df = models.Decoder()
        df.path = pkl_name
        df.name = classifier_name
        df.entry = models.TaskEntry.objects.using(db_name).get(id=min(train_hdf_ids))
        # # if you recorded hdf files in another machine and you want to read them in a new machine and save the classfier in this new machine:
        # #df.entry = models.TaskEntry.objects.using(db_name).get(id=an_id_in_our_current_db_where_we_used_a_decoder)
        # db_name = 'default'
        # df.entry = models.TaskEntry.objects.using(db_name).get(id=3578)
        df.save()
    elif len(dfs) == 1:
        pass # no new data base record needed
    elif len(dfs) > 1:
        print "More than one classifier with the same name! fix manually!"

# # --------------------
