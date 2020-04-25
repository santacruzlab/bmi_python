import tables
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
import pickle
import time

from ismore import brainamp_channel_lists
from ismore.invasive import rest_emg_classifier

from ismore.noninvasive.emg_feature_extraction import EMGMultiFeatureExtractor
from ismore.common_state_lists import *
from utils.constants import *
from db import dbfunctions as dbfn
from db.tracker import models

from sklearn import svm

def train_test_rest_classifier_from_te_list(train_hdf_ids, test_hdf_ids, 
    db_name = 'default', saveClassifier = True, use_scalar_fixed_var = False):

    channels_2train = brainamp_channel_lists.emg14_bip #take from data, take already the filtered emg data?
    emg_channels = brainamp_channel_lists.emg14_bip

    all_hdf_ids = np.append(train_hdf_ids, test_hdf_ids)

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

    filt_training_data = True

    ## Feature Extraction
    feature_names = ['WL']
    #feature_names = ['MAV', 'VAR', 'WL', 'RMS', 'LOGVAR']
    win_len = 0.2  # secs
    step_len = 0.020 # secs
    fs = 1000  # Hz

    extractor_kwargs = {
        'emg_channels':      emg_channels,
        'feature_names':     feature_names,
        'win_len':           win_len,
        'fs':                fs,
        'channels_2train':   channels_2train,
        'use_scalar_fixed_var': use_scalar_fixed_var,
        }

     #set svm classifier parameters
    C=1.0
    gamma=0.01

    extractor_cls = EMGMultiFeatureExtractor
    rest_classifier = rest_emg_classifier.SVM_rest_EMGClassifier(channels_2train, fs, 
        win_len, step_len, filt_training_data, extractor_cls, extractor_kwargs)
    rest_classifier.train_svm(C, gamma, train_hdf_names, test_hdf_names)

    print 'rest classifier trained'

    rest_classifier.training_ids = train_hdf_ids

    train_ids_str = str(min(train_hdf_ids)) + '_' + str(max(train_hdf_ids))

    subject_name  = models.TaskEntry.objects.using(db_name).get(id=train_hdf_ids[0]).subject.name

    classifier_name = 'emg_classifier_%s_%s_%s_scalar_var_%s' % (subject_name,train_ids_str, time.strftime('%Y%m%d_%H%M'), str(use_scalar_fixed_var))
    pkl_name = classifier_name + '.pkl'
    rest_classifier.classifier_name = classifier_name

    # # --------------------
    if saveClassifier:
        ## Store a record of the data file in the database
        storage_dir = '/storage/decoders'
        if not os.path.exists(storage_dir):
            os.popen('mkdir -p %s' % storage_dir)

        pickle.dump(rest_classifier, open(os.path.join(storage_dir, pkl_name), 'wb'))

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
