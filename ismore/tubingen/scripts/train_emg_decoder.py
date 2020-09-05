import tables
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
import pickle
import sys
import yaml

from ismore.tubingen import brainamp_channel_lists
from ismore.tubingen.noninvasive_tubingen import emg_decoding
from ismore.noninvasive import find_relevant_channels

from ismore import ismore_bmi_lib

from ismore.tubingen.noninvasive_tubingen.emg_feature_extraction import EMGMultiFeatureExtractor
from ismore.common_state_lists import *
from utils.constants import *
from db import dbfunctions as dbfn
from db.tracker import models

from ismore.tubingen.noninvasive_tubingen.sorting import SubjectFeedbackGroup


def main():
    ''' Function that will execute the rest of this script when called from the command line.'''
    
    try:
        config_id = sys.argv[1]
        train_hdf_ids = sys.argv[2].split(',')

        test_hdf_ids = []
        if len(sys.argv) > 3:
            test_hdf_ids = sys.argv[3]

        print("")
        print("Task entries used for decoder training of patient "+config_id+": "+str(train_hdf_ids))
        print("")
        train_emg_decoder(config_id=config_id, train_hdf_ids=train_hdf_ids, test_hdf_ids=test_hdf_ids)
    except Exception, e:
        print(" --> Could not train EMG decoder!")
        raise

def train_emg_decoder(config_id,train_hdf_ids,test_hdf_ids):
    """ Perform training of the EMG decoder

        Args:
            config_id (string): The id of the configuration file (usually the patient id)
            train_hdf_ids (list of strings): The list of task entries, whose data is used for training the decoder
            [optional] test_hdf_ids (list of int): The list of task entries, whose data is used for testing the decoder

    """

    # Load configuration from yml-file
    try:
        fname = '/home/tecnalia/code/ismore/tubingen/decoder_configs/'+config_id+'_emg.yml'
        with open(fname) as f:
            cfg = yaml.load(f)

            mirrored = cfg['mirrored']
            relevant_dofs = cfg['relevant_dofs']
            subset_muscles_names = cfg['subset_muscles_names']
            feature_names = cfg['feature_names']
            feature_fn_kwargs = cfg['feature_fn_kwargs']
            filt_training_data = cfg['filt_training_data']
            db_name = cfg['db_name']
            plant_type = cfg['plant_type']
            win_len = cfg['win_len']
            K = cfg['K']
            fs = cfg['fs']
            decode_all_dofs_with_all_muscles = cfg['decode_all_dofs_with_all_muscles']

            print('Mirrored', mirrored)
            print('relevant_dofs', relevant_dofs)
            print('subset_muscles_names', subset_muscles_names)
            print('feature_names', feature_names)
            print('feature_fn_kwargs', feature_fn_kwargs)
            print('filt_training_data', filt_training_data)
            print('db_name', db_name)
            print('plant_type', plant_type)
            print('win_len', win_len)
            print('K', K)
            print('fs', fs)
            print('decode_all_dofs_with_all_muscles', decode_all_dofs_with_all_muscles)

    except IOError as ioerr:
        print("Could not open {}".format(fname))
        print(ioerr)


    ################################################################
    # Variables that will probably not change during the whole experiment
    ################################################################

    # Compute variability of the data of each channel from the compliant run.
    # Compute mean of this variability for all the channels.
    # Use this value for normalization of each window of EMG activity of each channel.
    # That keeps the ratio of values between channels equal.

    fixed_var_scalar = True 

    # These variables are only needed when using the HD arrays
    
    # This is for classifier testing when using the high density array
    test_relevant_channels_hdf_ids = []

    # Limit for the total number of bipolarization combinations (because if there are too many channels
    #        the performance of the platform drops)
    nchannels_2select = 60 
    # Minimum number of channels from the HD-EMG that will be kept for the iterations used to find the relevant channels 
    min_HD_nchans = 10

    ################################################################

    train_hdf_names = []
    for id in train_hdf_ids:
        te = dbfn.TaskEntry(id, dbname= db_name)
        train_hdf_names.append(te.hdf_filename)
        te.close_hdf()
    test_relevant_channels_hdf_names = []
    for id in test_relevant_channels_hdf_ids:
        te = dbfn.TaskEntry(id, dbname= db_name)
        test_relevant_channels_hdf_names.append(te.hdf_filename)
        te.close_hdf()

    test_hdf_names = []
    for id in test_hdf_ids:
        te = dbfn.TaskEntry(id, dbname=db_name)
        test_hdf_names.append(te.hdf_filename)
        te.close_hdf()

    ssm = ismore_bmi_lib.SSM_CLS_DICT[plant_type]()
    states_to_decode = [s.name for s in ssm.states if s.order == 1]

    for state in states_to_decode:
        subset_muscles_names[state] = [i + '_filt' for i in subset_muscles_names[state]]

    # Select the channel list that is used to train
    channels_2train = brainamp_channel_lists.emg14_bip_filt

    # If we would like to use a subset of muscles for any of the DoFs, then define it here. Otherwise set it to an empty dict().
    subset_muscles = dict()

    if not(decode_all_dofs_with_all_muscles):
        for state in states_to_decode:
            subset_muscles[state] = [np.int(channels_2train.index(subset_muscles_names[state][i])) for i in np.arange(len(subset_muscles_names[state]))]

    if mirrored == True:
        # If the values of the rehand are positive when we recorded with the left hand (sess03), because we took all of them as possitive
        states_to_flip = ['aa_vx', 'aa_vpsi']
    else:
        states_to_flip = []

    extractor_kwargs = {
        'feature_names':     feature_names,
        'feature_fn_kwargs': feature_fn_kwargs,
        'win_len':           win_len,
        'fs':                fs,
        'K':                 K,
    }

    relevant_dofs = [str(dof) for dof in relevant_dofs]

    if channels_2train == brainamp_channel_lists.emg_48hd_6mono_filt:
        channel_finder = find_relevant_channels.FindRelevantChannels(train_hdf_names,test_relevant_channels_hdf_names,channels_2train, plant_type, filt_training_data, extractor_kwargs, nchannels_2select, relevant_dofs, min_HD_nchans, mirrored)
        opt_channels, opt_channels_dict, opt_channels_2train_dict = channel_finder()
        filt_training_data = True

    emg_feature_name_list = []

    try: 
        for feature in feature_names:
            emg_feature_name_list = emg_feature_name_list + [opt_channel + '_' + feature for opt_channel in opt_channels]
    except:
        for feature in feature_names:
            emg_feature_name_list = emg_feature_name_list + [opt_channel + '_' + feature for opt_channel in channels_2train]

    extractor_cls = EMGMultiFeatureExtractor
    if channels_2train == brainamp_channel_lists.emg_48hd_6mono_filt:
        extractor_kwargs = {
            'emg_channels':      opt_channels,
            'feature_names':     feature_names,
            'feature_fn_kwargs': feature_fn_kwargs,
            'win_len':           win_len,
            'fs':                fs,
            'channels_str_2discard': opt_channels_dict["channels_str_2discard"],
            'channels_str_2keep':    opt_channels_dict["channels_str_2keep"],
            'channels_diag1_1':  opt_channels_dict["channels_diag1_1"],
            'channels_diag1_2':  opt_channels_dict["channels_diag1_2"],
            'channels_diag2_1':  opt_channels_dict["channels_diag2_1"],
            'channels_diag2_2':  opt_channels_dict["channels_diag2_2"],
            'emg_feature_name_list': emg_feature_name_list,
            'subset_muscles': subset_muscles,
        }
    else:
        opt_channels_2train_dict = dict()
        extractor_kwargs = {
            'emg_channels':      channels_2train,
            'feature_names':     feature_names,
            'feature_fn_kwargs': feature_fn_kwargs,
            'win_len':           win_len,
            'fs':                fs,
            'emg_feature_name_list': emg_feature_name_list,
            'subset_muscles': subset_muscles,
            'fixed_var_scalar': fixed_var_scalar,
        }

    decoder = emg_decoding.LinearEMGDecoder(channels_2train, plant_type, fs, win_len, filt_training_data, extractor_cls, extractor_kwargs, opt_channels_2train_dict)

    decoder.train_ridge(K, train_hdf_names, test_hdf_names, states_to_flip)

    decoder.training_ids = train_hdf_ids

    train_ids_str = str(min(train_hdf_ids)) + '_' + str(max(train_hdf_ids))

    subject_name  = models.TaskEntry.objects.using(db_name).get(id=train_hdf_ids[0]).subject.name

    for key, value in SubjectFeedbackGroup.iteritems():
        if key == subject_name:
            Group = value

    try: 
        decoder.group = Group
    except UnboundLocalError as e:
        print(e)
        print(" --> WARNING: Could not find subject {} in randomization table --> Subject will be asigned to hybrid group.")
        decoder.group = 1
        
    decoder_name = 'emg_decoder_%s_%s' % (subject_name,train_ids_str)
    pkl_name = decoder_name + '.pkl'
    storage_dir = '/storage/decoders'

    decoder.path = os.path.join(storage_dir, pkl_name)
    decoder.decoder_name = decoder_name

    ## Store a record of the data file in the database
    if not os.path.exists(storage_dir):
        os.popen('mkdir -p %s' % storage_dir)

    pickle.dump(decoder, open(os.path.join(storage_dir, pkl_name), 'wb'))

    # Create a new database record for the decoder object if it doesn't already exist
    dfs = models.Decoder.objects.filter(name=decoder_name)
    if len(dfs) == 0:
        df = models.Decoder()
        df.path = pkl_name
        df.name = decoder_name
        df.entry = models.TaskEntry.objects.using(db_name).get(id=min(train_hdf_ids))
        df.save()
    elif len(dfs) == 1:
        pass # no new data base record needed
    elif len(dfs) > 1:
        print "More than one decoder with the same name! fix manually!"

# Call main function if script is called from the command line
if __name__ == "__main__":
    main()