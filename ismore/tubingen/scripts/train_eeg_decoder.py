import tables
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
import sys
import pickle
import copy
import yaml


from ismore import brainamp_channel_lists
from ismore.tubingen.noninvasive_tubingen import eeg_decoding

from ismore.tubingen.noninvasive_tubingen.eeg_feature_extraction import EEGMultiFeatureExtractor
from ismore.common_state_lists import *
from utils.constants import *
from db import dbfunctions as dbfn
from db.tracker import models


def main():
    ''' Function that will execute the rest of this script when called from the command line.'''
    
    try:
        config_id = sys.argv[1]
        train_hdf_ids = sys.argv[2].split(',')
        calibration_data = sys.argv[3]

        test_hdf_ids = []
        if len(sys.argv) > 4:
            test_hdf_ids = sys.argv[4]

        print("")
        print("Task entries used for decoder training of patient "+config_id+": "+str(train_hdf_ids))
        print("Calibrating for session type: "+calibration_data)
        print("")
        train_eeg_decoder(config_id=config_id, train_hdf_ids=train_hdf_ids, calibration_data=calibration_data, test_hdf_ids=test_hdf_ids)
    except IndexError as e:
        print("Could not train EEG decoder: Not enough parameters supplied.")
        print("Use as python train_eeg_decoder.py CONFIG_ID TRAIN_HDF_IDS CALIBRATION_DATA TEST_HDF_IDS.")
    except Exception, e:
        print(" --> Could not train EEG decoder!")
        raise

def train_eeg_decoder(config_id,train_hdf_ids,calibration_data,test_hdf_ids):
    """ Perform training of the EEG decoder

        Args:
            config_id (string): The id of the configuration file (usually the patient id)
            train_hdf_ids (list of strings): The list of task entries, whose data is used for training the decoder
            calibration_data (string): The task type of the data files (e.g. 'compliant')
            [optional] test_hdf_ids (list of int): The list of task entries, whose data is used for testing the decoder

    """

    # Load configuration from yml-file
    try:
        fname = '/home/tecnalia/code/ismore/tubingen/decoder_configs/'+config_id+'_eeg.yml'
        with open(fname) as f:
            cfg = yaml.load(f)

            channels_2train = cfg['channels_2train']
            trial_hand_side = cfg['trial_hand_side']
            freq_bands = cfg['freq_bands']
            bipolar_EOG = cfg['bipolar_EOG']
            artifact_rejection = cfg['artifact_rejection']
            NEIGHBOUR_CHANNELS_DICT = cfg['NEIGHBOUR_CHANNELS_DICT']
            filt_training_data = cfg['filt_training_data']
            feature_names = cfg['feature_names']
            db_name = cfg['db_name']
            feature_fn_kwargs = cfg['feature_fn_kwargs']
            plant_type = cfg['plant_type']
            win_len = cfg['win_len']
            buffer_len = cfg['buffer_len']
            fs = cfg['fs']

    except IOError as ioerr:
        print("Could not open {}".format(fname))
        print(ioerr)

    # Define only the channels that have to be taken from the source to compute the laplacian
    #???
    eeg_channels = copy.copy(channels_2train)

    print(eeg_channels)

    if filt_training_data == False:
        channels_numbers = [int(i[:i.find('_')]) for i in eeg_channels]
        for k, chan_neighbour in enumerate(NEIGHBOUR_CHANNELS_DICT):
            number = int(chan_neighbour[:chan_neighbour.find('_')])
            if number not in channels_numbers:
                channels_numbers = np.hstack([channels_numbers, number])
            for kk, chan_neighbour2 in enumerate(NEIGHBOUR_CHANNELS_DICT[chan_neighbour]):
                number2 = int(chan_neighbour2[:chan_neighbour2.find('_')])
                if number2 not in channels_numbers:
                    channels_numbers = np.hstack([channels_numbers,number2])
        channels_numbers_sorted = sorted(channels_numbers)
        eeg_channels = [str(i) + '_filt' for i in channels_numbers_sorted]
    else:
        channels_numbers = [int(i) for i in eeg_channels]
        for k, chan_neighbour in enumerate(NEIGHBOUR_CHANNELS_DICT):
            number = int(chan_neighbour)
            if number not in channels_numbers:
                channels_numbers = np.hstack([channels_numbers, number])
            for kk, chan_neighbour2 in enumerate(NEIGHBOUR_CHANNELS_DICT[chan_neighbour]):
                number2 = int(chan_neighbour2)
                if number2 not in channels_numbers:
                    channels_numbers = np.hstack([channels_numbers,number2])
        channels_numbers_sorted = sorted(channels_numbers)
        eeg_channels = [str(i) + '_filt' for i in channels_numbers_sorted]


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

    if channels_2train != sorted(NEIGHBOUR_CHANNELS_DICT.keys()):
        raise ValueError("ERROR: selected channels to train and neighbour channels dict entries do NOT match. Correct before training the decoder.")

    if artifact_rejection == True:
        hdf = tables.openFile(train_hdf_names[0])
        recorded_channels = hdf.root.brainamp.colnames
        if 'chanEOGV_filt' in recorded_channels:
            #eeg_channels always with filt
            eog_channels = brainamp_channel_lists.eog2_filt
            # Whether the EOG of the calibration and testing data was recorded in monopolar or bipolar mode
            bipolar_EOG = True
            neog_channs = 2
        elif 'chanEOGV' in recorded_channels:
            #eeg_channels always with filt
            eog_channels = brainamp_channel_lists.eog2_filt
            bipolar_EOG = True
            neog_channs = 2
        elif 'chanEOG1_filt' in recorded_channels:
            #eeg_channels always with filt
            eog_channels = brainamp_channel_lists.eog4_filt
            bipolar_EOG = False
            neog_channs = 4
        elif 'chanEOG1' in recorded_channels:
            #eeg_channels always with filt
            eog_channels = brainamp_channel_lists.eog4_filt
            bipolar_EOG = False
            neog_channs = 4

        hdf.close()

        eeg_channels += eog_channels

    else: 
    # just to avoid errors when EOG was not recorded
        eog_channels = list()
        bipolar_EOG = True
        neog_channs = 2

    ########################################################

    extractor_cls = EEGMultiFeatureExtractor
    extractor_kwargs = {
        'eeg_channels':      eeg_channels,
        'channels_2train':   channels_2train,
        'neighbour_channels': NEIGHBOUR_CHANNELS_DICT,
        'feature_names':     feature_names,
        'feature_fn_kwargs': feature_fn_kwargs,
        'win_len':           win_len,
        'fs':                fs,
        'artifact_rejection': artifact_rejection,
        'calibration_data': calibration_data,
        'bipolar_EOG': bipolar_EOG,
    }

    decoder = eeg_decoding.LinearEEGDecoder(channels_2train, plant_type, fs, win_len, buffer_len, filt_training_data, extractor_cls, extractor_kwargs, trial_hand_side)
    decoder.train_LDA(train_hdf_names, test_hdf_names)

    decoder.training_ids = train_hdf_ids

    train_ids_str = str(min(train_hdf_ids)) + '_' + str(max(train_hdf_ids)) + '_0'

    subject_name  = models.TaskEntry.objects.using(db_name).get(id=train_hdf_ids[0]).subject.name
    decoder_name = 'eeg_decoder_%s_%s' % (subject_name,train_ids_str)
    pkl_name = decoder_name + '.pkl'
    decoder.decoder_name = decoder_name

    # Store a record of the data file in the database
    storage_dir = '/storage/decoders'

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