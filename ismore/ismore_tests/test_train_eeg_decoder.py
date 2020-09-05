import tables
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
import pickle
import copy


from ismore import brainamp_channel_lists
from ismore.ismore_tests import eeg_decoding
#reload(eeg_decoding)
#from ismore.emg_decoding import LinearEMGDecoder
from ismore.ismore_tests.eeg_feature_extraction import EEGMultiFeatureExtractor
from ismore.common_state_lists import *
from utils.constants import *
from db import dbfunctions as dbfn
from db.tracker import models


################################################################
# The variables below need to be set before running this script!
################################################################


# Set 'okay_to_overwrite_decoder_file' to True to allow overwriting an existing 
#   file
okay_to_overwrite_decoder_file = True

##training dataset
#train_hdf_ids = [4056,3964] #EMG DATA
#train_hdf_ids = [4088] #EEG DATA
#train_hdf_ids = [718] #EEG DATA
train_hdf_ids = [4296,4298,4300] #AI EEG DATA
#train_hdf_ids = [4329,4330,4331] #EL EEG DATA
#train_hdf_ids = [1243,1244] #TEST DATA FOR PREEYA
#train_hdf_ids = [863,864]
#testing dataset
#test_hdf_ids = [3962] #EMG DATA
#test_hdf_ids = [4301] #AI EEG DATA
#test_hdf_ids = [4333]
test_hdf_ids = [4301]
#test_hdf_ids = []

trial_hand_side = 'left'
# Set 'channels' to be a list of channel names
channels_2train = ['13_filt','14_filt','18_filt','19_filt'] #determine here the electrodes that will be used for the training WITHOUT the channels that will be used for the Laplacian filter!!!
#channels = defined automatically below: these are the electrodes that will be used for the online decoding (always the _filt ones) INCLUDING the channels that will be used for the Laplacian filter!!!
brainamp_channels = brainamp_channel_lists.eeg32_raw_filt # channels to get record from the brainamptsource into the hdf file (these must always be the raw + filt ones. The data will be filtered as it arrives from the brainamp and stored both the raw and filtered data in the hdf file)

# channels_2train = ['Biceps_filt','Triceps_filt'] #determine here the electrodes that will be used for the training. 
# channels = ['Biceps_filt','Triceps_filt','Extra1_filt','Extra2_filt', 'Extra3_filt'] #determine here the electrodes that will be used for the online decoding (always the _filt ones) INCLUDING the channels that will be used for the Laplacian filter!!!
# brainamp_channels = brainamp_channel_lists.emg14_raw_filt # channels to get record from the brainamptsource into the hdf file (these must always be the raw + filt ones. The data will be filtered as it arrives from the brainamp and stored both the raw and filtered data in the hdf file)
# channels = brainamp_channel_lists.emg14_filt
freq_bands = dict()
freq_bands['13_filt'] = [[7,16]] #list with the freq bands of interest
freq_bands['14_filt'] = [[7,16]]#[[2,7],[9,16]]
freq_bands['18_filt'] = [[12,16]]
freq_bands['19_filt'] = [[12,16]]

if '_filt' not in channels_2train[0]:
    filt_training_data = True
    NEIGHBOUR_CHANNELS_DICT = { #define the neighbour channels for each channel (for the Laplacian filter)
    '1':  ['2','3','4'],
    '2':  ['5','6'],
    '3':  ['2','4','5'],
    }

else:
    filt_training_data = False
    # NEIGHBOUR_CHANNELS_DICT = { #define the neighbour channels for each channel (for the Laplacian filter)
    # '1_filt':  ['2_filt','3_filt','4_filt'],
    # '2_filt':  ['5_filt','6_filt'],
    # '3_filt':  ['2_filt','4_filt','5_filt'],
    # }
    # NEIGHBOUR_CHANNELS_DICT = { #define the neighbour channels for each channel (for the Laplacian filter)
    # '1_filt':  brainamp_channel_lists.eeg32_filt,
    # '2_filt':  brainamp_channel_lists.eeg32_filt,
    # '3_filt':  brainamp_channel_lists.eeg32_filt,
    # '4_filt':  brainamp_channel_lists.eeg32_filt,
    # '5_filt':  brainamp_channel_lists.eeg32_filt,
    # '6_filt':  brainamp_channel_lists.eeg32_filt,
    # '7_filt':  brainamp_channel_lists.eeg32_filt,
    # '8_filt':  ['3_filt', '4_filt','12_filt', '13_filt'],
    # '9_filt':  ['4_filt', '5_filt','13_filt', '14_filt'],
    # '10_filt':  ['5_filt', '6_filt','14_filt', '15_filt'],
    # '11_filt':  ['6_filt', '7_filt','15_filt', '16_filt'],
    # '12_filt':  brainamp_channel_lists.eeg32_filt,
    # '13_filt':  ['8_filt', '9_filt','18_filt', '19_filt'],
    # '14_filt':  ['9_filt', '10_filt','19_filt', '20_filt'],
    # '15_filt':  ['10_filt', '11_filt','20_filt', '21_filt'],
    # '16_filt':  brainamp_channel_lists.eeg32_filt,
    # '17_filt':  brainamp_channel_lists.eeg32_filt,
    # '18_filt':  ['12_filt', '13_filt','23_filt', '24_filt'],
    # '19_filt':  ['13_filt', '14_filt','24_filt', '25_filt'],
    # '20_filt':  ['14_filt', '15_filt','25_filt', '26_filt'],
    # '21_filt':  ['15_filt', '16_filt','26_filt', '27_filt'],
    # '22_filt':  brainamp_channel_lists.eeg32_filt,
    # '23_filt':  brainamp_channel_lists.eeg32_filt,
    # '24_filt':  ['18_filt', '19_filt','28_filt'],
    # '25_filt':  ['19_filt', '20_filt','28_filt', '30_filt'],
    # '26_filt':  ['20_filt', '21_filt','30_filt'],
    # '27_filt':  brainamp_channel_lists.eeg32_filt,
    # '28_filt':  brainamp_channel_lists.eeg32_filt,
    # '29_filt':  brainamp_channel_lists.eeg32_filt,
    # '30_filt':  brainamp_channel_lists.eeg32_filt,
    # '31_filt':  brainamp_channel_lists.eeg32_filt,
    # '32_filt':  brainamp_channel_lists.eeg32_filt,
    # }
    NEIGHBOUR_CHANNELS_DICT = { #define the neighbour channels for each channel (for the Laplacian filter)
    '13_filt':  ['8_filt', '9_filt','18_filt', '19_filt'],
    '14_filt':  ['9_filt', '10_filt','19_filt', '20_filt'],
    '18_filt':  ['12_filt', '13_filt','23_filt', '24_filt'],
    '19_filt':  ['13_filt', '14_filt','24_filt', '25_filt'],
    }
    # NEIGHBOUR_CHANNELS_DICT = { #define the neighbour channels for each channel (for the Laplacian filter)
    # 'Biceps_filt':  brainamp_channel_lists.emg14_filt,
    # 'Triceps_filt':  ['Extra2_filt', 'Extra3_filt'],
    # }

# Piece of code to define only the channels that have to be taken from the source to compute the laplacian
channels = copy.copy(channels_2train)
channels_numbers = [int(i[:i.find('_')]) for i in channels]
for k, chan_neighbour in enumerate(NEIGHBOUR_CHANNELS_DICT):
    number = int(chan_neighbour[:chan_neighbour.find('_')])
    if number not in channels_numbers:
        channels_numbers = np.hstack([channels_numbers, number])
    for kk, chan_neighbour2 in enumerate(NEIGHBOUR_CHANNELS_DICT[chan_neighbour]):
        number2 = int(chan_neighbour2[:chan_neighbour2.find('_')])
        if number2 not in channels_numbers:
            channels_numbers = np.hstack([channels_numbers,number2])
channels_numbers_sorted = sorted(channels_numbers)
channels = [str(i) + '_filt' for i in channels_numbers_sorted]


# Set 'feature_names' to be a list containing the names of features to use
#   (see emg_feature_extraction.py for options)
feature_names = ['AR'] # choose here the feature names that will be used to train the decoder

# Set 'train_hdf_names' to be a list of names of .hdf files to train from
# train_hdf_names = [
#     #'/home/lab/Desktop/test20150424_04.hdf',
#     #'/home/lab/Desktop/AS_F1S001R02.hdf',
#     #'/home/lab/Desktop/AS_F2S001R02.hdf'
#    # '/home/tecnalia/hybries/resources/EMGdecoder_code_testing/AS_F1S001R02.hdf'
#    # '/home/tecnalia/hybries/resources/EMGdecoder_code_testing/AS_F2S001R02.hdf'
#     #'/home/tecnalia/hybries/resources/EMGdecoder_code_testing/fs20150424_06.hdf', #B1_R1
#     #'/home/tecnalia/hybries/resources/EMGdecoder_code_testing/fs20150424_07.hdf', #B1_R2
#     #'/home/tecnalia/hybries/resources/EMGdecoder_code_testing/fs20150424_08.hdf', #B1_R3
#     #'/home/tecnalia/hybries/resources/EMGdecoder_code_testing/fs20150424_09.hdf', #B1_R4
#     #'/home/tecnalia/hybries/resources/EMGdecoder_code_testing/fs20150424_11.hdf' #B1_R5
# ]

# Set 'test_hdf_names' to be a list of names of .hdf files to test on (offline)
# test_hdf_names = [
#     '/home/tecnalia/hybries/resources/EMGdecoder_code_testing/fs20150424_11.hdf' #B1_R5   
# ]
# )
db_name = "default"
#db_name = "tubingen"

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



# Set 'feature_fn_kwargs' to be a dictionary where:
#   key: name of a feature (e.g., 'ZC')
#   value: a dictionary of the keyword arguments to be passed into the feature
#          function (e.g., extract_ZC) corresponding to this feature
#   (see emg_feature_extraction.py for how the feature function definitions)
feature_fn_kwargs = {
    'AR': {'freq_bands': freq_bands},  
}

# Set 'plant_type' (type of plant for which to create a decoder)
#   choices: 'ArmAssist', 'ReHand', or 'IsMore'
plant_type = 'IsMore'

# Set 'win_len'
win_len = 0.5  # secs
buffer_len = 120 #secs

# Set 'fs'
fs = 1000  # Hz 

########################################################


# if os.path.isfile(pkl_name) and not okay_to_overwrite_decoder_file:
#     raise Exception('A decoder file with that name already exists!') 

extractor_cls = EEGMultiFeatureExtractor
extractor_kwargs = {
    'channels':          channels,
    'channels_2train':   channels_2train,
    'brainamp_channels': brainamp_channels,
    'neighbour_channels': NEIGHBOUR_CHANNELS_DICT,
    'feature_names':     feature_names,
    'feature_fn_kwargs': feature_fn_kwargs,
    'win_len':           win_len,
    'fs':                fs,
}

decoder = eeg_decoding.LinearEEGDecoder(channels_2train, plant_type, fs, win_len, buffer_len, filt_training_data, extractor_cls, extractor_kwargs, trial_hand_side)
#decoder = LinearEMGDecoder(channels, plant_type, fs, win_len, extractor_cls, extractor_kwargs)
decoder.train_LDA(train_hdf_names, test_hdf_names)

decoder.training_ids = train_hdf_ids

train_ids_str = str(min(train_hdf_ids)) + '_' + str(max(train_hdf_ids)) + '_0'
#import pdb; pdb.set_trace()
subject_name  = models.TaskEntry.objects.using(db_name).get(id=train_hdf_ids[0]).subject.name
decoder_name = 'eeg_decoder_%s_%s' % (subject_name,train_ids_str)
pkl_name = decoder_name + '.pkl'
decoder.decoder_name = decoder_name
## Store a record of the data file in the database


storage_dir = '/storage/decoders'

if not os.path.exists(storage_dir):
    os.popen('mkdir -p %s' % storage_dir)

pickle.dump(decoder, open(os.path.join(storage_dir, pkl_name), 'wb'))

#db_name = 'default'
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
