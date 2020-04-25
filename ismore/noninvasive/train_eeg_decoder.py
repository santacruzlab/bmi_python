import tables
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
import pickle
import copy


from ismore import brainamp_channel_lists
from ismore.noninvasive import eeg_decoding
#reload(eeg_decoding)
#from ismore.emg_decoding import LinearEMGDecoder
from ismore.noninvasive.eeg_feature_extraction import EEGMultiFeatureExtractor
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
#train_hdf_ids = [4296,4298,4300] #AI EEG DATA
#train_hdf_ids = [5867, 5868, 5869, 5870, 5871]# AI S1- EEG BCI

train_hdf_ids = [5903,5904,5905,5906,5907]# EL -S1- EEG BCI
train_hdf_ids = [5922,5923,5925,5926,5927]# AS -S1- EEG BCI
train_hdf_ids = [5978, 5979, 5980, 5981, 5982]# TF -S1- EEG BCI
train_hdf_ids = [5996, 5997, 5998, 5999, 6000]#NI -S1- EEG BCI
train_hdf_ids = [6018, 6019, 6020, 6021, 6022]#FS -S1- EEG BCI
#train_hdf_ids = [4329,4330,4331] #EL EEG DATA
#train_hdf_ids = [1243,1244] #TEST DATA FOR PREEYA


train_hdf_ids = [5996, 5997] #test signal nerea


train_hdf_ids = [4296,4298,4300]
train_hdf_ids = [5996, 5997]
train_hdf_ids = [5996]
train_hdf_ids = [2600]
train_hdf_ids = [2635]
train_hdf_ids = [5996, 5997]
train_hdf_ids = [7995, 7996, 7997]
train_hdf_ids = [8229, 8230, 8231]
train_hdf_ids = [8243,8244,8245] # DK test patient hybrid-BCI
train_hdf_ids = [4283,4288]
train_hdf_ids = [5996] #test signal nerea
train_hdf_ids = [8210,8211,8212]
train_hdf_ids = [8678,8679,8683,8684,8685,8686]
#train_hdf_ids = [4329,4330,4331] #EL EEG DATA
#train_hdf_ids = [1243,1244] #TEST DATA FOR PREEYA

train_hdf_ids = [9066,9068]
train_hdf_ids = [4486,4487,4491,4495]

#train_hdf_ids = [9066,9068]

# train_hdf_ids = [4296,4298,4300]
# train_hdf_ids = [4296]
#train_hdf_ids = [5382]

#train_hdf_ids = [4296,4298,4300]
#train_hdf_ids = [5652,5653,5654,5655]

#testing dataset
#test_hdf_ids = [3962] #EMG DATA
#test_hdf_ids = [4301] #AI EEG DATA
#test_hdf_ids = [4333]


# Compliant left Wala 2017.06.26
train_hdf_ids = [9617,9620,9623,9625,9627]
# JO right active session
train_hdf_ids = [9520,9523]


# Test with Doris 23/11/2017
train_hdf_ids = [10170,10173,10174,10176]

#HUD1 - compliant session- paretic arm
train_hdf_ids = [4769,4770,4773,4774,4777,4778] 
train_hdf_ids = [14312,14313] 
train_hdf_ids = [14281,14282] 

test_hdf_ids = []

trial_hand_side = 'right'
# Set 'channels' to be a list of channel names
#channels_2train = ['13_filt','14_filt','18_filt','19_filt'] #determine here the electrodes that will be used for the training WITHOUT the channels that will be used for the Laplacian filter!!!
#channels_2train = ['9_filt'] 
#channels_2train = ['15'] # Write always the channels_2train in ascending order


channels_2train = ['13'] # Write always the channels_2train in ascending order
#channels_2train = ['6','7'] # Write always the channels_2train in ascending order

#channels = defined automatically below: these are the electrodes that will be used for the online decoding (always the _filt ones) INCLUDING the channels that will be used for the Laplacian filter!!!
CAR_channels = brainamp_channel_lists.eeg32_filt
#CAR_channels = brainamp_channel_lists.eeg32
# channels_2train = ['Biceps_filt','Triceps_filt'] #determine here the electrodes that will be used for the training. 
freq_bands = dict()
#freq_bands['10_filt'] = [[20,27]] #list with the freq bands of interest
#freq_bands['14_filt'] = [[7,25]]#[[2,7],[9,16]]
#freq_bands['18_filt'] = [[10,18]]
#freq_bands['9_filt'] = [[8,12]]

#freq_bands['15'] = [[11,12],[16,18]]
#freq_bands['7'] = [[23,27]]

# freq_bands['6'] = [[18,23]]
# freq_bands['6'] = [[25,28]]

#freq_bands['7'] = [[12,14]]
freq_bands['13'] = [[8,12]]

#freq_bands['3'] = [[9,13]]

#freq_bands['11_filt'] = [[9,10]]

#freq_bands['5_filt'] = [[8,9]]
# filt_training_data = True


# The variable 'calibration data' defines the type of data used to calibrate (i.e. train) the EEG decoder. 
# It can be 'screening' when data from a Screening session in which subject (tried to) open and close their hands is used
# or 'compliant' when data in which the subject wore the exoskeleton in his paretic arm and performed a series of tasks with full assistance (i.e. compliant movement)
#calibration_data = 'compliant'

calibration_data = 'screening'
#calibration_data = 'active'

#calibration_data = 'screening'
#calibration_data = 'active'
calibration_data = 'compliant_testing'
calibration_data = 'compliant'

bipolar_EOG = False

# Artifact rejection to remove EOG, movement and EMG artifacts. If artifact_rejection variable = True then the rejection is done. Otherwise no rejection is applied.
artifact_rejection = False

if '_filt' not in channels_2train[0]:
    filt_training_data = True
    NEIGHBOUR_CHANNELS_DICT = { #define the neighbour channels for each channel (for the Laplacian filter)
    # '1':  ['2','3','4'],
    # '2':  ['5','6'],
    #'3':  ['2','4','5'],
    # '6':  ['1', '5','7', '12'],

    #'3':  ['8', '10','21', '22'],#short Laplacian
    #'10':  ['4', '9','11', '16'],
    #'6':  ['1', '5','7', '12'],
    #'7':  ['2', '6','8', '13'],
    '13':  ['7', '12','14', '19'],
    #'16':  ['10', '15','17', '22'],
    #'23':  ['16', '18','26'],

    }


else:
    filt_training_data = False
    # Old 32-channels configuration
    #NEIGHBOUR_CHANNELS_DICT = { #define the neighbour channels for each channel (for the Laplacian filter)
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
    # '13_filt':  ['7_filt', '12_filt','14_filt', '19_filt'],
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
    #'24_filt':  ['18_filt', '19_filt','28_filt'],
    # '25_filt':  ['19_filt', '20_filt','28_filt', '30_filt'],
    #'26_filt':  ['20_filt', '21_filt','30_filt'],
    # '27_filt':  brainamp_channel_lists.eeg32_filt,
    # '28_filt':  brainamp_channel_lists.eeg32_filt,
    # '29_filt':  brainamp_channel_lists.eeg32_filt,
    # '30_filt':  brainamp_channel_lists.eeg32_filt,
    # '31_filt':  brainamp_channel_lists.eeg32_filt,
    # '32_filt':  brainamp_channel_lists.eeg32_filt,
    #}
    
    # New 32 channels montage
    # NEIGHBOUR_CHANNELS_DICT = { #define the neighbour channels for each channel (for the Laplacian filter)
    # '1_filt':  CAR_channels,
    # '2_filt':  ['6_filt', '8_filt','20_filt', '21_filt'],
    # '3_filt':  ['8_filt', '10_filt','21_filt', '22_filt'],
    # '4_filt':  CAR_channels,
    # '5_filt':  CAR_channels,
    # '6_filt':  ['1_filt', '5_filt','7_filt', '12_filt'],
    # '7_filt':  ['2_filt', '6_filt','8_filt', '13_filt'],
    # '8_filt':  ['2_filt', '3_filt','13_filt', '15_filt'],
    # '9_filt':  ['3_filt', '8_filt','10_filt', '15_filt'],
    # '10_filt':  ['4_filt', '9_filt','11_filt', '16_filt'],
    # '11_filt':  CAR_channels,
    # '12_filt':  ['6_filt', '13_filt','27_filt'],
    # '13_filt':  ['6_filt', '8_filt','27_filt', '28_filt'],
    # '14_filt':  ['8_filt', '13_filt','15_filt', '28_filt'],
    # '15_filt':  ['8_filt', '10_filt','28_filt', '29_filt'],
    # '16_filt':  ['10_filt','15_filt', '29_filt'],
    # '17_filt':  CAR_channels,
    # '18_filt':  CAR_channels,
    # '19_filt':  CAR_channels,
    # '20_filt':  CAR_channels,
    # '21_filt':  CAR_channels,
    # '22_filt':  CAR_channels,
    # '23_filt':  CAR_channels,
    # '24_filt':  CAR_channels,
    # '25_filt':  CAR_channels,
    # '26_filt':  CAR_channels,
    # '27_filt':  CAR_channels,
    # '28_filt':  CAR_channels,
    # '29_filt':  CAR_channels,
    # '30_filt':  CAR_channels,
    # '31_filt':  CAR_channels,
    # '32_filt':  CAR_channels,
    # }

    #New IsMore montage with higher density over the motor cortex
    NEIGHBOUR_CHANNELS_DICT = {
    # '1_filt':  ['2_filt', '6_filt','20_filt'], #short laplacian with 3 neighbours
    # '2_filt':  ['6_filt', '8_filt','20_filt', '21_filt'],#short Laplacian
    # '3_filt':  ['8_filt', '10_filt','21_filt', '22_filt'],#short Laplacian
    # '4_filt':  ['3_filt', '10_filt','22_filt'], #short laplacian with 3 neighbours
    # '5_filt':  CAR_channels,#CAR ????
    # '6_filt':  ['1_filt', '5_filt','7_filt', '12_filt'],
    # '7_filt':  ['2_filt', '6_filt','8_filt', '13_filt'],
    # '8_filt':  ['2_filt', '3_filt','13_filt', '15_filt'],
    # '9_filt':  ['3_filt', '8_filt','10_filt', '15_filt'],
    # '10_filt':  ['4_filt', '9_filt','11_filt', '16_filt'],
    # '11_filt':  CAR_channels,#CAR????
    # '12_filt':  ['6_filt', '13_filt','24_filt','27_filt'],
    #'13_filt':  ['7_filt', '12_filt','14_filt', '19_filt'],
    '13_filt':  ['7', '12','14', '19'],
    # '14_filt':  ['8_filt', '13_filt','15_filt', '28_filt'],
    # '15_filt':  ['8_filt', '10_filt','28_filt', '29_filt'],
    # '16_filt':  ['10_filt','15_filt','25_filt', '29_filt'],
    # '17_filt':  CAR_channels,#CAR
    # '18_filt':  CAR_channels,#CAR
    # '19_filt':  CAR_channels,#CAR
    # '20_filt':  [ '6_filt','17_filt', '19_filt', '21_filt'], #long laplacian 
    # '21_filt':  ['8_filt', '20_filt', '22_filt'], #long laplacian with 3 neighbours
    # '22_filt':  [ '10_filt','18_filt', '21_filt', '23_filt'], #long laplacian 
    # '23_filt':  CAR_channels, 
    # '24_filt':  ['6_filt','26_filt','27_filt'], #short laplacian with 3 neighbours
    # '25_filt':  ['10_filt','29_filt', '30_filt'], #short laplacian with 3 neighbours
    # '26_filt':  CAR_channels,
    # '27_filt': ['6_filt', '26_filt', '28_filt', '31_filt'], #long laplacian
    # '28_filt':  ['8_filt', '27_filt', '29_filt'], #long laplacian with 3 neighbours
    # '29_filt':  ['10_filt', '28_filt', '30_filt', '32_filt'], #long laplacian
    # '30_filt':  CAR_channels, #CAR
    # '31_filt':  CAR_channels, #CAR
    # '32_filt':  CAR_channels, #CAR
    }

    #NEIGHBOUR_CHANNELS_DICT = {
    # '1_filt':  ['2_filt', '6_filt','20_filt'], #short laplacian with 3 neighbours
    # '2_filt':  ['6_filt', '8_filt','20_filt', '21_filt'],#short Laplacian
    # '3_filt':  ['8_filt', '10_filt','21_filt', '22_filt'],#short Laplacian
    # '4_filt':  ['3_filt', '10_filt','22_filt'], #short laplacian with 3 neighbours
    # '5_filt':  CAR_channels,#CAR ????
    # '6_filt':  ['1_filt', '5_filt','7_filt', '12_filt'],
    # '7_filt':  ['2_filt', '6_filt','8_filt', '13_filt'],
    # '8_filt':  ['2_filt', '3_filt','13_filt', '15_filt'],
    #'9_filt':  ['3_filt', '8_filt','10_filt', '15_filt'],
    
    #'10_filt':  ['4_filt', '9_filt','11_filt', '16_filt'],
    # '11_filt':  CAR_channels,#CAR????
    # '12_filt':  ['6_filt', '13_filt','24_filt','27_filt'],
    # '13_filt':  ['6_filt', '8_filt','27_filt', '28_filt'],
    # '14_filt':  ['8_filt', '13_filt','15_filt', '28_filt'],
    # '15_filt':  ['8_filt', '10_filt','28_filt', '29_filt'],
    # '16_filt':  ['10_filt','15_filt','25_filt', '29_filt'],
    # '17_filt':  CAR_channels,#CAR
    # '18_filt':  CAR_channels,#CAR
    # '19_filt':  CAR_channels,#CAR
    # '20_filt':  [ '6_filt','17_filt', '19_filt', '21_filt'], #long laplacian 
    # '21_filt':  ['8_filt', '20_filt', '22_filt'], #long laplacian with 3 neighbours
    # '22_filt':  [ '10_filt','18_filt', '21_filt', '23_filt'], #long laplacian 
    # '23_filt':  CAR_channels, 
    # '24_filt':  ['6_filt','26_filt','27_filt'], #short laplacian with 3 neighbours
    # '25_filt':  ['10_filt','29_filt', '30_filt'], #short laplacian with 3 neighbours
    # '26_filt':  CAR_channels,
    # '27_filt': ['6_filt', '26_filt', '28_filt', '31_filt'], #long laplacian
    # '28_filt':  ['8_filt', '27_filt', '29_filt'], #long laplacian with 3 neighbours
    # '29_filt':  ['10_filt', '28_filt', '30_filt', '32_filt'], #long laplacian
    # '30_filt':  CAR_channels, #CAR
    # '31_filt':  CAR_channels, #CAR
    # '32_filt':  CAR_channels, #CAR
    #}

    # NEIGHBOUR_CHANNELS_DICT = {
    # '15_filt':  ['10_filt', '11_filt','20_filt', '21_filt'],
    # # '16_filt':  brainamp_channel_lists.eeg32_filt,
    # # '17_filt':  brainamp_channel_lists.eeg32_filt,
    # # '18_filt':  ['12_filt', '13_filt','23_filt', '24_filt'],
    # # '19_filt':  ['13_filt', '14_filt','24_filt', '25_filt'],
    # '20_filt':  ['14_filt', '15_filt','25_filt', '26_filt'],
    #}
# Piece of code to define only the channels that have to be taken from the source to compute the laplacian
eeg_channels = copy.copy(channels_2train)
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


#import pdb; pdb.set_trace()
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

db_name = 'default'
#db_name = 'tubingen'


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
    print 'ERROR: selected channels to train and neighbour channels dict entries do NOT match. Correct before training the decoder.'
  #  import pdb; pdb.set_trace()



if artifact_rejection == True:
    hdf = tables.openFile(train_hdf_names[0])
    recorded_channels = hdf.root.brainamp.colnames
#import pdb; pdb.set_trace()
    if 'chanEOGV_filt' in recorded_channels:
        eog_channels = brainamp_channel_lists.eog2_filt#eeg_channels always with filt
        bipolar_EOG = True# Whether the EOG of the calibration and testing data was recorded in monopolar or bipolar mode
        neog_channs = 2
    elif 'chanEOGV' in recorded_channels:
        eog_channels = brainamp_channel_lists.eog2_filt#eeg_channels always with filt
        bipolar_EOG = True
        neog_channs = 2
    elif 'chanEOG1_filt' in recorded_channels:
        eog_channels = brainamp_channel_lists.eog4_filt#eeg_channels always with filt
        bipolar_EOG = False
        neog_channs = 4
    elif 'chanEOG1' in recorded_channels:
        eog_channels = brainamp_channel_lists.eog4_filt#eeg_channels always with filt
        bipolar_EOG = False
        neog_channs = 4

    hdf.close()
    # if filt_training_data == True:
    #     eeg_channels += brainamp_channel_lists.eog4
    # else:
    #     eeg_channels += brainamp_channel_lists.eog4_filt
    eeg_channels += eog_channels#brainamp_channel_lists.eog4_filt
else: # just to avoid errors when EOG was not recorded
    eog_channels = list()
    bipolar_EOG = True
    neog_channs = 2
# Set 'feature_fn_kwargs' to be a dictionary where:
#   key: name of a feature (e.g., 'ZC')
#   value: a dictionary of the keyword arguments to be passed into the feature
#          function (e.g., extract_ZC) corresponding to this feature
#   (see emg_feature_extraction.py for how the feature function definitions)
#import pdb; pdb.set_trace()
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
# eeg_feature_name_list = []
# for feature in feature_names:
#     for channel in channels_2train:
#         eeg_feature_name_list = eeg_feature_name_list + [channel + '_freq_band_' + str(i) for i in np.arange(len(freq_bands[channel]))]

# if os.path.isfile(pkl_name) and not okay_to_overwrite_decoder_file:
#     raise Exception('A decoder file with that name already exists!') 

extractor_cls = EEGMultiFeatureExtractor
extractor_kwargs = {
    'eeg_channels':      eeg_channels,
    'channels_2train':   channels_2train,
    #'brainamp_channels': brainamp_channels,
    'neighbour_channels': NEIGHBOUR_CHANNELS_DICT,
    'feature_names':     feature_names,
    'feature_fn_kwargs': feature_fn_kwargs,
    'win_len':           win_len,
    'fs':                fs,
    'artifact_rejection': artifact_rejection,
    'calibration_data': calibration_data,
    'bipolar_EOG': bipolar_EOG,
}
#'eeg_feature_name_list': eeg_feature_name_list,
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
#import pdb; pdb.set_trace()

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
    #df.entry = models.TaskEntry.objects.using(db_name).get(id=892)   
    df.save()
elif len(dfs) == 1:
    pass # no new data base record needed
elif len(dfs) > 1:
    print "More than one decoder with the same name! fix manually!"


