import tables
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
import pickle


from ismore import brainamp_channel_lists
from ismore.noninvasive import find_relevant_channels
from ismore.invasive import emg_decoder
#from ismore.noninvasive import emg_decoding_command_vel
#from ismore.emg_decoding import LinearEMGDecoder
from ismore.noninvasive.emg_feature_extraction import EMGMultiFeatureExtractor
from ismore.common_state_lists import *
from ismore import ismore_bmi_lib
from utils.constants import *
from db import dbfunctions as dbfn
from db.tracker import models
import time

################################################################
# The variables below need to be set before running this script!
################################################################


# Set 'okay_to_overwrite_decoder_file' to True to allow overwriting an existing 
#   file
okay_to_overwrite_decoder_file = True
fixed_var_scalar = True

#train_hdf_ids = [2314] #, 2316, 2317, 2319, 2320, 2330, 2331, 2332, 2333, 2334, 2340, 2341, 2342, 2343, 2344] 


##training data from left compliant sessions03

#train_hdf_ids = [5592,5593,5596,5599] # MB01 B1
train_hdf_ids = [5592,5593,5596,5599,5736,5737,5738,5739] #MB01- B1 and B2
train_hdf_ids = [5736,5737] 
train_hdf_ids = [2314,2316,2317,2319] 


train_hdf_ids = [971] 
train_hdf_ids = [1796] 
train_hdf_ids = [5592,5593,5596,5599]
train_hdf_ids = [5465,5466,5468] 


# train_hdf_ids = [6035] 
train_hdf_ids = [6157,6158,6159,6160,6161]
train_hdf_ids = [6505,6506,6507,6512,6513]
train_hdf_ids = [6495,6496,6497,6498,6499]
# train_hdf_ids = [6297,6298,6299,6300,6301]
# train_hdf_ids = [6310,6311,6312,6313,6314]

#train_hdf_ids = [5738]
#train_hdf_ids = [2054]

train_hdf_ids = [6505,6506, 6507,6512,6513]
train_hdf_ids = [6495,6496,6497,6498,6499] # test data loss
train_hdf_ids = [6823,6824,6825,6826,6831]

# train_hdf_ids = [6505,6506,6507,6512,6513]

train_hdf_ids = [6810,6812,6815,6818,6819]

train_hdf_ids = [6142,6143,6144,6145,6146]
train_hdf_ids = [6905,6907,6908,6909,6910]
train_hdf_ids = [6994,6995,6996,6997,6998]
train_hdf_ids = [7005,7007,7008,7009,7010]
train_hdf_ids = [7015,7016,7017,7031,7034]
train_hdf_ids = [7467,7468,7469,7473,7474]
train_hdf_ids = [7547,7548,7549,7550,7551]
train_hdf_ids = [7652,7653,7654,7655,7656]
train_hdf_ids = [7821,7822,7823,7824,7825]
test_relevant_channels_hdf_ids = []
train_hdf_ids = [8603,8604,8609,8618,8619,8620,8621]
#train_hdf_ids = [6905,6907,6908,6909,6910]
# train_hdf_ids = [6810,6812,6815,6818,6819]
train_hdf_ids = [9647,9650]
#train_hdf_ids = [7467,7468,7469,7473,7474]
#train_hdf_ids = [8250,8251,8252,8253,8254]

# HUD1 - Pre1
#train_hdf_ids = [4737,4738,4742,4743,4746,4747] # healthy arm
#train_hdf_ids = [4737,4738] # healthy arm - B1 tasks
#train_hdf_ids = [4742,4743] # healthy arm - B2 tasks

# HUD1 - Post1
train_hdf_ids = [6967,6968,6971,6973,6974,6976,6979,6980,6982,6984,6987,6988] #healthy arm
#train_hdf_ids = [6967,6968] - B1 tasks
# HUD1 - Pre1 and Post1 data healthy arm
train_hdf_ids = [4737,4738,4742,4743,4746,4747,6967,6968,6971,6973,6974,6976,6979,6980,6982,6984,6987,6988,15270,15271,15272,15273] # healthy arm
train_hdf_ids = [15270,15271,15272,15273]
#train_hdf_ids = [15259] # healthy arm

#train_hdf_ids = [4737,4738,6967,6968,6987,6988] # healthy arm - B1 and F1 tasks
#train_hdf_ids = [4742,4743,6971,6973,6974] # healthy arm - B2 tasks#
#train_hdf_ids = [4746,4747,6976,6979,6980,6982,6984] # healthy arm - B1_B2 tasks#
# test_relevant_channels_hdf_ids = [6819]
#test_relevant_channels_hdf_ids = []
relevant_dofs = [0,1,2,3,4,5,6]
#relevant_dofs = [0,1,2]

# Select relevant DOFs among these ones:
# 0=Vx; 1=Vy; 2=Vpsi; 3=Vprono; 4=Vthumb; 5=Vindex; 6=V3Fing

#train_hdf_ids = [5465,5466,5467,5468,5469,5490,5495,5497,5498,5499]
#train_hdf_ids = [5601,5608]
test_hdf_ids = []
# HUD1 - Pre1
#test_hdf_ids = [4769,4770,4773,4774,4777,4778] # paretic arm 4769,4770,4773,4774,4777,4778
# HUD1 - Pre2
#test_hdf_ids = [4795,4796,4807,4811,4813] # paretic arm #4797
# HUD1 - Post1
#test_hdf_ids = [6937,6938,6946,6949,6950,6953,6954] # paretic arm
#test_hdf_ids = [6937,6938,6946] # paretic arm - B1 task
#test_hdf_ids = [6949, 6950] # paretic arm - B2 task 6949, 6950
# test_hdf_ids = [6954] # paretic arm - B1_B2 task, 6953
#test_hdf_ids = [6967,6968,6971,6973,6974,6976,6979,6980,6982,6984,6987,6988] #healthy arm
#test_hdf_ids = [6967,6968] #healthy arm - B1 tasks
#test_hdf_ids = [6971,6973,6974] # healthy_arm - B2 tasks
# HUD1 - Invasive Phase
# Compliant
# 24/10?2017
test_hdf_ids = [8278,8279,8280]
# 25/10?2017
test_hdf_ids = []
# 26/10?2017
#test_hdf_ids = [8453,8454,8455,8456]
#test_hdf_ids = [8453]#  B1 and F1 tasks
#test_hdf_ids = [8454]#  compliant block 2
#test_hdf_ids = [8678]
# Set 'test_hdf_names' to be a list of names of .hdf files to test on (offline)
# test_hdf_names = [
#     '/home/tecnalia/hybries/resources/EMGdecoder_code_testing/fs20150424_11.hdf' #B1_R5   
# ]

db_name = "default"
# db_name = 'tubingen'

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

# Set 'plant_type' (type of plant for which to create a decoder)
#   choices: 'ArmAssist', 'ReHand', or 'IsMore'
plant_type = 'IsMore'

ssm = ismore_bmi_lib.SSM_CLS_DICT[plant_type]()
states_to_decode = [s.name for s in ssm.states if s.order == 1]

# Set 'channels' to be a list of channel names

#Channels used to train the decoder
#channels_2train = brainamp_channel_lists.emg_48hd_6mono_filt
#channels_2train = brainamp_channel_lists.emg14_filt
subset_muscles_names = dict()
subset_muscles_names['aa_vx'] = ['Biceps',
    'Triceps',
    'FrontDelt',
    'MidDelt',
    'TeresMajor',
    'PectMajor']

subset_muscles_names['aa_vy'] = subset_muscles_names['aa_vx']
subset_muscles_names['aa_vpsi'] = subset_muscles_names['aa_vx']

subset_muscles_names['rh_vthumb'] = [
    'AbdPolLo','FlexDig',
    'FlexCarp']

subset_muscles_names['rh_vindex'] = ['InterFirst','ExtCU',
    'ExtCarp',
    'ExtDig',
    'FlexDig',
    'FlexCarp','PronTer']

subset_muscles_names['rh_vfing3'] = ['ExtCU',
    'ExtCarp',
    'ExtDig',
    'FlexDig',
    'FlexCarp','PronTer']

subset_muscles_names['rh_vprono'] = ['ExtCU',
    'ExtCarp',
    'ExtDig',
    'FlexDig',
    'FlexCarp',
    'PronTer',
    'Biceps']

for state in states_to_decode:
    subset_muscles_names[state] = [i + '_filt' for i in subset_muscles_names[state]]

channels_2train = brainamp_channel_lists.emg14_bip_filt
#channels_2train = emg_thumb_bip_filt
# channels_2train = brainamp_channel_lists.mono_96_filt


#subset_muscles = dict() #If we wanna use a subset of muscles for any of the DoFs, then define it here. Otherwise set it to an empty dict().
subset_muscles = dict()
# ATTENTION! Comment these two lines below if we wanna use the whole set of recorded muscles for the decoding of all DoFs!!!
for state in states_to_decode:
    subset_muscles[state] = [np.int(channels_2train.index(subset_muscles_names[state][i])) for i in np.arange(len(subset_muscles_names[state]))]




nchannels_2select = 60
min_HD_nchans = 10 #minimum number of channels from the HD-EMG that will be kept for the iterations used to find the relevant channels 

filt_training_data = True

# if '_filt' not in channels_2train[0]: #or any other type of configuration using raw signals
#     filt_training_data = True
# else:
#     filt_training_data = False


# Set 'feature_names' to be a list containing the names of features to use
#   (see emg_feature_extraction.py for options)
feature_names = ['MAV', 'VAR', 'WL', 'RMS', 'LOGVAR']

# Set 'feature_fn_kwargs' to be a dictionary where:
#   key: name of a feature (e.g., 'ZC')
#   value: a dictionary of the keyword arguments to be passed into the feature
#          function (e.g., extract_ZC) corresponding to this feature
#   (see emg_feature_extraction.py for how the feature function definitions)
feature_fn_kwargs = {
    'WAMP': {'threshold': 30},  
    'ZC':   {'threshold': 30},
    'SSC':  {'threshold': 700},
}

# If states are any of the DOFs 1-3, then win_len should be = 0.2. Otherwise win_len = 0.02
# Set 'win_len'
win_len = 0.2  # secs

# Set 'fs'
fs = 1000  # Hz

# Set 'K' to be the value of the ridge parameter
K = 10e3  
#K = 10

# Set 'states_to_flip' to be a list containing the state names for which
#   the beta vector of coefficients (trained using ridge regression) should be 
#   flipped (i.e., multiplied by -1)

#hdf = tables.openFile(train_hdf_names[2]); #set to second file because did not allow me to check the first one since the file was already open

#check the sing of the thumb angular position data to see if we need to flip the rehand states from left (training session) to right (tesing/online EMG feedback) session 
# negative values in rehand angluar positions appear only in all session 02 (left active) for all subjectc. In the rest of sessions (left or right) values are always positive
# if hdf.root.rehand[0]['data'][0] < 0:
#     states_to_flip = ['aa_vx', 'aa_vpsi', 'rh_vthumb','rh_vindex','rh_vfing3', 'rh_vprono'] # if the values of the rehand are negative when we recorded with the left hand (sess02)
# else:
#     states_to_flip = ['aa_vx', 'aa_vpsi'] # if the values of the rehand are positive when we recorded with the left hand (sess03), because we took all of them as possitive
mirrored = True #whether to train a mirrored decoder or not
if mirrored == True:
    states_to_flip = ['aa_vx', 'aa_vpsi'] # if the values of the rehand are positive when we recorded with the left hand (sess03), because we took all of them as possitive
else:
    states_to_flip = []
#states_to_flip = []

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

########################################################
# opt_channels = ['1str_filt','2str_filt','1diag1_filt','3diag2_filt']
# opt_channels_dict = dict()
# opt_channels_dict['channels_str_2discard'] = [5,11,17,23,29,35,41,47,49,51,53,55,57]
# opt_channels_dict['channels_str_2keep'] = [i for i in range(59) if i not in opt_channels_dict['channels_str_2discard']]
# opt_channels_dict['channels_diag1_1'] = [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46]
# opt_channels_dict['channels_diag1_2'] = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41]
# opt_channels_dict['channels_diag2_1'] = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40]
# opt_channels_dict['channels_diag2_2'] = [7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47]
# channels_str_2discard = [5,11,17,23,29,35,41,47,49,51,53,55,57]
# channels_str_2keep = [i for i in range(59) if i not in channels_str_2discard]
# channels_diag1_1 = [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46]
# channels_diag1_2 = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41]
# channels_diag2_1 = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40]
# channels_diag2_2 = [7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47]

# opt_channels = [str(i) + 'str_filt' for i in range(len(channels_str_2keep)-6)] + brainamp_channel_lists.emg_6bip_hd_filt + [str(j) + 'diag1_filt' for j in range(len(channels_diag1_1))] + [str(k) + 'diag2_filt' for k in range(len(channels_diag2_1))]

emg_feature_name_list = []
try: 
    for feature in feature_names:
        emg_feature_name_list = emg_feature_name_list + [opt_channel + '_' + feature for opt_channel in opt_channels]
except:
    for feature in feature_names:
        emg_feature_name_list = emg_feature_name_list + [opt_channel + '_' + feature for opt_channel in channels_2train]
# if os.path.isfile(pkl_name) and not okay_to_overwrite_decoder_file:
#     raise Exception('A decoder file with that name already exists!') 

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

decoder = emg_decoder.LinearEMGDecoder(channels_2train, plant_type, fs, win_len, filt_training_data, extractor_cls, extractor_kwargs, opt_channels_2train_dict)
# Using command_vel. It cannot be used the way it is now because during the periods in which the DoFs were not active I believe we set it to nan and when interpolating it, there are big peaks in data.
#decoder = emg_decoding_command_vel.LinearEMGDecoder(channels_2train, plant_type, fs, win_len, filt_training_data, extractor_cls, extractor_kwargs, opt_channels_2train_dict)

#decoder = LinearEMGDecoder(channels, plant_type, fs, win_len, extractor_cls, extractor_kwargs)
decoder.train_ridge(K, train_hdf_names, test_hdf_names, states_to_flip)

decoder.training_ids = train_hdf_ids

train_ids_str = str(min(train_hdf_ids)) + '_' + str(max(train_hdf_ids))

subject_name  = models.TaskEntry.objects.using(db_name).get(id=train_hdf_ids[0]).subject.name

#subject_name  = models.TaskEntry.objects.using("tubingen").get(id=train_hdf_ids[0]).subject.name

decoder_name = 'emg_decoder_%s_%s_PhaseVI_wrist_ext_only_%s' % (subject_name,train_ids_str, time.strftime('%Y%m%d_%H%M'))
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
    #df.entry = models.TaskEntry.objects.using(db_name).get(id=954)
    df.save()
elif len(dfs) == 1:
    pass # no new data base record needed
elif len(dfs) > 1:
    print "More than one decoder with the same name! fix manually!"
