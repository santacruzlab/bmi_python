from ismore.noninvasive import exg_tasks
#from ismore.noninvasive import train_eeg_decoder
from riglib import experiment
from features.hdf_features import SaveHDF
from ismore.brainamp_features import SimBrainAmpData, SimBrainAmpData_with_encoder, ReplayBrainAmpData
from ismore import bmi_ismoretasks
from ismore import brainamp_channel_lists
import numpy as np
import pandas as pd
#Train EEG Decoder: 
#run train_eeg_decoder.py

#Load EEG Decoder: 
import pickle

# Add decoder name here: 
#decoder = pickle.load(open(decoder_filename))
decoder = pickle.load(open('/storage/decoders/emg_decoder_NI_4928_4931.pkl'))
#decoder = pickle.load(open('/storage/decoders/eeg_decoder_AM_4329_4331_0.pkl'))
gamma = 0.5
use_emg_decoder = 1
#Choose brainamp channels here:
brainamp_channels = brainamp_channel_lists.emg14_raw_filt
hdf_file = '/storage/rawdata/hdf/ni20151214_73_te4931.hdf'
#Choose targets here:
targets = pickle.load(open('/storage/target_matrices/targets_matrix_TF_4777_B2.pkl'))
# targets = dict(targets, **targets)
# targets.update(targets)
#targets = pd.concat([targets.items(), targets.items()])
# targets2 = dict(targets.items() + targets.items())
# print targets2
#Initialize task from command line (note features are SimBrainAmpData and SaveHDF -- could also use SimBrainAmpData_with_encoder
Task = experiment.make(exg_tasks.EMGTrajDecodingEndPoint, (SaveHDF, ReplayBrainAmpData))

#If want to run task for a certain session length, can add "session_length" variable below (in seconds):
task = Task(targets, plant_type="ReHand", use_emg_decoder = use_emg_decoder, gamma = gamma, emg_decoder=decoder, targets_matrix = targets, brainamp_channels=brainamp_channels, session_length=430)

task.init()
task.run()

#When task is finished, save HDF: 
#task.decoder.save()

#Add HDF name here: 
new_hdf_name = 'test_emg_decoding.hdf'#/storage/raw_data/hdf/

#Copy HDF file 
import shutil
f = open(task.h5file.name)
f.close()

#Wait 
import time
time.sleep(2.)

#Wait after HDF cleaned up
task.cleanup_hdf()
import time
time.sleep(2.)
import pdb; pdb.set_trace()
# import os
# open(os.path.join('/storage/raw_data/hdf/','test_emg_decoding.hdf'))
# if not os.path.exists(new_hdf_name):
#     os.popen('mkdir -p %s' % new_hdf_name)


#Copy temp file to actual desired location
shutil.copy(task.h5file.name, new_hdf_name)
f = open(new_hdf_name)
f.close()
