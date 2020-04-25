from ismore.noninvasive import exg_tasks
#from ismore.noninvasive import train_eeg_decoder
from riglib import experiment
from features.hdf_features import SaveHDF
from ismore.brainamp_features import SimBrainAmpData, SimBrainAmpData_with_encoder
from ismore import bmi_ismoretasks
from ismore import brainamp_channel_lists

#Train EEG Decoder: 
#run train_eeg_decoder.py

#Load EEG Decoder: 
import pickle

# Add decoder name here: 
#decoder = pickle.load(open(decoder_filename))
decoder = pickle.load(open('/Users/preeyakhanna/Dropbox/Carmena_Lab/SpainBMI/eeg_decoder_AM_4329_4331_0.pkl'))

#Choose brainamp channels here:
brainamp_channels = brainamp_channel_lists.eeg32_raw_filt

#Choose targets here:
targets = pickle.load(open('/Users/preeyakhanna/Dropbox/Carmena_Lab/SpainBMI/targets_matrix_testing_4462_B1.pkl'))

#Initialize task from command line (note features are SimBrainAmpData and SaveHDF -- could also use SimBrainAmpData_with_encoder
Task = experiment.make(exg_tasks.SimEEGMovementDecoding_PK, (SaveHDF, SimBrainAmpData))

#If want to run task for a certain session length, can add "session_length" variable below (in seconds):
task = Task(targets, plant_type="ArmAssist", decoder=decoder, brainamp_channels=brainamp_channels, session_length=30)

task.init()
task.run()

#When task is finished, save HDF: 
task.decoder.save()

#Add HDF name here: 
new_hdf_name = ''

#Copy HDF file 
import shutil
f = open(task.h5file.name)
f.close()

#Wait 
import time
time.sleep(1.)

#Wait after HDF cleaned up
task.cleanup_hdf()
import time
time.sleep(1.)

#Copy temp file to actual desired location
shutil.copy(task.h5file.name, new_hdf_name)
f = open(new_hdf_name)
f.close()

