import tables
import pandas as pd
import numpy as np
import math
import pdb
import matplotlib.pyplot as plt
import sklearn
import copy
from copy import deepcopy


from sklearn.metrics import r2_score
from db import dbfunctions as dbfn
from scipy.signal import butter, lfilter, filtfilt
from scipy import stats
# from riglib.filter import Filter
from ismore.filter import Filter
from ismore.noninvasive.eeg_feature_extraction import extract_AR_psd
from ismore import brainamp_channel_lists, ismore_bmi_lib
from ismore.noninvasive.eeg_feature_extraction import EEGMultiFeatureExtractor
from utils.constants import *

import nitime.algorithms as tsa
from collections import OrderedDict
## dataset
#hdf_ids = [3962,3964]
hdf_ids = [4296, 4298, 4300,4301]#4296, 4298, 4300, 4301 AI subject
hdf_ids = [5867, 5868, 5869, 5870, 5871]# AI subject

# hdf_ids = [5903,5904,5905,5906,5907]#EL subject
# hdf_ids = [5922,5923,5925,5926,5927]#AS subject
hdf_ids = [5978, 5979, 5980, 5981, 5982]#TF subject
hdf_ids = [5996, 5997, 5998, 5999, 6000]#NI subject
hdf_ids = [5996,5997]
# hdf_ids = [6018, 6019, 6020, 6021, 6022]#FS subject
#hdf_ids = [2634,2635,2637]
#hdf_ids = [2634]
#hdf_ids = [5652,5653,5654,5655]00000
# hdf_ids = [4329,4330,4331] 
hdf_ids = [2635] 
hdf_ids = [3133,3134,3135]
hdf_ids = [7995, 7996, 7997]
hdf_ids = [8229, 8230, 8231]
hdf_ids = [8243,8244,8245] # DK test patient hybrid-BCI
#hdf_ids = [8210,8211,8212]
#hdf_ids = [8678,8679,8683,8684,8685,8686]
hdf_ids = [4486,4487,4491,4495]
#hdf_ids = [5996,5997]


#hdf_ids = [9617,9620,9623,9625,9627] # Complian session left WJ 2016.06.26
#hdf_ids = [9620] # Compliant session left WJ 2016.06.26
#hdf_ids = [9621,9622,9624,9626,9628] #MI session left WJ 2016.06.26

# JO right active session
hdf_ids = [9520,9523]#,9525,9526,9527]



# Test sleep task 25/05/2018
hdf_ids = [10572,10574,10575,10576]

channels_2visualize = brainamp_channel_lists.eeg32_filt #determine here the electrodes that we wanna visualize (either the filtered ones or the raw ones that will be filtered here)
#channels_2visualize = brainamp_channel_lists.eeg32
#channels_2visualize = brainamp_channel_lists.emg14_filt 
trial_hand_side = 'left'
trial_hand_side = 'right'

#HUD1 - compliant session- paretic arm
hdf_ids = [4769,4770,4773,4774,4777,4778] 
#HUD1 - screening session- paretic arm
hdf_ids = [4782,4783,4784,4785,4786]
#HUD1 - screening session 2- paretic arm
hdf_ids = [4818,4819,4820,4821,4822]
#HUD1 - compliant session 2- paretic arm
hdf_ids = [4795,4796,4807,4811,4813] 
hdf_ids = [5906,5911,5914,5916] 

# Test with Doris 23/11/2017
hdf_ids = [10170,10173,10174,10176]


#SleepTest1 - compliant - left arm
hdf_ids = [10596,10595] 
#test training decoder
hdf_ids = [14268, 14269, 14270, 14271] 
hdf_ids = [ 14312,14313] 

#channels_2visualize = brainamp_channel_lists.eeg32_filt #determine here the electrodes that we wanna visualize (either the filtered ones or the raw ones that will be filtered here)
channels_2visualize = brainamp_channel_lists.eeg32
#channels_2visualize = brainamp_channel_lists.emg14_filt 
#trial_hand_side = 'left'
#trial_hand_side = 'right'

db_name = "default"
#db_name = "tecnalia"
#db_name = "tubingen"

hdf_names = []
for id in hdf_ids:
    te = dbfn.TaskEntry(id, dbname= db_name)
    hdf_names.append(te.hdf_filename)
    te.close_hdf()

if '_filt' not in channels_2visualize[0]: #or any other type of configuration using raw signals
    filt_training_data = True
else:
    filt_training_data = False

# Set 'feature_names' to be a list containing the names of features to use
#   (see emg_feature_extraction.py for options)
feature_names = ['AR'] # choose here the feature names that will be used to train the decoder

#frequency_resolution = 3#Define the frequency bins of interest (the ones to be used to train the decoder)

# Set 'feature_fn_kwargs' to be a dictionary where:
#   key: name of a feature (e.g., 'ZC')
#   value: a dictionary of the keyword arguments to be passed into the feature
#          function (e.g., extract_ZC) corresponding to this feature
#   (see emg_feature_extraction.py for how the feature function definitions)
feature_fn_kwargs = {
    'AR': {'freq_bands': dict()},  # choose here the feature names that will be used to train the decoder 
    # 'ZC':   {'threshold': 30},
    # 'SSC':  {'threshold': 700},
}

# The variable 'calibration data' defines the type of data used to calibrate (i.e. train) the EEG decoder. 
# It can be 'screening' when data from a Screening session in which subject (tried to) open and close their hands is used
# or 'compliant' when data in which the subject wore the exoskeleton in his paretic arm and performed a series of tasks with full assistance (i.e. compliant movement)

#calibration_data = 'compliant'
calibration_data = 'screening'

calibration_data = 'compliant_testing'
calibration_data = 'EEG_BMI'
#calibration_data = 'screening'

#calibration_data = 'active'
# Artifact rejection to remove EOG, movement and EMG artifacts. If artifact_rejection variable = True then the rejection is done. Otherwise no rejection is applied.
#artifact_rejection = True
artifact_rejection = False

# neighbour_channels = { #define the neighbour channels for each channel (for the Laplacian filter)
#     '1':  [2,3,4],
#     '2':  [5,6],
#     '3':  [4,5],
# }



neighbour_channels = OrderedDict()
# neighbour_channels = { #define the neighbour channels for each channel (for the Laplacian filter)
#     '1_filt':  channels_2visualize,
#     '2_filt':  channels_2visualize,
#     '3_filt':  channels_2visualize,
#     '4_filt':  channels_2visualize,
#     '5_filt':  channels_2visualize,
#     '6_filt':  channels_2visualize,
#     '7_filt':  channels_2visualize,
#     '8_filt':  ['3_filt', '4_filt','12_filt', '13_filt'],
#     '9_filt':  ['4_filt', '5_filt','13_filt', '14_filt'],
#     '10_filt':  ['5_filt', '6_filt','14_filt', '15_filt'],
#     '11_filt':  ['6_filt', '7_filt','15_filt', '16_filt'],
#     '12_filt':  channels_2visualize,
#     '13_filt':  ['8_filt', '9_filt','18_filt', '19_filt'],
#     '14_filt':  ['9_filt', '10_filt','19_filt', '20_filt'],
#     '15_filt':  ['10_filt', '11_filt','20_filt', '21_filt'],
#     '16_filt':  channels_2visualize,
#     '17_filt':  channels_2visualize,
#     '18_filt':  ['12_filt', '13_filt','23_filt', '24_filt'],
#     '19_filt':  ['13_filt', '14_filt','24_filt', '25_filt'],
#     '20_filt':  ['14_filt', '15_filt','25_filt', '26_filt'],
#     '21_filt':  ['15_filt', '16_filt','26_filt', '27_filt'],
#     '22_filt':  channels_2visualize,
#     '23_filt':  channels_2visualize,
#     '24_filt':  ['18_filt', '19_filt','28_filt'],
#     '25_filt':  ['19_filt', '20_filt','28_filt', '30_filt'],
#     '26_filt':  ['20_filt', '21_filt','30_filt'],
#     '27_filt':  channels_2visualize,
#     '28_filt':  channels_2visualize,
#     '29_filt':  channels_2visualize,
#     '30_filt':  channels_2visualize,
#     '31_filt':  channels_2visualize,
#     '32_filt':  channels_2visualize,
#     }

#New IsMore montage with higher density over the motor cortex
if filt_training_data == False:
    neighbour_channels = {
        '1_filt':  ['2_filt', '6_filt','20_filt'], #short laplacian with 3 neighbours
        '2_filt':  ['6_filt', '8_filt','20_filt', '21_filt'],#short Laplacian
        '3_filt':  ['8_filt', '10_filt','21_filt', '22_filt'],#short Laplacian
        '4_filt':  ['3_filt', '10_filt','22_filt'], #short laplacian with 3 neighbours
        '5_filt':  channels_2visualize,#CAR ????
        '6_filt':  ['1_filt', '5_filt','7_filt', '12_filt'],
        '7_filt':  ['2_filt', '6_filt','8_filt', '13_filt'],
        '8_filt':  ['2_filt', '3_filt','13_filt', '15_filt'],
        '9_filt':  ['3_filt', '8_filt','10_filt', '15_filt'],
        '10_filt':  ['4_filt', '9_filt','11_filt', '16_filt'],
        '11_filt':  channels_2visualize,#CAR????
        '12_filt':  ['6_filt', '13_filt','24_filt','27_filt'],
        '13_filt':  ['6_filt', '8_filt','27_filt', '28_filt'],
        '14_filt':  ['8_filt', '13_filt','15_filt', '28_filt'],
        '15_filt':  ['8_filt', '10_filt','28_filt', '29_filt'],
        '16_filt':  ['10_filt','15_filt','25_filt', '29_filt'],
        '17_filt':  channels_2visualize,#CAR
        '18_filt':  channels_2visualize,#CAR
        '19_filt':  channels_2visualize,#CAR
        '20_filt':  [ '6_filt','17_filt', '19_filt', '21_filt'],#long laplacian 
        '21_filt':  ['8_filt', '20_filt', '22_filt'], #long laplacian with 3 neighbours
        '22_filt':  [ '10_filt','18_filt', '21_filt', '23_filt'], #long laplacian 
        '23_filt':  channels_2visualize, 
        '24_filt':  ['6_filt','26_filt','27_filt'], #short laplacian with 3 neighbours
        '25_filt':  ['10_filt','29_filt', '30_filt'], #short laplacian with 3 neighbours
        '26_filt':  channels_2visualize,
        '27_filt': ['6_filt', '26_filt', '28_filt', '31_filt'], #long laplacian
        '28_filt':  ['8_filt', '27_filt', '29_filt'],#long laplacian with 3 neighbours
        '29_filt':  ['10_filt', '28_filt', '30_filt', '32_filt'], #long laplacian
        '30_filt':  channels_2visualize, #CAR
        '31_filt':  channels_2visualize, #CAR
        '32_filt':  channels_2visualize, #CAR
        }
else:
    neighbour_channels = {
        '1':  ['2', '6','20'], #short laplacian with 3 neighbours
        '2':  ['6', '8','20', '21'],#short Laplacian
        '3':  ['8', '10','21', '22'],#short Laplacian
        '4':  ['3', '10','22'], #short laplacian with 3 neighbours
        '5':  channels_2visualize,#CAR ????
        '6':  ['1', '5','7', '12'],
        '7':  ['2', '6','8', '13'],
        '8':  ['2', '3','13', '15'],
        '9':  ['3', '8','10', '15'],
        '10':  ['4', '9','11', '16'],
        '11':  channels_2visualize,#CAR????
        '12':  ['6', '13','24','27'],
        '13':  ['6', '8','27', '28'],
        '14':  ['8', '13','15', '28'],
        '15':  ['8', '10','28', '29'],
        '16':  ['10','15','25', '29'],
        '17':  channels_2visualize,#CAR
        '18':  channels_2visualize,#CAR
        '19':  channels_2visualize,#CAR
        '20':  [ '6','17', '19', '21'],#long laplacian 
        '21':  ['8', '20', '22'], #long laplacian with 3 neighbours
        '22':  [ '10','18', '21', '23'], #long laplacian 
        '23':  channels_2visualize, 
        '24':  ['6','26','27'], #short laplacian with 3 neighbours
        '25':  ['10','29', '30'], #short laplacian with 3 neighbours
        '26':  channels_2visualize,
        '27': ['6', '26', '28', '31'], #long laplacian
        '28':  ['8', '27', '29'],#long laplacian with 3 neighbours
        '29':  ['10', '28', '30', '32'], #long laplacian
        '30':  channels_2visualize, #CAR
        '31':  channels_2visualize, #CAR
        '32':  channels_2visualize, #CAR
        }

# neighbour_channels['1_filt'] = ['2_filt', '6_filt','20_filt'] #short laplacian with 3 neighbours
# neighbour_channels['2_filt'] = ['6_filt', '8_filt','20_filt', '21_filt']#short Laplacian
# neighbour_channels['3_filt'] =  ['8_filt', '10_filt','21_filt', '22_filt']#short Laplacian
# neighbour_channels['4_filt'] = ['3_filt', '10_filt','22_filt'] #short laplacian with 3 neighbours
# neighbour_channels['5_filt'] = channels_2visualize
# neighbour_channels['6_filt'] =  ['1_filt', '5_filt','7_filt', '12_filt']
# neighbour_channels['7_filt'] =  ['2_filt', '6_filt','8_filt', '13_filt']
# neighbour_channels['8_filt'] =['2_filt', '3_filt','13_filt', '15_filt']
# neighbour_channels['9_filt'] = ['3_filt', '8_filt','10_filt', '15_filt']
# neighbour_channels['10_filt'] = ['4_filt', '9_filt','11_filt', '16_filt']
# neighbour_channels['11_filt'] = channels_2visualize#CAR????
# neighbour_channels['12_filt'] =  ['6_filt', '13_filt','24_filt','27_filt']
# neighbour_channels['13_filt'] = ['6_filt', '8_filt','27_filt', '28_filt']
# neighbour_channels['14_filt'] = ['8_filt', '13_filt','15_filt', '28_filt']
# neighbour_channels['15_filt'] = ['8_filt', '10_filt','28_filt', '29_filt']
# neighbour_channels['16_filt'] = ['10_filt','15_filt','25_filt', '29_filt']
# neighbour_channels['17_filt'] =  channels_2visualize#CAR
# neighbour_channels['18_filt'] = channels_2visualize#CAR
# neighbour_channels['19_filt'] =  channels_2visualize#CAR
# neighbour_channels['20_filt'] =  [ '6_filt','17_filt', '19_filt', '21_filt']#long laplacian 
# neighbour_channels['21_filt'] = ['8_filt', '20_filt', '22_filt'] #long laplacian with 3 neighbours
# neighbour_channels['22_filt'] = [ '10_filt','18_filt', '21_filt', '23_filt'] #long laplacian 
# neighbour_channels['23_filt'] = channels_2visualize
# neighbour_channels['24_filt'] =  ['6_filt','26_filt','27_filt'] #short laplacian with 3 neighbours
# neighbour_channels['25_filt'] = ['10_filt','29_filt', '30_filt'] #short laplacian with 3 neighbours
# neighbour_channels['26_filt'] =  channels_2visualize
# neighbour_channels['27_filt'] = ['6_filt', '26_filt', '28_filt', '31_filt'] #long laplacian
# neighbour_channels['28_filt'] = ['8_filt', '27_filt', '29_filt']#long laplacian with 3 neighbours
# neighbour_channels['29_filt'] = ['10_filt', '28_filt', '30_filt', '32_filt'] #long laplacian
# neighbour_channels['30_filt'] =  channels_2visualize #CAR
# neighbour_channels['31_filt'] = channels_2visualize #CAR
# neighbour_channels['32_filt'] = channels_2visualize #CAR
# All channels CAR filter
# for num, name in enumerate(neighbour_channels):
#     neighbour_channels[name] = channels_2visualize

# neighbour_channels = { #define the neighbour channels for each channel (for the Laplacian filter)
#     '1':  channels_2visualize,
#     '2':  channels_2visualize,
#     '3':  channels_2visualize,
#     '4':  channels_2visualize,
#     '5':  channels_2visualize,
#     '6':  channels_2visualize,
#     '7':  channels_2visualize,
#     '8':  ['3', '4','12', '13'],
#     '9':  ['4', '5','13', '14'],
#     '10':  ['5', '6','14', '15'],
#     '11':  ['6', '7','15', '16'],
#     '12':  channels_2visualize,
#     '13':  ['8', '9','18', '19'],
#     '14':  ['9', '10','19', '20'],
#     '15':  ['10', '11','20', '21'],
#     '16':  channels_2visualize,
#     '17':  channels_2visualize,
#     '18':  ['12', '13','23', '24'],
#     '19':  ['13', '14','24', '25'],
#     '20':  ['14', '15','25', '26'],
#     '21':  ['15', '16','26', '27'],
#     '22':  channels_2visualize,
#     '23':  channels_2visualize,
#     '24':  ['18', '19','28'],
#     '25':  ['19', '20','28', '30'],
#     '26':  ['20', '21','30'],
#     '27':  channels_2visualize,
#     '28':  channels_2visualize,
#     '29':  channels_2visualize,
#     '30':  channels_2visualize,
#     '31':  channels_2visualize,
#     '32':  channels_2visualize,
#     }

hdf = tables.openFile(hdf_names[0])
recorded_channels = hdf.root.brainamp.colnames
#import pdb; pdb.set_trace()
if 'chanEOGV_filt' in recorded_channels:
    eog_channels = brainamp_channel_lists.eog2_filt
    bipolar_EOG = True# Whether the EOG of the calibration and testing data was recorded in monopolar or bipolar mode
    neog_channs = 2
elif 'chanEOGV' in recorded_channels:
    eog_channels = brainamp_channel_lists.eog2
    bipolar_EOG = True
    neog_channs = 2
elif 'chanEOG1_filt' in recorded_channels:
    eog_channels = brainamp_channel_lists.eog4_filt
    bipolar_EOG = False
    neog_channs = 4
elif 'chanEOG1' in recorded_channels:
    eog_channels = brainamp_channel_lists.eog4
    bipolar_EOG = False
    neog_channs = 4
else: #IF EOG was not recorded- artifact rejection is not applied and this avoids errors when running this script
    eog_channels = list()
    bipolar_EOG = False
    neog_channs = 4
hdf.close()


fs = 1000
fs_down = 100
win_len = 0.5  # secs
if artifact_rejection == True:
    channels_2visualize += eog_channels
    if filt_training_data == True:
        rejection_channels = ['chan' + str(i+1) for i in np.arange(16)]
    else:
        rejection_channels = ['chan' + str(i+1) + '_filt' for i in np.arange(16)]

channel_names = ['chan' + name for name in channels_2visualize]
neighbours = dict()
for chan_neighbour in neighbour_channels:#Add the ones used for the Laplacian filter here
    # import pdb; pdb.set_trace()
    neighbours['chan' + chan_neighbour] = []
    for k,chans in enumerate(neighbour_channels[chan_neighbour]):
        new_channel = 'chan' + chans
        neighbours['chan' + chan_neighbour].append(new_channel)
neighbour_channels = copy.copy(neighbours)  


# import pdb; pdb.set_trace()     
# calculate coefficients for a 4th-order Butterworth BPF from 10-450 Hz
#band  = [0.1, 48]  # Hz
band  = [1,48]  # Hz
nyq   = 0.5 * fs
low   = band[0] / nyq
high  = band[1] / nyq
bpf_coeffs = butter(2, [low, high], btype='bandpass')

channel_filterbank_eeg = [None]*len(channel_names)
for k in range(len(channel_names)):
    filts = [Filter(bpf_coeffs[0], bpf_coeffs[1])]
    channel_filterbank_eeg[k] = filts

band  = [48,52]  # Hz
nyq   = 0.5 * fs
low   = band[0] / nyq
high  = band[1] / nyq
notchf_coeffs = butter(2, [low, high], btype='bandstop')

psd_points = win_len *fs_down# length of the window of eeg in which the psd is computed
psd_step = 0.05*fs_down # step size for the eeg window for the psdgm

extractor_cls = EEGMultiFeatureExtractor
f_extractor = extractor_cls(None, channels_2train = [], eeg_channels = channels_2visualize, feature_names = feature_names, feature_fn_kwargs = feature_fn_kwargs, win_len=win_len, fs=fs, neighbour_channels = neighbour_channels, artifact_rejection = artifact_rejection, calibration_data = calibration_data, eog_coeffs = None, TH_lowF = None, TH_highF = None, bipolar_EOG = bipolar_EOG)

features = []
labels = []
rest_lowF_feature_train = None
mov_lowF_feature_train = None
rest_highF_feature_train = None
mov_highF_feature_train = None
store_dir_supp = '/storage/supp_hdf/'



if artifact_rejection == True:

    eog_all = None
    eeg_all = None
    for hdf_name in hdf_names:
        # load EMG data from HDF file
        hdf = tables.openFile(hdf_name)
        
        store_dir_supp = '/storage/supp_hdf/'
        index_slash = hdf_name.encode('ascii','ignore').rfind('/')            
        hdf_supp_name = store_dir_supp + hdf_name.encode('ascii','ignore')[index_slash + 1:-3] + 'supp.hdf'
                
        cues = hdf.root.task_msgs[:]['msg']
        cues_trial_type = hdf.root.task[:]['trial_type']
        cues_events = hdf.root.task_msgs[:]['time']
        cues_times = hdf.root.task[:]['ts']

        n_win_pts = int(win_len * fs)
            
        #step_pts = 50  # TODO -- don't hardcode This is in sample points! Be careful if fps is changed!!
    
        try:
            hdf_supp = tables.open_file(hdf_supp_name)
            eeg = hdf_supp.root.brainamp[:][channel_names]
        except:
            eeg = hdf.root.brainamp[:][channel_names]

        original_ts = eeg[channel_names[0]]['ts_arrival']

        idx = 1            
        while original_ts[idx] == original_ts[0]:
            idx = idx + 1
  
        ts_step = 1./fs
        ts_eeg = np.arange(original_ts[0]-(ts_step*(idx-1)),original_ts[len(original_ts)-1],ts_step)

                
        if filt_training_data == True:
            dtype_eeg = np.dtype([('data', np.float64),
                                ('ts_arrival', np.float64)])
            #import pdb; pdb.set_trace()       
            for k in range(len(channel_names)): #for loop on number of electrodes
                eeg[channel_names[k]]['data'] = lfilter(bpf_coeffs[0],bpf_coeffs[1], eeg[channel_names[k]]['data']) 
                
            #import pdb; pdb.set_trace()  
                    # Use training data to compute coefficients of the regression to remove the EOG
                    # THINK IF WE WANNA BIPOLARIZE FIRST AND THEN FILTER!!!
                    # EOG channels are the last ones in the list
        #import pdb; pdb.set_trace()
        if bipolar_EOG == True:
            eog_v = eeg[channel_names[-1]]['data']
            eog_h = eeg[channel_names[-2]]['data']
        else:
            eog_v = eeg[channel_names[-2]]['data'] - eeg[channel_names[-1]]['data']
            eog_h = eeg[channel_names[-4]]['data'] - eeg[channel_names[-3]]['data']
        eog_v_h = np.vstack([eog_v,eog_h])

        # Put all the training data together to reject EOG artifacts and estimate the coefficients.
        if eog_all == None:
            eog_all = eog_v_h
            eeg_all = np.array([eeg[channel_names[i]]['data'] for i in np.arange(len(channel_names)-neog_channs)])
        else:
            eog_all = np.hstack([eog_all,eog_v_h])
            eeg_all = np.hstack([eeg_all,np.array([eeg[channel_names[i]]['data'] for i in np.arange(len(channel_names)-neog_channs)])])

        if calibration_data == 'screening':
            trial_events = cues_events[np.where(cues == 'trial')] #just in case the block was stopped in the middle of a trial
            if len(trial_events) > len(cues_events[np.where(cues == 'wait')[0][1:]]):
                trial_events = trial_events[:-1]
            rest_start_events = trial_events[np.where(cues_trial_type[trial_events] == 'relax')]
            mov_start_events = trial_events[np.where(cues_trial_type[trial_events] == trial_hand_side)]
            rest_end_events = cues_events[np.where(cues == 'wait')[0][1:]][np.where(cues_trial_type[trial_events]== 'relax')]
            mov_end_events = cues_events[np.where(cues == 'wait')[0][1:]][np.where(cues_trial_type[trial_events]== trial_hand_side)]
        
        elif calibration_data == 'compliant_testing':
            
            trial_cues = np.where(cues == 'trial')[0]
            mov_start_events = cues_events[trial_cues]#just in case the block was stopped in the middle of a trial
            # if len(mov_start_events) > len(cues_events[np.where(cues == 'wait')[0][1:]]):
            #     mov_start_events = mov_start_events[:-1]
            wait_cues = np.where(cues == 'wait')[0][1:]
            instruct_trial_type_cues = np.where(cues == 'instruct_trial_type')[0]
            mov_end_cues = [num for num in instruct_trial_type_cues if num - 1 in trial_cues]       
            mov_end_events = np.sort(cues_events[np.hstack([mov_end_cues,wait_cues])])
            rest_end_cues = [num for num in instruct_trial_type_cues if num not in mov_end_cues]
            rest_start_events = cues_events[np.where(cues == 'rest')]
            rest_end_events = cues_events[rest_end_cues]

        elif calibration_data == 'compliant':
            mov_start_events = cues_events[np.hstack([np.where(cues == 'trial'),np.where(cues == 'trial_return')])][0] #just in case the block was stopped in the middle of a trial
            # if len(mov_start_events) > len(cues_events[np.where(cues == 'wait')[0][1:]]):
            #     mov_start_events = mov_start_events[:-1]
            mov_end_events = cues_events[np.hstack([np.where(cues == 'instruct_rest_return'),(np.where(cues == 'wait')[0][1:],)])][0]
            rest_start_events = cues_events[np.hstack([np.where(cues == 'rest'),np.where(cues == 'rest_return')])][0]
            rest_end_events = cues_events[np.hstack([np.where(cues == 'instruct_trial_type'),np.where(cues == 'instruct_trial_return')])][0]
        
        elif calibration_data == 'EEG_BMI':

            mov_start_events = cues_events[np.hstack([np.where(cues == 'trial'),np.where(cues == 'trial_return')])][0] #just in case the block was stopped in the middle of a trial
            # if len(mov_start_events) > len(cues_events[np.where(cues == 'wait')[0][1:]]):
            #     mov_start_events = mov_start_events[:-1]
            
            mov_end_events = cues_events[np.hstack([np.where(cues == 'instruct_rest_return')[0],np.where(cues == 'instruct_rest')[0][1:],np.where(cues == 'wait')[0][-1]])]
            rest_start_events = cues_events[np.hstack([np.where(cues == 'rest'),np.where(cues == 'rest_return')])][0]
            rest_end_events = cues_events[np.hstack([np.where(cues == 'instruct_trial_type'),np.where(cues == 'instruct_trial_return')])][0]
        
        elif calibration_data == 'active':
            #import pdb; pdb.set_trace()
            # mov_start_events = cues_events[np.hstack([np.where(cues == 'trial'),np.where(cues == 'trial_return')])][0] #just in case the block was stopped in the middle of a trial
            mov_start_events = cues_events[np.where(cues == 'trial')]
            mov_end_events = cues_events[np.where(cues == 'rest')[0][1:]]
            rest_start_events =cues_events[np.where(cues == 'rest')[0][:-1]]
            rest_end_events = cues_events[np.where(cues == 'ready')]

        if  mov_end_events[-1] == len(cues_times):
            mov_end_events[-1] = mov_end_events[-1]-1
        mov_end_times = cues_times[mov_end_events]
        mov_start_times = cues_times[mov_start_events]
        rest_start_times = cues_times[rest_start_events]       
        if  rest_end_events[-1] == len(cues_times):
            rest_end_events[-1] = rest_end_events[-1]-1
        rest_end_times = cues_times[rest_end_events]
        

        #keep same number of trials of each class
        if len(rest_start_times) > len(mov_start_times):
            rest_start_times = rest_start_times[:len(mov_start_times)]
            rest_end_times =  rest_end_times[:len(mov_end_times)]
        elif len(mov_start_times) > len(rest_start_times):
            mov_start_times = mov_start_times[:len(rest_start_times)]
            mov_end_times =  mov_end_times[:len(rest_end_times)]
        # The amount and length of REST intervals and MOVEMENT intervals with the paretic hand should be the same
       
        rest_start_idxs_eeg = np.zeros(len(rest_start_times))
        mov_start_idxs_eeg = np.zeros(len(mov_start_times))
        rest_end_idxs_eeg = np.zeros(len(rest_end_times))
        mov_end_idxs_eeg = np.zeros(len(mov_end_times))
        for idx in range(len(rest_start_times)):
            # find indexes of eeg signal corresponding to start and end of rest/mov periods
            time_dif = [ts_eeg - rest_start_times[idx]]
                    
            bool_idx = time_dif == min(abs(time_dif[0]))
            if np.all(bool_idx == False):
                bool_idx = time_dif == -1*min(abs(time_dif[0]))
            rest_start_idxs_eeg[idx] = np.where(bool_idx[0])[0]

            time_dif = [ts_eeg - rest_end_times[idx]]
            bool_idx = time_dif == min(abs(time_dif[0]))
            if np.all(bool_idx == False):
                bool_idx = time_dif == -1*min(abs(time_dif[0]))
            rest_end_idxs_eeg[idx] = np.where(bool_idx[0])[0]

            time_dif = [ts_eeg - mov_start_times[idx]]
            bool_idx = time_dif == min(abs(time_dif[0]))
            if np.all(bool_idx == False):
                bool_idx = time_dif == -1*min(abs(time_dif[0]))
            mov_start_idxs_eeg[idx] = np.where(bool_idx[0])[0]

            time_dif = [ts_eeg - mov_end_times[idx]]
            bool_idx = time_dif == min(abs(time_dif[0]))
            if np.all(bool_idx == False):
                bool_idx = time_dif == -1*min(abs(time_dif[0]))
            mov_end_idxs_eeg[idx] = np.where(bool_idx[0])[0]
            
        # get rid of first rest and first trials segments to avoid using data with initial artifact from the filter
        rest_start_idxs_eeg = rest_start_idxs_eeg[1:]
        mov_start_idxs_eeg = mov_start_idxs_eeg[1:]
        rest_end_idxs_eeg = rest_end_idxs_eeg[1:]
        mov_end_idxs_eeg = mov_end_idxs_eeg[1:]

        r_lowF_feature_file = None
        m_lowF_feature_file = None
        r_highF_feature_file = None
        m_highF_feature_file = None
        # Compute power in delta and gamma bands for movement (lowF) and muscle (highF) artifact rejection
        for k in rejection_channels:  
                     
            if 'EOG' not in k:#k[4:] not in brainamp_channel_lists.eog4 + brainamp_channel_lists.eog4_filt:
                #import pdb; pdb.set_trace()
                r_lowF_feature_ch = None
                m_lowF_feature_ch = None
                r_highF_feature_ch = None
                m_highF_feature_ch = None
                
                for idx in range(len(rest_start_idxs_eeg)):
                    rest_window_all = eeg[k]['data'][rest_start_idxs_eeg[idx]:rest_end_idxs_eeg[idx]]
                    mov_window_all = eeg[k]['data'][mov_start_idxs_eeg[idx]:mov_end_idxs_eeg[idx]] 
           
                    rest_window = rest_window_all[np.arange(0,len(rest_window_all),fs/fs_down)]
                    mov_window = mov_window_all[np.arange(0,len(mov_window_all),fs/fs_down)]
                    found_index = k.find('n') + 1
                    chan_freq = k[found_index:]
                    #3.- Take windows of 500ms every 50ms (i.e. overlap of 450ms)
                    #4.- Extract features (AR-psd) of each of these windows 
                    n = 0
                    while n <= (len(rest_window) - psd_points) and n <= (len(mov_window) - psd_points):

                        if artifact_rejection == True:
                            r_lowF_feature = extract_AR_psd(rest_window[n:n+psd_points],[1,4])
                            m_lowF_feature = extract_AR_psd(mov_window[n:n+psd_points],[1,4])

                            r_highF_feature = extract_AR_psd(rest_window[n:n+psd_points],[30,48])
                            m_highF_feature = extract_AR_psd(mov_window[n:n+psd_points],[30,48])

                            if r_lowF_feature_ch == None:
                                r_lowF_feature_ch = r_lowF_feature
                                m_lowF_feature_ch = m_lowF_feature
                                r_highF_feature_ch = r_highF_feature
                                m_highF_feature_ch = m_highF_feature
                            else:
                                r_lowF_feature_ch = np.vstack([r_lowF_feature_ch, r_lowF_feature])
                                m_lowF_feature_ch = np.vstack([m_lowF_feature_ch, m_lowF_feature])
                                r_highF_feature_ch = np.vstack([r_highF_feature_ch, r_highF_feature])
                                m_highF_feature_ch = np.vstack([m_highF_feature_ch, m_highF_feature])
                
                        n += psd_step
                
                # Build feature array with columns being the channels and rows the features computed in each window
                if r_lowF_feature_file == None: 
                    r_lowF_feature_file = r_lowF_feature_ch
                    m_lowF_feature_file = m_lowF_feature_ch
                    r_highF_feature_file = r_highF_feature_ch
                    m_highF_feature_file = m_highF_feature_ch
                                
                else:
                    r_lowF_feature_file = np.hstack([r_lowF_feature_file, r_lowF_feature_ch])
                    m_lowF_feature_file = np.hstack([m_lowF_feature_file, m_lowF_feature_ch])
                    r_highF_feature_file = np.hstack([r_highF_feature_file, r_highF_feature_ch])
                    m_highF_feature_file = np.hstack([m_highF_feature_file, m_highF_feature_ch])
                

        if artifact_rejection == True:
            if rest_lowF_feature_train == None:
                rest_lowF_feature_train = r_lowF_feature_file
                mov_lowF_feature_train = m_lowF_feature_file
                rest_highF_feature_train = r_highF_feature_file
                mov_highF_feature_train = m_highF_feature_file

            else:
                rest_lowF_feature_train = np.vstack([rest_lowF_feature_train,r_lowF_feature_file])
                mov_lowF_feature_train = np.vstack([mov_lowF_feature_train,m_lowF_feature_file])
                rest_highF_feature_train = np.vstack([rest_highF_feature_train,r_highF_feature_file])
                mov_highF_feature_train = np.vstack([mov_highF_feature_train,m_highF_feature_file])

        # Close hdf files so taht they can be reopened below
        hdf.close()
        hdf_supp.close()
       
    # Compute the coefficients of the regression for the EOG artifact rejection
    #import pdb; pdb.set_trace()
    # UNCOMMENT
    
    # Curve fit
    #coeff, _ = curve_fit(self.func, eog_all, eeg_all)
    #b0, b1 = coeff[0], coeff[1]
  #  import scipy.io
   # scipy.io.savemat('/home/tecnalia/code/ismore/noninvasive/eog_eeg.mat', mdict = {'eog': eog_all, 'eeg': eeg_all})

    covariance = np.cov(np.vstack([eog_all, eeg_all]))
    autoCovariance_eog = covariance[:eog_all.shape[0],:eog_all.shape[0]]
    crossCovariance = covariance[:eog_all.shape[0],eog_all.shape[0]:]

    eog_coeffs = np.linalg.solve(autoCovariance_eog, crossCovariance)


#import pdb; pdb.set_trace()    
    
for name in hdf_names:
    # load EMG data from HDF file
    hdf = tables.openFile(name)

    
    index_slash = name.encode('ascii','ignore').rfind('/')            
    hdf_supp_name = store_dir_supp + name.encode('ascii','ignore')[index_slash + 1:-3] + 'supp.hdf'
            
          
    try:
        hdf_supp = tables.open_file(hdf_supp_name)
        eeg = hdf_supp.root.brainamp[:][channel_names]
    except:
        eeg = hdf.root.brainamp[:][channel_names]
            
    #eeg = hdf.root.brainamp[:][channel_names]
    cues = hdf.root.task_msgs[:]['msg']
    cues_trial_type = hdf.root.task[:]['trial_type']
    cues_events = hdf.root.task_msgs[:]['time']
    cues_times = hdf.root.task[:]['ts']

    n_win_pts = int(win_len * fs)
            
    step_pts = 50  # TODO -- don't hardcode This is in sample points! Be careful if fps is changed!!
    
    original_ts = eeg[channel_names[0]]['ts_arrival']
    idx = 1            
    while original_ts[idx] == original_ts[0]:
        idx = idx + 1
  
    ts_step = 1./fs
    ts_eeg = np.arange(original_ts[0]-(ts_step*(idx-1)),original_ts[-1],ts_step)
    #steps to follow to visualize the r2 plots

    #1.- Filter the whole signal - only if non-filtered data is extracted from the source/HDF file.
    # BP and NOTCH-filters might or might not be applied here, depending on what data (raw or filt) we are reading from the hdf file
    #filt_training_data = True
    if filt_training_data == True:
        
        for k in range(len(channel_names)): #for loop on number of electrodes
            # plt.plot(eeg[channel_names[k]]['data'])
            # plt.show(block = False)
            # import pdb; pdb.set_trace()
            for filt in channel_filterbank_eeg[k]:
                eeg[channel_names[k]]['data'] =  filt(eeg[channel_names[k]]['data'])
            #eeg[channel_names[k]]['data'] = lfilter(bpf_coeffs[0],bpf_coeffs[1], eeg[channel_names[k]]['data']) 
            #filtered_data = lfilter(bpf_coeffs[0],bpf_coeffs[1], eeg[channel_names[k]]['data']) 
            #eeg[channel_names[k]]['data'] = lfilter(notchf_coeffs[0],notchf_coeffs[1], eeg[channel_names[k]]['data']) 
            # plt.plot(filtered_data, color = 'red')
            # plt.show(block = False)
            # import pdb; pdb.set_trace()

    # Apply artifact rejection (optional)
    if artifact_rejection == True:
        # EOG removal
        if bipolar_EOG == True:
            eog_v = eeg[channel_names[-1]]['data']
            eog_h = eeg[channel_names[-2]]['data']
        else:
            eog_h = eeg[channel_names[-4]]['data'] - eeg[channel_names[-3]]['data']
            eog_v = eeg[channel_names[-2]]['data'] - eeg[channel_names[-1]]['data'] 
    
        #eog_v_h = np.vstack([eog_v,eog_h])

        for k in range(len(channel_names)): #for loop on number of electrodes
            if 'EOG' not in channel_names[k]:#[4:] not in brainamp_channel_lists.eog4 + brainamp_channel_lists.eog4_filt:
                #import pdb; pdb.set_trace()
                eeg[channel_names[k]]['data'] = eeg[channel_names[k]]['data'] - eog_coeffs[0,k]*eog_v - eog_coeffs[1,k]*eog_h
                #eeg[channel_names[k]]['data'] = eeg[channel_names[k]]['data'] - eog_coeffs[k,0].reshape(-1,1)*eog_v - eog_coeffs[k,1].reshape(-1,1)*eog_h
                
    # Laplacian filter -  this has to be applied always, independently of using raw or filt data            
    eeg = f_extractor.Laplacian_filter(eeg)
    #import pdb; pdb.set_trace()
    # savemat(os.path.expandvars('$HOME/code/ismore/filtered_Laplace_eeg15.mat'), dict(filtered_data = eeg['chan15']['data']))
    # savemat(os.path.expandvars('$HOME/code/ismore/filtered_Laplace_eeg7.mat'), dict(filtered_data = eeg['chan7']['data']))
    # import pdb; pdb.set_trace()
    #2.- Break done the signal into trial intervals (concatenate relax and right trials as they happen in the exp)
    if calibration_data == 'screening':
        trial_events = cues_events[np.where(cues == 'trial')] #just in case the block was stopped in the middle of a trial
        if len(trial_events) > len(cues_events[np.where(cues == 'wait')][1:]):
            trial_events = trial_events[:-1]
        rest_start_events = trial_events[np.where(cues_trial_type[trial_events] == 'relax')]
        # rest_start_times = cues_times[rest_start_events]
        mov_start_events = trial_events[np.where(cues_trial_type[trial_events] == trial_hand_side)]
        # mov_start_times = cues_times[mov_start_events]
        rest_end_events = cues_events[np.where(cues == 'wait')][1:][np.where(cues_trial_type[trial_events]== 'relax')]
        # if  rest_end_events[-1] == len(cues_times):
        #     rest_end_events[-1] = rest_end_events[-1]-1
        # rest_end_times = cues_times[rest_end_events]
        mov_end_events = cues_events[np.where(cues == 'wait')][1:][np.where(cues_trial_type[trial_events]== trial_hand_side)]

    #import pdb; pdb.set_trace()
        # if  mov_end_events[-1] == len(cues_times):
        #     mov_end_events[-1] = mov_end_events[-1]-1
        # mov_end_times = cues_times[mov_end_events]

    elif calibration_data == 'compliant':
        #import pdb; pdb.set_trace()
        mov_start_events = cues_events[np.hstack([np.where(cues == 'trial'),np.where(cues == 'trial_return')])][0] #just in case the block was stopped in the middle of a trial
        # if len(mov_start_events) > len(cues_events[np.where(cues == 'wait')[0][1:]]):
        #     mov_start_events = mov_start_events[:-1]
        mov_end_events = cues_events[np.hstack([np.where(cues == 'instruct_rest_return'),(np.where(cues == 'wait')[0][1:],)])][0]
        rest_start_events = cues_events[np.hstack([np.where(cues == 'rest'),np.where(cues == 'rest_return')])][0]
        rest_end_events = cues_events[np.hstack([np.where(cues == 'instruct_trial_type'),np.where(cues == 'instruct_trial_return')])][0]
    
    elif calibration_data == 'EEG_BMI':

            mov_start_events = cues_events[np.hstack([np.where(cues == 'trial'),np.where(cues == 'trial_return')])][0] #just in case the block was stopped in the middle of a trial
            # if len(mov_start_events) > len(cues_events[np.where(cues == 'wait')[0][1:]]):
            #     mov_start_events = mov_start_events[:-1]
            mov_end_events = cues_events[np.hstack([np.where(cues == 'instruct_rest_return')[0],np.where(cues == 'instruct_rest')[0][1:],np.where(cues == 'wait')[0][-1]])]
            rest_start_events = cues_events[np.hstack([np.where(cues == 'rest'),np.where(cues == 'rest_return')])][0]
            rest_end_events = cues_events[np.hstack([np.where(cues == 'instruct_trial_type'),np.where(cues == 'instruct_trial_return')])][0]
        
    elif calibration_data == 'compliant_testing':
            
        trial_cues = np.where(cues == 'trial')[0]
        mov_start_events = cues_events[trial_cues]#just in case the block was stopped in the middle of a trial
            # if len(mov_start_events) > len(cues_events[np.where(cues == 'wait')[0][1:]]):
            #     mov_start_events = mov_start_events[:-1]
        wait_cues = np.where(cues == 'wait')[0][1:]
        instruct_trial_type_cues = np.where(cues == 'instruct_trial_type')[0]
        mov_end_cues = [num for num in instruct_trial_type_cues if num - 1 in trial_cues]       
        mov_end_events = np.sort(cues_events[np.hstack([mov_end_cues,wait_cues])])
        rest_end_cues = [num for num in instruct_trial_type_cues if num not in mov_end_cues]
        rest_start_events = cues_events[np.where(cues == 'rest')]
        rest_end_events = cues_events[rest_end_cues]
    
    elif calibration_data == 'active':
        #import pdb; pdb.set_trace()
        # mov_start_events = cues_events[np.hstack([np.where(cues == 'trial'),np.where(cues == 'trial_return')])][0] #just in case the block was stopped in the middle of a trial
        mov_start_events = cues_events[np.where(cues == 'trial')]
        mov_end_events = cues_events[np.where(cues == 'rest')[0][1:]]
        rest_start_events =cues_events[np.where(cues == 'rest')[0][:-1]]
        rest_end_events = cues_events[np.where(cues == 'ready')]
    #import pdb; pdb.set_trace()
    if  mov_end_events[-1] == len(cues_times):
        mov_end_events[-1] = mov_end_events[-1]-1
    mov_end_times = cues_times[mov_end_events]
    mov_start_times = cues_times[mov_start_events]
    rest_start_times = cues_times[rest_start_events]       
    if  rest_end_events[-1] == len(cues_times):
        rest_end_events[-1] = rest_end_events[-1]-1
    rest_end_times = cues_times[rest_end_events]

    # trial_events = cues_events[np.where(cues == 'trial')] #just in case the block was stopped in the middle of a trial
    # if len(trial_events) > len(cues_events[np.where(cues == 'wait')][1:]):
    #     trial_events = trial_events[:-1]
    # rest_start_events = trial_events[np.where(cues_trial_type[trial_events] == 'relax')]
    # rest_start_times = cues_times[rest_start_events]
    # mov_start_events = trial_events[np.where(cues_trial_type[trial_events] == trial_hand_side)]
    # mov_start_times = cues_times[mov_start_events]
    # rest_end_events = cues_events[np.where(cues == 'wait')][1:][np.where(cues_trial_type[trial_events]== 'relax')]
    # if  rest_end_events[-1] == len(cues_times):
    #     rest_end_events[-1] = rest_end_events[-1]-1
    # rest_end_times = cues_times[rest_end_events]
    # mov_end_events = cues_events[np.where(cues == 'wait')][1:][np.where(cues_trial_type[trial_events]== trial_hand_side)]

    
    # if  mov_end_events[-1] == len(cues_times):
    #     mov_end_events[-1] = mov_end_events[-1]-1
    # mov_end_times = cues_times[mov_end_events]

    #keep same number of trials of each class
    if len(rest_start_times) > len(mov_start_times):
        rest_start_times = rest_start_times[:len(mov_start_times)]
        rest_end_times =  rest_end_times[:len(mov_end_times)]
    elif len(mov_start_times) > len(rest_start_times):
        mov_start_times = mov_start_times[:len(rest_start_times)]
        mov_end_times =  mov_end_times[:len(rest_end_times)]
    
    
    # The amount and length of REST intervals and MOVEMENT intervals with the paretic hand should be the same
    # Otherwise include a loop here to check the length of the arrays and limit them to the minimum length of all.
    rest_start_idxs_eeg = np.zeros(len(rest_start_times))
    mov_start_idxs_eeg = np.zeros(len(mov_start_times))
    rest_end_idxs_eeg = np.zeros(len(rest_end_times))
    mov_end_idxs_eeg = np.zeros(len(mov_end_times))
    for idx in range(len(rest_start_times)):
        time_dif = [ts_eeg - rest_start_times[idx]]
        bool_idx = time_dif == min(abs(time_dif[0]))
        if np.all(bool_idx == False):
            bool_idx = time_dif == -1*min(abs(time_dif[0]))
        #t_idx = np.where(bool_idx[0])
        #import pdb; pdb.set_trace()
        rest_start_idxs_eeg[idx] = np.where(bool_idx[0])[0]
        #rest_start_times_eeg[idx] = ts_eeg[t_idx]

        time_dif = [ts_eeg - rest_end_times[idx]]
        bool_idx = time_dif == min(abs(time_dif[0]))
        if np.all(bool_idx == False):
            bool_idx = time_dif == -1*min(abs(time_dif[0]))
        #t_idx = np.where(bool_idx[0])
        #rest_end_times_eeg[idx] = ts_eeg[t_idx]
        rest_end_idxs_eeg[idx] = np.where(bool_idx[0])[0]

        time_dif = [ts_eeg - mov_start_times[idx]]
        bool_idx = time_dif == min(abs(time_dif[0]))
        if np.all(bool_idx == False):
            bool_idx = time_dif == -1*min(abs(time_dif[0]))
        #t_idx = np.where(bool_idx[0])
        #mov_start_times_eeg[idx] = ts_eeg[t_idx]
        mov_start_idxs_eeg[idx] = np.where(bool_idx[0])[0]

        time_dif = [ts_eeg - mov_end_times[idx]]
        bool_idx = time_dif == min(abs(time_dif[0]))
        if np.all(bool_idx == False):
            bool_idx = time_dif == -1*min(abs(time_dif[0]))
        #t_idx = np.where(bool_idx[0])
        #mov_end_times_eeg[idx] = ts_eeg[t_idx]
        mov_end_idxs_eeg[idx] = np.where(bool_idx[0])[0]
    # Do not include the first trial to avoid initial artifact due to filter to alter the results
    rest_start_idxs_eeg = rest_start_idxs_eeg[1:]
    mov_start_idxs_eeg = mov_start_idxs_eeg[1:]
    rest_end_idxs_eeg = rest_end_idxs_eeg[1:]
    mov_end_idxs_eeg = mov_end_idxs_eeg[1:]
    #import pdb; pdb.set_trace()
    # fsample = 1000.00
    # f = 22 # in Hz
    # rest_amp = 33
    # mov_amp = 5
    if artifact_rejection == True:
        #import pdb; pdb.set_trace()
        channel_names_eeg = [chan for chan in channel_names if 'EOG' not in chan] # not in brainamp_channel_lists.eog4 + brainamp_channel_lists.eog4_filt + brainamp_channel_lists.eog2 + brainamp_channel_lists.eog]
    else:
        channel_names_eeg = channel_names
    for chan_n, k in enumerate(channel_names_eeg): 
        r_features_ch = None
        m_features_ch = None
        #eeg_power = None
        found_index = k.find('n') + 1
        chan_freq = k[found_index:]

        for idx in range(len(rest_start_idxs_eeg)):
            # if chan_n == 14:
            #     n = 0
            #     import pdb; pdb.set_trace()
            #     while n <= (len(eeg[k]['data']) - psd_points):
                
            #         eeg_power_all = f_extractor.extract_features(eeg[k]['data'][n:n+psd_points],chan_freq)  
            #         eeg_power_mean = (eeg_power_all[5:40])
            #         if eeg_power == None:
            #             eeg_power = eeg_power_mean.copy()
                        
            #         else:
            #             eeg_power = np.vstack([eeg_power, eeg_power_mean])
            #         n += 1                      
            #     import pdb; pdb.set_trace()
            #     import os
            #     from scipy.io import savemat

            #     savemat(os.path.expandvars('$HOME/code/ismore/noninvasive/only_eeg_power_5981.mat'), dict(eeg_power = eeg_power, ts_eeg = ts_eeg, rest_end_eeg = rest_end_idxs_eeg, mov_end_eeg = mov_end_idxs_eeg, rest_start_eeg = rest_start_idxs_eeg, mov_start_eeg = mov_start_idxs_eeg))    

            rest_window_all = eeg[k][rest_start_idxs_eeg[idx]:rest_end_idxs_eeg[idx]]['data']
            mov_window_all = eeg[k][mov_start_idxs_eeg[idx]:mov_end_idxs_eeg[idx]]['data']
            rest_window = rest_window_all[np.arange(0,len(rest_window_all),fs/fs_down)]
            mov_window = mov_window_all[np.arange(0,len(mov_window_all),fs/fs_down)]

            # # Fake the rest signal in some of the channels
            # # if chan_n == 12 or chan_n == 14:
            # t = np.arange(0, (rest_end_idxs_eeg[idx] - rest_start_idxs_eeg[idx])/fsample, 1/fsample)
            # rest_signal = np.zeros(len(t))
            # for i in np.arange(len(t)):
            #     rest_signal[i] = rest_amp * math.sin((f-1)*2*math.pi*t[i]) 
            #     #import pdb; pdb.set_trace()
            # if len(rest_signal) != rest_end_idxs_eeg[idx] - rest_start_idxs_eeg[idx]:
            #     import pdb; pdb.set_trace()
            # # rest_window = eeg[k][rest_start_idxs_eeg[idx]:rest_end_idxs_eeg[idx]]['data'] + rest_signal
            # rest_window = rest_signal
            # # else:
            # #     rest_window = eeg[k][rest_start_idxs_eeg[idx]:rest_end_idxs_eeg[idx]]['data']
            # # Fake the move signal in some of the channels
            # if chan_n == 12 or chan_n == 14:
            #     t = np.arange(0, (mov_end_idxs_eeg[idx] - mov_start_idxs_eeg[idx])/fsample, 1/fsample)
            #     mov_signal = np.zeros(len(t))
            #     for i in np.arange(len(t)):
            #         mov_signal[i] = mov_amp * math.sin((f-1)*2*math.pi*t[i]) 
            #         #import pdb; pdb.set_trace()
            #     if len(mov_signal) != mov_end_idxs_eeg[idx] - mov_start_idxs_eeg[idx]:
            #         import pdb; pdb.set_trace()
            #     # mov_window = eeg[k][mov_start_idxs_eeg[idx]:mov_end_idxs_eeg[idx]]['data'] + mov_signal
            #     mov_window = mov_signal
            # else:
            #     mov_window = rest_signal

            #3.- Take windows of 500ms every 50ms (i.e. overlap of 450ms)
            #4.- Extract features (AR-psd) of each of these windows 
            n = 0
            # if chan_n == 17:
            #     import pdb; pdb.set_trace()
            while n <= (len(rest_window) - psd_points) and n <= (len(mov_window) - psd_points):
                
                r_feats = f_extractor.extract_features(rest_window[n:n+psd_points],chan_freq)                        
                m_feats = f_extractor.extract_features(mov_window[n:n+psd_points],chan_freq)
                # if chan_n == 14:
                #     import pdb; pdb.set_trace()
                if r_features_ch == None:
                    r_features_ch = r_feats[0:50].copy()
                    m_features_ch = m_feats[0:50].copy()
                else:
                    r_features_ch = np.vstack([r_features_ch, r_feats[0:50]])
                    m_features_ch = np.vstack([m_features_ch, m_feats[0:50]])
                n += psd_step
            # if chan_n == 12:
                
            #     mean_PSD_mov =  np.mean(m_features_ch, axis = 0)
            #     mean_PSD_rest =  np.mean(r_features_ch, axis = 0)
                #plt.figure(); plt.plot(r_features_ch.T); plt.show()
                #import pdb; pdb.set_trace()
                # import os
                # savemat(os.path.expandvars('$HOME/code/ismore/all_psds.mat'), dict(r_features_ch = r_features_ch, m_features_ch = m_features_ch))
                #plt.figure(); plt.plot(mean_PSD_mov); plt.plot(mean_PSD_rest,color ='red'); plt.show()
        
        if  len(features) < len(channel_names_eeg):
            features.append(np.vstack([r_features_ch, m_features_ch]))
            labels.append(np.vstack([np.zeros(r_features_ch.shape), np.ones(m_features_ch.shape)]))

        else: #when we are using more than one hdf file
           
            features[chan_n] = np.vstack([features[chan_n],np.vstack([r_features_ch, m_features_ch])])
            labels[chan_n] = np.vstack([labels[chan_n],np.vstack([np.zeros(r_features_ch.shape), np.ones(m_features_ch.shape)])])

# Reject the trials that are not below the computed TH.
if artifact_rejection == True:
    # First iteration
    # Compute thresholds for low and high frequencies artifact rejection during the rest period.
    TH_lowF = np.mean(rest_lowF_feature_train, axis = 0) + 3*np.std(rest_lowF_feature_train, axis = 0)
    TH_highF = np.mean(rest_highF_feature_train, axis = 0) + 3*np.std(rest_highF_feature_train, axis = 0)

    # Separate features into rest and mov features and apply artifact rejection to each of them separately
    #import pdb; pdb.set_trace()

    rest_features_artifacted = list()
    mov_features_artifacted = list()
    for i in range(len(features)):
        rest_features_artifacted.append(features[i][np.where([labels[0][:,0] == 0])[1],:])
        mov_features_artifacted.append(features[i][np.where([labels[0][:,0] == 1])[1],:])#If any channel is contaminated, remove that point in all channels.
    
    
    # If we force to reject same amount of rest and mov data
    # index_keep_rest = np.all(self.rest_lowF_feature_train < self.TH_lowF, axis = 1) & np.all(self.rest_highF_feature_train < self.TH_highF, axis = 1)
    # index_keep_mov = np.all(self.mov_lowF_feature_train < self.TH_lowF, axis = 1) & np.all(self.mov_highF_feature_train < self.TH_highF, axis = 1)
    #index_keep_common = np.where([index_keep_mov & index_keep_rest])[1]
    # rest_features_clean1 = rest_features_artifacted[index_keep_common,:]
    # mov_features_clean1 = mov_features_artifacted[index_keep_common,:]
    # rest_lowF_feature_clean = self.rest_lowF_feature_train[index_keep_common,:]
    # mov_lowF_feature_clean = self.mov_lowF_feature_train[index_keep_common,:]
    # rest_highF_feature_clean = self.rest_highF_feature_train[index_keep_common,:]
    # mov_highF_feature_clean = self.mov_highF_feature_train[index_keep_common,:]
    # windows_rejected_train_it1 = self.rest_lowF_feature_train.shape[0] - len(index_keep_common)
    # total_windows_train_it1 = self.rest_lowF_feature_train.shape[0] 

    # We don't force to reject same amount of rest and mov windows
    index_keep_rest = np.where([np.all(rest_lowF_feature_train < TH_lowF, axis = 1) & np.all(rest_highF_feature_train < TH_highF, axis = 1)])[1]
    index_keep_mov = np.where([np.all(mov_lowF_feature_train < TH_lowF, axis = 1) & np.all(mov_highF_feature_train < TH_highF, axis = 1)])[1]
    
    rest_features_clean1 = list()
    mov_features_clean1 = list()
    for i in range(len(rest_features_artifacted)):
        rest_features_clean1.append(rest_features_artifacted[i][index_keep_rest,:])
        mov_features_clean1.append(mov_features_artifacted[i][index_keep_mov,:])
    rest_lowF_feature_clean = rest_lowF_feature_train[index_keep_rest,:]
    mov_lowF_feature_clean = mov_lowF_feature_train[index_keep_mov,:]
    rest_highF_feature_clean = rest_highF_feature_train[index_keep_rest,:]
    mov_highF_feature_clean = mov_highF_feature_train[index_keep_mov,:]
    windows_rest_rejected_train_it1 = rest_lowF_feature_train.shape[0] - len(index_keep_rest)
    windows_mov_rejected_train_it1 = mov_lowF_feature_train.shape[0] - len(index_keep_mov)
    total_windows_train_it1 = rest_lowF_feature_train.shape[0] 
    
    # Second iteration
    # Recompute new TH in cleaned rest periods
    TH_lowF = np.mean(rest_lowF_feature_clean, axis = 0) + 3*np.std(rest_lowF_feature_clean, axis = 0)
    TH_highF = np.mean(rest_highF_feature_clean, axis = 0) + 3*np.std(rest_highF_feature_clean, axis = 0)

    # If we force to reject same amount of rest and mov data
    # index_keep_rest = np.all(rest_lowF_feature_clean < self.TH_lowF, axis = 1) & np.all(rest_highF_feature_clean < self.TH_highF, axis = 1)
    # index_keep_mov = np.all(mov_lowF_feature_clean < self.TH_lowF, axis = 1) & np.all(mov_highF_feature_clean < self.TH_highF, axis = 1)    
    # index_keep_common = np.where([index_keep_mov & index_keep_rest])[1]
    # windows_rejected_train_it2 = rest_lowF_feature_clean.shape[0] - len(index_keep_common)
    # total_windows_train_it2 = rest_lowF_feature_clean.shape[0] 
    # self.rest_features_train_clean = rest_features_clean1[index_keep_common,:]
    # self.mov_features_train_clean = mov_features_clean1[index_keep_common,:]

    # We don't force to reject same amount of rest and mov windows
    index_keep_rest = np.where([np.all(rest_lowF_feature_clean < TH_lowF, axis = 1) & np.all(rest_highF_feature_clean < TH_highF, axis = 1)])[1]
    index_keep_mov = np.where([np.all(mov_lowF_feature_clean < TH_lowF, axis = 1) & np.all(mov_highF_feature_clean < TH_highF, axis = 1)])[1]
    windows_rest_rejected_train_it2 = rest_lowF_feature_clean.shape[0] - len(index_keep_rest)
    windows_mov_rejected_train_it2 = mov_lowF_feature_clean.shape[0] - len(index_keep_mov)
    total_windows_rest_train_it2 = rest_lowF_feature_clean.shape[0] 
    total_windows_mov_train_it2 = mov_lowF_feature_clean.shape[0] 
    rest_features_train_clean = list()
    mov_features_train_clean = list()
    for i in range(len(features)):
        rest_features_train_clean.append(rest_features_clean1[i][index_keep_rest,:])
        mov_features_train_clean.append(mov_features_clean1[i][index_keep_mov,:])

    # In this case we don't replicate data
   
    features = list()
    labels = list()
    for i in range(len(rest_features_train_clean)):
        features.append(np.vstack([rest_features_train_clean[i],mov_features_train_clean[i]]))
        labels.append(np.vstack([np.zeros([len(rest_features_train_clean[0]),len(features[0][0,:])]), np.ones([len(mov_features_train_clean[0]),len(features[0][0,:])])]))

if artifact_rejection == True:
    try:
        print str(windows_rest_rejected_train_it1), " out of ", str(total_windows_train_it1), " rest windows rejected in the first iteration of the training set"
        print str(windows_mov_rejected_train_it1), " out of ", str(total_windows_train_it1), " mov windows rejected in the first iteration of the training set"
        print str(windows_rest_rejected_train_it2), " out of ", str(total_windows_rest_train_it2), " rest windows rejected in the second iteration of the training set"
        print str(windows_mov_rejected_train_it2), " out of ", str(total_windows_mov_train_it2), " mov windows rejected in the second iteration of the training set"
    except:
        pass        
# import os
# from scipy.io import savemat
# savemat(os.path.expandvars('$HOME/code/ismore/noninvasive/data_eeg.mat'), dict(eeg = eeg, labels = labels, features = features, ts_eeg = ts_eeg, rest_end_eeg = rest_end_idxs_eeg, mov_end_eeg = mov_end_idxs_eeg, rest_start_eeg = rest_start_idxs_eeg, mov_start_eeg = mov_start_idxs_eeg))    
# # compute r2 coeff for each channel and freq (using all the training datafiles)
# #import pdb; pdb.set_trace()
# savemat(os.path.expandvars('$HOME/code/ismore/noninvasive/only_eeg_power_5981.mat'), dict(eeg_power = eeg_power, ts_eeg = ts_eeg, rest_end_eeg = rest_end_idxs_eeg, mov_end_eeg = mov_end_idxs_eeg, rest_start_eeg = rest_start_idxs_eeg, mov_start_eeg = mov_start_idxs_eeg))    
r2 = np.zeros([len(channel_names_eeg),features[0].shape[1]])
for k in range(len(features)):
    for kk in range(features[k].shape[1]):
        r2[k,kk] = stats.pearsonr(labels[k][:,kk].ravel(), features[k][:,kk].ravel())[0]**2


# import os
# from scipy.io import savemat
# savemat(os.path.expandvars('$HOME/code/ismore/r2_values.mat'), dict(r2 = r2))  
 
    #r2[k] = r2_score(labels[k], features[k], multioutput = 'raw_values')
    #r2[k] = np.corrcoef(labels[k], features[k], 0)[-1,252:]**2
#import pdb; pdb.set_trace()
mov_trials = np.where(labels[8][:,0] == 1)
rest_trials = np.where(labels[8][:,0] == 0)
mov_feat_mean = np.mean(features[8][mov_trials[0],:],axis = 0)
rest_feat_mean = np.mean(features[8][rest_trials[0],:],axis = 0)
plt.figure('features')
plt.plot(rest_feat_mean, 'b'); plt.plot(mov_feat_mean, 'r'); plt.show(block = False)
# plt.figure()
# for i in np.arange(100):
#     plt.plot(features[12][i,:],'b');  plt.plot(features[12][-i-1,:],'r'); plt.show(block = False)

# plot image of r2 values (freqs x channels)
plt.figure()
plt.imshow(r2, interpolation = 'none')
plt.axis([1,50,0,31])
plt.yticks(np.arange(32),['1-FP1','2-FP2','3-F7','4-F3','5-Fz','6-F4','7-F8','8-FC5','9-FC1','10-FC2','11-FC6','12-T7','13-C3','14-Cz','15-C4','16-T8','17-TP9','18-CP5','19-CP1','20-CP2','21-CP6','22-TP10','23-P7','24-P3','25-Pz','26-P4','27-P8','28-PO9','29-O1','30-Oz','31-O2','32-PO10'])
#plt.yticks(np.arange(32),['1-FC3','2-FC1','3-FC2','4-FC4','5-C5','6-C3','7-C1','8-Cz','9-C2','10-C4','11-C6','12-CP3','13-CP1','14-CPz','15-CP2','16-CP4','17-FP1','18-FP2','19-F7','20-F3','21-Fz','22-F4','23-F8','24-T7','25-T8','26-P7','27-P3','28-Pz','29-P4','30-P8','31-O1','32-O2'])
plt.yticks(np.arange(32),['1-FC3','2-FC1','3-FC2','4-FC4','5-C5','6-C3','7-C1','8-Cz','9-C2','10-C4','11-C6','12-CP3','13-CP1','14-CPz','15-CP2','16-CP4','17-FP1','18-FP2','19-F7','20-F3','21-Fz','22-F4','23-F8','24-CP5','25-CP6','26-P7','27-P3','28-Pz','29-P4','30-P8','31-O1','32-O2'])
plt.xlabel('frequency (Hz)')
plt.ylabel('channels')
plt.colorbar()
plt.show(block = False)
#plt.grid(True)

import pdb; pdb.set_trace() 