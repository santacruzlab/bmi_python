import tables
import pandas as pd
import numpy as np
import math
import pdb
import matplotlib.pyplot as plt
import sklearn
import copy

from sklearn.metrics import r2_score
from db import dbfunctions as dbfn
from scipy.signal import butter, lfilter, filtfilt
from scipy import stats
from riglib.filter import Filter
from ismore import brainamp_channel_lists, ismore_bmi_lib
from ismore.ismore_tests.eeg_feature_extraction import EEGMultiFeatureExtractor
from utils.constants import *

# Documentation
# Script to test the channel selection using a artificial signal (optimal signal)

## dataset
#hdf_ids = [3962,3964]
#hdf_ids = [4296, 4298, 4300, 4301]#4296, 4298, 4300, 4301 AI subject
#hdf_ids = [4329,4330,4331] # EL subject
hdf_ids = [4329]
channels_2visualize = brainamp_channel_lists.eeg32_filt #determine here the electrodes that we wanna visualize (either the filtered ones or the raw ones that will be filtered here)
#channels_2visualize = brainamp_channel_lists.emg14_filt 

#db_name = "default"
db_name = "tubingen"

hdf_names = []
for id in hdf_ids:
    te = dbfn.TaskEntry(id, dbname= db_name)
    hdf_names.append(te.hdf_filename)
    te.close_hdf()

# if '_filt' not in channels_2visualize[0]: #or any other type of configuration using raw signals
#     filt_training_data = True
# else:
#     filt_training_data = False
filt_training_data = True
# Set 'feature_names' to be a list containing the names of features to use
#   (see emg_feature_extraction.py for options)
feature_names = ['AR'] # choose here the feature names that will be used to train the decoder

#frequency_resolution = 3#Define the frequency bins of interest (the ones to be used to train the decoder)

# Set 'feature_fn_kwargs' to be a dictionary where:
#   key: name of a feature (e.g., 'ZC')
#   value: a dictionary of the keyword arguments to be passed into the feature
#          function (e.g., extract_ZC) corresponding to this feature
#   (see emg_feature_extraction.py for how the feature function definitions)
freq_bands = dict()
# freq_bands['13_filt'] = [] #list with the freq bands of interest
# freq_bands['14_filt'] = []#[[2,7],[9,16]]
# freq_bands['18_filt'] = []
# freq_bands['19_filt'] = []
feature_fn_kwargs = {
    'AR': {'freq_bands': freq_bands},  # choose here the feature names that will be used to train the decoder 
    # 'ZC':   {'threshold': 30},
    # 'SSC':  {'threshold': 700},
}

# neighbour_channels = { #define the neighbour channels for each channel (for the Laplacian filter)
#     '1':  [2,3,4],
#     '2':  [5,6],
#     '3':  [4,5],
# }

neighbour_channels = { #define the neighbour channels for each channel (for the Laplacian filter)
    '1_filt':  channels_2visualize,
    '2_filt':  channels_2visualize,
    '3_filt':  channels_2visualize,
    '4_filt':  channels_2visualize,
    '5_filt':  channels_2visualize,
    '6_filt':  channels_2visualize,
    '7_filt':  channels_2visualize,
    '8_filt':  ['3_filt', '4_filt','12_filt', '13_filt'],
    '9_filt':  ['4_filt', '5_filt','13_filt', '14_filt'],
    '10_filt':  ['5_filt', '6_filt','14_filt', '15_filt'],
    '11_filt':  ['6_filt', '7_filt','15_filt', '16_filt'],
    '12_filt':  channels_2visualize,
    '13_filt':  ['8_filt', '9_filt','18_filt', '19_filt'],
    '14_filt':  ['9_filt', '10_filt','19_filt', '20_filt'],
    '15_filt':  ['10_filt', '11_filt','20_filt', '21_filt'],
    '16_filt':  channels_2visualize,
    '17_filt':  channels_2visualize,
    '18_filt':  ['12_filt', '13_filt','23_filt', '24_filt'],
    '19_filt':  ['13_filt', '14_filt','24_filt', '25_filt'],
    '20_filt':  ['14_filt', '15_filt','25_filt', '26_filt'],
    '21_filt':  ['15_filt', '16_filt','26_filt', '27_filt'],
    '22_filt':  channels_2visualize,
    '23_filt':  channels_2visualize,
    '24_filt':  ['18_filt', '19_filt','28_filt'],
    '25_filt':  ['19_filt', '20_filt','28_filt', '30_filt'],
    '26_filt':  ['20_filt', '21_filt','30_filt'],
    '27_filt':  channels_2visualize,
    '28_filt':  channels_2visualize,
    '29_filt':  channels_2visualize,
    '30_filt':  channels_2visualize,
    '31_filt':  channels_2visualize,
    '32_filt':  channels_2visualize,
    }
channels_2train = ['13_filt','14_filt','18_filt','19_filt']
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

fs = 1000
channel_names = ['chan' + name for name in channels_2visualize]
neighbours = dict()
for chan_neighbour in neighbour_channels:#Add the ones used for the Laplacian filter here
    neighbours['chan' + chan_neighbour] = []
    for k,chans in enumerate(neighbour_channels[chan_neighbour]):
        new_channel = 'chan' + chans
        neighbours['chan' + chan_neighbour].append(new_channel)
neighbour_channels = copy.copy(neighbours)       
# calculate coefficients for a 4th-order Butterworth BPF from 10-450 Hz
band  = [0.5, 80]  # Hz
nyq   = 0.5 * fs
low   = band[0] / nyq
high  = band[1] / nyq
bpf_coeffs = butter(4, [low, high], btype='band')


band  = [48,52]  # Hz
nyq   = 0.5 * fs
low   = band[0] / nyq
high  = band[1] / nyq
notchf_coeffs = butter(2, [low, high], btype='bandstop')

extractor_cls = EEGMultiFeatureExtractor
f_extractor = extractor_cls(None, channels_2train = [], channels = channels_2visualize, feature_names = feature_names, feature_fn_kwargs = feature_fn_kwargs, fs=fs, neighbour_channels = neighbour_channels)

features = []
labels = []

for name in hdf_names:
    # load EMG data from HDF file
    hdf = tables.openFile(name)
            
    eeg = hdf.root.brainamp[:][channel_names]
    len_file = len(hdf.root.brainamp[:][channel_names[0]])
    # Parameters to generate artificial EEG data
    fsample = 1000.00 #Sample frequency in Hz
    t = np.arange(0, 10 , 1/fsample)#[0 :1/fsample:10-1/fsample]; #time frames of 10seg for each state
    time = np.arange(0, len_file/fsample, 1/fsample) #time vector for 5min
    f = 10 # in Hz
    rest_amp = 10
    move_amp = 5; #mov state amplitude 


    #steps to follow to visualize the r2 plots

    #1.- Filter the whole signal - only if non-filtered data is extracted from the source/HDF file.
    # BP and NOTCH-filters might or might not be applied here, depending on what data (raw or filt) we are reading from the hdf file
    cnt = 1
    cnt_noise = 1
    for k in range(len(channel_names)): #for loop on number of electrodes
        if channel_names[k] in ['chan8_filt', 'chan9_filt', 'chan13_filt', 'chan14_filt', 'chan18_filt', 'chan19_filt']:
            rest_noise = rest_amp*0.1*np.random.randn(len(t)) #10% of signal amplitude
            rest_signal = np.zeros(len(t))
            move_noise = move_amp*0.1*np.random.randn(len(t)) #10% of signal amplitude
            move_signal = np.zeros(len(t))
            for i in np.arange(len(t)):
                rest_signal[i] = rest_amp*cnt * math.sin((f+cnt-1)*2*math.pi*t[i]) + rest_noise[i] #rest sinusoidal signal
                move_signal[i] = move_amp*cnt * math.sin((f+cnt-1)*2*math.pi*t[i]) + move_noise[i]
            cnt += 1
            signal = []
            # label = []
            for i in np.arange(30):
                signal = np.hstack([signal, rest_signal, move_signal])
                # label = np.hstack([label, np.ones([len(rest_signal)]), np.zeros([len(move_signal)])])
            
        else:
            rest_signal = rest_amp*0.1*cnt_noise*np.random.randn(len(t)) #10% of signal amplitude. only noise 
            move_signal = rest_amp*0.1*cnt_noise*np.random.randn(len(t)) #10% of signal amplitude
            cnt_noise += 1
            signal = []
            # label = []
            for i in np.arange(30):
                signal = np.hstack([signal, rest_signal, move_signal])
        
        
        eeg[channel_names[k]]['data'] = signal[:len_file].copy()        
        eeg[channel_names[k]]['data'] = lfilter(bpf_coeffs[0],bpf_coeffs[1], eeg[channel_names[k]]['data']) 
        eeg[channel_names[k]]['data'] = lfilter(notchf_coeffs[0],notchf_coeffs[1], eeg[channel_names[k]]['data']) 
       

    # Laplacian filter -  this has to be applied always, independently of using raw or filt data
    #import pdb; pdb.set_trace()
    # from scipy.io import savemat
    # import os
    # savemat(os.path.expandvars('$HOME/code/ismore/filtered_eeg15.mat'), dict(filtered_data = eeg['chan15']['data']))
    # savemat(os.path.expandvars('$HOME/code/ismore/filtered_eeg7.mat'), dict(filtered_data = eeg['chan7']['data']))
    #import pdb; pdb.set_trace()
    eeg = f_extractor.Laplacian_filter(eeg)
    # import os
    # savemat(os.path.expandvars('$HOME/code/ismore/filtered_Laplace_eeg15.mat'), dict(filtered_data = eeg['chan15']['data']))
    # savemat(os.path.expandvars('$HOME/code/ismore/filtered_Laplace_eeg7.mat'), dict(filtered_data = eeg['chan7']['data']))
    #import pdb; pdb.set_trace()
    #2.- Break done the signal into trial intervals (concatenate relax and right trials as they happen in the exp)

    
    
    rest_start_idxs_eeg = np.arange(0,len(time), fsample *20)
    mov_start_idxs_eeg = np.arange(10*1000,len(time), fsample *20)
    rest_end_idxs_eeg = np.arange(10*1000-1,len(time), fsample *20)
    mov_end_idxs_eeg = np.arange(20*1000-1,len(time), fsample *20)
    
    if len(mov_end_idxs_eeg) < len(rest_end_idxs_eeg):
        rest_end_idxs_eeg = rest_end_idxs_eeg[:len(mov_end_idxs_eeg)]
        rest_start_idxs_eeg = rest_start_idxs_eeg[:len(mov_end_idxs_eeg)]
        mov_start_idxs_eeg = mov_start_idxs_eeg[:len(mov_end_idxs_eeg)]

    for chan_n, k in enumerate(channel_names): 
        r_features_ch = None
        m_features_ch = None
        # k = 'chan8_filt'
        # chan_n = 7
        found_index = k.find('n') + 1
        chan_freq = k[found_index:]
        #import pdb; pdb.set_trace()

        for idx in range(len(rest_start_idxs_eeg)):
            # mirar si es mejor sacar los features en todos los channels a la vez o channel por channel!
            #rest_window = eeg[k][rest_start_times_eeg[idx]:rest_end_times_eeg[idx]]['data']
            #import pdb; pdb.set_trace()
            rest_window = eeg[k][rest_start_idxs_eeg[idx]:rest_end_idxs_eeg[idx]+1]['data']
            mov_window = eeg[k][mov_start_idxs_eeg[idx]:mov_end_idxs_eeg[idx]+1]['data']
            # import pdb; pdb.set_trace()
            #3.- Take windows of 500ms every 50ms (i.e. overlap of 450ms)
            #4.- Extract features (AR-psd) of each of these windows 
            n = 0
            while n <= (len(rest_window) - 500) and n <= (len(mov_window) - 500):
                # if k == 'chan8_filt':
                #     import pdb; pdb.set_trace()
                r_feats = f_extractor.extract_features(rest_window[n:n+500],chan_freq)                        
                m_feats = f_extractor.extract_features(mov_window[n:n+500],chan_freq)
                # if k == 'chan8_filt':
                # if chan_n == 7:
                #     import pdb; pdb.set_trace()
                if r_features_ch == None:
                    r_features_ch = r_feats.copy()
                    m_features_ch = m_feats.copy()
                else:
                    r_features_ch = np.vstack([r_features_ch, r_feats])
                    m_features_ch = np.vstack([m_features_ch, m_feats])
                n +=50
            # if chan_n == 12:
                
            #     mean_PSD_mov =  np.mean(m_features_ch, axis = 0)
            #     mean_PSD_rest =  np.mean(r_features_ch, axis = 0)
                #plt.figure(); plt.plot(r_features_ch.T); plt.show()
                #import pdb; pdb.set_trace()
                # import os
                # savemat(os.path.expandvars('$HOME/code/ismore/all_psds.mat'), dict(r_features_ch = r_features_ch, m_features_ch = m_features_ch))
                #plt.figure(); plt.plot(mean_PSD_mov); plt.plot(mean_PSD_rest,color ='red'); plt.show()
        # if chan_n == 12:
        #     import pdb; pdb.set_trace()
        if  len(features) < len(channel_names):
            features.append(np.vstack([r_features_ch, m_features_ch]))
            labels.append(np.vstack([np.zeros(r_features_ch.shape), np.ones(m_features_ch.shape)]))
        else:
            features[chan_n] = np.vstack([features[chan_n],np.vstack([r_features_ch, m_features_ch])])
            labels[chan_n] = np.vstack([labels[chan_n],np.vstack([np.zeros(r_features_ch.shape), np.ones(m_features_ch.shape)])])
# import os
# from scipy.io import savemat
# savemat(os.path.expandvars('$HOME/code/ismore/labels_features_eegall.mat'), dict(labels = labels, features = features))    
# compute r2 coeff for each channel and freq (using all the training datafiles)
#import pdb; pdb.set_trace()

r2 = np.zeros([len(channel_names),features[0].shape[1]])
for k in range(len(features)):
    for kk in range(features[k].shape[1]):
        r2[k,kk] = stats.pearsonr(labels[k][:,kk], features[k][:,kk])[0]**2

plt.figure('features-ch 8')
for i in np.arange(100):
    plt.plot(features[7][i,:],'b');  plt.plot(features[7][-i-1,:],'r'); plt.show(block = False)

plt.figure('features-ch 9')
for i in np.arange(100):
    plt.plot(features[8][i,:],'b');  plt.plot(features[8][-i-1,:],'r'); plt.show(block = False)

plt.figure('features-ch 10')
for i in np.arange(100):
    plt.plot(features[9][i,:],'b');  plt.plot(features[9][-i-1,:],'r'); plt.show(block = False)

mov_trials = np.where(labels[12][:,0] == 1)
rest_trials = np.where(labels[12][:,0] == 0)
mov_feat_mean = np.mean(features[12][mov_trials[0],:],axis = 0)
rest_feat_mean = np.mean(features[12][rest_trials[0],:],axis = 0)
plt.figure('mean features- ch 13')
plt.plot(rest_feat_mean); plt.plot(mov_feat_mean, 'r'); plt.show(block = False)

#plt.hold(True);

# import os
# from scipy.io import savemat
# savemat(os.path.expandvars('$HOME/code/ismore/r2_values.mat'), dict(r2 = r2))  
# import pdb; pdb.set_trace()  
    #r2[k] = r2_score(labels[k], features[k], multioutput = 'raw_values')
    #r2[k] = np.corrcoef(labels[k], features[k], 0)[-1,252:]**2

# plot image of r2 values (freqs x channels)
plt.figure()
plt.imshow(r2, interpolation = 'none')
plt.axis([0,50,0,31])
plt.yticks(np.arange(32),['1-FP1','2-FP2','3-F7','4-F3','5-Fz','6-F4','7-F8','8-FC5','9-FC1','10-FC2','11-FC6','12-T7','13-C3','14-Cz','15-C4','16-T8','17-TP9','18-CP5','19-CP1','20-CP2','21-CP6','22-TP10','23-P7','24-P3','25-Pz','26-P4','27-P8','28-PO3','29-POz','30-PO4','31-O1','32-O2'])
plt.xlabel('frequency (Hz)')
plt.ylabel('channels')
plt.colorbar()
plt.show(block = False)
#plt.grid(True)

import pdb; pdb.set_trace()

plt.figure('feat')
plt.plot(PSD,'b')
plt.show(block = False)

plt.figure('win')
plt.plot(rest_window[n:n+500],'g')
plt.figure('feat')
plt.plot(r_feats,'g')
plt.show(block = False)

plt.figure('win')
plt.plot(mov_window[n:n+500],'g')
plt.figure('feat')
plt.plot(m_feats,'g')
plt.show(block = False)