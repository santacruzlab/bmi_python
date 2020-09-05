import tables
import pickle
import scipy.io
from db.tracker import models
from db import dbfunctions as dbfn
#from scipy.ndimage.filters import laplace
import nitime.algorithms as tsa
from scipy.signal import butter, lfilter
import numpy as np
import tables
try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
except:
    from sklearn.lda import LDA
import matplotlib.pyplot as plt
hdf_id = 8683

path_to_decoder = '/storage/decoders/eeg_decoder_DK_8678_8686_0.pkl'


# te = dbfn.TaskEntry(hdf_id)  
# hdf = te.hdf_supp
hdf_supp_name = '/storage/supp_hdf/dk20170310_20_te8683.supp.hdf'
hdf_supp = tables.open_file(hdf_supp_name) 
decoder = pickle.load(open(path_to_decoder))
t = scipy.io.loadmat('/home/tecnalia/code/ismore/noninvasive/t_vectors.mat')
t_limit = t. get('t_limit')

#scipy.io.loadmat('/home/tecnalia/code/ismore/noninvasive/t_vectors.mat', mdict = {'t_limit': t_limit, 't_final': t_final_eeg})
#
t_final = t. get('t_final_eeg') 

#
t_final = t_final - 1
channels = ['3', '8', '9', '10', '15']
neighbour_channels = { #define the neighbour channels for each channel (for the Laplacian filter)
    '9':  ['3', '8','10', '15'],
}
FB_channel = '9'
freq_bands = dict()
freq_bands['9'] = [[8,12]]
channels = ['chan' + c for c in channels]

eeg = hdf_supp.root.brainamp[:][channels]
#import pdb; pdb.set_trace()
fs = 1000
fs_down = 100
band  = [1, 48]  # Hz
nyq   = 0.5 * fs
low   = band[0] / nyq
high  = band[1] / nyq
bpf_coeffs = butter(2, [low, high], btype='band')

# Filter
for k in range(len(channels)): #for loop on number of electrodes
    eeg[channels[k]]['data'] = lfilter(bpf_coeffs[0],bpf_coeffs[1], eeg[channels[k]]['data']) 

for k, neighbour in enumerate(neighbour_channels): #for loop on number of electrodes
    samples_laplace = eeg['chan' + neighbour].copy()
    #import pdb; pdb.set_trace()
    for n in range(len(neighbour_channels[neighbour])):
        samples_laplace = np.vstack([samples_laplace, eeg['chan' + neighbour_channels[neighbour][n]]]) 
    eeg['chan' + neighbour]['data'] = samples_laplace[0,:]['data'] - np.mean(samples_laplace[1:,:]['data'], axis = 0)


eeg_FB_chan = eeg['chan' + FB_channel]['data'][t_limit[0] == 1]

# uncomment from  here to compute features
# Extract features
# features = None
# order = 20
# window = 500 # window used to extract features. In samples
# window_norm = 1238  # window used to normalize the signal. In samples
# n = 0
# while n <= (len(eeg_FB_chan) - window) and n <= (len(eeg_FB_chan) - window):
# 	samples = eeg_FB_chan[n:n+window]
# 	samples_down = samples[np.arange(0,len(samples),fs/fs_down)]

#     # nitime using levinson durbin method
# 	AR_coeffs_LD, sigma_v = tsa.autoregressive.AR_est_LD(samples_down, order)

# 	n_freqs = fs_down    
# 	norm_freqs, PSD = tsa.autoregressive.AR_psd (AR_coeffs_LD, sigma_v, n_freqs, sides = 'onesided')
#     #import pdb; pdb.set_trace()
# 	PSD = PSD[1:] #get rid of PSD in freq = 0
# 	PSD = np.log(PSD) #compute the Log of the PSD
# 	PSD_mean = np.mean(PSD[freq_bands[FB_channel][0][0]-1:freq_bands[FB_channel][0][1]])
#    	# if freq_bands != []:
# 		#import pdb; pdb.set_trace()
# 		# if type(freq_bands[FB_channel]) == list:# If more than one band is chosen per channel
# 		# 	for i in np.arange(len(freq_bands[FB_channel])):
#   #               #import pdb; pdb.set_trace()
# 		# 		try:
# 		# 			PSD_mean = np.hstack([PSD_mean, np.mean(PSD[freq_bands[FB_channel][0][0]-1:freq_bands[FB_channel][0][1]])])
# 		# 		except NameError:
# 		# 			PSD_mean = np.mean(PSD[freq_bands[FB_channel][0][0]-1:freq_bands[FB_channel][0][1]])
                   
# 		# else:
# 		# 	try:
# 		# 		PSD_mean = np.hstack([PSD_mean, np.mean(PSD[freq_bands[0]-1:freq_bands[1]])])
# 		# 	except NameError:
# 		# 		PSD_mean = np.mean(PSD[freq_bands[0]-1:freq_bands[1]])
#             #print freq_bands[0]
#         #import pdb; pdb.set_trace()
#         #print 'PSD', PSD
# 		# PSD = PSD_mean.copy()
# 	PSD = PSD_mean.copy()
# 	if features == None:
# 		features = PSD
# 	else:
		
# 		features = np.hstack([features, PSD])

# 	n += 1

# eeg_features = features[t_final[0][0:-1]-1]

# # normalization
# mean_feat = np.mean(eeg_features[0:window_norm])
# std_feat = np.std(eeg_features[0:window_norm])
# eeg_features_norm = (eeg_features - mean_feat)/std_feat
# eeg_features_short = eeg_features[window_norm:]
# #import pdb; pdb.set_trace()
# eeg_features_final = eeg_features_norm[window_norm:]

# uncomment until here to compute features
#import pdb; pdb.set_trace()
# save eeg features
#scipy.io.savemat('/home/tecnalia/code/ismore/noninvasive/eeg_features.mat', mdict = {'eeg_features_norm': eeg_features_norm, 'eeg_features_final': eeg_features_final, 'eeg_features': eeg_features, 'eeg_features_short':eeg_features_short})
feat = scipy.io.loadmat('/home/tecnalia/code/ismore/noninvasive/eeg_features.mat')
eeg_features_final = feat.get('eeg_features_final')
import pdb; pdb.set_trace()
# prediction
eeg_output = np.empty(len(eeg_features_final))
eeg_output_prob = np.empty(len(eeg_features_final))
for i in range(len(eeg_features_final)):
	eeg_output[i] = decoder.decoder.predict(eeg_features_final[i]) 
	#
	eeg_output_prob[i] = decoder.decoder.predict_proba(eeg_features_final[i])[0][1]
import pdb; pdb.set_trace()
scipy.io.savemat('/home/tecnalia/code/ismore/noninvasive/eeg_output_prob.mat', mdict = {'eeg_output_prob': eeg_output_prob })
# Apply filter to eeg output
prev_output = 0
eeg_filtered_output = np.ones(len(eeg_output))
eeg_filtered_output[0] = 0
consec_mov_outputs = 0
consec_rest_outputs = 0
for n in range(len(eeg_output)):

	if n != 0:
		prev_output = eeg_output[n-1]
		eeg_filtered_output[n] = eeg_filtered_output[n-1]
	if eeg_output[n] == 1 and prev_output == 1:
        # we need 5 consecutive outputs of the same type
		consec_mov_outputs +=1
        if consec_mov_outputs == 5 and eeg_filtered_output[n] == 0:
			consec_rest_outputs = 0
	elif eeg_output[n] == 1 and prev_output == 0:
		if eeg_filtered_output[n] == 1: #if it's moving
			consec_rest_outputs = 0
		else:
			consec_mov_outputs = 1
	elif eeg_output[n] == 0 and prev_output == 0:
		consec_rest_outputs +=1
		if consec_rest_outputs == 5 and eeg_filtered_output[n] == 1:
			consec_mov_outputs = 0
	elif eeg_output[n] == 0 and prev_output == 1:
		if eeg_filtered_output[n] == 1: #if it's moving
			consec_rest_outputs = 1
		else:
			consec_mov_outputs = 0
	if consec_mov_outputs >= 5:
		eeg_filtered_output[n] = 1
        #set all the velocities to a constant value towards the end point
	elif consec_rest_outputs >=5:
		eeg_filtered_output[n] = 0
plt.plot(eeg_filtered_output)
plt.show(block = False)

scipy.io.savemat('/home/tecnalia/code/ismore/noninvasive/eeg_output.mat', mdict = {'eeg_output': eeg_output, 'eeg_filtered_output': eeg_filtered_output })
import pdb; pdb.set_trace() 
# scipy.io.savemat('/home/tecnalia/code/ismore/noninvasive/eog_eeg.mat', mdict = {'eog': eog_all, 'eeg': eeg_all})