### EEG-EMG coactivation
import pickle
import os
from db import dbfunctions as dbfn
import tables
import numpy as np
import matplotlib.pyplot as plt

train_hdf_ids = [7500,7511] # comliant session Left Arm (Motor Learning study)
test_hdf_ids = [8221]#,8222,8223,8224] #hybrid left arm

#load classifier trained with train_hdf_ids
#CB
classifier_name = 'emg_classifier_CB_7500_7511_20170214_1636' #training with AbdPolLo and upper arm electrodes
#classifier_name = 'emg_classifier_CB_7500_7511_20170214_1524' #using only upper arm electrodes
classifier_name = 'emg_classifier_CB_7482_7541_20170221_1259' # using 2 compliant runs of 4 sessions

#DK
classifier_name = 'emg_classifier_DK_8280_8283_20170215_1359'
classifier_name = 'emg_classifier_DK_8282_8282_20170222_1810'
classifier_name = 'emg_classifier_DK_8281_8281_20170223_1104'
classifier_name = 'emg_classifier_DK_8294_8294_20170223_1156' #trained with data of a hybrid BCI session

storage_dir = '/storage/decoders'
pkl_name = classifier_name + '.pkl'
emg_classifier = pickle.load(open(os.path.join(storage_dir, pkl_name), 'rb'))
   
emg_channels_train = emg_classifier.emg_channels
emg_channels_train = emg_classifier.extractor_kwargs['channels_2train']


eeg_emg_coactivated_perc_mov     = []
eeg_emg_not_activated_perc_mov   = []
eeg_activated_emg_not_perc_mov   = []
emg_activated_eeg_not_perc_mov   = []

eeg_emg_coactivated_perc_rest    = []
eeg_emg_not_activated_perc_rest  = []
eeg_activated_emg_not_perc_rest  = []
emg_activated_eeg_not_perc_rest  = []



#load EEG and EMG data to run the mov detection classification : we only need the data stored at task freq

# hdf file
db_name = 'tubingen'
#CB
ids = [8218, 8221,8222,8223,8224]
ids = [8218]#,8221,8222,8223,8224]
#DK
ids = [8292, 8294,8295]
ids = [8295]#, 8294,8295]

for ind_id, id in enumerate(ids):
    te = dbfn.TaskEntry(id, dbname=db_name)
    te.close_hdf()
    hdf_name = te.hdf_filename
    hdf = tables.openFile(hdf_name)

    #hdf_supp file
    store_dir_supp = '/storage/supp_hdf/'
    index_slash = hdf_name.encode('ascii','ignore').rfind('/')            
    hdf_supp_name = store_dir_supp + hdf_name.encode('ascii','ignore')[index_slash + 1:-3] + 'supp.hdf'
    hdf_supp = tables.open_file(hdf_supp_name)      
                
    brainamp = hdf_supp.root.brainamp[:]

    emg_chan_all = hdf_supp.root.brainamp.colnames
    emg_chan_filt_idx = [ch_idx for ch_idx,ch_name in enumerate(emg_chan_all) if emg_chan_all[ch_idx].endswith('_filt')]
    emg_chan_filt_name = [ch_name for ch_idx,ch_name in enumerate(emg_chan_all) if emg_chan_all[ch_idx].endswith('_filt')]


    #look for last channel containing 'diag'. From that channel on, they are EEG channels we need to discard
    for chan_idx,chan_name in enumerate(emg_chan_filt_name):
    	#print "chan_name.find('diag') ", chan_name.find('diag')
    	if chan_name.find('diag')>-1:
    	 	last_EMG_chan_idx = chan_idx

    emg_decoding_ch_names = emg_chan_filt_name[0:last_EMG_chan_idx+1]
    
    
    #find chan indexes of the emg channels used for training the EMG classifier
    train_chan_index = []
    for chan_idx,chan_name in enumerate(emg_channels_train):
    	train_chan_index.append(emg_decoding_ch_names.index('chan'+chan_name))
    

    # find idx of feature used for training the classifier: WL is the 4th feature
    feature_names = ['MAV', 'VAR', 'WL', 'RMS', 'LOGVAR']
    used_feature = 'WL'
    ind_feature = feature_names.index(used_feature)

    emg_features = hdf.root.task[:]['emg_features']
    emg_features_Z = hdf.root.task[:]['emg_features_Z'] #normalized with mean and std calculated in sliding windows in testing data

    #take channels from training emg electrodes and selected feature. in emg_features, the first block of rows is the first featurexchannels, then second feature etc.
    ch_emg_feat_indexes = []
    for i in range(len(emg_channels_train)):
    	ch_emg_feat_idx = ind_feature*len(emg_decoding_ch_names) + train_chan_index[i]
     	ch_emg_feat_indexes.append(ch_emg_feat_idx)	
  

    emg_feat_Z_2classify = emg_features_Z[:,ch_emg_feat_indexes]

    # emg_feat_2classify = emg_features[:,ch_emg_feat_indexes] # if not normalized not working well
    # features_mean_train = emg_classifier.classifier_MovNoMov.features_mean_train
    # features_std_train = emg_classifier.classifier_MovNoMov.features_std_train
    # emg_feat_Z_2classify = (emg_feat_Z_2classify - features_mean_train)/features_std_train

    import pdb; pdb.set_trace()

    emg_mov_detec = emg_classifier.classifier_MovNoMov(emg_feat_Z_2classify)
    plt.plot(emg_feat_Z_2classify)
    plt.plot(emg_mov_detec)
    plt.show()
    
    #eeg_decoder_output = hdf.root.task[:]['decoder_output'] #memory error
    # eeg_state_decoder = hdf.root.task[:]['state_decoder']
    eeg_state_decoder = hdf.root.task[:]['decoder_output']

    #EEG-EMG activations
    eeg_activated_idx = np.where(eeg_state_decoder ==1)[0]
    emg_activated_idx = np.where(emg_mov_detec ==1)[0]

    eeg_not_activated_idx = np.where(eeg_state_decoder ==0)[0]
    emg_not_activated_idx = np.where(emg_mov_detec ==0)[0]

    eeg_emg_coactivated_idx 	= sorted(set(eeg_activated_idx) & set(emg_activated_idx))
    eeg_emg_not_activated_idx  	= sorted(set(eeg_not_activated_idx) & set(emg_not_activated_idx))
    eeg_activated_emg_not_idx 	= sorted(set(eeg_activated_idx) & set(emg_not_activated_idx))
    emg_activated_eeg_not_idx 	= sorted(set(emg_activated_idx) & set(eeg_not_activated_idx))

    total_num_points = len(emg_mov_detec)

    eeg_emg_coactivated_perc  	= (len(eeg_emg_coactivated_idx)*100) / total_num_points 
    eeg_emg_not_activated_perc  = (len(eeg_emg_not_activated_idx)*100) / total_num_points
    eeg_activated_emg_not_perc 	= (len(eeg_activated_emg_not_idx)*100) / total_num_points
    emg_activated_eeg_not_perc 	= (len(emg_activated_eeg_not_idx)*100) / total_num_points

    print "EEG and EMG coactivated : ", eeg_emg_coactivated_perc , " %"
    print "EEG and EMG NOT activated : ", eeg_emg_not_activated_perc , " %"
    print "EEG activated / EMG NOT activated : ", eeg_activated_emg_not_perc , " %"
    print "EEG NOT activated / EMG activated : ", emg_activated_eeg_not_perc , " %"


    #evaluate movement periods and rest periods separately
    task_msgs = hdf.root.task_msgs
    state_types = set(task_msgs[:]['msg'])
    task_state = task_msgs[:]['msg']
    task_state_idx = task_msgs[:]['time']

    mov_states = []
    no_mov_states = []
               
    for state in state_types:
        if state.startswith('trial'): #states where the subject was moving (we also consider that during the return instruction the subject was moving since tehre was not a rest period in between)
            mov_states.append(state)        
    no_mov_states = state_types-set(mov_states)

    mov_idx = []
    for state in mov_states:
        print 'state : ', state
        #look for indices         
        state_idx = [idx for (idx, msg) in enumerate(task_msgs[:]['msg']) if msg == state]
        print 'state_idx ', state_idx
        for trial_idx in state_idx:

            trial_idx_start = task_state_idx[trial_idx]
            trial_idx_end = task_state_idx[trial_idx+1]
        
            mov_idx = np.append(mov_idx,[range(trial_idx_start,trial_idx_end)])


    eeg_emg_coactivated_idx_mov	= sorted(set(eeg_emg_coactivated_idx) & set(mov_idx))    
    eeg_emg_not_activated_idx_mov  	= sorted(set(eeg_emg_not_activated_idx) & set(mov_idx))
    eeg_activated_emg_not_idx_mov 	= sorted(set(eeg_activated_emg_not_idx) & set(mov_idx))
    emg_activated_eeg_not_idx_mov 	= sorted(set(emg_activated_eeg_not_idx) & set(mov_idx))

    total_num_points_mov = len(mov_idx)

    # import pdb; pdb.set_trace()

    eeg_emg_coactivated_perc_mov.append((len(eeg_emg_coactivated_idx_mov)*100) / total_num_points_mov)
    eeg_emg_not_activated_perc_mov.append((len(eeg_emg_not_activated_idx_mov)*100) / total_num_points_mov)
    eeg_activated_emg_not_perc_mov.append((len(eeg_activated_emg_not_idx_mov)*100) / total_num_points_mov)
    emg_activated_eeg_not_perc_mov.append((len(emg_activated_eeg_not_idx_mov)*100) / total_num_points_mov)

    print "---- MOVEMENTS PERIODS ---- "
    print "EEG and EMG coactivated : ", eeg_emg_coactivated_perc_mov[ind_id] , " %"
    print "EEG and EMG NOT activated : ", eeg_emg_not_activated_perc_mov[ind_id] , " %"
    print "EEG activated / EMG NOT activated : ", eeg_activated_emg_not_perc_mov[ind_id] , " %"
    print "EEG NOT activated / EMG activated : ", emg_activated_eeg_not_perc_mov[ind_id] , " %"



    ## rest, preparation and instruction periods
    all_idx = range(len(emg_mov_detec))
    rest_idx = set(all_idx) - set(mov_idx)

    eeg_emg_coactivated_idx_rest		= sorted(set(eeg_emg_coactivated_idx) & set(rest_idx))
    eeg_emg_not_activated_idx_rest  	= sorted(set(eeg_emg_not_activated_idx) & set(rest_idx))
    eeg_activated_emg_not_idx_rest 	= sorted(set(eeg_activated_emg_not_idx) & set(rest_idx))
    emg_activated_eeg_not_idx_rest 	= sorted(set(emg_activated_eeg_not_idx) & set(rest_idx))

    total_num_points_rest = len(rest_idx)

    eeg_emg_coactivated_perc_rest.append((len(eeg_emg_coactivated_idx_rest)*100) / total_num_points_rest )
    eeg_emg_not_activated_perc_rest.append((len(eeg_emg_not_activated_idx_rest)*100) / total_num_points_rest)
    eeg_activated_emg_not_perc_rest.append((len(eeg_activated_emg_not_idx_rest)*100) / total_num_points_rest)
    emg_activated_eeg_not_perc_rest.append((len(emg_activated_eeg_not_idx_rest)*100) / total_num_points_rest)

    print "---- REST PERIODS ---- "
    print "EEG and EMG coactivated : ", eeg_emg_coactivated_perc_rest[ind_id] , " %"
    print "EEG and EMG NOT activated : ", eeg_emg_not_activated_perc_rest[ind_id] , " %"
    print "EEG activated / EMG NOT activated : ", eeg_activated_emg_not_perc_rest[ind_id] , " %"
    print "EEG NOT activated / EMG activated : ", emg_activated_eeg_not_perc_rest[ind_id] , " %"


    # import pdb; pdb.set_trace()


    




hdf.close()
hdf_supp.close()


