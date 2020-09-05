import pickle
import sys
from db import dbfunctions as dbfn
from db.tracker import models
import tables
import numpy as np
from scipy.signal import butter
from ismore.filter import Filter

def main():
    ''' Function that will execute the rest of this script when called from the command line.'''
    try:
        te_compliant = sys.argv[1]
        name_emg_decoder = sys.argv[2]
        print("Task entry compliant:", te_compliant, "Name of the EMG decoder:",name_emg_decoder)
        update_var(compliant_te=te_compliant,name_for_emg_decoder=name_emg_decoder)
    except IndexError as e:
        print("Could not update decoder variance: Not enough parameters supplied.")
        print("Use like: python update_fixed_var_from_compliant.py TE_COMPLIANT NAME_EMG_DECODER\n")
    except Exception, e:
        print(" --> Could not update variance!")
        raise

def update_var(compliant_te, te_for_grasp_class="", name_for_emg_decoder=""):
    # Method to update the 'recent_features_std_train' in grasp classifier
    # and recent_features_std in emg decoder using the EMG variance computed
    # from compliant movements

    ####### GET COMPLIANT FILES ######
    tsk = dbfn.TaskEntry(compliant_te)
    #supp_hdf = tables.openFile('/storage/supp_hdf/'+tsk.name+'.supp.hdf')
    supp_hdf = tables.open_file('/storage/supp_hdf/'+tsk.name+'.supp.hdf')
    

    ###### GET and SAVE EMG DECODER FTS ######
    if name_for_emg_decoder:
        emg_decoder_path = '/storage/decoders/'+name_for_emg_decoder+'.pkl'
        emg_decoder = pickle.load(open(emg_decoder_path))

        ### Only update if fixed var decoder, else it'll be updated in the task by 
        ### sliding window: 
        if emg_decoder.fixed_var_scalar:
            print ' Updating emg decoder: ',  emg_decoder_path
            print ' Using compliant features of task entry: ', compliant_te
            emg_decoder_fts = get_emg_features(supp_hdf, tsk.hdf, emg_decoder)

            ft_ix = {}
            std = np.zeros((emg_decoder_fts.shape[1]))
            for ft in emg_decoder.extractor_kwargs['feature_names']:
                ix = np.array([i for i, j in enumerate(emg_decoder.extractor_kwargs['emg_feature_name_list']) if ft in j])
                ft_ix[ft] = ix
                std[ix] = np.mean(np.std(emg_decoder_fts[:, ix], axis=0))

            emg_decoder.recent_features_mean = np.mean(emg_decoder_fts, axis=0)
            emg_decoder.recent_features_std = std
            pickle.dump(emg_decoder, open(emg_decoder_path, 'wb'))
    
    ###### GET and SAVE EMG CLASSIFER FTS ######
    if te_for_grasp_class:
        emg_id = dbfn.TaskEntry(te_for_grasp_class)
        emg_id_class = models.Decoder.objects.get(pk=emg_id.grasp_emg_classifier)
        emg_classifier_path = '/storage/decoders/'+emg_id_class.path
        emg_classifier = pickle.load(open(emg_classifier_path))

        if emg_classifier.scalar_fixed_var:
            print 'updating emg classifier ', emg_classifier_path
            print ' using compliant features: ', compliant_te
            emg_class_fts = get_emg_features(supp_hdf, tsk.hdf, emg_classifier)
            emg_class_fts = emg_class_fts[:, emg_classifier.extractor_kwargs['subset_muscles_ix']]
            emg_classifier.recent_features_mean_train = np.mean(emg_class_fts, axis=0)
            emg_classifier.recent_features_std_train = np.mean(np.std(emg_class_fts, axis=0))

            pickle.dump(emg_classifier, open(emg_classifier_path, 'wb'))

def get_emg_features(supp_hdf, hdf, emg_decoder):

    # Get timestamps of hdf rows: 
    ts = hdf.root.task[:]['ts'][:, 0]

    # Brainamp data: 
    ts_emg = supp_hdf.root.brainamp[:]['chanAbdPolLo']['ts_arrival']
    #ts_emg = (np.arange(0, len(supp_hdf.root.brainamp))*1./1000.) + t0
        
    # define extractor: 
    f_extractor = emg_decoder.extractor_cls(None, 
        emg_channels = emg_decoder.extractor_kwargs['emg_channels'], 
        feature_names = emg_decoder.extractor_kwargs['feature_names'], 
        win_len = emg_decoder.extractor_kwargs['win_len'], 
        fs=emg_decoder.extractor_kwargs['fs'])  
    win_len = emg_decoder.extractor_kwargs['win_len']

    ### Use filtered channels -- then extract features: 
    emg = []

    for chan in emg_decoder.extractor_kwargs['emg_channels']:
        
        # Get filtered channel: 
        try:
            data = supp_hdf.root.brainamp[:]['chan'+chan]['data']
        except:
            # Remove 'fitl'
            data = supp_hdf.root.brainamp[:]['chan'+chan[:-5]]['data']

        # Filter this! 
        filterbank = get_emg_filterbank(1)
        for filt in filterbank[0]:
            data = filt(data)

        emg.append(data[:, np.newaxis])

    # T x nchannels
    emg = np.hstack((emg))

    emg_fts = []

    # Now go through step by step and extract features: 
    for t1 in ts:
        
        # Take the last 'win_len' before this time point: 
        t0 = t1 - win_len

        # get indices: 
        ix = np.nonzero(np.logical_and(t0 < ts_emg, ts_emg <= t1))[0]

        # get features
        features = f_extractor.extract_features(emg[ix, :].T)

        emg_fts.append(features)

    return np.vstack((emg_fts))

def get_emg_filterbank(n_channels, fs=1000.):
    band  = [10, 450]  # Hz
    nyq   = 0.5 * fs
    low   = band[0] / nyq
    high  = band[1] / nyq
    bpf_coeffs = butter(4, [low, high], btype='band')


    # calculate coefficients for multiple 2nd-order notch filers
    notchf_coeffs = []
    for freq in [50, 150, 250, 350]:
        band  = [freq - 1, freq + 1]  # Hz
        nyq   = 0.5 * fs
        low   = band[0] / nyq
        high  = band[1] / nyq
        notchf_coeffs.append(butter(2, [low, high], btype='bandstop'))

    notch_filters = []
    for b, a in notchf_coeffs:
        notch_filters.append(Filter(b=b, a=a))

    channel_filterbank = [None]*n_channels
    for k in range(n_channels):
        filts = [Filter(bpf_coeffs[0], bpf_coeffs[1])]
        for b, a in notchf_coeffs:
            filts.append(Filter(b=b, a=a))
        channel_filterbank[k] = filts
    return channel_filterbank

# Call main function if script is called from the command line
if __name__ == "__main__":
    main()