from ismore.noninvasive import eeg_feature_extraction
from ismore.noninvasive.eeg_decoding import LinearEEGDecoder
import pickle
import scipy
import tables
from db import dbfunctions as dbfn
import numpy as np
import copy
import matplotlib.pyplot as plt

def load_files(dec_file, hdf_file, dbname='default'):

    if dec_file is None:
        print 'error:files to be replayed should be determined'
        # #Files to be replayed (decoder and hdf files):
        # dec_file = '/storage/decoders/eeg_decoder_AM_4329_4331_0.pkl'
        # hdf_file = '/storage/rawdata/hdf/am20160111_02_te4988.hdf'
        #dbname = 'ibmi'

    #Load decoder and 
    dec = pickle.load(open(dec_file))
    hdf = tables.openFile(hdf_file)
    
    extractor_cls = dec.extractor_cls
    extractor_kwargs = dec.extractor_kwargs
    eeg_extractor = extractor_cls(source=None, **extractor_kwargs)

    #Modify 'get_samples' method of eeg_extractor to get samples from hdf file:
    funcType = type(eeg_extractor.get_samples)
    eeg_extractor.get_samples = funcType(get_samples_from_hdf_file, 
        eeg_extractor, LinearEEGDecoder)

    eeg_extractor.hdf = hdf
    eeg_extractor.max_task_loop = 0
    eeg_extractor.task_loop_cnt = -1

    return hdf, dec, eeg_extractor

#New 'get_samples()' function
def get_samples_from_hdf_file(self):

    #init time stamps if needed
    if not hasattr(self, 'task_ts'):
        self.brain_amp_ts = self.hdf.root.brainamp[:]['chan1']['ts_arrival']
        self.task_ts = np.squeeze(np.vstack((np.zeros((1,1)), self.hdf.root.task[:]['ts'])))
        self.hdf_chan_names = ['chan'+str(c) for c in self.channels]
        self.task_loop_cnt = 0
        self.max_task_loop = len(self.task_ts)-1

    ts_hi = self.task_ts[self.task_loop_cnt+1]
    ts_lo = self.task_ts[self.task_loop_cnt]

    brain_amp_ix = np.nonzero(scipy.logical_and(
            self.brain_amp_ts<ts_hi, self.brain_amp_ts>= ts_lo))[0]

    self.task_loop_cnt += 1

    samples = np.vstack(( [self.hdf.root.brainamp[brain_amp_ix][c]['data'] 
        for c in self.hdf_chan_names] ))
    #import pdb; pdb.set_trace()
    if samples.shape[1] <= 20:
        #add_zeros = 21 - samples.shape[1]
        #samples = np.hstack((samples, np.zeros((samples.shape[0], add_zeros))))
        samples = self.previous_samples
    self.previous_samples = samples.copy()
    #import pdb; pdb.set_trace()
    return samples

def rerun_task_with_hdf_data(hdf, eeg_extractor, decoder):

    #Initialize feature and decoder output lists:
    fts = []
    dec_output = []

    #Loop through task timestamps:
    #eeg_extractor.max_task_loop = len(np.squeeze(np.vstack((np.zeros((1,1)), hdf.root.task[:]['ts'])))) - 1
    
    while eeg_extractor.task_loop_cnt < eeg_extractor.max_task_loop:
        eeg_features = eeg_extractor()    
        #eeg_features = hdf.root.task[:]['eeg_features'][eeg_extractor.task_loop_cnt -1]
        print eeg_extractor.task_loop_cnt, eeg_features
        eeg_features = np.reshape(eeg_features,(1,eeg_features.shape[0]))
        fts.append(eeg_features)

        #Run Decoder:
        decoder_output = decoder(eeg_features)
        dec_output.append(decoder_output)

    return np.squeeze(np.vstack((fts))), np.squeeze(np.vstack((dec_output)))

def reconstruct_eeg_dec(dec_file, hdf_file, te_num, dbname='default'):
    hdf, decoder, eeg_extractor = load_files(dec_file, hdf_file, dbname=dbname)
    
    #Using HDF file for brainamp data -- construct decoder output using task 
    #decoder and task feature extractor:
    ft, dec_output = rerun_task_with_hdf_data(hdf, eeg_extractor, decoder)

    plt.plot(hdf.root.task[:]['decoder_output'], label='decoder_output')
    plt.plot(dec_output+.1, label='sim_decoder output')
    plt.legend()
    plt.show()


