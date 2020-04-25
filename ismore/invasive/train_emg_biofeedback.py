import tables
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os, glob, tables
import pickle

from ismore import brainamp_channel_lists
from ismore.noninvasive import find_relevant_channels
from ismore.invasive import emg_decoder
from ismore.noninvasive.emg_feature_extraction import EMGMultiFeatureExtractor
from ismore.common_state_lists import *
from ismore import ismore_bmi_lib
from utils.constants import *
from db import dbfunctions as dbfn
from db.tracker import models

################################################################
# The variables below need to be set before running this script!
################################################################

# rad = 8

# # subj = 'nerea' 
# # baseline = 200
# # max_activity = 10000
# # subset_muscles = 'ExtCarp'
# #  mult = 1

# # baseline = 500
# # max_activity = 10000
# # subset_muscles = 'FlexCarp'
# # mult = -1

# subj = 'hud1'
# baseline = 300
# max_activity = 8000
# subset_muscles = 'FlexCarp'
# mult = -1

# subj = 'hud1'
# baseline = 2000
# max_activity = 10000
# subset_muscles = 'ExtCarp'
# suffx = '_2'
# mult = 1


# subj = 'hud1'
# baseline = 500
# max_activity = 12000
# subset_muscles = 'ExtCarp'
# suffx = '_3'
# mult = 1

# subj = 'hud1'
# baseline = 3000
# max_activity = 10000
# subset_muscles = 'ExtCarp'
# suffx = '_4'
# mult = 1

# subj = 'hud1'
# baseline = -1000
# max_activity = 2000
# subset_muscles = ['ExtCarp', 'FlexCarp']
# suffx = '_diff_1'
# mult = 1

def train_all(te_id, rad = 8):

    files = glob.glob('/storage/rawdata/hdf/*te'+str(te_id)+'.hdf')
    assert len(files) == 1

    hdf = tables.openFile(files[0])
    features = hdf.root.task[:]['zsc_emg_fts']
    ch = brainamp_channel_lists.emg14_bip_filt

    #for subset_muscles in [['ExtCarp_filt'], ['FlexCarp_filt'], ['ExtCarp_filt', 'FlexCarp_filt']]:
    for subset_muscles in [['ExtCarp_filt'], ['FlexDig_filt'], ['ExtCarp_filt', 'FlexDig_filt']]:

        subset_muscles_ix = [ch.index(subset_muscles[i]) for i in range(len(subset_muscles))]
        feature_names = ['WL']

        # Set 'win_len'
        win_len = 0.2  # secs

        # Set 'fs'
        fs = 1000  # Hz

        extractor_kwargs = {
            'feature_names':      feature_names,
            'win_len':            win_len,
            'fs':                 fs,
            'subset_muscles_ix':  subset_muscles_ix,
            'subset_muscles':     subset_muscles,
        }

        for d, (std_lims, name) in enumerate(zip([(-1, 1), (-1.5, 2)], ['easy', 'med'])):
            if len(subset_muscles) > 1:
                dist = features[:, subset_muscles_ix[0]] - features[:, subset_muscles_ix[1]]
                baseline = np.nanmean(dist) + std_lims[0]*np.nanstd(dist)
                max_activity = np.nanmean(dist) + std_lims[1]*np.nanstd(dist)
                decoder = emg_decoder.EMGBioFeedbackDiff(baseline, max_activity, rad, 1)
                namez = 'diff_'+name+'_te_id_'+str(te_id)
            
            else:
                if subset_muscles[0][0] == 'E':
                    dist = features[:, subset_muscles_ix[0]]
                    mult = 1
                elif subset_muscles[0][0] == 'F':
                    dist = features[:, subset_muscles_ix[0]]
                    mult = -1
                baseline = np.nanmean(dist) + std_lims[0]*np.nanstd(dist)
                max_activity = np.nanmean(dist) + std_lims[1]*np.nanstd(dist)
                decoder = emg_decoder.EMGBioFeedback(baseline, max_activity, rad, mult)
                namez = subset_muscles[0]+'_'+name+'_te_id_'+str(te_id)
            decoder.extractor_kwargs = extractor_kwargs 
            decoder_name = 'emgbiofeedback_decoder_%s' % namez
            pkl_name = decoder_name + '.pkl'
            decoder.decoder_name = decoder_name

            print baseline, max_activity, name, subset_muscles
            
            storage_dir = '/storage/decoders'
            pickle.dump(decoder, open(os.path.join(storage_dir, pkl_name), 'wb'))

            # Create a new database record for the decoder object if it doesn't already exist
            dfs = models.Decoder.objects.filter(name=decoder_name)
            if len(dfs) == 0:
                df = models.Decoder()
                df.path = pkl_name
                df.name = decoder_name
                df.entry = models.TaskEntry.objects.get(id=10000)
                df.save()

            elif len(dfs) == 1:
                pass # no new data base record needed
            elif len(dfs) > 1:
                print "More than one decoder with the same name! fix manually!"
