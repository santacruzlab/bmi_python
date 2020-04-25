from db import dbfunctions as dbfn
from db.tracker import models
from riglib.bmi import train, extractor

import numpy as np
import tables, os, time
import glob, pickle
import matplotlib.pyplot as plt
import scipy.ndimage
from ismore.invasive.analysis import tuning_analysis
from ismore import ismore_bmi_lib
import trigger_decoder

velocity_bins = np.arange(-0.4, 0.5, 0.1)
velocity_bins_xy = np.arange(-3.5, 3.5, 0.5)
neural_bins = np.arange(0.5, 5.5, 1.)

def train_ridge_decoder(compliant_te_list, nsec=0.4, save=True, scale_factor=30.):
    # for i in compliant_te_list:
    #     te = TaskEntry.objects.get(pk=i)
    #     length, units = models.parse_blackrock_file(te.nev_file, te.nsx_files, te)
    ridge, units = train_test_trig(compliant_te_list[0], compliant_te_list, nsec=nsec, save=save)
    ridge_filt = trigger_decoder.RidgeFilt(ridge)
    ssm = ismore_bmi_lib.SSM_CLS_DICT['IsMore']()

    ridge_decoder = trigger_decoder.RidgeDecoder(ridge_filt, units, ssm, extractor.BinnedSpikeCountsExtractor,
        dict(units=units), binlen=nsec, n_subbins=1)
    ridge_decoder.filt.scale_factor = scale_factor
    
    if save: 
        #save decoder:
        decoder_name = 'decoder_phaseVII_%s_%s' % (str(compliant_te_list[0]), time.strftime('%Y%m%d_%H%M'))
        pkl_name = decoder_name + '.pkl'
        storage_dir = '/storage/decoders'

        ridge_decoder.path = os.path.join(storage_dir, pkl_name)
        ridge_decoder.decoder_name = decoder_name
        pickle.dump(ridge_decoder, open(os.path.join(storage_dir, pkl_name), 'wb'))

        # Create a new database record for the decoder object if it doesn't already exist
        dfs = models.Decoder.objects.filter(name=decoder_name)
        df = models.Decoder()
        df.path = pkl_name
        df.name = decoder_name
        df.entry = models.TaskEntry.objects.get(id=min(compliant_te_list))
        df.save()

    return ridge_decoder

def train_test_trig(compliant_te, compliant_list, nsec=0.2, select_motor_cells_w_M1=True,
    min_fr = 0., ridge_param=10000., filter_kin=True, save=False):
    ''' 

    Method to take in compliant movements datafiles
    (which are assumed to have a preparatory period), and 
    use these files to train a trigger decoder (ismore.invasive.trigger_decoder)

    Method uses Ridge regression to read out intended velocity from preparatory 
    activity

    '''

    # Take the first compliant movement to get 
    try:
        te = dbfn.TaskEntry(compliant_te)
        nev_hdf_fname = [j for j in comp_te.blackrock_filenames if '.nev.hdf' in j][0]
    except:
        nev_hdf_fname = glob.glob('/storage/rawdata/blackrock/*te'+str(compliant_te)+'.nev.hdf')[0]
        
    extractor_kwargs = dict(keep_zero_units=True)

    ### Unit selection based on optimal mutual information lag during compliant ###
    units, _ = tuning_analysis.get_cellnames_and_ts([compliant_te], cellnames=None, include_wf=False,
        noise_rejection=False)

    ### Get all non-sorted units ###
    motor_units = np.vstack(([u for u in units if u[1] != 10]))

    if select_motor_cells_w_M1:

        MI = get_mutual_info(compliant_te, min_fr=min_fr)

        ### Now get the average optimal lag for each unit: 
        motor_units = plot_MI(MI, compliant_te, save=save)
        import pdb; pdb.set_trace()
    # 1/ (Number of HDF rows per nsec)
    rows_to_bigrows = float(0.05/nsec)
    
    ## Initialize lists: 
    NF = []
    KF = []
    P = []
    add_pairs = 0

    for comp in compliant_list:
        pairs = []
        hdf_filename = glob.glob('/storage/rawdata/hdf/*te'+str(comp)+'.hdf')[0]
        nev_fname = glob.glob('/storage/rawdata/blackrock/*te'+str(comp)+'.nev')[0]
        nev_hdf_fname = glob.glob('/storage/rawdata/blackrock/*te'+str(comp)+'.nev.hdf')[0]
        files = dict(hdf=hdf_filename, blackrock=[nev_hdf_fname])

        # First get rows in the prep phase
        hdf_rows0 = get_paired_state_timestamps('prep', 
        'target', hdf_filename, nev_hdf_fname, return_hdf_rows=True)[1]

        # Then get rows during the target: 
        hdf_rows1 = get_paired_state_timestamps('target', 
        ['hold', 'timeout_penalty'], hdf_filename, nev_hdf_fname, return_hdf_rows=True)[1]

        # Now map the hdf_rows to time periods
        # Map beginning of target neural activity w/ middle of kinemactis: 

        # loop through the indices betwen prep and target: 
        for h, (hdf_pair0, hdf_pair1) in enumerate(zip(hdf_rows0, hdf_rows1)):

            # Take 1/4 of the way through the trial: 
            x2 = (( hdf_pair1[1] - hdf_pair1[0] ) *0.25) + hdf_pair1[0]
            x2 = x2*rows_to_bigrows

            # Take all preparatory bins: 
            x1_arr = np.arange(int(rows_to_bigrows*hdf_pair0[0]), int(rows_to_bigrows*hdf_pair0[1]))

            ## add all the prep phase data points: 
            for x1i in x1_arr:
                pairs.append([int(x1i), int(x2)])
                P.append([add_pairs+int(x1i), add_pairs+int(x2)])

        ### Get neural features and kinematics associated with this 
        ## get kinematic data
        files = dict(hdf=hdf_filename, blackrock=[nev_fname, nev_hdf_fname])
        tmask, rows = train._get_tmask(files, None, sys_name='task')
        kin = train.get_plant_pos_vel(files, nsec, tmask, 
            pos_key='plant_pos', vel_key=None, update_rate_hz=20.)

        extractor_kwargs = dict(keep_zero_units=True)
        neural_features, motor_units, extractor_kwargs = train.get_neural_features(files, nsec,
            extractor.BinnedSpikeCountsExtractor.extract_from_file, 
            extractor_kwargs, tslice=None, units=motor_units, source='task', strobe_rate=20.)

        # Remove 1st kinematic sample and last neural features sample to align the 
        # velocity with the neural features
        kin = kin[1:].T
        neural_features = neural_features[:-1].T
        
        if filter_kin:
            filts = get_filterbank(fs=1./nsec)
            kin_filt = np.zeros_like(kin)
            for chan in range(14):
                for filt in filts[chan]:
                    kin_filt[chan, :] = filt(kin[chan, :])

        ###     
        NF.append(neural_features)
        KF.append(kin_filt)

    NF = np.hstack((NF))
    KF = np.hstack((KF))

    from sklearn.linear_model import Ridge
    decoder = Ridge(ridge_param, fit_intercept=True, normalize=False)
    X = []
    Y = []

    for pear in P:
        X.append(NF[:, pear[0]])
        Y.append(KF[:, pear[1]])

    # Convert these hdf rows to 
    decoder.fit(np.vstack((X)), np.vstack((Y)))

    decoder.extractor_kwargs = dict(units=units)
    decoder.extractor_cls = extractor.BinnedSpikeCountsExtractor
    return decoder, motor_units

def get_mutual_info(Id, min_fr=5.):
    ''' Method to get mutual information plots'''

    te = dbfn.TaskEntry(Id)
    hdf_fname = te.hdf_filename
    nev_hdf_fname = [f for f in te.blackrock_filenames if '.nev.hdf' in f][0]

    ts_stamps, hdf_rows = get_paired_state_timestamps('instruct_trial_type', 
        ['hold', 'timeout_penalty'], hdf_fname, nev_hdf_fname, return_hdf_rows=True)

    units, _ = tuning_analysis.get_cellnames_and_ts([Id], cellnames=None, include_wf=False,
        noise_rejection=False)

    ### Get non-sorted units ###
    units = np.vstack(([u for u in units if u[1] != 10]))

    ### Get binned kin and neural activity: ###
    kin, neural_data = tuning_analysis.extract_neural_bins(hdf_fname, nev_hdf_fname, 0.05, units)

    nkins, T = kin.shape

    # And a tmask to ensure that the we only extract trials: 
    tmask = np.zeros((T))
    for pair in hdf_rows:
        for p in range(pair[0], pair[1]):
            tmask[p] = 1
    tmask_ix = np.nonzero(tmask==1)[0]

    neural_bins = np.arange(0.5, 5.5, 1.)

    MI = dict()
    for ic, unit in enumerate(units):
        if np.sum(neural_data[ic, :])/(0.05*len(neural_data[ic, :])) > min_fr:

            MI[tuple(unit)] = np.zeros((15, 14)) # Temoporal delays, DOFs
            for it, temp_offset in enumerate(np.arange(-7, 8)):
                
                # Modify the tmask_ix to shift neural activity
                ix = tmask_ix+temp_offset
                
                # Make sure no indices are < 0 or > len(neural_data):
                ix_safe = np.nonzero(np.logical_and(ix >= 0, ix < T))[0]
                                  
                # Get shifted neural data: 
                # for a temp_offset < 0, the neural data is shifted earlier
                # making the relationship more 'motor'
                neur = neural_data[ic, ix[ix_safe]]
                
                for k in range(7, 14):
                    kin_dof = kin[k, tmask_ix[ix_safe]]
                    
                    # Comput mutual information between kin_dof and neural data: 
                    # Bin the kinematics into the their bin: 
                    if k in [7, 8]:
                        vel_bin = velocity_bins_xy
                    else:
                        vel_bin = velocity_bins

                    binned_kin = np.digitize(kin_dof, vel_bin)
                    binned_neur = np.digitize(neur, neural_bins)
                
                    # This fcn calculates the emperical probability of 
                    MI[tuple(unit)][it, k] = compute_MI(binned_kin, binned_neur, 
                        velocity_bins=vel_bin, max_neur_cnts=6, nshuffles=0)
    return MI

def plot_MI(MI, compliant, save=False):
    # Get index, fing3, and pron/supp results: 
    # Plot Base X, Y, PSI vs. index, fing3, prono

    X = np.zeros((2, len(MI.keys()), 15))
    units = MI.keys()
    for iu, u in enumerate(units):
        # Now get the mean for base and hand
        X[0, iu, :] = np.mean(MI[u][:, 7+np.array([0, 1, 2])], axis=1) - np.mean(MI[u][:, 7+np.array([0, 1, 2])])
        X[1, iu, :] = np.mean(MI[u][:, 7+np.array([4, 5, 6])], axis=1) - np.mean(MI[u][:, 7+np.array([4, 5, 6])])

    # Now order the units by highest mutual info: 
    # ix0 = np.array([np.argmax(X[0, i, :]) for i in range(X.shape[1])]) # units showing mutual info to base DoFs kinematics
    print "Selecting units tuned to hand DoFs"
    ix0 = np.array([np.argmax(X[1, i, :]) for i in range(X.shape[1])])  # units showing mutual info to hand DoFs kinematics

    ix0_ = []
    for j in range(15): 
        ix0_.append(np.nonzero(ix0==j)[0])

    if save:
        f, ax = plt.subplots(ncols=2)
        ax[0].pcolormesh(0.05*np.arange(-7, 8), np.arange(X.shape[1]), X[0, np.hstack((ix0_)), :], vmin=0, vmax=0.004)
        cax = ax[1].pcolormesh(0.05*np.arange(-7, 8), np.arange(X.shape[1]), X[1, np.hstack((ix0_)), :], vmin=0, vmax=0.004)
        plt.colorbar(cax, ax=ax[1])
        ax[1].set_title('Hand (no thumb) MI')
        ax[0].set_title('Base MI')
        # f.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/Projects/SpainBMI/ismore_analysis/data/temp_data/Mut_Inf_Comp_'+str(compliant)+'.pdf')

    # Now get unit names you care about: 
    unit_ix = np.nonzero(ix0<=7)[0]
    return np.vstack((units))[unit_ix, :]

def compute_MI(binned_kin, neur, velocity_bins=velocity_bins, max_neur_cnts=5,
    nshuffles=20):

    #Compute mutual information: 
    MI = _MI(binned_kin, neur, 
        np.zeros((len(velocity_bins)+1, max_neur_cnts)))
    # Compute boot-strapped w/ shuffling x 20: 
    MI_shuffle = []
    for shuffle in range(nshuffles):
        perm_ix = np.random.permutation(len(neur))
        binned_kin_shuff = binned_kin[perm_ix]

        MI_shuffle.append(_MI(binned_kin_shuff, neur, 
            np.zeros((len(velocity_bins)+1, max_neur_cnts))))
    if len(MI_shuffle) == 0:
        return MI
    else:
        return MI - np.mean(MI_shuffle)

def _MI(binned_kin, neur, pvn):
    for i, (bk, bn) in enumerate(zip(binned_kin, neur)):
        pvn[int(bk), int(bn)] += 1

    ### 2D convolution to make pvn non-zero: 
    g = scipy.ndimage.filters.gaussian_filter(pvn, [0, 1.], 
        mode='nearest')

    g[g==0] = np.min(g)
    pvn = g / float(np.sum(g))

    pv = np.sum(pvn, axis=1)
    pv = pv / float(np.sum(pv))
    
    pn = np.sum(pvn, axis=0)
    pn = pn / float(np.sum(pn))

    MI = 0
    for iin, n in enumerate(pn):
        if n != 0:
            for iv, v in enumerate(pv):
                if v != 0 and pvn[iv, iin] != 0:
                    MI += pvn[iv, iin]*np.log2(pvn[iv, iin]/(v*n))
    return MI

def get_filterbank(n_channels=14, fs=1000.):
    from ismore.filter import Filter
    from scipy.signal import butter
    band  = [.001, 1]  # Hz
    nyq   = 0.5 * fs
    low   = band[0] / nyq
    high  = band[1] / nyq
    high = np.min([high, 0.99])
    bpf_coeffs = butter(4, [low, high], btype='band')

    channel_filterbank = [None]*n_channels
    for k in range(n_channels):
        filts = [Filter(bpf_coeffs[0], bpf_coeffs[1])]
        channel_filterbank[k] = filts
    return channel_filterbank

def get_paired_state_timestamps(statename_start, statename_end, hdf_fname, nev_hdf_fname, task_fs =20.,
    return_hdf_rows=False):

    hdf = tables.openFile(hdf_fname)
    task_msgs = hdf.root.task_msgs[:]
    msg_ix = np.array([i for i, j in enumerate(task_msgs[:]['msg']) if j in statename_end])
    
    msg_pairs = []
    for m in msg_ix:
        i = 1
        skip = False
        while task_msgs[m-i]['msg'] != statename_start:
            i += 1
            if task_msgs[m-i]['msg'] == 'wait':
                skip = True
                break
        if not skip:
            msg_pairs.append([task_msgs[m-i]['time'], task_msgs[m]['time']])
    try:
        msg_pairs = np.vstack((msg_pairs))
    except:
        msg_pairs = []

    # Get arduino timestamps from .NEV file 
    tmask, rows = train._get_tmask(files=dict(blackrock=[hdf_fname, nev_hdf_fname], ), tslice=None)

    # Get timestamps of msg_pairs: 
    ts_pairs = [[rows[m[0]], rows[m[1]]] for m in msg_pairs]

    if return_hdf_rows:
        return ts_pairs, msg_pairs
    else:
        return ts_pairs

