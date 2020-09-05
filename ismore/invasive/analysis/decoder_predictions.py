from riglib.bmi import train, extractor 
from db import dbfunctions as dbfn
from ismore import ismore_bmi_lib
from ismore import plants
from ismore.ismore_tests import test_train_decoder
import tables
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.stats
from ismore import common_state_lists
from scipy.signal import butter, lfilter
from ismore.filter import Filter
fs_synch = 20 #Frequency at which emg and kin data are synchronized
nyq   = 0.5 * fs_synch
cuttoff_freq  = 3 / nyq
bpf_kin_coeffs = butter(2, cuttoff_freq, btype='low')
n_dof             = 7
vel_filter = [None] * n_dof
for k in range(n_dof):
    vel_filter[k] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1])   
cmap_list = ['maroon', 'orangered', 'darkgoldenrod', 'olivedrab', 'teal', 'steelblue', 'midnightblue', 'darkmagenta']



def predict_from_VFB(vfb_te_num, plant_type='IsMore', units='all', tslice=None, zscore=False, **kwargs):
    '''
    Using a visual feedback entry, 
    a) train a decoder, 
    b) use the decoder to predict performance during the VFB task
    c) plot C matrix, K matrix
    '''

    use_kwarg_files = kwargs.pop('use_kwarg_files', False)
    if use_kwarg_files:
        nev_fname = [kwargs['nev']]
        hdf_fname = kwargs['hdf']
    else:
        task_entry = dbfn.TaskEntry(vfb_te_num)
        ## get kinematic data
        nev_fname = [j for i, j in enumerate(task_entry.blackrock_filenames) if j[-4:] == '.hdf']
        hdf_fname = task_entry.hdf_filename

    tmask, rows = train._get_tmask(dict(blackrock=nev_fname), tslice)
    kin = train.get_plant_pos_vel(dict(hdf=hdf_fname), 0.1, tmask, pos_key='plant_pos', vel_key=None, update_rate_hz=20)

    ## get neural features
    if units == 'all':
        units = []
        nev_hdf = tables.openFile(nev_fname[0][:-4]+'.hdf')
        for i in range(1, 97):
            try:
                ch = getattr(nev_hdf.root.channel, 'channel'+str(i).zfill(5))
                un = np.unique(ch.spike_set[:]['Unit'])
                for j in un:
                    units.append(np.array([i, j]))
            except:
                pass
        units = np.vstack((units))

    neural_features, units, extractor_kwargs = train.get_neural_features(dict(blackrock=[nev_fname[0][:-4]], hdf = hdf_fname), 0.1, extractor.BinnedSpikeCountsExtractor.extract_from_file, 
        dict(), tslice=tslice, units=units, source='task')

    # Remove 1st kinematic sample and last neural features sample to align the 
    # velocity with the neural features
    kin = kin[1:].T
    neural_features = neural_features[:-1].T
    ssm = ismore_bmi_lib.SSM_CLS_DICT[plant_type]()
    decoder = train.train_KFDecoder_abstract(ssm, kin, neural_features, units, 0.1, tslice=tslice, zscore=zscore)
    pos_pred, vel_pred = predict_activity_w_decoder(decoder, neural_features)
    
    ix = np.sum(ssm.state_order==0)

    pos_real = kin[:ix, :]
    vel_real = kin[ix:, :]

    #### PLOT PREDICTIONS ####
    for i in range(ix):
        f, ax = plt.subplots(ncols = 2)
        for j, (met_pred, met_real) in enumerate(zip([pos_pred, vel_pred], [pos_real, vel_real])):
            axi = ax[j]

            # Pos
            axi.plot(np.arange(len(met_pred))/20., met_pred[:, i])
            axi.plot(np.arange(len(met_pred))/20., met_real[i, :])
            _, _, rv, _, _ = scipy.stats.linregress(met_pred[:, i], met_real[i, :])
            axi.set_title(str(np.round((rv**2)*100)/100.) + ' param: '+ssm.state_names[j*ix + i])
            axi.set_xlabel('Seconds')
        plt.show()
 

def predict_xval_from_compliant(compliant_list_list, use_decoder = [], plot=False):

    R = {}
    NRMSE = {}
    for i in range(7):
        R[i] = []
        NRMSE[i] = []

    for ic, compliant_list in enumerate(compliant_list_list):
        if len(compliant_list) >= 4:

            if len(use_decoder) > 0:
                decoder_te = use_decoder[ic]
                te_dec = dbfn.TaskEntry(decoder_te)
                try:
                    decoder = te_dec.decoder
                    units = decoder.units
                except:
                    decoder = None
                    units = None
            else:
                units = None
                decoder = None

            try:
                decoder_new, K_test, N_test = test_train_decoder.test_train_decoder_from_te_list(compliant_list, None, 'test',
                cursor_or_ismore = 'ismore', cellnames=units, only_use_part=0.1, only_sorted_units=True)
                skip = False
            
                if decoder is None:
                    decoder = decoder_new
            except:
                print 'skipping: ', compliant_list
                skip = True            
            
            if not skip:
                vel_pred = []
                vel_pred_filt = []
                vel_real = []



                zero = decoder.filt.state.mean.copy()

                for i, te in enumerate(compliant_list):
                    decoder.filt.state.mean = zero
                    T, nunits= N_test[te].shape
                    vel_pred_te_filt = []

                    for t in range(T):
                        tmp = decoder(N_test[te][t, :])
                        v = decoder['qdot']
                        vel_pred.append(v)
                        vel_real.append(K_test[te][t, 7:])
                        
                        flt = []
                        for dof in range(7):
                            tmp = vel_filter[dof](v[dof])
                            flt.append(tmp)
                        vel_pred_te_filt.append(np.hstack((flt)))
                    vel_pred_filt.append(np.vstack((vel_pred_te_filt)))

                vel_real = np.vstack((vel_real))
                vel_pred = np.vstack((vel_pred))
                vel_pred_filt = np.vstack((vel_pred_filt))

                for dof in range(7):
                    if plot:
                        f, ax = plt.subplots()
                        ax.plot(vel_pred[:, dof])
                        ax.plot(vel_pred_filt[:, dof])
                        ax.plot(vel_real[:, dof])

                    # R-squared
                    R[dof].append(scipy.stats.pearsonr(vel_real[:, dof], vel_pred_filt[:, dof])[0])

                    # NRMSE: 
                    SE = np.sqrt(np.square(vel_pred_filt[:, dof] - vel_real[:, dof]))
                    nrmse = np.sum(SE) / float(len(SE) * (np.max(vel_real[:, dof]) - np.min(vel_real[:, dof])))
                    NRMSE[dof].append(nrmse)
    return R, NRMSE

def plot_R_and_NRMSE(R, NRMSE):
    is_vel = common_state_lists.ismore_vel_states
    f, ax = plt.subplots()
    for dof in range(7):
        ax.plot(R[dof], label=is_vel[dof], color=cmap_list[dof])
    ax.set_ylabel('Pearson R')
    ax.set_xlabel('Different Days of Training')
    ax.set_ylim([-.4, 0.8])
    plt.legend(fontsize=10)
    ax.set_title('R for DOFs, Train on 0.9 compliant , test on 0.1')


    f, ax = plt.subplots()
    for dof in range(7):
        ax.plot(NRMSE[dof], label=is_vel[dof],color=cmap_list[dof])
    ax.set_ylabel('NRMSE')
    ax.set_ylim([0, 1.2])
    ax.set_title('NRMSE for DOFs, Train on 0.9 compliant , test on 0.1')
    plt.legend(fontsize=10)
    ax.set_xlabel('Different Days of Training')


    

def predict_activity_w_decoder(decoder, neural_features):
    # plant = plants.IsMorePlantUDP()
    # plant.blocking_joints = None
    # plant.init()
    # plant.start()

    pos = []
    vel = []

    # Initialize decoder
    nunits, T = neural_features.shape
    for t in range(T):
        tmp = decoder(neural_features[:, t])
        v = decoder['qdot']
        p = decoder['q'] + 0.1*v
        pos.append(p)
        vel.append(v)
    return np.vstack((pos)), np.vstack((vel))

#def simulate_CLDA(decoder, neural_features):