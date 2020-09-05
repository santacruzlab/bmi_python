rh_vel_states = ['aa_vx', 'aa_vy', 'aa_vpsi', 'rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']
import matplotlib.pyplot as plt
from db import dbfunctions as dbfn
import numpy as np
import pickle
import scipy.stats

emg_decoder = pickle.load(open('/storage/decoders/emg_decoder_HUD1_4737_6988.pkl'))

emg14_bip = [
    'InterFirst',
    'AbdPolLo',
    'ExtCU',
    'ExtCarp',
    'ExtDig',
    'FlexDig',
    'FlexCarp',
    'PronTer',
    'Biceps',
    'Triceps',
    'FrontDelt',
    'MidDelt',
    'TeresMajor',
    'PectMajor',
]

emg_fts = []
for name in emg_decoder.feature_names:
    for ch in emg14_bip:
        emg_fts.append(ch+'_'+name)


def plot_fings(te_num, task_only=True):
    te = dbfn.TaskEntry(te_num)

    features = te.hdf.root.task[:]['emg_decoder_features_Z'].T

    if task_only is True:
        for i in range(0, 7):
            f, ax = plt.subplots()
            ax.plot(te.hdf.root.task[:]['target_pos'][:, i], label='targ')
            ax.plot(te.hdf.root.task[:]['plant_pos'][:, i], label='pos')
            ax.plot(te.hdf.root.task[:]['emg_vel_raw_scaled'][:, i], label='emg')
            ax.plot(te.hdf.root.task[:]['drive_velocity_raw_brain'][:, i], label='brain')
            ax.plot(te.hdf.root.task[:]['drift_correction'][:, i+7], label='drift')
            ax.set_title(rh_vel_states[i])
            plt.legend(loc=4)

        f, ax = plt.subplots(nrows=2)
        ax[0].plot(te.hdf.root.task[:]['emg_vel_raw_scaled'][:, 4], label='index')
        ax[0].plot(te.hdf.root.task[:]['emg_vel_raw_scaled'][:, 5], label='fing3')
        ax[0].plot([0, len(te.hdf.root.task)], [0, 0], 'k-')
        plt.legend()
        ax[1].plot(te.hdf.root.task[:]['drive_velocity_raw_brain'][:, 4], label='index')
        ax[1].plot(te.hdf.root.task[:]['drive_velocity_raw_brain'][:, 5], label='fing3')
        ax[1].plot([0, len(te.hdf.root.task)], [0, 0], 'k-')
        plt.legend(loc=4)


        
    if task_only is False:
        for i in range(0, 7):
            y = False
            f, ax = plt.subplots()
            ax2 = plt.subplot(2, 1, 1)
            state = rh_vel_states[i]
            ax2.plot(te.hdf.root.task[:]['emg_vel_raw_scaled'][:, i], color='k', label='emg')
            ax3 = plt.subplot(2, 1, 2)
            for j in range(len(emg_decoder.subset_features[state])):
                ft_ix = emg_decoder.subset_features[state][j]
                decoder_output = emg_decoder.beta[state][j].T.dot(features[ft_ix, :].reshape(-1,1).T)

                # if R**2 > 0.7? 
                si, ii, pi, ri, ei = scipy.stats.linregress(decoder_output, te.hdf.root.task[:]['emg_vel_raw_scaled'][:, i])
                if ri**2 > 0.1:
                    ax3.plot(decoder_output, label=emg_fts[ft_ix])
                    print state, ft_ix, emg_fts[ft_ix], decoder_output.shape
                    y = True
                elif np.abs(np.sum(decoder_output))/np.abs(np.sum(te.hdf.root.task[:]['emg_vel_raw_scaled'][:, i])) > 0.3:
                    ax3.plot(decoder_output, label=emg_fts[ft_ix])
                    y = True
                elif np.var(decoder_output) >0.3*np.var(te.hdf.root.task[:]['emg_vel_raw_scaled'][:, i]):
                    ax3.plot(decoder_output, label=emg_fts[ft_ix])
                    y = True
            ax3.set_title(rh_vel_states[i]+'_emgs')
            if y:
                plt.legend()
        # # Now plot contribution of muscles velocities 
        # f, ax = plt.subplots(nrows=2)

        # # EMG first:
        # for i in range(3, 7):
        #   ax[0].plot(te.hdf.root.task[:]['emg_vel_raw_scaled'][:, i], label=rh_vel_states[i-3])
        # plt.legend()

        # for i in range(3, 7):
        #   ax[1].plot(te.hdf.root.task[:]['drive_velocity_raw_brain'][:, i], label=rh_vel_states[i-3])
        # plt.legend()

        # ax[0].set_title('EMG Vels')
        # ax[1].set_title('Brain Vels')

    trial_type = te.hdf.root.task[:]['trial_type']
    tt = ''
    for i, t in enumerate(trial_type):
        if t != tt:
            print i, t
            tt = t
            

    # targs = np.unique(te.hdf.root.task[:]['trial_type'])
    # trial_starts_ts = [i['time'] for i in te.hdf.root.task_msgs[:] if i['msg'] == 'target']
    # trial_start_tt = te.hdf.root.task[trial_starts_ts]['trial_type']
    # trial_start_ti = te.hdf.root.task[trial_starts_ts]['target_index']
    # five_sec = 20*20
    # emg_decoder = pickle.load(open('/storage/decoders/emg_decoder_HUD1_4737_6988.pkl'))

    # for it, targ in enumerate(targs):
    #   ix = np.nonzero(trial_start_tt==targ)[0]
    #   if len(ix) > 0:
        
    #       f, ax = plt.subplots(nrows=1, ncols=2)

    #       for ti in range(2):
    #           ix2 = np.nonzero(trial_start_ti[ix] == ti)[0]
    #           if len(ix2) > 0:
    #               ix3 = ix[ix2]

    #               # Plot mean of jt vels for first 5 sec of target: 
    #               ax[0].set_title(targ)

    #               #for i, dof in enumerate(range(3, 6)):
    #               plant_pos = []
    #               targ_pos = []
    #               emg_vel_unsc = []
    #               brain_vel = []
    #               emg_vel_unsc_unz = []

    #               for j in ix3:
    #                   start = trial_starts_ts[j]
    #                   if (start + five_sec) < len(te.hdf.root.task):
    #                       #plant_pos.append(te.hdf.root.task[start:start+five_sec]['plant_pos'][:, dof])
    #                       #targ_pos.append(te.hdf.root.task[start:start+five_sec]['target_pos'][:, dof])
    #                       emg_vel_unsc.append(te.hdf.root.task[start:start+five_sec]['emg_decoder_features'][:, [3, 4, 5, 6]])
    #                       #emg_vel_unsc.append(te.hdf.root.task[start:start+five_sec]['emg_vel_raw'][:, dof])
    #                       #brain_vel.append(te.hdf.root.task[start:start+five_sec]['drive_velocity_raw_brain'][:, dof])
                            
    #                       # non_z = te.hdf.root.task[start:start+five_sec]['emg_decoder_features'][:, :]
    #                       # tmp = []
    #                       # for t in range(non_z.shape[0]):
    #                       #   v = emg_decoder(non_z[t, :])
    #                       #   tmp.append(v[dof])

    #                       # emg_vel_unsc_unz.append(np.hstack((tmp)))
    #               axi = ax[ti]
    #               #axi2 = ax[i, 1]
    #               #axi.plot(np.mean(np.vstack((plant_pos)), axis=0), label='plant_pos')
    #               #axi.plot(np.mean(np.vstack((targ_pos)), axis=0), label='targ_pos')
    #               try:
    #                   axi.plot(np.mean(np.dstack((emg_vel_unsc)), axis=2), label='emg_scaled')
    #               except:
    #                   pass
    #               #tmp = np.mean(np.vstack((emg_vel_unsc)), axis=0)
    #               #axi2.plot(tmp / np.std(tmp), label='emg_vel_unsc')
    #               #unsc_unz = np.mean(np.vstack((emg_vel_unsc_unz)), axis=0) 
    #               #unsc_unz = (unsc_unz - np.mean(unsc_unz))/np.std(unsc_unz)
    #               #axi2.plot(unsc_unz, label='emg_vel_unsc_unz')
    #               #axi.plot(np.mean(np.vstack((brain_vel)), axis=0), label='brain_vel')
    #               axi.plot([0, 300], [0, 0], 'k-', label='zero')
    #               axi.plot([300], [0], 'k*', label='15sec')
    #               axi.set_ylim([0, 80])
    #               plt.legend()
    #           else:
    #               print ix, ix2, trial_start_ti[ix]




