from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import butter, lfilter
from db import dbfunctions as dbfn
import numpy as np
import tables
import matplotlib.pyplot as plt
from ismore.noninvasive.emg_feature_extraction import EMGMultiFeatureExtractor
## Extract R, G, B trials from invasive and non-invasive phases, over many days

emg14_bip = ['InterFirst', 'AbdPolLo', 'ExtCU', 'ExtCarp', 'ExtDig', 'FlexDig',
    'FlexCarp', 'PronTer', 'Biceps', 'Triceps', 'FrontDelt', 'MidDelt', 
    'TeresMajor', 'PectMajor']

healthy_te = [4737,4738, 4742,4743]#,4746,4747]
#paretic_te = [4769,4770, 4773,4774]#,4777,4778]
paretic_te = [8453, 8454, 8455, 8456] # 10/26
#paretic_te = [8368, 8369, 8372, 8373, 8409]
healthy_te2 = [6967,6968,6971,6973]#,6974,6976,6979,6980,6982,6984,6987,6988]


nyq = 0.5 * 1000.
low = 10 / nyq
high = 450 / nyq
bfilter, afilter = butter(4, [low, high], btype='band')

notchf_coeffs = []
for freq in [50, 150, 250, 350]:
    band  = [freq - 1, freq + 1]  # Hz
    nyq   = 0.5 * 1000
    low   = band[0] / nyq
    high  = band[1] / nyq
    notchf_coeffs.append(butter(2, [low, high], btype='bandstop'))

targ_ix_dict = {}
targ_ix_dict['red', 0] = [[10, 11], [13], [5, 6]]
targ_ix_dict['red', 1] = [[9], [5, 6]] # 11
targ_ix_dict['green', 0] = [[10, 11], [5, 6]]
targ_ix_dict['green', 1] = [[11], [5, 6]]
targ_ix_dict['blue', 0] = [[9, 11], [5, 6]] # add 9
targ_ix_dict['blue', 1] = [[8, 11], [6]] # rm 5
targ_ix_dict['up', 0] = [[5, 6], [8]]
targ_ix_dict['up', 1] = [[2, 4], [5, 6]]
targ_ix_dict['down', 0] = [[2, 4], [5, 6], [7]] #
targ_ix_dict['down', 1] = [[2, 4], [5, 6], [8]]
targ_ix_dict['point', 0] = [[2, 4], [5, 6]] #
targ_ix_dict['point', 1] = [ [2, 4], [5, 6]] # add 2, 4
targ_ix_dict['grasp', 0] = [[1], [2, 4], [5, 6]]
targ_ix_dict['grasp', 1] = [[5, 6], [11]] #add 11

# norm = EMG_healthy_vs_paretic.get_norm_params(healthy_te)
# norm2 = EMG_healthy_vs_paretic.get_norm_params(healthy_te2)
# normp = EMG_healthy_vs_paretic.get_norm_params(paretic_te)
# targ_healthy, ba_healthy = EMG_healthy_vs_paretic.get_WL(healthy_te, norm)
# targ_healthy2, ba_healthy2 = EMG_healthy_vs_paretic.get_WL(healthy_te2, norm2)
# targ_paretic, ba_paretic = EMG_healthy_vs_paretic.get_WL(paretic_te, normp)


def get_norm_params(te_list):
    ''' Take all EMG from healthy and paretic datasets and use to compute norm and std of
    notch + BP filtered + WFL signal
    '''
    
    emg_data = []
    for t, te in enumerate(te_list):
        taskentry = dbfn.TaskEntry(te)
        wfl, emg, ts_wfl = get_filt_emg2(taskentry.name)
        emg_data.append(wfl)

    emg_data = np.vstack((emg_data))
    return np.mean(emg_data, axis=0)

def compare_EMG_by_targ(targ_healthy, ba_healthy, targ_paretic, ba_paretic, skip_first_n_sec=2):
    target_indices_h = np.array([int(float(t)) for t in targ_healthy[:, 1]])
    target_indices_p = np.array([int(float(t)) for t in targ_healthy[:, 1]])
    
    for targ in ['red', 'green', 'blue', 'up', 'down', 'point', 'grasp']:
        ih = np.nonzero(targ_healthy[:, 0]==targ)[0]
        ip = np.nonzero(targ_paretic[:, 0]==targ)[0]
        
        for r in range(2):
            ihh = np.nonzero(target_indices_h[ih]==r)[0]
            ipp = np.nonzero(target_indices_p[ip]==r)[0]

            b_h = []
            b_p = []
            for h in ihh:
                b_h.append(ba_healthy[ih[h]][skip_first_n_sec*20:, :])

            for p in ipp:
                b_p.append(ba_paretic[ip[p]][skip_first_n_sec*20:, :])

            b_h = np.vstack((b_h))
            ix = np.nonzero(~np.isnan(b_h))
            b_h = b_h[np.unique(ix[0]), :]
            ix1 = int(len(b_h)/2.)

            b_p = np.vstack((b_p))
            ix = np.nonzero(~np.isnan(b_p))
            b_p = b_p[np.unique(ix[0]), :]
        
            ix2 = int(len(b_p)/2.)

            X = np.vstack((b_h[:ix1, :], b_p[:ix2]))
            X2 = np.vstack((b_h[ix1:, :], b_p[ix2:]))
            X3 = np.vstack((b_h, b_p))
            
            y = np.hstack((np.zeros((ix1)), np.ones((ix2))))
            y2 = np.hstack((np.zeros((len(b_h)-ix1)), np.ones((len(b_p)-ix2))))
            y3 = np.hstack((np.zeros((len(b_h))), np.ones((len(b_p)))))
            
            clf = LinearDiscriminantAnalysis()

            clf.fit(X, y)
            est = clf.predict(X)
            clf2 = LinearDiscriminantAnalysis()
            clf2.fit(np.vstack((X, X2)), np.hstack((y, y2)))

            print targ, r, clf.score(X2, y2)
            plt.close('all')
            
            zh = (np.mean(b_h, axis=0) - clf2.xbar_) * np.squeeze(clf2.scalings_)
            zp = (np.mean(b_p, axis=0) - clf2.xbar_) * np.squeeze(clf2.scalings_)
            f, ax = plt.subplots(nrows=2)
            ax[0].plot(zh)
            ax[0].plot(zp)
            ax[1].plot(np.mean(b_h, axis=0))
            ax[1].plot(np.mean(b_p, axis=0))
            import pdb; pdb.set_trace()
            input('cont: ')

def define_healthy_vectors(targ_healthy, ba_healthy, targ_healthy2, ba_healthy2,
    skip_first_n_sec=2, targ_ix_dict=targ_ix_dict):

    vector_targ = {}

    for targ in ['red', 'green', 'blue', 'up', 'down', 'point', 'grasp']:
        ih = np.nonzero(targ_healthy[:, 0]==targ)[0]
        ih2 = np.nonzero(targ_healthy2[:, 0]==targ)[0]
        
        for r in range(2):
            ihh = np.nonzero(targ_healthy[ih, 1].astype(float)==r)[0]
            ihh2 = np.nonzero(targ_healthy2[ih2, 1].astype(float)==r)[0]

            b_h = []
            for h in ihh:
                b_h.append(ba_healthy[ih[h]][skip_first_n_sec*20:, :])
            for h2 in ihh2:
                b_h.append(ba_healthy2[ih2[h2]][skip_first_n_sec*20:, :])

            b_h = np.vstack((b_h))
            b_h_agg = []
            for g, grp in enumerate(targ_ix_dict[targ, r]):
                b_h_agg.append(np.mean(b_h[:, grp], axis=1)[:, np.newaxis])
            b_h_agg = np.hstack((b_h_agg))
            if len(b_h_agg.shape) < 2:
                b_h_agg = b_h_agg[:, np.newaxis]
            ix = np.nonzero(~np.isnan(b_h_agg))
            b_h_agg = b_h_agg[np.unique(ix[0]), :]
            vect = np.mean(b_h_agg, axis=0)
            vector_targ[targ, r] = vect
    return vector_targ

def test_vectors(vectors, targ_healthy, ba_healthy, targ_paretic, ba_paretic, targ_ix_dict,
    skip_first_n_sec=2):
    
    angs = {}

    for targ in ['red', 'green', 'blue', 'up', 'down', 'point', 'grasp']:
        ih = np.nonzero(targ_healthy[:, 0]==targ)[0]
        ip = np.nonzero(targ_paretic[:, 0]==targ)[0]
        
        for r in range(2):
            ihh = np.nonzero(targ_healthy[ih, 1].astype(float)==r)[0]
            ipp = np.nonzero(targ_paretic[ip, 1].astype(float)==r)[0]

            if len(vectors[targ, r]) > 1:
                vector_healthy = vectors[targ, r]/np.linalg.norm(vectors[targ, r])
                one_dim = False
            else:
                vector_healthy = vectors[targ, r]
                one_dim = True

            b_h = []
            b_p = []
            for h in ihh:
                b_h.append(ba_healthy[ih[h]][skip_first_n_sec*20:, :])

            for p in ipp:
                b_p.append(ba_paretic[ip[p]][skip_first_n_sec*20:, :])

            b_h = np.vstack((b_h))
            ix = np.nonzero(~np.isnan(b_h))
            b_h = b_h[np.unique(ix[0]), :]

            b_p = np.vstack((b_p))
            ix = np.nonzero(~np.isnan(b_p))
            b_p = b_p[np.unique(ix[0]), :]

            # Get angles: 
            ang_h = []
            ang_p = []

            for i in range(b_h.shape[0]):
                vect = b_h[i, :]
                g = []
                for grp in targ_ix_dict[targ, r]:
                    g.append(np.mean(vect[grp]))
                vect = np.hstack((g))
                if one_dim:
                    a = np.abs(vect - vector_healthy) / np.abs(vector_healthy)
                else:
                    vect = vect / np.linalg.norm(vect)
                    a  =np.arccos(np.clip(np.dot(vect, vector_healthy), -1.0, 1.0))
                ang_h.append(a)
            
            for i in range(b_p.shape[0]):
                g = []
                vect = b_p[i, :]
                for grp in targ_ix_dict[targ, r]:
                    g.append(np.mean(vect[grp]))
                vect = np.hstack((g))
                if one_dim:
                    a = np.abs(vect - vector_healthy) / np.abs(vector_healthy)
                else:
                    vect = vect / np.linalg.norm(vect)
                    a = np.arccos(np.clip(np.dot(vect, vector_healthy), -1.0, 1.0))
                ang_p.append(a)

            angs[targ, r, 'h'] = ang_h
            angs[targ, r, 'p'] = ang_p

            plt.close('all')
            plt.plot(ang_h)
            plt.plot(ang_p)
            import pdb; pdb.set_trace()

def get_WL(te_list, meanz):
    ba_all = {}
    targs = []
    tecnt = 0

    for h in te_list:
        te = dbfn.TaskEntry(h)
        hdf = te.hdf
        msgs = te.hdf.root.task_msgs
        task = te.hdf.root.task
        print np.unique(te.hdf.root.task[:]['trial_type'])

        if te.task.name == 'ismore_EXGEndPointMovement_testing':
            start_ix = np.nonzero(te.hdf.root.task_msgs[:]['msg']=='trial')[0]
            end_ix = start_ix + 1
        elif te.task.name == 'compliant_move':
            start_ix = np.nonzero(te.hdf.root.task_msgs[:]['msg']=='target')[0]
            end_ix = start_ix + 1

        wfl, emg, ts_wfl = get_filt_emg2(te.name)
        zwfl = wfl - meanz[np.newaxis, :]
        ba_all, targs_i, n = tsk_msgs_to_ba(start_ix, end_ix, hdf, emg, tecnt, ba_all, zwfl, ts_wfl)
        targs.append(targs_i)
        tecnt = tecnt+ n
    return np.vstack((targs)), ba_all

def get_filt_emg2(name):
    feature_names = ['WL']
    win_len = 1  # secs
    step_len = 0.02 # secs
    fs = 1000  # Hz
    n_win_pts = int(win_len * fs)
    step_pts = int(step_len * fs)

    extractor_kwargs = {
        'emg_channels':      emg14_bip,
        'feature_names':     feature_names,
        'win_len':           win_len,
        'fs':                fs,
        'source':           None}
    
    supp_hdf_f = '/storage/supp_hdf/'+name + '.supp.hdf'
    emg = tables.openFile(supp_hdf_f)
    ts = emg.root.brainamp[:]['chanBiceps']['ts_arrival']
    emg_proc = []
    
    for e in emg14_bip:
        emg_proc.append(emg.root.brainamp[:]['chan'+e]['data'])
    
    emg_proc = np.vstack((emg_proc)).T
    
    # Notch Filter
    for n in notchf_coeffs:
        emg_proc = lfilter(n[0], n[1], emg_proc, axis=0)

    # Bandpass filter
    trl_filt = lfilter(bfilter, afilter, emg_proc, axis=0)

    start_idxs = np.arange(0, trl_filt.shape[0] - n_win_pts + 1, step_pts)
    f_extractor = EMGMultiFeatureExtractor(**extractor_kwargs)
    features = np.zeros((len(start_idxs), f_extractor.n_features))
    ts_features = []
    for i, start_idx in enumerate(start_idxs):
        end_idx = start_idx + n_win_pts 
        samples = trl_filt[start_idx:end_idx, :]
        features[i, :] = f_extractor.extract_features(samples.T).T
        ts_features.append(np.mean(ts[start_idx:end_idx]))
    return features, emg, np.hstack((ts_features))

def get_filt_emg(name, norm_params, h_or_p):
    supp_hdf_f = '/storage/supp_hdf/'+name + '.supp.hdf'
    emg = tables.openFile(supp_hdf_f)
    emg_proc = []
    
    for e in emg14_bip:
        emg_proc.append(emg.root.brainamp[:]['chan'+e]['data'])
    
    emg_proc = np.vstack((emg_proc)).T
    
    # Notch Filter
    for n in notchf_coeffs:
        emg_proc = lfilter(n[0], n[1], emg_proc, axis=0)

    # Bandpass filter
    trl_filt = np.abs(lfilter(bfilter, afilter, emg_proc, axis=0))

    # Waveform length
    wfl_full = np.abs(np.diff(trl_filt, axis=0))

    # 5 Hz filter 
    # wfl_full = lfilter(blp, alp, wfl_, axis=0)

    # Normalize:
    wfl = []
    for e in range(14):
        tmp = wfl_full[:, e] - norm_params[h_or_p, 'mean'][e] #/norm_params[h_or_p, 'std'][e]
        wfl.append(tmp)
    wfl = np.vstack((wfl)).T
    return wfl, emg

def tsk_msgs_to_ba(msgs_start_ix, msgs_end_ix, hdf, emg, tecnt, ba_all, wfl, ts_wfl):
    task = hdf.root.task
    start_ts = hdf.root.task_msgs[msgs_start_ix]['time']
    end_ts = hdf.root.task_msgs[msgs_end_ix]['time']
    mid = np.round(np.mean(np.hstack((start_ts[:, np.newaxis], end_ts[:, np.newaxis])), axis=1)).astype(int)
    try:
        targs = np.hstack(( hdf.root.task[mid]['trial_type'][:, np.newaxis], 
            hdf.root.task[mid]['goal_idx']))
    except:
        targs = np.hstack(( hdf.root.task[mid]['trial_type'][:, np.newaxis], 
            hdf.root.task[mid]['target_index']))

    #ba_ts = emg.root.brainamp[:]['chanTriceps']['ts_arrival']
    ba_ts = ts_wfl
    task_epochs = np.hstack((start_ts[:, np.newaxis], end_ts[:, np.newaxis]))
    ts_epochs = {}

    for i, trl in enumerate(task_epochs):
        ts_epochs[i] = []
        ba_trl = []

        for t_ in np.arange(trl[0], trl[1]):
            ts_epochs[i].append(np.argmin(np.abs(ba_ts - task[t_]['ts'])))

        for it, t_ in enumerate(ts_epochs[i][:-1]):
            ba_trl.append(np.mean(wfl[t_:ts_epochs[i][it+1]], axis=0))

        ba_all[i+tecnt] = np.vstack((ba_trl))
    return ba_all, targs, task_epochs.shape[0]

def plot_mEMG_by_targ(targ_healthy, ba_healthy,
    targ_healthy2, ba_healthy2, targ_paretic, ba_paretic, skip_first_n_sec=2):

    target_indices_h = np.array([int(float(t)) for t in targ_healthy[:, 1]])
    target_indices_h2 = np.array([int(float(t)) for t in targ_healthy2[:, 1]])
    target_indices_p = np.array([int(float(t)) for t in targ_paretic[:, 1]])
    
    for targ in ['red', 'green', 'blue', 'up', 'down', 'point', 'grasp']:
    #for targ in [ 'point', 'grasp']:
        ih = np.nonzero(targ_healthy[:, 0]==targ)[0]
        ih2 = np.nonzero(targ_healthy2[:, 0]==targ)[0]
        ip = np.nonzero(targ_paretic[:, 0]==targ)[0]
        
        for r in range(2):
            ihh = np.nonzero(target_indices_h[ih]==r)[0]
            ihh2 = np.nonzero(target_indices_h2[ih2]==r)[0]
            ipp = np.nonzero(target_indices_p[ip]==r)[0]

            b_h = []
            b_h2 = []
            b_p = []

            for h in ihh:
                b_h.append(ba_healthy[ih[h]][skip_first_n_sec*20:, :])

            for h2 in ihh2:
                b_h2.append(ba_healthy2[ih2[h2]][skip_first_n_sec*20:, :])

            for p in ipp:
                b_p.append(ba_paretic[ip[p]][skip_first_n_sec*20:, :])

            b_h = np.vstack((b_h))
            ix = np.nonzero(~np.isnan(b_h))
            b_h = b_h[np.unique(ix[0]), :]

            b_h2 = np.vstack((b_h2))
            ix = np.nonzero(~np.isnan(b_h2))
            b_h2 = b_h2[np.unique(ix[0]), :]

            b_p = np.vstack((b_p))
            ix = np.nonzero(~np.isnan(b_p))
            b_p = b_p[np.unique(ix[0]), :]
        
            plt.close('all')
            
            f, ax = plt.subplots()
            ax.plot(np.mean(b_h, axis=0))
            ax.plot(np.mean(b_h2, axis=0))
            ax.plot(np.mean(b_p, axis=0))
            import pdb; pdb.set_trace()












