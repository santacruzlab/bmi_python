import numpy as np
from db import dbfunctions as dbfn
import matplotlib.pyplot as plt

# Week 1
te_list1 = [7597, 7624, 7665, 7696, 7735, 7746, 7773, 7829, 7830, 7832]
# Week 2
te_list2 = [7853, 7888, 7944, 7946, 7990,8038, 8040]
clda_te_list = [7597, 7624, 7665, 7696, 7735, 7746, 7773, 7829, 7830, 7832]+[7853, 7888, 7944, 7946, 7990,8038, 8040]

# BMI List: 
# Week 1
te1 = [7599, 7600, 7601, 7625, 7628, 7666, 7670, 7672, 7740, 7742, 7778, 7781, 7833, 7834, 7835]
# Week 2
te2 = [7859, 7860, 7868, 7860, 7871, 7872, 7873, 7889, 7893, 7894, 7895, 7942, 7948, 7950, 7994, 7998, 8001, 8043, 8044]
te_list = te1 + te2

def plot_decoder_wts(te_list=te_list, jts=np.arange(7, 14)):
    T = 0
    f, ax = plt.subplots(nrows=len(jts))

    for te_ in te_list:
        te = dbfn.TaskEntry(te_)

        if 'clda' in te.task.name or 'CLDA' in te.task.name:
            clda = te.hdf.root.clda
            C = clda[:]['filt_C']

            for i, j in enumerate(jts):
                i_norm = np.linalg.norm(C[:, :, j], axis=1)
                ax[i].plot(T+np.arange(len(i_norm)), i_norm)
            T += len(i_norm)

def plot_kalman_gain(te_list=te_list, jts=np.arange(7, 14)):
    T = 0
    f, ax = plt.subplots(nrows=len(jts))
    f2, ax2 = plt.subplots(nrows=len(jts))
    for te_ in te_list:
        te = dbfn.TaskEntry(te_)
        F, K = te.decoder.filt.get_sskf()
        for i, j in enumerate(jts):
            i_norm = np.linalg.norm(K[j, :])
            ax[i].plot(T, i_norm, '.', markersize=10)

            v_norm = np.mean(np.abs(te.hdf.root.task[:]['plant_vel'][:, j-7]))
            ax2[i].plot(T, v_norm, '.', markersize=10)
        T += 1
    for i, j in enumerate(jts):
        ax[i].plot([len(te1), len(te1)], [0, 1], 'k-')
        ax2[i].plot([len(te1), len(te1)], [0, 1], 'k-')

def plot_neural_contribution(te_list=te_list, jts=np.arange(7, 14)):
    T = 0
    f, ax = plt.subplots(nrows=len(jts))
    
    for te_ in te_list:
        te = dbfn.TaskEntry(te_)
        F, K = te.decoder.filt.get_sskf()

        sc = te.hdf.root.task[:]['spike_counts'][:, :, 0]
        sc_bin = bin_sc(sc)

        kin = np.hstack((te.hdf.root.task[:]['plant_pos'], te.hdf.root.task[:]['plant_vel'], np.ones((len(te.hdf.root.task), 1)))) 
        kin_bin = kin[np.arange(1, len(te.hdf.root.task), 2), :]
        
        if te_ in te2:
            mFR = te.decoder.mFR
            sdFR = te.decoder.sdFR
            assert te.decoder.zscore == True
            sc_bin = (sc_bin - mFR[np.newaxis, :]) / sdFR[np.newaxis, :]

        nc = np.mat(K)*np.mat(sc_bin).T
        kc = np.mat(F)*np.mat(kin_bin).T

        for i, j in enumerate(jts):
            n_norm = np.linalg.norm(nc[j, :])
            k_norm = np.linalg.norm(kc[j, :])
            ax[i].plot(T, n_norm, '.', markersize=10)
            ax[i].plot(T, k_norm, '*', markersize=10)
        T += 1

    for i, j in enumerate(jts):
        ax[i].plot([len(te1), len(te1)], [0, 1], 'k-')

def plot_unit_contrib_to_neural(te_num, jts=np.array([7, 8])):
    te = dbfn.TaskEntry(te_num)
    F, K = te.decoder.filt.get_sskf()
    f, ax = plt.subplots(nrows=2)

    sc = te.hdf.root.task[:]['spike_counts'][:, :, 0]
    sc_bin = bin_sc(sc)
    _, n_units = sc_bin.shape

    if te_num in te2:
        mFR = te.decoder.mFR
        sdFR = te.decoder.sdFR
        assert te.decoder.zscore == True
        sc_bin = (sc_bin - mFR[np.newaxis, :]) / sdFR[np.newaxis, :]

    for j in jts:
        j_ = []
        j2_ = []

        for iu in range(n_units):
            nc = np.linalg.norm(K[j, iu]*sc_bin[:, iu])
            nc2 = np.mean(K[j, iu]*sc_bin[:, iu])
            j_.append(nc)
            j2_.append(nc2)
        ax[0].plot(np.hstack((j_)))
        ax[1].plot(np.hstack((j2_)))

def plot_mFR_sdFR_mismatch(te_nums=te2):
    f, ax = plt.subplots()
    kf, kax = plt.subplots(nrows=2)

    for it, te_num in enumerate(te_nums):
        if te_num in [7994, 7998, 8001]:
            col = 'r'
        elif te_num in [8043, 8044]:
            col = 'b'
        else:
            col = 'k'


        te = dbfn.TaskEntry(te_num)
    
        sc = te.hdf.root.task[:]['spike_counts'][:, :, 0]
        sc_bin = bin_sc(sc)
        _, n_units = sc_bin.shape

        assert te.decoder.zscore == True

        mFR = te.decoder.mFR
        sdFR = te.decoder.sdFR

        F, K = te.decoder.filt.get_sskf()

        zsc_mu = []
        for iu in np.arange(n_units):
            sci = sc_bin[:, iu]
            zmi = (np.mean(sci) - mFR[iu]) / sdFR[iu]
            kax[0].plot(iu, zmi*K[7, iu], '.', color=col)
            kax[1].plot(iu, zmi*K[8, iu], '.', color=col)
            
            zsc_mu.append(zmi)
        ax.plot(np.hstack((zsc_mu)), label=str(te_num), color=col)

    plt.show()
    plt.legend()
    return ax

def bin_sc(sc):
    T = sc.shape[0]

    sc_bin = np.zeros((int(T/2.), sc.shape[1]))
    for i, j in enumerate(np.arange(1, T, 2)):
        sc_bin[i, :] = np.sum(sc[j-1:j+1, :], axis=0)
    return sc_bin

def calc_kalman_gain(P, C, C_xpose_Q_inv):
    nX = P.shape[0]
    I = np.mat(np.eye(nX))
    D = C_xpose_Q_inv*C
    L = C_xpose_Q_inv
    K = P * (I - D*P*(I + D*P).I) * L
    return K

def get_sskf(A, W, C, Q, C_xpose_Q_inv, tol=1e-15):

    A, W, C, Q = np.mat(A), np.mat(W), np.mat(C), np.mat(Q)

    nS = A.shape[0]
    P = np.mat(np.zeros([nS, nS]))
    I = np.mat(np.eye(nS))

    D = C_xpose_Q_inv*C

    last_K = np.mat(np.ones(C.T.shape))*np.inf
    K = np.mat(np.ones(C.T.shape))*0

    K_hist = []

    iter_idx = 0
    last_P = None
    while np.linalg.norm(K-last_K) > tol and iter_idx < 5:
        P = A*P*A.T + W 
        last_K = K
        K = calc_kalman_gain(P, C, C_xpose_Q_inv)
        K_hist.append(K)
        KC = P*(I - D*P*(I + D*P).I)*D
        last_P = P
        P -= KC*P;
        iter_idx += 1
    
    n_state_vars, n_state_vars = A.shape
    F = (np.mat(np.eye(n_state_vars, n_state_vars)) - KC) * A
    return K

def plot_clda_wt_changes(te_entry_list=[8840, 8831, 8793, 8784]):
    '''
    For each CLDA session, compute the 
    change in weights over time. Summarize 
    these changes in a plot where each 
    session has one plot with 7 lines. 

    Each line starts at zero. 

    Deviation from the line corresponds to 
    norm diff in Kalman Gain from 
    initial decoder over time over all neurons.
    '''

    for te_entry in te_entry_list:
        te = dbfn.TaskEntry(te_entry)
        clda = te.hdf.root.clda
        T = len(clda)

        K0 = get_sskf(te.decoder.filt.A, te.decoder.filt.W, 
            te.decoder.filt.C, te.decoder.filt.Q, te.decoder.filt.C_xpose_Q_inv)

        filt_C = clda[:]['filt_C']
        filt_Q = clda[:]['filt_Q']
        filt_C_xpose_Q_inv = clda[:]['filt_C_xpose_Q_inv']

        dK_all = []
        amp = []
        f, ax = plt.subplots(nrows=2)

        for t in range(T):
            K = get_sskf(te.decoder.filt.A, te.decoder.filt.W, 
               filt_C[t, :, :], filt_Q[t, :, :], filt_C_xpose_Q_inv[t, :, :])

            dK_all.append(np.linalg.norm(K - K0, axis=1))
            amp.append(np.linalg.norm(K, axis=1))

        dK_all = np.vstack((dK_all))
        amp = np.vstack((amp))
        ax[0].plot(amp[:, [7, 8]])
        ax[0].set_ylim([0, 3])
        ax[1].plot(amp[:, range(9, 15)])
        ax[1].set_ylim([0, .1])
        ax[0].set_title(str(te_entry)+': '+te.task.name)
        import gc
        gc.collect()




