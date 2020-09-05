import numpy as np
from ismore.invasive.analysis import tuning_analysis
from db import dbfunctions as dbfn
import tables
from riglib.bmi import train, extractor

def test_train_decoder():

    import os
    path_arc = '/home/lab/preeya'
    path_local = os.path.expandvars('/Users/preeyakhanna/Dropbox/Carmena_Lab/SpainBMI')

    br_filename = path_local + '/test20151111_87_te1112.nev'
    hdf_filename = path_local + '/test20151111_87_te1112.hdf'

    #Note the list form for blackrock files; 
    files = dict(blackrock=[br_filename], hdf = hdf_filename)

    from riglib.bmi.extractor import BinnedSpikeCountsExtractor
    extractor_cls = BinnedSpikeCountsExtractor

    from db.tracker import models
    length, units = models.parse_blackrock_file(br_filename, None)

    units_new = []
    for u in units:
        units_new.append((u[0], u[1]+1))

    extractor_kwargs = dict(n_subbins = 1., units = units_new)

    import tables
    import ismore_bmi_lib

    hdf = tables.openFile(hdf_filename)
    ss = []
    if 'armassist' in hdf.root:
        ss.append(ismore_bmi_lib.StateSpaceArmAssist)
    if 'rehand' in hdf.root:
        ss.append(ismore_bmi_lib.StateSpaceReHand)
    if len(ss) > 1:
        ss = [ismore_bmi_lib.StateSpaceIsMore]

    tmp = ss[0]
    ssm = tmp()

    from riglib.bmi import train
    kin_extractor = train.get_plant_pos_vel

    update_rate = 0.1
    tslice = (0., length)

    decoder = train.train_KFDecoder(files, extractor_cls, extractor_kwargs, kin_extractor, ssm, np.array(units_new), 
        tslice=tslice, pos_key='plant_pos')

    return decoder

def test_train_cursor_decoder(te_num):

    import os
    import db.dbfunctions as dbfn
    te = dbfn.TaskEntry(te_num)
    br_filenames = te.blackrock_filenames
    hdf_filename = te.hdf_filename

    #Note the list form for blackrock files; 
    files = dict(blackrock=br_filenames, hdf = hdf_filename)

    from riglib.bmi.extractor import BinnedSpikeCountsExtractor
    extractor_cls = BinnedSpikeCountsExtractor

    from db.tracker import models
    br_filename = [x for x in br_filenames if x[-4:]=='.nev']
    length, units = models.parse_blackrock_file(br_filename[0], None, None)

    units_new = []
    for u in units:
        units_new.append((u[0], u[1]+1))

    extractor_kwargs = dict(n_subbins = 1., units = units_new)

    import tables
    hdf = tables.openFile(hdf_filename)



    from riglib.bmi import train
    kin_extractor = train.get_plant_pos_vel

    update_rate = 0.1
    tslice = (0., length)

    decoder = train.train_KFDecoder(files, extractor_cls, extractor_kwargs, kin_extractor, ssm, np.array(units_new), 
        tslice=tslice, pos_key='cursor')

    return decoder

def test_train_decoder_from_te_list(te_list, decoder_save_id, suffix,
    cursor_or_ismore = 'cursor', cellnames=None, update_rate=0.1, 
    tslice=None, zscore=False, kin_source='task', driftKF=False, 
    noise_rej = False, noise_rej_cutoff = 3.5*96):

    binlen = update_rate
    if cursor_or_ismore == 'cursor':
        from riglib.bmi.state_space_models import StateSpaceEndptVel2D
        ssm = StateSpaceEndptVel2D()
        pos_key = 'cursor'
    elif cursor_or_ismore in ['ismore', 'IsMore', 'Ismore']:
        from bmilist import bmi_state_space_models
        ssm = bmi_state_space_models['ISMORE']
        pos_key = 'plant_pos'

    ### Start OF GET NEURAL AND KIN FEATURES ###

    K = {}
    N = {}
    CIX = {}

    if cellnames is None:
        units, _ = tuning_analysis.get_cellnames_and_ts(te_list, None, skip_ts=True)

    else:
        units = cellnames

    for te in te_list:
        task_entry = dbfn.TaskEntry(te)
        files = dict(hdf=task_entry.hdf_filename, blackrock=task_entry.blackrock_filenames)

        #channel indices for this: 
        if type(task_entry.blackrock_filenames) is list:
            nev_hdf = [j for i,j in enumerate(task_entry.blackrock_filenames) if j[-8:] == '.nev.hdf'][0]
        elif task_entry.blackrock_filenames[-8:] == '.nev.hdf':
            nev_hdf = task_entry.blackrock_filenames
        else:
            raise Exception('Missing .nev.hdf file fpr %s'%te)
        hdf2 = tables.openFile(nev_hdf)

        # get unit indices for this te: 
        CIX[te] = []
        units_te = []
        for iu, u in enumerate(units):
            chan = 'channel'+str(u[0]).zfill(5)
            try:
                ss = getattr(hdf2.root.channel, chan)
                ix = np.nonzero(ss.spike_set[:]['Unit'] == u[1])[0]
                if len(ix) > 0:
                    CIX[te].append(iu)
                    units_te.append(u)
            except:
                pass

        tmask, rows = train._get_tmask(files, None, sys_name=kin_source)

        # Double Check if tmask oversteps HDF file len: 
        hdf_task_len = len(task_entry.hdf.root.task)
        if len(tmask) > hdf_task_len:
            tmask[-1] = False
            tslice = (rows[0]-.01, rows[-2]+.01)
        else:
            tslice = None
        kin = train.get_plant_pos_vel(files, binlen, tmask, update_rate_hz=20., pos_key=pos_key, vel_key=None)

        ## get neural features
        extractor_cls = extractor.BinnedSpikeCountsExtractor
        extractor_kwargs = dict(units=np.vstack((units_te)))

        neural_features, _, extractor_kwargs = train.get_neural_features(files, binlen, extractor_cls.extract_from_file, 
            extractor_kwargs, tslice=tslice, units=np.vstack((units_te)), source=kin_source)

        K[te] = kin

        neural_features2 = np.zeros((neural_features.shape[0], len(units)))
        for i, ii in enumerate(CIX[te]):
            neural_features2[:, i] = neural_features[:, i]
        
        N[te] = neural_features2

    # Combine Kin and Neural Features List: 
    K_master = []
    N_master = []

    for te in te_list:
        K_master.append(K[te][1:, :])
        N_master.append(N[te][1:, :])

    K_master = np.vstack((K_master))
    N_master = np.vstack((N_master))

    kin = K_master[1:, :].T
    neural_features = N_master[:-1, :].T

    ### END OF GET NEURAL AND KIN FEATURES ###
    ### END OF GET NEURAL AND KIN FEATURES ###
    ### END OF GET NEURAL AND KIN FEATURES ###
    kwargs = {}
    kwargs['driftKF'] = driftKF
    
    # INSERT HACK TO FIX KIN VS. NEURAL FEATURES SIZE MISMATCH
    # import pdb; pdb.set_trace()
    kwargs['noise_rej'] = noise_rej
    kwargs['noise_rej_cutoff'] = noise_rej_cutoff
    
    decoder = train.train_KFDecoder_abstract(ssm, kin, neural_features, np.vstack((units)), update_rate, 
        tslice=tslice, zscore=zscore, **kwargs)

    extractor_kwargs['units'] = np.vstack((units))
    decoder.extractor_cls = extractor_cls
    decoder.extractor_kwargs = extractor_kwargs

    from riglib.bmi import kfdecoder_fcns
    kfdecoder_fcns.save_new_dec(decoder_save_id, decoder, suffix)

    return decoder


from riglib import experiment
from ismore.invasive import bmi_ismoretasks
from features.hdf_features import SaveHDF
from ismore.ismore_tests import test_clda
import multiprocessing as mp

def test_adapting_state_inds():
    IsM, Base = mp.Pipe()

    kwargs=dict(assist_level_time=1., assist_level=(1.,0.),session_length=10.)
    task_cls = experiment.make(bmi_ismoretasks.SimBMIControl, [SaveHDF])
    pnm_base = mp.Process(target=test_train_decoder.run_clda, args=(kwargs, 'is_base_', task_cls, Base))
    pnm_base.start()

    pnm_base.join()

    pnm_f_base = IsM.recv()
    print 'Conenction received!'

    #pnm_f_base = '/Users/preeyakhanna/ismore/ismore_tests/sim_data/is_base_20160407_08_38_57.pkl'

    #Init an ismore decoder and encoder
    task_cls2 = experiment.make(bmi_ismoretasks.SimCLDAControl, [SaveHDF])
    kwargs_rh_clda=dict(assist_level_time=400., assist_level=(1.,0.),session_length=60.,
        half_life=(120., 120), half_life_time = 400., timeout_time=60., enc_path=pnm_f_base, 
        dec_path=pnm_f_base, clda_adapting_ssm='ReHand')   

    pnm_rh = mp.Process(target=run_clda, args=(kwargs_rh_clda, 'is_rh_clda_', task_cls2))
    pnm_rh.start()
    pnm_rh.join()


    #Init an ismore decoder and encoder
    kwargs_full_clda=dict(assist_level_time=400., assist_level=(1.,0.),session_length=60.,
        half_life=(120., 120), half_life_time = 400., timeout_time=60., enc_path=pnm_f_base, 
        dec_path=pnm_f_base)   

    pnm_full = mp.Process(target=run_clda, args=(kwargs_full_clda, 'is_full_clda_', task_cls2))
    pnm_full.start()
    pnm_full.join()

def run_clda(kwargs, nm_pref, task_cls, conn=None):
    #Init an ismore decoder and encoder
    targets = bmi_ismoretasks.SimBMIControl.ismore_simple(length=100)
    task_is = task_cls(targets, plant_type="IsMore", **kwargs)
    task_is.init()
    task_is.run()

    pnm_= test_clda.save_dec_enc(task_is, pref=nm_pref)
    if conn is not None:
        conn.send(pnm_)
        print 'conenction sent!'
    print pnm_
    return pnm_

import os, pickle
def compare_decoders(pnm_baseline, pnm_rh_clda, pnm_full_clda):
    path = os.path.expandvars('$ISMORE/ismore_tests/sim_data/')
    pnm_baseline = path + 'is_baseline_20160407_08_33_10.pkl'
    pnm_full_clda = path + 'is_full_clda_20160407_08_47_24.pkl'
    pnm_rh_clda = path + 'is_rh_clda_20160407_08_44_19.pkl'

    f, ax = plt.subplots(nrows=4, ncols= 3)
    d = []
    for fi, fname in enumerate([pnm_baseline, pnm_full_clda, pnm_rh_clda]):
        dec = pickle.load(open(fname))
        d.append(dec.corresp_dec.filt)
        axi = ax[0, fi]
        axi.pcolormesh(np.array(dec.corresp_dec.filt.C), vmin=0, vmax = 1)


