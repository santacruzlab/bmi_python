import ismore.invasive.bmi_ismoretasks as bmi_ismoretasks
from riglib import experiment
from features.hdf_features import SaveHDF
from features.arduino_features import PlexonSerialDIORowByte
from ismore.brainamp_features import SimBrainAmpData
import datetime
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

testing=False

if testing:
    full_session_length = 1
else:
    full_session_length = 100

def test_brainamp(plant='aa'):
    Task = experiment.make(bmi_ismoretasks.SimBMIControl, [SaveHDF, PlexonSerialDIORowByte, SimBrainAmpData])
    Task.pre_init()
    if plant=='is':
        targets = bmi_ismoretasks.SimBMIControl.ismore_simple(length=1)
        plant_type = 'IsMore'
    elif plant=='aa':
        targets = bmi_ismoretasks.SimBMIControl.armassist_simple(length=1)
        plant_type = 'ArmAssist'

    kwargs=dict(assist_level_time=400., assist_level=(1.,0.),session_length=10.,
        half_life=(20., 120), half_life_time = 400., timeout_time=60.)

    task = Task(targets, plant_type=plant_type, **kwargs)
    task.run_sync()
    return task

def arm_assist_main(plant='aa', conn=None):
    '''
    Summary: method to run ArmAssist only simulation
    Input param: : none
    Output param: returns task
    '''

    Task = experiment.make(bmi_ismoretasks.SimCLDAControl, [SaveHDF])

    if plant=='aa':
        targets = bmi_ismoretasks.SimBMIControl.armassist_simple(length=100)
        plant_type = 'ArmAssist'
    elif plant=='is':
        targets = bmi_ismoretasks.SimBMIControl.ismore_simple(length=100)
        plant_type = 'IsMore'


    kwargs=dict(assist_level_time=400., assist_level=(1.,0.),session_length=full_session_length,
        half_life=(20., 120), half_life_time = 400., timeout_time=60.)

    task = Task(targets, plant_type=plant_type, **kwargs)
    task.run_sync()
    fnm = save_dec_enc(task, pref='aa_')
    print fnm
    if conn is not None:
        conn.send(fnm)
        print 'Connection sent!'
    
    return fnm

def save_dec_enc(task, pref='enc_'):
    '''
    Summary: method to save encoder / decoder and hdf file information from task in sim_data folder
    Input param: task: task, output from arm_assist_main, or generally task object
    Input param: pref: prefix to saved file names (defaults to 'enc' for encoder)
    Output param: pkl file name used to save encoder/decoder
    '''
    enc = task.encoder
    task.decoder.save()
    enc.corresp_dec = task.decoder

    #Save task info
    import pickle
    ct = datetime.datetime.now()
    #pnm = '/Users/preeyakhanna/ismore/ismore_tests/sim_data/'+pref + ct.strftime("%Y%m%d_%H_%M_%S") + '.pkl'
    pnm = '/home/lab/code/ismore/ismore_tests/sim_data/'+pref + ct.strftime("%m%d%y_%H%M") + '.pkl'
    pnm2 = '/Users/preeyakhanna/code/ismore/ismore_tests/sim_data/'+pref + ct.strftime("%m%d%y_%H%M") + '.pkl'
    
    try:
        pickle.dump(enc, open(pnm,'wb'))
    except:
        pickle.dump(enc, open(pnm2, 'wb'))
        pnm = pnm2

    #Save HDF file
    new_hdf = pnm[:-4]+'.hdf'
    import shutil
    f = open(task.h5file.name)
    f.close()

    #Wait 
    import time
    time.sleep(1.)

    #Wait after HDF cleaned up
    task.cleanup_hdf()
    import time
    time.sleep(1.)

    #Copy temp file to actual desired location
    shutil.copy(task.h5file.name, new_hdf)
    f = open(new_hdf)
    f.close()

    #Return filename
    return pnm

def from_aa_to_full(fnm, option='no_clda'):
    '''
    Summary: Method to test performance of armassist decoder on generalization to ismore task
    Input param: : none
    Output param: none, two saved files sets -- 'aaxxxxx' and 'isxxxxxx'
    '''

    print 'received fname: ', fnm
    #Use same decoder / encoder on Ismore task (no clda)
    Task = experiment.make(bmi_ismoretasks.SimCLDAControl, [SaveHDF])
    targets = bmi_ismoretasks.SimBMIControl.ismore_simple(length=100)

    if option=='no_clda':
        kwargs=dict(assist_level_time=1., assist_level=(1.,0.),session_length=full_session_length,
            half_life=(20., 120), half_life_time = 1., timeout_time=60., enc_path=fnm, dec_path=fnm)

    elif option == 'rh_clda':
        kwargs=dict(assist_level_time=400., assist_level=(1.,0.),session_length=full_session_length,
            half_life=(20., 120), half_life_time = 400., timeout_time=60., enc_path=fnm, dec_path=fnm,
            clda_adapting_ssm='ReHand')        

    elif option == 'full_clda':
        kwargs=dict(assist_level_time=400., assist_level=(1.,0.),session_length=full_session_length,
            half_life=(20., 120), half_life_time = 400., timeout_time=60., enc_path=fnm, dec_path=fnm,
            clda_adapting_ssm='IsMore')

    else:
        Exception('clda option not yet implemented')

    task_is = Task(targets, plant_type="IsMore", **kwargs)

    task_is.run_sync()
    pnm = save_dec_enc(task_is, pref='is_')
    print pnm
    return pnm

def extract_trials(hdf):     
    '''
    Summary: method to extract trials (all types) from file and plot 
    them according to target pos and whether CLDA was on or not

    Input param: hdf file
    Output param: axes object for plot
    '''


    wait_ix = np.array([im for im, m in enumerate(hdf.root.task_msgs) if m[0] == 'wait'])
    end_ix = np.array([m[1] for im, m in enumerate(hdf.root.task_msgs[:-1]) if hdf.root.task_msgs[im+1][0]=='wait'])
    co_ix = np.array([hdf.root.task_msgs[m][1] for im, m in enumerate(wait_ix+4) if m+4 < len(hdf.root.task_msgs)])

    time_clda_done_arr = np.nonzero(hdf.root.task[:]['update_bmi'])[0]
    if len(time_clda_done_arr)>0:
        time_clda_done = time_clda_done_arr[-1]
    else:
        time_clda_done = 0
    targ_ix = get_target_ix(hdf.root.task[:]['target_pos'])


    f, ax = plt.subplots(nrows = len(np.unique(targ_ix)), ncols=2)

    for i, (w, e) in enumerate(zip(co_ix, end_ix)):
        tix = targ_ix[e-5]
        if e < time_clda_done:
            ax[tix, 0].plot(hdf.root.task[w:e]['plant_pos'][:,0], hdf.root.task[w:e]['plant_pos'][:,1], '.-')
        else:
            ax[tix, 1].plot(hdf.root.task[w:e]['plant_pos'][:,0], hdf.root.task[w:e]['plant_pos'][:,1], '.-')
    return ax

def get_target_ix(targ_pos):
    '''
    Summary: Helper function to get target index (arranged by increasing angle around circle)
    Input param: targ_pos: hdf.root.task[:]['target_pos']
    Output param: indices, same length as targ_pos
    '''

    #Target Index: 
    b = np.ascontiguousarray(targ_pos).view(np.dtype((np.void, targ_pos.dtype.itemsize * targ_pos.shape[1])))
    _, idx = np.unique(b, return_index=True)
    unique_targ = targ_pos[idx,:]

    #Order by theta: 
    theta = np.arctan2(unique_targ[:,1],unique_targ[:,0])
    thet_i = np.argsort(theta)
    unique_targ = unique_targ[thet_i, :]
    
    targ_ix = np.zeros((targ_pos.shape[0]), )
    ndim_targ = targ_pos.shape[1]
    for ig, targ_coord in enumerate(targ_pos):
        targ_ix[ig] = np.nonzero(np.sum(targ_pos[ig,:]==unique_targ, axis=1)==ndim_targ)[0]
    return targ_ix

def test_index_adaptation():
    full_session_length = 10
    fnm = '/Users/preeyakhanna/ismore/ismore_tests/sim_data/aa_20160414_09_53_38.pkl'

    kwargs=dict(assist_level_time=400., assist_level=(1.,0.),session_length=10.,
        half_life=(20., 120), half_life_time = 400., timeout_time=60., enc_path=fnm, dec_path=fnm,
        clda_adapting_ssm='ArmAssist', clda_stable_neurons='1a, 2a, 3a, 4a, 5a, 6a')


    targets = bmi_ismoretasks.SimBMIControl.armassist_simple(length=100)
    Task = experiment.make(bmi_ismoretasks.SimCLDAControl, [SaveHDF])

    task = Task(targets, plant_type="ArmAssist", **kwargs)
    task.run_sync()
    pnm = save_dec_enc(task_is, pref='aa_')
    hdf = tables.openFile(pnm[:-4]+'.hdf')
    b = hdf.root.clda[0]['filt_C']
    a = hdf.root.clda[200]['filt_C']
    a - b

if __name__ == "__main__":
    
    for clda_type in ['no_clda', 'rh_clda', 'full_clda']:
        print 'starting mp pipe'
        IsM, ArmA = mp.Pipe()

        aa_p = mp.Process(target=arm_assist_main, args=('aa', ArmA))
        print 'starting aa mp process'
        aa_p.start()

        aa_p.join()
        print 'aa process joined'
        fnm = IsM.recv()
        print 'Ism received'
        is_p = mp.Process(target=from_aa_to_full, args=(fnm, clda_type))
        is_p.start()
        print 'ism process started'
        is_p.join()
        print 'ism process joined'