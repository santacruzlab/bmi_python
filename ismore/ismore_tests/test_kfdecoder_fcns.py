'''
Test all functions in kfdecoder_fcns using SimBMIControl, SimCLDAControl
'''
import ismore.invasive.bmi_ismoretasks as bmi_ismoretasks
from riglib import experiment
from features.hdf_features import SaveHDF
from features.arduino_features import PlexonSerialDIORowByte
from ismore.brainamp_features import SimBrainAmpData
import datetime
import numpy as np
#import test_clda
import ismore.ismoretasks as ismoretasks

from ismore import ismore_bmi_lib
import tables
from riglib.bmi import train, kfdecoder_fcns
import pickle
import datetime

def train_decoder_simulation(desired_update_rate=.1, plant_type='ArmAssist', zscore_flag=False, return_hdf = False):
    # From Visual Feedback:
    Task = experiment.make(bmi_ismoretasks.SimBMIControl, [SaveHDF])
    if plant_type is 'ArmAssist':
        targets = bmi_ismoretasks.SimBMIControl.armassist_simple(length=100)
        ssm = ismore_bmi_lib.StateSpaceArmAssist()
        pref = 'aa_decoder'
    elif plant_type is 'ReHand':
        targets = bmi_ismoretasks.SimBMIControl.rehand_simple(length=100)
        ssm = ismore_bmi_lib.StateSpaceReHand()
        pref = 'rh_decoder'
    elif plant_type is 'IsMore':
        targets = bmi_ismoretasks.SimBMIControl.ismore_simple(length=100)
        ssm = ismore_bmi_lib.StateSpaceIsMore()
        pref = 'is_decoder'

    kwargs=dict(session_length=180., assist_level = (1., 1.), assist_level_time=60.,
        timeout_time=60.,)
    task = Task(targets, plant_type=plant_type, **kwargs)
    task.run_sync()
    fnm = test_clda.save_dec_enc(task)
    
    # Open HDF:
    hdf = tables.openFile(fnm[:-4]+'.hdf')

    # 20 Hz
    pos = hdf.root.task[:]['plant_pos']
    dt = 1/20.
    vel = np.diff(pos, axis=0)/dt
    pos = np.zeros_like(vel)
    kin = np.hstack((pos, vel)).T

    # Binned spikes:
    neural_features = hdf.root.task[:-1]['spike_counts'][:, :, 0].T
    units = np.array([[i, 1] for i in range(neural_features.shape[0])])
    update_rate = .05

    bin_nf, bin_kf, update_rate = kfdecoder_fcns.bin_(kin, neural_features, update_rate, desired_update_rate)
    
    # Neural features: 
    kf = train.train_KFDecoder_abstract(ssm, bin_kf[:, 2:], bin_nf[:, 2:], units, update_rate, 
        tslice=None, zscore=zscore_flag)
    kf.extractor_cls = {}
    kf.extractor_kwargs = {}
    kf.corresp_encoder = task.encoder
    
    # Train Decoder:
    ct = datetime.datetime.now()
    try:
        fname = '/home/lab/code/ismore/ismore_tests/sim_data/' + pref + ct.strftime("%m%d%y_%H%M") + '.pkl'
        pickle.dump(kf, open(fname, 'wb'))
    except:
        fname = '/Users/preeyakhanna/ismore/ismore_tests/sim_data/' + pref + ct.strftime("%m%d%y_%H%M") + '.pkl'
        pickle.dump(kf, open(fname, 'wb'))
    
    if return_hdf:
        return fname, fnm[:-4]+'.hdf'
    else:
        return fname

def ismore_clda_simulation(decoder_path, plant_type='ArmAssist'):
    '''
    Summary: method to run ArmAssist only simulation
    Input param: : none
    Output param: returns task
    '''

    Task = experiment.make(bmi_ismoretasks.SimCLDAControl, [SaveHDF])

    if plant_type=='ArmAssist':
        targets = bmi_ismoretasks.SimBMIControl.armassist_simple(length=100)
        pref = 'aa_'
    elif plant_type=='IsMore':
        targets = bmi_ismoretasks.SimBMIControl.ismore_simple(length=100)
        pref = 'is_'
    elif plant_type == 'ReHand':
        targets = bmi_ismoretasks.SimBMIControl.rehand_simple(length=100)
        pref = 'rh_'

    kwargs=dict(assist_level_time=30., assist_level=(1.,0.),session_length=60.,
        half_life=(450., 600), half_life_time = 400., timeout_time=60., decoder_path=decoder_path,
        regularizer = None, clda_adapting_ssm='ArmAssist', adapt_mFR_stats=False)

    task = Task(targets, plant_type=plant_type, **kwargs)
    task.run_sync()
    fnm = test_clda.save_dec_enc(task, pref=pref)
    decoder = task.decoder
    decoder.corresp_encoder = task.encoder

    ct = datetime.datetime.now()
    fname = '/home/lab/code/ismore/ismore_tests/sim_data/' + pref + ct.strftime("%m%d%y_%H%M") + '_decoder_clda.pkl'
    pickle.dump(decoder, open(fname, 'wb'))
    print 'clda decoder (pls corresp_encoder): ', fname
    print 'encoder: ', fnm
    print 'HDF: ', fnm[:-4]+'.hdf'

def ismore_bmi_simulation(decoder_path, decoder = None, plant_type='ArmAssist', safety_grid_name=None):
    Task = experiment.make(bmi_ismoretasks.SimBMIControl, [SaveHDF])

    if plant_type=='ArmAssist':
        targets = bmi_ismoretasks.SimBMIControl.armassist_simple(length=100)
        pref = 'aa_'
    elif plant_type=='IsMore':
        targets = bmi_ismoretasks.SimBMIControl.ismore_simple(length=100)
        pref = 'is_'
    elif plant_type == 'ReHand':
        targets = bmi_ismoretasks.SimBMIControl.rehand_simple(length=100)
        pref = 'rh_'

    kwargs=dict(assist_level_time=60., assist_level=(0., 0.), session_length=60.,
        timeout_time=60., decoder_path=decoder_path, safety_grid_name=safety_grid_name)

    task = Task(targets, plant_type=plant_type, **kwargs)
    task.run_sync()
    fnm = test_clda.save_dec_enc(task, pref=pref)
    print fnm

def test_adding_units(decoder_path):
    units = np.array([[100,1], [200, 1]])
    kfdecoder_fcns.add_rm_units(5813, units, 'add', True, 'ismore_test', **dict(decoder_path=decoder_path))

def test_subtract_units(decoder_path):
    units = np.array([[100,1], [200, 1]])
    kfdecoder_fcns.add_rm_units(5813, units, 'rm', False, 'ismore_test', **dict(decoder_path=decoder_path))

def test_zscore_substitution(decoder_path, hdf_path, te_id=5813):
    decoder, suffx = kfdecoder_fcns.zscore_units(None, None, pos_key = 'plant_pos', decoder_entry_id=None, 
        training_method=train.train_KFDecoder, retrain=False, **dict(decoder_path=decoder_path, hdf_path=hdf_path, te_id = te_id))
    pickle.dump(decoder, open(decoder_path[:-4]+suffx+'.pkl', 'wb'))
    print decoder_path[:-4]+suffx+'.pkl'