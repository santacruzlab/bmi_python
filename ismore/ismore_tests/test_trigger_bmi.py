import ismore.invasive.bmi_ismoretasks as bmi_ismoretasks
from riglib import experiment
from ismore import ismoretasks
from features.hdf_features import SaveHDF
from ismore.brainamp_features import SimBrainAmpData
import pickle
from ismore.invasive import discrete_movs_emg_classification 
import time, shutil
import numpy as np
import subprocess

#['test_20180529_1526.hdf', 'test_20180529_1528.hdf', 'test_20180529_1536.hdf', 'test_20180529_1538.hdf',
'test_20180529_1542.hdf','test_20180529_1548.hdf', 'test_20180529_1550']
#[0, 0.01, 0.05, 0.1, 0.4, .75, 1.0]

class SimNeurData(object):
    def __init__(self):
        self.sim = True
        self.t0 = time.time()
        self.dtype = np.dtype([("ts", np.float), ("chan", np.int32), ("unit", np.int32), ("arrival_ts", np.float64)])
    
    def get(self):
        return np.array([time.time() - self.t0, np.random.randint(1, 71), 1, time.time()], dtype=self.dtype)


def main(rh_levels=[0.0, 0.01, 0.05, 0.1, 0.4, 0.75, 1.0]):
    fnames = []
    for rh in rh_levels:
        fnames.append(test(rh_assist_level_i=rh))
    return fnames, rh_levels

def test(rh_assist_level_i):
    full_session_length = 60.
    targets = ismoretasks.IsMoreBase.blk_B1_grasp()
    Task = experiment.make(bmi_ismoretasks.Hybrid_GraspClass_w_RestEMG_PhaseV, [SaveHDF, SimBrainAmpData])    
    kwargs=dict(aa_assist_level=(0.1,0.1), rh_assist_level=(rh_assist_level_i, rh_assist_level_i), 
        session_length=full_session_length,
        emg_decoder=pickle.load(open('/storage/decoders/emg_decoder_HUD1_4746_6984.pkl')),
        rest_mov_emg_classifier=pickle.load(open('/storage/decoders/emg_classifier_HUD1_13158_13158_20180524_1845_scalar_var_False.pkl')),
        decoder = pickle.load(open('/storage/decoders/trigger_decoder_phaseV_13802_20180529_1445.pkl')),
        #decoder = pickle.load(open('/Users/preeyakhanna/Downloads/refollowuptotaskimplementationcomments/trigger_decoder_phaseV_13978_20180528_1241.pkl')),
        targets_matrix=pickle.load(open('/storage/target_matrices/targets_HUD1_new_more_fing_ext-0.7_thumb-0.15.pkl')),
        safety_grid_file = '/storage/rawdata/safety/phaseIII_safetygrid_same_minprono_updated_more_fing_ext_-0.55_real_more_fing_ext_-0.7_more_fing_ext_-0.8.pkl',
        language = 'castellano', timeout_time=5., assist_speed='high', emg_weight_aa=0.0, emg_weight_rh=0.0,
        decoder_drift_hl=90.)
        #blocking_opts='ArmAssist_and_Pron')
 
    task = Task(targets, plant_type='IsMorePlantHybridBMISoftSafety', **kwargs)
    task.neurondata = SimNeurData()
    task.run_sync()

    import time
    time.sleep(3)

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
    new_fname = 'test_%s.hdf' %time.strftime('%Y%m%d_%H%M')
    shutil.copy(task.h5file.name, new_fname)
    f = open(new_fname)
    f.close()
    return new_fname

def plot_decoder():
    direct = '/Users/preeyakhanna/Downloads/resummarytodaystherapy'

    hdf_list = ['hud120180529_43_te14046.hdf',
                'hud120180529_44_te14047.hdf',
                'hud120180529_45_te14048.hdf',
                'test_20180529_1526.hdf',
                'test_20180529_1528.hdf',
                'test_20180529_1536.hdf',
                'test_20180529_1538.hdf',
                'test_20180529_1542.hdf',
                'test_20180529_1548.hdf',
                'test_20180529_1550.hdf']

    test = ['test']*7
    lab = ['14046', '14047', '14048']+test

    # Plot drift correlation for these guys: 
    off = 0
    f, ax = plt.subplots(nrows=2)

    for ih, hdf in enumerate(hdf_list):
        if lab[ih] == 'test':
            fl = tables.openFile('/Users/preeyakhanna/ismore/ismore_tests/'+hdf)
        else:
            fl = tables.openFile(direct+'/'+hdf)
        T = np.arange(len(fl.root.task[:]['drift_correction']))
        assist = fl.root.task.attrs.rh_assist_level[0]
        ax[0].plot(T+off, fl.root.task[:]['drift_correction'][:, 7+4], label=lab[ih]+', assist: '+str(assist))

        # Divide by scale
        ax[1].plot(T+off, fl.root.task[:]['drive_velocity_raw_brain'][:, 4]/30.)

        ix = np.nonzero(fl.root.task_msgs[:]['msg']=='target')[0]
        ts = fl.root.task_msgs[ix]['time']
        ax[1].plot(ts+off, fl.root.task[ts]['drive_velocity_raw_brain'][:, 4]/30., 'r.')

        # Add drift correction
        off += len(T)

    ax[0].ylabel('Drift Correction Index')
    ax[1].ylabel('Brain Vel Index')
    ax[0].legend()
    

