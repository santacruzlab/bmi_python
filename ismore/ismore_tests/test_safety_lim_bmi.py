import ismore.invasive.bmi_ismoretasks as bmi_ismoretasks
from riglib import experiment
from features.hdf_features import SaveHDF
from ismore.ismore_tests.test_clda import save_dec_enc
from ismore.ismore_tests import test_kfdecoder_fcns
import pickle
import datetime, shutil, time
import tables
from ismore import settings
from ismore.safetygrid import SafetyGrid
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def run_MC_data():
    Task = experiment.make(bmi_ismoretasks.ManualControl, [SaveHDF])
    targets = bmi_ismoretasks.SimBMIControl.ismore_simple(length=1)
    plant_type = 'IsMore'
    kwargs=dict(session_length=60.)
    task = Task(targets, plant_type=plant_type, **kwargs)
    task.run_sync()

    pref = 'ismore_mc_'
    ct = datetime.datetime.now()
    pnm = '/Users/preeyakhanna/ismore/ismore_tests/sim_data/'+pref + ct.strftime("%Y%m%d_%H_%M_%S") + '.pkl'
    #pnm = '/home/lab/code/ismore/ismore_tests/sim_data/'+pref + ct.strftime("%m%d%y_%H%M") + '.pkl'
    #Save HDF file
    new_hdf = pnm[:-4]+'.hdf'

    f = open(task.h5file.name)
    f.close()

    #Wait 
    time.sleep(1.)

    #Wait after HDF cleaned up
    task.cleanup_hdf()
    time.sleep(1.)

    #Copy temp file to actual desired location
    shutil.copy(task.h5file.name, new_hdf)
    f = open(new_hdf)
    f.close()

    #Return filename
    print new_hdf


def get_grid_from_hdf(hdf_name = '/Users/preeyakhanna/ismore/ismore_tests/sim_data/enc_20170724_17_20_03.hdf', local_dist = 5, 
    plant_type='IsMore', delta=.5, simulation = False):

    if simulation: 
        rh_flag = True
        mn_mat = np.min(settings.MAT_SIZE) - 1
        tmp = np.arange(0, mn_mat, 1)
        tmp2 = np.arange(0, mn_mat, 4)
        
        x = np.hstack(( np.arange(0, mn_mat, 1), np.zeros((len(tmp))) + mn_mat, np.arange(mn_mat, 0, -1), np.zeros((len(tmp)))))  
        y = np.hstack(( np.zeros((len(tmp))), np.arange(0, mn_mat, 1),  np.zeros((len(tmp))) + mn_mat, np.arange(mn_mat, 0, -1) ))

        psi = np.arange(-np.pi/2, 2*np.pi/3)
        prono = np.arange(0, np.pi/2, np.pi/16)
        
        aa_data = np.zeros((len(psi)*len(y) + len(psi)*(len(tmp2)**2)), dtype=[('aa_px', 'f8'), ('aa_py', 'f8'), ('aa_ppsi', 'f8')])
        rh_data = np.zeros((len(prono)*len(y) + len(prono)*(len(tmp2)**2)), dtype=[('aa_px', 'f8'), ('aa_py', 'f8'), ('aa_ppsi', 'f8'), ('rh_pprono', 'f8'), 
            ('rh_pthumb', 'f8'), ('rh_pindex', 'f8'), ('rh_pfing3', 'f8')])

        cnt = 0
        cnt2 = 0
        for i, (xi, yi) in enumerate(zip(x, y)):
            for j, (pi, pr) in enumerate(zip(psi, prono)):
                aa_data[cnt] = (xi, yi, pi)
                cnt += 1

                rh_data[cnt2] = (xi, yi, pi, pr, np.random.randn(),  np.random.randn(),  np.random.randn())
                cnt2 += 1

        bound_aa = aa_data[:cnt].copy()

        for i, xi in enumerate(tmp2):
            for iy, yi in enumerate(tmp2):
                for j, pi in enumerate(psi):
                    aa_data[cnt] = (xi, yi, pi)
                    cnt += 1

                for k, pr in enumerate(prono):
                    rh_data[cnt2] = (xi, yi, pi, pr, np.random.randn(),  np.random.randn(),  np.random.randn())
                    cnt2 += 1

    else:
        hdf = tables.openFile(hdf_name)
        aa_data = hdf.root.armassist[:]['data']
        aa = hdf.root.armassist
        rh_flag = 'rehand' in hdf.root

        if rh_flag:
            rh = hdf.root.rehand
            rh_data = hdf.root.rehand[:]['data']

    # create a SafetyGrid object
    mat_size = settings.MAT_SIZE
    #delta = 0.5  # size (length/width in cm) of each square in the SafetyGrid
    safety_grid = SafetyGrid(mat_size, delta)

    if simulation:
        boundary_positions = np.array([bound_aa[:]['aa_px'], bound_aa[:]['aa_py']]).T
    else:
        boundary_positions = np.array([aa_data[:]['aa_px'], aa_data[:]['aa_py']]).T
    safety_grid.set_valid_boundary(boundary_positions)

    interior_pos = [np.mean(aa_data[:]['aa_px']), np.mean(aa_data[:]['aa_py'])]
    safety_grid.mark_interior_as_valid(interior_pos)

    safety_grid.plot_valid_area()
    print 'Total valid area: %.2f cm^2' % safety_grid.calculate_valid_area()


    # load SafetyGrid object from the .pkl file
    #safety_grid = pickle.load(open(args.pkl_name, 'rb'))

    # for each recorded psi angle, update the min/max psi values (if necessary)
    #   of all positions within a local_dist cm radius
    for i in range(len(aa_data)):
        pos = np.array([aa_data[i]['aa_px'], aa_data[i]['aa_py']])
        psi = aa_data[i]['aa_ppsi']
        safety_grid.update_minmax_psi(pos, psi, local_dist)

    if rh_flag:
        if simulation:
            for i in range(len(rh_data)):
                pos = np.array([rh_data[i]['aa_px'], rh_data[i]['aa_py']])
                pron = rh_data[i]['rh_pprono']
                safety_grid.update_minmax_prono(pos, pron, local_dist)
        else:
            # since the ArmAssist and ReHand data are asynchronous, we need to 
            #   interpolate the ArmAssist data onto the times (in ts_interp) at 
            #   which ReHand data is saved, so that we know the corresponding 
            #   ArmAssist xy-position for each prono value
            ts    = aa[:]['ts_arrival']
            aa_px = aa[:]['data']['aa_px']
            aa_py = aa[:]['data']['aa_py']
            ts_interp = rh[:]['ts_arrival']
            
            ts_interp2 = ts_interp[np.logical_and(ts_interp>=ts[0], ts_interp<=ts[-1])]
            rh_data = rh_data[np.logical_and(ts_interp>=ts[0], ts_interp<=ts[-1])]
            interp_aa_px = interp1d(ts, aa_px)(ts_interp2)
            interp_aa_py = interp1d(ts, aa_py)(ts_interp2)

            # for each recorded psi angle, update the min/max prono values 
            #   (if necessary) of all positions within a local_dist cm radius
            max_prono = np.max(rh_data[:]['rh_pprono'])
            min_prono = np.min(rh_data[:]['rh_pprono'])

            for i in range(len(rh_data)):
                pos = np.array([interp_aa_px[i], interp_aa_py[i]])
                # prono = rh_data[i]['rh_pprono']
                # Update min / max
                safety_grid.update_minmax_prono(pos, max_prono, local_dist)
                safety_grid.update_minmax_prono(pos, min_prono, local_dist)

        safety_grid.define_finger_minmax(rh_data)

    safety_grid.plot_minmax_psi()
    safety_grid.plot_minmax_prono()

    print ''
    if safety_grid.is_psi_minmax_set():
        print 'Min/max psi values were set for all valid grid squares!'
    else:
        print 'Min/max psi values were NOT set for all valid grid squares!'
        print 'Try one of the following:'
        print '  1) recording calibration movements that cover more of the workspace'
        print '  2) increase the local_dist parameter'
    print ''

    print ''
    if safety_grid.is_prono_minmax_set():
        print 'Min/max prono values were set for all valid grid squares!'
    else:
        print 'Min/max prono values were NOT set for all valid grid squares!'
        print 'Try one of the following:'
        print '  1) recording calibration movements that cover more of the workspace'
        print '  2) increase the local_dist parameter'
    print ''


    #save the SafetyGrid to a .pkl file with the same name as the .hdf file
    output_pkl_name = hdf_name[:-4] + '_safetey_grid.pkl'
    pickle.dump(safety_grid, open(output_pkl_name, 'wb'))
    print output_pkl_name
    return output_pkl_name


def test_grid_from_hdf(plant_type, sl_name = None, dec_fname = None, hdf_name = None, delta_grid = .5, 
    simulation = False, local_dist=1):

    # Visual Feedback used to train a decoder: 
    if dec_fname is None:
        dec_fname, hdf_name = test_kfdecoder_fcns.train_decoder_simulation(desired_update_rate=.1, plant_type=plant_type, zscore_flag=False, return_hdf = True)

    if sl_name is None:
        # Get safety limit: 
        sl_name = get_grid_from_hdf(hdf_name=hdf_name, local_dist=local_dist, delta=delta_grid, simulation = simulation)

    # Run simulated BMI w/ sl: 
    test_kfdecoder_fcns.ismore_bmi_simulation(dec_fname, decoder = None, plant_type=plant_type, safety_grid_name=sl_name)
    return sl_name

def save_safety_grid(safety_grid_name, te_num):
    from db.tracker import dbq
    dbq.save_data(safety_grid_name, 'misc', te_num, move=True, local=True, custom_suffix='_safetey_grid.pkl')


if __name__ == '__main__':
    te_num = 5887
    d = '/home/lab/code/ismore/ismore_tests/sim_data/is_decoder072517_1501.pkl'
    hdf = '/home/lab/code/ismore/ismore_tests/sim_data/enc_072517_1501.hdf'
    sl_name = test_grid_from_hdf('IsMore', dec_fname=d, hdf_name=hdf, delta_grid=2., simulation=True, local_dist=2.)
    save_safety_grid(sl_name, te_num)