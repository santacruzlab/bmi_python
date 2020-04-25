
import numpy as np
import pandas as pd
import pickle
from ismore import common_state_lists
from db.tracker import models
from db import dbfunctions as dbfn

def generate_target_matrices_for_sim():
    ''' 
    Point: use these as input to sim_passive_movements so you can test target_matrices
    '''
    ## B1 ##
    tm = pickle.load(open('/storage/target_matrices/targets_HUD1_4767_B1_testing_20170712_1032.pkl'))
    targets_matrix = {}
    up = np.array([.6, .52, .75, .768])
    down = np.array([.6, .52, .75, 1.5])
    rest = np.array([.64, 1.0, 1.4, 1.2])
    point = np.array([.7, .5, 1.5, 1.2])
    grasp = np.array([.4, .4, .5, 1.2])

    #if generator_name == 'B1':
    for k in tm.keys():
        if k != 'subgoal_names':
            targets_matrix[k] = tm[k]

            # F1
            if k not in ['rest']:
                for k_ in tm.keys():
                    if k_ not in ['subgoal_names', 'rest'] and k_!= k:
                        targets_matrix[k+'_to_'+k_] = pd.concat([tm[k][0], tm[k_][0], tm[k][1]], axis=1, ignore_index=True)

                # B1 B2: 
                for i, (k2_nm, k2) in enumerate(zip(['up', 'point', 'grasp', 'down'], [up, point, grasp, down])):
                    v = targets_matrix[k]
                    v[0][3:] = k2
                    targets_matrix[k+'_'+k2_nm] = pd.concat([v[0], tm['rest'][0]], axis=1, ignore_index=True)

    # B2: 
    for i, (k2_nm, k2) in enumerate(zip(['up', 'point', 'grasp', 'down'], [up, point, grasp, down])):
        v = tm['rest'].copy()
        v[0][3:] = k2
        targets_matrix[k2_nm] = pd.concat([v[0], tm['rest'][0]], axis=1, ignore_index=True)

    # Circular
    # Clockwise: 
    targets_matrix['circular_clockwise'] = pd.concat([tm['red'][0], tm['green'][0], tm['blue'][0], tm['rest'][0]], axis=1, ignore_index=True)
    targets_matrix['circular_anticlockwise'] = pd.concat([ tm['blue'][0], tm['green'][0], tm['red'][0], tm['rest'][0]], axis=1, ignore_index=True)

    # Linear:
    lr =  pd.concat([tm['red'][0], pd.Series(np.array([-1., 0., 0., 0., 0., 0., 0.]), common_state_lists.ismore_vel_states)], axis=0, ignore_index=True)
    lb =  pd.concat([tm['blue'][0], pd.Series(np.array([1., -1., 0., 0., 0., 0., 0.]), common_state_lists.ismore_vel_states)], axis=0, ignore_index=True)
    lg =  pd.concat([tm['green'][0], pd.Series(np.array([0., -1., 0., 0., 0., 0., 0.]), common_state_lists.ismore_vel_states)], axis=0, ignore_index=True)
    lrest =  pd.concat([tm['rest'][0], pd.Series(np.array([1., .5, 0., 0., 0., 0., 0.]), common_state_lists.ismore_vel_states)], axis=0, ignore_index=True)
    
    targets_matrix['linear_red'] = pd.concat([lr, lrest, lr, lrest], axis=1, ignore_index=True)
    targets_matrix['linear_green'] = pd.concat([lb, lrest, lb, lrest], axis=1, ignore_index=True)
    targets_matrix['linear_blue'] = pd.concat([lg, lrest, lg, lrest], axis=1, ignore_index=True)
    pickle.dump(targets_matrix, open('/storage/target_matrices/compliant_move_test.pkl', 'wb'))
    subject_name = 'Testing'
    pkl_name = '/storage/target_matrices/compliant_move_test.pkl'
    tmfn = 'compliant_move_test.pkl'

    ## Store a record of the data file in the database
    storage_dir = '/storage/target_matrices'

    dfs = models.DataFile.objects.filter(path = pkl_name)
    if len(dfs) == 0:
        data_sys = models.System.make_new_sys('misc')
        data_sys.name = tmfn
        data_sys.entry = models.TaskEntry.objects.using('default').get(id=6099)
        data_sys.save_to_file( targets_matrix, pkl_name, obj_name=None, entry_id=6099)

    elif len(dfs) > 1:
        print "Warning: More than one targets_matrix with the same name! File will be overwritten but name wont show up twice at the database"
    
def test_te_safety(te_num, safety_file_name = None):
    task_entry = dbfn.TaskEntry(te_num)
    if safety_file_name is None:
        safety_sys = models.System.objects.get(name='safety')
        safety_grid_file = models.DataFile.objects.filter(system=safety_sys, id = task_entry.safety_grid_file)
        if len(safety_grid_file) == 1:
            path = safety_grid_file[0].path
        else:
            raise Exception('Too many safety grid entries that match the id saved in te_num...')
    else:
        path = safety_file_name
        
    plant_pos = task_entry.hdf.root.task[:]['plant_pos']

    if plant_pos.shape[1] == 7:
        plant_type = 'IsMore'
    elif plant_pos.shape[1] == 3:
        plant_type = 'ArmAssist'
    elif plant_pos.shape[1] == 4:
        plant_type = 'ReHand'

    T = plant_pos.shape[0]

    # Iterate through each position
    safety_grid = pickle.load(open(path))

    psi_fail = []
    prono_fail = []
    fing_fail = dict()
    fing_nms = ['rh_pthumb', 'rh_pindex', 'rh_pfing3']
    for f in fing_nms:
        fing_fail[f] = []

    for t in range(1, T):
        if plant_type in ['ArmAssist', 'IsMore']:
            x, y = plant_pos[t, [0, 1]]

            # min/max psi: 
            mn, mx = safety_grid.get_minmax_psi([x, y])

            psi = plant_pos[t, 2]
            if np.logical_or(psi > mx, psi < mn):
                psi_fail.append(t)

        if plant_type in ['IsMore']:
            # min/max pprono: 
            mn, mx = safety_grid.get_minmax_prono([x, y])
            prono = plant_pos[t, 6]
            if np.logical_and(prono > mx, prono < mn):
                prono_fail.append(t)

        if plant_type in ['IsMore', 'ReHand']:
            if plant_type == 'ReHand':
                fings = plant_pos[t, [0, 1, 2]]
            elif plant_type == 'IsMore':
                fings = plant_pos[t, [3, 4, 5]]

            for i, (val, nm) in enumerate(zip(fings, fing_nms)):
                mn, mx = safety_grid.get_rh_minmax(nm)
                if np.logical_and(val > mx, val < mn):
                    fing_fail[nm].append(t)
    return psi_fail, prono_fail, fing_fail, T

def test_compliant_move_safety(targets_matrix_file_name, safety_file_name, skip_targ=[]):
    safety_grid = pickle.load(open(safety_file_name))
    targ_mat = pickle.load(open(targets_matrix_file_name))
    
    xy_fail = []
    psi_fail = []
    prono_fail = []
    fing_fail = dict()
    fing_nms = ['rh_pthumb', 'rh_pindex', 'rh_pfing3']
    for f in fing_nms:
        fing_fail[f] = []

    for i, t in enumerate(targ_mat.keys()):
        if t not in skip_targ+['subgoal_names']:
            trial = targ_mat[t]
            N = trial.shape[1]
            for n in range(N):
                targ = trial[n]
                x, y = targ[[0, 1]]
                if not safety_grid.is_valid_pos((x, y)):
                    if t == 'rest':
                        xy_fail.append('rest')
                    else:
                        xy_fail.append([t, targ_mat['subgoal_names'][t][n][0]])
                else:
                    # min/max psi: 
                    mn, mx = safety_grid.get_minmax_psi([x, y])

                    psi = targ[2]

                    if np.logical_or(psi > mx, psi < mn):
                        psi_fail.append([t, targ_mat['subgoal_names'][t][n][0]])

                
                    mn, mx = safety_grid.get_minmax_prono([x, y])
                    prono = targ[6]
                    if np.logical_and(prono > mx, prono < mn):
                        prono_fail.append([t, targ_mat['subgoal_names'][t][n][0]])

                    fings = targ[[3, 4, 5]]

                    for i, (val, nm) in enumerate(zip(fings, fing_nms)):
                        mn, mx = safety_grid.get_rh_minmax(nm)
                        if np.logical_and(val > mx, val < mn):
                            fing_fail[nm].append([t, targ_mat['subgoal_names'][t][n][0]])
    return xy_fail, psi_fail, prono_fail, fing_fail









