from db import dbfunctions as dbfn
import numpy as np
from db.tracker import models
import pickle

'''
Method  to test the safety of targets in a targets matrix
'''

def test_targ_mat_safety(te_num = 11631, targets_matrix_file = None, safety_file_name=None,
    plant_type='IsMore'):
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
        
    # Iterate through each position
    try:
        safety_grid = pickle.load(open(path))
    except:
        safety_grid = pickle.load(open('/storage/rawdata/safety/'+path))

    psi_fail = []
    prono_fail = []
    fing_fail = dict()
    fing_nms = ['rh_pthumb', 'rh_pindex', 'rh_pfing3']
    for f in fing_nms:
        fing_fail[f] = []

    targets_matrix = pickle.load(open(targets_matrix_file))
    T = targets_matrix.keys()

    for t in T:
        if t != 'subgoal_names':
            for tt in [0, 1]:
                if plant_type in ['ArmAssist', 'IsMore']:
                    x, y = targets_matrix[t][tt][[0, 1]]

                    # min/max psi: 
                    mn, mx = safety_grid.get_minmax_psi([x, y])

                    psi = targets_matrix[t][tt][2]
                    if np.logical_or(psi > mx, psi < mn):
                        psi_fail.append(t)
                        print t, targets_matrix[t][tt], mx, mn, x, y

                if plant_type in ['IsMore']:
                    # min/max pprono: 
                    mn, mx = safety_grid.get_minmax_prono([x, y])
                    prono = targets_matrix[t][tt][6]
                    if np.logical_and(prono > mx, prono < mn):
                        prono_fail.append(t)

                if plant_type in ['IsMore', 'ReHand']:
                    if plant_type == 'ReHand':
                        fings = targets_matrix[t][tt][[3, 4, 5]]
                    elif plant_type == 'IsMore':
                        fings = targets_matrix[t][tt][[3, 4, 5]]

                    for i, (val, nm) in enumerate(zip(fings, fing_nms)):
                        mn, mx = safety_grid.get_rh_minmax(nm)
                        if np.logical_and(val > mx, val < mn):
                            fing_fail[nm].append(t)
    return psi_fail, prono_fail, fing_fail, T


