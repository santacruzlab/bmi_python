'''Script for processing the data saved in an HDF file to define the outer 
safety boundary. Saves the resulting SafetyGrid object to a .pkl file.'''

import argparse
import numpy as np
import tables
import pickle
import matplotlib.pyplot as plt
import os
from db import dbfunctions as dbfn
from db.tracker import models

from ismore import settings
from ismore import safetygrid
from ismore import common_state_lists
import time

# if __name__ == '__main__':

#     ### parse command line arguments ###
#     parser = argparse.ArgumentParser(description='TODO')
#     parser.add_argument('te', help='task entry number used with saftey_grid_task')
#     parser.add_argument('boundary_tol')
#     parser.add_argument('angle_tol_degrees')
#     parser.add_argument('suffx', help='suffix')
#     args = parser.parse_args()

def make_grid(te_list, local_dist, local_dist_gte_45x, suffx, attractor_pt_dist_from_rest):

    ## load armassist data from hdf file ###
    hdf_list = []
    for te in te_list:
        te = dbfn.TaskEntry(te)
        hdf_fname = te.hdf_filename
        hdf_list.append(hdf_fname)

    ### create a SafetyGrid object ###
    mat_size = settings.MAT_SIZE
    delta = 0.5  # size (length/width in cm) of each square in the SafetyGrid
    #boundary_tolerence = float(boundary_tol) # tolerance of boudary
    #angle_tolerence = np.pi/180.*float(angle_tol_degrees) # 5 degress of angle tolerence
    safety_grid = safetygrid.SafetyGridTargetsMatrix(mat_size, delta, 0, 0, hdf_list, local_dist, local_dist_gte_45x)

    ### Defining Boundary: ###
    safety_grid.set_valid_boundary()
    safety_grid.plot_valid_area()
    print 'Total valid area: %.2f cm^2' % safety_grid.calculate_valid_area()

    safety_grid.update_minmax_psi_prono(psi_or_prono='prono')
    safety_grid.update_minmax_psi_prono(psi_or_prono='psi')
  
    safety_grid.plot_minmax_psi()
    safety_grid.plot_minmax_prono()

    ### HARD CODE RH POSITIONS ###
    ### HARD CODE RH POSITIONS ###
    ### HARD CODE RH POSITIONS ###
    print ''
    print ''
    print 'hard-coding hand-positions: rh_index, rh_fing3: ', -0.39, 0.38
    safety_grid.rh_pindex = [-0.39, 0.38]
    safety_grid.rh_pfing3 = [-0.39, 0.38]

    print 'hard-coding max prono to zero: '
    safety_grid._grid['max_prono'][~np.isnan(safety_grid._grid['max_prono'])] = 0

    print 'hard-coding thumb max to: ', -0.49, -0.1
    safety_grid.rh_pthumb = [-0.49, -0.1]
    ### HARD CODE RH POSITIONS ###
    ### HARD CODE RH POSITIONS ###
    ### HARD CODE RH POSITIONS ###

    rest = np.vstack(([(safety_grid.targets_xy[s]['aa_px'],safety_grid.targets_xy[s]['aa_py']) for s in safety_grid.targets_xy.keys()]))
    ix = np.argmin(rest[:, 1])
    print ' Assuming: ', rest[ix, :], ' is Rest target.'
    safety_grid.define_attractor(rest[ix, 0], rest[ix, 1], attractor_pt_dist_from_rest)

    plt.show()

    storage = '/storage/rawdata/safety',
    pkl_name = os.path.join(hdf_fname[:-4] + '_safetygrid_'+suffx+'.pkl')
    print pkl_name
    safety_grid.hdf = None

    pickle.dump(safety_grid, open(os.path.join(storage, pkl_name), 'wb'))

    dfs = models.DataFile.objects.filter(path = pkl_name)

    if len(dfs) == 0:
        data_sys = models.System.make_new_sys('safety')
        data_sys.name = pkl_name
        data_sys.entry = models.TaskEntry.objects.get(id=int(te.id))
        data_sys.save_to_file( safety_grid, pkl_name, obj_name=None, entry_id=int(te.id))

    elif len(dfs) >= 1:
         print "Warning: Safety grid with the same name! Choose another suffx!"

def modify_grid_w_targ_mat(safety_grid_name, target_matrix_name, trials_to_add, suffx, te_num):
    safety_grid = pickle.load(open(safety_grid_name))
    targ_mat = pickle.load(open(target_matrix_name))
    safety_grid.update_targ_pos(targ_mat, trials_to_add)

    positions = np.vstack(([targ_mat[t[0]][t[1]][common_state_lists.aa_xy_states] for t in trials_to_add]))
    safety_grid.set_valid_boundary(positions=positions)
    safety_grid.mark_interior_as_valid(np.mean(positions, axis=0))
    safety_grid.plot_valid_area()
    print 'Total valid area: %.2f cm^2' % safety_grid.calculate_valid_area()

    safety_grid.update_minmax_psi_prono(psi_or_prono='prono')
    safety_grid.update_minmax_psi_prono(psi_or_prono='psi')
  
    safety_grid.plot_minmax_psi()
    safety_grid.plot_minmax_prono()
    plt.show()

    storage = '/storage/rawdata/safety',
    pkl_name = safety_grid_name[:-4] + '_ammended_'+suffx+'.pkl'
    safety_grid.hdf = None

    pickle.dump(safety_grid, open(os.path.join(storage, pkl_name), 'wb'))

    dfs = models.DataFile.objects.filter(path = pkl_name)

    if len(dfs) == 0:
        data_sys = models.System.make_new_sys('safety')
        data_sys.name = pkl_name
        data_sys.entry = models.TaskEntry.objects.get(id=int(te_num))
        data_sys.save_to_file( safety_grid, pkl_name, obj_name=None, entry_id=int(te_num))

    elif len(dfs) >= 1:
         print "Warning: Safety grid with the same name! Choose another suffx!"












