
''' 
 Method to combine task entry sessions to make a full target matrix for compliant movements at the beginning of each day 
'''

import numpy as np
from db import dbfunctions as dbfn
from collections import OrderedDict
import pandas as pd
from ismore import common_state_lists
from db.tracker import models
import time, os, pickle

pos_states = common_state_lists.ismore_pos_states
vel_states = common_state_lists.ismore_vel_states
db_name = 'default'

def combine_from_TEs(B1_targets=None, B2_targets=None, B1_B2_targets=None, B3_targets=None, 
    B1_B3_targets=None, Circular=None, file_nm_suffx='', trial_type_replace=None, 
    trial_type_replace_te=None):
    '''
    ONLY FOR ISMORE PLANT

    B1_targets --> task entry for B1, RecordGoalTargets
    B2_targets --> task entry for B2, RecordGoalTargets
    B1_B2_targets --> task entry for B1_B2, RecordGoalTargets
    Circular targets --> task entry for Circular

    [F1 from B1, Linear from B1, F1_B2 from B1_B2, Circular]

    trial_type_replace: List of trial types to use from trial_type_replace_te:
        e.g. ['red', 'rest', 'green'] # trial_type and index of trial
    trial_type_replace_te: TE to use for replacement trial types
    '''

    # First parse replacement trial types: 
    if trial_type_replace is not None:
        replacement_targets_matrix = {}
        
        taskentry = dbfn.TaskEntry(trial_type_replace_te)
        hdf = taskentry.hdf
        task = hdf.root.task
        trial_accept_reject_idx = [idx for (idx, msg) in enumerate(task[:]['trial_accept_reject']) if msg == 'accept']
        
        for idx in trial_accept_reject_idx:
            trial_type = task[idx-1]['trial_type']
            target_position_i = task[idx-1]['plant_pos']
            pos = pd.Series(target_position_i, pos_states)
            replacement_targets_matrix[trial_type] = pos
        
        for t in trial_type_replace:
            assert t in replacement_targets_matrix.keys()

    targets_matrix = OrderedDict()
    position = OrderedDict()
    subgoal_names = OrderedDict()

    ####################################################
    ### Make B1_targets, B2_targets, B1_B2_targets, B3_targets ###
    ####################################################
    target_rest = dict()
    target_name_dict = dict()

    for it, (te, te_name) in enumerate(zip([B1_targets, B2_targets, B1_B2_targets, B3_targets, B1_B3_targets], ['B1', 'B2', 'B1_B2','B3', 'B1_B3'])):
        if te is not None:
            # Extract HDF file: 
            taskentry = dbfn.TaskEntry(te)
            hdf = taskentry.hdf
            task = hdf.root.task
            trial_accept_reject_idx = [idx for (idx, msg) in enumerate(task[:]['trial_accept_reject']) if msg == 'accept']

            for idx in trial_accept_reject_idx:

                #need to look at previous index
                trial_type = task[idx-1]['trial_type']

                if trial_type in trial_type_replace:
                    pos = replacement_targets_matrix[trial_type]
                    print 'replacing: ', trial_type, ' from te: ', te, 'in ', te_name, ' with ', trial_type, ' from te: ', trial_type_replace_te
                else:
                    target_position_i = task[idx-1]['plant_pos']
                    pos = pd.Series(target_position_i, pos_states)
                
                if trial_type == 'rest':
                    target_rest[te_name] = pos

                targets_matrix[trial_type] = pos
                target_name_dict[trial_type] = te_name

                subgoal_names[trial_type] =  OrderedDict()
                subgoal_names[trial_type][0] = [trial_type]
                subgoal_names[trial_type][1] = ['back']
        else:
            print ' No TE: ', te_name            

    for trial_type in targets_matrix.keys():
        te_name = target_name_dict[trial_type]
        rest_ = target_rest[te_name]

        ### HACK for 10_11 targets: fix pron angle
        if te_name == 'B3':
            rest_[-1] = -0.74525

        targets_matrix[trial_type] = pd.concat([targets_matrix[trial_type], rest_], axis = 1, ignore_index = True)    
    target_rest_dict = target_rest

    #####################
    ### Make Circular ###
    #####################

    te = Circular
    if te is not None:
        te_name = 'Circular'
        
        target_position = OrderedDict()
        vel_dir_dict = OrderedDict() 

        vel_dir_dict['circular_clockwise'] = [[-1, 0],[0, 1], [1, 0],[0, -1]]
        vel_dir_dict['circular_anticlockwise'] = [[1, 0],[0, 1],[-1, 1], [0, -1]]

        taskentry = dbfn.TaskEntry(te)
        hdf = taskentry.hdf
        task = hdf.root.task
        trial_accept_reject_idx = [idx for (idx, msg) in enumerate(task[:]['trial_accept_reject']) if msg == 'accept']
        trial_circ_list = []

        for n, idx in enumerate(trial_accept_reject_idx):
            trial_type = task[idx-1]['trial_type']
            if trial_type == 'circular_clockwise':
                trial_type_ammend = 'stirr_clockwise'
            elif trial_type == 'circular_anticlockwise':
                trial_type_ammend = 'stirr_anticlockwise'

            if trial_type not in targets_matrix.keys() and trial_type != 'rest':
                trial_circ_list.append(trial_type_ammend)
                
                targets_matrix[trial_type_ammend] = []
                print 'adding targets for trial type : circular ', trial_type, trial_type_ammend

                #look for indexes of this trial type
                ind_trial_type = [ idx for (idx, msg) in enumerate(task[trial_accept_reject_idx-np.ones((len(trial_accept_reject_idx),),dtype=np.int)]['trial_type']) if msg == trial_type]
                
                subgoal_names[trial_type_ammend] =  OrderedDict() 
                
                for n, idx in enumerate(ind_trial_type):
                    target_position[n] = task[trial_accept_reject_idx[idx]]['plant_pos']
              
                # Approach A: only 4 points will be defined and the vel_end vectors will be hardcoded to the tangential vectors to the circular trajectory
                for n in range(len(target_position)):
                    pos = pd.Series(target_position[n], pos_states)
                    
                    if n == len(target_position)-1: #last point recorded
                        pos_next = pd.Series(target_position[0], pos_states)
                    else:
                        pos_next= pd.Series(target_position[n+1], pos_states)        

                    vel_dir = vel_dir_dict[trial_type][n]
                    vel_dir_psi = np.sign(pos_next['aa_ppsi']-pos['aa_ppsi']) #calculate the direction of angular velocity in psi to move towards the next target position
                    
                    vel_mod = 7
                    vel_mod_psi = np.deg2rad(3)
                    
                    vel_xy = pd.Series([vel_mod *vel_value for vel_value in vel_dir ], vel_states[:2])
                    vel_psi = pd.Series([vel_mod_psi * vel_dir_psi], [vel_states[2]])
                    
                    vel_zeros = pd.Series(np.zeros(len(vel_states)-3), vel_states[3:])
                    vel = pd.concat([vel_xy, vel_psi, vel_zeros])
                    target = pd.concat([pos,vel], axis = 0)

                    if n == 0:
                        target_rest = pd.concat([pos,vel], axis = 0)
                    if n == 1:
                        targets_matrix[trial_type_ammend] = target
                        targets_matrix[trial_type_ammend] = pd.concat([targets_matrix[trial_type_ammend], target], axis = 1, ignore_index = True)
                    elif n > 1:
                        targets_matrix[trial_type_ammend] = pd.concat([targets_matrix[trial_type_ammend], pd.DataFrame(target)], axis = 1, ignore_index = True)
                    
                    #subgoal_names[trial_type_ammend][n] =  OrderedDict() 
                    subgoal_names[trial_type_ammend][n] =  ['circular']

                
            elif trial_type == 'rest':
                ind_trial_type = [ idx for (idx, msg) in enumerate(task[trial_accept_reject_idx-np.ones((len(trial_accept_reject_idx),),dtype=np.int)]['trial_type']) if msg == trial_type]
                subgoal_names[trial_type_ammend].append('back')

                # Take the first rest: 
                idx = ind_trial_type[0]
                target_rest_circ = task[trial_accept_reject_idx[idx]]['plant_pos']
                
            for trial_type_ammend in trial_circ_list:
                targets_matrix[trial_type_ammend] = pd.concat([targets_matrix[trial_type_ammend], pd.DataFrame(target_rest_circ)], axis = 1, ignore_index = True)

    ############################################################
    ##### Make F1_targets from B1, F1_B2 from B1_B2_targets ####
    ############################################################
    for it, (te, te_name) in enumerate(zip([B1_targets, B1_B2_targets], ['F1', 'F1_B2'])):
        if te is not None:
            taskentry = dbfn.TaskEntry(te)
            hdf = taskentry.hdf
            task = hdf.root.task
            trial_accept_reject_idx = [idx for (idx, msg) in enumerate(hdf.root.task[:]['trial_accept_reject']) if msg == 'accept']
            targets_matrix_init = OrderedDict()
            for idx in trial_accept_reject_idx:
                trial_type = task[idx-1]['trial_type']
                target_position = task[idx-1]['plant_pos']
                if trial_type in trial_type_replace:
                    pos = replacement_targets_matrix[trial_type]
                    print 'replacing: ', trial_type, ' from te: ', te, 'in ', te_name, ' with ', trial_type, ' from te: ', trial_type_replace_te
                else:
                    pos = pd.Series(target_position, pos_states)
                targets_matrix_init[trial_type] = pos

            single_targets = targets_matrix_init.keys()
            single_targets.remove('rest')

            double_targets = []
            for target_1 in single_targets:
                for target_2 in single_targets:
                    if target_1 != target_2 and target_1[:4] != target_2[:4]:
                        double_targets.append(target_1 + ' to ' + target_2)

                        ### Double Targets ###: 
                        targets_matrix[double_targets[-1]] = pd.concat([targets_matrix_init[target_1],targets_matrix_init[target_2],targets_matrix_init['rest']], axis = 1)
                        
                        subgoal_names[double_targets[-1]] = OrderedDict()                
                        subgoal_names[double_targets[-1]][0] =  [target_1]
                        subgoal_names[double_targets[-1]][1] =  [target_2]
                        subgoal_names[double_targets[-1]][2] =  ['back']
        else:
            print ' No TE: ', te_name

    ##############################################
    ##### Make Cleanign targets from B1 ##########
    ##############################################
    targ_prog = ['red','blue','red','blue','rest']
    tlist = []
    sg = OrderedDict()
    for i, (t0, t1) in enumerate(zip(targ_prog[:-1], targ_prog[1:])):
        targ = get_vel(targets_matrix[t0][0], targets_matrix[t1][0], vel_states)
        tlist.append(targ)
        sg[i] = [t0]

    rest_pos = pd.Series(target_rest_dict['B1'], pos_states)
    vel_zeros = pd.Series(np.zeros(len(vel_states)), vel_states)
    rest = pd.concat([rest_pos, vel_zeros], axis=0)

    tlist.append(rest)
    sg[i+1] = ['back']
    subgoal_names['clean_horizontal'] = sg
    targets_matrix['clean_horizontal'] = pd.concat(tlist, axis=1)

    targ_prog = ['green','rest','green','rest']
    tlist = []
    sg = OrderedDict()
    for i, (t0, t1) in enumerate(zip(targ_prog[:-1], targ_prog[1:])):
        targ = get_vel(targets_matrix[t0][0], targets_matrix[t1][0], vel_states)
        tlist.append(targ)
        sg[i] = [t0]

    rest_pos = pd.Series(target_rest_dict['B1'], pos_states)
    vel_zeros = pd.Series(np.zeros(len(vel_states)), vel_states)
    rest = pd.concat([rest_pos, vel_zeros], axis=0)

    tlist.append(rest)
    sg[i+1] = ['back']
    subgoal_names['clean_vertical'] = sg
    targets_matrix['clean_vertical'] = pd.concat(tlist, axis=1)

    ###########################################
    ############### LINEAR ####################
    ###########################################
    if B1_targets is not None:
        for trial_type in ['red', 'green', 'blue']:      
            if trial_type in targets_matrix.keys():
                
                targets_matrix['linear_'+trial_type] = []

                #look for indexes of this trial type
                ind_trial_type = [ idx for (idx, msg) in enumerate(task[trial_accept_reject_idx-np.ones((len(trial_accept_reject_idx),),dtype=np.int)]['trial_type']) if msg == trial_type]
               
                target_position = OrderedDict()
                target_position[1] = target_rest_dict['B1']
                target_position[0] = targets_matrix[trial_type][0]

                xy_mod = 5

                for n in range(len(target_position)):
                    pos = pd.Series(target_position[n], pos_states)

                    if n == 0: #starting point of the linear movement of this trial_type -- proximal
                        # x,y.psi
                        pos_diff = target_position[n][:3] - target_position[n+1][:3] #from proximal to distal
                        vel_dir = np.sign(pos_diff)
                        mod = np.sqrt(pos_diff[0]**2 + pos_diff[1]**2)

                        vel_mod_x = np.abs(xy_mod*(pos_diff[0]/ mod))
                        vel_mod_y = np.abs(xy_mod*(pos_diff[1]/ mod))
                        vel_mod_psi = np.deg2rad(1)

                        vel_x = pd.Series([vel_mod_x * vel_dir[0]], [vel_states[0]])
                        vel_y = pd.Series([vel_mod_y * vel_dir[1]], [vel_states[1]])
                        vel_psi = pd.Series([vel_mod_psi * vel_dir[2]], [vel_states[2]])
                
                        # rehand DoFs
                        vel_zeros = pd.Series(np.zeros(len(vel_states)-3), vel_states[3:])
                        vel = pd.concat([vel_x, vel_y, vel_psi, vel_zeros])
                        target_rest = pd.concat([pos,vel], axis = 0)
                      
                    else: 

                        # x,y.psi
                        pos_diff = target_position[n][:3]-target_position[0][:3] #from distal to proximal
                        vel_dir = np.sign(pos_diff)
                        mod = np.sqrt(pos_diff[0]**2 + pos_diff[1]**2)

                        vel_mod_x = np.abs(xy_mod*(pos_diff[0]/ mod))
                        vel_mod_y = np.abs(xy_mod*(pos_diff[1]/ mod))
                        vel_mod_psi = np.deg2rad(1)

                        vel_x = pd.Series([vel_mod_x * vel_dir[0]], [vel_states[0]])
                        vel_y = pd.Series([vel_mod_y * vel_dir[1]], [vel_states[1]])
                        vel_psi = pd.Series([vel_mod_psi * vel_dir[2]], [vel_states[2]])
                
                        # rehand DoFs
                        vel_zeros = pd.Series(np.zeros(len(vel_states)-3), vel_states[3:])
                        vel = pd.concat([vel_x, vel_y, vel_psi, vel_zeros])
                        target = pd.concat([pos,vel], axis = 0)
                        targets_matrix['linear_'+trial_type] = pd.concat([target, target_rest], axis = 1, ignore_index = True)
    else:
        print 'no linear targets'

    subgoal_names['rest'] = OrderedDict()
    subgoal_names['rest'][0] = ['rest']
    targets_matrix['subgoal_names'] = subgoal_names

    # print "target matrix : ",  targets_matrix
    # print "subgoal names : ",  targets_matrix['subgoal_names']

    subject_name  = models.TaskEntry.objects.get(id=B1_targets).subject.name

    #add date and time to differenciate between targets_matrices
    targets_matrix_file_name = 'targets_%s_%s_%s_%s_%s_%s' % (subject_name, str(B1_targets), str(B2_targets), str(B1_B2_targets), str(Circular), time.strftime('%Y%m%d_%H%M'))
    targets_matrix_file_name = targets_matrix_file_name + file_nm_suffx
    pkl_name = targets_matrix_file_name + '.pkl'

    ## Store a record of the data file in the database
    storage_dir = '/storage/target_matrices'

    if not os.path.exists(storage_dir):
        os.popen('mkdir -p %s' % storage_dir)

    pickle.dump(targets_matrix, open(os.path.join(storage_dir, pkl_name), 'wb'))

    dfs = models.DataFile.objects.filter(path = pkl_name)

    if len(dfs) == 0:
        data_sys = models.System.make_new_sys('misc')
        data_sys.name = targets_matrix_file_name
        data_sys.entry = models.TaskEntry.objects.using(db_name).get(id=B1_targets)
        data_sys.save_to_file( targets_matrix, pkl_name, obj_name=None, entry_id=B1_targets)
    else:
        print 'error! This name already exists: ', pkl_name


def get_vel(targ0, targ1, vel_states):
    pos = pd.Series(targ0, pos_states)
    pos_diff = targ1[:3] - targ0[:3]
    vel_dir = np.sign(pos_diff).values
    mod = np.sqrt(pos_diff[0]**2 + pos_diff[1]**2)

    xypsi_mod = np.abs(pos_diff/mod*5.).values
    xypsi_mod[2] = np.deg2rad(1.)
    vel_aa = pd.Series(xypsi_mod*vel_dir, vel_states[:3])

    vel_zeros = pd.Series(np.zeros(len(vel_states)-3), vel_states[3:])
    vel = pd.concat([vel_aa, vel_zeros])
    target = pd.concat([pos,vel], axis = 0)
    return target