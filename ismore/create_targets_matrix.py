'''
Script to read a hdf file recorded with a RecordGoalTargets task to create a .pkl file that contains
the goal targets of every trial type
'''

import tables
import argparse
import os
import pickle
import numpy as np
import pandas as pd


from common_state_lists import *
import time


from db import dbfunctions as dbfn
from db.tracker import models
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Parse ArmAssist and/or ReHand \
        trajectories from a .hdf file (corresponding to a task for recording or \
        playing back trajectories) and save them to a .pkl file. Interpolates \
        trajectories from a "record" task, but not for a "playback" task.')
parser.add_argument('id', help='Task entry id from which to parse trajectories')
parser.add_argument('task_type', help='Task type for which to generate target positions')
parser.add_argument('--temp', help='', action="store_true")
args = parser.parse_args()   

# load task, and armassist and/or rehand data from hdf file
db_name = 'default'
te = dbfn.TaskEntry(int(args.id), dbname= db_name)
hdf = te.hdf

task      = hdf.root.task
task_msgs = hdf.root.task_msgs

if te.plant_type  == 'ArmAssist':
    n_joints = 3
    pos_states = aa_pos_states
    vel_states = aa_vel_states

elif te.plant_type == 'ReHand':
    n_joints = 4
    pos_states = rh_pos_states
    vel_states = rh_vel_states

elif te.plant_type == 'IsMore':
    n_joints = 7
    pos_states = aa_pos_states + rh_pos_states
    vel_states = aa_vel_states + rh_vel_states

targets_matrix = OrderedDict()
target_position = OrderedDict()
position = OrderedDict()
subgoal_names = OrderedDict()


#idx where the trial was acceped --> the exo was placed in the goal target position 
trial_accept_reject_idx = [idx for (idx, msg) in enumerate(task[:]['trial_accept_reject']) if msg == 'accept']

# Depending on the type of task, the target matrices are created in a different way

# ----------------------------------------------------- B1: SINGLE TARGET  and B2: SINGLE HAND MOVEMENT ------------------------------------------------------------------------
if args.task_type in ['B1', 'B2']:

    for idx in trial_accept_reject_idx:
        print idx
        #need to look at previous index
        trial_type = task[idx-1]['trial_type']
        target_position = task[idx-1]['plant_pos']
        pos = pd.Series(target_position, pos_states)
                
        targets_matrix[trial_type] = pos

        subgoal_names[trial_type] =  OrderedDict()
        subgoal_names[trial_type][0] = [trial_type]

    for trial_type in targets_matrix.keys():
        targets_matrix[trial_type] = pd.concat([targets_matrix[trial_type]], axis =1)


elif args.task_type in ['B1_w_rest', 'B2_w_rest', 'B1_B2_w_rest', 'B3_w_rest']:

    for idx in trial_accept_reject_idx:
        print idx
        #need to look at previous index
        trial_type = task[idx-1]['trial_type']
        target_position = task[idx-1]['plant_pos']
        pos = pd.Series(target_position, pos_states)
        
        if trial_type == 'rest':
            target_rest = pos

        targets_matrix[trial_type] = pos

        subgoal_names[trial_type] =  OrderedDict()
        subgoal_names[trial_type][0] = [trial_type]
        subgoal_names[trial_type][1] = ['back']
    
    for trial_type in targets_matrix.keys():
        targets_matrix[trial_type] = pd.concat([targets_matrix[trial_type], target_rest], axis = 1, ignore_index = True)    


# ----------------------------------------------------  F1: DOUBLE TARGET  ------------------------------------------------------------------------
elif args.task_type == 'F1':
    targets_matrix_init = OrderedDict()
    for idx in trial_accept_reject_idx:
        print idx
        #need to look at previous index
        trial_type = task[idx-1]['trial_type']
        target_position = task[idx-1]['plant_pos']
        pos = pd.Series(target_position, pos_states)
        targets_matrix_init[trial_type] = pos

    targets_matrix['rest'] = pd.concat([targets_matrix_init['rest']], axis =1)

    single_targets = targets_matrix_init.keys()
    if 'rest' in single_targets:
        single_targets.remove('rest')

    double_targets = []
    for target_1 in single_targets:
        for target_2 in single_targets:
            if target_1 != target_2:
                double_targets.append(target_1 + ' to ' + target_2)
                targets_matrix[double_targets[-1]] = pd.concat([targets_matrix_init[target_1],targets_matrix_init[target_2]], axis = 1)
                
                subgoal_names[double_targets[-1]] = OrderedDict()                
                subgoal_names[double_targets[-1]][0] =  [target_1]
                subgoal_names[double_targets[-1]][1] =  [target_2]

elif args.task_type in ['F1_w_rest', 'F1_B2_w_rest']:
    targets_matrix_init = OrderedDict()
    for idx in trial_accept_reject_idx:
        print idx
        #need to look at previous index
        trial_type = task[idx-1]['trial_type']
        target_position = task[idx-1]['plant_pos']
        pos = pd.Series(target_position, pos_states)
        targets_matrix_init[trial_type] = pos

    #targets_matrix['rest'] = pd.concat([targets_matrix_init['rest']], axis =1)

    single_targets = targets_matrix_init.keys()
    if 'rest' in single_targets:
        single_targets.remove('rest')

    double_targets = []
    for target_1 in single_targets:
        for target_2 in single_targets:
            if target_1 != target_2:
                double_targets.append(target_1 + ' to ' + target_2)
                targets_matrix[double_targets[-1]] = pd.concat([targets_matrix_init[target_1],targets_matrix_init[target_2],targets_matrix_init['rest'] ], axis = 1)
                
                subgoal_names[double_targets[-1]] = OrderedDict()                
                subgoal_names[double_targets[-1]][0] =  [target_1]
                subgoal_names[double_targets[-1]][1] =  [target_2]
                subgoal_names[double_targets[-1]][2] =  ['back']




# cyclic movs for EEG control
# elif args.task_type == 'targets_linear':
#     targets_matrix_init = OrderedDict()
#     for idx in trial_accept_reject_idx:
#         print idx
#         #need to look at previous index
#         trial_type = task[idx-1]['trial_type']
#         import pdb; pdb.set_trace()
#         if 'to' in trial_type:
#             target_position = task[idx-1]['plant_pos']
#             pos = pd.Series(target_position, pos_states)
#             targets_matrix_init[trial_type] = pos
#             single_targets = targets_matrix_init.keys()
#             double_targets = []
#             for target_1 in single_targets:
#                 for target_2 in single_targets:
#                     if target_1 != target_2:
#                         double_targets.append(target_1 + ' to ' + target_2)
#                         targets_matrix[double_targets[-1]] = pd.concat([targets_matrix_init[target_1],targets_matrix_init[target_2],targets_matrix_init[target_1],targets_matrix_init[target_2],targets_matrix_init[target_1],targets_matrix_init[target_2]], axis = 1)
                
#                         subgoal_names[double_targets[-1]] = OrderedDict()                
#                         subgoal_names[double_targets[-1]][0] =  [target_1]
#                         subgoal_names[double_targets[-1]][1] =  [target_2]
#                         subgoal_names[double_targets[-1]][2] =  [target_1]
#                         subgoal_names[double_targets[-1]][3] =  [target_2]
#                         subgoal_names[double_targets[-1]][4] =  [target_1]
#                         subgoal_names[double_targets[-1]][5] =  [target_2]
#     # targets_matrix['rest'] = pd.concat([targets_matrix_init['rest']], axis =1)

#     # single_targets = targets_matrix_init.keys()
#     # if 'rest' in single_targets:
#     #     single_targets.remove('rest')
#     import pdb; pdb.set_trace()
#     for idx in trial_accept_reject_idx:
#         print idx
       
#         trial_type = task[idx-1]['trial_type']  #need to look at previous index
#         #pos = pd.Series(task[idx-1]['plant_pos'], pos_states)

#         if trial_type not in targets_matrix.keys():
            
#             targets_matrix[trial_type] = []
#             print 'adding targets for trial type : linear ', trial_type
#            # import pdb; pdb.set_trace()
#             #look for indexes of this trial type
#             ind_trial_type = [ idx for (idx, msg) in enumerate(task[trial_accept_reject_idx-np.ones((len(trial_accept_reject_idx),),dtype=np.int)]['trial_type']) if msg == trial_type]
#             if 'to' not in trial_type:
#                 subgoal_names[trial_type] =  OrderedDict()
#                 subgoal_names[trial_type][0] = [trial_type]
#                 subgoal_names[trial_type][1] = ['back']

#                 import pdb; pdb.set_trace()
#             for n, idx in enumerate(ind_trial_type):
#                 position[n] = task[trial_accept_reject_idx[idx]]['plant_pos']
#             pos_rest = pd.Series(position[0], pos_states)
#             pos_target = pd.Series(position[1], pos_states)
#             targets_matrix[trial_type] = pd.concat([pos_target,pos_rest,pos_target,pos_rest,pos_target,pos_rest], axis = 1)
#             import pdb
#             pdb.set_trace()

    # for target_1 in single_targets:
    #     targets_matrix[target_1] = pd.concat([targets_matrix_init[target_1],targets_matrix_init['rest'],targets_matrix_init[target_1],targets_matrix_init['rest'],targets_matrix_init[target_1],targets_matrix_init['rest']], axis = 1)
    #     subgoal_names[target_1] = OrderedDict()                
    #     subgoal_names[target_1][0] =  [target_1]
    #     subgoal_names[target_1][1] =  ['back']
    #     subgoal_names[target_1][2] =  [target_1]
    #     subgoal_names[target_1][3] =  ['back']
    #     subgoal_names[target_1][4] =  [target_1]
    #     subgoal_names[target_1][5] =  ['back']




# ----------------------------------------------------- B1_B2: SINGLE REACHING with Hand & Wrist ------------------------------------------------------------------------

elif args.task_type == 'B1_B2':
    for idx in trial_accept_reject_idx:
        print idx
        #need to look at previous index
        trial_type = task[idx-1]['trial_type']
        target_position = task[idx-1]['plant_pos']
        pos = pd.Series(target_position, pos_states)
                
        targets_matrix[trial_type] = pos
    
        subgoal_names[trial_type] =  OrderedDict()
        subgoal_names[trial_type][0] = [trial_type]
        subgoal_names[trial_type][1] = ['back']

    for trial_type in targets_matrix.keys():
        #if trial_type != 'rest':
        targets_matrix[trial_type] = pd.concat([targets_matrix[trial_type]], axis =1)
        
        # if trial_type != 'rest':
        #     ind_between_subtasks = trial_type.find('_')
        #     subtask1 = trial_type[:ind_between_subtasks]
        #     subtask2 = trial_type[ind_between_subtasks+1:]
        #     subgoal_names[trial_type] =  OrderedDict()
        #     subgoal_names[trial_type][0] = [subtask1, subtask2] 

# ----------------------------------------------------- F1_B2: DOUBLE REACHING with Hand & Wrist ------------------------------------------------------------------------
   
elif args.task_type == 'F1_B2':
    targets_matrix_init = OrderedDict()
    for idx in trial_accept_reject_idx:
        print idx
        #need to look at previous index
        trial_type = task[idx-1]['trial_type']
        target_position = task[idx-1]['plant_pos']
        pos = pd.Series(target_position, pos_states)
        targets_matrix_init[trial_type] = pos

    targets_matrix['rest'] = pd.concat([targets_matrix_init['rest']], axis =1)

    single_targets = targets_matrix_init.keys()
    if 'rest' in single_targets:
        single_targets.remove('rest')

    double_targets = []
    for target_1 in single_targets:
        for target_2 in single_targets:
            if target_1 != target_2:
                double_targets.append(target_1 + ' to ' + target_2)
                targets_matrix[double_targets[-1]] = pd.concat([targets_matrix_init[target_1],targets_matrix_init[target_2]], axis = 1)
                
                subgoal_names[double_targets[-1]] = OrderedDict()
                
                ind_between_subtasks1 = target_1.find('_')
                ind_between_subtasks2 = target_2.find('_')

                subtask1_1 = target_1[:ind_between_subtasks1]
                subtask1_2 = target_1[ind_between_subtasks1+1:]

                subtask2_1 = target_2[:ind_between_subtasks2]
                subtask2_2 = target_2[ind_between_subtasks2+1:]

                #we only put the name of the position type
                subgoal_names[double_targets[-1]][0] =  [subtask1_1]#, subtask1_2]
                subgoal_names[double_targets[-1]][1] =  [subtask2_1]#, subtask2_2] 
    
    for trial_type in targets_matrix.keys():
        targets_matrix[trial_type] = pd.concat([targets_matrix[trial_type]], axis =1)

subgoal_names['rest'] = OrderedDict()
subgoal_names['rest'][0] = ['rest']
targets_matrix['subgoal_names'] = subgoal_names

# to readapt - nerea
# ------------------------------------------------------- CYCLIC LINEAR ------------------------------------------------------------------------
if args.task_type == 'linear_EEG': 

    for idx in trial_accept_reject_idx:
        print idx
       
        trial_type = task[idx-1]['trial_type']  #need to look at previous index
        #pos = pd.Series(task[idx-1]['plant_pos'], pos_states)

        if trial_type not in targets_matrix.keys():
            targets_matrix[trial_type] = []
            print 'adding targets for trial type : linear ', trial_type

            #look for indexes of this trial type
            ind_trial_type = [ idx for (idx, msg) in enumerate(task[trial_accept_reject_idx-np.ones((len(trial_accept_reject_idx),),dtype=np.int)]['trial_type']) if msg == trial_type]
            if 'to' in trial_type:
                subgoal_names[trial_type] =  OrderedDict()
                subgoal_names[trial_type][0] = ['blue']
                subgoal_names[trial_type][1] = ['red']
                subgoal_names[trial_type][2] = ['blue']
                subgoal_names[trial_type][3] = ['red']
                subgoal_names[trial_type][4] = ['blue']
                subgoal_names[trial_type][5] = ['red']
            else:
                subgoal_names[trial_type] =  OrderedDict()
                subgoal_names[trial_type][0] = [trial_type]
                subgoal_names[trial_type][1] = ['back']
                subgoal_names[trial_type][2] = [trial_type]
                subgoal_names[trial_type][3] = ['back']
                subgoal_names[trial_type][4] = [trial_type]
                subgoal_names[trial_type][5] = ['back']


            for n, idx in enumerate(ind_trial_type):
                target_position[n] = task[trial_accept_reject_idx[idx]]['plant_pos']
            target_1 = pd.Series(target_position[0], pos_states)
            target_2 = pd.Series(target_position[1], pos_states)
            targets_matrix[trial_type] = pd.concat([target_2,target_1,target_2,target_1,target_2,target_1], axis = 1)
    targets_matrix['subgoal_names'] = subgoal_names

if args.task_type == 'linear': 

    for idx in trial_accept_reject_idx:
        print idx
       
        trial_type = task[idx-1]['trial_type']  #need to look at previous index
        #pos = pd.Series(task[idx-1]['plant_pos'], pos_states)

        if trial_type not in targets_matrix.keys():
            targets_matrix[trial_type] = []
            print 'adding targets for trial type : linear ', trial_type

            #look for indexes of this trial type
            ind_trial_type = [ idx for (idx, msg) in enumerate(task[trial_accept_reject_idx-np.ones((len(trial_accept_reject_idx),),dtype=np.int)]['trial_type']) if msg == trial_type]
           
            subgoal_names[trial_type] =  OrderedDict()
            subgoal_names[trial_type][0] = [trial_type]
            subgoal_names[trial_type][1] = ['back']


            for n, idx in enumerate(ind_trial_type):
                target_position[n] = task[trial_accept_reject_idx[idx]]['plant_pos']

            xy_mod = 5

            for n in range(len(target_position)):
                pos = pd.Series(target_position[n], pos_states)

                # check if there are pos values 


                
                #compute velocity states for LQR controler 
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
                    targets_matrix[trial_type] = pd.concat([target, target_rest], axis = 1, ignore_index = True)


# -------------------------------------------------------- CYCLIC CIRCULAR ------------------------------------------------------------------------  
elif args.task_type == 'circular':

    vel_dir_dict = OrderedDict() 
    vel_dir_dict['circular_clockwise'] = [[-1, 0],[0, 1], [1, 0],[0, -1]]
    # vel_dir_dict['circular_anticlockwise'] = [[1, 0],[0, 1], [-1, 0],[0, -1]]
    vel_dir_dict['circular_anticlockwise'] = [[1, 0],[0, 1],[-1, 1], [0, -1]]


    for n, idx in enumerate(trial_accept_reject_idx):

        trial_type = task[idx-1]['trial_type']
     
        if trial_type not in targets_matrix.keys():

            targets_matrix[trial_type] = []
            print 'adding targets for trial type : circular ', trial_type

            #look for indexes of this trial type
            ind_trial_type = [ idx for (idx, msg) in enumerate(task[trial_accept_reject_idx-np.ones((len(trial_accept_reject_idx),),dtype=np.int)]['trial_type']) if msg == trial_type]
            
            subgoal_names[trial_type] =  OrderedDict() 
            # subgoal_names[trial_type] = ['circular']
             

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
                    targets_matrix[trial_type] = target
                elif n != 0:
                    targets_matrix[trial_type] = pd.concat([targets_matrix[trial_type], target], axis = 1, ignore_index = True)
                
                subgoal_names[trial_type][n] =  OrderedDict() 
                subgoal_names[trial_type][n] =  ['circular']


            targets_matrix[trial_type] = pd.concat([targets_matrix[trial_type], target_rest], axis = 1, ignore_index = True)    
    
            # # Approach B: only 4 points will be defined and the vel_end vectors will be hardcoded to the tangential vectors to the circular trajectory
            # for n in range(len(target_position)):
            #     pos = pd.Series(target_position[n], pos_states)
                
            #     if n == len(target_position)-1: #last point recorded
            #         pos_next = pd.Series(target_position[0], pos_states)
            #     else:
            #         pos_next= pd.Series(target_position[n+1], pos_states)        

             
            #     mod_xy = np.sqrt((pos_next['aa_px']-pos['aa_px'])**2 + (pos_next['aa_py']-pos['aa_py'])**2)
            #     vel_dir = [(pos_next['aa_px']-pos['aa_px'])/mod_xy,(pos_next['aa_py']-pos['aa_py'])/mod_xy] #calculate the direction of angular velocity in psi to move towards the next target position
            #     vel_dir_psi = np.sign(pos_next['aa_ppsi']-pos['aa_ppsi']) #calculate the direction of angular velocity in psi to move towards the next target position
                    
            #     vel_mod =1
            #     vel_mod_psi = np.deg2rad(5)

            #     vel_xy = pd.Series([vel_mod *vel_value for vel_value in vel_dir], vel_states[:2])
            #     vel_psi = pd.Series([vel_mod_psi * vel_dir_psi], [vel_states[2]])

            #     # rehand vel = 0
            #     vel_zeros = pd.Series(np.zeros(len(vel_states)-3), vel_states[3:])
            #     vel = pd.concat([vel_xy, vel_psi, vel_zeros])
            #     target = pd.concat([pos,vel], axis = 0)

            #     if n == 0: #n = 0 always will correspond to te rest position (initial position)
            #         target_rest = pd.concat([pos,vel], axis = 0)

            #     if n == 1:
            #         targets_matrix[trial_type] = target
            #     elif n != 0:
            #         targets_matrix[trial_type] = pd.concat([targets_matrix[trial_type], target], axis = 1, ignore_index = True)
            #     # import pdb; pdb.set_trace()
                
            #     subgoal_names[trial_type][n] =  OrderedDict()  
            #     subgoal_names[trial_type][n] = ['circular']

            # targets_matrix[trial_type] = pd.concat([targets_matrix[trial_type], target_rest], axis = 1, ignore_index = True)    
    

            # import pdb; pdb.set_trace()    
            # Approach C: as many points as desired can be used to define the circular trajectory. The direction of the vel_end vectors will be computed as the direction from the target_position to the next target in the trajectory
            # for n in range(len(target_position)):
            #     pos = pd.Series(target_position[n], pos_states)
            #     if n == len(target_position) - 1:
            #         vel_dir = (target_position[0][:2] - target_position[n][:2])/ np.linalg.norm(target_position[0][:2] - target_position[n][:2])

            #     else:
            #         vel_dir = (target_position[n+1][:2] - target_position[n][:2])/ np.linalg.norm(target_position[n+1][:2] - target_position[n][:2])
            #         #import pdb; pdb.set_trace()
            #     vel_mod = 5 
            #     vel_xy = pd.Series([vel_mod *vel_value for vel_value in vel_dir], vel_states[:2])
            #     vel_zeros = pd.Series(np.zeros(len(vel_states)-2), vel_states[2:])
            #     vel = pd.concat([vel_xy, vel_zeros])        
            #     if n == 0:
            #         target_rest = pd.concat([pos,vel], axis = 0)
            #         vel_rest = pd.Series(np.zeros(len(vel_states)), vel_states)
            #         target_init = pd.concat([pos,vel_rest], axis = 0)
            #         targets_matrix['rest'] = pd.concat([target_init], axis = 1)
            #     elif n == 1:
            #         targets_matrix['circular'] = pd.concat([pos,vel], axis = 0)
            #     else:
            #         #import pdb; pdb.set_trace()
            #         target = pd.concat([pos,vel], axis = 0)
            #         targets_matrix['circular'] = pd.concat([targets_matrix['circular'], target], axis = 1, ignore_index = True)
            # targets_matrix['circular'] = pd.concat([targets_matrix['circular'], target_rest], axis = 1, ignore_index = True) 

elif args.task_type == 'circular_EEG':

    # parser.add_argument('repetitions_cycle', help='Number of cycle repetitions')
    # print int(args.repetitions_cycle)
    for n, idx in enumerate(trial_accept_reject_idx):

        trial_type = task[idx-1]['trial_type']
     
        if trial_type not in targets_matrix.keys():

            targets_matrix[trial_type] = []
            print 'adding targets for trial type : circular ', trial_type

            #look for indexes of this trial type
            ind_trial_type = [ idx for (idx, msg) in enumerate(task[trial_accept_reject_idx-np.ones((len(trial_accept_reject_idx),),dtype=np.int)]['trial_type']) if msg == trial_type]
            
            subgoal_names[trial_type] =  OrderedDict() 

            for n, idx in enumerate(ind_trial_type):

                target_position[n] = task[trial_accept_reject_idx[idx]]['plant_pos']
          
            # Approach A: only 4 points will be defined and the vel_end vectors will be hardcoded to the tangential vectors to the circular trajectory
            for n in range(len(target_position)):
                pos = pd.Series(target_position[n], pos_states)
                vel = pd.Series(np.zeros(len(vel_states)), vel_states)

                target = pd.concat([pos,vel], axis = 0)

                if n == 0:
                    target_rest = pd.concat([pos,vel], axis = 0)
                if n == 1:
                    targets_matrix[trial_type] = target
                elif n != 0:
                    targets_matrix[trial_type] = pd.concat([targets_matrix[trial_type], target], axis = 1, ignore_index = True)
            
                subgoal_names[trial_type][n] =  OrderedDict() 
                subgoal_names[trial_type][n] =  ['circular']

            targets_matrix[trial_type] = pd.concat([targets_matrix[trial_type], target_rest], axis = 1, ignore_index = True)    
            targets_matrix[trial_type] = pd.concat([targets_matrix[trial_type], targets_matrix[trial_type], targets_matrix[trial_type]], axis = 1, ignore_index = True)    
            # for k in np.arange(4,12,1):
            #     subgoal_names[trial_type][4] =  ['circular']
            subgoal_names[trial_type][4] =  ['circular']
            subgoal_names[trial_type][5] =  ['circular']
            subgoal_names[trial_type][6] =  ['circular']
            subgoal_names[trial_type][7] =  ['circular']
            subgoal_names[trial_type][8] =  ['circular']
            subgoal_names[trial_type][9] =  ['circular']
            subgoal_names[trial_type][10] =  ['circular']
            subgoal_names[trial_type][11] =  ['circular']


# --------------------------------------------------------- SEQUENCE ------------------------------------------------------------------------
elif args.task_type == 'sequence':
    for n, idx in enumerate(trial_accept_reject_idx):

        trial_type = task[idx-1]['trial_type']
    
        if trial_type not in targets_matrix.keys():
            targets_matrix[trial_type] = []
            print 'adding targets for trial type : sequence ', trial_type

            #look for indexes of this trial type
            ind_trial_type = [ idx for (idx, msg) in enumerate(task[trial_accept_reject_idx-np.ones((len(trial_accept_reject_idx),),dtype=np.int)]['trial_type']) if msg == trial_type]
            
            for n, idx in enumerate(ind_trial_type):
                target_position[n] = task[trial_accept_reject_idx[idx]]['plant_pos']
            
            for n in range(len(target_position)):                
                pos = pd.Series(target_position[n], pos_states)

                # rest position - last recorded position
                if n == len(target_position)-1: #last recorded postion
                    vel = pd.Series(np.zeros(len(vel_states)), vel_states)
                    target_rest = pd.concat([pos,vel], axis = 0)
                #transition and subtrials positions
                else:
                    pos_next= pd.Series(target_position[n+1], pos_states) #go to the following position recorded
                    
                    # not generalized for different plant_types; ToDo
                    # X, Y and psi
                    vel_dir = [np.sign(pos_next['aa_px']-pos['aa_px']), np.sign(pos_next['aa_py']-pos['aa_py'])] #calculate the direction of angular velocity in psi to move towards the next target position
                    mod = np.sqrt((pos_next['aa_px']-pos['aa_px'])**2 + (pos_next['aa_py']-pos['aa_py'])**2)
                    vel_dir_psi = np.sign(pos_next['aa_ppsi']-pos['aa_ppsi']) #calculate the direction of angular velocity in psi to move towards the next target position
                    vel_mod_x = 3*np.abs((pos_next['aa_px']-pos['aa_px'])) / mod
                    vel_mod_y = 3*np.abs((pos_next['aa_py']-pos['aa_py']))/ mod
                    vel_mod_psi = np.deg2rad(5)
                    vel_x = pd.Series([vel_mod_x * vel_dir[0]], [vel_states[0]])
                    vel_y = pd.Series([vel_mod_y * vel_dir[1]], [vel_states[1]])
                    vel_psi = pd.Series([vel_mod_psi * vel_dir_psi], [vel_states[2]])

                    # rehand DoFs
                    vel_mod_rh = np.deg2rad(1)
                    # vel_dir_rh = np.sign(pos_next[rh_pos_states]-pos[rh_pos_states]) 
                    # to improve!! - nerea
                    vel_dir_rh = np.array(-1)
                    vel_rh = pd.Series([vel_mod_rh * vel_dir_rh], rh_vel_states)
                    
                    # vel_zeros = pd.Series(np.zeros(len(vel_states)-3), vel_states[3:])
                    vel = pd.concat([vel_x, vel_y, vel_psi, vel_rh])

                    #in the state where the grasping is occurring, make the vel state = 0 for all DoFs
                    if (n == 1):
                        vel = pd.Series(np.zeros(len(vel_states)), vel_states)
 
                    target = pd.concat([pos,vel], axis = 0)
                    if n == 0:
                        targets_matrix[trial_type] = target
                    elif n > 0:
                        targets_matrix[trial_type] = pd.concat([targets_matrix[trial_type], target], axis = 1, ignore_index = True)
            # append the rest target, the las one recorded
            targets_matrix[trial_type] = pd.concat([targets_matrix[trial_type], target_rest], axis = 1, ignore_index = True)    
            
           
print "target matrix : ",  targets_matrix
print "subgoal names : ",  targets_matrix['subgoal_names']

hdf.close()

subject_name  = models.TaskEntry.objects.get(id=args.id).subject.name

#add date and time to differenciate between targets_matrices
targets_matrix_file_name = 'targets_%s_%s_%s_%s' % (subject_name, int(args.id), args.task_type, time.strftime('%Y%m%d_%H%M'))

pkl_name = targets_matrix_file_name + '.pkl'


## Store a record of the data file in the database
storage_dir = '/storage/target_matrices'

if not os.path.exists(storage_dir):
    os.popen('mkdir -p %s' % storage_dir)

pickle.dump(targets_matrix, open(os.path.join(storage_dir, pkl_name), 'wb'))

#save mean times as a datafile in the database
# data_sys = models.System.make_new_sys('misc')
# data_sys.save_to_file( targets_matrix, pkl_name, obj_name=None, entry_id=-1)
dfs = models.DataFile.objects.filter(path = pkl_name)
# dfs = models.System.objects.filter(name=targets_matrix_file_name)
#import pdb; pdb.set_trace()

if len(dfs) == 0:
    data_sys = models.System.make_new_sys('misc')
    # data_sys.path = pkl_name
    data_sys.name = targets_matrix_file_name
    data_sys.entry = models.TaskEntry.objects.using(db_name).get(id=int(args.id))
    #data_sys.entry = models.TaskEntry.objects.using(db_name).get(id=int(15261))
    data_sys.save_to_file( targets_matrix, pkl_name, obj_name=None, entry_id=-1)
    #data_sys.save_to_file( targets_matrix, pkl_name, obj_name=None, entry_id=15261)
    #data_sys.save_to_file(targets_matrix, pkl_name, obj_name=None, entry_id=int(args.id))

    # # df = models.Decoder()
    # data_sys.path = pkl_name
    # data_sys.name = decoder_name
    # data_sys.entry = models.TaskEntry.objects.using(db_name).get(id=id)
    # #df.entry = models.TaskEntry.objects.using(db_name).get(id=954)
    # df.save()

elif len(dfs) == 1:
    pass # no new data base record needed
elif len(dfs) > 1:
     print "Warning: More than one targets_matrix with the same name! File will be overwritten but name wont show up twice at the database"
     







