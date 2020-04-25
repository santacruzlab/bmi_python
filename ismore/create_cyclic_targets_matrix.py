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
parser.add_argument('task_type', help='Task type for which to generate target positions') # should be either linear or circular
parser.add_argument('--temp', help='', action="store_true")
args = parser.parse_args()
    
    
# load task, and armassist and/or rehand data from hdf file
db_name = 'default'
te = dbfn.TaskEntry(int(args.id), dbname= db_name)
hdf = te.hdf
# hdf_name = '/storage/rawdata/hdf/test20151120_20_te717.hdf' 
# hdf = tables.openFile(hdf_name)

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
# subtargets = OrderedDict()
#idx where the trial was acceped --> the exo was placed in the goal target position 
trial_accept_reject_idx = [idx for (idx, msg) in enumerate(task[:]['trial_accept_reject']) if msg == 'accept']
target_position = OrderedDict()

# ------------------------------------------------------------ CYCLIC LINEAR ------------------------------------------------------------------------
# rest (i.e. initial) position is always the last one in the targets_matrix of each trial_type (i.e. circular, linear_blue, linear_red, etc.. trial types)
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
           
            for n, idx in enumerate(ind_trial_type):
                target_position[n] = task[trial_accept_reject_idx[idx]]['plant_pos']

            xy_mod = 15

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
                    

# ------------------------------------------------------------ CYCLIC CIRCULAR ------------------------------------------------------------------------  
elif args.task_type == 'circular':

    for n, idx in enumerate(trial_accept_reject_idx):

        trial_type = task[idx-1]['trial_type']
     
        if trial_type not in targets_matrix.keys():

            targets_matrix[trial_type] = []
            print 'adding targets for trial type : circular ', trial_type

            #look for indexes of this trial type
            ind_trial_type = [ idx for (idx, msg) in enumerate(task[trial_accept_reject_idx-np.ones((len(trial_accept_reject_idx),),dtype=np.int)]['trial_type']) if msg == trial_type]
            
            for n, idx in enumerate(ind_trial_type):

                target_position[n] = task[trial_accept_reject_idx[idx]]['plant_pos']
            # print "len target pos: ", len(target_position)
            # print "target_position : ", target_position
            # Approach A: only 4 points will be defined and the vel_end vectors will be hardcoded to the tangential vectors to the circular trajectory
            for n in range(len(target_position)):
                pos = pd.Series(target_position[n], pos_states)
                
                if n == len(target_position)-1: #last point recorded
                    pos_next = pd.Series(target_position[0], pos_states)
                else:
                    pos_next= pd.Series(target_position[n+1], pos_states)        

                if n == 0: #n = 0 always will correspond to te rest position (initial position)
                    vel_dir = [-1, 0]
                    vel_dir_psi = np.sign(pos_next['aa_ppsi']-pos['aa_ppsi']) #calculate the direction of angular velocity in psi to move towards the next target position
                    vel_mod = 10
                    vel_mod_psi = np.deg2rad(5)
                    vel_xy = pd.Series([vel_mod *vel_value for vel_value in vel_dir], vel_states[:2])
                    vel_psi = pd.Series([vel_mod_psi * vel_dir_psi], [vel_states[2]])
                    vel_zeros = pd.Series(np.zeros(len(vel_states)-3), vel_states[3:])
                    vel = pd.concat([vel_xy, vel_psi, vel_zeros])
                    target_rest = pd.concat([pos,vel], axis = 0)
                elif n == 1:
                    vel_dir = [0, 1]
                elif n == 2:
                    vel_dir = [1, 0]
                elif n == 3:
                    vel_dir = [0, -1]
                vel_dir_psi = np.sign(pos_next['aa_ppsi']-pos['aa_ppsi']) #calculate the direction of angular velocity in psi to move towards the next target position
                vel_mod = 10
                vel_mod_psi = np.deg2rad(5)
                vel_xy = pd.Series([vel_mod *vel_value for vel_value in vel_dir], vel_states[:2])
                vel_psi = pd.Series([vel_mod_psi * vel_dir_psi], [vel_states[2]])
                vel_zeros = pd.Series(np.zeros(len(vel_states)-3), vel_states[3:])
                vel = pd.concat([vel_xy, vel_psi, vel_zeros])
                target = pd.concat([pos,vel], axis = 0)

                if n == 1:
                    targets_matrix[trial_type] = target
                elif n != 0:
                    targets_matrix[trial_type] = pd.concat([targets_matrix[trial_type], target], axis = 1, ignore_index = True)
                # import pdb; pdb.set_trace()
            targets_matrix[trial_type] = pd.concat([targets_matrix[trial_type], target_rest], axis = 1, ignore_index = True)    
    # import pdb; pdb.set_trace()    
    # Approach B: as many points as desired can be used to define the circular trajectory. The direction of the vel_end vectors will be computed as the direction from the target_position to the next target in the trajectory
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

# ------------------------------------------------------------ SEQUENCE ------------------------------------------------------------------------
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


                if n == len(target_position)-1:
                    pos_next = pd.Series(target_position[0], pos_states)
                else:
                    pos_next= pd.Series(target_position[n+1], pos_states)        


                if n == 0: #n = 0 always will correspond to te rest position (initial position)
                    mod = np.sqrt((pos_next['aa_px']-pos['aa_px'])**2 + (pos_next['aa_py']-pos['aa_py'])**2)
                    
                    vel_dir = [np.sign(pos_next['aa_px']-pos['aa_px']), np.sign(pos_next['aa_py']-pos['aa_py'])] #calculate the direction of angular velocity in psi to move towards the next target position
                    vel_dir_psi = np.sign(pos_next['aa_ppsi']-pos['aa_ppsi']) #calculate the direction of angular velocity in psi to move towards the next target position

                    vel_mod_x = np.abs(2*(pos_next['aa_px']-pos['aa_px']) / mod)
                    vel_mod_y = np.abs(2*(pos_next['aa_py']-pos['aa_py']) / mod)
                    vel_mod_psi = np.deg2rad(1)
               
                    vel_x = pd.Series([vel_mod_x * vel_dir[0]], [vel_states[0]])
                    vel_y = pd.Series([vel_mod_y * vel_dir[1]], [vel_states[1]])
                    vel_psi = pd.Series([vel_mod_psi * vel_dir_psi], [vel_states[2]])
                    vel_zeros = pd.Series(np.zeros(len(vel_states)-3), vel_states[3:])
                    vel = pd.concat([vel_x, vel_y, vel_psi, vel_zeros])

                    #nerea
                    #set the velocity in all DoFs to 0 since when it reaches the rest position we want the exo to be completenly stopped
                    vel = pd.Series(np.zeros(len(vel_states)), vel_states)

                    target_rest = pd.concat([pos,vel], axis = 0)
                elif n != 0:
                    vel_dir = [np.sign(pos_next['aa_px']-pos['aa_px']), np.sign(pos_next['aa_py']-pos['aa_py'])]


                mod = np.sqrt((pos_next['aa_px']-pos['aa_px'])**2 + (pos_next['aa_py']-pos['aa_py'])**2)
                vel_dir_psi = np.sign(pos_next['aa_ppsi']-pos['aa_ppsi']) #calculate the direction of angular velocity in psi to move towards the next target position
                vel_mod_x = 3*(pos_next['aa_px']-pos['aa_px']) / mod
                vel_mod_y = 3*(pos_next['aa_py']-pos['aa_py']) / mod
                vel_mod_psi = np.deg2rad(5)
                vel_x = pd.Series([vel_mod_x * vel_dir[0]], [vel_states[0]])
                vel_y = pd.Series([vel_mod_y * vel_dir[1]], [vel_states[1]])
                vel_psi = pd.Series([vel_mod_psi * vel_dir_psi], [vel_states[2]])
                vel_zeros = pd.Series(np.zeros(len(vel_states)-3), vel_states[3:])
                vel = pd.concat([vel_x, vel_y, vel_psi, vel_zeros])
                
                #testing nerea
                vel = pd.Series(np.zeros(len(vel_states)), vel_states)

                #point in which the object is grasped
                if n == 2:
                    vel = pd.Series(np.zeros(len(vel_states)), vel_states)

                target = pd.concat([pos,vel], axis = 0)
             
                if n == 1:
                    targets_matrix[trial_type] = target
                elif n > 1:

                    targets_matrix[trial_type] = pd.concat([targets_matrix[trial_type], target], axis = 1, ignore_index = True)
            # import pdb; pdb.set_trace()
            targets_matrix[trial_type] = pd.concat([targets_matrix[trial_type], target_rest], axis = 1, ignore_index = True)    
               





#import pdb; pdb.set_trace() 
print "target matrix : ",  targets_matrix
hdf.close()


subject_name  = models.TaskEntry.objects.get(id=args.id).subject.name
#subject_name = 'testing'
# targets_matrix_file_name = 'targets_matrix_%s_%s_%s' % (subject_name, int(args.id), args.task_type)
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
    # data_sys.save_to_file( targets_matrix, pkl_name, obj_name=None, entry_id=-1)
    data_sys.save_to_file( targets_matrix, pkl_name, obj_name=None, entry_id=int(args.id))


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
     





