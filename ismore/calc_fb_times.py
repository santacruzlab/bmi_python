import argparse
import os
import pickle
import numpy as np


from collections import defaultdict
from common_state_lists import *
from ismore.common_state_lists import *

from db import dbfunctions as dbfn
from riglib.plants import RefTrajectories

import argparse
import os
import tables
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import pickle
from db import dbfunctions as dbfn

from ismore.common_state_lists import *
from utils.constants import *

from riglib.plants import RefTrajectories
import matplotlib.pyplot as plt

from riglib.filter import Filter

from db.tracker import models





def parse_trajectories(hdf, INTERPOLATE_TRAJ=True):
    #hdf = tables.open_file(hdf_name)

    task      = hdf.root.task
    task_msgs = hdf.root.task_msgs
    aa_flag = 'armassist' in hdf.root
    rh_flag = 'rehand' in hdf.root
    if aa_flag:
        armassist = hdf.root.armassist
    if rh_flag:
        rehand = hdf.root.rehand


    # idxs into task_msgs corresponding to instances when the task entered the 
    # 'trial' state
    trial_start_msg_idxs = [idx for (idx, msg) in enumerate(task_msgs[:]['msg']) if msg == 'trial']

    traj = [None]*len(trial_start_msg_idxs)

    # iterate over trials
    for k, msg_idx in enumerate(trial_start_msg_idxs):
        # task iteration at which this trial started
        idx_start = task_msgs[msg_idx]['time']

        trial_type = task[idx_start]['trial_type']

        # only save one trajectory for each trial type (the first one)

        if 1: #trial_type not in traj:
            #print 'adding trajectory for trial type', trial_type
            
            traj[k] = dict()
            traj[k]['trial_type'] = trial_type

            # task iteration at which this trial ended
            idx_end = task_msgs[msg_idx+1]['time'] - 1
            
            # actual start and end times of this trial 
            ts_start = task[idx_start]['ts']  # secs
            ts_end   = task[idx_end]['ts']    # secs

            traj[k]['ts_start'] = ts_start
            traj[k]['ts_end']   = ts_end

            # save task data
            idxs = [idx for idx in range(len(task[:])) if idx_start <= idx <= idx_end]
            traj[k]['task']= task[idxs]
            
            traj[k]['plant_pos']= task[idxs]['plant_pos']

            traj[k]['plant_vel']= task[idxs]['plant_vel']
            #df_1 = pd.DataFrame(task[idxs]['plant_pos'])#, columns = aa_pos_states+rh_pos_states)
            #df_2 = pd.DataFrame(task[idxs]['ts'])#, columns=['ts'])
            #df   = pd.concat([df_1, df_2], axis=1)

            #traj[k]['all'] = df

            #print traj[k]['all']


            if INTERPOLATE_TRAJ:
                # finely-spaced vector of time-stamps onto which we will interpolate armassist and rehand data
                ts_step = 0.010  # seconds (equal to 10 ms)
                ts_interp = np.arange(ts_start, ts_end, ts_step)
                df_ts_interp = pd.DataFrame(ts_interp, columns=['ts'])

            # save armassist data
            if 0:#aa_flag:
                idxs = [i for (i, ts) in enumerate(armassist[:]['ts_arrival']) if ts_start <= ts <= ts_end]
                    
                if INTERPOLATE_TRAJ:
                    # add one more idx to the beginning and end, if possible
                    if idxs[0] != 0:
                        idxs = [idxs[0]-1] + idxs
                    if idxs[-1] != len(armassist[:])-1:
                        idxs = idxs + [idxs[-1]+1]

                    df_aa = df_ts_interp.copy()
                    vel_data = dict()
                    for state in aa_pos_states:
                        ts_data    = armassist[idxs]['ts_arrival']
                        state_data = armassist[idxs]['data'][state]

                        ## make state_data constant at beginning and end
                        n_const_samples = 60 # about 1 s for armassist
                        state_data = np.hstack([[state_data[0]]*n_const_samples, state_data, [state_data[-1]]*n_const_samples])
                        t_samp = np.mean(np.diff(ts_data))
                        ts_data = np.hstack([ts_data[0]+np.arange(n_const_samples)[::-1]*-t_samp, ts_data, ts_data[-1]+np.arange(n_const_samples)*t_samp])

                        # linear interpolation
                        if state == 'aa_ppsi':
                            # y_coeffs = np.hstack([1, [-0.9**k for k in range(1, 20)]])
                            # y_coeffs /= np.sum(y_coeffs)
                            x_coeffs = np.array([0.9**k for k in range(20)])
                            x_coeffs /= np.sum(x_coeffs)

                            lpf = Filter(b=x_coeffs, a=[1])
                            smooth_state_data = lpf(state_data)
                            interp_fn = interp1d(ts_data, smooth_state_data)
                            interp_state_data = interp_fn(ts_interp)

                            # noisy_vel = np.hstack([0, np.diff(interp_state_data)])
                            # vel_lpf = Filter(b=x_coeffs, a=[1])
                            # vel_data[state] = vel_lpf(noisy_vel)
                            
                            support_size = 40
                            vel_data[state] = np.hstack([np.zeros(support_size), interp_state_data[support_size:] - interp_state_data[:-support_size]]) / (ts_step*support_size)
                            
                        else:
                            # spline interpolation
                            from scipy.interpolate import splrep, splev
                            tck = splrep(ts_data, state_data, s=7)
                            interp_state_data = splev(ts_interp, tck)
                            vel_data[state] = splev(ts_interp, tck, der=1)

                        # plt.figure()
                        # plt.subplot(2, 1, 1)
                        # plt.hold(True)
                        # plt.plot(ts_interp, interp_state_data)
                        # plt.plot(ts_data, state_data)
                        # plt.subplot(2, 1, 2)
                        # plt.plot(ts_interp, vel_data[state])
                        
                        # plt.title('%s %s' % (trial_type, state))

                        df_tmp = pd.DataFrame(interp_state_data, columns=[state])
                        df_aa  = pd.concat([df_aa, df_tmp], axis=1)

                    # Add interpolated velocity data to the table
                    from itertools import izip
                    for pos_state, vel_state in izip(aa_pos_states, aa_vel_states):
                        df_tmp = pd.DataFrame(vel_data[pos_state], columns=[vel_state])
                        df_aa  = pd.concat([df_aa, df_tmp], axis=1)

                else:
                    df_aa1 = pd.DataFrame(armassist[idxs]['data'],       columns=aa_pos_states)
                    df_aa2 = pd.DataFrame(armassist[idxs]['ts_arrival'], columns=['ts'])
                    df_aa  = pd.concat([df_aa1, df_aa2], axis=1)

                traj[k]['armassist'] = df_aa

            # save rehand data
            if 0:#rh_flag:
                idxs = [i for (i, ts) in enumerate(rehand[:]['ts_arrival']) if ts_start <= ts <= ts_end]
                    
                if INTERPOLATE_TRAJ:
                    # add one more idx to the beginning and end, if possible
                    if idxs[0] != 0:
                        idxs = [idxs[0]-1] + idxs
                    if idxs[-1] != len(rehand[:])-1:
                        idxs = idxs + [idxs[-1]+1]

                    df_rh = df_ts_interp.copy()
                    for state in rh_pos_states+rh_vel_states:
                        ts_data    = rehand[idxs]['ts_arrival']
                        state_data = rehand[idxs]['data'][state]
                        interp_fn = interp1d(ts_data, state_data)
                        interp_state_data = interp_fn(ts_interp)
                        df_tmp = pd.DataFrame(interp_state_data, columns=[state])
                        df_rh  = pd.concat([df_rh, df_tmp], axis=1)
        
                else:
                    df_rh1 = pd.DataFrame(rehand[idxs]['data'],       columns=rh_pos_states+rh_vel_states)
                    df_rh2 = pd.DataFrame(rehand[idxs]['ts_arrival'], columns=['ts'])
                    df_rh  = pd.concat([df_rh1, df_rh2], axis=1)

                traj[k]['rehand'] = df_rh

            # also save armassist+rehand data into a single combined dataframe
            if INTERPOLATE_TRAJ:
                df_traj = df_ts_interp.copy()

                if aa_flag:
                    for state in aa_pos_states + aa_vel_states:
                        df_traj = pd.concat([df_traj, df_aa[state]], axis=1)
                
                if rh_flag:
                    for state in rh_pos_states + rh_vel_states:
                        df_traj = pd.concat([df_traj, df_rh[state]], axis=1)
                
                traj[k]['traj'] = df_traj
    
    return traj

from ismoretasks import targetsB1, targetsB2, targetsF1_F2
def get_task_type(trial_type):
    if trial_type in targetsB1:
        task_type = 'B1'
        n_subtasks = 2
    elif trial_type in targetsB2:
        task_type = 'B2'
        n_subtasks = 2
    elif trial_type in targetsF1_F2:
        task_type = 'F1'
        n_subtasks = 3
    return task_type#, n_subtasks


def _set_subgoals(task_type, traj,pos_states=aa_pos_states):
    trial_type = traj['trial_type']
    
    if task_type == 'B1':
        pos_states = aa_pos_states
        #pos_traj = np.array(traj['armassist'][pos_states])
        target_margin_rest = np.array([3, 3])
        fails = 0

        pos_traj = np.array(traj['plant_pos'])
        pos_traj_diff = pos_traj - pos_traj[0]
        max_xy_displ_idx = np.argmax(map(np.linalg.norm, pos_traj_diff[:,0:2]))

        
        # distal_goal = pos_traj[max_xy_displ_idx]
        # proximal_goal = pos_traj[len(pos_traj)-1]

        # subgoals = [distal_goal, proximal_goal]
        #reached_rest = False

        # if 'max_xy_displ_idx' not in locals(): 
        #     max_xy_displ_idx = 0
        # Find the first index in which the exo is within the rest area
        for kk in range(max_xy_displ_idx, len(pos_traj)-1):
            if np.all(np.abs(pos_traj[kk,0:2]-pos_traj[len(pos_traj)-1,0:2]) < target_margin_rest):# and reached_rest == False:
                target_goal_rest_idx = kk #+ 10 #Add 10 to compute the time when the exo is stopped in the target instead of the very first point in which the exo reaches the target area
                #reached_rest = True         
                break

        # Both in B1 and B2 the first target will always be reached with this algorithm
        if 'target_goal_rest_idx' in locals(): 
            subgoal_inds = [max_xy_displ_idx, target_goal_rest_idx]
            subgoal_times = traj['task'][subgoal_inds]['ts'].ravel() - traj['task'][0]['ts']
            
        else:     
            subgoal_inds = [max_xy_displ_idx, np.nan]
            subgoal_times = traj['task'][subgoal_inds[0]]['ts'].ravel() - traj['task'][0]['ts']
            subgoal_times = [subgoal_times,np.nan]  
            fails = 1
        # if max_xy_displ_idx == 0 and 'target_goal_rest_idx' in locals(): 
        #     subgoal_inds = [np.nan, target_goal_rest_idx]
        #     subgoal_times = traj['task'][target_goal_rest_idx]['ts'].ravel() - traj['task'][0]['ts']
        #     subgoal_times = [np.nan,subgoal_times]
        # elif max_xy_displ_idx == 0 and 'target_goal_rest_idx' not in locals():        
        #     subgoal_inds = [np.nan, np.nan]
        #     subgoal_times = [np.nan,np.nan] 
        # elif max_xy_displ_idx != 0 and 'target_goal_rest_idx' in locals(): 
        #     subgoal_inds = [max_xy_displ_idx, target_goal_rest_idx]
        #     subgoal_times = traj['task'][subgoal_inds]['ts'].ravel() - traj['task'][0]['ts']
        # elif max_xy_displ_idx != 0 and 'target_goal_rest_idx' not in locals():          
        #     subgoal_inds = [max_xy_displ_idx, np.nan]
        #     subgoal_times = traj['task'][subgoal_inds[0]]['ts'].ravel() - traj['task'][0]['ts']
        #     subgoal_times = [subgoal_times,np.nan]  
        


                #print traj['task'][subgoal_inds]['ts']
        #subgoal_times = traj['task'][subgoal_inds]['ts'].ravel() - traj['task'][0]['ts']
        
        #subgoal_times = traj['armassist'].ix[subgoal_inds]['ts'].ravel() - traj['armassist'].ix[0]['ts']
    
    elif task_type == 'B2':

        pos_traj = (traj['plant_pos'])

        target_margin = np.deg2rad(10)
        fails = 0

        if trial_type == 'Up':
            grasp_goal_idx = np.argmin(pos_traj[:,6].ravel()) #pronosupination
            # Find the first index in which the exo is within the rest area
            for kk in range(grasp_goal_idx, len(pos_traj)-1):
                if np.all(np.abs(pos_traj[kk,6]-pos_traj[len(pos_traj)-1,6]) < target_margin):
                    target_goal_rest_idx = kk #+ 10 #Add 10 to compute the time when the exo is stopped in the target instead of the very first point in which the exo reaches the target area
                    break
        elif trial_type == 'Down':
            grasp_goal_idx = np.argmax(pos_traj[:,6].ravel()) #pronosupination
            # Find the first index in which the exo is within the rest area
            for kk in range(grasp_goal_idx, len(pos_traj)-1):
                if np.all(np.abs(pos_traj[kk,6]-pos_traj[len(pos_traj)-1,6]) < target_margin):
                    target_goal_rest_idx = kk #+ 10 #Add 10 to compute the time when the exo is stopped in the target instead of the very first point in which the exo reaches the target area
                    break
        elif trial_type == 'Point':
            grasp_goal_idx = np.argmin(pos_traj[:,4].ravel()) # index
            # Find the first index in which the exo is within the rest area
            for kk in range(grasp_goal_idx, len(pos_traj)-1):
                if np.all(np.abs(pos_traj[kk,4]-pos_traj[len(pos_traj)-1,4]) < target_margin):
                    target_goal_rest_idx = kk #+ 10 #Add 10 to compute the time when the exo is stopped in the target instead of the very first point in which the exo reaches the target area
                    break
        elif trial_type == 'Pinch':
            grasp_goal_idx = np.argmax(pos_traj[:,4].ravel()) #index
            # Find the first index in which the exo is within the rest area
            for kk in range(grasp_goal_idx, len(pos_traj)-1):
                if np.all(np.abs(pos_traj[kk,4]-pos_traj[len(pos_traj)-1,4]) < target_margin):
                    target_goal_rest_idx = kk #+ 10 #Add 10 to compute the time when the exo is stopped in the target instead of the very first point in which the exo reaches the target area
                    break
        elif trial_type == 'Grasp':
            grasp_goal_idx = np.argmin(pos_traj[:,4].ravel()) #index
            # Find the first index in which the exo is within the rest area
            for kk in range(grasp_goal_idx, len(pos_traj)-1):
                if np.all(np.abs(pos_traj[kk,4]-pos_traj[len(pos_traj)-1,4]) < target_margin):
                    target_goal_rest_idx = kk #+ 10 #Add 10 to compute the time when the exo is stopped in the target instead of the very first point in which the exo reaches the target area
                    break
        # Both in B1 and B2 the first target will always be reached
        if 'target_goal_rest_idx' not in locals():
            subgoal_inds = [grasp_goal_idx, np.nan]
            subgoal_times = traj['task'][subgoal_inds[0]]['ts'].ravel() - traj['task'][0]['ts']
            subgoal_times = [subgoal_times,np.nan]
            fails = 1
        else: 
            subgoal_inds = [grasp_goal_idx, target_goal_rest_idx]
            subgoal_times = traj['task'][subgoal_inds]['ts'].ravel() - traj['task'][0]['ts']
        

        #subgoal_inds = [grasp_goal_idx, len(traj['rehand'])-2]
        #subgoal_times = traj['rehand'].ix[subgoal_inds]['ts'].ravel() - traj['rehand'].ix[0]['ts']

    elif task_type == 'F1':
        pos_states = aa_pos_states
        # fit the largest triangle possible to the trajectory
        #pos_traj = np.array(traj['armassist'][pos_states])
        pos_traj = np.array(traj['plant_pos'])

        pos_traj_diff = pos_traj - pos_traj[0]

        
        target1 =  True
        target2 =  True
        targetrest =  True
        failed_target1 = 0
        failed_target2 = 0
        failed_rest = 0

        # Method 1: Compute Tsuccess times based on local minima using distance to rest
        # diff = map(np.linalg.norm, pos_traj_diff[:,0:2])
        
        # #print len(diff)
        # # if trial_type == 'Green to Blue':
        # #     print trial_type
        # #     print (pos_traj[:,0:2])
        # local_minima = np.zeros(len(pos_traj_diff))
        # T = len(pos_traj_diff)
        # support = 50 #10 good value for 2554 ref trajectory #135
        # for k in range(support, T-support):
        #     local_minima[k] = np.all(diff[k-support:k+support] <= (diff[k]+0.4)) 
 
        # local_minima[diff < 14] = 0 # exclude anything closer than 5 cm

        # local_minima_inds, = np.nonzero(local_minima)


        # subgoal_inds = np.hstack([local_minima_inds, len(pos_traj)-2])
        
        # idx_ok = (np.diff(subgoal_inds) > 10)
        # idx_ok = np.hstack([True, idx_ok])
        
        # subgoal_inds = subgoal_inds[idx_ok] 
      
        # subgoals = [pos_traj[idx] for idx in subgoal_inds]
        

        # subgoal_times = traj['task'][subgoal_inds]['ts'].ravel() - traj['task'][0]['ts']
        # #subgoal_times = traj['armassist'].ix[subgoal_inds]['ts'].ravel() - traj['armassist'].ix[0]['ts']
        # #assert len(subgoal_inds) == 3

        # Method 2: Define target area (based on x and y coordinates) for each target type
        target_goal_Red = np.array([28, 35])
        target_goal_Blue = np.array([54, 33])
        target_goal_Green = np.array([39, 45])
        target_goal_Brown = np.array([52, 46])
        target_margin = np.array([7, 5]) #np.array([2., 2., np.deg2rad(20),  np.deg2rad(10), np.deg2rad(10), np.deg2rad(10), np.deg2rad(10)])
        target_margin_rest = np.array([7, 5])

        if trial_type == 'Red to Brown' or trial_type == 'Blue to Brown' or trial_type == 'Green to Brown':
            target_goal_pos_2 = target_goal_Brown
        if trial_type == 'Red to Green' or trial_type == 'Blue to Green' or trial_type == 'Brown to Green':
            target_goal_pos_2 = target_goal_Green
        if trial_type == 'Red to Blue' or trial_type == 'Brown to Blue' or trial_type == 'Green to Blue':
            target_goal_pos_2 = target_goal_Blue
        if trial_type == 'Brown to Red' or trial_type == 'Blue to Red' or trial_type == 'Green to Red':
            target_goal_pos_2 = target_goal_Red
        if trial_type == 'Red to Brown' or trial_type == 'Red to Blue' or trial_type == 'Red to Green':
            target_goal_pos_1 = target_goal_Red
        if trial_type == 'Blue to Brown' or trial_type == 'Blue to Red' or trial_type == 'Blue to Green':
            target_goal_pos_1 = target_goal_Blue
        if trial_type == 'Green to Brown' or trial_type == 'Green to Blue' or trial_type == 'Green to Red':
            target_goal_pos_1 = target_goal_Green
        if trial_type == 'Brown to Red' or trial_type == 'Brown to Blue' or trial_type == 'Brown to Green':
            target_goal_pos_1 = target_goal_Brown


        # Find the first index in which the exo is within the target1 area
        for kk in range(0, len(pos_traj)-1):
            if np.all(np.abs(pos_traj[kk,0:2]-target_goal_pos_1) < target_margin):
                target_goal_1_idx = kk #+ 10 #Add 10 to compute the time when the exo is stopped in the target instead of the very first point in which the exo reaches the target area
                break
        #np.where(np.abs(pos_traj[:,0:2]-target_goal_pos_1) <= target_margin)
        if 'target_goal_1_idx' not in locals(): 
            target1 = False
            target_goal_1_idx = 0
            
             
        # Find the first index in which the exo is within the target2 area
        for kk in range (target_goal_1_idx+30,len(pos_traj)-1): # Find the moment when the second target was reached imposing the condition that it should happen 30 time points after target1 at least
            if np.all(np.abs(pos_traj[kk,0:2]-target_goal_pos_2) < target_margin):
                target_goal_2_idx = kk  #Add 10 to compute the time when the exo is stopped in the target instead of the very first point in which the exo reaches the target area
                break
        if 'target_goal_2_idx' not in locals(): 
            target2 = False
            target_goal_2_idx = target_goal_1_idx
            

        # Find the first index in which the exo is within the rest area
        for kk in range(target_goal_2_idx, len(pos_traj)-1):
            if np.all(np.abs(pos_traj[kk,0:2]-pos_traj[len(pos_traj)-1,0:2]) < target_margin_rest):
                target_goal_rest_idx = kk #+ 10 #Add 10 to compute the time when the exo is stopped in the target instead of the very first point in which the exo reaches the target area

                break
       
        if 'target_goal_rest_idx' not in locals() or target_goal_rest_idx == target_goal_2_idx: 
            target_goal_rest_idx = target_goal_2_idx
            targetrest = False
         
        

        subgoal_inds = np.array([target_goal_1_idx, target_goal_2_idx, target_goal_rest_idx])        
        #subgoals = [pos_traj[idx] for idx in subgoal_inds]        
        subgoal_times = traj['task'][subgoal_inds]['ts'].ravel() - traj['task'][0]['ts']

        

        if target1 == False: 
            subgoal_inds = np.array([np.nan, subgoal_inds[1], subgoal_inds[2]])      
            subgoal_times[0] = np.nan
            failed_target1 = 1
            
        if target2 == False: 
            subgoal_inds = np.array([subgoal_inds[0], np.nan, subgoal_inds[2]])   
            subgoal_times[1] = np.nan
            failed_target2 = 1
            
        if targetrest == False: 
            subgoal_inds = np.array([subgoal_inds[0], subgoal_inds[1], np.nan]) 
            subgoal_times[2] = np.nan
            failed_rest = 1
        
        fails = failed_target1 + failed_target2 + failed_rest   

        
        # subgoal_times[0] = traj['task'][subgoal_inds[0]]['ts'].ravel() - traj['task'][0]['ts']
        # subgoal_times[1] = traj['task'][subgoal_inds[1]]['ts'].ravel() - traj['task'][subgoal_inds[0]]['ts'].ravel()
        # subgoal_times[2] = traj['task'][subgoal_inds[2]]['ts'].ravel() - traj['task'][subgoal_inds[1]]['ts'].ravel()
        
        # Method 3: find points based on velocities
        # pos_vel = np.array(traj['plant_vel'])
        
        # #find indices in which vel Y is zero
            
        # velXY = map(np.linalg.norm, pos_vel[:,0:2])


        #vel_0 = np.where(np.array(diffvel) <= 0.1)
        # vel_0 = np.where(velXY <=0.3)
        # print 'vel_0'
        # print vel_0


        # diffvel = np.array(np.diff(vel_0)) 
        # diffvel_copy = diffvel
        # idx_max_diff0 = np.argmax(diffvel)
        
        # diffvel[:,idx_max_diff0] = 0

        # idx_max_diff1 = np.argmax(diffvel)

        # diffvel[:,idx_max_diff1] = 0

        # idx_max_diff2 = np.argmax(diffvel)
        # print 'diffvel'
        # print diffvel_copy

        # print idx_max_diff0+1
        # print idx_max_diff1+1
        # print idx_max_diff2+1

        # plt.figure()
        # plt.plot(pos_traj[:,0],pos_traj[:,1])  
        # #plt.plot(np.abs(pos_vel[:,1]) ) 
        # plt.show()               
        # #plt.close('all')


    return subgoal_times, fails


# parse command line arguments
parser = argparse.ArgumentParser(description='Plot a recorded reference \
#     trajectory from the saved reference .pkl file, and plot the corresponding \
#     playback trajectory from the saved playback .pkl file.')
parser.add_argument('id', help='Task entry id from which to parse trajectories')
args = parser.parse_args()
id = int(args.id)

#id = 2637 #3344 TF-F1 #2560 AS-F1 #2668 #FS-F1 #2634#b2 #3243 #F1 #2623 #B1


te = dbfn.TaskEntry(id)
trajectories = parse_trajectories(te.hdf, INTERPOLATE_TRAJ=True)
subgoal_times = defaultdict(list)
subgoal_times_abs = defaultdict(list)
subgoal_fails = defaultdict(list)


for traj_pbk in trajectories:
    trial_type = traj_pbk['trial_type']
    task_type = get_task_type(trial_type)
    
    subgoal_times_trial = np.hstack([ 0, _set_subgoals(task_type, traj_pbk)[0]])
    subgoal_times_trial_abs = _set_subgoals(task_type, traj_pbk)[0]
    failed_rest_amount = _set_subgoals(task_type, traj_pbk)[1]
    #subgoal_times_trial = np.hstack([0, _set_subgoals(task_type, traj_pbk)])
    subgoal_times[trial_type].append(subgoal_times_trial)
    subgoal_times_abs[trial_type].append(subgoal_times_trial_abs)
    subgoal_fails[trial_type].append(failed_rest_amount)


mean_subgoal_times = dict()
mean_subgoal_times_abs = dict()
mean_subgoal_fails = dict()
import numpy, scipy.io
storage_dir = '/storage/feedback_times'
for trial_type in subgoal_times:
    subgoal_times_tr = np.diff(np.vstack(subgoal_times[trial_type]), axis=1)
    subgoal_times_tr_abs = np.vstack(subgoal_times_abs[trial_type])
    subgoal_fails_tr = np.vstack(subgoal_fails[trial_type])
    
    mean_subgoal_times_abs[trial_type] = np.nanmean(subgoal_times_tr_abs, axis=0)
    # print'times_absolute'
    # print mean_subgoal_times_abs
    mean_subgoal_times[trial_type] = np.nanmean(subgoal_times_tr, axis=0)
    # print'times_relative'
    # print mean_subgoal_times
    mean_subgoal_fails[trial_type] = np.sum(subgoal_fails_tr, axis=0)
    print trial_type
    trial_name = str(trial_type)
    fb_file_name = 'feedback_times_%s_%s' % (str(id),trial_name) 
    scipy.io.savemat(os.path.join(storage_dir, fb_file_name), mdict={'mean_subgoal_fails': mean_subgoal_fails[trial_type],'mean_subgoal_times': mean_subgoal_times[trial_type], 'mean_subgoal_times_abs': mean_subgoal_times_abs[trial_type]})
    # print 'fails'
    # print mean_subgoal_fails
    #dict_data.update({'trial_type': mean_subgoal_fails[trial_type],'mean_subgoal_times': mean_subgoal_times[trial_type], 'mean_subgoal_times_abs': mean_subgoal_times_abs[trial_type]})


te.close_hdf()


# if task_type == 'B1' or 'B2':
#     scipy.io.savemat('/storage/feedback_times/mydata.mat', mdict={'mean_subgoal_fails': mean_subgoal_fails[0],'mean_subgoal_times': mean_subgoal_times[0], 'mean_subgoal_times_abs': mean_subgoal_times_abs[0]})
# else:
#     scipy.io.savemat('/storage/feedback_times/mydata.mat', mdict={'mean_subgoal_fails': mean_subgoal_fails[trial_type],'mean_subgoal_times': mean_subgoal_times[trial_type], 'mean_subgoal_times_abs': mean_subgoal_times_abs[trial_type]})



#scipy.io.savemat('/storage/feedback_times/mydata.mat', dict={'mean_subgoal_fails': mean_subgoal_fails})
#save mean times


subject_name  = models.TaskEntry.objects.get(id=id).subject.name
task_name = str(task_type)
fb_file_name = 'feedback_times_%s_%s' % (subject_name, task_name)
pkl_name = fb_file_name + '.pkl'
#hdf_name = fb_file_name + '.hdf'


# ## Store a record of the data file in the database


storage_dir = '/storage/feedback_times'

if not os.path.exists(storage_dir):
    os.popen('mkdir -p %s' % storage_dir)

pickle.dump(mean_subgoal_times, open(os.path.join(storage_dir, pkl_name), 'wb'))
#hdf.dump(mean_subgoal_times, open(os.path.join(storage_dir, hdf_name), 'wb'))


#save mean times as a datafile in the database
from db.tracker import models
data_sys = models.System.make_new_sys('misc')

data_sys.save_to_file( mean_subgoal_times, pkl_name, obj_name=None, entry_id=-1)
