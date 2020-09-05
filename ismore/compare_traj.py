import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from ismore.common_state_lists import *
from utils.util_fns import norm_vec
from utils.constants import *

from ismore.parse_traj import parse_trajectories
import plotutil

from db import dbfunctions as dbfn
from db.tracker import models

from ismore import analysis
import types


# parse command line arguments
parser = argparse.ArgumentParser(description='Plot a recorded reference \
    trajectory from the saved reference .pkl file, and plot the corresponding \
    playback trajectory from the saved playback .pkl file.')
parser.add_argument('id', help='Task entry id from which to parse trajectories')
parser.add_argument('trial_type', help='E.g., Blue, Red, "Brown to Red", etc.')
args = parser.parse_args()

id = int(args.id)
trial_type = args.trial_type


plot_closest_idx_lines = False
plot_xy_aim_lines      = False
plot_psi_aim_lines     = False

dbname = 'default'

# playback task
#te = dbfn.TaskEntry(id, dbname='tubingen')
te = dbfn.TaskEntry(id, dbname= dbname)

# Load ref trajectories of the played back traj 
#traj_ref_rec = models.DataFile.objects.using('tubingen').get(id=te.ref_trajectories)
traj_ref_rec = models.DataFile.objects.using(dbname).get(id=te.ref_trajectories)

traj_ref = traj_ref_rec.get()


# Calc playback trajectories
traj_pbk = parse_trajectories(te.hdf, INTERPOLATE_TRAJ=False)

aa_flag = 'armassist' in traj_pbk[trial_type]
rh_flag = 'rehand' in traj_pbk[trial_type]

if aa_flag:
    aa_ref = traj_ref[trial_type]['armassist']
    aa_pbk = traj_pbk[trial_type]['armassist']
    # print "length of aa_ref:", len(aa_ref)
    # print "length of aa_pbk:", len(aa_pbk)
    # print "length of aa_ref (secs):", aa_ref['ts'][aa_ref.index[-1]] - aa_ref['ts'][0]
    # print "length of aa_pbk (secs):", aa_pbk['ts'][aa_pbk.index[-1]] - aa_pbk['ts'][0]

if rh_flag:
    rh_ref = traj_ref[trial_type]['rehand']
    rh_pbk = traj_pbk[trial_type]['rehand']
    # print "length of rh_ref:", len(rh_ref)
    # print "length of rh_pbk:", len(rh_pbk)
    # print "length of rh_ref (secs):", rh_ref['ts'][rh_ref.index[-1]] - rh_ref['ts'][0]
    # print "length of rh_pbk (secs):", rh_pbk['ts'][rh_pbk.index[-1]] - rh_pbk['ts'][0]


#play backed trajectory info
task_pbk       = traj_pbk[trial_type]['task']
aim_pos        = task_pbk['aim_pos']
idx_aim        = task_pbk['idx_aim']
idx_aim_psi    = task_pbk['idx_aim_psi']
#print idx_aim_psi
plant_pos      = task_pbk['plant_pos']
plant_vel      = task_pbk['plant_vel']
command_vel    = task_pbk['command_vel'] #already filtered
command_vel_raw    = task_pbk['command_vel_raw']
playback_vel   = task_pbk['playback_vel']
idx_traj       = task_pbk['idx_traj']



try:
    emg_vel    = task_pbk['emg_vel']
    emg_flag = True
except:
    emg_flag = False
    print 'no emg data'
    pass




#emg_flag = np.any(task_pbk['emg_vel'])
#if emg_flag: 
#    emg_vel    = task_pbk['emg_vel']



############
# PLOTTING #
############

task_tvec = task_pbk['ts'] - task_pbk['ts'][0]

color_ref = 'red'
color_pbk = 'blue'

tight_layout_kwargs = {
    'pad':   0.5,
    'w_pad': 0.5,
    'h_pad': 0.5,
}


##figure only for X and Y
if aa_flag:

    plt.figure()

    ## Plot the trajectories in X and Y
    plt.plot(aa_ref['aa_px'], aa_ref['aa_py'], color='blue', label='reference')
    plt.plot(aa_pbk['aa_px'], aa_pbk['aa_py'], color='green', label='achieved')
    plt.plot(aa_pbk['aa_px'][0:50], aa_pbk['aa_py'][0:50], color='red', label='start') #start
    plt.plot(aa_ref['aa_px'][0:50], aa_ref['aa_py'][0:50], color='red', label='start') #start
    plt.show()
   #plotutil.legend(base_traj_axes[0,0])
   #plotutil.clean_up_ticks(base_traj_axes)


##in a big window, first plot X and Y

if aa_flag:

    plt.figure(figsize=(14,8))

    ## Plot the trajectories
    base_traj_axes = plotutil.subplots2(1, 1, right_fig_offset_frac=0.75, bottom_fig_frac_offset=0.75, aspect=1)
    base_traj_axes[0,0].plot(aa_ref['aa_px'], aa_ref['aa_py'], color='blue', label='reference')
    base_traj_axes[0,0].plot(aa_pbk['aa_px'], aa_pbk['aa_py'], color='green', label='achieved')
    #base_traj_axes[0,0].plot(aa_pbk['aa_px'][0:50], aa_pbk['aa_py'][0:50], color='red', label='start') #start
    base_traj_axes[0,0].plot(aa_ref['aa_px'][0:100], aa_ref['aa_py'][0:100], color='red', label='start') #start

    ## Plot the trajectories
    base_traj_axes = plotutil.subplots2(1, 1, right_fig_offset_frac=0.75, bottom_fig_frac_offset=0.75, aspect=1)
    base_traj_axes[0,0].plot(aa_ref['aa_px'], aa_ref['aa_py'], color='blue', label='reference')
    base_traj_axes[0,0].plot(aa_pbk['aa_px'], aa_pbk['aa_py'], color='green', label='achieved')
    base_traj_axes[0,0].plot(aa_pbk['aa_px'][0:50], aa_pbk['aa_py'][0:50], color='red', label='start') #start
    base_traj_axes[0,0].plot(aa_ref['aa_px'][0:100], aa_ref['aa_py'][0:100], color='red', label='start') #start


    plotutil.legend(base_traj_axes[0,0])
    plotutil.clean_up_ticks(base_traj_axes)


### plot the achieved trajectory against the aimed trajectory
aim_pos = traj_pbk[trial_type]['task']['aim_pos']
plant_pos = traj_pbk[trial_type]['task']['plant_pos']
ts = traj_pbk[trial_type]['task']['ts'].ravel()
ts -= ts[0]

sl = slice(None, -1)

aim_point_axes = plotutil.subplots2(1, 7, border=0.05, left_fig_offset_frac=0.25, bottom_fig_frac_offset=0.75)

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


##plot all trajectories in the first line separately for each DoF

for k in range(n_joints):
    aim_point_axes[0,k].plot(ts[sl], aim_pos[sl,k], label='aim')
    aim_point_axes[0,k].plot(ts[sl], plant_pos[sl,k], label='ach.')

plotutil.legend(aim_point_axes[0,0], loc=(-1,0.5))
plotutil.clean_up_ticks(aim_point_axes)

plotutil.ylabel(aim_point_axes[0,0], 'x-pos (cm)')
plotutil.ylabel(aim_point_axes[0,1], 'y-pos (cm)')
plotutil.ylabel(aim_point_axes[0,2], 'psi angle (deg)')
plotutil.ylabel(aim_point_axes[0,3], ' thumb ang (rad)')
plotutil.ylabel(aim_point_axes[0,4], ' index ang (rad)')
plotutil.ylabel(aim_point_axes[0,5], ' 3-fing ang (rad)')
plotutil.ylabel(aim_point_axes[0,6], ' prono ang (rad)')
plotutil.xlabel(aim_point_axes, 'Time (s)')

plotutil.set_title(aim_point_axes[0,3], 'Aimed versus achieved positions')


### plot velocities
vel_axes = plotutil.subplots2(3, 7, border=0.05, top_fig_frac_offset=0.3, y=3) # subplot, 3 lines, 7 columns
grid = (4, 1)





##old way, taking values from the traj and not the task 
traj_reference = traj_ref[trial_type]['traj'] ##parsed trajectory

for i, state in enumerate(vel_states):
    
    #convert to degrees for psi angle
    if state == 'aa_ppsi':
        scale = rad_to_deg    
    else:
        scale = 1
    scale = 1

    #first row: reference vel (the velocity profile computed from the recorded reference trajectory)
    ref_ts = traj_reference['ts']
    ref_ts -= ref_ts[0]
    #ref_vel = analysis.pos_to_vel(traj_reference[state].ravel(), rate=1./np.mean(np.diff(ref_ts)))
    ref_vel = traj_reference[state].ravel()
    vel_axes[0,i].plot(ref_ts, ref_vel)


    #second row: achieved velocity in the played back trajectory
    #plant_vel = analysis.pos_to_vel(plant_pos[:,i], rate=1./np.mean(np.diff(ts))) # Andrea: pos_to_vel computes the velocity and also filters the data. 
    
    vel_axes[1,i].plot(task_tvec, plant_vel[:,i])
    #vel_axes[1,i].plot(task_tvec, plant_vel[:, i])


    # third row: playback vel and command vel (if there is EMG influence they are not the same, otherwise they are the same)
    ax = vel_axes[2, i]
    # ax.set_title(state + ' command velocity')
    
    ax.plot(task_tvec, scale * playback_vel[:, i], color=color_ref)
    ax.plot(task_tvec, scale * command_vel_raw[:, i], color='green')

    # try:
    #     ax.plot(task_tvec, scale * emg_vel[:, i], color=color_pbk)
    # except:
    #     pass




plotutil.set_title(vel_axes[0,i], state)
plotutil.row_label(vel_axes[0,0], 'Reference velocity', line_offset=2, fontproperties=plotutil.bold_font)
plotutil.row_label(vel_axes[1,0], 'Achieved velocity', line_offset=2, fontproperties=plotutil.bold_font)
plotutil.row_label(vel_axes[2,0], 'Command velocity raw', line_offset=2, fontproperties=plotutil.bold_font)


plotutil.ylabel(vel_axes[:,0:2], 'cm/s')
plotutil.ylabel(vel_axes[0,1], 'cm')
plotutil.ylabel(vel_axes[:,2], 'deg/s')
plotutil.ylabel(vel_axes[:,3:7], 'rad/s')
plotutil.xlabel(vel_axes, 'Time (s)')

plotutil.clean_up_ticks(vel_axes)
plt.draw()
if emg_flag:
    for i, state in enumerate(pos_states):
        # ax = plt.subplot2grid(grid, (i+1, 0))
        plt.figure()
        # ax.set_title(state + ' command velocity')
    
        #convert to degrees for psi angle
        if state == 'aa_ppsi':
            scale = rad_to_deg    
        else:
            scale = 1
        scale = 1

#     aim_point_axes[0,k].plot(ts[sl], aim_pos[sl,k], label='aim')
#     aim_point_axes[0,k].plot(ts[sl], plant_pos[sl,k], label='ach.')

# plotutil.legend(aim_point_axes[0,0], loc=(-1,0.5))

        try:
            plt.plot(task_tvec, scale * emg_vel[:, i], color=color_pbk, label='emg')
        except:
            pass

        #plt.plot(task_tvec, scale * emg_vel[:, i], color=color_pbk, label='emg')

        plt.plot(task_tvec, scale * playback_vel[:, i], color=color_ref, label='playback')
        plt.plot(task_tvec, scale * command_vel[:, i], color='green', label='command')
        
        plt.show()



###extra plot to compare reference vel, command vel and achieved vel




te.close_hdf()