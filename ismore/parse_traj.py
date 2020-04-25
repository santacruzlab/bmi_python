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
from scipy.signal import lfilter

plt.close('all')

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

    
    # code below will create a dictionary of trajectories, indexed by trial_type
    traj = RefTrajectories()

    # idxs into task_msgs corresponding to instances when the task entered the 
    # 'trial' state
    trial_start_msg_idxs = [idx for (idx, msg) in enumerate(task_msgs[:]['msg']) if msg == 'trial']

    # iterate over trials
    for msg_idx in trial_start_msg_idxs:

        # task iteration at which this trial started
        idx_start = task_msgs[msg_idx]['time']

        trial_type = task[idx_start]['trial_type']

        # only save one trajectory for each trial type (the first one)
        if trial_type not in traj:
            print 'adding trajectory for trial type', trial_type
            
            traj[trial_type] = dict()

            # task iteration at which this trial ended
            idx_end = task_msgs[msg_idx+1]['time'] - 1

            # actual start and end times of this trial 
            ts_start = task[idx_start]['ts']  # secs
            ts_end   = task[idx_end]['ts']    # secs

            traj[trial_type]['ts_start'] = ts_start
            traj[trial_type]['ts_end']   = ts_end


            # save task data
            idxs = [idx for idx in range(len(task[:])) if idx_start <= idx <= idx_end]
            traj[trial_type]['task'] = task[idxs]

            if INTERPOLATE_TRAJ:
                # finely-spaced vector of time-stamps onto which we will interpolate armassist and rehand data
                ts_step = 0.010  # seconds (equal to 10 ms)
                ts_interp = np.arange(ts_start, ts_end, ts_step)
                df_ts_interp = pd.DataFrame(ts_interp, columns=['ts'])

            # save armassist data
            if aa_flag:
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
                            #smooth_state_data = lfilter(b= x_coeffs, a=[1], x = state_data) #andrea
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

                traj[trial_type]['armassist'] = df_aa

            # save rehand data
            if rh_flag:
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
                        interp_fn = interp1d(ts_data, state_data, bounds_error = False, fill_value=0)
                        interp_state_data = interp_fn(ts_interp)
                        df_tmp = pd.DataFrame(interp_state_data, columns=[state])
                        df_rh  = pd.concat([df_rh, df_tmp], axis=1)
        
                else:
                    df_rh1 = pd.DataFrame(rehand[idxs]['data'],       columns=rh_pos_states+rh_vel_states)
                    df_rh2 = pd.DataFrame(rehand[idxs]['ts_arrival'], columns=['ts'])
                    df_rh  = pd.concat([df_rh1, df_rh2], axis=1)

                traj[trial_type]['rehand'] = df_rh

            # also save armassist+rehand data into a single combined dataframe
            if INTERPOLATE_TRAJ:
                df_traj = df_ts_interp.copy()

                if aa_flag:
                    for state in aa_pos_states + aa_vel_states:
                        df_traj = pd.concat([df_traj, df_aa[state]], axis=1)
                
                if rh_flag:
                    for state in rh_pos_states + rh_vel_states:
                        df_traj = pd.concat([df_traj, df_rh[state]], axis=1)
                
                traj[trial_type]['traj'] = df_traj
    hdf.close()
    return traj
    

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Parse ArmAssist and/or ReHand \
        trajectories from a .hdf file (corresponding to a task for recording or \
        playing back trajectories) and save them to a .pkl file. Interpolates \
        trajectories from a "record" task, but not for a "playback" task.')
    parser.add_argument('id', help='Task entry id from which to parse trajectories')
    parser.add_argument('--temp', help='', action="store_true")
    args = parser.parse_args()
    
    
    # load task, and armassist and/or rehand data from hdf file
    te = dbfn.TaskEntry(int(args.id), dbname='default')


    hdf = te.hdf

    traj = parse_trajectories(te.hdf)


    plt.show()

    
    if not args.temp:
        ## Store a record of the data file in the database
        from db.tracker import models
        
        ref_traj_dir = '/storage/rawdata/ref_trajectories'
        # get the 'ref trajeectories' system
        try:
            sys_ = models.System.objects.get(name='ref_trajectories')
        except:
            sys_ = models.System()
            sys_.name = 'ref_trajectories'
            sys_.path = ref_traj_dir
            sys_.save()
        
        sys_ = models.System.objects.get(name='ref_trajectories')
        
        if not os.path.exists(ref_traj_dir):
            os.popen('mkdir -p %s' % ref_traj_dir)
        
        pkl_name = 'parsed_trajectories_%s.pkl' % args.id

        pickle.dump(traj, open(os.path.join(ref_traj_dir, pkl_name), 'wb'))
        
        
        # Store a link to the data file
        # Search the database for data files of the same name
        dfs = models.DataFile.objects.filter(system__name='ref_trajectories', path=pkl_name)
        
        if len(dfs) == 0:
            df = models.DataFile()
            df.path = pkl_name
            df.system = sys_
            df.entry = models.TaskEntry.objects.get(id=int(args.id))
            df.archived = False
            df.local = True
            df.save()
        elif len(dfs) > 1:
            print "More than one datafile with the same name recorded in the database! fix manually!"

