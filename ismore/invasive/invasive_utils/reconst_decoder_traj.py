import numpy as np
import pickle
from riglib.filter import Filter
from riglib.bmi.goal_calculators import ZeroVelocityGoal_ismore
from scipy.signal import butter,lfilter
from ismore import ismore_bmi_lib
import tables
from ismore.invasive.make_global_armassist_hull import global_hull

# Path: 
# 7742 -- constant assist (xy = 0, ang = 0.1)

class RerunDecoding(object):
    
    def __init__(self, hdf_file, decoder_file, safety_grid_file, xy_assist, 
        ang_assist, attractor_speed):

        hdf = tables.openFile(hdf_file)

        self.plant_pos = hdf.root.task[:]['plant_pos']
        self.plant_vel = hdf.root.task[:]['plant_vel']

        self.target = hdf.root.task[:]['target_pos']
        spike_counts = hdf.root.task[:]['spike_counts'] 

        self.spike_counts = np.array(spike_counts, dtype=np.float64)
        
        self.internal_state = hdf.root.task[:]['internal_decoder_state']
        self.decoder_state = hdf.root.task[:]['decoder_state']

        self.raw_command = hdf.root.task[:]['command_vel_raw']
        self.pre_safe_command = hdf.root.task[:]['command_vel_sent_pre_safety']
        self.proc_command = hdf.root.task[:]['command_vel_sent']
        self.pre_drive_state = hdf.root.task[:]['pre_drive_state']

        self.state_list = hdf.root.task_msgs[:]

        self.dec = pickle.load(open(decoder_file))        
        self.drives_neurons = self.dec.drives_neurons;
        self.drives_neurons_ix0 = np.nonzero(self.drives_neurons)[0][0]
        self.update_bmi_ix = np.nonzero(np.diff(np.squeeze(self.internal_state[:, self.drives_neurons_ix0, 0])))[0]+1
        
        self.xy_assist = hdf.root.task[:]['xy_assist_level']
        self.ang_assist = hdf.root.task[:]['ang_assist_level']
        self.hdf = hdf

        self.safety_grid = pickle.load(open(safety_grid_file))
        #hdf_global = tables.openFile('/Users/preeyakhanna/Dropbox/Carmena_Lab/SpainBMI/ismore_analysis/data/hud120171010_72_te7721.hdf')
        #pts = hdf_global.root.task[:]['plant_pos'][:, [0, 1, 2]]
        #self.safety_grid.global_hull.hull_xy._points = pts[:, [0, 1]]
        #self.safety_grid.global_hull.hull3d._points = pts
        #self.safety_grid.global_hull.hull_xy.simplices = self.safety_grid.global_hull.hull_xy.vertices.copy()
        #self.safety_grid.global_hull.hull3d.simplices = self.safety_grid.global_hull.hull3d.vertices.copy()

        fs_synch = 20
        nyq   = 0.5 * fs_synch
        cuttoff_freq  = 5 / nyq
        bpf_kin_coeffs = butter(4, cuttoff_freq, btype='low')

        self.accel_lim_armassist = np.inf
        self.accel_lim_psi = np.inf
        self.accel_lim_rehand = np.inf

        self.command_lpfs = dict()
        for state in ['aa_vx', 'aa_vy', 'aa_vpsi','rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']: 
            self.command_lpfs[state] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1]) # low-pass filter to smooth out command velocities

        self.target = hdf.root.task[:]['target_pos']
        self.goal_calculator = ZeroVelocityGoal_ismore(ismore_bmi_lib.SSM_CLS_DICT['IsMore'], 
            pause_states = ['rest', 'wait', 'instruct_rest', 'instruct_trial_type'])

        asst_kwargs = {
            'call_rate': 20,
            'xy_cutoff': 3,
            'speed':    'high',
        }

        self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT['IsMore'](**asst_kwargs)
        self.assist_level = np.array([xy_assist, ang_assist])
        self.rh_pfings = [[0, 'rh_pthumb'], [1, 'rh_pindex'], [2, 'rh_pfing3']]

        self.spike_counts = hdf.root.task[:]['spike_counts'][:, :, 0]
        self.attractor_speed = attractor_speed
    
    def run_decoder(self):
        '''
        Summary: method to use the 'predict' function in the decoder object
        Input param: spike_counts: unbinned spike counts in iter x units x 1
        Input param: cutoff:  cutoff in iterations
        '''
        spike_counts = self.spike_counts

        T = spike_counts.shape[0]
        decoded_state = []
        
        spike_accum = np.zeros_like(spike_counts[0,:])
        dec_last = np.zeros_like(self.dec.predict(spike_counts[0,:]))
        self.prev_vel_bl_aa = np.zeros((3, ))*np.nan
        self.prev_vel_bl_rh = np.zeros((4, ))*np.nan
        
        tot_spike_accum = np.zeros_like(spike_counts[0,:])-1
        self.dec.filt._init_state()
        self.state = 'wait'
        self.sent_vel  = [np.zeros((7, ))]
        self.raw_vel = [np.zeros((7, ))]
        self.pre_safe_vel = [np.zeros((7, ))]
        self.raw_pos = [np.zeros((7, ))]
        
        for t in range(T):
            spike_accum = spike_accum+spike_counts[t,:]

            if t in self.state_list[:]['time']:
                ix = np.nonzero(self.state_list[:]['time']==t)[0]
                self.state = self.state_list[ix[-1]]['msg']

            if t in self.update_bmi_ix:
                # Assister
                target_state = self.get_target_BMI_state(self.plant_pos[t-1, :], self.plant_vel[t-1, :], self.target[t, :])
                current_state = np.hstack((self.plant_pos[t-1, :], self.plant_vel[t-1, :], [1]))
                assist_kwargs = self.assister(current_state, target_state[:,0].reshape(-1,1), 
                    self.assist_level, mode=self.state)

                # 
                if self.dec.zscore == True:
                    spike_accum = (spike_accum - self.dec.mFR)/self.dec.sdFR
                spike_accum[np.isnan(spike_accum)] = 0

                # Call decoder
                dec_new = self.dec.predict(spike_accum, **assist_kwargs)
                spike_accum = np.zeros_like(spike_counts[0,:])
            else:
                dec_new = self.dec.get_state()

            vel_bl = dec_new[7:14]
            if np.any(np.isnan(vel_bl)):
                vel_bl[np.isnan(vel_bl)] = 0
            self.dec.filt.state.mean[7:14, 0] = np.mat(vel_bl.copy()).T
            self.raw_vel.append(vel_bl.copy())
            self.raw_pos.append(dec_new[:7])
            v = vel_bl.copy()
            vel_post_drive = self.ismore_plant_drive(v, t)

            # Send imaginary velocity command
            # Set decoder['q'] to plant pos: 
            self.dec['q'] = self.plant_pos[t]
            self.sent_vel.append(vel_post_drive)

        self.raw_vel = np.vstack((self.raw_vel))
        self.sent_vel = np.vstack((self.sent_vel))
        self.raw_pos = np.vstack((self.raw_pos))
        self.pre_safe_vel = np.vstack((self.pre_safe_vel))

    def ismore_plant_drive(self, vel_bl, t):

        current_state = self.pre_drive_state[t, :]

        ### Velocity processing in plants.drive ###
        vel_bl_aa = vel_bl[0:3]
        vel_bl_rh = vel_bl[3:7]

        ### Accel Limit Velocitites ###
        ### Accel Limit Velocitites ###
        ### Accel Limit Velocitites ###

        if not np.all(np.isnan(np.hstack((self.prev_vel_bl_aa, self.prev_vel_bl_rh)))):
            aa_output_accel = vel_bl_aa - self.prev_vel_bl_aa
            rh_output_accel = vel_bl_rh - self.prev_vel_bl_rh

            ### AA XY ###
            for i in np.arange(2):
                if aa_output_accel[i] > self.accel_lim_armassist:
                    vel_bl_aa[i] = self.prev_vel_bl_aa[i] + self.accel_lim_armassist
                elif aa_output_accel[i] < -1*self.accel_lim_armassist:
                    vel_bl_aa[i] = self.prev_vel_bl_aa[i] - self.accel_lim_armassist
            
            ### AA PSI ###
            if aa_output_accel[2] > self.accel_lim_psi:
                vel_bl_aa[2] = self.prev_vel_bl_aa[2] + self.accel_lim_psi
            elif aa_output_accel[2] < -1*self.accel_lim_psi:
                vel_bl_aa[2] = self.prev_vel_bl_aa[2] - self.accel_lim_psi

            ### RH All ###
            for i in np.arange(4):
                if rh_output_accel[i] > self.accel_lim_rehand:
                    vel_bl_rh[i] = self.prev_vel_bl_rh[i] + self.accel_lim_rehand
                elif rh_output_accel[i] < -1*self.accel_lim_rehand:
                    vel_bl_rh[i] = self.prev_vel_bl_rh[i] - self.accel_lim_rehand

        ### LPF Filter Velocities ###
        ### LPF Filter Velocities ###
        ### LPF Filter Velocities ###

        for s, state in enumerate(['aa_vx', 'aa_vy', 'aa_vpsi']):
            tmp = vel_bl_aa[s].copy()
            vel_bl_aa[s] = self.command_lpfs[state](tmp)
            if np.isnan(vel_bl_aa[s]):
                vel_bl_aa[s] = 0

        for s, state in enumerate(['rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']):
            tmp = vel_bl_rh[s].copy()
            vel_bl_rh[s] = self.command_lpfs[state](tmp)
            if np.isnan(vel_bl_rh[s]):
                vel_bl_rh[s] = 0


        #If the next position is outside of safety then damp velocity to only go to limit: 
        self.pre_safe_vel.append(np.hstack((vel_bl_aa, vel_bl_rh)))

        pos_pred = current_state + 0.05*np.hstack((vel_bl_aa, vel_bl_rh))
        pos_pred_aa = pos_pred[0:3]
        pos_pred_rh = pos_pred[3:7]
    
        #Make sure predicted AA PX, AA PY within bounds: 
        xy_change = True

        x_tmp = self.safety_grid.is_valid_pos(pos_pred_aa[[0, 1]])
        if x_tmp == False:
            #d_pred = np.linalg.norm(self.safety_grid.interior_pos - pos_pred_aa[[0, 1]])
            #d_curr = np.linalg.norm(self.safety_grid.interior_pos - self.plant_pos[t-1, [0, 1]])
            
            # if d_pred < d_curr:
            #     xy_change = True
            # else:
            #     xy_change = False
            #     vel_bl_aa[[0, 1]] = 0

            current_pos = current_state[[0, 1]]
            d_to_valid, pos_valid = self.safety_grid.dist_to_valid_point(current_pos)
            vel_bl_aa[[0, 1]] = self.attractor_speed*(pos_valid - current_pos)/0.05
            pos_pred_aa[[0, 1]] = current_pos + 0.05*vel_bl_aa[[0, 1]]
            #print 'plant adjust: ', vel_bl_aa[self.aa_plant.aa_xy_ix], pos_pred_aa[self.aa_plant.aa_xy_ix]
            xy_change = True

        # Make sure AA Psi within bounds: 
        # If X/Y ok
        if xy_change:
            mn, mx = self.safety_grid.get_minmax_psi(pos_pred_aa[[0, 1]])
            predx, predy, predpsi = pos_pred_aa[[0, 1, 2]]

        # If x/y not ok: 
        else:
            mn, mx = self.safety_grid.get_minmax_psi(self.plant_pos[t-1, [0, 1]])
            predx, predy, predpsi = self.plant_pos[t-1, [0, 1, 2]]

        # Set psi velocity : 
        psi_ok = False
        if np.logical_and(pos_pred_aa[2] >= mn, pos_pred_aa[2] <= mx):
            # Test if globally ok: 
            global_ok = self.safety_grid.global_hull.hull3d.find_simplex(np.array([predx, predy, pos_pred_aa[2]]))
            if global_ok:
                psi_ok = True

        if psi_ok == False:
            vel_bl_aa[2] = 0
            #print 'stop psi vel: ', mn, mx, pos_pred_aa[self.aa_plant.aa_psi_ix]

        # Make sure RH Prono within bounds (if SSM is only RH, use settings.starting_pos for AAPX, AAPY)

        # If X/Y ok
        if xy_change:
            mn, mx = self.safety_grid.get_minmax_prono(pos_pred_aa[[0, 1]])

        # If x/y not ok or not moving bc not part of state pace : 
        else:
            mn, mx = self.safety_grid.get_minmax_prono(self.plant_pos[t-1, [0, 1]])

        # Set prono velocity : 
        if np.logical_and(pos_pred_rh[3] >= mn, pos_pred_rh[3] <= mx):
            pass
        else:
            tmp_pos = pos_pred_rh[3]
            if tmp_pos < mn:
                tmp = self.attractor_speed*(mn - tmp_pos)/0.05
            elif tmp_pos > mx:
                tmp = self.attractor_speed*(mn - tmp_pos)/0.05
            else:
                tmp = 0
            vel_bl_rh[3] = tmp
        
        # Assure RH fingers are within range: 
        for i, (ix, nm) in enumerate(self.rh_pfings):
            mn, mx = self.safety_grid.get_rh_minmax(nm)
            if np.logical_and(pos_pred_rh[ix] >= mn, pos_pred_rh[ix] <= mx):
                pass
            else:
                if pos_pred_rh[ix] > mx:
                    tmp = self.attractor_speed*(mx - pos_pred_rh[ix])/0.05
                elif pos_pred_rh[ix] < mn:
                    tmp = self.attractor_speed*(mn - pos_pred_rh[ix])/0.05
                else:
                    tmp = 0
                vel_bl_rh[ix] = tmp
        return np.hstack((vel_bl_aa, vel_bl_rh))

    def get_target_BMI_state(self, plant_pos, plant_vel, targ_pos, *args):
        '''Run the goal calculator to determine the current target state.'''
        current_state = np.hstack([plant_pos, plant_vel, 1])[:, None]
        if np.any(np.isnan(current_state)):
            current_state[np.isnan(current_state)] = 0

        data, solution_updated = self.goal_calculator(targ_pos, self.state, 
            **dict(current_state=current_state))
        
        target_state = data[0].reshape(-1, 1)
        if np.any(np.isnan(target_state)):
            target_state[np.isnan(target_state)] = 0        
        #target_state = np.hstack(( self.target_pos.values, [1]))
        #target_state = target_state.reshape(-1, 1)
        return np.tile(np.array(target_state).reshape(-1, 1), [1, self.dec.n_subbins])


import scipy.stats
import matplotlib.pyplot as plt
def plot_results(RRD):
    ### Raw Velocities ###
    for i in range(7):
        s, ii, r, p, e = scipy.stats.linregress(RRD.raw_vel[:-1, i], RRD.raw_command[:, i])
        print 'raw vel: ', i, ', r2 = ', r**2

    print ''
    print ''
    print ''

    ### Presafe Velocities ###
    for i in range(7):
        s, ii, r, p, e = scipy.stats.linregress(RRD.pre_safe_vel[:-1, i], RRD.pre_safe_command[:, i])
        print 'presafe vel: ', i, ', r2 = ', r**2
        #f, ax = plt.subplots()
        #ax.plot(RRD.pre_safe_vel[:, i])
        #ax.plot(RRD.pre_safe_command[:, i])

    print ''
    print ''
    print ''


    #### Proc Velocities ###
    for i in range(7):
        s, ii, r, p, e = scipy.stats.linregress(RRD.sent_vel[:-1, i], RRD.proc_command[:, i])
        print 'proc vel: ', i, ', r2 = ', r**2

        f, ax = plt.subplots()
        ax.plot(RRD.proc_command[:, i])
        ax.plot(RRD.sent_vel[:, i])
    print ''
    print ''
    print ''


wsxy = np.array([0.5, ])
wpsi = np.array([0.01])
wsang = np.array([0.005])
ws = np.hstack((wsxy.reshape(-1, 1), wsang.reshape(-1, 1), wsang.reshape(-1, 1)))

def sweep_W(R, ws=ws):
    ax = []
    ax2 = []

    for i in range(7):
        f, axi = plt.subplots()
        axi.plot(R.pre_safe_command[:, i], label='og, presafe')
        ax.append(axi)
        
        #f, axi = plt.subplots()
        axi.plot(R.raw_command[:, i], label='og, raw')
        #ax2.append(axi)
        
    for i, (xy, psi, ang) in enumerate(ws):
        R = change_W(R, xy, psi, ang)
        R = def_LPF(R, 5)

        R.run_decoder()
        
        for j in range(7):
            axi = ax[j]
            axi.plot(R.raw_vel[:, j], label=str(xy)+','+str(ang)+' raw ')

            #axi = ax2[j]
            axi.plot(R.pre_safe_vel[:, j], label=str(xy)+','+str(ang)+' presafe ')


def change_W(RRD, w_vel_xy, w_vel_psi, w_vel_ang):
    for i in [7, 8]:
        RRD.dec.filt.W[i, i] = w_vel_xy

    RRD.dec.filt.W[9, 9] = w_vel_psi

    for i in [10, 11, 12]:
        RRD.dec.filt.W[i, i] = w_vel_ang
    return RRD

def def_LPF(RRD, lpf_cutoff):
    fs_synch = 20
    nyq   = 0.5 * fs_synch
    cuttoff_freq  = lpf_cutoff / nyq
    bpf_kin_coeffs = butter(4, cuttoff_freq, btype='low')

    RRD.command_lpfs = dict()
    for state in ['aa_vx', 'aa_vy', 'aa_vpsi','rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']: 
        RRD.command_lpfs[state] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1]) # low-pass filter to smooth out command velocities
    return RRD

def compare_decoders(RRD):
    ### PLot filtered velocities of RRD 1 vs. unfiltered velocities of RRD 2
    for i in range(7):
        f, ax = plt.subplots()
        ax.plot(RRD.pre_safe_command[:, i], label='actual_pre_safe_command')
        ax.plot(RRD.raw_vel[:, i], label='predicted_pre_filt')
    print ''
    print ''
    print ''


