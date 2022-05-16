from riglib.bmi import kfdecoder
import numpy as np
from riglib.experiment import traits
from .bmimultitasks import SimpleEndpointAssister
import pickle
import os
from datetime import date
import copy
import random
import pandas as pd

import time 
from tracker.models import TaskEntry

class CurlFieldKalmanFilter(kfdecoder.KalmanFilter):
    def _calc_kalman_gain(self, P):
        '''
        see KalmanFilter._calc_kalman_gain
        '''
        K = super(CurlFieldKalmanFilter, self)._calc_kalman_gain(P)
        v_norm = np.linalg.norm(np.array(self.state.mean[3:6,0]))
        rot_angle_deg = self.rot_factor*v_norm
        theta = np.deg2rad(rot_angle_deg)

        R = np.mat([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        K[[0,2],:] = R * K[[0,2],:]
        K[[3,5],:] = R * K[[3,5],:]
        return K

class VisRotKalmanFilter(kfdecoder.KalmanFilter):
    def _calc_kalman_gain(self, P):
        '''
        see KalmanFilter._calc_kalman_gain
        '''
        K = super(VisRotKalmanFilter, self)._calc_kalman_gain(P)
        theta = np.deg2rad(self.rot_angle_deg)

        R = np.mat([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        K[[0,2],:] = R * K[[0,2],:]
        K[[3,5],:] = R * K[[3,5],:]

        fn = self.filename_KG
        cwd = os.path.abspath(os.getcwd())
        os.chdir('/media/samantha/ssd/storage/rawdata/bmi')
        with open(fn, 'ab') as f:
            pickle.dump(K, f)
            f.close()
        os.chdir(cwd)

        return K

class ShuffledKalmanFilter(kfdecoder.KalmanFilter):

    def _calc_kalman_gain(self, P):

        if self.shuffle_state == True:

            if self.flag == 0:
                '''Block 2 - First Update ONLY | Should only run once.'''
                print("SHUFFLE")

                self.flag = 1
                
                K = super(ShuffledKalmanFilter, self)._calc_kalman_gain(P)

                shuffleInds = self.shuffleInds

                print('ORIGINAL:',  K[[0,2,3,5],:][:,shuffleInds] ) 
                
                dfShuffle = pd.DataFrame({ 'Kx_pos':np.ravel((copy.deepcopy(K[0,shuffleInds]))),
                                           'Ky_pos':np.ravel((copy.deepcopy(K[2,shuffleInds]))),
                                           'Kx_vel':np.ravel((copy.deepcopy(K[3,shuffleInds]))),
                                           'Ky_vel':np.ravel((copy.deepcopy(K[5,shuffleInds])))}) 

                
                temp = dfShuffle.sample(frac=1).reset_index(drop=True).to_numpy().T
                

                #K[[0,2,3,5],:][:,shuffleInds] = temp #WHY DOES THIS NOT WORK???

                K[0, shuffleInds] = temp[0,:]
                K[2, shuffleInds] = temp[1,:]
                K[3, shuffleInds] = temp[2,:]
                K[5, shuffleInds] = temp[3,:]

                print('SHUFFLE:', K[[0,2,3,5],:][:,shuffleInds])

                self.shuffledK = K 
                self.shuffledInds = shuffleInds #saved in HDF file under 'hdf.root.task._v_attrs.indsToShuffle'

            elif self.flag == 1:
                '''REMAINING PERTURBATION: Block 2 (after the shuffle) and Block 3 | Maintain shuffled decoder from single instance of shuffling at start of Block 2.'''
                K = self.shuffledK
                #print('K1 == K3?: ', np.mean(self.shuffledK == self.baseline_decoder))

        elif self.shuffle_state == False:
            '''BLock 1 - Baseline Decoder'''
            
            try: 
                temp = self.flag
            except: 
                self.flag = None
            
            if (self.flag == None) or (self.flag == 0): 
                K = super(ShuffledKalmanFilter, self)._calc_kalman_gain(P)
                self.baseline_decoder = K #This will overwrite the baseline decoder until it reaches steady state (first few bins).
                self.flag = 0

            elif self.flag == 1:
                '''Block 4 - Washout | Reinstate intial decoder'''
                K = self.baseline_decoder

        
        fn = self.filename_KG
        cwd = os.path.abspath(os.getcwd())
        os.chdir('/media/samantha/ssd/storage/rawdata/bmi')
        with open(fn, 'ab') as f:
            pickle.dump(K, f)
            f.close()
        os.chdir(cwd)

        return K

from .bmimultitasks import BMIControlMulti, BMIResetting
class BMICursorKinematicCurlField(BMIResetting):
    rot_factor = traits.Float(10., desc='scaling factor from speed to rotation angle in degrees')
    def load_decoder(self):
        super(BMICursorKinematicCurlField, self).load_decoder()
        # Conver the KF to a curl-field generating KF
        dec = self.decoder
        filt = CurlFieldKalmanFilter(A=self.decoder.filt.A, W=self.decoder.filt.W, C=dec.filt.C, Q=dec.filt.Q, is_stochastic=dec.filt.is_stochastic)
        filt.C_xpose_Q_inv = dec.filt.C_xpose_Q_inv
        filt.C_xpose_Q_inv_C = dec.filt.C_xpose_Q_inv_C
        filt.rot_factor = self.rot_factor
        self.decoder.filt = filt

from riglib import plants
class CursorErrorClampPlant(plants.CursorPlant):
    def __init__(self, *args, **kwargs):
        super(CursorErrorClampPlant, self).__init__(*args, **kwargs)
        self.axis = None

    def set_intrinsic_coordinates(self, pt):
        if self.axis is None:
            self.position = pt
        else:
            origin, terminus = self.axis
            vec_to_pt = pt - origin
            vec_to_targ = terminus - origin

            scale = np.dot(vec_to_targ, vec_to_pt) / np.dot(vec_to_targ, vec_to_targ)
            self.position = scale*vec_to_targ + origin
        self.draw()

class CursorErrorClamp(object):
    exclude_parent_traits = ['plant_type']
    sequence_generators = ['center_out_error_clamp_infrequent']
    def __init__(self, *args, **kwargs):
        self.plant = CursorErrorClampPlant(endpt_bounds=(-25, 25, 0., 0., -14, 14))
        super(CursorErrorClamp, self).__init__(*args, **kwargs)

        #print('HS: CursorErrorClamp INIT Function')
        '''Naming convention based on task entry in Django interface. (Not pretty, but it works, I guess.)'''
        #(1) Obtain information for all entries.
        db_name='default'
        entry    = TaskEntry.objects.using(db_name)
        #(2) Determine the total number of entries.
        lenEntry = len(entry)
        #(3) Assign entry to be the current task entry.
        entry    = entry[lenEntry-1]
        #(4) Pull desired entry information.
        subj     = entry.subject.name[:4].lower()
        te_id    = entry.id
        date     = time.strftime('%Y%m%d')
        self.fn = "{}{}_te{}_KG_SHKF.pkl".format(subj, date, te_id) #SHKF: SHuffled Kalman Filter
        

    def _parse_next_trial(self):
        if isinstance(self.next_trial, dict):
            for key in self.next_trial:
                setattr(self, '_gen_%s' % key, self.next_trial[key])

        self.targs = self._gen_targs
        if self._gen_error_clamp:
            self.plant.axis = (self._gen_targs[0], self._gen_targs[1])
        else:
            self.plant.axis = None

        self.reportstats['Perturbation'] = str(self._gen_curl)
        self.reportstats['Error clamp'] = str(self._gen_error_clamp)
        self.reportstats['Block Type'] = str(self._gen_block_type)

    def init(self):
        self.add_dtype('error_clamp', 'i', (1,))
        self.add_dtype('pert', 'i', (1,))
        self.add_dtype('block_type', 'i', (1,))
        self.add_dtype('toShuffle', 'i', (1,))
        super(CursorErrorClamp, self).init()


    def _cycle(self):
        self.task_data['error_clamp'] = self._gen_error_clamp
        self.task_data['pert'] = self._gen_curl
        self.task_data['block_type'] = self._gen_block_type
        self.task_data['toShuffle'] = self._gen_toShuffle
        super(CursorErrorClamp, self)._cycle()

    @staticmethod 
    def center_out_error_clamp_infrequent(ntargets=8, distance=10, n_baseline_blocks=8, n_pert_learning_blocks=8, n_pert_err_clamp_blocks=8, n_washout_blocks=16):
        # need to return: 1) target sequences, 2) baseline or perturbation, 3) error clamp axis

        error_clamp_trials_per_block = 1

        n_meta_blocks = int(ntargets/error_clamp_trials_per_block)
        n_baseline_metablocks = int(n_baseline_blocks/n_meta_blocks)
        n_pert_learning_metablocks = int(n_pert_learning_blocks/n_meta_blocks)
        n_pert_err_clamp_metablocks = int(n_pert_err_clamp_blocks/n_meta_blocks)
        n_washout_metablocks = int(n_washout_blocks/n_meta_blocks)
        

        target_angles = np.arange(-np.pi, np.pi, 2*np.pi/ntargets)
        targs = np.vstack([np.cos(target_angles), np.zeros_like(target_angles), np.sin(target_angles)]).T * distance
        targ_seqs = [dict(targs=np.vstack([np.zeros(3), targ]), error_clamp=False, curl=False, block_type=1) for targ in targs]

        metablock = []
        for k in range(n_meta_blocks):
            metablock.append([x.copy() for x in targ_seqs])

        for k in range(ntargets):
            metablock[k][k]['error_clamp'] = True

        trials = []
        import copy
        from random import shuffle

        for _ in range(n_baseline_metablocks):
            _metablock = copy.deepcopy(metablock)
            for row in _metablock:
                shuffle(row)
            shuffle(_metablock)
            for row in _metablock:
                trials += row
                for tr in row:
                    tr['toShuffle'] = False

        for _ in range(n_pert_learning_metablocks):
            _metablock = copy.deepcopy(metablock)
            for row in _metablock:
                shuffle(row)
                for tr in row:
                    tr['curl'] = True
                    tr['error_clamp'] = False
                    tr['block_type'] = 2
                    tr['toShuffle'] = True


            shuffle(_metablock)
            for row in _metablock:
                trials += row

        for _ in range(n_pert_err_clamp_metablocks):
            _metablock = copy.deepcopy(metablock)
            for row in _metablock:
                shuffle(row)
                for tr in row:
                    tr['curl'] = True
                    tr['block_type'] = 3
                    tr['toShuffle'] = True#previously False HANNAH

            shuffle(_metablock)
            for row in _metablock:
                trials += row

        for _ in range(n_washout_metablocks):
            _metablock = copy.deepcopy(metablock)
            for row in _metablock:
                shuffle(row)
                for tr in row:
                    tr['curl'] = False
                    tr['error_clamp'] = False
                    tr['block_type'] = 4
                    tr['toShuffle'] = False

            shuffle(_metablock)
            for row in _metablock:
                trials += row
       
        return trials

    @staticmethod 
    def center_out_error_clamp_NONE(ntargets=8, distance=10, n_baseline_blocks=48):
        # need to return: 1) target sequences
        #HMS: added 20220411

        #error_clamp_trials_per_block = 0

        n_meta_blocks = 8 #int(ntargets/error_clamp_trials_per_block)
        n_baseline_metablocks = int(n_baseline_blocks/n_meta_blocks)

        target_angles = np.arange(-np.pi, np.pi, 2*np.pi/ntargets)
        targs = np.vstack([np.cos(target_angles), np.zeros_like(target_angles), np.sin(target_angles)]).T * distance
        targ_seqs = [dict(targs=np.vstack([np.zeros(3), targ]), error_clamp=False, curl=False, block_type=1) for targ in targs]

        metablock = []
        for k in range(n_meta_blocks):
            metablock.append([x.copy() for x in targ_seqs])

        trials = []
        import copy
        from random import shuffle

        for _ in range(n_baseline_metablocks):
            _metablock = copy.deepcopy(metablock)
            for row in _metablock:
                shuffle(row)
                for tr in row:
                    tr['curl'] = False
                    tr['error_clamp'] = False
                    tr['block_type'] = 1
                    tr['toShuffle'] = False

            shuffle(_metablock)
            for row in _metablock:
                trials += row
       
        return trials


class BMICursorKinematicCurlErrorClamp(BMICursorKinematicCurlField, CursorErrorClamp):
    exclude_parent_traits = ['plant_type']
    sequence_generators = ['center_out_error_clamp_infrequent']    
    def _parse_next_trial(self):
        super(BMICursorKinematicCurlErrorClamp, self)._parse_next_trial()
        self.decoder.filt.rot_factor = self.rot_factor * int(self._gen_curl)

class BMICursorVisRotErrorClamp(CursorErrorClamp, BMIResetting):
    #fps = 60. # YZ the original fps is set to 60, not sure why here change into 20
    background = (0,0,0,1)
    exclude_parent_traits = ['plant_type', 'timeout_penalty_time', 'marker_num', 'plant_hide_rate', 'plant_visible', 'cursor_radius', 'show_environment', 'rand_start', 'hold_penalty_time']
    sequence_generators = ['center_out_error_clamp_infrequent']    
    rot_angle_deg = traits.Float(-90., desc='scaling factor from speed to rotation angle in degrees')    

    def _parse_next_trial(self):
        super(BMICursorVisRotErrorClamp, self)._parse_next_trial()
        self.decoder.filt.rot_angle_deg = self.rot_angle_deg * int(self._gen_curl)
        self.decoder.filt.filename_KG = self.fn

    def create_assister(self):
        kwargs = dict(decoder_binlen=self.decoder.binlen, target_radius=self.target_radius)
        if hasattr(self, 'assist_speed'):
            kwargs['assist_speed'] = self.assist_speed    
        self.assister = SimpleEndpointAssister(**kwargs)

    def load_decoder(self):
      
        super(BMICursorVisRotErrorClamp, self).load_decoder()
        # Convert the KF to a curl-field generating KF
        dec = self.decoder
        filt = VisRotKalmanFilter(A=dec.filt.A, W=dec.filt.W, C=dec.filt.C, Q=dec.filt.Q, is_stochastic=dec.filt.is_stochastic)
        filt.C_xpose_Q_inv = dec.filt.C_xpose_Q_inv
        filt.C_xpose_Q_inv_C = dec.filt.C_xpose_Q_inv_C
        filt.rot_angle_deg = self.rot_angle_deg
        self.decoder.filt = filt        

class BMICursorShuffleErrorClamp(CursorErrorClamp, BMIResetting):
    '''
        This task is designed to shuffle 50% of the Kalman gains (K) within each row (e.g., x-velocity Kalman gain (row 3) for each unit is randomized).
            This creates a completely random (i.e., shuffled) decoder in the perturbation blocks.
        
        The single instance of shuffling occurs on the first KF update of block 2 (first perturbation block) and is reverted at the start of block 4 (washout block).
        
        Tasked added by HS on 20220218
    '''
    #from riglib.bmi.bmi import Decoder 
    background = (0,0,0,1)
    exclude_parent_traits = ['plant_type', 'timeout_penalty_time', 'marker_num', 'plant_hide_rate', 'plant_visible', 'cursor_radius', 'show_environment', 'rand_start', 'hold_penalty_time']
    sequence_generators = ['center_out_error_clamp_infrequent'] 
    ordered_traits = ['indsToShuffle',  'reward_time_SHUFFLE', 'reward_time', 'timeout_time']  

    reward_time_SHUFFLE = traits.Float(0.5, desc="Length of juice reward AFTER BASELINE") 
    #subject = traits.String('test', desc="Four-character identifier for NHP")
    #trial_run = traits.String('0000', desc="Experiment ID of shuffle task (te_id)")
    indsToShuffle = traits.String('', desc="String of indices to shuffle (tuningCurve_HDF.py)")
  
 
    def _parse_next_trial(self):
        super(BMICursorShuffleErrorClamp, self)._parse_next_trial()
        self.decoder.filt.shuffle_state = int(self._gen_toShuffle)

        # self.decoder.filt.trial_run = self.trial_run 
        # self.decoder.filt.subject = self.subject

        self.decoder.filt.filename_KG = self.fn

        if self.decoder.filt.shuffle_state == 1:
            self.reward_time = self.reward_time_SHUFFLE

        if self.indsToShuffle == '':
            print("NO INDICES TO SHUFFLE!!")
            self.decoder.filt.shuffleInds = [0]
            
        else:
            temp = self.indsToShuffle.split(', ')
            self.decoder.filt.shuffleInds = [int(i) for i in temp]
        

    def create_assister(self):
        kwargs = dict(decoder_binlen=self.decoder.binlen, target_radius=self.target_radius)
        if hasattr(self, 'assist_speed'):
            kwargs['assist_speed'] = self.assist_speed    
        self.assister = SimpleEndpointAssister(**kwargs)

    def load_decoder(self):

        '''The decoder is loaded every 0.1ms (10Hz).'''

        super(BMICursorShuffleErrorClamp, self).load_decoder()
        # Shuffle KF once at start of perturbation block.
        dec = self.decoder 
        filt = ShuffledKalmanFilter(A=dec.filt.A, W=dec.filt.W, C=dec.filt.C, Q=dec.filt.Q, is_stochastic=dec.filt.is_stochastic)
        filt.C_xpose_Q_inv = dec.filt.C_xpose_Q_inv
        filt.C_xpose_Q_inv_C = dec.filt.C_xpose_Q_inv_C
        self.decoder.filt = filt  

class BASELINE_BMICursorShuffle(BMICursorShuffleErrorClamp):

    '''Task for collecting data to compute tuning curves prior to running shuffle task.'''

    exclude_parent_traits = ['indsToShuffle', 'reward_time_SHUFFLE']#plant_type', 'timeout_penalty_time', 'marker_num', 'plant_hide_rate', 'plant_visible', 'cursor_radius', 'show_environment', 'rand_start', 'hold_penalty_time']
    sequence_generators = ['center_out_error_clamp_NONE']#['center_out_error_clamp_infrequent'] #['centerout_2D_discrete']#
    ordered_traits = ['reward_time', 'timeout_time']  

    def _parse_next_trial(self):
        super(BASELINE_BMICursorShuffle, self)._parse_next_trial()        
        self.decoder.filt.filename_KG = self.fn
        self.decoder.filt.shuffle_state = False

            
