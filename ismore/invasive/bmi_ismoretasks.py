from __future__ import division
from collections import OrderedDict

import time
import datetime
import os
import re
import pdb
import pickle
import tables
import math
import traceback
import numpy as np
import pandas as pd
import random
import multiprocessing as mp
import subprocess
from random import shuffle
import random
import copy
# from django.db import models
# from db.tracker import TaskEntry, Task 
from riglib.experiment import traits, Sequence, generate, FSMTable, StateTransitions
from riglib.stereo_opengl.window import WindowDispl2D, FakeWindow
from riglib.stereo_opengl.primitives import Circle, Sector, Line
from riglib.bmi import clda, extractor, train
from riglib.bmi.bmi import Decoder, BMISystem, GaussianStateHMM, BMILoop, GaussianState, MachineOnlyFilter
from riglib.plants import RefTrajectories
from riglib.filter import Filter
from riglib.bmi.goal_calculators import ZeroVelocityGoal_ismore, ZeroVelocityGoal

from ismore import plants, settings, ismore_bmi_lib, brainamp_channel_lists
from ismore.common_state_lists import *
from ismore.invasive.rest_emg_classifier import SVM_rest_EMGClassifier
from ismore.invasive.emg_decoder import LinearEMGDecoder
from ismore.invasive.discrete_movs_emg_classification import SVM_mov_EMGClassifier

from features.bmi_task_features import LinearlyDecreasingAssist, LinearlyDecreasingHalfLife, LinearlyDecreasingXYAssist, LinearlyDecreasingAngAssist
from features.simulation_features import SimTime, SimHDF
from features.generator_features import Autostart

from utils.angle_utils import *
from utils.util_fns import *
from utils.constants import *
from ismore.ismoretasks import IsMoreBase

from utils.ringbuffer import RingBuffer
import pygame
import time
speed_options = ['very-low','low', 'medium','high', 'very-high', 'super-high']

np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)

from ismore.filter import Filter
from scipy.signal import butter,lfilter

class SimClock(object):
    def __init__(self, *args, **kwargs):
        super(SimClock, self).__init__(*args, **kwargs)

    def tick(self, *args, **kwargs):
        pass

class SimClockTick(object):
    '''
    Summary: A simulation pygame.clock to use in simulations that inherit from experiment.Experiment, to overwrite
    the pygame.clock.tick in the ._cycle function ( self.clock.tick(self.fps) )
    '''
    def __init__(self, *args, **kwargs):
        '''
        Summary: Constructor for SimClock
        Input param: *args:
        Input param: **kwargs:
        Output param: 
        '''
        super(SimClockTick, self).__init__(*args, **kwargs)

    def init(self, *args, **kwargs):
        self.clock = SimClock()
        super(SimClockTick, self).init(*args, **kwargs)

class IsmoreSimTime(SimTime):
    @property
    def update_rate(self):
        return 1/20.

#######################################################################
COLORS = {
    'black': (0, 0, 0, 1),
    'red':   (1, 0, 0, 1),
    'green': (0, 1, 0, 1),
    'blue':  (0, 0, 1, 1),
    'white': (1, 1, 1, 1),
}

plant_type_options = ['ArmAssist', 'ReHand', 'IsMore', '']
clda_update_methods = ['RML', 'Smoothbatch', 'Baseline']
languages_list = ['english', 'deutsch', 'castellano', 'euskara']
clda_intention_est_methods = ['OFC', 'OFC_w_rest']
clda_adapting_states_opts = ['ArmAssist', 'ReHand', 'IsMore']
clda_adapt_mFR_stats = ['Yes', 'No']
no_yes = ['No', 'Yes']

blocking_options = ['No', 'ArmAssist', 'ArmAssist_and_Pron','ArmAssist_and_Thumb','ReHand', 'Fingers_Only', 'Fingers_ArmAssist', 
'Index', 'Prono_only', 'Psi_and_Fingers', 'Thumb', 'ArmAssist_and_Pron_and_Thumb', 'Pron_and_Thumb', 
'ArmAssist_and_Fing3', 'ArmAssist_and_Fing3_Thumb', 'CLDA_blk', 'Psi_and_Prono', 'DoFs_non_involved_block', 'DoFs_non_involved_block_and_thumb', 'ReHand_and_Psi', 'All_but_3fingers', 'All_but_thumb', 'All_but_index']

CLDA_blk = {}
CLDA_blk['red to green'] = [3, 4, 5, 6]
CLDA_blk['green'] = [2, 3, 4, 5, 6]
CLDA_blk['blue'] = [2, 3, 4, 5, 6]
CLDA_blk['red'] = [2, 3, 4, 5, 6]
CLDA_blk['blue to red'] = [3, 4, 5, 6]
CLDA_blk['up'] = [0, 1, 2, 3, 4, 5]
CLDA_blk['down'] = [0, 1, 2, 3, 4, 5]
CLDA_blk['point'] = [0, 1, 2, 6]
CLDA_blk['grasp'] = [0, 1, 2, 6]
CLDA_blk['red_up'] = [2, 3, 4, 5]
CLDA_blk['red_down'] = [2, 3, 4, 5]
CLDA_blk['green_point'] = [2, 6]
CLDA_blk['blue_grasp'] = [2, 6]
CLDA_blk['blue_point'] = [2, 6]
CLDA_blk['red_point_down'] = [2]
CLDA_blk['blue_grasp_up'] = [2]
CLDA_blk['green_grasp_down'] = [2]
CLDA_blk['red_grasp_up'] = [2]
CLDA_blk['red_grasp_down'] = [2]
CLDA_blk['blue_point_down'] = [2]
CLDA_blk['red_grasp'] = [2, 6]
CLDA_blk['green_grasp'] = [2, 6]


DoFs_non_involved_block = {}
DoFs_non_involved_block['green'] = [3, 4, 5, 6]
DoFs_non_involved_block['blue'] = [3, 4, 5, 6]
DoFs_non_involved_block['red'] = [3, 4, 5, 6]
DoFs_non_involved_block['red to green'] = [3, 4, 5, 6]
DoFs_non_involved_block['blue to red'] = [3, 4, 5, 6]
DoFs_non_involved_block['up'] = [0, 1, 2, 3, 4, 5]
DoFs_non_involved_block['down'] = [0, 1, 2, 3, 4, 5]
DoFs_non_involved_block['point'] = [0, 1, 2, 6]
DoFs_non_involved_block['grasp'] = [0, 1, 2, 6]
DoFs_non_involved_block['red_grasp'] = [6]
DoFs_non_involved_block['green_grasp'] = [6]
DoFs_non_involved_block['blue_grasp'] = [6]
DoFs_non_involved_block['red_up'] = [2, 3, 4, 5]
DoFs_non_involved_block['green_up'] = [2, 3, 4, 5]
DoFs_non_involved_block['blue_up'] = [2, 3, 4, 5]
DoFs_non_involved_block['red_down'] = [2, 3, 4, 5]
DoFs_non_involved_block['green_down'] = [2, 3, 4, 5]
DoFs_non_involved_block['blue_down'] = [2, 3, 4, 5]
DoFs_non_involved_block['red_point'] = [6]
DoFs_non_involved_block['green_point'] = [6]
DoFs_non_involved_block['blue_point'] = [6]
DoFs_non_involved_block['red_point_down'] = []
DoFs_non_involved_block['blue_grasp_up'] = []
DoFs_non_involved_block['green_grasp_down'] = []
DoFs_non_involved_block['red_grasp_up'] = []
DoFs_non_involved_block['red_grasp_down'] = []
DoFs_non_involved_block['blue_point_down'] = []
DoFs_non_involved_block['grasp_up'] = [0, 1, 2]
DoFs_non_involved_block['grasp_down'] = [0, 1, 2]
DoFs_non_involved_block['wrist_ext'] = [0, 1, 2, 3, 4, 6] #all but 3-fingers

thumb_idx = 3
DoFs_non_involved_block_and_thumb = {}
for trial_type in DoFs_non_involved_block.keys():
    if thumb_idx not in DoFs_non_involved_block[trial_type]:
        DoFs_non_involved_block_and_thumb[trial_type] = DoFs_non_involved_block[trial_type] + [thumb_idx]
        DoFs_non_involved_block_and_thumb[trial_type].sort()

    else:
        DoFs_non_involved_block_and_thumb[trial_type] = DoFs_non_involved_block[trial_type]
# for tt in CLDA_blk.keys():
#     DoFs_non_involved_block[tt] = CLDA_blk[tt]
#     if 2 in DoFs_non_involved_block[tt]:
#         DoFs_non_involved_block[tt].remove(2)

blocking_dict = dict(No=[], ArmAssist=[0, 1, 2], ArmAssist_and_Pron=[0, 1, 2, 6], ArmAssist_and_Thumb=[0, 1, 2, 3], ReHand=[3, 4, 5, 6],
    Fingers_Only=[3, 4, 5], Fingers_ArmAssist=[0, 1, 2, 3, 4, 5], Prono_only=[6], Psi_and_Fingers=[2, 3, 4, 5], 
    Index = [4], Thumb = [3], ArmAssist_and_Pron_and_Thumb=[0, 1, 2, 3, 6], Pron_and_Thumb=[3, 6], CLDA_blk=CLDA_blk,
    ArmAssist_and_Fing3=[0, 1, 2, 5], ArmAssist_and_Fing3_Thumb = [0, 1, 2, 3, 5], Psi_and_Prono = [2, 6], DoFs_non_involved_block = DoFs_non_involved_block, DoFs_non_involved_block_and_thumb = DoFs_non_involved_block_and_thumb, ReHand_and_Psi = [2, 3, 4, 5, 6], All_but_3fingers = [0, 1, 2, 3, 4, 6], All_but_thumb = [0, 1, 2, 4, 5, 6], All_but_index = [0, 1, 2, 3, 5, 6])

#######################################################################
class PlantControlBase(FakeWindow, Sequence, IsMoreBase):
    '''Abstract base class for controlling plants through a sequence of targets.'''
    exclude_parent_traits = ['reward_time', 'rand_start', 'simulate', 'show_environment', 'arm_side']
    arm_side = 'right'
    fps = 20 
    cnt = 0
    status = {
        'wait': {
            'start_trial':         'rest',
            'stop':                None},
        
        'rest': {
            'rest_complete':    'instruct_trial_type',
            'stop':             None},

        'instruct_trial_type': {
            'end_instruct':        'target' },
        
        'target': {
            'enter_target':        'hold',
            'timeout':             'timeout_penalty',
            'stop':                None},
        
        'hold': {
            'leave_early':         'hold_penalty',
            'hold_complete':       'targ_transition'},
        
        'targ_transition': {
            'trial_complete':      'reward',
            'trial_incomplete':    'instruct_trial_type'},
        
        'timeout_penalty': {
            'timeout_try_again': 'instruct_trial_type',
            'timeout_skip_target': 'instruct_trial_type',
            'timeout_end_of_trial': 'wait'},
        
        'hold_penalty': {
            'hold_penalty_try_again':    'instruct_trial_type',
            'hold_penalty_skip_target':  'instruct_trial_type',
            'hold_penalty_end_of_trial':  'wait'},
        
        'reward': {
            'reward_end': 'wait'}
    }

    state = "wait"  # initial state
    trial_end_states = ['timeout_penalty', 'hold_penalty', 'reward']
    sequence_generators = ['B1_targets', 'B2_targets', 'B3_targets','F1_targets',
        'B1_B2_targets','F1_B2_targets', 'targets_circular', 'targets_linear', 
        'compliant_movements_obj', 'compliant_movements_non_obj', 'compliant_move_blk1', 
        'compliant_move_blk2', 'compliant_move_blk3', 'compliant_move_blk4', 'clda_blk',
        'compliant_move_blk3_4_combo', 'active_movements_blk', 'blk_B1_grasp', 'blk_grasp_combo']

    channel_list_options = brainamp_channel_lists.channel_list_options
    reset = False

    replace_ya_w_pausa = traits.OptionsList(*no_yes, bmi3d_input_options=no_yes)

    # settable parameters on web interface
    wait_time            = .1 #traits.Float(2,  desc='Time to remain in the wait state.')
    rest_time            = traits.Float(2., desc='Time to remain in rest state')
    reward_time          = .1 #traits.Float(.5, desc='Time in reward state.')
    hold_time            = traits.Float(.2, desc='Hold time required at targets.')
    hold_penalty_time    = .1 #traits.Float(1,  desc='Penalty time for target hold error.')
    timeout_time         = traits.Float(15, desc='Time allowed to go between targets.')
    timeout_penalty_time = .1 #traits.Float(1,  desc='Penalty time for timeout error.')
    max_attempt_per_targ = traits.Int(2, desc='The number of attempts at a target before skipping to the next one.')
    target_radius_x      = traits.Float(2,  desc='Radius of targets.')
    target_radius_y      = traits.Float(2,  desc='Radius of targets.')
    tol_deg_psi          = traits.Float(5, desc='Angular orientation must be within +/- this amount of target angle for success.')
    tol_deg_pron         = traits.Float(5, desc='Angular orientation must be within +/- this amount of target angle for success.')
    tol_deg_fing         = traits.Float(5, desc='Angular orientation must be within +/- this amount of target angle for success.')
    tol_deg_thumb         = traits.Float(5, desc='Angular orientation must be within +/- this amount of target angle for success.')
    
    safety_grid_file     = traits.DataFile(object, bmi3d_query_kwargs=dict(system__name='safety'))
    channel_list_name    = traits.OptionsList(*channel_list_options, bmi3d_input_options=channel_list_options) 

    blocking_opts = traits.OptionsList(*blocking_options, bmi3d_input_options=blocking_options)
    ignore_correctness_jts = traits.OptionsList(*blocking_options, bmi3d_input_options=blocking_options)

    target_index = -1  # helper variable to keep track of which target to display within a trial
    n_attempts = 0     # helper variable to keep track of the number of failed attempts at a given trial
    rehand_angular_noise_tol = np.deg2rad(5.)

    # Sounds directory 
    language = traits.OptionsList(*languages_list, bmi3d_input_options=languages_list)

    # Target matrix, trained: 
    targets_matrix = traits.DataFile(object,desc='test', bmi3d_query_kwargs=dict(system__name='misc'))
    
    keep_attempts = False

    def __init__(self, *args, **kwargs):
        super(PlantControlBase, self).__init__(*args, **kwargs)

        self.command_vel = pd.Series(0.0, self.vel_states)
        self.target_pos  = pd.Series(0.0, self.pos_states+self.vel_states)


        print ' ALERT: IF youre using JUST the ReHand, be aware that youre initial target \
        position will be very wrong!' 

        self.target_pos[0] = 33.
        self.target_pos[1] = 28.
        self.target_pos[2] = 0.9

        self.add_dtype('target_pos',  'f8', (len(self.target_pos),))
        self.add_dtype('target_index', 'i', (1,))
        self.add_dtype('command_vel_raw', 'f8', (len(self.command_vel),))
        self.add_dtype('command_vel_sent', 'f8', (len(self.command_vel),))
        self.add_dtype('command_vel_sent_pre_safety', 'f8', (len(self.command_vel),))
        self.add_dtype('pre_drive_state', 'f8', (len(self.command_vel),))

        self.add_dtype('trial_type', np.str_, 40)
        self.add_dtype('iter_time', 'f8', (1, ))
        self.add_dtype('plant_pos_raw', 'f8', (len(self.plant_vel),))
        self.add_dtype('plant_vel_raw', 'f8', (len(self.plant_vel),))

        self.last_time2 = time.time()
        self.init_target_display()

        self.blocking_joints = blocking_dict[self.blocking_opts]
        if type(self.blocking_joints) is dict:
            self.blocking_joints = None

        self.plant.blocking_joints = self.blocking_joints

        self.ignore_correctness = blocking_dict[self.ignore_correctness_jts]
        
        if type(self.ignore_correctness) is dict:
            self.ignore_correctness_dict = self.ignore_correctness
        else:
            self.ignore_correctness_dict = None

 

        self.plant.enable()
        self.brainamp_channels = getattr(brainamp_channel_lists, self.channel_list_name)  

        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)
        pygame.mixer.init()

        # Filter for smoothing command_velocities
        # 4th order butterworth filter for command_vel
        fs_synch = self.fps #Frequency at which the task is running
        nyq   = 0.5 * fs_synch
        cuttoff_freq  = 5 / nyq
        bpf_kin_coeffs = butter(4, cuttoff_freq, btype='low')

        self.command_lpfs = dict()
        for state in self.vel_states:
            self.command_lpfs[state] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1]) # low-pass filter to smooth out command velocities
        self.plant.command_lpfs = self.command_lpfs
        self.plant.task_state = 'wait'

    def move_plant(self):
        raise NotImplementedError  # implement in subclasses

    def init_target_display(self):
        if self.plant_type in ['ArmAssist', 'IsMore']:
            # use circle & sector to represent the ArmAssist part of a target (xy position + orientation)
            self.target1        = Circle(np.array([0, 0]), self.target_radius_x, COLORS['green'], False)
            self.target2        = Circle(np.array([0, 0]), self.target_radius_x, COLORS['green'], False)
            self.target1_sector = Sector(np.array([0, 0]), 2*self.target_radius_x, [0, 0], COLORS['white'], False)
            self.target2_sector = Sector(np.array([0, 0]), 2*self.target_radius_x, [0, 0], COLORS['white'], False)

            self.add_model(self.target1)
            self.add_model(self.target2)
            self.add_model(self.target1_sector)
            self.add_model(self.target2_sector)

        if self.plant_type in ['ReHand', 'IsMore']:
            # use sectors to represent the ReHand part of a target (4 angles)
            self.rh_sectors = {}
            for state in rh_pos_states:
                s = Sector(self.rh_angle_line_positions[state], 5, [0, 0], COLORS['white'])
                self.rh_sectors[state] = s
                self.add_model(s)

    def _parse_next_trial(self):
        self.trial_type = self.next_trial
        self.targets = self.targets_matrix[self.trial_type]
        self.chain_length = len(self.targets.keys())

        if self.ignore_correctness_dict is not None:
            self.ignore_correctness = self.ignore_correctness_dict[self.trial_type]
        
        if self.blocking_joints is None: # in ['CLDA_blk', 'DoFs_non_involved_block', 'DoFs_non_involved_block_and_thumb']: #dictionary with specific joints based of trial_type
            self.blocking_joints = blocking_dict[self.blocking_opts][self.trial_type]
 

    def load_safety(self):
        if self.safety_grid_file is not None:
            try:
                self.plant.safety_grid = pickle.load(open(self.safety_grid_file))
            except:
                self.plant.safety_grid = self.safety_grid_file

    def hide_targets(self):
        if self.plant_type in ['ArmAssist', 'IsMore']:
            self.target1.visible        = False
            self.target2.visible        = False
            self.target1_sector.visible = False
            self.target2_sector.visible = False
        
        if self.plant_type in ['ReHand', 'IsMore']:
            for state in rh_pos_states:
                self.rh_sectors[state].visible = False

    def _cycle(self):
        '''Runs self.fps times per second (see Experiment class).'''
        # get latest position/velocity information before calling move_plant()
        super(PlantControlBase, self)._cycle()
        self.plant.task_state = self.state
        self.plant_pos[:] = self.plant.get_pos()
        self.plant_vel[:] = self.plant.get_vel()
        #self.move_plant()

        self.update_plant_display()

        self.task_data['plant_pos']    = self.plant_pos.values
        self.task_data['plant_vel']    = self.plant_vel.values
        
        self.task_data['plant_pos_raw']    = self.plant.get_pos_raw()
        self.task_data['plant_vel_raw']    = self.plant.get_vel_raw()

        self.task_data['target_pos']   = self.target_pos.values
        self.task_data['target_index'] = self.target_index

        self.task_data['command_vel_raw'] = self.plant.drive_velocity_raw
        self.task_data['command_vel_sent_pre_safety'] = self.plant.drive_velocity_sent_pre_safety
        
        self.task_data['command_vel_sent'] = self.plant.drive_velocity_sent
        self.task_data['pre_drive_state'] = self.plant.pre_drive_state

        self.task_data['trial_type'] = self.trial_type
        self.task_data['iter_time'] = time.time() - self.last_time2
        self.last_time2 = time.time()
        self.cnt += 1

    #### TEST FUNCTIONS ####

    # helper function
    def armassist_inside_target(self, target=None):

        if target is None:
            target = self.target_pos
        # assume rest target is last target: 
        
        if self.plant.safety_grid is not None:
            rest = self.plant.safety_grid.interior_pos # targets_matrix[self.trial_type][L]
            angle_between = np.arctan2(target['aa_py'] - rest[1], target['aa_px'] - rest[0])
            dmn_tg = target[['aa_px', 'aa_py']] - rest
            dmn_pt = self.plant_pos[['aa_px', 'aa_py']] - rest

            # # make a rotation matrix:
            R = np.mat([[np.cos(-1*angle_between), -1*np.sin(-1*angle_between)],
                        [np.sin(-1*angle_between), np.cos(-1*angle_between)]])

            rotated_pt = R*np.mat(dmn_pt).T
            rotated_tg = R*np.mat(dmn_tg).T

            pt = rotated_pt - rotated_tg

            dx = np.abs(pt[0]) # <= self.target_radius_x)
            dy = np.abs(pt[1]) #<= self.target_radius_y)

        else:
            dx = np.abs(self.plant_pos['aa_px'] - target['aa_px'])
            dy = np.abs(self.plant_pos['aa_py'] - target['aa_py'])

        if 0 in self.ignore_correctness and 1 not in self.ignore_correctness:
            var = ['aa_py']
            inside_target = dy <= self.target_radius_y

        elif 1 in self.ignore_correctness and 0 not in self.ignore_correctness:
            var = ['aa_px']
            inside_target = dx <= self.target_radius_x

        elif 0 in self.ignore_correctness and 1 in self.ignore_correctness:
            var = []
            inside_target = True

        elif 0 not in self.ignore_correctness and 1 not in self.ignore_correctness:
            var = ['aa_px', 'aa_py']
            inside_target = np.logical_and(dx <= self.target_radius_x, dy <= self.target_radius_y)

        if 2 in self.ignore_correctness:
            inside_angular_target = True
        else:
            target_psi = target['aa_ppsi']
            inside_angular_target = angle_inside_range(self.plant_pos['aa_ppsi'],
                                                   target_psi - np.deg2rad(self.tol_deg_psi),
                                                       target_psi + np.deg2rad(self.tol_deg_psi))
        return inside_target and inside_angular_target

    # helper function
    def rehand_inside_target(self, target=None):
        if target is None:
            target = self.target_pos

        for i, state in enumerate(rh_pos_states):
            if i + 3 not in self.ignore_correctness:
                angle = self.plant_pos[state]
                target_angle = target[state]

                if state == 'rh_pprono':
                    max_angle = target_angle + np.deg2rad(self.tol_deg_pron)
                    min_angle = target_angle - np.deg2rad(self.tol_deg_pron)
                elif state == 'rh_pthumb':
                    max_angle = target_angle + np.deg2rad(self.tol_deg_thumb)
                    min_angle = target_angle - np.deg2rad(self.tol_deg_thumb)                    
                else:
                    max_angle = target_angle + np.deg2rad(self.tol_deg_fing)
                    min_angle = target_angle - np.deg2rad(self.tol_deg_fing)
                
                if self.state == 'hold':
                    max_angle += self.rehand_angular_noise_tol
                    min_angle -= self.rehand_angular_noise_tol
                
                if not angle_inside_range(angle, min_angle, max_angle):
                    return False
        return True  # must be True if we reached here

    def _test_end_instruct(self, *args, **kwargs):
        return self.sound_counter >= self.sound_chain
        #return not pygame.mixer.music.get_busy()

    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause

    def _test_rest_complete(self, ts):
        return ts > self.rest_time and not self.pause and not pygame.mixer.music.get_busy()

    def _test_enter_target(self, ts):
        if self.plant_type == 'ArmAssist':
            return self.armassist_inside_target()
        elif self.plant_type == 'ReHand':
            return self.rehand_inside_target()
        elif 'IsMore' in self.plant_type:
            return self.armassist_inside_target() and self.rehand_inside_target()
        
    def _test_leave_early(self, ts):
        return not self._test_enter_target(ts)

    def _test_hold_complete(self, ts):
        print 'test hold: ', self.hold_sound_counter, self.hold_sound_chain, self.soundlist_hold
        return ts >= self.hold_time and self.hold_sound_counter >= self.hold_sound_chain

    def _test_timeout(self, ts):
        return ts > self.timeout_time

    def _test_timeout_penalty_end(self, ts):
        return ts > self.timeout_penalty_time

    def _test_hold_penalty_end(self, ts):
        return ts > self.hold_penalty_time

    def _test_trial_complete(self, ts):
        if self.target_index == self.chain_length - 1 :
            self.n_attempts = 0
            return True
        else:
            return False

    def _test_trial_incomplete(self, ts):
        if (not self._test_trial_complete(ts)) and (self.n_attempts < self.max_attempt_per_targ) :
            self.n_attempts = 0
            return True
        else:
            return False

    def _test_timeout_end_of_trial(self, ts):
        return (ts > self.timeout_penalty_time) and (self.n_attempts == self.max_attempt_per_targ) and (self.target_index == self.chain_length-1)

    def _test_timeout_skip_target(self, ts):
        if ts > self.timeout_penalty_time:
            if self.n_attempts == self.max_attempt_per_targ and (self.target_index < self.chain_length - 1):
                self.n_attempts = 0
                return True
            else:
                return False
        else:
            return False

    def _test_timeout_try_again(self, ts):
        if ts > self.timeout_penalty_time:
            if self.n_attempts < self.max_attempt_per_targ:
                self.target_index -= 1
                return True
            else:
                return False
        else:
            return False

    def _test_hold_penalty_try_again(self, ts):
        if ts > self.hold_penalty_time:
            if self.n_attempts < self.max_attempt_per_targ:
                self.target_index -= 1
                return True
            else:
                return False
        else:
            return False  

    def _test_hold_penalty_skip_target(self, ts):
        if ts > self.hold_penalty_time:
            if self.n_attempts == self.max_attempt_per_targ:
                self.n_attempts = 0
                return True
            else:
                return False
        else:
            return False      

    def _test_reward_end(self, ts):
        return ts > self.reward_time

    #### STATE FUNCTIONS ####

    def _start_wait(self):
        super(PlantControlBase, self)._start_wait()
        self._helper_start_wait()

    def _start_rest(self):
        self._play_sound(self.sounds_dir, ['rest'])

    def _start_instruct_trial_type(self):
        self.target_index += 1

        try:
            self.target_pos[:] = self.targets[self.target_index]
        except:
            self.target_pos[:] = pd.Series(np.hstack(( self.targets[self.target_index][self.pos_states], np.zeros((len(self.targets[self.target_index][self.pos_states]))) )), self.pos_states+self.vel_states)
        
        self.soundlist = self.get_sounds(self.sounds_dir, self.targets_matrix['subgoal_names'][self.trial_type][self.target_index])
        self.sound_counter = 0 # Increments after playing sound
        self.sound_chain = len(self.soundlist)

    def _while_instruct_trial_type(self):
        if pygame.mixer.music.get_busy():
            pass
        else:
            sound = self.soundlist[self.sound_counter]
            pygame.mixer.music.load(sound)
            pygame.mixer.music.play()
            self.sound_counter += 1

    def _helper_start_wait(self):
        self.n_attempts = 0
        self.target_index = -1
        self.hide_targets()

    def _start_target(self):

        if self.plant_type in ['ArmAssist', 'IsMore']:
            self.target1.color = COLORS['red']
            self.target2.color = COLORS['red']

            x   = self.target_pos['aa_px']
            y   = self.target_pos['aa_py']
            psi = self.target_pos['aa_ppsi']

            # move a target to current location (target1 and target2 alternate moving)
            if self.target_index % 2 == 0:
                target = self.target1
                target_sector = self.target1_sector
            else:
                target = self.target2
                target_sector = self.target2_sector
            
            d2r = deg_to_rad

            target.center_pos        = np.array([x, y])
            target.visible           = True
            target_sector.center_pos = np.array([x, y])
            target_sector.ang_range  = [psi-(self.tol_deg_psi*d2r), psi+(self.tol_deg_psi*d2r)]
            target_sector.visible    = True

        if self.plant_type in ['ReHand', 'IsMore']:
            for state in rh_pos_states:
                sector = self.rh_sectors[state]
                target_angle = self.target_pos[state]
                if state == 'rh_pprono':
                    sector.ang_range = [target_angle-(self.tol_deg_pron*d2r), target_angle+(self.tol_deg_pron*d2r)]
                else:
                    sector.ang_range = [target_angle-(self.tol_deg_fing*d2r), target_angle+(self.tol_deg_fing*d2r)]
                sector.visible = True
        
    def _start_hold(self):
        if self.plant_type in ['ArmAssist', 'IsMore']:
            # make next target visible unless this is the final target in the trial
            if (self.target_index + 1) < self.chain_length:
                next_target_pos = pd.Series(self.targets[self.target_index+1], self.pos_states)
                x   = next_target_pos['aa_px']
                y   = next_target_pos['aa_py']
                psi = next_target_pos['aa_ppsi']

                if self.target_index % 2 == 0:
                    target = self.target2
                    target_sector = self.target2_sector
                else:
                    target = self.target1
                    target_sector = self.target1_sector
                
                target.center_pos        = np.array([x, y])
                target.visible           = True
                target_sector.center_pos = np.array([x, y])
                target_sector.ang_range  = [psi-np.deg2rad(self.tol_deg_psi), psi+np.deg2rad(self.tol_deg_psi)]
                target_sector.visible    = True

        # unlike ArmAssist target circles, we only have one set of ReHand 
        # target sectors objects, so we can't display next ReHand targets
        self.soundlist_hold = self.get_sounds(self.sounds_dir, ['beep1'])
        self.hold_sound_counter = 0 # Increments after playing sound
        self.hold_sound_chain = len(self.soundlist_hold)

    def _while_hold(self):
        if pygame.mixer.music.get_busy():
            pass
        else:
            sound = self.soundlist_hold[self.hold_sound_counter]
            pygame.mixer.music.load(sound)
            pygame.mixer.music.play()
            self.hold_sound_counter += 1

    def _end_hold(self):
        if self.plant_type in ['ArmAssist', 'IsMore']:
            # change current target color to green
            if self.target_index % 2 == 0:            
                self.target1.color = COLORS['green']
            else:            
                self.target2.color = COLORS['green']

    # helper function
    def start_penalty(self):
        self.hide_targets()
        self.n_attempts += 1

    def _start_hold_penalty(self):
        self.start_penalty()

    def _start_timeout_penalty(self):
        self.start_penalty()

    def _start_targ_transition(self):
        self.hide_targets()
    
    def _start_reward(self):
        #super(PlantControlBase, self)._start_reward()
        #self._play_sound(self.sounds_dir, ['beep1'])

        if self.plant_type in ['ArmAssist', 'IsMore']:    
            if self.target_index % 2 == 0:
                self.target1.visible        = True
                self.target1_sector.visible = True
            else:
                self.target2.visible        = True
                self.target2_sector.visible = True
    
    def get_sounds(self, fpath, fname):
        soundfiles = []
        if hasattr(self, 'replace_ya_w_pausa'):
            if self.replace_ya_w_pausa == 'Yes':
                if fname[0] == 'go':
                    fname = ['rest']

        for filename in fname:
            # print 'filename ', filename
            if filename == 'circular':
                filename = 'circular_big'
                sound_fname = os.path.join(fpath, filename + '.wav')
                soundfiles.append(sound_fname)                
            
            elif '_' in filename or ' ' in filename:
                # First see if there's a file with exact name: 
                if os.path.isfile(os.path.join(fpath, filename + '.wav')):
                    soundfiles.append(os.path.join(fpath, filename + '.wav'))  
                else:
                    # try:
                    # Next try replacing with spaces: 
                    # Red to green
                    if '_' in filename:
                        filename = filename.replace('_', ' ')
                        key = ' '
                    elif ' ' in filename:
                        filename = filename.replace(' ', '_')
                        key = '_'
                    
                    if os.path.isfile(os.path.join(fpath, filename + '.wav')):
                        soundfiles.append(os.path.join(fpath, filename + '.wav'))
                    else: 
                        #try:
                        # Next try splitting up the names:
                        fi1 = filename.find(key)  
                        filename1 = filename[:fi1]

                        if os.path.isfile(os.path.join(fpath, filename1 + '.wav')):
                            #sound_fname = os.path.join(fpath, filename1 + '.wav')
                            soundfiles.append(os.path.join(fpath, filename1 + '.wav'))
                            
                            filename2 = filename[filename.find(key)+1:]
                            if os.path.isfile(os.path.join(fpath, filename2 + '.wav')):
                                soundfiles.append(os.path.join(fpath, filename2 + '.wav'))
                            else:
                                # 3 legged: 
                                fi2 = filename.find(key, fi1+1)
                                filename2 = filename[fi1+1:fi2]
                                filename3 = filename[fi2+1:]

                                sound_fname = os.path.join(fpath, filename2 + '.wav')
                                soundfiles.append(sound_fname)
                                
                                sound_fname2 = os.path.join(fpath, filename3 + '.wav')
                                soundfiles.append(sound_fname2)

                        else:
                            print 'cant play: ', filename
            elif filename == 'beep':
                soundfiles.append(os.path.join(fpath, 'beep1' + '.wav'))
            else:
                sound_fname = os.path.join(fpath, filename + '.wav')
                soundfiles.append(sound_fname)
            return soundfiles

    def cleanup_hdf(self):
        self.plant.disable()
        super(PlantControlBase, self).cleanup_hdf()
         
class ManualControl(PlantControlBase):
    '''Allow the subject to manually move the plant to targets with disabled plant'''
        
    is_bmi_seed = True
    def __init__(self, *args, **kwargs):
        super(ManualControl, self).__init__(*args, **kwargs)
        self.plant.disable()
        print 'Motors disabled'

    def move_plant(self):
        '''Do nothing here -- plant is moved manually.'''
        pass

class CompliantMovements(PlantControlBase):
    '''Moves the plant automatically to targets using an assister.'''
    is_bmi_seed = True
    speed = traits.OptionsList(*speed_options, bmi3d_input_options= speed_options)

    def __init__(self, *args, **kwargs):
        super(CompliantMovements, self).__init__(*args, **kwargs)
        assister_kwargs = {
            'call_rate': self.fps,
            'xy_cutoff': np.min([self.target_radius_x, self.target_radius_y]),
            'speed':    self.speed,
        }

        ### PREEYA TESTING 3-14-18
        self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT_OG[self.plant_type](**assister_kwargs)
        self.goal_calculator = ZeroVelocityGoal_ismore(ismore_bmi_lib.SSM_CLS_DICT[self.plant_type], 
            pause_states = ['rest', 'wait', 'instruct_rest', 'instruct_trial_type'])
        

        # Filter for smoothing command_velocities
        # 4th order butterworth filter for command_vel
        fs_synch = self.fps #Frequency at which the task is running
        nyq   = 0.5 * fs_synch
        cuttoff_freq  = 1.5 / nyq
        bpf_kin_coeffs = butter(4, cuttoff_freq, btype='low')

        self.command_lpfs = dict()
        for state in self.vel_states:
            self.command_lpfs[state] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1]) # low-pass filter to smooth out command velocities


    def move_plant(self):
        current_state = np.hstack([self.plant_pos.values, self.plant_vel.values, 1])[:, None]

        data, solution_updated = self.goal_calculator(self.target_pos.values, self.state, 
            **dict(current_state=current_state))
        
        target_state = data[0].reshape(-1, 1)

        # use assister:     
        assist_output = self.assister(current_state, target_state, 1.)
        
        try:
            # For LQR based assisters
            Bu = pd.Series(np.array(assist_output["x_assist"]).ravel(), self.ssm_states)
        except:
            # For ismore_bmi_lib.IsMoreAssister (old)
            Bu = pd.Series(np.squeeze(np.array(assist_output['Bu'])), self.ssm_states)

        command_vel_raw = Bu[self.vel_states]
        self.plant.drive_velocity_raw = command_vel_raw

        if np.any(np.isnan(command_vel_raw)):
            print 'setting command_vel_raw nans equal to zero'
            command_vel_raw[np.isnan(command_vel_raw)] = 0

        #filter command_vel
        command_vel  = pd.Series(0.0, self.vel_states)
        
        if self.state in ['wait','rest', 'instruct_rest', 'instruct_trial_type']: 
            command_vel_raw[:] = 0

        for state in self.vel_states:
            # command_vel[state] = self.command_lpfs[state](command_vel[state])
            command_vel[state] = self.command_lpfs[state](command_vel_raw[state])
            if np.isnan(command_vel[state]):
                command_vel[state] = 0
        
        self.plant.drive_velocity_sent = command_vel.values
        self.plant.send_vel(command_vel.values) 

    def _cycle(self):
        super(CompliantMovements, self)._cycle()
        self.move_plant()
        
        #self.task_data['trial_accept_reject'] = self.experimenter_acceptance_of_trial

class CompliantMovements_w_prep(CompliantMovements):

    status = {
        'wait': {
            'start_trial':         'rest',
            'stop':                None},
        
        'rest': {
            'rest_complete':    'instruct_trial_type',
            'stop':             None},

        'instruct_trial_type': {
            'end_instruct':        'prep' },
        
        'prep': {
            'end_prep':             'target'},
            
        'target': {
            'enter_target':        'hold',
            'timeout':             'timeout_penalty',
            'stop':                None},
        
        'hold': {
            'leave_early':         'hold_penalty',
            'hold_complete':       'targ_transition'},
        
        'targ_transition': {
            'trial_complete':      'reward',
            'trial_incomplete':    'instruct_trial_type'},
        
        'timeout_penalty': {
            'timeout_try_again': 'instruct_trial_type',
            'timeout_skip_target': 'instruct_trial_type',
            'timeout_end_of_trial': 'wait'},
        
        'hold_penalty': {
            'hold_penalty_try_again':    'instruct_trial_type',
            'hold_penalty_skip_target':  'instruct_trial_type',
            'hold_penalty_end_of_trial':  'wait'},
        
        'reward': {
            'reward_end': 'wait'}
    }
    prep_min = traits.Float(1, desc='minimum of prep phase')
    prep_max = traits.Float(1, desc='maximum of prep phase')
   # sequence_generators = ['blk_B1_grasp']
    vel_scale_aa = traits.Float(1., desc='none')
    vel_scale_rh = traits.Float(1., desc='none')

    def __init__(self, *args, **kwargs):
        super(CompliantMovements_w_prep, self).__init__(*args, **kwargs)
        self.add_dtype('command_vel_non_scaled', 'f8', (len(self.command_vel),))

    def move_plant(self):
        current_state = np.hstack([self.plant_pos.values, self.plant_vel.values, 1])[:, None]

        data, solution_updated = self.goal_calculator(self.target_pos.values, self.state, 
            **dict(current_state=current_state))
        
        target_state = data[0].reshape(-1, 1)

        # use assister:     
        assist_output = self.assister(current_state, target_state, 1.)
        
        try:
            # For LQR based assisters
            Bu = pd.Series(np.array(assist_output["x_assist"]).ravel(), self.ssm_states)
        except:
            # For ismore_bmi_lib.IsMoreAssister (old)
            Bu = pd.Series(np.squeeze(np.array(assist_output['Bu'])), self.ssm_states)

        command_vel_raw = Bu[self.vel_states]
        self.plant.drive_velocity_raw = command_vel_raw

        if np.any(np.isnan(command_vel_raw)):
            print 'setting command_vel_raw nans equal to zero'
            command_vel_raw[np.isnan(command_vel_raw)] = 0

        #filter command_vel
        command_vel  = pd.Series(0.0, self.vel_states)
        command_vel_non_scaled  = pd.Series(0.0, self.vel_states)
        if self.state in ['wait','rest', 'instruct_rest', 'instruct_trial_type', 'prep']: 
            command_vel_raw[:] = 0

        # Make sure blocked joints don't move! 
        if self.blocking_opts in ['CLDA_blk', 'DoFs_non_involved_block', 'DoFs_non_involved_block_and_thumb']: #dictionary with specific joints based of trial_type
            self.blocking_joints = blocking_dict[self.blocking_opts][self.trial_type]
        
        if len(self.blocking_joints) > 0:
            for joint in self.blocking_joints:
                command_vel_raw[self.vel_states[joint]] = 0


        for state in self.vel_states:
            # command_vel[state] = self.command_lpfs[state](command_vel[state])
            command_vel_non_scaled[state] = self.command_lpfs[state](command_vel_raw[state])
            if np.isnan(command_vel_non_scaled[state]):
                command_vel_non_scaled[state] = 0
            
            if state in aa_vel_states:
                command_vel[state] = command_vel_non_scaled[state]*self.vel_scale_aa

            if state in rh_vel_states:
                command_vel[state] = command_vel_non_scaled[state]*self.vel_scale_rh
        
        self.task_data['command_vel_non_scaled'] = command_vel_non_scaled 

        self.plant.drive_velocity_sent = command_vel.values
        self.plant.send_vel(command_vel.values) 

    def _start_prep(self):
        self.preparation_time = ( np.random.rand() * (self.prep_max - self.prep_min) ) + self.prep_min
 
    def _start_target(self):
        # self.soundlist_go = self.get_sounds(self.sounds_dir, ['go']) #removed 2018.05.28
        self.soundlist_go = []
        self.go_sound_counter = 0
        self.go_sound_chain = len(self.soundlist_go)
        super(CompliantMovements_w_prep, self)._start_target()


    def _while_target(self):
        if pygame.mixer.music.get_busy():
            pass
        else:
            if self.go_sound_counter >= self.go_sound_chain:
                pass
            else:
                sound = self.soundlist_go[self.go_sound_counter]
                pygame.mixer.music.load(sound)
                pygame.mixer.music.play()
                self.go_sound_counter += 1

    def _test_end_prep(self, ts):
        #if self.go_sound_counter >= self.go_sound_chain:
        return ts >= self.preparation_time

class EMGControl_w_Binary_EMG(CompliantMovements):
    exclude_parent_traits = ['plant_type']
    rest_mov_emg_classifier = traits.InstanceFromDB(SVM_rest_EMGClassifier, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='emg_classifier'))
    emg_decoder             = traits.InstanceFromDB(LinearEMGDecoder, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='emg_decoder'))
    emg_rest_time           = traits.Float(2., desc='none')
    emg_weight_aa              = traits.Float(0.5, desc = 'weight of the EMG in the final velocity command')
    emg_weight_rh              = traits.Float(0.5, desc = 'weight of the EMG in the final velocity command')
    scale_emg_pred_aa       = traits.Float(1., desc='none')
    scale_emg_pred_rh       = traits.Float(1., desc='none')
    attractor_speed         = traits.Float(0., desc='speed to use for safety grid bounce')
    attractor_speed_const   = traits.Float(0., desc='attractor speed on all the time')
    plant_type              = 'IsMoreEMGControl'
    

    status = {
        'wait': {
            'start_trial':         'rest',
            'stop':                None},        
        'rest': {
            'rest_complete':    'emg_rest'},

        'emg_rest': {
            'emg_rest_complete':    'instruct_trial_type'},

        'instruct_trial_type': {
            'end_instruct':        'target' },
        
        'target': {
            'enter_target':        'hold',
            'timeout':             'timeout_penalty',
            'stop':                None},
        
        'hold': {
            'leave_early':         'hold_penalty',
            'hold_complete':       'targ_transition'},
        
        'targ_transition': {
            'trial_complete':      'reward',
            'trial_incomplete':    'instruct_trial_type'},
        
        'timeout_penalty': {
            'timeout_try_again': 'instruct_trial_type',
            'timeout_skip_target': 'instruct_trial_type',
            'timeout_end_of_trial': 'wait'},
        
        'hold_penalty': {
            'hold_penalty_try_again':    'instruct_trial_type',
            'hold_penalty_skip_target':  'instruct_trial_type',
            'hold_penalty_end_of_trial':  'wait'},
        
        'reward': {
            'reward_end': 'wait'}
    }

    def __init__(self, *args, **kwargs):
        super(EMGControl_w_Binary_EMG, self).__init__(*args, **kwargs)
        self.channels = self.emg_decoder.extractor_kwargs['emg_channels']
        self.add_dtype('assist_output', 'f8', (len(self.command_vel),))
    
    def init(self):
        self.load_safety() # load safety grid after plant has been loaded
        self.plant.attractor_speed = self.attractor_speed
        self.plant.attractor_speed_const = self.attractor_speed_const

        self.emg_classifier_extractor = self.rest_mov_emg_classifier.extractor_cls(None, 
            emg_channels = self.rest_mov_emg_classifier.extractor_kwargs['emg_channels'], 
            feature_names = self.rest_mov_emg_classifier.extractor_kwargs['feature_names'], 
            win_len = self.rest_mov_emg_classifier.extractor_kwargs['win_len'], 
            fs = self.rest_mov_emg_classifier.extractor_kwargs['fs'])

        self.emg_decoder_extractor = self.emg_decoder.extractor_cls(None, emg_channels = self.emg_decoder.extractor_kwargs['emg_channels'], feature_names = self.emg_decoder.extractor_kwargs['feature_names'], feature_fn_kwargs = self.emg_decoder.extractor_kwargs['feature_fn_kwargs'], win_len=self.emg_decoder.extractor_kwargs['win_len'], fs=self.emg_decoder.extractor_kwargs['fs'])
        
        #self.emg_decoder_name = self.emg_decoder.decoder_name

        # for calculating/updating mean and std of EMG features online
        self.features_buffer = RingBuffer(
            item_len=self.emg_decoder_extractor.n_features,
            capacity=60*self.fps,  # 60 secs
        )

        # for low-pass filtering decoded EMG velocities
        # self.emg_vel_buffer = RingBuffer(
        #     item_len=len(self.vel_states),
        #     capacity=10,
        # )
        
        self.emg_decoder_extractor.source = self.brainamp_source
        self.emg_classifier_extractor.source = self.brainamp_source
        self.plant.rest_emg_output = 1 # Full movement
        self.plant.emg_vel = np.zeros((len(self.vel_states), ))

        # If I save the ones from the decoder the WL is already included. Probably we can avoid saving it twice. 
        self.add_dtype('emg_decoder_features','f8', (self.emg_decoder_extractor.n_features,))
        self.add_dtype('emg_decoder_features_Z','f8', (self.emg_decoder_extractor.n_features,))
        #self.add_dtype('emg_classifier_features','f8', (self.emg_classifier_extractor.n_features,))
        #self.add_dtype('emg_classifier_features_Z','f8', (self.emg_classifier_extractor.n_features,))
        self.add_dtype('rest_emg_output',  float, (1,))
        self.add_dtype('emg_vel_raw',         'f8', (len(self.vel_states),))

        super(EMGControl_w_Binary_EMG, self).init()

    def move_plant(self):
        current_state = np.hstack([self.plant_pos.values, self.plant_vel.values, 1])[:, None]
        #vfb_kin = pd.Series(np.squeeze(np.array(self.assister(current_state, target_state, 1)['Bu'])),
        #    self.ssm_states)

        data, solution_updated = self.goal_calculator(self.target_pos.values, self.state, 
            **dict(current_state=current_state))
        
        #data, solution_updated = self.goal_calculator(self.target_pos.values)
        target_state = data[0].reshape(-1, 1)

        # Preeya, 10/3/17 removed
        # assist_output = self.assister(current_state, target_state, 1.)
        # vfb_kin = pd.Series(np.squeeze(np.array(assist_output["x_assist"]).ravel()), self.ssm_states)

        # command_vel = vfb_kin[self.vel_states]
        # self.plant.drive_velocity = command_vel
        # self.plant.send_vel(command_vel)#, self.fps)

        # if self.state in ['wait','rest', 'instruct_rest', 'instruct_trial_type']: 
        #     #command_vel_raw[:] = 0
        #     target_state = current_state
    
        assist_output = self.assister(current_state, target_state, 1.)
        Bu = pd.Series(np.array(assist_output["x_assist"]).ravel(), self.ssm_states)
        assist_vel_raw = Bu[self.vel_states]

        if self.emg_classifier_extractor is not None:
            emg_classifier_features = self.emg_classifier_extractor() # emg_features is of type 'dict' 
            emg_classifier_features_Z = (emg_classifier_features[self.emg_classifier_extractor.feature_type] - self.rest_mov_emg_classifier.features_mean_train) / self.rest_mov_emg_classifier.features_std_train       
            # classifier_output = self.rest_mov_emg_classifier(emg_features[self.emg_extractor.feature_type])
            classifier_output = self.rest_mov_emg_classifier(emg_classifier_features_Z)
            # output = 1 if moving, 0 if rest
            self.plant.rest_emg_output = classifier_output

        
        # Feature extraction for EMG decoder
        emg_decoder_features = self.emg_decoder_extractor() # emg_features is of type 'dict'

        #emg_features[self.emg_extractor.feature_type] = emg_features[self.emg_extractor.feature_type][len(emg_features[self.emg_extractor.feature_type])/2:]
        self.features_buffer.add(emg_decoder_features[self.emg_decoder_extractor.feature_type])
        #print 'emg_features[self.emg_extractor.feature_type]', emg_features[self.emg_extractor.feature_type].shape
        
        if self.features_buffer.num_items() > 60 * self.fps: # changed to 60 seconds, 14/05/2018
                # if we have more than 1 second of recent EMG data, then
                #   calculate mean and std from this data
            recent_features = self.features_buffer.get_all()
            #print 'recent_features', recent_features.shape
            features_mean = np.mean(recent_features, axis=1)
            features_std  = np.std(recent_features, axis=1)
        else:
                # else use mean and std from the EMG data that was used to 
                #   train the decoder
            features_mean = self.emg_decoder.features_mean
            features_std  = self.emg_decoder.features_std

        features_std[features_std == 0] = 1

        # z-score the EMG features
        emg_decoder_features_Z = (emg_decoder_features[self.emg_decoder_extractor.feature_type] - features_mean) / features_std 
        # compute velocity from EMG decoder
        self.plant.emg_vel = self.emg_decoder(emg_decoder_features_Z)
        # self.emg_vel_buffer.add(self.plant.emg_vel[self.vel_states])

        # Filter output velocity from EMG decoder using a weighted moving avge filter (optional). Otherwise filter at 5Hz, as neural signal
        # n_items = self.emg_vel_buffer.num_items()
        # buffer_emg = self.emg_vel_buffer.get(n_items)
        # win = min(9,n_items)
        # weights = np.arange(1./win, 1 + 1./win, 1./win)
        # try:
        #     emg_vel_lpf = np.sum(weights*buffer_emg[:,n_items-win:n_items+1], axis = 1)/np.sum(weights)
        # except:
        #     pass
            
        # If I save the ones from the decoder the WL is already included. Probably we can avoid saving it twice. 
        self.task_data['emg_decoder_features'] = emg_decoder_features[self.emg_decoder_extractor.feature_type]
        self.task_data['emg_decoder_features_Z'] = emg_decoder_features_Z 
        #self.task_data['emg_classifier_features'] = emg_classifier_features[self.emg_classifier_extractor.feature_type]
        #self.task_data['emg_classifier_features_Z'] = emg_classifier_features_Z       
        self.task_data['rest_emg_output'] = self.plant.rest_emg_output
        self.task_data['emg_vel_raw'] = self.plant.emg_vel
        self.task_data['assist_output'] = assist_vel_raw
        #self.plant.rest_emg_output =0

        # Fuse EMG velocity component with assistive component       
        # SPLIT AA / RH
        assist_vel_raw_aa = assist_vel_raw[[0, 1, 2]]
        assist_vel_raw_rh = assist_vel_raw[[3, 4, 5, 6]]

        emg_vel_aa = self.plant.emg_vel[[0, 1, 2]]*self.scale_emg_pred_aa
        emg_vel_rh = self.plant.emg_vel[[3, 4, 5, 6]]*self.scale_emg_pred_rh

        command_vel_raw_aa = assist_vel_raw_aa*(1-self.emg_weight_aa) + emg_vel_aa*self.emg_weight_aa
        command_vel_raw_rh = assist_vel_raw_rh*(1-self.emg_weight_rh) + emg_vel_rh*self.emg_weight_rh
        command_vel_raw = np.hstack((command_vel_raw_aa, command_vel_raw_rh))
        command_vel_raw = pd.Series(command_vel_raw, self.vel_states)

        self.plant.drive_velocity_raw = command_vel_raw

        if np.any(np.isnan(command_vel_raw)):
            print 'setting command_vel_raw nans equal to zero'
            command_vel_raw[np.isnan(command_vel_raw)] = 0

        #filter command_vel
        command_vel  = pd.Series(0.0, self.vel_states)
        
        if self.state in ['wait','rest', 'instruct_rest', 'instruct_trial_type']: 
            command_vel_raw[:] = 0

        for state in self.vel_states:
            # command_vel[state] = self.command_lpfs[state](command_vel[state])
            command_vel[state] = self.command_lpfs[state](command_vel_raw[state])
            if np.isnan(command_vel[state]):
                command_vel[state] = 0
        
        self.plant.drive_velocity_sent = command_vel.values
        self.plant.send_vel(command_vel.values) 

    def _cycle(self): # This can be skipped
        super(EMGControl_w_Binary_EMG, self)._cycle()
    
    def _test_emg_rest_complete(self, ts):
        return ts >= self.emg_rest_time

class BMIControl(BMILoop, LinearlyDecreasingXYAssist, LinearlyDecreasingAngAssist, PlantControlBase):
#class BMIControl(BMILoop, LinearlyDecreasingAssist, PlantControlBase):

    '''Target capture task with plant controlled by BMI output.
    Cursor movement can be assisted toward target by setting assist_level > 0.
    '''
    debug = True
    #max_attempts = traits.Int(3, desc='Max attempts allowed to a target before skipping to the next one')
    assist_opts = ['endpt', 'ortho_damp']
    assist_type = traits.OptionsList(*assist_opts, bmi3d_input_options=assist_opts, desc='Assist Type')
    ordered_traits = ['session_length', 'assist_level', 'assist_level_time', 'reward_time']
    assist_speed = traits.OptionsList(*speed_options, bmi3d_input_options= speed_options)
    attractor_speed = traits.Float(0., desc='speed to use for safety grid bounce')
    attractor_speed_const = traits.Float(0., desc='attractor speed on all the time')
    decoder_drift_hl = traits.Float(0., desc='half')

    def __init__(self, *args, **kwargs):
        super(BMIControl, self).__init__(*args, **kwargs)
        self.add_dtype('drift_correction', 'f8', (len(self.pos_states+self.vel_states+[1]), ))

    def _cycle(self):
        try:
            self.task_data['drift_correction'] = np.squeeze(np.array(self.decoder.filt.drift_corr))
        except:
            pass
        super(BMIControl, self)._cycle()  


    def init(self):
        self.load_safety() # load safety grid after plant has been loaded
        self.plant.attractor_speed = self.attractor_speed
        self.plant.attractor_speed_const = self.attractor_speed_const
        super(BMIControl, self).init()

    # overrides BMILoop.init_decoder_state, without calling it
    def init_decoder_state(self):
        if self.decoder_drift_hl > 0:
            drift_rho = np.exp(np.log(0.5) / (self.decoder_drift_hl/ self.decoder.binlen))
        else:
            drift_rho = 1.
        self.decoder.filt.drift_rho = drift_rho
        vel_ix = np.nonzero(self.decoder.ssm.state_order==1)[0]
        self.decoder.filt.vel_ix = vel_ix
        self.decoder.filt._init_state()
        self.decoder['q'] = self.starting_pos[self.pos_states].values
        self.init_decoder_mean = self.decoder.filt.state.mean
        self.decoder.set_call_rate(self.fps)

    def create_assister(self):
        kwargs = {
            'call_rate': self.fps,
            'xy_cutoff': np.min([self.target_radius_x, self.target_radius_y]),
            'speed':    self.assist_speed,
        }
        if self.assist_type == 'endpt':
            #self.assister = ismore_bmi_lib.LFC_ASSISTER_CLS_DICT[self.plant_type](**kwargs)
            self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT[self.plant_type](**kwargs)
        
            #self.assister = ismore_bmi_lib.ASSISTER_CLS_DICT[self.plant_type](**kwargs)
        elif self.assist_type == 'ortho_damp':
            self.assister = ismore_bmi_lib.ORTHO_DAMP_ASSIST_CLS_DICT[self.plant_type](**kwargs)

    def get_current_assist_level(self):
        return np.array([self.current_aa_assist_level, self.current_rh_assist_level])

    def create_goal_calculator(self):
        #self.goal_calculator = ZeroVelocityGoal(self.decoder.ssm)
        self.goal_calculator = ZeroVelocityGoal_ismore(self.decoder.ssm, 
            pause_states = ['rest', 'wait', 'instruct_rest', 'instruct_trial_type'])
        #self.goal_calculator = ismore_bmi_lib.GOAL_CALCULATOR_CLS_DICT[self.plant_type](self.decoder.ssm)

    def get_current_state(self):
        self.plant_pos[:] = self.plant.get_pos()
        current_state = np.array(self.decoder.filt.state.mean).ravel()
        current_state = pd.Series(current_state, self.ssm_states)
        current_state = current_state.values.reshape(-1, 1)
        return current_state        

    def get_target_BMI_state(self, *args):
        '''Run the goal calculator to determine the current target state.'''
        current_state = np.hstack([self.plant_pos.values, self.plant_vel.values, 1])[:, None]
        if np.any(np.isnan(current_state)):
            current_state[np.isnan(current_state)] = 0

        data, solution_updated = self.goal_calculator(self.target_pos.values, self.state, 
            **dict(current_state=current_state))
        
        target_state = data[0].reshape(-1, 1)
        if np.any(np.isnan(target_state)):
            target_state[np.isnan(target_state)] = 0        
        #target_state = np.hstack(( self.target_pos.values, [1]))
        #target_state = target_state.reshape(-1, 1)
        return np.tile(np.array(target_state).reshape(-1, 1), [1, self.decoder.n_subbins])

    # def _end_timeout_penalty(self):
    #     if self.reset:
    #         self.decoder.filt.state.mean = self.init_decoder_mean
    #         self.hdf.sendMsg("reset")

    def cleanup_hdf(self):
        super(BMIControl, self).cleanup_hdf()
        self.decoder.save_attrs(self.h5file.name, 'task')

    def set_state(self, *args, **kwargs):
        print args
        super(BMIControl, self).set_state(*args, **kwargs)

jingle_namelist = ['jingle2','jingle3', 'jingle4']

class BMIControl_1D_along_traj(BMIControl):
    '''
    Task used in sleep study with hud1
    '''
    sequence_generators = ['sleep_gen']
    exclude_parent_traits = ['assist_type', 'attractor_speed', 'attractor_speed_const',
    'decoder_drift_hl', 'assist_speed', 'rh_assist_level', 'rh_assist_level_time',
    'aa_assist_level','aa_assist_level_time', 'max_attempt_per_targ', 
    'timeout_penalty_time', 'arm_side', 'language', 'show_FB_window']
    arm_side = 'right'
    assist_type = 'endpt'
    attractor_speed = 0.01
    attractor_speed_const = 0.
    decoder_drift_hl = 0.
    assist_speed = 'high'
    rh_assist_level = (1., 1.)
    rh_assist_level_time = 1.
    aa_assist_level = (1., 1.)
    aa_assist_level_time = 1.
    jingle_name = traits.OptionsList(*jingle_namelist, bmi3d_input_options=jingle_namelist) 
    max_attempt_per_targ = 1.
    timeout_penalty_time = .1
    last_tmp = np.ones((1, 1))
    language = 'castellano'
    baseline_timeout_time = 30.
    speed= 'high'
    n_decoder_steps = 1
    plant_type = 'IsMore'

    status = dict(wait = dict(start_trial ='rest', stop=None),
        rest = dict(rest_complete = 'baseline_check', stop=None),
        baseline_check = dict(baseline_reached = 'instruct_trial_type', baseline_timeout = 'wait', stop=None),
        instruct_trial_type = dict(end_instruct = 'target', stop=None),
        target = dict(enter_target = 'hold', timeout = 'timeout_penalty', stop=None),
        hold = dict(leave_early = 'hold_penalty', hold_complete='reward', stop=None),
        reward = dict(reward_end = 'drive_to_start', stop=None),
        hold_penalty = dict(hold_penalty_end ='drive_to_start', stop=None),
        timeout_penalty = dict(timeout_penalty_end = 'drive_to_start', stop=None),
        drive_to_start = dict(at_start = 'wait', stop=None))

    def init(self):
        self.add_dtype('e1_minus_e2', float, (1, ))
        super(BMIControl_1D_along_traj, self).init()

        assister_kwargs = dict(call_rate=self.fps, xy_cutoff=np.min([self.target_radius_x, self.target_radius_y]),
            speed =self.speed)

        # "CursorGoal" Assister a la Gilja 2012
        self.assister = ismore_bmi_lib.IsMoreAssister(**assister_kwargs)
        self.decoder.filt.init_from_task(**dict(nsteps=self.n_decoder_steps))
        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, 'sleep')


    def _test_baseline_timeout(self, ts):
        return ts > self.baseline_timeout_time

    def _test_baseline_reached(self, ts):
        return self.decoder.filt.baseline

    def _start_drive_to_start(self):
        self.target_index += 1
        try:
            self.target_pos[:] = self.targets[self.target_index]
        except:
            self.target_pos[:] = pd.Series(np.hstack(( self.targets[self.target_index][self.pos_states], np.zeros((len(self.targets[self.target_index][self.pos_states]))) )), self.pos_states+self.vel_states)
        self.soundlist = self.get_sounds(self.sounds_dir, ['back'])
        self.sound_counter = 0 # Increments after playing sound
        self.sound_chain = len(self.soundlist)        

    def _start_instruct_trial_type(self):
        # Play 'go' + jingle: 
        self.target_index += 1
        try:
            self.target_pos[:] = self.targets[self.target_index]
        except:
            self.target_pos[:] = pd.Series(np.hstack(( self.targets[self.target_index][self.pos_states], np.zeros((len(self.targets[self.target_index][self.pos_states]))) )), self.pos_states+self.vel_states)
        self.soundlist = self.get_sounds(self.sounds_dir, ['go']) + self.get_sounds(self.sounds_dir, [self.jingle_name]) 
        self.sound_counter = 0 # Increments after playing sound
        self.sound_chain = len(self.soundlist)

    def _while_drive_to_start(self):
        self.ignore_decoder = True
        if pygame.mixer.music.get_busy():
            pass
        else:
            if self.sound_counter < self.sound_chain:
                sound = self.soundlist[self.sound_counter]
                pygame.mixer.music.load(sound)
                pygame.mixer.music.play()
                self.sound_counter += 1

    def _start_wait(self):
        self.ignore_decoder = False
        super(BMIControl_1D_along_traj, self)._start_wait()
    
    def _test_at_start(self, ts):
        in_target = self.armassist_inside_target() and self.rehand_inside_target()
        done_sound = self.sound_counter >= self.sound_chain
        return in_target and done_sound
        
    def move_plant(self, **kwargs):

        # get features: 
        features = self.get_features()
        
        for key, val in features.items():
            self.task_data[key] = val 
        neural_features = features[self.extractor.feature_type]  
        if len(neural_features) == 0:
            print 'NO NEURAL FEATURES STREAMING'
            neural_features = np.zeros((self.decoder.n_features, 1))
        # Get assist arguments 
        current_state = np.hstack([self.plant_pos.values, self.plant_vel.values, 1])[:, None]
        data, solution_updated = self.goal_calculator(self.target_pos.values, self.state, 
            **dict(current_state=current_state))
        
        target_state = data[0].reshape(-1, 1)
        assist_output = self.assister(current_state, target_state, 1.)

        # call decoder: 
        kwargs.update(assist_output)
        tmp = self.call_decoder(neural_features, target_state, **kwargs)

        # Save decoder output
        self.task_data['decoder_state'] = tmp
        self.task_data['e1_minus_e2'] = self.decoder.filt.FR

        # get alpha value: 
        try:
            assist = assist_output['x_assist']
            Bu = pd.Series(np.array(assist_output["x_assist"]).ravel(), self.ssm_states)
        except:
            Bu = pd.Series(np.squeeze(np.array(assist_output['Bu'])), self.ssm_states)
        
        # Remove any nans, shouldn't be any (assister also does this step)
        command_vel_raw = Bu[self.vel_states]
        command_vel_raw[np.isnan(command_vel_raw)]=0
        
        if self.ignore_decoder:
            self.decoder.qdot = np.squeeze(np.array(command_vel_raw))
        else:
            self.decoder.qdot = np.squeeze(np.array(command_vel_raw*tmp[0,0]))
        
        self.plant.drive(self.decoder)

    @staticmethod
    def sleep_gen(length=100, sleep_target=1):
        return ['sleep_target']*length
        
class BMIControl_w_Binary_EMG(BMIControl):
    rest_mov_emg_classifier    = traits.InstanceFromDB(SVM_rest_EMGClassifier, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='emg_classifier'))
    emg_rest_time =        traits.Float(2., desc='none')

    status = {
        'wait': {
            'start_trial':         'rest',
            'stop':                None},        
        'rest': {
            'rest_complete':    'emg_rest'},

        'emg_rest': {
            'emg_rest_complete':    'instruct_trial_type'},

        'instruct_trial_type': {
            'end_instruct':        'target' },
        
        'target': {
            'enter_target':        'hold',
            'timeout':             'timeout_penalty',
            'stop':                None},
        
        'hold': {
            'leave_early':         'hold_penalty',
            'hold_complete':       'targ_transition'},
        
        'targ_transition': {
            'trial_complete':      'reward',
            'trial_incomplete':    'instruct_trial_type'},
        
        'timeout_penalty': {
            'timeout_try_again': 'instruct_trial_type',
            'timeout_skip_target': 'instruct_trial_type',
            'timeout_end_of_trial': 'wait'},
        
        'hold_penalty': {
            'hold_penalty_try_again':    'instruct_trial_type',
            'hold_penalty_skip_target':  'instruct_trial_type',
            'hold_penalty_end_of_trial':  'wait'},
        
        'reward': {
            'reward_end': 'wait'}
    }


    def init(self):
        self.emg_extractor = self.rest_mov_emg_classifier.extractor_cls(None, 
            emg_channels = self.rest_mov_emg_classifier.extractor_kwargs['emg_channels'], 
            feature_names = self.rest_mov_emg_classifier.extractor_kwargs['feature_names'], 
            win_len = self.rest_mov_emg_classifier.extractor_kwargs['win_len'], 
            fs = self.rest_mov_emg_classifier.extractor_kwargs['fs'])

        self.emg_extractor.source = self.brainamp_source
        self.plant.rest_emg_output = 1 # Full movement

        self.add_dtype('emg_features','f8', (self.emg_extractor.n_features,))
        self.add_dtype('emg_features_Z','f8', (self.emg_extractor.n_features,))
        self.add_dtype('rest_emg_output',  float, (1,))

        super(BMIControl_w_Binary_EMG, self).init()

    def _cycle(self):
        super(BMIControl_w_Binary_EMG, self)._cycle()

        if self.emg_extractor is not None:
            emg_features = self.emg_extractor() # emg_features is of type 'dict' 
            emg_features_Z = (emg_features[self.emg_extractor.feature_type] - self.rest_mov_emg_classifier.features_mean_train) / self.rest_mov_emg_classifier.features_std_train       
            # classifier_output = self.rest_mov_emg_classifier(emg_features[self.emg_extractor.feature_type])
            classifier_output = self.rest_mov_emg_classifier(emg_features_Z)
           
            # output = 1 if moving, 0 if rest
            self.plant.rest_emg_output = classifier_output

        self.task_data['emg_features'] = emg_features[self.emg_extractor.feature_type]
        self.task_data['emg_features_Z'] = emg_features_Z       
        self.task_data['rest_emg_output'] = self.plant.rest_emg_output
        #self.plant.rest_emg_output =0
    
    def _test_emg_rest_complete(self, ts):
        return ts >= self.emg_rest_time

class Hybrid_BMIControl_w_Binary_EMG(BMIControl):
    exclude_parent_traits = ['plant_type']
    rest_mov_emg_classifier = traits.InstanceFromDB(SVM_rest_EMGClassifier, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='emg_classifier'))
    emg_decoder             = traits.InstanceFromDB(LinearEMGDecoder, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='emg_decoder'))
    emg_rest_time           = traits.Float(2., desc='none')
    emg_weight_aa           = traits.Float(0.5, desc = 'weight of the EMG in the final velocity command')
    emg_weight_fingers      = traits.Float(0.5, desc = 'weight of the EMG in the final velocity command of the fingers DoFs')
    emg_weight_prono        = traits.Float(0.5, desc = 'weight of the EMG in the final velocity command of the pronosup DoF')
    #scale_emg_pred_aa       = traits.Float(1., desc='none')
    #scale_emg_pred_rh       = traits.Float(1., desc='none')
    plant_type              = 'IsMoreHybridControl'
    fb_gain = np.array([1, 1, 1, 1.5, 2, 2.5, 1.5])

    status = {
        'wait': {
            'start_trial':         'rest',
            'stop':                None},        
        'rest': {
            'rest_complete':    'emg_rest'},

        'emg_rest': {
            'emg_rest_complete':    'instruct_trial_type'},

        'instruct_trial_type': {
            'end_instruct':        'target' },
        
        'target': {
            'enter_target':        'hold',
            'timeout':             'timeout_penalty',
            'stop':                None},
        
        'hold': {
            'leave_early':         'hold_penalty',
            'hold_complete':       'targ_transition'},
        
        'targ_transition': {
            'trial_complete':      'reward',
            'trial_incomplete':    'instruct_trial_type'},
        
        'timeout_penalty': {
            'timeout_try_again': 'instruct_trial_type',
            'timeout_skip_target': 'instruct_trial_type',
            'timeout_end_of_trial': 'wait'},
        
        'hold_penalty': {
            'hold_penalty_try_again':    'instruct_trial_type',
            'hold_penalty_skip_target':  'instruct_trial_type',
            'hold_penalty_end_of_trial':  'wait'},
        
        'reward': {
            'reward_end': 'wait'}
    }

    def __init__(self, *args, **kwargs):
        super(Hybrid_BMIControl_w_Binary_EMG, self).__init__(*args, **kwargs)
        self.channels = self.emg_decoder.extractor_kwargs['emg_channels']
        self.add_dtype('drive_velocity_raw_brain',  'f8', (len(self.vel_states),))
        self.add_dtype('drive_velocity_raw_fb_gain', 'f8', (len(self.vel_states),))
        self.add_dtype('emg_vel_raw_scaled',  'f8', (len(self.vel_states),))

    def init(self):
        self.emg_classifier_extractor = self.rest_mov_emg_classifier.extractor_cls(None, 
            emg_channels = self.rest_mov_emg_classifier.extractor_kwargs['emg_channels'], 
            feature_names = self.rest_mov_emg_classifier.extractor_kwargs['feature_names'], 
            win_len = self.rest_mov_emg_classifier.extractor_kwargs['win_len'], 
            fs = self.rest_mov_emg_classifier.extractor_kwargs['fs'])

        self.emg_decoder_extractor = self.emg_decoder.extractor_cls(None, emg_channels = self.emg_decoder.extractor_kwargs['emg_channels'], 
            feature_names = self.emg_decoder.extractor_kwargs['feature_names'], 
            feature_fn_kwargs = self.emg_decoder.extractor_kwargs['feature_fn_kwargs'], 
            win_len=self.emg_decoder.extractor_kwargs['win_len'], fs=self.emg_decoder.extractor_kwargs['fs'])

        #self.emg_decoder_name = self.emg_decoder.decoder_name

        # for calculating/updating mean and std of EMG features online
        self.features_buffer = RingBuffer(
            item_len=self.emg_decoder_extractor.n_features,
            capacity=120*self.fps,  # 60 secs
        )

        # for low-pass filtering decoded EMG velocities
        # self.emg_vel_buffer = RingBuffer(
        #     item_len=len(self.vel_states),
        #     capacity=10,
        # )

        self.emg_classifier_extractor.source = self.brainamp_source
        self.emg_decoder_extractor.source = self.brainamp_source
        self.plant.rest_emg_output = 1 # Full movement
        self.plant.emg_weight_aa = self.emg_weight_aa
        self.plant.emg_weight_fingers = self.emg_weight_fingers
        self.plant.emg_weight_prono = self.emg_weight_prono
        self.plant.fb_vel_gain = self.fb_gain

        # Ratio of EMG to BMI to make them equally distributed
        andrea = np.array([6.14073, 3.21564, 3.13008, 20.94372, 5.58169, 4.54888, 14.17686])
        self.plant.scale_emg_pred_arr = andrea
        self.plant.emg_vel = np.zeros((len(self.vel_states), ))

        # If I save the ones from the decoder the WL is already included. Probably we can avoid saving it twice. 
        self.add_dtype('emg_decoder_features','f8', (self.emg_decoder_extractor.n_features,))
        self.add_dtype('emg_decoder_features_Z','f8', (self.emg_decoder_extractor.n_features,))
        self.add_dtype('emg_decoder_features_mn','f8', (self.emg_decoder_extractor.n_features,))
        self.add_dtype('emg_decoder_features_std','f8', (self.emg_decoder_extractor.n_features,))
        self.add_dtype('rest_emg_output',  float, (1,))
        self.add_dtype('emg_vel_raw',         'f8', (len(self.vel_states),))

        super(Hybrid_BMIControl_w_Binary_EMG, self).init()

    def _cycle(self):
        super(Hybrid_BMIControl_w_Binary_EMG, self)._cycle()

        if self.emg_classifier_extractor is not None:
            emg_classifier_features = self.emg_classifier_extractor() # emg_features is of type 'dict' 
            emg_classifier_features_Z = (emg_classifier_features[self.emg_classifier_extractor.feature_type] - self.rest_mov_emg_classifier.features_mean_train) / self.rest_mov_emg_classifier.features_std_train       
            # classifier_output = self.rest_mov_emg_classifier(emg_features[self.emg_extractor.feature_type])
            classifier_output = self.rest_mov_emg_classifier(emg_classifier_features_Z)
            # output = 1 if moving, 0 if rest
            self.plant.rest_emg_output = classifier_output

        
        # Feature extraction for EMG decoder
        emg_decoder_features = self.emg_decoder_extractor() # emg_features is of type 'dict'

        #emg_features[self.emg_extractor.feature_type] = emg_features[self.emg_extractor.feature_type][len(emg_features[self.emg_extractor.feature_type])/2:]
        self.features_buffer.add(emg_decoder_features[self.emg_decoder_extractor.feature_type])
        #print 'emg_features[self.emg_extractor.feature_type]', emg_features[self.emg_extractor.feature_type].shape
        
        if self.features_buffer.num_items() > 60 * self.fps: # changed to 60 seconds, 14/05/2018
                # if we have more than 1 second of recent EMG data, then
                #   calculate mean and std from this data
            recent_features = self.features_buffer.get_all()
            #print 'recent_features', recent_features.shape
            self.features_mean_emg_dec = np.mean(recent_features, axis=1)

            if hasattr(self.emg_decoder, 'fixed_var_scalar') and self.emg_decoder.fixed_var_scalar:
                self.features_std_emg_dec  = self.emg_decoder.recent_features_std
            else:
                self.features_std_emg_dec = np.std(recent_features, axis=1)

        else:
            try:
                # first try to use most recent saved data: 
                self.features_mean_emg_dec = self.emg_decoder.recent_features_mean
                self.features_std_emg_dec  = self.emg_decoder.recent_features_std 

            except:
                # else use mean and std from the EMG data that was used to 
                # train the decoder
                self.features_mean_emg_dec = self.emg_decoder.features_mean
                self.features_std_emg_dec  = self.emg_decoder.features_std

        try:
            self.features_std_emg_dec[self.features_std_emg_dec == 0] = 1
        except:
            pass

        # z-score the EMG features
        emg_decoder_features_Z = (emg_decoder_features[self.emg_decoder_extractor.feature_type] - self.features_mean_emg_dec) / self.features_std_emg_dec 
        
        # compute velocity from EMG decoder
        self.plant.emg_vel = self.emg_decoder(emg_decoder_features_Z)
        # self.emg_vel_buffer.add(self.plant.emg_vel[self.vel_states])

            
        # If I save the ones from the decoder the WL is already included. Probably we can avoid saving it twice. 
        self.task_data['emg_decoder_features'] = emg_decoder_features[self.emg_decoder_extractor.feature_type]
        self.task_data['emg_decoder_features_Z'] = emg_decoder_features_Z 
        self.task_data['rest_emg_output'] = self.plant.rest_emg_output
        self.task_data['emg_vel_raw'] = self.plant.emg_vel
        self.task_data['emg_vel_raw_scaled'] = self.plant.emg_vel_raw_scaled
        self.task_data['drive_velocity_raw_brain'] = self.plant.drive_velocity_raw_brain
        self.task_data['drive_velocity_raw_fb_gain'] = self.plant.drive_velocity_raw_fb_gain
        self.task_data['emg_decoder_features_mn'] = self.features_mean_emg_dec
        self.task_data['emg_decoder_features_std'] = self.features_std_emg_dec
            
    def _test_emg_rest_complete(self, ts):
        return ts >= self.emg_rest_time

    def cleanup(self, database, saveid, **kwargs):
        super(Hybrid_BMIControl_w_Binary_EMG, self).cleanup(database, saveid, **kwargs)

        # Save z-scored features from EMG decoder:
        self.emg_decoder.recent_features_mean = self.features_mean_emg_dec
        self.emg_decoder.recent_features_std = self.features_std_emg_dec
        self.emg_decoder.recent_features_te = saveid

        # Save this
        import pickle
        pickle.dump(self.emg_decoder, open(self.emg_decoder.path, 'wb'))

        print ' saving EMG decoder zscore features: ', self.emg_decoder.path

class Hybrid_BMIControl_w_Binary_EMG_BackToStart(Hybrid_BMIControl_w_Binary_EMG):
    rest_back_attractor_speed = traits.Float(0.025, desc='attr')
    rest_back_time = traits.Float(2., desc='none')

    status = {
        'wait': {
            'start_trial':         'rest_back',
            'stop':                None}, 

        'rest_back': {
            'rest_back_complete':   'rest',
            'stop':                 None},

        'rest': {
            'rest_complete':    'emg_rest'},

        'emg_rest': {
            'emg_rest_complete':    'instruct_trial_type'},

        'instruct_trial_type': {
            'end_instruct':        'target' },
        
        'target': {
            'enter_target':        'hold',
            'timeout':             'timeout_penalty',
            'stop':                None},
        
        'hold': {
            'leave_early':         'hold_penalty',
            'hold_complete':       'targ_transition'},
        
        'targ_transition': {
            'trial_complete':      'reward',
            'trial_incomplete':    'instruct_trial_type'},
        
        'timeout_penalty': {
            'timeout_try_again': 'instruct_trial_type',
            'timeout_skip_target': 'instruct_trial_type',
            'timeout_end_of_trial': 'wait'},
        
        'hold_penalty': {
            'hold_penalty_try_again':    'instruct_trial_type',
            'hold_penalty_skip_target':  'instruct_trial_type',
            'hold_penalty_end_of_trial':  'wait'},
        
        'reward': {
            'reward_end': 'wait'}
    }

    def init(self):
        self.plant.rest_back_attractor_speed = self.rest_back_attractor_speed
        super(Hybrid_BMIControl_w_Binary_EMG_BackToStart, self).init()

    def _start_rest_back(self):
        self._play_sound(self.sounds_dir, ['rest'])

    def _test_rest_back_complete(self, ts):
        return ts > self.rest_back_time and not pygame.mixer.music.get_busy()

    def _start_rest(self):
        pass

class Hybrid_BMIControl_w_BinEMG_Back2Start_Prep(Hybrid_BMIControl_w_Binary_EMG_BackToStart):
    status = {
        'wait': {
            'start_trial':         'rest_back',
            'stop':                None}, 

        'rest_back': {
            'rest_back_complete':   'rest',
            'stop':                 None},

        'rest': {
            'rest_complete':    'emg_rest'},

        'emg_rest': {
            'emg_rest_complete':    'instruct_trial_type'},

        'instruct_trial_type': {
            'end_instruct':        'prep' },

        'prep': {
            'end_prep':             'target'},

        'target': {
            'enter_target':        'hold',
            'timeout':             'timeout_penalty',
            'stop':                None},
        
        'hold': {
            'leave_early':         'hold_penalty',
            'hold_complete':       'targ_transition'},
        
        'targ_transition': {
            'trial_complete':      'reward',
            'trial_incomplete':    'instruct_trial_type'},
        
        'timeout_penalty': {
            'timeout_try_again': 'instruct_trial_type',
            'timeout_skip_target': 'instruct_trial_type',
            'timeout_end_of_trial': 'wait'},
        
        'hold_penalty': {
            'hold_penalty_try_again':    'instruct_trial_type',
            'hold_penalty_skip_target':  'instruct_trial_type',
            'hold_penalty_end_of_trial':  'wait'},
        
        'reward': {
            'reward_end': 'wait'}
    }

    prep_min = traits.Float(1, desc='minimum of prep phase')
    prep_max = traits.Float(1, desc='maximum of prep phase')

    def _start_prep(self):
        self.preparation_time = ( np.random.rand() * (self.prep_max - self.prep_min) ) + self.prep_min
        
    def _start_target(self):
        # self.soundlist_go = self.get_sounds(self.sounds_dir, ['go'])
        self.soundlist_go = [] #removed 2018.05.28
        self.go_sound_counter = 0
        self.go_sound_chain = len(self.soundlist_go)
        super(Hybrid_BMIControl_w_BinEMG_Back2Start_Prep, self)._start_target()

    def _while_target(self):
        if pygame.mixer.music.get_busy():
            pass
        else:
            if self.go_sound_counter >= self.go_sound_chain:
                pass
            else:
                sound = self.soundlist_go[self.go_sound_counter]
                pygame.mixer.music.load(sound)
                pygame.mixer.music.play()
                self.go_sound_counter += 1
        #super(Hybrid_BMIControl_w_BinEMG_Back2Start_Prep, self)._while_target()

    def _test_end_prep(self, ts):
        #if self.go_sound_counter >= self.go_sound_chain:
        return ts >= self.preparation_time
        #else:
        #    return False

class Hybrid_EMGHandClassification_BMIControl_w_BinEMG_Back2Start_Prep(Hybrid_BMIControl_w_BinEMG_Back2Start_Prep):
    grasp_emg_classifier = traits.InstanceFromDB(SVM_mov_EMGClassifier,
        bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='grasp_emg_classifier'))
    
    #grasp_emg_classifier_path = '/storage/decoders/20180306_HUD1_grasp_emg_classifier.pkl'
    #grasp_emg_classifier_path = '/storage/decoders/20180306_HUD1_grasp_emg_classifier_scalarvar_True.pkl'
    #try:
    #grasp_emg_classifier = grasp_emg_classifier0
    #except:
    #grasp_emg_classifier = pickle.load(open(grasp_emg_classifier_path))

    # FB gains same as HYBRID BMI to start. 
    fb_gain = np.array([1, 1, 1, 1.5, 2, 2.5, 1.5])
    def init(self):
        # Start by making the RH variables we're controlling == 1 -- > then compute the correct ratio needed
        # to scale to ~ BMI size. 
        self.grasp_emg_classifier_extractor = self.grasp_emg_classifier.extractor_cls(None, 
            emg_channels = self.grasp_emg_classifier.extractor_kwargs['emg_channels'], 
            feature_names = self.grasp_emg_classifier.extractor_kwargs['feature_names'], 
            win_len = self.grasp_emg_classifier.extractor_kwargs['win_len'], 
            fs = self.grasp_emg_classifier.extractor_kwargs['fs'])
        
        self.chanix = self.grasp_emg_classifier.extractor_kwargs['subset_muscles_ix']
        self.b = self.grasp_emg_classifier.b
        self.m = self.grasp_emg_classifier.m
        self.grasp_emg_classifier_extractor.source = self.brainamp_source
        self.add_dtype('grasp_emg_classifier_features','f8', (self.grasp_emg_classifier.n_features,)) 
        self.add_dtype('grasp_emg_classifier_features_Z','f8', (self.grasp_emg_classifier.n_features,)) 
        self.add_dtype('grasp_emg_classifier_prob_output','f8', 1) 
        self.grasp_emg_control_dofs = ['rh_vthumb', 'rh_vindex', 'rh_vfing3']
        self.add_dtype('grasp_emg_output',  float, (len(self.grasp_emg_control_dofs),))
        self.add_dtype('grasp_emg_classifier_features_mean','f8', (self.grasp_emg_classifier.n_features,)) 
        self.add_dtype('grasp_emg_classifier_features_std','f8', (self.grasp_emg_classifier.n_features,)) 

        # for calculating/updating mean and std of EMG features online
        self.grasp_classifier_features_buffer = RingBuffer(
            item_len=self.grasp_emg_classifier_extractor.n_features,
            capacity=60*self.fps,  # 60 secs
        )
        self.vel_output_grasp_emg_classifier = np.zeros(len(self.grasp_emg_control_dofs))
        super(Hybrid_EMGHandClassification_BMIControl_w_BinEMG_Back2Start_Prep, self).init()
        
        # overwrite scale_emg predictions
        andrea2 = np.array([6.14073, 3.21564, 3.13008,  2.06547,  1.32943,  1.3678, 14.17686])
        self.plant.scale_emg_pred_arr = andrea2.copy()

    def _cycle(self):

        super(Hybrid_EMGHandClassification_BMIControl_w_BinEMG_Back2Start_Prep, self)._cycle()

        if self.grasp_emg_classifier_extractor is not None:

            # Get features: 
            grasp_emg_classifier_features = self.grasp_emg_classifier_extractor() 
            
            # Add features to the buffer
            self.grasp_classifier_features_buffer.add(grasp_emg_classifier_features[self.grasp_emg_classifier.extractor_cls.feature_type])

            # If there's mroe than 1 second of data in the buffer: 
            if self.grasp_classifier_features_buffer.num_items() > 60 * self.fps: # changed to 60 seconds, 14/05/2018
                # if we have more than 1 second of recent EMG data, then
                #   calculate mean and std from this data
                grasp_emg_classifier_recent_features = self.grasp_classifier_features_buffer.get_all()
                #print 'recent_features', recent_features.shape
                self.grasp_emg_classifier_features_mean = np.mean(grasp_emg_classifier_recent_features[self.chanix,:] , axis=1)

                # If we're using a fixed scalar as the standard deviation: 
                use_fixed = False
                if hasattr(self.grasp_emg_classifier, 'scalar_fixed_var'):
                    if self.grasp_emg_classifier.scalar_fixed_var == True:
                        use_fixed = True
                        self.grasp_emg_classifier_features_std = self.grasp_emg_classifier.recent_features_std_train
                
                if not use_fixed:
                    self.grasp_emg_classifier_features_std  = np.std(grasp_emg_classifier_recent_features[self.chanix,:] , axis=1)
            
            else:
                # Try to use recently trained data: 
                try:
                    self.grasp_emg_classifier_features_mean = self.grasp_emg_classifier.recent_features_mean_train
                    self.grasp_emg_classifier_features_std  = self.grasp_emg_classifier.recent_features_std_train

                except:
                    # else use mean and std from the EMG data that was used to train the classifier
                    self.grasp_emg_classifier_features_mean = self.grasp_emg_classifier.features_mean_train
                    self.grasp_emg_classifier_features_std  = self.grasp_emg_classifier.features_std_train

            try:
                self.grasp_emg_classifier_features_std[self.grasp_emg_classifier_features_std == 0] = 1
            except:
                pass
            # z-score the EMG features
            grasp_emg_classifier_features_Z = (grasp_emg_classifier_features[self.grasp_emg_classifier.extractor_cls.feature_type][self.chanix] - self.grasp_emg_classifier_features_mean) / self.grasp_emg_classifier_features_std 
        
            # compute prob for grasp/close -- it return the prob being 1 -- close, 0 -- open (to map velocity sign)
            self.grasp_emg_classifier_prob_output = self.grasp_emg_classifier(grasp_emg_classifier_features_Z,'prob')


            for ind_grasp_control in range(len(self.grasp_emg_control_dofs)):
                self.vel_output_grasp_emg_classifier[ind_grasp_control] = self.m[ind_grasp_control] * self.grasp_emg_classifier_prob_output + self.b[ind_grasp_control]
        
            self.task_data['grasp_emg_classifier_features'] = grasp_emg_classifier_features[self.grasp_emg_classifier.extractor_cls.feature_type][self.chanix]
            self.task_data['grasp_emg_classifier_features_Z'] = grasp_emg_classifier_features_Z 
            self.task_data['grasp_emg_classifier_prob_output'] = self.grasp_emg_classifier_prob_output
            self.task_data['grasp_emg_output'] = self.vel_output_grasp_emg_classifier.copy()
            self.task_data['grasp_emg_classifier_features_mean'] = self.grasp_emg_classifier_features_mean
            self.task_data['grasp_emg_classifier_features_std'] = self.grasp_emg_classifier_features_std
            self.plant.emg_vel[self.grasp_emg_control_dofs] = self.vel_output_grasp_emg_classifier.copy()

    def cleanup(self, database, saveid, **kwargs):
        super(Hybrid_EMGHandClassification_BMIControl_w_BinEMG_Back2Start_Prep, self).cleanup(database, saveid, **kwargs)
        self.grasp_emg_classifier.recent_features_mean_train = self.grasp_emg_classifier_features_mean 
        self.grasp_emg_classifier.recent_features_std_train = self.grasp_emg_classifier_features_std

        # Save this
        import pickle
        pickle.dump(self.grasp_emg_classifier, open(self.grasp_emg_classifier.path, 'wb'))

        print ' saving EMG classifier zscore features'

class Hybrid_GraspClass_w_RestEMG_PhaseV(Hybrid_BMIControl_w_Binary_EMG):
    
    plant_type = 'IsMorePlantHybridBMISoftSafety'
    inter_block_rest_time = traits.Float(5.0, desc='rest time between blocks')
    trials_per_block = traits.Int(5, desc='The number of target ATTEMPTS per block')
    back_to_target_speed = traits.Float(0.05, desc='speed to use to go back to target')
    preparation_time = traits.Float(2.0, desc='prep time')
    thumb_only_assist = traits.Float(0.0, desc='thumb assist value')
    fb_gain = np.array([1, 1, 1, 1., 1., 1., 1.])
    
    status = {

        'wait': {
            'start_trial':         'rest'},

        'rest': {
            'start_block':          'drive_to_start',
            'stop':                 None},

        'drive_to_start': {
            'enter_start_target':   'instruct_trial_type',
            'stop':                 None},

        'instruct_trial_type': {
            'end_instruct':        'prep' },

        'prep': {
            'end_prep':             'target'},

        'target': {
            'enter_target':        'hold',
            'timeout':             'timeout_penalty',
            'stop':                None},
        
        'hold': {
            'hold_complete':       'test_block_end'},
        
        'test_block_end': {
            'block_complete':       'drive_to_rest',
            'block_incomplete':     'drive_to_start'},

        'drive_to_rest': {
            'enter_rest':   'wait'},
        
        'timeout_penalty': {
            'timeout_penalty_end': 'test_block_end'},
            }

    def __init__(self, *args, **kwargs):
        super(Hybrid_GraspClass_w_RestEMG_PhaseV, self).__init__(*args, **kwargs)
        fs_synch = self.fps #Frequency at which the task is running
        nyq   = 0.5 * fs_synch
        cuttoff_freq  = 2.5 / nyq
        bpf_kin_coeffs = butter(4, cuttoff_freq, btype='low')

        self.command_lpfs = dict()
        for state in self.vel_states:
            self.command_lpfs[state] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1]) # low-pass filter to smooth out command velocities
        self.plant.command_lpfs = self.command_lpfs

    def init(self):
        self.plant.back_to_target_speed = self.back_to_target_speed
        if self.thumb_only_assist > 0:
            self.decoder.filt.thumb_assist = tuple((True, self.thumb_only_assist))
        else:
            self.decoder.filt.thumb_assist = tuple((False, np.nan))

        if self.decoder_drift_hl > 0:
            drift_rho = np.exp(np.log(0.5) / (self.decoder_drift_hl/ self.decoder.binlen))
        else:
            drift_rho = 1.

        # Send drift correctino to the decoder
        self.decoder.filt.drift_rho = drift_rho

        super(Hybrid_GraspClass_w_RestEMG_PhaseV, self).init()

    def get_current_state(self):
        current_state = np.hstack([self.plant_pos.values, self.plant_vel.values, 1])[:, None]
        if np.any(np.isnan(current_state)):
            current_state[np.isnan(current_state)] = 0
        return current_state

    def _test_start_block(self, ts):
        start_block = ts >= self.inter_block_rest_time
        if start_block:
            # What to do to start a block: 
            self.ntrials_completed = 0
        return start_block

    def _test_end_prep(self, ts):
        # if self.go_sound_counter >= self.go_sound_chain:
        return ts >= self.preparation_time

    def _start_drive_to_rest(self):
        self.plant.drive_to_start_target = self.targets[1]

    def _start_drive_to_start(self):
        # Block just started -- drive to rest target
        if self.target_index in [-1]:
            self.plant.drive_to_start_target = self.targets[1] # "vuelta pos"

        # Just got/attempted open position: 
        elif self.target_index in [0]:
            self.plant.drive_to_start_target = self.targets[0] # "open position"
        
        # Just got/attempted close position
        elif self.target_index in [1]:
            self.plant.drive_to_start_target = self.targets[1] # vuelta position

    def _test_enter_start_target(self, ts):
        if self.plant_type == 'ArmAssist':
            return self.armassist_inside_target(target=self.plant.drive_to_start_target)
        elif self.plant_type == 'ReHand':
            return self.rehand_inside_target(target=self.plant.drive_to_start_target)
        elif 'IsMore' in self.plant_type:
            return self.armassist_inside_target(target=self.plant.drive_to_start_target) and self.rehand_inside_target(target=self.plant.drive_to_start_target)

    def _start_timeout_penalty(self):
        self._play_sound(self.sounds_general_dir, ['timeout'])
        self.last_trial_fail = True
        super(Hybrid_GraspClass_w_RestEMG_PhaseV, self)._start_timeout_penalty()

    def _start_hold(self):
        self.last_trial_fail = False
        super(Hybrid_GraspClass_w_RestEMG_PhaseV, self)._start_hold()

    def _start_test_block_end(self):
        self.ntrials_completed += 1
        # Last trial was a success -- if target index is 1, then 
        # it'll get incremented to a 2 on the next instruct_trial_index
        # which will give an error.

        # we know the 2nd part of the trial has been completed successfully
        # So now want to do the first part again. 

        if self.target_index == 1:
            self.target_index = -1
        
    def _test_block_complete(self, ts):
        return self.ntrials_completed >= self.trials_per_block

    def _test_block_incomplete(self, ts):
        return self.ntrials_completed < self.trials_per_block

    def _test_enter_rest(self, ts):
        # Assume ismore plant. 
        #return self.armassist_inside_target(target=self.plant.safety_grid.attractor_point) and self.rehand_inside_target(target=self.plant.safety_grid.attractor_point)
        return self.armassist_inside_target(target=self.plant.drive_to_start_target) and self.rehand_inside_target(target=self.plant.drive_to_start_target)

    def _cycle(self):
        self.decoder.filt.task_state = self.state
        super(Hybrid_GraspClass_w_RestEMG_PhaseV, self)._cycle()

class CLDAControl(LinearlyDecreasingHalfLife, BMIControl):
    '''
    BMI task that periodically refits the decoder parameters based on intended
    movements toward the targets.
    '''

    batch_time  = traits.Float(0.1, desc='The length of the batch in seconds (RML: 0.1, SmoothBatch: ')
    clda_update_method = traits.OptionsList(*clda_update_methods, bmi3d_input_options=clda_update_methods)
    clda_intention_est_method = traits.OptionsList(*clda_intention_est_methods, bmi3d_input_options=clda_intention_est_methods)
    clda_adapting_ssm_name = traits.OptionsList(*clda_adapting_states_opts, bmi3d_input_options=clda_adapting_states_opts)
    clda_adapt_mFR_stats = traits.OptionsList(*clda_adapt_mFR_stats, bmi3d_input_options=clda_adapt_mFR_stats)
    clda_stable_neurons = traits.String('', desc='Units to keep stable in current decoder')
    ordered_traits = ['session_length', 'assist_level', 'assist_level_time', 'half_life', 'half_life_decay_time']
    
    def __init__(self, *args, **kwargs):
        super(CLDAControl, self).__init__(*args, **kwargs)
        self.learn_flag = True

    def init_decoder_state(self):
        #Add adapting inds option to decoder to pass to clda intialization
        cls_ = ismore_bmi_lib.SSM_CLS_DICT[self.clda_adapting_ssm_name]

        cellname = re.compile(r'(\d{1,3})\s*(\w{1})')
        cells = [ (int(c), ord(u) - 96) for c, u in cellname.findall(self.clda_stable_neurons)]
        stable_ids = []
        for i, (c, u) in enumerate(cells):
            ix = np.nonzero(self.decoder.units[:,0]==c)[0]
            ixx = np.nonzero(self.decoder.units[ix, 1]==u)[0]
            if len([ixx]) > 0:
                stable_ids.append(ix[int(ixx)])

        adapting_neur_inds = [i for i in range(len(self.decoder.units)) if i not in stable_ids]
        self.decoder.adapting_neur_inds = adapting_neur_inds
        self.decoder.adapting_state_inds = cls_()
        if self.clda_adapt_mFR_stats == 'Yes':
            self.decoder.adapt_mFR_stats = True
        else:
            self.decoder.adapt_mFR_stats = False
        super(CLDAControl, self).init_decoder_state()

    def init(self):
        '''
        Secondary init function. Decoder has already been created by inclusion
        of the 'bmi' feature in the task. 
        '''
        # self.load_decoder() will be called again in the call to super(...).init() (in BMILoop.init)
        # but there is no way around this
        self.load_decoder()
        self.batch_size = int(self.batch_time / self.decoder.binlen)
        super(CLDAControl, self).init()

    def create_learner(self):
        if self.clda_intention_est_method == 'OFC':
            self.learner = ismore_bmi_lib.OFC_LEARNER_CLS_DICT[self.plant_type](self.batch_size)
        elif self.clda_intention_est_method == 'simple':
            # the simple, non-OFC learners just create/use an assister object
            assister_kwargs = {
                'call_rate': self.fps,
                'xy_cutoff': np.min([self.target_radius_x, self.target_radius_y]),
            }
            self.learner = ismore_bmi_lib.LEARNER_CLS_DICT[self.plant_type](self.batch_size, **assister_kwargs)
        elif self.clda_intention_est_method == 'OFC_w_rest':
            self.learner = ismore_bmi_lib.OFC_LEARNER_CLS_DICT_w_REST[self.plant_type](self.batch_size)
        else:
            NotImplementedError("Unrecognized CLDA intention estimation method: %s" % self.clda_intention_est_method)
        
        self.learn_flag = True

    def create_updater(self):
        if self.clda_update_method == 'RML':
            self.updater = clda.KFRML(self.batch_time, self.half_life[0], adapt_C_xpose_Q_inv_C=True)
        elif self.clda_update_method == 'Smoothbatch':
            half_life_start, half_life_end = self.half_life
            self.updater = clda.KFSmoothbatch(self.batch_time, half_life_start)
        elif self.clda_update_method == 'Baseline':
            self.updater = clda.KFRML_baseline(self.batch_time, self.half_life[0], adapt_C_xpose_Q_inv_C=True)
            self.clda_adapt_mFR_stats = 'Yes'
        else:
            raise NotImplementedError("Unrecognized CLDA update method: %s" % self.clda_update_method)

class Hybrid_CLDAControl_w_Binary_EMG(Hybrid_BMIControl_w_Binary_EMG, CLDAControl):
    plant_type = 'IsMoreHybridControl'
    decoder_drift_hl        = 0
    fb_gain = np.array([1, 1, 1, 1, 1, 1, 1])

    def __init__(self, *args, **kwargs):
        super(Hybrid_CLDAControl_w_Binary_EMG, self).__init__(*args, **kwargs)

class Hybrid_EMGHandClass_CLDAControl_w_BinEMG(Hybrid_EMGHandClassification_BMIControl_w_BinEMG_Back2Start_Prep,CLDAControl):
    
    rest_back_time = 0.

    status = {
        'wait': {
            'start_trial':         'rest_back',
            'stop':                None}, 

        'rest_back': {
            'rest_back_complete':   'rest',
            'stop':                 None},

        'rest': {
            'rest_complete':    'emg_rest'},

        'emg_rest': {
            'emg_rest_complete':    'instruct_trial_type'},

        'instruct_trial_type': {
            'end_instruct':        'target' },

        'target': {
            'enter_target':        'hold',
            'timeout':             'timeout_penalty',
            'stop':                None},
        
        'hold': {
            'leave_early':         'hold_penalty',
            'hold_complete':       'targ_transition'},
        
        'targ_transition': {
            'trial_complete':      'reward',
            'trial_incomplete':    'instruct_trial_type'},
        
        'timeout_penalty': {
            'timeout_try_again': 'instruct_trial_type',
            'timeout_skip_target': 'instruct_trial_type',
            'timeout_end_of_trial': 'wait'},
        
        'hold_penalty': {
            'hold_penalty_try_again':    'instruct_trial_type',
            'hold_penalty_skip_target':  'instruct_trial_type',
            'hold_penalty_end_of_trial':  'wait'},
        
        'reward': {
            'reward_end': 'wait'}
    }
    fb_gain = np.array([1, 1, 1, 1.5, 2, 2.5, 1.5])

    def __init__(self, *args, **kwargs):
        super(Hybrid_EMGHandClass_CLDAControl_w_BinEMG, self).__init__(*args, **kwargs)

class ReplayBMI_wo_Audio(PlantControlBase):
    te_id_to_replay = traits.Float(2., desc='task entry ID to replay velocities from')
    status = dict(wait=dict(trial_start_replay='drive_to_trial_start', stop=None),
                  drive_to_trial_start=dict(at_pos='trial', stop=None),
                  trial=dict(end_trial='wait',stop=None))
    start_pos_replay = np.zeros((7, ))
    drive_to_trial_vel = np.zeros((7, ))

    def __init__(self, *args, **kwargs):
        super(ReplayBMI_wo_Audio, self).__init__(*args, **kwargs)

    def init(self, *args, **kwargs):
        self.load_commands()

        # Load safety grid
        self.load_safety()
        self.replay_state = 'wait'
        self.goal_calculator = ZeroVelocityGoal_ismore(ismore_bmi_lib.SSM_CLS_DICT[self.plant_type], 
            pause_states = ['rest', 'wait', 'instruct_rest', 'instruct_trial_type'])
        
        assister_kwargs = {
            'call_rate': self.fps,
            'xy_cutoff': 2,
            'speed':    'high',
                }
        self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT_OG[self.plant_type](**assister_kwargs)
        self.add_dtype('command_count',  'f8', (1,))
        self.add_dtype('target_pos_drive_to',  'f8', (7,))

        super(ReplayBMI_wo_Audio, self).init(*args, **kwargs)

    def load_commands(self):
        import glob
        fnm = glob.glob('/storage/rawdata/hdf/*te'+str(int(self.te_id_to_replay))+'.hdf') 
        if len(fnm) == 1:
            hdf = tables.openFile(fnm[0])
        else:
            print "HDF: ", fnm, len(fnm), '/storage/rawdata/hdf/*te'+str(int(self.te_id_to_replay))+'.hdf'
        self.commands_to_send = hdf.root.task[:]['command_vel_sent']

        # Make sure blocked joints don't move! 
        if len(self.blocking_joints) >0:
            for joint in self.blocking_joints:
                self.commands_to_send[:, joint] = 0

        self.plant_pos_from_hdf = hdf.root.task[:]['plant_pos']

        self.command_count = 0
        self.tsk_msgs = hdf.root.task_msgs[:]
        ix = np.nonzero(hdf.root.task_msgs[:]['msg']=='target')[0]
        self.targ_tsk_msgs = hdf.root.task_msgs[ix]['time']

        self.replay_trial_type = hdf.root.task[:]['trial_type']
        self.maxN = len(hdf.root.task)
        print 'maxN: ', self.maxN

    def move_plant(self):
        if self.state == 'trial':
            self.plant.send_vel(self.commands_to_send[self.command_count, :])
            self.plant.drive_velocity_sent = self.commands_to_send[self.command_count,:]
        elif self.state == 'drive_to_trial_start':
            self.plant.drive_velocity_sent = self.drive_to_trial_vel


        self.command_count += 1
        if self.command_count >= self.maxN:
            self.end_task()

    def _cycle(self):
        super(ReplayBMI_wo_Audio, self)._cycle()
        self.move_plant()
        self.task_data['command_count'] = self.command_count
        self.task_data['target_pos_drive_to'] = self.start_pos_replay

        if self.command_count in self.tsk_msgs[:]['time']:
            tmp = np.nonzero(self.tsk_msgs[:]['time']==self.command_count)[0]
            self.replay_state = self.tsk_msgs[tmp[0]]['msg']
            #print  ' new state: ', self.replay_state, self.replay_state == 'rest_back'

    def _test_trial_start_replay(self, ts):
        return self.replay_state == 'rest_back'

    def _start_drive_to_trial_start(self):
        # Deprecated -- doesn't really make sense: 
        # Find exo position at next 'target' if it exists:
        # next_target_n = self.targ_tsk_msgs[self.targ_tsk_msgs > self.command_count]
        # if len(next_target_n) > 0:
        #     self.start_pos_replay = self.plant_pos_from_hdf[next_target_n[0]]
        # else:
        #     self.start_pos_replay = self.plant_pos.values

        # Updated April 11, 2018: Drive to the neutral position
        # Drive to the safety grid: 
        self.start_pos_replay = self.plant.safety_grid.attractor_point

    def _while_drive_to_trial_start(self):
        current_state = np.hstack([self.plant_pos.values, self.plant_vel.values, 1])[:, None]
        data, solution_updated = self.goal_calculator(self.start_pos_replay, 'target', 
            **dict(current_state=current_state))
        
        #data, solution_updated = self.goal_calculator(self.target_pos.values)
        target_state = data[0].reshape(-1, 1)
        assist_output = self.assister(current_state, target_state, 1.)
        Bu = pd.Series(np.array(assist_output["x_assist"]).ravel(), self.ssm_states)
        command_vel_raw = Bu[self.vel_states]
        if np.any(np.isnan(command_vel_raw)):
            print 'setting command_vel_raw nans equal to zero'
            command_vel_raw[np.isnan(command_vel_raw)] = 0       
        self.plant.send_vel(command_vel_raw) 
        self.drive_to_trial_vel = command_vel_raw

    def _test_at_pos(self, ts):
        self.target_pos = pd.Series(np.hstack(( self.start_pos_replay, np.zeros((7, )) )),
            self.pos_states + self.vel_states)
        return self.armassist_inside_target() and self.rehand_inside_target()

    def _test_end_trial(self, ts):
        return self.replay_state == 'wait'

class ReplayBMI_w_Audio(ReplayBMI_wo_Audio):
    te_id_to_replay = traits.Float(2., desc='task entry ID to replay velocities from')

    status = dict(wait=dict(trial_start_replay='drive_to_trial_start', stop=None),
                  drive_to_trial_start=dict(at_pos='trial', audio = 'instruct_trial_type_replay', stop=None),
                  trial=dict(end_trial='wait', audio= 'instruct_trial_type_replay', stop=None),
                  instruct_trial_type_replay=dict(end_instruct_replay='trial', end_instruct_replay2='drive_to_trial_start'))

    replay_tt = ''
    def _start_wait(self):
        self.replay_tt = ''
        super(ReplayBMI_w_Audio, self)._start_wait()

    def _test_audio(self, ts):
        speak = False
        if self.command_count in self.tsk_msgs[:]['time']:
            self.thing_to_say = []
            ix = np.nonzero(self.tsk_msgs[:]['time']==self.command_count)[0]
            for i in ix:
                if self.tsk_msgs[i]['msg'] in ['instruct_trial_type', 'rest', 'target', 'reward']:
                    speak = True
                    self.last_state = self.state
                    self.thing_to_say.append(self.tsk_msgs[i]['msg'])
        return speak

    def _start_instruct_trial_type_replay(self):
        self.soundlist = []
        for message in self.thing_to_say:
            if message == 'rest':
                self.soundlist.append(self.get_sounds(self.sounds_dir, ['rest']))
            elif message == 'target':
                self.soundlist.append(self.get_sounds(self.sounds_dir, ['go']))
            elif message == 'reward':
                self.soundlist.append(self.get_sounds(self.sounds_dir, ['beep1']))
            else:
                if self.replay_trial_type[self.command_count] == self.replay_tt:
                    self.replay_target_index += 1
                else:
                    self.replay_target_index = 0
                    self.replay_tt = self.replay_trial_type[self.command_count]
                
                targ = self.targets_matrix['subgoal_names'][self.replay_trial_type[self.command_count]][self.replay_target_index]
                self.soundlist.append(self.get_sounds(self.sounds_dir, targ))
            print self.soundlist
        self.soundlist = np.hstack(( self.soundlist))
        self.sound_counter = 0 # Increments after playing sound
        self.sound_chain = len(self.soundlist)            

    def _while_instruct_trial_type_replay(self):
        if pygame.mixer.music.get_busy():
            pass
        else:
            sound = self.soundlist[self.sound_counter]
            pygame.mixer.music.load(sound)
            pygame.mixer.music.play()
            self.sound_counter += 1

    def _test_end_instruct_replay(self, *args, **kwargs):
        if self.last_state == 'trial':
            return self.sound_counter >= self.sound_chain
        else:
            return False

    def _test_end_instruct_replay2(self, *args, **kwargs):
        if self.last_state == 'drive_to_trial_start':
            return self.sound_counter >= self.sound_chain
        else:
            return False        


########################
## simulation classes ##
########################
from riglib.bmi.sim_neurons import KalmanEncoder
from features.simulation_features import SimKalmanEnc, SimKFDecoderSup, SimCosineTunedEnc, SimKalmanEnc, SimHDF
from riglib.bmi.feedback_controllers import LQRController

class SimBMIControl(SimKalmanEnc, SimKFDecoderSup, BMIControl):
    sequence_generators = ['B1_targets'] 
    safety_grid_file = '/storage/rawdata/safety/phaseIII_safetygrid_same_minprono_updated.pkl'
    language = 'castellano'

    def __init__(self, *args, **kwargs):
        super(SimBMIControl, self).__init__(*args, **kwargs)
        self.targets_matrix = pickle.load(open('/storage/target_matrices/targets_HUD1_7727_7865_8164_None_HUD1_20171122_1502_fixed_thumb_point_all_targs_blue_mod_fix_cha_cha_cha_fix_B3_fix_rest.pkl'))
        ssm_class = ismore_bmi_lib.SSM_CLS_DICT[self.plant_type]
        self.ssm = ssm = ssm_class()

        if self.plant_type == "ArmAssist":
            self.fb_ctrl = ismore_bmi_lib.arm_assist_controller_bmi_sims
        elif self.plant_type == "ReHand":
            self.fb_ctrl = ismore_bmi_lib.rehand_controller2
        elif self.plant_type == "IsMore":
            self.fb_ctrl = ismore_bmi_lib.ismore_controller

        if 'Q' in kwargs:
            self.set_assisterQ = kwargs['Q']
            self.set_assisterR = kwargs['R']
            self.update_assister = 0
        else:
            self.update_assister = 1

        self.assist_level_time =  kwargs.get('assist_level_time', 30.)
        self.assist_level = kwargs.get('assist_level', (1.,1.))
        self.dec_path =  kwargs.get('decoder_path', None)
        if self.dec_path is not None:
            self.decoder = pickle.load(open(self.dec_path))
        self.enc_path = kwargs.get('enc_path', None)
        self.session_length = kwargs.get('session_length',0.)
        self.timeout_time = kwargs.get('timeout_time', 15.)
        #self.safety_grid_file = kwargs.get('safety_grid_name', None)

    def _cycle(self):
        self.task_data['ctrl_input'] = np.squeeze(np.array(self.extractor.sim_ctrl))
        if not self.update_assister:
            self.assister.fb_ctrl.Q = self.set_assisterQ
            self.assister.fb_ctrl.R = self.set_assisterR
            self.assister.fb_ctrl.F = self.assister.fb_ctrl.dlqr(self.assister.fb_ctrl.A, 
                self.assister.fb_ctrl.B, self.assister.fb_ctrl.Q, 
                self.assister.fb_ctrl.R)
            self.update_assister = 1
        super(SimBMIControl, self)._cycle()

    def _init_neural_encoder(self):
        ## Simulation neural encoder
        load_new_enc = True
        if hasattr(self, 'decoder'):
            n_features = self.decoder.n_features
            if hasattr(self.decoder, 'corresp_encoder'):
                print 'loading old encoder from decoder'
                self.encoder = self.decoder.corresp_encoder
                load_new_enc = False
        else:
            n_features = 50
        
        if load_new_enc:
            print 'loading new encoder: ', n_features, ' units'
            self.encoder = KalmanEncoder(self.ssm, n_features, int_neural_features=False, scale_noise=0.1)

class SimBMIControlReplayFile(BMIControl_1D_along_traj):
    sequence_generators = ['sleep_gen'] 
    safety_grid_file = '/storage/rawdata/safety/phaseIII_safetygrid_same_minprono_updated_more_fing_ext_-0.55_real.pkl'
    language = 'castellano'
    try:
        targets_matrix = pickle.load(open('/storage/target_matrices/phaseIII_TM_w_sleep_targ_GGU.pkl'))
    except:
        pass
    ssm_class = ismore_bmi_lib.SSM_CLS_DICT['IsMore']
    ssm = ssm_class()
    #fb_ctrl = ismore_bmi_lib.ismore_controller
    aa_assist_level = tuple((1. ,1.))
    rh_assist_level = tuple((1. ,1.))
    plant_type = 'IsMore'
    blocking_opts = 'No'
    target_radius_x = 2
    target_radius_y = 4
    tol_deg_fing = 5
    tol_deg_pron = 5
    tol_deg_psi = 15
    tol_deg_thumb = 8
    hold_time = 0.01
    neurondata = None
    attractor_speed = 0.01
    attractor_speed_const = 0
    
    def __init__(self, *args, **kwargs):
        super(SimBMIControlReplayFile, self).__init__(*args, **kwargs)
        self.cnt = 0
        self.decoder = kwargs['decoder']
        self.decoder.filt.init_from_task()
        self.replay_neural_features = kwargs['replay_neural_features']

        if 'targets_matrix' in kwargs and kwargs['targets_matrix'] is not None:
            print 'kwargs[targetmatrix] = '
            self.targets_matrix = kwargs['targets_matrix']
        self.maxN = self.replay_neural_features.shape[0]
        self.session_length = kwargs.pop('session_length', 0)
        print 'session length: ', self.session_length

    def move_plant(self, **kwargs):

        # get features: 
        features = self.replay_neural_features[self.cnt]

        if self.cnt + 5 > self.maxN:
            self.end_task()
        
        # Get assist arguments 
        current_state = np.hstack([self.plant_pos.values, self.plant_vel.values, 1])[:, None]
        data, solution_updated = self.goal_calculator(self.target_pos.values, self.state, 
            **dict(current_state=current_state))
        
        target_state = data[0].reshape(-1, 1)
        assist_output = self.assister(current_state, target_state, 1.)

        # call decoder: 
        kwargs.update(assist_output)
    
        #if decode:
        tmp = self.call_decoder(features[:, np.newaxis], target_state, **kwargs)

        self.task_data['e1_minus_e2'] = self.decoder.filt.FR
        self.task_data['spike_counts'] = features[:, np.newaxis]
        self.task_data['decoder_state'] = tmp

        # get alpha value: 
        try:
            assist = assist_output['x_assist']
            Bu = pd.Series(np.array(assist_output["x_assist"]).ravel(), self.ssm_states)
        except:
            Bu = pd.Series(np.squeeze(np.array(assist_output["Bu"])), self.ssm_states)

        command_vel_raw = Bu[self.vel_states]
        command_vel_raw[np.isnan(command_vel_raw)]=0
        if self.ignore_decoder:
            self.decoder.qdot = np.squeeze(np.array(command_vel_raw))
        else:
            self.decoder.qdot = np.squeeze(np.array(command_vel_raw*tmp[0,0]))
        self.plant.drive(self.decoder)

class SimCLDAControl(SimBMIControl, CLDAControl):
    
    rand_start = (0., 0.)
    def __init__(self, *args, **kwargs):
        super(SimCLDAControl, self).__init__(*args, **kwargs)
        
        self.clda_update_method = kwargs.pop('clda_update_method', 'RML')

        if self.clda_update_method == 'RML':
            self.batch_time = 0.1
        else:
            self.batch_time = kwargs.get('batch_time', 10.)

        self.half_life  = kwargs.get('half_life', (20.0, 20.0))
        self.half_life_time = kwargs.get('half_life_time', 600.)
        self.adapt_mFR_stats = kwargs.get('adapt_mFR_stats', False)

        if 'clda_adapting_ssm' in kwargs:
            self.clda_adapting_ssm_name = kwargs['clda_adapting_ssm']
            print self.clda_adapting_ssm_name, 'clda clda_adapting_ssm'
        else:
            print 'no clda_adapting_ssm'

        if 'clda_stable_neurons' in kwargs:
            self.clda_stable_neurons = kwargs['clda_stable_neurons']
            print self.clda_stable_neurons,'clda clda_stable_neurons'

        if 'clda_adapt_mFR' in kwargs:
            self.clda_adapt_mFR_stats = kwargs['clda_adapt_mFR']
            print self.clda_adapt_mFR_stats, 'clda adapt mFR'

    def _cycle(self):
        super(SimCLDAControl, self)._cycle()
        elapsed_time = self.get_time() - self.task_start_time
        if elapsed_time > self.assist_level_time and self.learn_flag:
            self.disable_clda()
            print 'assist OFF:'
            print '*' * 80
