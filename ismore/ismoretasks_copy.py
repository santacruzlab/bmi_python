'''Tasks specific to the IsMore project.'''

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

from riglib.experiment import traits, Sequence, generate, FSMTable, StateTransitions
from riglib.stereo_opengl.window import WindowDispl2D
from riglib.stereo_opengl.primitives import Circle, Sector, Line
from riglib.bmi import clda, extractor, train
from riglib.bmi.bmi import Decoder, BMISystem, GaussianStateHMM, BMILoop, GaussianState, MachineOnlyFilter

from ismore import plants, settings
from ismore.common_state_lists import *
from features.bmi_task_features import LinearlyDecreasingAssist
from features.simulation_features import SimTime #, SimHDF
import ismore_bmi_lib
from utils.angle_utils import *
from utils.util_fns import *
from utils.constants import *

from utils.ringbuffer import RingBuffer

from features.generator_features import Autostart

import pygame

from riglib.plants import RefTrajectories
from riglib.filter import Filter


np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)


#######################################################################
COLORS = {
    'black': (0, 0, 0, 1),
    'red':   (1, 0, 0, 1),
    'green': (0, 1, 0, 1),
    'blue':  (0, 0, 1, 1),
    'white': (1, 1, 1, 1),
}

#######################################################################

plant_type_options = ['ArmAssist', 'ReHand', 'IsMore']
clda_update_methods = ['Smoothbatch', 'RML']
class IsMoreBase(WindowDispl2D):
    '''
    A base class for all IsMore tasks. Creates the appropriate plant object
    and updates the display of the plant at every iteration of the task.
    '''

    # settable parameters on web interface
    session_time = traits.Float(0, desc='Time until task stops (0 means no auto stop).')
    window_size = traits.Tuple((1000, 560), desc='Size of window to display the plant position/angle')

    starting_pos = settings.starting_pos

    update_rest = True
    plant_type_options = plant_type_options
    plant_type = traits.Enum(*plant_type_options)

    simulate = traits.Bool(False, desc='Use simulation "plant" without UDP communication')
    
    def __init__(self, *args, **kwargs):
        super(IsMoreBase, self).__init__(*args, **kwargs)

        self.ssm = ismore_bmi_lib.SSM_CLS_DICT[self.plant_type]()
        self.ssm_states = [s.name for s in self.ssm.states]
        self.pos_states = [s.name for s in self.ssm.states if s.order == 0]
        self.vel_states = [s.name for s in self.ssm.states if s.order == 1]

        if 0: #self.simulate:
            # use locally running simulated ArmAssist and/or ReHand
            #   for which we can magically set the initial position
            self.plant = plants.NONUDP_PLANT_CLS_DICT[self.plant_type]()
            self.plant.set_pos(self.starting_pos[self.pos_states].values)
        else:
            self.plant = plants.UDP_PLANT_CLS_DICT[self.plant_type]()

        self.plant_pos = pd.Series(self.plant.get_pos(), self.pos_states)
        self.plant_vel = pd.Series(self.plant.get_vel(), self.vel_states)

        self.add_dtype('plant_pos', 'f8', (len(self.plant_pos),))
        self.add_dtype('plant_vel', 'f8', (len(self.plant_vel),))
        
        self.init_plant_display()
        self.update_plant_display()

    def _set_workspace_size(self):
        MAT_SIZE = settings.MAT_SIZE

        border = 10.  # TODO -- difference between this and self.display_border?
        self.workspace_bottom_left = np.array([ 0. - border, 
                                                    0. - border])
        self.workspace_top_right   = np.array([MAT_SIZE[0] + border, 
                                                   MAT_SIZE[1] + border])

    def init(self):
        self.plant.init()
        super(IsMoreBase, self).init()

        if settings.WATCHDOG_ENABLED:
            self.plant.watchdog_enable(settings.WATCHDOG_TIMEOUT)

    def run(self):
        self.plant.start()
        try:
            super(IsMoreBase, self).run()
        finally:
            self.plant.stop()

    def _cycle(self):
        if settings.VERIFY_PLANT_DATA_ARRIVAL:
            self.verify_plant_data_arrival(settings.VERIFY_PLANT_DATA_ARRIVAL_TIME)
        
        super(IsMoreBase, self)._cycle()
        # Note: All classes that inherit from this class should probably call
        # the following code at some point during their _cycle methods
            # self.plant_pos[:] = self.plant.get_pos()
            # self.plant_vel[:] = self.plant.get_vel()
            # self.update_plant_display()
            # self.task_data['plant_pos'] = self.plant_pos.values
            # self.task_data['plant_vel'] = self.plant_vel.values

    def verify_plant_data_arrival(self, n_secs):
        time_since_started = time.time() - self.plant.ts_start_data
        last_ts_arrival = self.plant.last_data_ts_arrival()

        if self.plant_type in ['ArmAssist', 'ReHand']:
            if time_since_started > n_secs:
                if last_ts_arrival == 0:
                    print 'No %s data has arrived at all' % self.plant_type
                else:
                    t_elapsed = time.time() - last_ts_arrival
                    if t_elapsed > n_secs:
                        print 'No %s data in the last %.1f s' % (self.plant_type, t_elapsed)

        elif self.plant_type == 'IsMore':
            for plant_type in ['ArmAssist', 'ReHand']:
                if time_since_started > n_secs:
                    if last_ts_arrival[plant_type] == 0:
                        print 'No %s data has arrived at all' % plant_type
                    else:
                        t_elapsed = time.time() - last_ts_arrival[plant_type]
                        if t_elapsed > n_secs:
                            print 'No %s data in the last %.1f s' % (plant_type, t_elapsed)

        else:
            raise Exception('Unrecognized plant type!')

    def init_plant_display(self):
        if self.plant_type in ['ArmAssist', 'IsMore']:
            # use circle & line to represent ArmAssist's xy position & orientation
            self.xy_cursor = Circle(np.array([0, 0]), 0.5, COLORS['white'])
            self.psi_line  = Line(np.array([0, 0]), 6, 0.1, 0, COLORS['white'])

            self.add_model(self.xy_cursor)
            self.add_model(self.psi_line)

            # use circle & line to represent ArmAssist's xy position & orientation of REST POSITION
            self.xy_cursor_rest = Circle(np.array([0, 0]), 0.5, COLORS['white'])
            self.psi_line_rest  = Line(np.array([0, 0]), 6, 0.1, 0, COLORS['green'])

            self.add_model(self.xy_cursor_rest)
            self.add_model(self.psi_line_rest)

        if self.plant_type in ['ReHand', 'IsMore']:
            # use (rotating) lines to represent ReHand's angles
            px = self.starting_pos['aa_px']
            self.rh_angle_line_positions = {
                'rh_pthumb': np.array([px - 15, 0]),
                'rh_pindex': np.array([px - 5,  0]),
                'rh_pfing3': np.array([px + 5,  0]),
                'rh_pprono': np.array([px + 15, 0]),
            }
            
            self.rh_angle_lines = {}
            self.rh_angle_lines_rest = {}

            for state in rh_pos_states:
                l = Line(self.rh_angle_line_positions[state], 6, 0.1, 0, COLORS['white'])
                self.rh_angle_lines[state] = l 
                self.add_model(l)

                l_rest= Line(self.rh_angle_line_positions[state], 6, 0.1, 0, COLORS['green'])
                self.rh_angle_lines_rest[state]= l_rest 
                self.add_model(l_rest)

            # #display the first position in red --> REST position, reference to start and finish trajectories
            # if self.plant_type in ['ArmAssist', 'IsMore']:
            #     self.xy_cursor.center_pos = self.plant_pos[aa_xy_states].values
            #     self.psi_line.start_pos   = self.plant_pos[aa_xy_states].values
            #     self.psi_line.angle       = self.plant_pos['aa_ppsi']+ 90*deg_to_rad #to show it in a more intuitive way according to the forearm position

            # if self.plant_type in ['ReHand', 'IsMore']:
            #     for state in rh_pos_states:
            #         self.rh_angle_lines[state].angle = self.plant_pos[state]

    def update_plant_display(self):
        if (self.update_rest == True) and (any(self.plant_pos[aa_xy_states].values > 0.1)):
            if self.plant_type in ['ArmAssist', 'IsMore']:
                    self.xy_cursor_rest.center_pos = self.plant_pos[aa_xy_states].values
                    self.psi_line_rest.start_pos   = self.plant_pos[aa_xy_states].values
                    self.psi_line_rest.angle       = self.plant_pos['aa_ppsi']+ 90*deg_to_rad #to show it in a more intuitive way according to the forearm position
                    
            if self.plant_type in ['ReHand', 'IsMore']:
                for state in rh_pos_states:
                    self.rh_angle_lines_rest[state].angle = self.plant_pos[state]

            self.update_rest = False


        if self.plant_type in ['ArmAssist', 'IsMore']:

            self.xy_cursor.center_pos = self.plant_pos[aa_xy_states].values
            self.psi_line.start_pos   = self.plant_pos[aa_xy_states].values
            self.psi_line.angle       = self.plant_pos['aa_ppsi']+ 90*deg_to_rad #to show it in a more intuitive way according to the forearm position

        if self.plant_type in ['ReHand', 'IsMore']:

            for state in rh_pos_states:
                self.rh_angle_lines[state].angle = self.plant_pos[state]        


class CalibrationMovements(IsMoreBase):
    '''TODO.'''

    sequence_generators = []

    status = {
        'move': {'stop':  None},
    }
    
    state = 'move'  # initial state

    def _cycle(self):
        self.plant_pos[:] = self.plant.get_pos()
        self.plant_vel[:] = self.plant.get_vel()

        self.update_plant_display()

        self.task_data['plant_pos']  = self.plant_pos.values
        self.task_data['plant_vel']  = self.plant_vel.values

        super(CalibrationMovements, self)._cycle()


targetsB1 =  ['Blue', 'Brown', 'Green', 'Red']
targetsB2 = [ 'Up', 'Down', 'Grasp', 'Pinch', 'Point']
targetsF1_F2 = [
        'Green to Brown', 
        'Green to Blue', 
        'Brown to Red',
        'Brown to Green',
        'Red to Green',
        'Red to Brown',
        'Red to Blue',
        'Green to Red',
        'Brown to Blue',
        'Blue to Red',
        'Blue to Green',
        'Blue to Brown'
    ] 



def device_to_use(trial_type):
    '''Return 'ArmAssist' or 'ReHand' depending on whether xy position
    or ReHand angular positions should be used for the given trial_type
    for identifying the current point on trajectory playback.'''
    
    if (trial_type in targetsB1):
        return 'ArmAssist'
        
    elif (trial_type in targetsB2):
        return 'ReHand'

    elif (trial_type in targetsF1_F2):
        return 'ArmAssist'


class NonInvasiveBase(Autostart, Sequence, IsMoreBase):
    '''Abstract base class for noninvasive IsMore tasks (e.g., tasks for
    recording and playing back trajectories). This class defines some 
    common sequence generators for those inheriting classes.
    '''
    
    sounds_dir = os.path.expandvars('$HOME/sounds')

    sequence_generators = [
        'B1_targets',
        'B2_targets',
        'F1_targets',
        'F2_targets',
        'FreeMov_targets',
        'double_target',
    ]

    def __init__(self, *args, **kwargs):
        super(NonInvasiveBase, self).__init__(*args, **kwargs)
        pygame.mixer.init()

    def _play_sound(self, fname):
        pygame.mixer.music.load(fname)
        pygame.mixer.music.play()

    @staticmethod
    def _make_block_rand_targets(length, available_targets, shuffle=False):
        targets = []
        for k in range(length):
            a_ = available_targets[:]
            if shuffle:
                random.shuffle(a_)
            targets += a_
        return targets

    @staticmethod
    def B1_targets(length=5, Blue=1, Brown=1, Green=1, Red=1, shuffle=1):
        available_targets = []
        if Blue: available_targets.append('Blue')
        if Brown: available_targets.append('Brown')
        if Green: available_targets.append('Green')
        if Red: available_targets.append('Red')

        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=shuffle)
        return targets

    @staticmethod
    def B2_targets(length=5, Up=1, Down=1, Grasp=1, Pinch=1, Point=1, shuffle=1):
        available_targets = []
        if Up: available_targets.append('Up')
        if Down: available_targets.append('Down')
        if Grasp: available_targets.append('Grasp')
        if Pinch: available_targets.append('Pinch')
        if Point: available_targets.append('Point')

        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=shuffle)
        return targets  
  
    @staticmethod
    def F1_targets(length=1, green_to_brown=1, green_to_blue=1, brown_to_red=1, brown_to_green=1, 
                   red_to_green=1, red_to_brown=1, red_to_blue=1, green_to_red=1, brown_to_blue=1,
                   blue_to_red=1, blue_to_green=1, blue_to_brown=1, shuffle=1):
        '''
        Generate target sequence for the F1 task.
        '''
        available_targets = []
        if green_to_brown: available_targets.append('Green to Brown')
        if green_to_blue: available_targets.append('Green to Blue')
        if brown_to_red: available_targets.append('Brown to Red')
        if brown_to_green: available_targets.append('Brown to Green')
        if red_to_green: available_targets.append('Red to Green')
        if red_to_brown: available_targets.append('Red to Brown')
        if red_to_blue: available_targets.append('Red to Blue')
        if green_to_red: available_targets.append('Green to Red')
        if brown_to_blue: available_targets.append('Brown to Blue')
        if blue_to_red: available_targets.append('Blue to Red')
        if blue_to_green: available_targets.append('Blue to Green')
        if blue_to_brown: available_targets.append('Blue to Brown')

        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=shuffle)
        return targets

    @staticmethod
    def F2_targets(length=1):
        
        return length * [
            'Green to Blue',
            'Brown to Green',
            'Green to Red',
            'Blue to Brown',
            'Brown to Blue',
            'Red to Brown',
            'Blue to Red',
            'Red to Green',
            'Brown to Red',
            'Green to Brown',
            'Red to Blue',
            'Blue to Green'
        ]    

        
        #nerea  
        #targets = ['Red to Brown'] 
        return targets   

    @staticmethod
    def FreeMov_targets(length=1):
        return length * ['Go.1s']      


    @staticmethod
    def double_target(length=10):
        colors = ['Blue', 'Brown', 'Green', 'Red']
        trial_types = []
        for c1 in colors:
            for c2 in colors:
                if c1 != c2:
                    trial_types.append(c1 + ' to ' + c2)
        
        return length * trial_types


class RecordTrajectoriesBase(NonInvasiveBase):
    '''
    Base class for all tasks involving recording trajectories.
    '''
    
    fps = 10 

    status = {
        'wait': {
            'start_trial': 'instruct_rest',
            'stop': None},
        'instruct_rest': {
            'end_instruct': 'rest',
            'stop':      None},            
        'rest': {
            'end_rest': 'instruct_trial_type',
            'stop':      None},
        'instruct_trial_type': {
            'end_instruct': 'trial',
            'stop':      None},
        'trial': {
            # 'end_trial': 'wait',
            'accept_trial': 'wait',
            'reject_trial': 'instruct_rest',
            'stop':      None},    
    }
    state = 'wait'  # initial state

    # settable parameters on web interface
    rest_interval = traits.Tuple((2., 4.), desc='Min and max time to remain in the rest state.')
    trial_time    = traits.Float(10,       desc='Time to remain in the trial state.') 
    ready_time    = traits.Float(0,        desc='Time to remain in the ready state.')

    def __init__(self, *args, **kwargs):
        ## Init the pygame mixer for playing back sounds
        super(RecordTrajectoriesBase, self).__init__(*args, **kwargs)

        self.add_dtype('trial_type', np.str_, 40)
        self.add_dtype('trial_accept_reject', np.str_, 10)
        self.add_dtype('ts',         'f8',    (1,))

        self.plant.disable()
        #self.plant.enable() # to record trajectories with motors enable, easier to control
        #self.armassist.enable()

        self.experimenter_acceptance_of_trial = ''

    def _cycle(self):
        '''Runs self.fps times per second.'''

        self.plant_pos[:] = self.plant.get_pos()
        self.plant_vel[:] = self.plant.get_vel()

        self.update_plant_display()

        self.task_data['plant_pos']  = self.plant_pos.values
        self.task_data['plant_vel']  = self.plant_vel.values
        self.task_data['trial_type'] = self.trial_type
        self.task_data['ts']         = time.time()
        self.task_data['trial_accept_reject'] = self.experimenter_acceptance_of_trial

        super(RecordTrajectoriesBase, self)._cycle()

    def _test_end_instruct(self, *args, **kwargs):
        return not pygame.mixer.music.get_busy()

    def _start_instruct_rest(self):
        self._play_sound(os.path.join(self.sounds_dir, 'Rest.wav'))
        self.experimenter_acceptance_of_trial = ''

    def _start_instruct_trial_type(self):
        sound_fname = os.path.join(self.sounds_dir, self.trial_type + '.wav')
        self._play_sound(sound_fname)

    def _start_rest(self):
        # determine the random length of time to stay in the rest state
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time
    
    def _parse_next_trial(self):
        self.trial_type = self.next_trial

    def _test_end_rest(self, ts):
        return ts > self.rest_time  # and not self.pause -- needed?

    def _start_trial(self):
        print self.trial_type

    def _test_end_trial(self, ts):
        return ts > self.trial_time

    def _test_accept_trial(self, *args, **kwargs):
        return self.experimenter_acceptance_of_trial == 'accept'

    def _test_reject_trial(self, *args, **kwargs):
        return self.experimenter_acceptance_of_trial == 'reject'

class PlaybackTrajectories(NonInvasiveBase):
    '''
    Plays back trajectories stored in a file of reference trajectories.
    '''
    fps = 20
    
    status = {
        'wait': {
            'start_trial': 'go_to_start', 
            'stop': None},
        'go_to_start': {
            'at_starting_config': 'instruct_rest',
            'stop': None},              
        'instruct_rest': {
            'end_instruct': 'rest',
            'stop': None},            
        'rest': {
            'time_expired': 'instruct_trial_type',
            'stop': None},
        'instruct_trial_type': {
            'end_instruct': 'trial',
            'stop': None},
        'trial': {
            'end_trial': 'wait',
            'stop': None},
        }
    
    state = 'wait'  # initial state

    # settable parameters on web interface
    rest_interval    = traits.Tuple((2., 3.), desc='Min and max time to remain in the rest state.')
    min_advance_t    = traits.Float(0.005, desc='Minimum time to advance trajectory playback at each step of playback.') 
    search_win_t     = traits.Float(0.200, desc='Search within this time window from previous known point on trajectory.')#nerea. 0.2 --> 0.1
    aim_ahead_t      = traits.Float(0.150, desc='Aim this much time ahead of the current point on the trajectory.') #nerea 0.1 --> 0.05
    aim_ahead_t_psi  = traits.Float(0.200, desc='Specific to psi - aim this much time ahead of the current point on the trajectory.') #changed from 0.200 to 0.100 nerea
    gamma            = traits.Float(0.0,   desc='Gamma value for incorporating EMG decoded velocity.')
    ref_trajectories = traits.Instance(RefTrajectories)
    emg_decoder_file = traits.String('',   desc='Full path to EMG decoder file.')
    emg_playback_file = traits.String('',   desc='Full path to recorded EMG data file.')

    debug = True

    subtrial_start_time_last = np.inf
    
    def __init__(self, *args, **kwargs):
        super(PlaybackTrajectories, self).__init__(*args, **kwargs)

        self.add_dtype('trial_type',   np.str_, 40)
        self.add_dtype('ts',           'f8',    (1,))
        self.add_dtype('playback_vel', 'f8',    (len(self.vel_states),))
        self.add_dtype('command_vel',  'f8',    (len(self.vel_states),))
        self.add_dtype('aim_pos',      'f8',    (len(self.vel_states),))
        self.add_dtype('idx_aim',       int,    (1,))
        self.add_dtype('idx_aim_psi',   int,    (1,))
        self.add_dtype('idx_traj',      int,    (1,))
        self.add_dtype('audio_feedback_start',      int,    (1,))
        self.add_dtype('subtrial_idx',      int,    (1,))        

        self.subtrial_idx = np.nan
        self.subtrial_start_time_last = np.inf

        self.feedback_time = [0, 0, 0] #[3, 3]
        #area considered as acceptable as rest around the rest position (first position in trial) or the subtask end position
        self.target_rect = np.array([1., 1., np.deg2rad(20),  np.deg2rad(10), np.deg2rad(10), np.deg2rad(10), np.deg2rad(10)])
        
        # if EMG decoder file was specified, load EMG decoder and create feature extractor 
        try:
            self.emg_decoder = pickle.load(open(self.emg_decoder_file, 'rb'))
        except IOError:
            self.emg_decoder = None
        else:
            if self.emg_playback_file == '':
                # create EMG extractor object (it's 'source' will be set later in the init method)
                extractor_cls    = self.emg_decoder.extractor_cls
                extractor_kwargs = self.emg_decoder.extractor_kwargs
                self.emg_playback = False
            else:
                # replay old EMG data
                from ismore.emg_feature_extraction import ReplayEMGMultiFeatureExtractor
                import tables
                extractor_cls = ReplayEMGMultiFeatureExtractor

                emg_hdf = tables.open_file(self.emg_playback_file)
                channels = ['chan' + x for x in self.emg_decoder.channels]
                extractor_kwargs = dict(hdf_table=emg_hdf.root.brainamp, cycle_rate=self.fps, channels=channels)
                self.emg_playback = True
            self.emg_extractor = extractor_cls(source=None, **extractor_kwargs)

            self.add_dtype('emg_features',    'f8', (self.emg_extractor.n_features,))
            self.add_dtype('emg_features_Z',  'f8', (self.emg_extractor.n_features,))
            self.add_dtype('emg_vel',         'f8', (len(self.vel_states),))
            self.add_dtype('emg_vel_lpf',     'f8', (len(self.vel_states),))


            # for calculating/updating mean and std of EMG features online
            self.features_buffer = RingBuffer(
                item_len=self.emg_extractor.n_features,
                capacity=60*self.fps,  # 60 secs
            )

            # for low-pass filtering decoded EMG velocities
            self.emg_vel_buffer = RingBuffer(
                item_len=len(self.vel_states),
                capacity=10,
            )


        # for low-pass filtering command psi velocities
        self.psi_vel_buffer = RingBuffer(
            item_len=1,
            capacity=10,
        )

        self.plant.enable() 

        self.command_lpfs = dict()
        for state in self.vel_states:
            self.command_lpfs[state] = Filter(b=[1., -0.3], a=[1.]) # low-pass filter to smooth out command velocities

    def _set_task_type(self):
        if self.trial_type in targetsB1:
            self.task_type = 'B1'
            self.n_subtasks = 2
        elif self.trial_type in targetsB2:
            self.task_type = 'B2'
            self.n_subtasks = 2
        elif self.trial_type in targetsF1_F2:
            self.task_type = 'F1'
            self.n_subtasks = 3
        
        # return task_type

    def pos_diff(self, x1, x2):
        '''
        Calculate x1 - x2, but with a specialized definition of "-"
        '''
        if self.plant_type == 'ArmAssist':
            sub_fns = [operator.sub, operator.sub, angle_subtract]
        elif self.plant_type == 'ReHand':
            sub_fns = [angle_subtract, angle_subtract, angle_subtract, angle_subtract]
        elif self.plant_type == 'IsMore':
            sub_fns = [operator.sub, operator.sub, angle_subtract, angle_subtract, angle_subtract, angle_subtract, angle_subtract]

        x1 = np.array(x1).ravel()
        x2 = np.array(x2).ravel()
        diff_ = []
        for sub_fn, i, j in izip(sub_fns, x1, x2):
            diff_.append(sub_fn(i, j))
        return np.array(diff_)

    def _set_subgoals(self):
        if self.task_type == 'B1':
            traj = self.ref_trajectories[self.trial_type]['traj']
            pos_traj = np.array(traj[self.pos_states])

            pos_traj_diff = pos_traj - pos_traj[0]
            max_xy_displ_idx = np.argmax(map(np.linalg.norm, pos_traj_diff[:,0:2]))

            distal_goal = pos_traj[max_xy_displ_idx]
            proximal_goal = pos_traj[-1]

            self.subgoals = [distal_goal, proximal_goal]
            self.subgoal_inds = [max_xy_displ_idx, len(pos_traj)-2]
        # elif self.task_type == 'B2':
        #     raise NotImplementedError
        elif self.task_type == 'F1':
            # fit the largest triangle possible to the trajectory
            traj = self.ref_trajectories[self.trial_type]['traj']
            pos_traj = np.array(traj[self.pos_states])

            pos_traj_diff = pos_traj - pos_traj[0]
            diff = map(np.linalg.norm, pos_traj_diff[:,0:2])
            local_minima = np.zeros(len(pos_traj_diff))
            T = len(pos_traj_diff)
            support = 200
            for k in range(support, T-support):
                local_minima[k] = np.all(diff[k-support:k+support] <= diff[k]) 

            local_minima[diff < 5] = 0 # exclude anything closer than 5 cm

            local_minima_inds, = np.nonzero(local_minima)
            self.subgoal_inds = np.hstack([local_minima_inds, len(pos_traj)-2])
            print self.subgoal_inds


            self.subgoals = [pos_traj[idx] for idx in self.subgoal_inds]

            # self.subgoals = [distal_goal, proximal_goal]
            # self.subgoal_inds = [max_xy_displ_idx, len(pos_traj)-2]
        else:
            #raise ValueError("Unrecognized task type:%s " % task_type)
            pass

    def _while_trial(self):
        # determine if subgoals have been accomplished
        goal_pos_state = self.subgoals[self.subtrial_idx]
        pos_diff = self.pos_diff(self.plant.get_pos(), goal_pos_state)
        
        if np.all(np.abs(pos_diff) < np.abs(self.target_rect[:len(self.pos_states)])) and self.subgoal_reached == False or self.idx_traj > self.subgoal_inds[self.subtrial_idx and self.subgoal_reached == False]:
            print "subgoal reached"
            # print pos_diff
            # print np.all(pos_diff < self.target_rect[:len(self.pos_states)])
            # print self.idx_traj > self.subgoal_inds[self.subtrial_idx]
            # if subtrial has been accomplished (close enough to goal), move on to the next subtrial
            self.subgoal_reached = True
        print np.any(np.abs(pos_diff) > np.abs(self.target_rect[:len(self.pos_states)]))
        
        if np.any(np.abs(pos_diff) > np.abs(self.target_rect[:len(self.pos_states)])) and self.subgoal_reached == True:
            print "leaving subgoal"
            # print pos_diff
            # print np.all(pos_diff < self.target_rect[:len(self.pos_states)])
            # print self.idx_traj > self.subgoal_inds[self.subtrial_idx]
            # if subtrial has been accomplished (close enough to goal), move on to the next subtrial

            self.subtrial_start_time = self.get_time()
            self.subtrial_idx += 1
            
        self.subgoal_reached = False
        self.subtrial_idx = min(self.subtrial_idx, self.n_subtasks-1)

        if ((self.get_time() - self.subtrial_start_time) > self.feedback_time[self.subtrial_idx]) and not self.feedback_given[self.subtrial_idx]:
            self.feedback_given[self.subtrial_idx] = True
            self.task_data['audio_feedback_start'] = 1
            self._play_sound(os.path.join(self.sounds_dir, 'beep.wav'))
        else:
            self.task_data['audio_feedback_start'] = 0

    def move_plant(self):
        '''Docstring.'''

        playback_vel = pd.Series(0.0, self.vel_states)
        command_vel  = pd.Series(0.0, self.vel_states)
        aim_pos      = pd.Series(0.0, self.pos_states)

        traj = self.ref_trajectories[self.trial_type]['traj']

        self.plant_pos[:] = self.plant.get_pos()

        # number of points in the reference trajectory for the current trial type
        len_traj = traj.shape[0]

        # do this simply to avoid having to write "self." everywhere
        idx_traj = self.idx_traj
        states = self.states
        dist_fn = self.dist_fn

        # index into the current trajectory playback
        # search locally in range [start_ts, end_ts)
        # depending on the type of trial, determine where we are along the trajectory by
        # finding the idx of the point in the reference trajectory that is closest to the
        # current state of plant in either xy euclidean distance or angular l1 distance
        start_ts = traj['ts'][idx_traj] + self.min_advance_t
        end_ts   = start_ts + self.search_win_t
        search_idxs = [idx for (idx, ts) in enumerate(traj['ts']) if start_ts <= ts < end_ts]
        min_dist = np.inf
        for idx in search_idxs:
            d = dist_fn(self.plant_pos[states], traj[states].ix[idx]) 
            if idx == search_idxs[0] or d < min_dist:
                min_dist = d
                idx_traj = idx

        # find the idx of the point in the reference trajectory to aim towards
        idx_aim = idx_traj
        idx_aim_psi = idx_traj

        while idx_aim < len_traj - 1:
            if (traj['ts'][idx_aim] - traj['ts'][idx_traj]) < self.aim_ahead_t:
                idx_aim += 1
                idx_aim_psi = idx_aim 
            else:
                break

        # # aim towards a possibly different point for psi
        # idx_aim_psi = idx_traj
        # while idx_aim_psi < len_traj - 1:
        #     if (traj['ts'][idx_aim_psi] - traj['ts'][idx_traj]) < self.aim_ahead_t_psi:
        #         idx_aim_psi += 1
        #     else:
        #         break
        

        if self.plant_type in ['ArmAssist', 'IsMore']:
            if idx_traj == len_traj - 1:
                playback_vel[aa_vel_states] = np.zeros(3)
                self.finished_traj_playback = True

                # Fill in the aim pos for any after-analysis
                aim_pos['aa_px'] = traj['aa_px'][idx_traj]
                aim_pos['aa_py'] = traj['aa_py'][idx_traj]
                aim_pos['aa_ppsi'] = traj['aa_ppsi'][idx_traj]
            else:
                # ArmAssist xy
                aim_pos[aa_xy_states] = traj[aa_xy_states].ix[idx_aim]
                xy_dir = norm_vec(traj[aa_xy_states].ix[idx_aim] - self.plant_pos[aa_xy_states])


                # since armassist does not provide velocity feedback, 
                # need to calculate the xy speed for the reference trajectory
                t1   = traj['ts'][idx_traj]
                t2   = traj['ts'][idx_aim]
                pos1 = traj[aa_xy_states].ix[idx_traj]
                pos2 = traj[aa_xy_states].ix[idx_aim]
                xy_speed = np.linalg.norm((pos2 - pos1) / (t2 - t1))  # cm/s


                # apply xy-distance-dependent min and max xy speed
                xy_dist = dist(traj[aa_xy_states].ix[idx_aim], self.plant_pos[aa_xy_states])
                max_xy_speed_1 = 15                          # cm/s
                max_xy_speed_2 = xy_dist / self.aim_ahead_t  # cm/s
                max_xy_speed   = min(max_xy_speed_1, max_xy_speed_2)
                min_xy_speed   = 0  #min(0.25 * max_xy_speed_2, max_xy_speed)
                xy_speed       = bound(xy_speed, min_xy_speed, max_xy_speed) 
                #xy_speed       =  max_xy_speed 


                # ArmAssist psi (orientation) -- handle separately from xy
                aim_pos['aa_ppsi'] = traj['aa_ppsi'][idx_aim_psi]
                psi_dir = np.sign(angle_subtract(traj['aa_ppsi'][idx_aim_psi], self.plant_pos['aa_ppsi']))
                
                # since armassist does not provide velocity feedback, 
                # need to calculate the psi speed for the reference trajectory
                t1   = traj['ts'][idx_traj]
                t2   = traj['ts'][idx_aim_psi]
                psi1 = traj['aa_ppsi'][idx_traj]
                psi2 = traj['aa_ppsi'][idx_aim_psi]
                psi_speed = np.abs(angle_subtract(psi2, psi1) / (t2 - t1))  # rad/s

                # apply psi-distance-dependent min and max psi speed
                psi_dist = abs(angle_subtract(traj['aa_ppsi'][idx_aim_psi], self.plant_pos['aa_ppsi']))
                max_psi_speed_1 = 30*deg_to_rad                     # rad/s
                max_psi_speed_2 = psi_dist / self.aim_ahead_t_psi  # rad/s
                max_psi_speed   = min(max_psi_speed_1, max_psi_speed_2)
                min_psi_speed   = 0

             
                psi_speed       = bound(psi_speed, min_psi_speed, max_psi_speed)


                playback_vel[['aa_vx', 'aa_vy']] = xy_speed  * xy_dir
                playback_vel['aa_vpsi']          = (psi_speed * psi_dir) #/2

                #self.x_vel_buffer.add(playback_vel['aa_vx'])
                #self.y_vel_buffer.add(playback_vel['aa_vy'])
                
                #playback_vel[['aa_vx']] = np.mean(self.x_vel_buffer.get_all(), axis=1)
                #playback_vel[['aa_vy']] = np.mean(self.y_vel_buffer.get_all(), axis=1)
                
                # Moving average filter for the output psi angular velocity
                
                self.psi_vel_buffer.add(playback_vel['aa_vpsi'])
                std_psi_vel = np.std(self.psi_vel_buffer.get_all(), axis=1)
                mean_psi_vel = np.mean(self.psi_vel_buffer.get_all(), axis=1)

                psi_vel_points = np.array(self.psi_vel_buffer.get_all())
                z1 = psi_vel_points < (mean_psi_vel + 2*std_psi_vel)
                z2 = psi_vel_points[z1] > (mean_psi_vel - 2*std_psi_vel ) 
                psi_vel_points_ok = psi_vel_points[z1]                
                psi_vel_lpf = np.mean(psi_vel_points_ok[z2])

                

                if math.isnan(psi_vel_lpf) == False:
                    playback_vel['aa_vpsi'] = psi_vel_lpf
                #else:
                #    playback_vel['aa_vpsi'] = (psi_speed * psi_dir)/2


            if (device_to_use(self.trial_type) == 'ArmAssist' and self.plant_type == 'IsMore') :
               playback_vel[rh_vel_states] = 0

        if self.plant_type in ['ReHand', 'IsMore']:
            if idx_traj == len_traj - 1:  # reached the end of the trajectory
                playback_vel[rh_vel_states] = np.zeros(4)
                self.finished_traj_playback = True
            else:
                aim_pos[rh_pos_states] = traj[rh_pos_states].ix[idx_aim]

                ang_dir = np.sign(angle_subtract_vec(traj[rh_pos_states].ix[idx_aim], self.plant_pos[rh_pos_states]))

                vel = traj[rh_vel_states].ix[idx_traj]
                ang_speed = np.abs(vel)
                

                # apply angular-distance-dependent min and max angular speeds
                for i, state in enumerate(rh_pos_states):
                    ang_dist = abs(angle_subtract(traj[state][idx_aim], self.plant_pos[state]))
                    max_ang_speed_1 = 40*deg_to_rad     #changed nerea            # rad/s    
                    max_ang_speed_2 = ang_dist / self.aim_ahead_t  # rad/s
                    max_ang_speed   = min(max_ang_speed_1, max_ang_speed_2)
                    min_ang_speed   = 0
                    ang_speed[i]    = bound(ang_speed[i], min_ang_speed, max_ang_speed) 
                    #ang_speed[i]   = max_ang_speed

                    playback_vel[rh_vel_states] = ang_speed * ang_dir


            # if recorded ReHand trajectory is being used as the reference when playing back,
            # then don't move the ArmAssist at all
            if (device_to_use(self.trial_type) == 'ReHand' and self.plant_type == 'IsMore') :
               playback_vel[aa_vel_states] = 0
            #if recorded trajectory is B1 and plant is IsMore, do not move ReHand. need to put it again here again so that RH vel are set to 0. #nerea
            #elif (device_to_use(self.trial_type) == 'ArmAssist' and self.plant_type == 'IsMore') :
            #   playback_vel[rh_vel_states] = 0


        command_vel = playback_vel # This is different for EMG decoding!


        # print 'command_vel'
        # print command_vel



        #Apply low-pass filter to command velocities
        for state in self.vel_states:
            command_vel[state] = self.command_lpfs[state](command_vel[state])
        
        # iterate over actual State objects, not state names
        #for state in self.ssm.states:
        #    if state.name in self.vel_states:
        #        command_vel[state.name] = bound(command_vel[state.name], state.min_val, state.max_val)

        # Command zero velocity if the task is in a non-moving state
        if self.state in ['wait','rest', 'instruct_rest', 'instruct_trial_type']: 
            command_vel[:] = 0
        elif self.state == 'go_to_start':
            pos_diff = self.pos_diff(traj[self.pos_states].ix[0],self.plant_pos[self.pos_states])
            signs = np.sign(pos_diff)
            max_vel      = pd.Series(0.0, ismore_vel_states)
            max_vel[:] = np.array([1., 1., np.deg2rad(1), np.deg2rad(1), np.deg2rad(1), np.deg2rad(1), np.deg2rad(1)])
            command_vel[:] = max_vel[self.vel_states].ravel() * signs
            
            idx_aim = 0
            idx_traj = 0
        self.plant.send_vel(command_vel.values) #send velocity command to EXO
        self.idx_traj = idx_traj

    def _cycle(self):
        '''Runs self.fps times per second.'''

        # get latest position/velocity information before calling move_plant()
        self.plant_pos[:] = self.plant.get_pos()
        self.plant_vel[:] = self.plant.get_vel()
        self.move_plant()

        self.update_plant_display()

        # print self.subtrial_idx
        if not self.state == 'trial':
            self.task_data['audio_feedback_start'] = 0
            self.task_data['subtrial_idx'] = -1
        else:
            self.task_data['subtrial_idx'] = self.subtrial_idx
        self.task_data['plant_pos']  = self.plant_pos.values
        self.task_data['plant_vel']  = self.plant_vel.values
        self.task_data['trial_type'] = self.trial_type
        self.task_data['ts']         = time.time()

        super(PlaybackTrajectories, self)._cycle()

    #### STATE AND TEST FUNCTIONS ####
    def _start_wait(self):
        # determine the random length of time to stay in the rest state
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time

        self.finished_traj_playback = False
        self.idx_traj = 0

        self.subtrial_idx = 0
        self.feedback_given = [False, False, False]
        super(PlaybackTrajectories, self)._start_wait()

    def _parse_next_trial(self):
        self.trial_type = self.next_trial

        device = device_to_use(self.trial_type)
        if device == 'ArmAssist':
            self.dist_fn = dist
            self.states = aa_xy_states
        elif device == 'ReHand':
            self.dist_fn = l1_ang_dist
            self.states = rh_pos_states

    def _test_end_instruct(self, *args, **kwargs):
        return not pygame.mixer.music.get_busy()

    def _start_instruct_rest(self):
        self._play_sound(os.path.join(self.sounds_dir, 'Rest.wav'))
        print 'rest'

    def _start_instruct_trial_type(self):
        sound_fname = os.path.join(self.sounds_dir, self.trial_type + '.wav')
        self._play_sound(sound_fname)

    def _start_trial(self):
        print self.trial_type
        self._set_task_type()
        self._set_subgoals()
        self.subtrial_start_time = self.get_time()

    def _test_end_trial(self, ts):
        return self.finished_traj_playback

    def _test_at_starting_config(self, *args, **kwargs):
        traj = self.ref_trajectories[self.trial_type]['traj']
        diff_to_start = np.abs(self.plant.get_pos() - traj[self.pos_states].ix[0].ravel())
        print diff_to_start

        return np.all(diff_to_start < self.target_rect[:len(self.pos_states)])


from riglib.bmi.assist import FeedbackControllerAssist
from riglib.bmi.feedback_controllers import FeedbackController
from riglib.bmi.extractor import DummyExtractor
import operator
from itertools import izip

class TrajectoryFollowingController(FeedbackController):
    def __init__(self, plant_type='IsMore'):
        self.plant_type = plant_type

        if self.plant_type == 'ArmAssist':
            self.states = aa_pos_states + aa_vel_states + ['offset']
            self.pos_states = aa_pos_states
        elif self.plant_type == 'ReHand':
            self.states = rh_pos_states + rh_vel_states + ['offset']
            self.pos_states = rh_pos_states
        elif self.plant_type == 'IsMore':
            self.states = ismore_pos_states + ismore_vel_states + ['offset']
            self.pos_states = ismore_pos_states
        else:
            raise Exception

        self.vel_lpfs = []
        for k in range(len(self.pos_states)):
            x_coeffs = np.array([0.9**k for k in range(20)])
            x_coeffs /= np.sum(x_coeffs)
            self.vel_lpfs.append(Filter(b=x_coeffs, a=[1]))

        self.input_pos_lpfs = []
        for k in range(len(self.pos_states)):
            self.input_pos_lpfs.append(Filter(b=[1], a=[1, 0.6]))

    def pos_diff(self, x1, x2):
        '''
        Calculate x1 - x2, but with a specialized definition of "-"
        '''
        if self.plant_type == 'ArmAssist':
            sub_fns = [operator.sub, operator.sub, angle_subtract]
        elif self.plant_type == 'ReHand':
            sub_fns = [angle_subtract, angle_subtract, angle_subtract, angle_subtract]
        elif self.plant_type == 'IsMore':
            sub_fns = [operator.sub, operator.sub, angle_subtract, angle_subtract, angle_subtract, angle_subtract, angle_subtract]

        x1 = np.array(x1).ravel()
        x2 = np.array(x2).ravel()
        diff_ = []
        for sub_fn, i, j in izip(sub_fns, x1, x2):
            diff_.append(sub_fn(i, j))
        return np.array(diff_)

    def calc_next_state(self, current_state, target_state, mode=None):
        if 0:
            print "current_state"
            print np.array(current_state).ravel()
            print "target state"
            print np.array(target_state).ravel()

        # filter the position state
        # for k in range(len(self.pos_states)):
        #     current_state[k,0] = self.input_pos_lpfs[k](current_state[k,0])

        # TODO this does not work for IsMore or ReHand plants
        state_pos_diff = self.pos_diff(target_state, current_state)

        # normalize the state_diff
        if np.linalg.norm(state_pos_diff[0:2]) > 0:
            state_pos_diff[0:2] /= np.linalg.norm(state_pos_diff[0:2])
        state_pos_diff[2:] = np.sign(state_pos_diff[2:])

        # vel = ref_speed * diff
        playback_vel = np.array(target_state[3:6,0]).ravel()
        playback_vel[0:2] = state_pos_diff[0:2] * np.linalg.norm(playback_vel[0:2])
        playback_vel[2:] = np.abs(playback_vel[2:]) * state_pos_diff[2:]

        # print "state pos diff"
        # print state_pos_diff

        # print "playback_vel"
        # print playback_vel

        # Apply low-pass filters to velocity
        for k in range(len(playback_vel)):
            playback_vel[k] = self.vel_lpfs[k](playback_vel[k])

        # bound the velocity if needed in any dimension
        max_vel = np.array([15./np.sqrt(2), 15./np.sqrt(2), np.deg2rad(30)]) #, np.deg2rad(40), np.deg2rad(40), np.deg2rad(40), np.deg2rad(40)])
        # max_vel = np.array([10./np.sqrt(2), 10./np.sqrt(2), np.deg2rad(30), np.deg2rad(40), np.deg2rad(40), np.deg2rad(40), np.deg2rad(40)])
        min_vel = -max_vel
        below_min = min_vel > playback_vel 
        above_max = playback_vel > max_vel
        playback_vel[below_min] = min_vel[below_min]
        playback_vel[above_max] = max_vel[above_max]

        
        # print "bounded playback vel"
        # print playback_vel


        # return a "fake" next state where the position is not updated (since it's not a tracked variable anyway..)
        ns = current_state.copy()
        ns[3:6,0] = playback_vel.reshape(-1,1)
        return ns


class PlaybackTrajectories2(BMILoop, NonInvasiveBase):
    '''
    Should be functionally the same as PlaybackTrajectories, but implemented using the generic BMI architecture.
    '''
    fps = 20
    current_assist_level = 1
    debug = False
    
    status = {
        'wait': {
            'start_trial': 'go_to_start',
            'stop': None},
        'go_to_start': {
            'at_starting_config': 'instruct_rest',
            'stop': None},            
        'instruct_rest': {
            'end_instruct': 'rest',
            'stop': None},            
        'rest': {
            'time_expired': 'instruct_trial_type',
            'stop': None},
        'instruct_trial_type': {
            'end_instruct': 'trial',
            'stop': None},
        'trial': {
            'end_trial': 'wait',
            'stop': None},
        }
    
    state = 'wait'  # initial state

    # settable parameters on web interface
    rest_interval    = traits.Tuple((2., 3.), desc='Min and max time to remain in the rest state.')
    min_advance_t    = traits.Float(0.005, desc='Minimum time to advance trajectory playback at each step of playback.') 
    search_win_t     = traits.Float(0.200, desc='Search within this time window from previous known point on trajectory.')#nerea. 0.2 --> 0.1
    aim_ahead_t      = traits.Float(0.150, desc='Aim this much time ahead of the current point on the trajectory.') #nerea 0.1 --> 0.05
    aim_ahead_t_psi  = traits.Float(0.200, desc='Specific to psi - aim this much time ahead of the current point on the trajectory.') #changed from 0.200 to 0.100 nerea
    gamma            = traits.Float(0.0,   desc='Gamma value for incorporating EMG decoded velocity.')
    ref_trajectories = traits.Instance(RefTrajectories)

    target_rect = np.array([2., 2., np.deg2rad(15)])#this is the area in X, Y and psi aroun the rest position that you accept as ok to consider that the plant has reached already the rest position
    
    def __init__(self, *args, **kwargs):
        super(PlaybackTrajectories2, self).__init__(*args, **kwargs)

        self.add_dtype('trial_type',   np.str_, 40)
        self.add_dtype('ts',           'f8',    (1,))
        self.add_dtype('playback_vel', 'f8',    (len(self.vel_states),))
        self.add_dtype('command_vel',  'f8',    (len(self.vel_states),))
        self.add_dtype('aim_pos',      'f8',    (len(self.vel_states),))
        self.add_dtype('idx_aim',       int,    (1,))
        self.add_dtype('idx_aim_psi',   int,    (1,))
        self.add_dtype('idx_traj',      int,    (1,))

    def load_decoder(self):
        from ismore_bmi_lib import StateSpaceIsMore, StateSpaceArmAssist, StateSpaceReHand
        if self.plant_type == 'IsMore':
            self.ssm = StateSpaceIsMore()
        elif self.plant_type == 'ArmAssist':
            self.ssm = StateSpaceArmAssist()
        elif self.plant_type == 'ReHand':
            self.ssm = StateSpaceReHand()

        A, B, W = self.ssm.get_ssm_matrices()
        filt = MachineOnlyFilter(A, W)
        units = []
        self.decoder = Decoder(filt, units, self.ssm, binlen=0.1)
        self.decoder.n_features = 1

    def create_feature_extractor(self):
        self.extractor = DummyExtractor()
        self._add_feature_extractor_dtype()

    def create_goal_calculator(self):
        pass

    def create_assister(self):
        fb_ctrl = TrajectoryFollowingController(plant_type=self.plant_type)
        self.assister = FeedbackControllerAssist(fb_ctrl, style='mixing')

    def get_target_BMI_state(self, *args):
        '''
        Determine where the exo is supposed to be, either based on the trajectory following 
        algorithm or standing still (in certain phases of the task)
        '''
        if self.state in ['wait','rest', 'instruct_rest', 'instruct_trial_type']:
            # Plant should not move during any of these states
            current_pos = self.plant.get_pos()
            target_state = np.hstack([current_pos, np.zeros_like(current_pos), 1])
            target_state = target_state.reshape(-1, 1)
            idx_aim = -1
            aim_pos = np.nan * np.ones(len(current_pos))
            ref_vel = np.zeros(len(current_pos))

        elif self.state in ['trial', 'go_to_start']:
            # index into the current trajectory playback
            # search locally in range [start_ts, end_ts)
            # depending on the type of trial, determine where we are along the trajectory by
            # finding the idx of the point in the reference trajectory that is closest to the
            # current state of plant in either xy euclidean distance or angular l1 distance

            aim_pos = pd.Series(0.0, self.pos_states)
            ref_vel = pd.Series(0.0, self.vel_states)

            traj = self.ref_trajectories[self.trial_type]['traj']
            self.plant_pos[:] = self.plant.get_pos()


            if self.state == 'trial':
                # number of points in the reference trajectory for the current trial type
                len_traj = traj.shape[0]

                # do this simply to avoid having to write "self." everywhere
                idx_traj = self.idx_traj
                states = self.states
                dist_fn = self.dist_fn

                # find the point on the reference trajectory closest to the current position
                start_ts = traj['ts'][idx_traj] + self.min_advance_t
                end_ts   = start_ts + self.search_win_t
                search_idxs = [idx for (idx, ts) in enumerate(traj['ts']) if start_ts <= ts < end_ts]
                min_dist = np.inf
                for idx in search_idxs:
                    d = dist_fn(self.plant_pos[states], traj[states].ix[idx]) 
                    if idx == search_idxs[0] or d < min_dist:
                        min_dist = d
                        idx_traj = idx


                # find the idx of the point in the reference trajectory to aim towards
                idx_aim = idx_traj

                while idx_aim < len_traj - 1:
                    if (traj['ts'][idx_aim] - traj['ts'][idx_traj]) < self.aim_ahead_t:
                        idx_aim += 1
                    else:
                        break

                # save the pos to aim to
                aim_pos[self.pos_states] = traj[self.pos_states].ix[idx_aim]
                # read out the velocity based on the selected point along the trajectory
                ref_vel[self.vel_states] = traj[self.vel_states].ix[idx_traj]
            elif self.state == 'go_to_start':
                idx_aim = 0
                idx_traj = 0

                # save the pos to aim to (first point in the reference trajectory for this trial)
                aim_pos[self.pos_states] = traj[self.pos_states].ix[idx_aim]
                # read out the velocity based on the selected point along the trajectory
                ref_vel[self.vel_states] = np.array([1., 1., np.deg2rad(1)]) #np.abs(traj[self.vel_states].ix[idx_traj])     


            # store pointer in the trajectory for next time
            self.idx_traj = idx_traj
            if idx_traj == len(traj) - 1:
                self.finished_traj_playback = True

            ##### variable stuff ####
            if self.debug:
                print idx_aim
                print self.plant_pos
                print 

            aim_pos = aim_pos.ravel()
            ref_vel = ref_vel.ravel()
            target_state = np.hstack([aim_pos, ref_vel, 1])
            target_state = target_state.reshape(-1, 1)
            self.plant.send_vel(ref_vel) #send velocity command to EXO #nerea
        else:
            raise NotImplementedError("Unrecognized task state: %s!" % self.state)


        self.task_data['idx_aim'] = idx_aim
        self.task_data['aim_pos'] = aim_pos
        self.task_data['playback_vel'] = ref_vel

        return target_state

    def _cycle(self):
        
        self.update_plant_display()

        self.task_data['plant_pos']  = self.plant_pos.values
        self.task_data['plant_vel']  = self.plant_vel.values
        self.task_data['trial_type'] = self.trial_type
        self.task_data['ts']         = time.time()
        self.task_data['command_vel'] = self.decoder['qdot']
        

        print self.task_data['command_vel']
        super(PlaybackTrajectories2, self)._cycle()

    #### STATE AND TEST FUNCTIONS ####
    def _start_wait(self):
        # determine the random length of time to stay in the rest state
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time

        self.finished_traj_playback = False
        self.idx_traj = 0
        super(PlaybackTrajectories2, self)._start_wait()

    def _parse_next_trial(self):
        self.trial_type = self.next_trial

        device = device_to_use(self.trial_type)
        if device == 'ArmAssist':
            self.dist_fn = dist
            self.states = aa_xy_states
        elif device == 'ReHand':
            self.dist_fn = l1_ang_dist
            self.states = rh_pos_states

    def _test_end_instruct(self, *args, **kwargs):
        return not pygame.mixer.music.get_busy()

    def _start_instruct_rest(self):
        self._play_sound(os.path.join(self.sounds_dir, 'Rest.wav'))

    def _start_instruct_trial_type(self):
        sound_fname = os.path.join(self.sounds_dir, self.trial_type + '.wav')
        self._play_sound(sound_fname)

    def _start_trial(self):
        print self.trial_type

    def _test_end_trial(self, ts):
        return self.finished_traj_playback

    def _test_at_starting_config(self, *args, **kwargs):
        traj = self.ref_trajectories[self.trial_type]['traj']
        diff_to_start = np.abs(self.plant.get_pos() - traj[self.pos_states].ix[0].ravel())
        print diff_to_start
        return np.all(diff_to_start < self.target_rect)



class EMGTrajectoryDecoding(PlaybackTrajectories):
    emg_decoder_file = traits.String('',   desc='Full path to EMG decoder file.')

    def __init__(self, *args, **kwargs):
        super(EMGTrajectoryDecoding, self).__init__(*args, **kwargs)

        self.add_dtype('trial_type',   np.str_, 40)
        self.add_dtype('ts',           'f8',    (1,))
        self.add_dtype('playback_vel', 'f8',    (len(self.vel_states),))
        self.add_dtype('command_vel',  'f8',    (len(self.vel_states),))
        self.add_dtype('aim_pos',      'f8',    (len(self.vel_states),))
        self.add_dtype('idx_aim',       int,    (1,))
        self.add_dtype('idx_aim_psi',   int,    (1,))
        self.add_dtype('idx_traj',      int,    (1,))

        # if EMG decoder file was specified, load EMG decoder and create feature extractor 
        self.emg_decoder = pickle.load(open(self.emg_decoder_file, 'rb'))

        # create EMG extractor object (it's 'source' will be set later in the init method)
        extractor_cls    = self.emg_decoder.extractor_cls
        extractor_kwargs = self.emg_decoder.extractor_kwargs
        self.emg_playback = False

        self.emg_extractor = extractor_cls(source=None, **extractor_kwargs)

        self.add_dtype('emg_features',    'f8', (self.emg_extractor.n_features,))
        self.add_dtype('emg_features_Z',  'f8', (self.emg_extractor.n_features,))
        self.add_dtype('emg_vel',         'f8', (len(self.vel_states),))
        self.add_dtype('emg_vel_lpf',     'f8', (len(self.vel_states),))

        # for calculating/updating mean and std of EMG features online
        self.features_buffer = RingBuffer(
            item_len=self.emg_extractor.n_features,
            capacity=60*self.fps,  # 60 secs
        )

        # for low-pass filtering decoded EMG velocities
        self.emg_vel_buffer = RingBuffer(
            item_len=len(self.vel_states),
            capacity=10,
        )

        # for low-pass filtering command psi velocities
        self.psi_vel_buffer = RingBuffer(
            item_len=1,
            capacity=10,
        )
        
        # # for low-pass filtering command x and y velocities
        # self.x_vel_buffer = RingBuffer(
        #     item_len=1,
        #     capacity=5,
        # )

        # self.y_vel_buffer = RingBuffer(
        #     item_len=1,
        #     capacity=5,
        # )
        
        self.plant.enable() 

        self.command_lpfs = dict()
        for state in self.vel_states:
            self.command_lpfs[state] = Filter(b=[1., -0.3], a=[1.]) # low-pass filter to smooth out command velocities

    def init(self):
        self.emg_extractor.source = self.brainamp_source
        super(EMGTrajectoryDecoding, self).init()

    def move_plant(self):
        '''Docstring.'''

        playback_vel = pd.Series(0.0, self.vel_states)
        command_vel  = pd.Series(0.0, self.vel_states)
        aim_pos      = pd.Series(0.0, self.pos_states)
        emg_vel      = pd.Series(0.0, self.vel_states) #nerea

        traj = self.ref_trajectories[self.trial_type]['traj']

        self.plant_pos[:] = self.plant.get_pos()
        # print 'self.plant_pos'
        # print self.plant_pos

 
        # t= time.clock()  # t is wall seconds elapsed (floating point)

        # print 'time'
        # print t


        # number of points in the reference trajectory for the current trial type
        len_traj = traj.shape[0]

        # do this simply to avoid having to write "self." everywhere
        idx_traj = self.idx_traj
        states = self.states
        dist_fn = self.dist_fn

        # index into the current trajectory playback
        # search locally in range [start_ts, end_ts)
        # depending on the type of trial, determine where we are along the trajectory by
        # finding the idx of the point in the reference trajectory that is closest to the
        # current state of plant in either xy euclidean distance or angular l1 distance
        start_ts = traj['ts'][idx_traj] + self.min_advance_t
        end_ts   = start_ts + self.search_win_t
        search_idxs = [idx for (idx, ts) in enumerate(traj['ts']) if start_ts <= ts < end_ts]
        min_dist = np.inf
        for idx in search_idxs:
            d = dist_fn(self.plant_pos[states], traj[states].ix[idx]) 
            if idx == search_idxs[0] or d < min_dist:
                min_dist = d
                idx_traj = idx

        # find the idx of the point in the reference trajectory to aim towards
        idx_aim = idx_traj
        idx_aim_psi = idx_traj #nerea

        while idx_aim < len_traj - 1:
            if (traj['ts'][idx_aim] - traj['ts'][idx_traj]) < self.aim_ahead_t:
                idx_aim += 1
                idx_aim_psi = idx_aim #nerea
            else:
                break

        


        '''
        # aim towards a possibly different point for psi
        idx_aim_psi = idx_traj
        while idx_aim_psi < len_traj - 1:
            if (traj['ts'][idx_aim_psi] - traj['ts'][idx_traj]) < self.aim_ahead_t_psi:
                idx_aim_psi += 1
            else:
                break
        '''



        if self.plant_type in ['ArmAssist', 'IsMore']:
            if idx_traj == len_traj - 1:
                playback_vel[aa_vel_states] = np.zeros(3)
                self.finished_traj_playback = True

                # Fill in the aim pos for any after-analysis
                aim_pos['aa_px'] = traj['aa_px'][idx_traj]
                aim_pos['aa_py'] = traj['aa_py'][idx_traj]
                aim_pos['aa_ppsi'] = traj['aa_ppsi'][idx_traj]


            else:
                # ArmAssist xy
                aim_pos[aa_xy_states] = traj[aa_xy_states].ix[idx_aim]
                xy_dir = norm_vec(traj[aa_xy_states].ix[idx_aim] - self.plant_pos[aa_xy_states])


                # since armassist does not provide velocity feedback, 
                # need to calculate the xy speed for the reference trajectory
                t1   = traj['ts'][idx_traj]
                t2   = traj['ts'][idx_aim]
                pos1 = traj[aa_xy_states].ix[idx_traj]
                pos2 = traj[aa_xy_states].ix[idx_aim]
                xy_speed = np.linalg.norm((pos2 - pos1) / (t2 - t1))  # cm/s

                
                                               
                # apply xy-distance-dependent min and max xy speed
                xy_dist = dist(traj[aa_xy_states].ix[idx_aim], self.plant_pos[aa_xy_states])
                max_xy_speed_1 = 10   #changed from 2 to 5                        # cm/s
                max_xy_speed_2 = xy_dist / self.aim_ahead_t  # cm/s
                max_xy_speed   = min(max_xy_speed_1, max_xy_speed_2)
                min_xy_speed   = 0  #min(0.25 * max_xy_speed_2, max_xy_speed)
                #xy_speed       = bound(xy_speed, min_xy_speed, max_xy_speed) #nerea
                xy_speed       =  max_xy_speed

                # ArmAssist psi (orientation) -- handle separately from xy
                aim_pos['aa_ppsi'] = traj['aa_ppsi'][idx_aim_psi]
                psi_dir = np.sign(angle_subtract(traj['aa_ppsi'][idx_aim_psi], self.plant_pos['aa_ppsi']))
                
                # since armassist does not provide velocity feedback, 
                # need to calculate the psi speed for the reference trajectory
                t1   = traj['ts'][idx_traj]
                t2   = traj['ts'][idx_aim_psi]
                psi1 = traj['aa_ppsi'][idx_traj]
                psi2 = traj['aa_ppsi'][idx_aim_psi]
                psi_speed = np.abs(angle_subtract(psi2, psi1) / (t2 - t1))  # rad/s

                # apply psi-distance-dependent min and max psi speed
                psi_dist = abs(angle_subtract(traj['aa_ppsi'][idx_aim_psi], self.plant_pos['aa_ppsi']))
                max_psi_speed_1 = 30*deg_to_rad                     # rad/s
                max_psi_speed_2 = psi_dist / self.aim_ahead_t_psi  # rad/s
                max_psi_speed   = min(max_psi_speed_1, max_psi_speed_2)
                min_psi_speed   = 0

             
                #psi_speed       = bound(psi_speed, min_psi_speed, max_psi_speed) #nerea
                psi_speed       = max_psi_speed 

                playback_vel[['aa_vx', 'aa_vy']] = xy_speed  * xy_dir
                playback_vel['aa_vpsi']          = (psi_speed * psi_dir)/2


                #self.x_vel_buffer.add(playback_vel['aa_vx'])
                #self.y_vel_buffer.add(playback_vel['aa_vy'])
                
                #playback_vel[['aa_vx']] = np.mean(self.x_vel_buffer.get_all(), axis=1)
                #playback_vel[['aa_vy']] = np.mean(self.y_vel_buffer.get_all(), axis=1)
                
                # Moving average filter for the output psi angular velocity
                
                self.psi_vel_buffer.add(playback_vel['aa_vpsi'])
                std_psi_vel = np.std(self.psi_vel_buffer.get_all(), axis=1)
                mean_psi_vel = np.mean(self.psi_vel_buffer.get_all(), axis=1)

                psi_vel_points = np.array(self.psi_vel_buffer.get_all())
                z1 = psi_vel_points < (mean_psi_vel + 2*std_psi_vel)
                z2 = psi_vel_points[z1] > (mean_psi_vel - 2*std_psi_vel ) 
                psi_vel_points_ok = psi_vel_points[z1]                
                psi_vel_lpf = np.mean(psi_vel_points_ok[z2])

                

                if math.isnan(psi_vel_lpf) == False:
                    playback_vel['aa_vpsi'] = psi_vel_lpf
                #else:
                #    playback_vel['aa_vpsi'] = (psi_speed * psi_dir)/2


        if self.plant_type in ['ReHand', 'IsMore']:
            if idx_traj == len_traj - 1:  # reached the end of the trajectory
                playback_vel[rh_vel_states] = np.zeros(4)
                self.finished_traj_playback = True
            else:
                aim_pos[rh_pos_states] = traj[rh_pos_states].ix[idx_aim]

                ang_dir = np.sign(angle_subtract_vec(traj[rh_pos_states].ix[idx_aim], self.plant_pos[rh_pos_states]))

                vel = traj[rh_vel_states].ix[idx_traj]
                ang_speed = np.abs(vel)
                

                # apply angular-distance-dependent min and max angular speeds
                for i, state in enumerate(rh_pos_states):
                    ang_dist = abs(angle_subtract(traj[state][idx_aim], self.plant_pos[state]))
                    max_ang_speed_1 = 40*deg_to_rad     #changed nerea            # rad/s    
                    max_ang_speed_2 = ang_dist / self.aim_ahead_t  # rad/s
                    max_ang_speed   = min(max_ang_speed_1, max_ang_speed_2)
                    min_ang_speed   = 0
                    #ang_speed[i]    = bound(ang_speed[i], min_ang_speed, max_ang_speed) #nerea
                    ang_speed[i]   = max_ang_speed

                    playback_vel[rh_vel_states] = ang_speed * ang_dir


        # if recorded ReHand trajectory is being used as the reference when playing back,
        # then don't move the ArmAssist at all
        if (device_to_use(self.trial_type) == 'ReHand' and self.plant_type == 'IsMore') :
            playback_vel[aa_vel_states] = 0
            #if recorded trajectory is B1 and plant is IsMore, do not move ReHand. need to put it again here again so that RH vel are set to ` #nerea
            #elif (device_to_use(self.trial_type) == 'ArmAssist' and self.plant_type == 'IsMore') :
            #   playback_vel[rh_vel_states] = 0

        #print 'playback_vel:'
        #print playback_vel

        # run EMG feature extractor and decoder
        if self.emg_decoder is not None:
            emg_features = self.emg_extractor() # emg_features is of type 'dict'

            self.features_buffer.add(emg_features[self.emg_extractor.feature_type])
            if 1: #self.features_buffer.num_items() > 1 * self.fps:
                # if we have more than 1 second of recent EMG data, then
                #   calculate mean and std from this data
                recent_features = self.features_buffer.get_all()
                features_mean = np.mean(recent_features, axis=1)
                features_std  = np.std(recent_features, axis=1)
            else:
                # else use mean and std from the EMG data that was used to 
                #   train the decoder
                features_mean = self.emg_decoder.features_mean
                features_std  = self.emg_decoder.features_std

            # z-score the EMG features
            emg_features_Z = (emg_features[self.emg_extractor.feature_type] - features_mean) / features_std 
            emg_vel = self.emg_decoder(emg_features_Z)

            self.emg_vel_buffer.add(emg_vel[self.vel_states])

            #print 'any zeros in std vector?:', any(features_std == 0.0)


            emg_vel_lpf = np.mean(self.emg_vel_buffer.get_all(), axis=1)

            self.task_data['emg_features']   = emg_features[self.emg_extractor.feature_type]
            self.task_data['emg_features_Z'] = emg_features_Z
            self.task_data['emg_vel']        = emg_vel
            self.task_data['emg_vel_lpf']    = emg_vel_lpf
                  

            # combine EMG decoded velocity and playback velocity into one velocity command
            norm_playback_vel = np.linalg.norm(playback_vel)
            epsilon = 1e-6
            if (norm_playback_vel < epsilon):
                # if norm of the playback velocity is 0 or close to 0,
                #   then just set command velocity to 0s
                command_vel[:] = 0.0

            else:

                #feedback 1
                term1 = self.gamma * emg_vel_lpf
                term2 = (1 - self.gamma) * playback_vel

                #feedback 2
                # term1 = self.gamma * ((np.dot(emg_vel_lpf, playback_vel) / (norm_playback_vel**2)) * playback_vel)
                # term2 = (1 - self.gamma) * playback_vel


                #term1 = self.gamma * self.emg_decoder.lambda_coeffs * ((np.dot(emg_vel_lpf, playback_vel) / (norm_playback_vel**2)) * playback_vel)
                #term2 = (1 - self.gamma * self.emg_decoder.lambda_coeffs) * playback_vel
                

                command_vel = term1 + term2


                if (device_to_use(self.trial_type) == 'ReHand' and self.plant_type == 'IsMore') :
                    command_vel[aa_vel_states] = 0.0
        else:
            command_vel = playback_vel




        # # # # Apply low-pass filter to command velocities
        # for state in self.vel_states:
        #     print command_vel[state]
        #     command_vel[state] = self.command_lpfs[state](command_vel[state])

              
        # iterate over actual State objects, not state names
        # for state in self.ssm.states:
        #     if state.name in self.vel_states:
        #         command_vel[state.name] = bound(command_vel[state.name], state.min_val, state.max_val)

        # Command zero velocity if the task is in a non-moving state
        if self.state in ['wait','rest', 'instruct_rest', 'instruct_trial_type']: 
            command_vel[:] = 0

        self.plant.send_vel(command_vel.values)
        self.idx_traj = idx_traj

        self.task_data['playback_vel'] = playback_vel.values
        self.task_data['command_vel']  = command_vel.values
        self.task_data['aim_pos']      = aim_pos.values
        self.task_data['idx_aim']      = idx_aim
        self.task_data['idx_aim_psi']  = idx_aim_psi
        self.task_data['idx_traj']     = idx_traj 



    def _cycle(self):
        '''Runs self.fps times per second.'''

        # get latest position/velocity information before calling move_plant()
        self.plant_pos[:] = self.plant.get_pos()
        self.plant_vel[:] = self.plant.get_vel()
        self.move_plant()

        self.update_plant_display()

        self.task_data['plant_pos']  = self.plant_pos.values
        self.task_data['plant_vel']  = self.plant_vel.values
        self.task_data['trial_type'] = self.trial_type
        self.task_data['ts']         = time.time()

        super(PlaybackTrajectories, self)._cycle()


    #### STATE AND TEST FUNCTIONS ####

    def _start_wait(self):
        # get the next trial type in the sequence
        #try:
        #    self.trial_type = self.gen.next()
        #except StopIteration:
        #    self.end_task()

        # determine the random length of time to stay in the rest state
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time

        self.finished_traj_playback = False
        self.idx_traj = 0
        super(PlaybackTrajectories, self)._start_wait()

    def _test_end_rest(self, ts):
        return ts > self.rest_time  # and not self.pause -- needed?

    def _start_trial(self):
        print self.trial_type
        print pygame.mixer.music.get_busy()

    def _test_end_trial(self, ts):
        return self.finished_traj_playback

    def _parse_next_trial(self):
        self.trial_type = self.next_trial


class SimEMGTrajectoryDecoding(EMGTrajectoryDecoding):
    '''
    Same as above, but only for debugging purposes, so uses an old HDF file for EMG data instead of live streaming data
    '''
    emg_playback_file = traits.String('', desc='file from which to replay old EMG data. Leave blank to stream EMG data from the brainamp system')

#############################################################################
##### Derivative tasks to record specific types of trajectories #############
#############################################################################

# tasks for only EMG recording (they do NOT include a "ready" period)

class RecordB1_EMG(RecordTrajectoriesBase):
    '''Task class for recording trajectories for the center-out task to four different targets.'''
    pass

      
class RecordB2_EMG(RecordTrajectoriesBase):
    '''Task class for recording trajectories for the center-out task to four different targets.'''
    pass


class RecordF1_EMG(RecordTrajectoriesBase):
    '''Task class for recording trajectories for the center-out task to four different targets.'''
    pass
    

class RecordF2_EMG(RecordTrajectoriesBase):
    '''Task class for recording trajectories for the center-out task to four different targets.'''
    pass

class RecordFreeMov_EMG(RecordTrajectoriesBase):
    '''Task class for recording free movements.'''
    pass

# tasks for EEG&EMG recording (they include a "ready" period)

class RecordB1(RecordTrajectoriesBase):
    '''Task class for recording trajectories for the center-out task.'''

    status = {
        'rest': {
            'end_rest': 'ready',
            'stop':      None},
        'ready': {
            'end_ready': 'trial',
            'stop':      None},
        'trial': {
            'end_trial': 'rest',
            'stop':      None},
    }
    state = 'rest'  # initial state

    # settable parameters on web interface
    rest_interval = traits.Tuple((2., 4.), desc='Min and max time to remain in the rest state.')
    ready_time    = traits.Float(2,        desc='Time to remain in the ready state.')
    trial_time    = traits.Float(10,       desc='Time to remain in the trial state.')

    def _start_rest(self):
        filename = os.path.join(self.sounds_dir, 'Rest.wav')
        play_audio(filename)

        # get the next trial type in the sequence
        try:
            self.trial_type = self.gen.next()
        except StopIteration:
            self.end_task()

        # determine the random length of time to stay in the rest state
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time
        
    def _test_end_rest(self, ts):
        return ts > self.rest_time  # and not self.pause -- needed?

    def _start_ready(self):
        filename = os.path.join(self.sounds_dir, self.trial_type + '.wav')
        play_audio(filename)

    def _test_end_ready(self, ts):
        return ts > self.ready_time

    def _start_trial(self):
        print self.trial_type
        filename = os.path.join(self.sounds_dir,'Go.1s.wav')
        play_audio(filename)

    def _test_end_trial(self, ts):
        return ts > self.trial_time


class RecordF2(RecordTrajectoriesBase):
    '''Task class for recording trajectories for the spherical grip task.'''

    status = {
        'rest': {
            'end_rest':  'trial',
            'stop':      None},
        'trial': {
            'end_trial': 'rest',
            'stop':      None},
    }
    state = 'rest'  # initial state

    # settable parameters on web interface
    rest_interval = traits.Tuple((2., 4.), desc='Min and max time to remain in the rest state.')
    trial_time    = traits.Float(8,        desc='Time to remain in the trial state.')

    def _start_rest(self):
        filename = os.path.join(self.sounds_dir, 'Rest.wav')
        play_audio(filename)

        # get the next trial type in the sequence
        try:
            self.trial_type = self.gen.next()
        except StopIteration:
            self.end_task()

        # determine the random length of time to stay in the rest state
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time
        
    def _test_end_rest(self, ts):
        return ts > self.rest_time  # and not self.pause -- needed?

    def _start_trial(self):
        print self.trial_type
        filename = os.path.join(self.sounds_dir, self.trial_type + '.wav')
        play_audio(filename)

    def _test_end_trial(self, ts):
        return ts > self.trial_time

#task to record only ReHand data -->  for testing pursposes
class Record_ReHand_data(RecordTrajectoriesBase):
    '''Task class for recording free movements.'''

    status = {
        'trial': {
            'end_trial': 'rest',
            'stop':      None},
    }
    state = 'trial'  # initial state

    # settable parameters on web interface
    trial_time    = traits.Float(180,      desc='Time to remain in the trial state.')

    def _start_trial(self):
        
        self.plant_type = 'ReHand'

        # get the next trial type in the sequence
        try:
            self.trial_type = self.gen.next()
        except StopIteration:
            self.end_task()

        print 'Recording ReHand data'

        #print self.trial_type
        #filename = os.path.join(self.sounds_dir, self.trial_type + '.wav')
        #play_audio(filename)
        

    def _test_end_trial(self, ts):
        return ts > self.trial_time

class Disable_System(NonInvasiveBase):

    def __init__(self, *args, **kwargs):
        super(Disable_System, self).__init__(*args, **kwargs)
        self.plant.disable()

        print 'Motors disabled'



#############################################################################
#############################################################################

class PlantControlBase(Sequence, IsMoreBase):
    '''Abstract base class for controlling plants through a sequence of targets.'''
    
    #fps = 10
    fps = 10 #changed nerea

    status = {
        'wait': {
            'start_trial':         'target',
            'stop':                None},
        'target': {
            'enter_target':        'hold',
            'timeout':             'timeout_penalty',
            'stop':                None},
        'hold': {
            'leave_early':         'hold_penalty',
            'hold_complete':       'targ_transition'},
        'targ_transition': {
            'trial_complete':      'reward',
            'trial_abort':         'wait',
            'trial_incomplete':    'target'},
        'timeout_penalty': {
            'timeout_penalty_end': 'targ_transition'},
        'hold_penalty': {
            'hold_penalty_end':    'targ_transition'},
        'reward': {
            'reward_end': 'wait'}
    }
    state = "wait"  # initial state
    trial_end_states = ['timeout_penalty', 'hold_penalty', 'reward']

    sequence_generators = ['armassist_simple', 'rehand_simple', 'ismore_simple']

    # settable parameters on web interface
    wait_time            = traits.Float(2,  desc='Time to remain in the wait state.')
    reward_time          = traits.Float(.5, desc='Time in reward state.')
    hold_time            = traits.Float(.2, desc='Hold time required at targets.')
    hold_penalty_time    = traits.Float(1,  desc='Penalty time for target hold error.')
    timeout_time         = traits.Float(15, desc='Time allowed to go between targets.')
    timeout_penalty_time = traits.Float(1,  desc='Penalty time for timeout error.')
    max_attempts         = traits.Int(10,   desc='The number of attempts at a target before skipping to the next one.')
    target_radius        = traits.Float(2,  desc='Radius of targets.')
    angular_range        = traits.Float(5*deg_to_rad, desc='Angular orientation must be within +/- this amount of target angle for success.')

    target_index = -1  # helper variable to keep track of which target to display within a trial
    n_attempts = 0     # helper variable to keep track of the number of failed attempts at a given trial
    
    def __init__(self, *args, **kwargs):
        super(PlantControlBase, self).__init__(*args, **kwargs)

        self.command_vel = pd.Series(0.0, self.vel_states)
        self.target_pos  = pd.Series(0.0, self.pos_states)

        self.add_dtype('command_vel', 'f8', (len(self.command_vel),))
        self.add_dtype('target_pos',  'f8', (len(self.target_pos),))
        self.add_dtype('target_index', 'i', (1,))

        self.init_target_display()

    def move_plant(self):
        raise NotImplementedError  # implement in subclasses

    def init_target_display(self):
        if self.plant_type in ['ArmAssist', 'IsMore']:
            # use circle & sector to represent the ArmAssist part of a target (xy position + orientation)
            self.target1        = Circle(np.array([0, 0]), self.target_radius, COLORS['green'], False)
            self.target2        = Circle(np.array([0, 0]), self.target_radius, COLORS['green'], False)
            self.target1_sector = Sector(np.array([0, 0]), 2*self.target_radius, [0, 0], COLORS['white'], False)
            self.target2_sector = Sector(np.array([0, 0]), 2*self.target_radius, [0, 0], COLORS['white'], False)

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
        self.plant_pos[:] = self.plant.get_pos()
        self.plant_vel[:] = self.plant.get_vel()
        self.move_plant()

        self.update_plant_display()

        self.task_data['plant_pos']    = self.plant_pos.values
        self.task_data['plant_vel']    = self.plant_vel.values
        self.task_data['target_pos']   = self.target_pos.values
        self.task_data['target_index'] = self.target_index

        super(PlantControlBase, self)._cycle()


    #### TEST FUNCTIONS ####

    # helper function
    def armassist_inside_target(self):
        d = dist(self.plant_pos[['aa_px', 'aa_py']], self.target_pos[['aa_px', 'aa_py']])
        inside_target = d <= self.target_radius
        
        target_psi = self.target_pos['aa_ppsi']
        inside_angular_target = angle_inside_range(self.plant_pos['aa_ppsi'],
                                                   target_psi - self.angular_range,
                                                   target_psi + self.angular_range)

        return inside_target and inside_angular_target

    # helper function
    def rehand_inside_target(self):
        for state in rh_pos_states:
            angle = self.plant_pos[state]
            target_angle = self.target_pos[state]
            if not angle_inside_range(angle,
                                      target_angle - self.angular_range,
                                      target_angle + self.angular_range):
                return False

        return True  # must be True if we reached here

    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause

    def _test_enter_target(self, ts):
        if self.plant_type == 'ArmAssist':
            return self.armassist_inside_target()
        elif self.plant_type == 'ReHand':
            return self.rehand_inside_target()
        elif self.plant_type == 'IsMore':
            return self.armassist_inside_target() and self.rehand_inside_target()
        
    def _test_leave_early(self, ts):
        return not self._test_enter_target(ts)

    def _test_hold_complete(self, ts):
        return ts >= self.hold_time

    def _test_timeout(self, ts):
        return ts > self.timeout_time

    def _test_timeout_penalty_end(self, ts):
        return ts > self.timeout_penalty_time

    def _test_hold_penalty_end(self, ts):
        return ts > self.hold_penalty_time

    def _test_trial_complete(self, ts):
        return self.target_index == self.chain_length - 1

    def _test_trial_incomplete(self, ts):
        return (not self._test_trial_complete(ts)) and (self.n_attempts < self.max_attempts)

    def _test_trial_abort(self, ts):
        return (not self._test_trial_complete(ts)) and (self.n_attempts == self.max_attempts)

    def _test_reward_end(self, ts):
        return ts > self.reward_time

    def _test_stop(self, ts):
        if self.session_time > 0 and (time.time() - self.task_start_time) > self.session_time:
            self.end_task()
        return self.stop


    #### STATE FUNCTIONS ####

    def _start_wait(self):
        super(PlantControlBase, self)._start_wait()
        self.n_attempts = 0
        self.target_index = -1
        
        self.hide_targets()

        # get target locations for this trial
        self.targets = self.next_trial

        # number of sequential targets in a single trial
        self.chain_length = self.targets.shape[0] 

    def _start_target(self):
        self.target_index += 1
        self.target_pos[:] = self.targets[self.target_index]
        
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
            
            target.center_pos        = np.array([x, y])
            target.visible           = True
            target_sector.center_pos = np.array([x, y])
            target_sector.ang_range  = [psi-self.angular_range, psi+self.angular_range]
            target_sector.visible    = True

        if self.plant_type in ['ReHand', 'IsMore']:
            for state in rh_pos_states:
                sector = self.rh_sectors[state]
                target_angle = self.target_pos[state]
                sector.ang_range = [target_angle-self.angular_range, target_angle+self.angular_range]
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
                target_sector.ang_range  = [psi-self.angular_range, psi+self.angular_range]
                target_sector.visible    = True

        # unlike ArmAssist target circles, we only have one set of ReHand 
        # target sectors objects, so we can't display next ReHand targets
    
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
        self.target_index = -1

    def _start_hold_penalty(self):
        self.start_penalty()

    def _start_timeout_penalty(self):
        self.start_penalty()

    def _start_targ_transition(self):
        self.hide_targets()

    def _start_reward(self):
        super(PlantControlBase, self)._start_reward()

        if self.plant_type in ['ArmAssist', 'IsMore']:    
            if self.target_index % 2 == 0:
                self.target1.visible        = True
                self.target1_sector.visible = True
            else:
                self.target2.visible        = True
                self.target2_sector.visible = True

    @staticmethod
    def armassist_simple(length=10):
        '''Create an array of pairs of 3-dim (x, y, psi) targets, with the first 
        target in every pair always at the same starting position.

        Parameters
        ----------
        length : int
            The number of target pairs in the sequence.

        Returns
        -------
        pairs : [length*ntargets x 2 x 3] array of pairs of target locations
        '''

        
        aa_pos_states = ['aa_px', 'aa_py', 'aa_ppsi']
        start = settings.starting_pos[aa_pos_states].values

        target_offsets = []
        target_offsets.append(np.array([ 10., 10., 40.]))
        target_offsets.append(np.array([-10., 10., 20.]))
        target_offsets.append(np.array([  0., 10.,  0.]))
        for offset in target_offsets:
            offset[2] *= deg_to_rad

        targets = [start+offset for offset in target_offsets]

        ntargets = len(targets)
        pairs = np.zeros([length * ntargets, 2, 3])
        for block in range(length):
            for i, target in enumerate(targets):
                pairs[ntargets*block + i, 0, :] = start
                pairs[ntargets*block + i, 1, :] = target

        return pairs

    @staticmethod
    def rehand_simple(length=10):
        '''Create an array of pairs of 4-dim (thumb, index, fing3, prono) angular 
        targets, with the first target in every pair always at the same starting
        position.

        Parameters
        ----------
        length : int
            The number of target pairs in the sequence.

        Returns
        -------
        pairs : [length*ntargets x 2 x 4] array of pairs of target locations
        '''

        
        rh_pos_states = ['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono']
        start = settings.starting_pos[rh_pos_states].values

        target_offsets = []
        target_offsets.append(np.array([40., 40., 40., 8.]))
        for offset in target_offsets:
            offset *= deg_to_rad

        targets = [start+offset for offset in target_offsets]

        ntargets = len(targets)
        pairs = np.zeros([length * ntargets, 2, 4])
        for block in range(length):
            for i, target in enumerate(targets):
                pairs[ntargets*block + i, 0, :] = start
                pairs[ntargets*block + i, 1, :] = target

        return pairs

    @staticmethod
    def ismore_simple(length=10):
        '''
        Create an array of pairs of 7-dim (x, y, psi, thumb, index, fing3, prono)
        targets, with the first target in every pair always at the same starting
        position. Intended for use with full ArmAssist+ReHand.

        Parameters
        ----------
        length : int
            The number of target pairs in the sequence.

        Returns
        -------
        pairs : [length*ntargets x 2 x 7] array of pairs of target locations
        '''

        
        start = settings.starting_pos.values
        print 'start', start

        target_offsets = []
        target_offsets.append(np.array([ 10., 10., 40., 20., 20., 20., 20.]))
        target_offsets.append(np.array([-10., 10., 20., 20., 20., 20., 20.]))
        target_offsets.append(np.array([  0., 10.,  0., 20., 20., 20., 20.]))
        for offset in target_offsets:
            offset[2:7] *= deg_to_rad

        targets = [start+offset for offset in target_offsets]

        ntargets = len(targets)
        pairs = np.zeros([length * ntargets, 2, 7])
        for block in range(length):
            for i, target in enumerate(targets):
                pairs[ntargets*block + i, 0, :] = start
                pairs[ntargets*block + i, 1, :] = target

        return pairs


class ManualControl(PlantControlBase):
    '''Allow the subject to manually move the plant to targets.'''
        
    is_bmi_seed = True

    def move_plant(self):
        '''Do nothing here -- plant is moved manually.'''
        pass


class VisualFeedback(PlantControlBase):
    '''Moves the plant automatically to targets using an assister.'''

    is_bmi_seed = True

    def __init__(self, *args, **kwargs):
        super(VisualFeedback, self).__init__(*args, **kwargs)

        assister_kwargs = {
            'call_rate': self.fps,
            'xy_cutoff': self.target_radius,
        }
        if USE_LFC_ASSISTER:
            self.assister = ismore_bmi_lib.LFC_ASSISTER_CLS_DICT[self.plant_type](**assister_kwargs)
        else:
            self.assister = ismore_bmi_lib.ASSISTER_CLS_DICT[self.plant_type](**assister_kwargs)
        
        self.goal_calculator = ismore_bmi_lib.GOAL_CALCULATOR_CLS_DICT[self.plant_type](self.ssm)

    def move_plant(self):
        current_state = np.hstack([self.plant_pos.values, self.plant_vel.values, 1])[:, None]

        data, solution_updated = self.goal_calculator(self.target_pos.values)
        target_state = data[0].reshape(-1, 1)

        # use an assister with assist=100% to generate visual feedback kinematics
        vfb_kin = pd.Series(np.squeeze(np.array(self.assister(current_state, target_state, 1)[0])),
                            self.ssm_states)

        command_vel = vfb_kin[self.vel_states]

        self.plant.send_vel(command_vel)
        self.task_data['command_vel'] = command_vel


class BMIControl(BMILoop, LinearlyDecreasingAssist, PlantControlBase):
    '''Target capture task with plant controlled by BMI output.
    Cursor movement can be assisted toward target by setting assist_level > 0.
    '''

    max_attempts = traits.Int(3, desc='Max attempts allowed to a target before skipping to the next one')
    
    ordered_traits = ['session_time', 'assist_level', 'assist_level_time', 'reward_time']

    # overrides BMILoop.init_decoder_state, without calling it
    def init_decoder_state(self):
        self.decoder.filt._init_state()
        self.decoder['q'] = self.starting_pos[self.pos_states].values
        self.init_decoder_mean = self.decoder.filt.state.mean

        self.decoder.set_call_rate(self.fps)

    def create_assister(self):
        kwargs = {
            'call_rate': self.fps,
            'xy_cutoff': self.target_radius,
        }
        if USE_LFC_ASSISTER:
            self.assister = ismore_bmi_lib.LFC_ASSISTER_CLS_DICT[self.plant_type](**kwargs)
        else:
            self.assister = ismore_bmi_lib.ASSISTER_CLS_DICT[self.plant_type](**kwargs)

    def create_goal_calculator(self):
        self.goal_calculator = ismore_bmi_lib.GOAL_CALCULATOR_CLS_DICT[self.plant_type](self.decoder.ssm)

    def move_plant(self, **decoder_kwargs):
        feature_data = self.get_features()
        for key, val in feature_data.items():
            self.task_data[key] = val

        neural_features = feature_data[self.extractor.feature_type]

        # TODO -- set self.decoder's velocities to self.plant's velocities?

        # determine the target state of the current sub-trial if CLDA and/or assist is on
        if self.current_assist_level > 0 or self.learn_flag:
            target_state = self.get_target_BMI_state(self.decoder.states)
            self.task_data['target_state'] = target_state            
        else:
            target_state = np.zeros([self.decoder.n_states, self.decoder.n_subbins])

        # want to use as current of a position as possible for assister
        # assister only needs position to determine Bu, so no need to get/use
        #   plant's current velocity
        self.plant_pos[:] = self.plant.get_pos()

        current_state = np.array(self.decoder.filt.state.mean).ravel()
        current_state = pd.Series(current_state, self.ssm_states)
        current_state[self.pos_states] = self.plant_pos[self.pos_states]
        current_state = current_state.values.reshape(-1, 1)

        # if assist is at 0%, assister will return (None, 0)
        Bu, assist_weight = self.assister(current_state, target_state, self.current_assist_level, mode=self.state)

        # run the decoder
        if self.state not in self.static_states:
            decoder_kwargs['weighted_avg_lfc'] = USE_LFC_ASSISTER
            self.call_decoder(neural_features, target_state, Bu=Bu, assist_level=assist_weight, feature_type=self.extractor.feature_type, **decoder_kwargs)

        # reset position in decoder state vector based on feedback from plant
        # this way, if move_plant is called at 10 Hz and if learner's 
        #  input_state_index is -1, then on next iteration when self.call_decoder()
        #  is called, learner will use the ArmAssist's true position from 100 ms ago
        self.plant_pos[:] = self.plant.get_pos()
        self.decoder['q'] = self.plant_pos.values

        # bound the decoder state based on the endpoint constraints
        # TODO -- keep this line?
        self.decoder.bound_state()

        command_vel = self.decoder[self.vel_states]
        self.plant.send_vel(command_vel)
        self.task_data['command_vel'] = command_vel

        decoder_state = self.decoder.get_state(shape=(-1, 1))
        self.task_data['decoder_state'] = decoder_state

        return decoder_state

    def _cycle(self):
        self.task_data['loop_time'] = self.iter_time()
        # loop_time = self.iter_time()
        # self.task_data['loop_time'] = loop_time
        # print 'printing loop time in BMIControl._cycle:', loop_time
        super(BMIControl, self)._cycle()

    def get_target_BMI_state(self, *args):
        '''Run the goal calculator to determine the current target state.'''
        data, solution_updated = self.goal_calculator(self.target_pos.values)
        target_state = data[0]

        return np.tile(np.array(target_state).reshape(-1, 1), [1, self.decoder.n_subbins])

    def _end_timeout_penalty(self):
        if self.reset:
            self.decoder.filt.state.mean = self.init_decoder_mean
            self.hdf.sendMsg("reset")

    def cleanup_hdf(self):
        super(BMIControl, self).cleanup_hdf()
        self.decoder.save_attrs(self.h5file.name, 'task')

clda_intention_est_methods = ['OFC', 'simple']
class CLDAControl(BMIControl):
    '''
    BMI task that periodically refits the decoder parameters based on intended
    movements toward the targets.
    '''

    batch_time           = traits.Float(0.1, desc='The length of the batch in seconds')
    half_life            = traits.Tuple((20., 20.), desc='Half life of the adaptation in seconds')
    decoder_sequence     = traits.String('test', desc='signifier to group together sequences of decoders')
    half_life_decay_time = traits.Float(900.0, desc='Time to go from initial half life to final')

    clda_update_method_options = clda_update_methods
    clda_update_method = traits.Enum(*clda_update_methods)
    clda_intention_est_method_options = clda_intention_est_methods
    clda_intention_est_method = traits.Enum(*clda_intention_est_methods)    

    ordered_traits = ['session_time', 'assist_level', 'assist_level_time', 'batch_time', 'half_life', 'half_life_decay_time']

    def __init__(self, *args, **kwargs):
        super(CLDAControl, self).__init__(*args, **kwargs)
        self.learn_flag = True

    def init(self):
        '''
        Secondary init function. Decoder has already been created by inclusion
        of the 'bmi' feature in the task. 
        '''

        # self.load_decoder() will be called again in the call to super(...).init() (in BMILoop.init)
        # but there is no way around this
        self.load_decoder()
        self.batch_size = int(self.batch_time / self.decoder.binlen)

        self.add_dtype('half_life', 'f8', (1,))


        ## TODO -- remove creation of ref_learner below
        # this is just for testing purposes (e.g., comparing an OFC
        # learner to a non-OFC learner)

        # the simple, non-OFC learners just create/use an assister object
        assister_kwargs = {
            'call_rate': self.fps,
            'xy_cutoff': self.target_radius,
        }
        ref_learner = ismore_bmi_lib.LEARNER_CLS_DICT[self.plant_type](self.batch_size, **assister_kwargs)
        self.bmi_system.ref_learner = ref_learner

    def create_learner(self):
        if self.clda_intention_est_method == 'OFC':
            self.learner = ismore_bmi_lib.OFC_LEARNER_CLS_DICT[self.plant_type](self.batch_size)
        elif self.clda_intention_est_method == 'simple':
            # the simple, non-OFC learners just create/use an assister object
            assister_kwargs = {
                'call_rate': self.fps,
                'xy_cutoff': self.target_radius,
            }
            self.learner = ismore_bmi_lib.LEARNER_CLS_DICT[self.plant_type](self.batch_size, **assister_kwargs)
        else:
            NotImplementedError("Unrecognized CLDA intention estimation method: %s" % self.clda_intention_est_method)
        
        self.learn_flag = True

    def create_updater(self):
        if self.clda_update_method == 'RML':
            self.updater = clda.KFRML(self.batch_time, self.half_life[0])
        elif self.clda_update_method == 'Smoothbatch':
            half_life_start, half_life_end = self.half_life
            self.updater = clda.KFSmoothbatch(self.batch_time, half_life_start)
        else:
            raise NotImplementedError("Unrecognized CLDA update method: %s" % self.clda_update_method)


########################
## simulation classes ##
########################


class FakeHDF(object):
    def __init__(self):
        self.msgs = []

    def sendMsg(self, msg):
        self.msgs.append(msg)


class SimHDF(object):
    '''
    An interface-compatbile HDF for simulations which do not require saving an
    HDF file
    '''
    def __init__(self, *args, **kwargs):
        from collections import defaultdict
        self.data = defaultdict(list)
        self.msgs = []        
        self.hdf = FakeHDF()
        from riglib import sink
        self.sinks = sink.sinks
        super(SimHDF, self).__init__(*args, **kwargs)

    def sendMsg(self, msg):
        self.msgs.append((msg, -1))



class KalmanEncoder:
    '''Models a BMI user as someone who, given an intended state x,
    generates a vector of neural features y according to the KF observation
    model equation: y = Cx + q.'''

    def __init__(self, ssm, n_features):
        self.ssm = ssm
        self.n_features = n_features

        drives_neurons = ssm.drives_obs
        nX = ssm.n_states

        C = np.random.standard_normal([n_features, nX])
        C[:, ~drives_neurons] = 0
        Q = np.identity(n_features)

        self.C = C
        self.Q = Q

    def __call__(self, intended_state):
        q = np.random.multivariate_normal(np.zeros(self.Q.shape[0]), self.Q).reshape(-1, 1)
        neural_features = np.dot(self.C, intended_state)# + q

        return neural_features


class SimBMIControl(BMIControl):
    def init(self):
        self._init_neural_encoder()
        super(SimBMIControl, self).init()
        self.wait_time = 0
        self.pause = False

        self.intended_state_generator = ismore_bmi_lib.ASSISTER_CLS_DICT[self.plant_type]()
        
    def create_feature_extractor(self):
        # if using a KalmanEncoder, this isn't actually used; only reason
        # that this is here is because the task expects the variable
        # "self.extractor.feature_type" to exist
        self.extractor = extractor.BinnedSpikeCountsExtractor(None, units=self.decoder.units)  

        if isinstance(self.extractor.feature_dtype, tuple):
            self.add_dtype(*self.extractor.feature_dtype)
            print 'self.extractor.feature_dtype:', self.extractor.feature_dtype
        else:
            for x in self.extractor.feature_dtype:
                print 'x:', x
                self.add_dtype(*x)

    def get_features(self):
        target_state = self.get_target_BMI_state(self.decoder.states)
        current_state = self.decoder.filt.state.mean
        intended_state = self.intended_state_generator(current_state, 
                                                       target_state, 
                                                       1, 
                                                       mode=self.state)[0]
        print("Yes we go here SimBMIControl")
        neural_features = self.encoder(intended_state)

        return dict(spike_counts=neural_features)

    def _test_penalty_end(self, ts):
        # no penalty when using simulated neurons
        return True

    def _init_neural_encoder(self):
        ssm = ismore_bmi_lib.SSM_CLS_DICT[self.plant_type]()
        
        n_features = 10
        self.encoder = KalmanEncoder(ssm, n_features)


class SimCLDAControl(SimTime, SimHDF, SimBMIControl, CLDAControl):
    assist_level = (1., 1.)
    rand_start = (0., 0.)
    def __init__(self, *args, **kwargs):
        super(SimCLDAControl, self).__init__(*args, **kwargs)
        if self.clda_update_method == 'RML':
            self.batch_time = 0.1
        else:
            self.batch_time = 10
        self.half_life  = 20.0, 20.0
        self.assist_level_time = 30.

    def load_decoder(self):
        ssm = self.encoder.ssm
        units = np.array([(i+1, 0) for i in range(self.encoder.n_features)])  # fake unit numbers
        self.decoder = train._train_KFDecoder_2D_sim_2(ssm, units)

    def _cycle(self):
        super(SimCLDAControl, self)._cycle()
        elapsed_time = self.get_time() - self.task_start_time
        if elapsed_time > 30 and self.learn_flag:
            self.disable_clda()
            print '*' * 80


class SimRecordB1(SimTime, SimHDF, RecordB1):
    pass

class SimVisualFeedback(SimTime, SimHDF, VisualFeedback):
    pass


if __name__ == '__main__':

    SIMPLE_GENFNS_DICT = {
        'ArmAssist': PlantControlBase.armassist_simple,
        'ReHand':    PlantControlBase.rehand_simple,
        'IsMore':    PlantControlBase.ismore_simple,
    }

    task_cls = SimCLDAControl
    # task_cls = SimVisualFeedback
    # task_cls = SimRecordTrajectories

    seq = SIMPLE_GENFNS_DICT['IsMore']()     # actual sequence of targets

    task = task_cls(seq)
    task.init()
    task.run()
