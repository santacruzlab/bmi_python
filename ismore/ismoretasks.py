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
import copy
# from django.db import models
# from db.tracker import TaskEntry, Task 
from riglib.experiment import traits, Sequence, generate, FSMTable, StateTransitions
from riglib.stereo_opengl.window import WindowDispl2D, FakeWindow
from riglib.stereo_opengl.primitives import Circle, Sector, Line
from riglib.bmi import clda, extractor, train
from riglib.bmi.bmi import Decoder, BMISystem, GaussianStateHMM, BMILoop, GaussianState, MachineOnlyFilter

from ismore import plants, settings, ismore_bmi_lib
from ismore.common_state_lists import *
from features.bmi_task_features import LinearlyDecreasingAssist, LinearlyDecreasingHalfLife
from features.simulation_features import SimTime, SimHDF
from ismore.brainamp import rda
from utils.angle_utils import *
from utils.util_fns import *
from utils.constants import *

# from db.tracker import models
from utils.ringbuffer import RingBuffer

from features.generator_features import Autostart

import pygame
from riglib.plants import RefTrajectories
from ismore.filter import Filter
from scipy.signal import butter,lfilter
import brainamp_channel_lists
from utils.constants import *
#import playsound

np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)

####################################   DEFINITIONS   ---------------------------------------------- 

###### Colors ######
COLORS = {
    'black': (0, 0, 0, 1),
    'red':   (1, 0, 0, 1),
    'red_grasp':   (1, 0, 0, 1),
    'grasp':   (1, 0, 0, 1),
    'pinch':   (1, 0, 0, 1),
    'green': (0, 1, 0, 1),
    'green_point': (0, 1, 0, 1),
    'point': (0, 1, 0, 1),
    'blue':  (0, 0, 1, 1),
    'blue_up':  (0, 0, 1, 1),
    'up':  (0, 0, 1, 1),
    'rest': (1, 1, 1, 1),
    'white': (1, 1, 1, 1),
    'magenta': (0, 1, 0, 0), 
    'brown': (29, 74, 100, 24),
    'yellow': (0, 0, 1, 0),
    'down':   (1, 0, 0, 1),
    'linear_red':   (1, 0, 0, 1),
    'circular':  (0, 0, 1, 1),
    'wrist_ext': (1, 0, 0, 1),

}

###### Options to select in interface ######
plant_type_options  = ['IsMore','ArmAssist', 'ReHand', 'DummyPlant', 'IsMorePlantHybridBMISoftSafety']
DoF_control_options = ['IsMore','ArmAssist', 'ReHand', 'ReHand-Pronosup', 'ReHand-Pronosup-FingersDisabled']
DoF_target_options  = ['IsMore','ArmAssist', 'ReHand', 'ReHand-Pronosup', 'ReHand-Pronosup-FingersDisabled']
arm_side_options = ['left','right']
clda_update_methods = ['RML', 'Smoothbatch', ]
languages_list = ['english', 'deutsch', 'castellano', 'euskara']
speed_options = ['very-low','low', 'medium','high']
fb_input_options = ['standard', 'andreita']
channel_list_options = brainamp_channel_lists.channel_list_options

#----------------------------------------------   DEFINITIONS   ###############################################

################################################   FUNCTIONS   ################################################

def check_plant_and_DoFs(plant_type, DoF_control, DoF_target):
    '''
    Function to check if the connected plant_type and the selected DoFs for control and target accomplishment are compatible.
    Output: the indexes of the selected DoFs depending on the connected plant_type.
    '''
    plant_and_DoFs_correct = True
    #check if the selected DoF_control is possible with the selected plant_type
    if plant_type in ['ArmAssist', 'ReHand']:
        if DoF_control.startswith(plant_type) == False:
            plant_and_DoFs_correct = False
            print "DoF_control selected not possible for the selected plant_type"
            
    #check if the selected DoF_target is possible with the selected DoF_control
    if DoF_control != 'IsMore':
        if DoF_target.startswith(DoF_control) == False:
            plant_and_DoFs_correct = False
            print "DoF_target selected not possible for the selected DoF_control"

    if plant_and_DoFs_correct == True:
        # define DoF target indexes for each case
        if DoF_target == 'ArmAssist' and plant_type in ['ArmAssist', 'IsMore']:
            DoF_target_idx_init = 0
            DoF_target_idx_end  = 3 
        elif DoF_target == 'ReHand' and plant_type == 'IsMore':
            DoF_target_idx_init = 3
            DoF_target_idx_end  = 7
        elif DoF_target in ['ReHand-Pronosup', 'ReHand-Pronosup-FingersDisabled'] and plant_type == 'IsMore':
            DoF_target_idx_init = 6
            DoF_target_idx_end  = 7
        elif DoF_target == 'ReHand' and plant_type == 'ReHand':
            DoF_target_idx_init = 0
            DoF_target_idx_end  = 3
        elif DoF_target in ['ReHand-Pronosup', 'ReHand-Pronosup-FingersDisabled'] and plant_type == 'ReHand':
            DoF_target_idx_init = 3
            DoF_target_idx_end  = 4
        elif DoF_target == 'IsMore' and plant_type == 'IsMore':
            DoF_target_idx_init = 0
            DoF_target_idx_end  = 7

        # define DoF control indexes for each case
        if DoF_control == 'ArmAssist' and plant_type =='IsMore':
            DoF_not_control_idx_init = 3
            DoF_not_control_idx_end  = 7
        elif DoF_control == 'ArmAssist' and plant_type == 'ArmAssist':
            DoF_not_control_idx_init = np.nan
            DoF_not_control_idx_end  = np.nan
        elif DoF_control == 'ReHand' and plant_type == 'IsMore':
            DoF_not_control_idx_init = 0
            DoF_not_control_idx_end  = 3
        elif DoF_control in ['ReHand-Pronosup', 'ReHand-Pronosup-FingersDisabled'] and plant_type == 'IsMore':
            DoF_not_control_idx_init = 0 
            DoF_not_control_idx_end  = 6
        elif DoF_control == 'ReHand' and plant_type == 'ReHand':
            DoF_not_control_idx_init = np.nan
            DoF_not_control_idx_end  = np.nan
        elif DoF_control in ['ReHand-Pronosup', 'ReHand-Pronosup-FingersDisabled'] and plant_type == 'ReHand':
            DoF_not_control_idx_init = 0
            DoF_not_control_idx_end  = 3
        elif DoF_control == 'IsMore' and plant_type == 'IsMore':
            DoF_not_control_idx_init = np.nan
            DoF_not_control_idx_end  = np.nan
    else:
        print "ERROR!!! Plant and selected target or control DoFs incorrect!!!"

    return [DoF_target_idx_init,DoF_target_idx_end, DoF_not_control_idx_init,DoF_not_control_idx_end]


##############################################  BASIC CLASSES   ################################################

class IsMoreBase(WindowDispl2D):
    '''
    A base class for all IsMore tasks. Creates the appropriate plant object
    and updates the display of the plant at every iteration of the task.
    '''

    window_size = traits.Tuple((500, 281), desc='Size of window to display the plant position/angle')
    # window_size = traits.Tuple((1920, 1080), desc='Size of window to display the plant position/angle')
    starting_pos = settings.starting_pos
    update_rest = True
    plant_type = traits.OptionsList(*plant_type_options, bmi3d_input_options=plant_type_options, desc='Device connected, data will be acquired from this plant')
    #simulate = traits.Bool(False, desc='Use simulation "plant" without UDP communication')
    arm_side = traits.OptionsList(*arm_side_options, bmi3d_input_options=arm_side_options, desc='arm side wearing the exo')
    show_FB_window = traits.OptionsList(*fb_input_options, bmi3d_input_options=fb_input_options, desc='')

    exclude_parent_traits = ["show_environment"]
    
    def __init__(self, *args, **kwargs):
        super(IsMoreBase, self).__init__(*args, **kwargs)

        self.ssm = ismore_bmi_lib.SSM_CLS_DICT[self.plant_type]()
        self.ssm_states = [s.name for s in self.ssm.states]
        self.pos_states = [s.name for s in self.ssm.states if s.order == 0]
        self.vel_states = [s.name for s in self.ssm.states if s.order == 1]
        print 'self.vel_states', self.vel_states
        if 0: #self.simulate:
            # use locally running IsMoreBasesimulated ArmAssist and/or ReHand
            #   for which we can magically set the initial position
            self.plant = plants.NONUDP_PLANT_CLS_DICT[self.plant_type]()
            self.plant.set_pos(self.starting_pos[self.pos_states].values)
        else:
            self.plant = plants.UDP_PLANT_CLS_DICT[self.plant_type]()
                
        print 'self.pos_states', self.pos_states
        print 'plant_type', self.plant_type

        self.plant_pos_raw = pd.Series(self.plant.get_pos_raw(), self.pos_states)
        self.plant_pos = pd.Series(self.plant.get_pos(), self.pos_states)

        self.plant_vel_raw = pd.Series(self.plant.get_vel_raw(), self.vel_states)
        self.plant_vel = pd.Series(self.plant.get_vel(), self.vel_states)

        self.add_dtype('plant_pos', 'f8', (len(self.plant_pos_raw),))
        #self.add_dtype('plant_pos_filt', 'f8', (len(self.plant_pos),))

        self.add_dtype('plant_vel', 'f8', (len(self.plant_vel_raw),))
        #self.add_dtype('plant_vel_filt', 'f8', (len(self.plant_vel),))
        
        self.add_dtype('plant_type',   np.str_, 40)
        self.add_dtype('ts',         'f8',    (1,))
        # self.add_dtype('DoF_control',   np.str_, 40)
        # self.add_dtype('DoF_target',   np.str_, 40)
        
        self.init_plant_display()
        self.update_plant_display()
        pygame.mixer.init()
        
        #if a targets_matrix is being used in the task, show the target positions in the display window
        if 'targets_matrix' in locals()['kwargs']:
            self.display_targets()    
        else: 
            print 'no targets matrix'

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

    def _play_sound(self, fpath, fname):
        print 'play sound: ', fname
        if hasattr(self, 'replace_ya_w_pausa'):
            if self.replace_ya_w_pausa == 'Yes':
                if fname[0] == 'go':
                    fname = ['rest']

        for filename in fname:
            # print 'filename ', filename
            if filename == 'circular':
                filename = 'circular_big'
                sound_fname = os.path.join(fpath, filename + '.wav')
                pygame.mixer.music.load(sound_fname)
                pygame.mixer.music.play()                  
            
            elif '_' in filename or ' ' in filename:
                # First see if there's a file with exact name: 
                if os.path.isfile(os.path.join(fpath, filename + '.wav')):
                    pygame.mixer.music.load(os.path.join(fpath, filename + '.wav'))
                    pygame.mixer.music.play()   
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
                        pygame.mixer.music.load(os.path.join(fpath, filename + '.wav'))
                        pygame.mixer.music.play()
                    else: 
                        #try:
                        # Next try splitting up the names:
                        fi1 = filename.find(key)  
                        filename1 = filename[:fi1]

                        if os.path.isfile(os.path.join(fpath, filename1 + '.wav')):
                            #sound_fname = os.path.join(fpath, filename1 + '.wav')
                            pygame.mixer.music.load(os.path.join(fpath, filename1 + '.wav'))
                            pygame.mixer.music.play()
                            
                            x = 0
                            while pygame.mixer.music.get_busy():
                                x += 1
                            
                            filename2 = filename[filename.find(key)+1:]
                            if os.path.isfile(os.path.join(fpath, filename2 + '.wav')):
                                pygame.mixer.music.load(os.path.join(fpath, filename2 + '.wav'))
                                pygame.mixer.music.play()
                            else:
                                # 3 legged: 
                                fi2 = filename.find(key, fi1+1)
                                filename2 = filename[fi1+1:fi2]
                                filename3 = filename[fi2+1:]

                                sound_fname = os.path.join(fpath, filename2 + '.wav')
                                pygame.mixer.music.load(sound_fname)
                                pygame.mixer.music.play()
                                y = 0
                                while pygame.mixer.music.get_busy():
                                    y+=1
                                sound_fname = os.path.join(fpath, filename3 + '.wav')
                                pygame.mixer.music.load(sound_fname)
                                pygame.mixer.music.play()

                        else:
                            print 'cant play: ', filename
            else:
                sound_fname = os.path.join(fpath, filename + '.wav')
                pygame.mixer.music.load(sound_fname)
                pygame.mixer.music.play()    

    def _cycle(self):
        self.task_data['ts']= time.time()
        self.plant.write_feedback()
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
                    
                    # Run IsMore application automatically in a second terminal
                    # if (self.plant_type == 'ArmAssist' and t_elapsed > 3):
                    #    os.system("gnome-terminal -e 'bash -c \"cd /home/tecnalia/code/armassist && sudo /home/tecnalia/code/armassist/IsMore; exec bash\"'")


        elif 'IsMore' in self.plant_type:
            for plant_type in ['ArmAssist', 'ReHand']:
                if time_since_started > n_secs:
                    if last_ts_arrival[plant_type] == 0:
                        print 'No %s data has arrived at all' % plant_type
                    else:
                        t_elapsed = time.time() - last_ts_arrival[plant_type]
                        if t_elapsed > n_secs:
                            print 'No %s data in the last %.1f s' % (plant_type, t_elapsed)

                        # Run IsMore application automatically in a second terminal
                        # if (plant_type == 'ArmAssist' and t_elapsed > 3):
                        #    os.system("gnome-terminal -e 'bash -c \"cd /home/tecnalia/code/armassist && sudo /home/tecnalia/code/armassist/IsMore; exec bash\"'")

        else:
            raise Exception('Unrecognized plant type!')

    def init_plant_display(self):
        if self.plant_type in ['ArmAssist', 'IsMore']:
            # use circle & line to represent ArmAssist's xy position & orientation
            self.xy_cursor = Circle(np.array([0, 0]), 1, COLORS['white'])
            self.psi_line  = Line(np.array([0, 0]), 10, 1, 0, COLORS['white'])

            self.add_model(self.xy_cursor)
            self.add_model(self.psi_line)


        if self.plant_type in ['ReHand', 'IsMore']:
            # use (rotating) lines to represent ReHand's angles
            px = self.starting_pos['aa_px']
            self.rh_angle_line_positions = {
                'rh_pthumb': np.array([px - 30, 30]),
                'rh_pindex': np.array([px - 10,  30]),
                'rh_pfing3': np.array([px + 10,  30]),
                'rh_pprono': np.array([px + 30, 30]),
            }
            
            self.rh_angle_lines = {}

            for state in rh_pos_states:
                l = Line(self.rh_angle_line_positions[state], 13, 1, 5, COLORS['white'])
                self.rh_angle_lines[state] = l 
                self.add_model(l)

    def update_plant_display(self):
        # if (self.update_rest == True) and (any(self.plant_pos[aa_xy_states].values > 0.1)):
        #     if self.plant_type in ['ArmAssist', 'IsMore']:
        #             self.xy_cursor_rest.center_pos = self.plant_pos[aa_xy_states].values
        #             self.psi_line_rest.start_pos   = self.plant_pos[aa_xy_states].values
        #             self.psi_line_rest.angle       = self.plant_pos['aa_ppsi']+ 90*deg_to_rad #to show it in a more intuitive way according to the forearm position
                    
        #     if self.plant_type in ['ReHand', 'IsMore']:
        #         for state in rh_pos_states:
        #             self.rh_angle_lines_rest[state].angle = self.plant_pos[state]

        #     self.update_rest = False

        if self.plant_type in ['ArmAssist', 'IsMore']:

            self.xy_cursor.center_pos = self.plant_pos[aa_xy_states].values
            self.psi_line.start_pos   = self.plant_pos[aa_xy_states].values
            self.psi_line.angle       = self.plant_pos['aa_ppsi']+ 90*deg_to_rad #to show it in a more intuitive way according to the forearm position

        if self.plant_type in ['ReHand', 'IsMore']:

            for state in rh_pos_states:
                self.rh_angle_lines[state].angle = self.plant_pos[state]

    def display_targets(self):

        # colors = ['magenta', 'blue', 'yellow', 'green', 'red' , 'blue', 'yellow', 'green', 'red'] 
        colors = ['white', 'red', 'green', 'blue' , 'red', 'green', 'blue', 'red', 'green', 'magenta', 'blue','white', 'red', 'green', 'blue' ] 
        
        targets = self.targets_matrix.keys()
        if 'subgoal_names' in targets:
            targets.remove('subgoal_names')

        num_trial_types = len(targets)

        if self.plant_type in ['ArmAssist', 'IsMore']:

            # positions of each target
            self.aa_xy_cursor_targets_positions = OrderedDict()
            self.aa_psi_line_targets_angles = OrderedDict()

            for trial_type_index in range(num_trial_types):
                
                trial_type = targets[trial_type_index]

                num_targets = len(self.targets_matrix[targets[trial_type_index]].keys())

                # print 'self.targets_matrix', self.targets_matrix
                # print 'targets ', targets
                # print 'trial_type_index ', trial_type_index
                # print "num_targets ", num_targets

                for target_index in range(num_targets):
                # for target_index in range(1):
                    px = self.targets_matrix[trial_type][target_index][aa_pos_states].aa_px
                    py = self.targets_matrix[trial_type][target_index][aa_pos_states].aa_py
                    ppsi = self.targets_matrix[trial_type][target_index][aa_pos_states].aa_ppsi
                    self.aa_xy_cursor_targets_positions[target_index] = np.array([px, py])
                    self.aa_psi_line_targets_angles[target_index] = np.array(ppsi)
                    
                    if trial_type.find(' ') == -1:
                        target_name = trial_type
                        if trial_type.find('_') != -1:
                            target_name = target_name[:target_name.find('_')]
                    else:
                        target_name = trial_type[:trial_type.find(' ')]
                        if trial_type.find('_') != -1:
                            target_name = target_name[:target_name.find('_')]

                    self.aa_xy_targets_cursors = {}
                    self.aa_psi_targets_lines = {}
            
                    # color_target = COLORS[colors[trial_type_index]] 
                    # color_target = COLORS[target_name] #andrea
                    color_target = COLORS[colors[target_index]] 

                    xy_cursor = Circle(self.aa_xy_cursor_targets_positions[target_index], 1, color_target)
                    self.aa_xy_targets_cursors[target_index] = xy_cursor 
                    self.add_model(xy_cursor)

                    psi_line = Line(self.aa_xy_cursor_targets_positions[target_index], 10, 1, 0, color_target)
                    self.aa_psi_targets_lines[target_index] = psi_line 
                    self.add_model(psi_line)

                    self.aa_psi_targets_lines[target_index].angle =  self.aa_psi_line_targets_angles[target_index] + 90*deg_to_rad
         
        if self.plant_type in ['ReHand', 'IsMore']:

            # use (rotating) lines to represent ReHand's angles
            px = self.starting_pos['aa_px']
            self.rh_angle_line_positions = {
                'rh_pthumb': np.array([px - 30, 30]),
                'rh_pindex': np.array([px - 10,  30]),
                'rh_pfing3': np.array([px + 10,  30]),
                'rh_pprono': np.array([px + 30, 30]),
            }

            self.rh_angle_lines_targets = OrderedDict()

            for trial_type_index in range(num_trial_types):

                trial_type = targets[trial_type_index]
                num_targets = len(self.targets_matrix[targets[trial_type_index]].keys())

                for target_index in range(num_targets):
                    
                    self.rh_angle_lines_targets[target_index]= self.targets_matrix[trial_type][target_index][rh_pos_states]
                    
                    if num_targets == 1:
                        color_target = COLORS[colors[trial_type_index]]
                    else:
                        color_target = COLORS[colors[target_index]]

                    for state in rh_pos_states:
                        rh_line = Line(self.rh_angle_line_positions[state], 10, 0.5, 5, color_target)
                        self.rh_angle_lines_targets[state] = rh_line
                        self.add_model(rh_line)
                        self.rh_angle_lines_targets[state].angle =  self.rh_angle_lines_targets[target_index][state]

    def define_safety(self):

            safety_states_plant_min_max = safety_states_min_max[self.plant_type]
            self.safety_margin = pd.Series(np.array([3, 3, np.deg2rad(20),np.deg2rad(20), np.deg2rad(10),  np.deg2rad(10), np.deg2rad(10), np.deg2rad(10),np.deg2rad(10),  np.deg2rad(10), np.deg2rad(10), np.deg2rad(10)]), safety_states_min_max['IsMore'])

            #self.safety_margin = pd.Series(np.array([3, 3, -np.deg2rad(20),np.deg2rad(20), -np.deg2rad(10),  -np.deg2rad(10), -np.deg2rad(10), -np.deg2rad(10),np.deg2rad(10),  -np.deg2rad(10), -np.deg2rad(10), -np.deg2rad(10)]), safety_states_min_max['IsMore'])


            trial_types_list_all = self.targets_matrix.keys()
            trial_types_list_all.remove('subgoal_names')
            # trial_types_list.remove('rest')

            print "trial_types_list " , trial_types_list_all
            if self.plant_type in ['ArmAssist', 'IsMore']:
                # base -- 2D movement
                x_pos_rest = self.targets_matrix['rest'][0]['aa_px']
                y_pos_rest = self.targets_matrix['rest'][0]['aa_py']

                distXY_rest = []
                psi_values = []

                trial_types_list = trial_types_list_all
                trial_types_list.remove('rest')

                for ind_trial_type, trial_type in enumerate(trial_types_list):
                    # if '_' in trial_type:
                    #     trial_type = trial_type[:trial_type.find('_')]
                    # if ' ' in trial_type:
                    #     trial_type = trial_type[:trial_type.find(' ')]

                    distX_rest = self.targets_matrix[trial_type][0]['aa_px'] - x_pos_rest
                    distY_rest = self.targets_matrix[trial_type][0]['aa_py'] - y_pos_rest

                    distXY_rest.append(np.sqrt(distX_rest**2 + distY_rest**2))

                    psi_values.append(self.targets_matrix[trial_type][0]['aa_ppsi'])


                max_distXY_rest = max(distXY_rest)            
                min_distXY_rest = 0

                # forearm angle
                max_ppsi = max(psi_values)
                min_ppsi = min(psi_values)

                self.safety_area_plant_aa = pd.Series(np.array([min_distXY_rest, max_distXY_rest + self.safety_margin['max_aa_distXY'], min_ppsi - self.safety_margin['min_aa_ppsi'] , max_ppsi + self.safety_margin['max_aa_ppsi']]), safety_states_min_max['ArmAssist'])

            if self.plant_type in ['ReHand', 'IsMore']:
                rh_angles = {}
                max_rh_angles = pd.Series()
                min_rh_angles = pd.Series()
                for rh_state in rh_pos_states:
                    rh_angles[rh_state] = []
                    trial_types_list = trial_types_list_all
                    for ind_trial_type, trial_type in enumerate(trial_types_list):           
                    
                        rh_angles[rh_state].append(self.targets_matrix[trial_type][0][rh_state])

                    min_rh_angles['min_' + rh_state] = min(rh_angles[rh_state]) - self.safety_margin['min_' + rh_state]
                    max_rh_angles['max_' + rh_state] = max(rh_angles[rh_state]) + self.safety_margin['max_' + rh_state]
                    
                self.safety_area_plant_rh = pd.concat([min_rh_angles , max_rh_angles])

            if self.plant_type=='IsMore':
                self.safety_area_plant = pd.concat([self.safety_area_plant_aa,self.safety_area_plant_rh])
            if self.plant_type=='ArmAssist':
                self.safety_area_plant = self.safety_area_plant_aa
            if self.plant_type=='ReHand':
                self.safety_area_plant = self.safety_area_plant_rh

            print "self.safety_area_plant" , self.safety_area_plant

    def display_safety_area(self):
    # use (rotating) lines to represent ReHand's angles
        px = self.starting_pos['aa_px']
        self.rh_angle_line_positions = {
            'rh_pthumb': np.array([px - 30, 30]),
            'rh_pindex': np.array([px - 10,  30]),
            'rh_pfing3': np.array([px + 10,  30]),
            'rh_pprono': np.array([px + 30, 30]),
        }

        self.rh_angle_lines_safety = OrderedDict()
        self.rh_angle_lines_safety_angles = self.safety_area_plant_rh
        # self.rh_angle_lines_safety_angles = self.safety_area_plant_rh[safety_states_min_max['ReHand']]
        print "self.rh_angle_lines_safety_angles ", self.rh_angle_lines_safety_angles
                
        for state in rh_pos_states:
            print "state : ", state
            print "min + state ", ['min_' + state]
            rh_line_min = Line(self.rh_angle_line_positions[state], 10, 0.5, 5, COLORS['white'])
            rh_line_max = Line(self.rh_angle_line_positions[state], 10, 0.5, 5, COLORS['white'])
            self.rh_angle_lines_safety['min_' + state] = rh_line_min
            self.rh_angle_lines_safety['max_' + state] = rh_line_max
            self.add_model(rh_line_min)   
            self.add_model(rh_line_max)            
            self.rh_angle_lines_safety['min_' + state].angle =  self.rh_angle_lines_safety_angles['min_' + state]   
            self.rh_angle_lines_safety['max_' + state].angle =  self.rh_angle_lines_safety_angles['max_' + state]  

    def check_safety(self,command_vel):
        # print "self.safety_area_plant ", self.safety_area_plant
        # print "current pos ", self.plant_pos[:]['aa_px']
        # print "command_vel", command_vel
        
        if self.plant_type in ['ArmAssist', 'IsMore']:
            #aa base
            distX = self.plant_pos[:]['aa_px'] - self.targets_matrix['rest'][0]['aa_px']
            distY = self.plant_pos[:]['aa_py'] - self.targets_matrix['rest'][0]['aa_py']
            aa_distXY = np.sqrt(distX**2 + distY**2)

            velX = self.plant_vel[:]['aa_vx'] 
            velY = self.plant_vel[:]['aa_vy'] 
            aa_velXY = np.sqrt(velX**2 + velY**2)

        safety_states_plant = safety_states[self.plant_type]
        safety_states_vel_plant = safety_states_vel[self.plant_type]

        border_safety_lower = {}
        border_safety_upper = {}

        for ind_safety_dof, safety_dof in enumerate(safety_states_plant):
            if safety_dof == 'aa_distXY':
                border_safety_lower[safety_dof] = aa_distXY <= self.safety_area_plant['min_' + safety_dof]
                border_safety_upper[safety_dof] = aa_distXY >=self.safety_area_plant['max_' + safety_dof]
                
            else:

                border_safety_lower[safety_dof] = self.plant_pos[:][safety_dof] <= self.safety_area_plant['min_' + safety_dof]
                border_safety_upper[safety_dof] = self.plant_pos[:][safety_dof] >= self.safety_area_plant['max_' + safety_dof]

        # check actual velocity directions to see if it is driving the DoFs out of the range of security
            if border_safety_lower[safety_dof] == True:
                print "plant going out of safety area (lower boundary)"
                print "safety_dof ", safety_dof

                if safety_dof != 'aa_distXY':
                    if command_vel[safety_states_vel_plant[ind_safety_dof]] < 0:
                        command_vel[safety_states_vel_plant[ind_safety_dof]] == 0


            if border_safety_upper[safety_dof] == True:
                print "plant going out of safety area (upper boundary)"
                print "safety_dof " ,safety_dof
                if safety_dof != 'aa_distXY':
                    if command_vel[safety_states_vel_plant[ind_safety_dof]] > 0:
                        command_vel[safety_states_vel_plant[ind_safety_dof]] == 0

                elif safety_dof == 'aa_distXY':

                    next_X = self.plant_pos[:]['aa_px'] + command_vel['aa_vx']
                    next_Y = self.plant_pos[:]['aa_py'] + command_vel['aa_vy']
                    next_distXY = np.sqrt(next_X**2 + next_Y**2)
                    if next_distXY > border_safety_upper[safety_dof]:
                        command_vel['aa_vx'] = 0
                        command_vel['aa_vy'] = 0

        return command_vel
                    



    # ------------------ GOAL TARGETS SEQUENCE GENERATORS --------------------------#
    #depending on the taks, some will have different sequences for recording the goal targets (with an extra trial_type to record REST position) or without it
    
    @staticmethod
    def B1_targets(length=5, red=1, green = 1, brown = 0, blue=1, shuffle=1):
        available_targets = []
        if red: available_targets.append('red')
        if green: available_targets.append('green')
        if brown: available_targets.append('brown')
        if blue: available_targets.append('blue')

        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=shuffle)
        return targets

    @staticmethod
    def single_target(length=5, target=1):
        available_targets = []
        if target: available_targets.append('target')

        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=shuffle)
        return targets

    @staticmethod
    def B2_targets(length=5, grasp=1, pinch=1, point=1, up=1, down=1, grasp_up=1, wrist_ext = 1, shuffle=1 ):
        available_targets = []
        if grasp: available_targets.append('grasp')
        if pinch: available_targets.append('pinch')
        if point: available_targets.append('point')
        if up: available_targets.append('up')
        if down: available_targets.append('down') 
        if wrist_ext: available_targets.append('wrist_ext')        

        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=shuffle)
        return targets  
    
    @staticmethod
    def B3_targets(length=5, grasp_up=1, grasp_down=1, point_up=1, point_down=1, shuffle=1 ):
        available_targets = []
        if grasp_up: available_targets.append('grasp_up')
        if grasp_down: available_targets.append('grasp_down')
        if point_up: available_targets.append('point_up')
        if point_down: available_targets.append('point_down')     

        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=shuffle)
        return targets  

    @staticmethod
    def F1_targets(length=1, red_to_green=1, red_to_blue=1, green_to_red=1,  green_to_blue=1, blue_to_green=1, blue_to_red=1,   shuffle=1):
        '''
        Generate target sequence for the F1 task.
        '''
        available_targets = []
        if green_to_blue: available_targets.append('green to blue')
        if red_to_green: available_targets.append('red to green')
        if red_to_blue: available_targets.append('red to blue')
        if green_to_red: available_targets.append('green to red')
        if blue_to_red: available_targets.append('blue to red')
        if blue_to_green: available_targets.append('blue to green')

        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=shuffle)
        return targets

    @staticmethod
    def B1_B2_targets(length=1, red_grasp=1, red_point=1, red_up=1, green_grasp=1, green_point=1, green_up=1, blue_grasp=1, blue_point=1, blue_up=1, shuffle=1):
        '''
        Generate target sequence for the B1_B2 task.
        '''
        available_targets = []
        if red_grasp: available_targets.append('red_grasp')
        if red_point: available_targets.append('red_point')
        if red_up: available_targets.append('red_up')
        if green_grasp: available_targets.append('green_grasp')
        if green_point: available_targets.append('green_point')
        if green_up: available_targets.append('green_up')
        if blue_grasp: available_targets.append('blue_grasp')
        if blue_point: available_targets.append('blue_point')
        if blue_up: available_targets.append('blue_up')
        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=shuffle)

        return targets

    @staticmethod
    def compliant_movements_obj(length=1,
        stirr_clockwise=1, stirr_anticlockwise=1, 
        clean_horizontal = 1, clean_vertical = 1):

        available_targets = []

        ##### Circular #####
        for i, (t, tn) in enumerate(zip([stirr_clockwise, stirr_anticlockwise], 
            ['stirr_clockwise', 'stirr_anticlockwise'])):
            if t:
                available_targets.append(tn)        
        
        ##### Linear #####
        for i, (t, tn) in enumerate(zip([clean_horizontal, clean_vertical], 
            ['clean_horizontal', 'clean_vertical'])):
            if t:
                available_targets.append(tn)
        return available_targets

    @staticmethod
    def compliant_move_blk1(length=1, red=1, blue=1, green=1, red_to_blue=1, red_to_green=1,
        blue_to_red=1, blue_to_green=1, green_to_red=0, green_to_blue=0, non_red_first=False, shuffle=0):
        available_targets = []
        
        #### B1 targets ####
        for i, (t, tn) in enumerate(zip([red, blue, green], ['red', 'blue', 'green'])):
            if t:
                available_targets.append(tn)   

        #### F1 targets ####
        for i, (t, tn) in enumerate(zip([red_to_blue, red_to_green, blue_to_red, blue_to_green], ['red to blue', 'red to green', 'blue to red', 'blue to green'])):
            if t:
                available_targets.append(tn)

        for i, (t, tn) in enumerate(zip([green_to_blue, green_to_red], ['green to blue', 'green to red'])):
            if t:
                available_targets.append(tn)

        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=shuffle)
        if non_red_first:
            while 'red' in targets[0]:
                targets.insert(0, targets.pop())
        return targets 

    @staticmethod
    def compliant_move_blk2(length=1, grasp=1, point=1, up=1, down=1, grasp_up=1, grasp_down=1, point_up=1, 
        point_down=1, shuffle=1):
        available_targets = []
        
        ##### B2 targets #### 
        for i, (t, tn) in enumerate(zip([grasp, point, up, down], ['grasp', 'point', 'up', 'down'])):
            if t:
                available_targets.append(tn)    

        ##### B3 #########
        for i, (t, tn) in enumerate(zip([grasp_up, grasp_down, point_up, point_down], ['grasp_up', 'grasp_down', 'point_up', 'point_down'])):
            if t:
                available_targets.append(tn)
        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=shuffle)
        return targets 

    @staticmethod
    def compliant_move_blk3(length=1, red_grasp=0, red_point=0, red_up=1, red_down = 1, 
        green_grasp=0, green_point=1, green_up=0, green_down =0, 
        blue_grasp=1, blue_point=0, blue_up=0, blue_down = 0, shuffle=0):
        available_targets = []
        
        ##### B1_B2 targets #### 
        for i, (t, tn) in enumerate(zip([red_grasp, red_point, red_up, green_grasp, green_point, green_up, blue_grasp, blue_point, blue_up], 
            ['red_grasp', 'red_point', 'red_up', 'green_grasp', 'green_point', 'green_up', 'blue_grasp', 'blue_point', 'blue_up'])):
            if t:
                available_targets.append(tn)

        for i, (t, tn) in enumerate(zip([red_down, green_down, blue_down], ['red_down', 'green_down', 'blue_down'])):
            if t:
                available_targets.append(tn)

        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=shuffle)
        return targets 

    @staticmethod
    def compliant_move_blk4(length=1, red_grasp_up=1, red_grasp_down=0, red_point_up=0, red_point_down=1,
        green_grasp_up=0, green_grasp_down=1, green_point_up=0, green_point_down=0,
        blue_grasp_up=1, blue_grasp_down=0, blue_point_up=0, blue_point_down=0, shuffle=0):
        available_targets = []
        
        ##### B1_B3 #########
        if red_grasp_up: available_targets.append('red_grasp_up')
        if red_grasp_down: available_targets.append('red_grasp_down')
        if green_grasp_up: available_targets.append('green_grasp_up')
        if green_grasp_down: available_targets.append('green_grasp_down')
        if blue_grasp_up: available_targets.append('blue_grasp_up')
        if blue_grasp_down: available_targets.append('blue_grasp_down')

        if red_point_up: available_targets.append('red_point_up')
        if red_point_down: available_targets.append('red_point_down')
        if green_point_up: available_targets.append('green_point_up')
        if green_point_down: available_targets.append('green_point_down')
        if blue_point_up: available_targets.append('blue_point_up')
        if blue_point_down: available_targets.append('blue_point_down') 

        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=shuffle)
        return targets 

    @staticmethod
    def compliant_move_blk3_4_combo(length=1):
        available_targets = ['red_up', 'green_point','blue_grasp', 'red_down', 'blue_point', 
        'red_grasp_up','green_grasp_down', 'blue_grasp_up', 'red_point_down', 'blue_point_down', 'grasp']
        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=0)
        return targets

    @staticmethod
    def blk_B1_grasp(length=2):
        available_targets = ['red', 'green','blue', 'red_grasp', 'green_grasp', 
        'blue_grasp','grasp']
        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=1)
        return targets
 
    @staticmethod
    def blk_grasp_combo(length=2):
        available_targets = ['grasp', 'grasp_up','grasp_down', 'red_grasp', 'green_grasp', 
        'blue_grasp']
        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=1)
        return targets

    @staticmethod
    def clda_blk(length=1, shuffle=0):
        available_targets = ['red', 'green', 'blue', 'up', 'down', 'point', 'grasp', 'red_up',
            'green_point', 'blue_grasp', 'red_point_down', 'blue_grasp_up', 'green_grasp_down']
        return NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=shuffle)

    @staticmethod
    def compliant_movements_non_obj(length= 1, red=1, blue=1, green=1, double_reach=1, double_reach_B2=1,
        grasp=1, point=1, up=1, down=1, red_grasp=1, 
        
        red_point=1, red_up=1, red_down = 1, 
        green_grasp=1, green_point=1, green_up=1, green_down =1, 
        blue_grasp=1, blue_point=1, blue_up=1, blue_down = 1, 
        
        grasp_up=1, grasp_down=1, point_up=1, point_down=1,
        
        red_grasp_up=1, red_grasp_down=1, red_point_up=1, red_point_down=1,
        green_grasp_up=1, green_grasp_down=1, green_point_up=1, green_point_down=1,
        blue_grasp_up=1, blue_grasp_down=1, blue_point_up=1, blue_point_down=1,
        
        shuffle=0):

        available_targets = []

        #### B1 targets ####
        for i, (t, tn) in enumerate(zip([red, blue, green], ['red', 'blue', 'green'])):
            if t:
                available_targets.append(tn)

        ##### B2 targets #### 
        for i, (t, tn) in enumerate(zip([grasp, point, up, down], ['grasp', 'point', 'up', 'down'])):
            if t:
                available_targets.append(tn)        

        ##### B1_B2 targets #### 
        for i, (t, tn) in enumerate(zip([red_grasp, red_point, red_up, green_grasp, green_point, green_up, blue_grasp, blue_point, blue_up], 
            ['red_grasp', 'red_point', 'red_up', 'green_grasp', 'green_point', 'green_up', 'blue_grasp', 'blue_point', 'blue_up'])):
            if t:
                available_targets.append(tn)

        if double_reach:
            ##### F1 targets #### 
            for i, (t, tn) in enumerate(zip([red, blue, green], ['red', 'blue', 'green'])):
                for i2, (t2, tn2) in enumerate(zip([red, blue, green], ['red', 'blue', 'green'])):
                    if tn != tn2:
                        if t and t2:        
                            available_targets.append(tn+' to '+tn2)
  

        ##### F1_B2 targets #### 
        if double_reach_B2:
            for i, (t, tn) in enumerate(zip([red_grasp, red_point, red_up, green_grasp, green_point, green_up, blue_grasp, blue_point, blue_up], 
                ['red_grasp', 'red_point', 'red_up', 'green_grasp', 'green_point', 'green_up', 'blue_grasp', 'blue_point', 'blue_up'])):

                for i2, (t2, tn2) in enumerate(zip([red_grasp, red_point, red_up, green_grasp, green_point, green_up, blue_grasp, blue_point, blue_up], 
                    ['red_grasp', 'red_point', 'red_up', 'green_grasp', 'green_point', 'green_up', 'blue_grasp', 'blue_point', 'blue_up'])):

                    if t and t2:
                        if tn != tn2 and tn[:4] != tn2[:4]:
                            available_targets.append(tn+' to '+tn2)

        ##### B3 #########
        for i, (t, tn) in enumerate(zip([grasp_up, grasp_down, point_up, point_down], ['grasp_up', 'grasp_down', 'point_up', 'point_down'])):
            if t:
                available_targets.append(tn)              

        ##### B1_B3 #########
        if red_grasp_up: available_targets.append('red_grasp_up')
        if red_grasp_down: available_targets.append('red_grasp_down')
        if green_grasp_up: available_targets.append('green_grasp_up')
        if green_grasp_down: available_targets.append('green_grasp_down')
        if blue_grasp_up: available_targets.append('blue_grasp_up')
        if blue_grasp_down: available_targets.append('blue_grasp_down')

        if red_point_up: available_targets.append('red_point_up')
        if red_point_down: available_targets.append('red_point_down')
        if green_point_up: available_targets.append('green_point_up')
        if green_point_down: available_targets.append('green_point_down')
        if blue_point_up: available_targets.append('blue_point_up')
        if blue_point_down: available_targets.append('blue_point_down') 
        
        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=shuffle)
        return targets

    @staticmethod
    def active_movements():
        return []

    @staticmethod
    def active_movements_blk(length=1, red_w_bottle=0, green_w_bottle=0, blue_w_bottle=0, 
        bottle_to_box_low=0, bottle_to_box_high=0, 
        bottle_to_mouth=0, cup_to_mouth = 0,
        wring_towel_up=0, wring_towel_down = 0,
        wrist_ext=0, wrist_ext_w_res =0,
        thumb_ext=0, thumb_ext_w_res=0, thumb_abd_add=0, 
        grasp_w_res=0, up_w_res = 0, 
        scissors = 0,
        cylinder_bimanual = 0, tray_to_front_bimanual = 0, open_box_bimanual = 0,
        hold_bottle_green = 0, 
        # from 3rd April 2018 on
        grasp = 0, wrist_rotation =0, fingers_abd_add = 0, rolling_pin_front_up = 0, cup_to_box_low = 0, cup_to_box_high = 0, stir_w_spoon = 0,
        fingers_abd_add_w_res=0, zeros= 0, eights=0, hold_arm_up = 0, close_hand = 0,
        # from 7th Nov 2018 on
        hold_fingers_ext = 0, hold_wrist_ext=0, 
        new_exercise = 0, shuffle=0):
     

        available_targets = []
        
        if red_w_bottle: available_targets.append('red_w_bottle')
        if green_w_bottle: available_targets.append('green_w_bottle')
        if blue_w_bottle: available_targets.append('blue_w_bottle')

        if bottle_to_box_low: available_targets.append('bottle_to_box_low')
        if bottle_to_box_high: available_targets.append('bottle_to_box_high')
        if bottle_to_mouth: available_targets.append('bottle_to_mouth')

        if wring_towel_up: available_targets.append('wring_towel_up')
        if wring_towel_down: available_targets.append('wring_towel_down')

        if wrist_ext: available_targets.append('wrist_ext')
        if wrist_ext_w_res: available_targets.append('wrist_ext_w_res')
        if thumb_ext: available_targets.append('thumb_ext')
        if thumb_ext_w_res: available_targets.append('thumb_ext_w_res')
        if thumb_abd_add: available_targets.append('thumb_abd_add')
        if grasp_w_res: available_targets.append('grasp_w_res') 
        if up_w_res: available_targets.append('up_w_res')
        if scissors: available_targets.append('scissors')
        if cylinder_bimanual: available_targets.append('cylinder_bimanual') 
        if tray_to_front_bimanual: available_targets.append('tray_to_front_bimanual') 
        if open_box_bimanual: available_targets.append('open_box_bimanual') 
        if hold_bottle_green: available_targets.append('hold_bottle_green')
        if cup_to_mouth: available_targets.append('cup_to_mouth')
        # from 3rd April 2018 on
        if grasp: available_targets.append('grasp')
        if wrist_rotation: available_targets.append('wrist_rotation')
        if fingers_abd_add: available_targets.append('fingers_abd_add')
        if rolling_pin_front_up: available_targets.append('rolling_pin_front_up')
        if cup_to_box_low: available_targets.append('cup_to_box_low')
        if cup_to_box_high: available_targets.append('cup_to_box_high')
        # from 22nd May 2018 on
        if fingers_abd_add_w_res: available_targets.append('fingers_abd_add_w_res')
        # from 24th May 2018 on
        if stir_w_spoon: available_targets.append('stir_w_spoon')
        if zeros: available_targets.append('zeros')
        if eights: available_targets.append('eights')
        # from 18th July 2018 on
        if hold_arm_up: available_targets.append('hold_arm_up')
        if close_hand: available_targets.append('close_hand')
        # from 7th Nov 2018 on
        if hold_fingers_ext: available_targets.append('hold_fingers_ext')
        if hold_wrist_ext: available_targets.append('hold_wrist_ext')

        if new_exercise: available_targets.append('new_exercise')

        print available_targets, length, shuffle
        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=shuffle)
        return targets 

    @staticmethod
    def mirror_therapy_movements_blk():
        return []


    @staticmethod
    def mirror_therapy_movements_blk_new(length=1,
        m1_open_book=0, m2_roll_up=0, m3_roll_front=0, m4_glass_front=0, m5_glass_mouth=0, 
        m6_glasses_face=0, m7_glasses_front=0, m8_rolling_pin=0, m9_turn_page_calendar=0, 
        m10_bowl_to_front =0, m11_clap =0, m12_tray_to_front=0, m13_open_box=0, 
        m14_fold_towel_vert=0, m15_fold_towel_horiz=0, m16_roll_towel_up=0, m17_hang_towel=0, 
        m18_mix_w_fork=0,  m19_play_symbols=0, m21_play_drum=0, m22_fingers_touch_thumb=0,
        m23_open_close_hand=0, m24_open_close_thumb = 0, m25_wrist_pronosup = 0, m26_wrist_flex_ext = 0, 
        m27_wrist_flex_inhibit = 0, m28_4fingers_abd_add = 0, m29_hang_towel_arm_ext = 0, 
        m30_wrist_ext_hold_elastic_res = 0, m31_wrist_ext_hold_elastic_res_arm_ext = 0,
        m32_wrist_and_fingers_ext_flex = 0, m33_wrist_and_fingers_ext_flex_w_res = 0, m34_zeros = 0, m35_eights = 0,
        m36_wrist_flex_inhibit_w_paretic_idem= 0, m37_push_box_w_paretic_idem = 0, 
        m38_pinky_flex_ext = 0, m39_ring_flex_ext = 0, m40_middle_flex_ext = 0, m41_index_flex_ext = 0 , m42_grasp_inhibit_reflex=0,
        m_new_exercise = 0 ):

        available_targets = []
        
        if m1_open_book: available_targets.append('m1_open_book')
        if m2_roll_up: available_targets.append('m2_roll_up')
        if m3_roll_front: available_targets.append('m3_roll_front')
        if m4_glass_front: available_targets.append('m4_glass_front')
        if m5_glass_mouth: available_targets.append('m5_glass_mouth')
        if m6_glasses_face: available_targets.append('m6_glasses_face')
        if m7_glasses_front: available_targets.append('m7_glasses_front')
        if m8_rolling_pin: available_targets.append('m8_rolling_pin')
        if m9_turn_page_calendar: available_targets.append('m9_turn_page_calendar')
        if m10_bowl_to_front: available_targets.append('m10_bowl_to_front')
        if m11_clap: available_targets.append('m11_clap')
        if m12_tray_to_front: available_targets.append('m12_tray_to_front')
        if m13_open_box: available_targets.append('m13_open_box')
        if m14_fold_towel_vert: available_targets.append('m14_fold_towel_vert')
        if m15_fold_towel_horiz: available_targets.append('m15_fold_towel_horiz')
        if m16_roll_towel_up: available_targets.append('m16_roll_towel_up')
        if m17_hang_towel: available_targets.append('m17_hang_towel')
        if m18_mix_w_fork: available_targets.append('m18_mix_w_fork')
        if m19_play_symbols: available_targets.append('m19_play_symbols')
        if m21_play_drum: available_targets.append('m21_play_drum')
        if m22_fingers_touch_thumb: available_targets.append('m22_fingers_touch_thumb')
        if m23_open_close_hand: available_targets.append('m23_open_close_hand')
        if m24_open_close_thumb: available_targets.append('m24_open_close_thumb')
        if m25_wrist_pronosup: available_targets.append('m25_wrist_pronosup')
        if m26_wrist_flex_ext: available_targets.append('m26_wrist_flex_ext')
        # from 9rd April 2018 on 
        if m27_wrist_flex_inhibit: available_targets.append('m27_wrist_flex_inhibit')
        if m28_4fingers_abd_add: available_targets.append('m28_4fingers_abd_add')
        if m29_hang_towel_arm_ext: available_targets.append('m29_hang_towel_arm_ext')
        if m30_wrist_ext_hold_elastic_res: available_targets.append('m30_wrist_ext_hold_elastic_res')
        if m31_wrist_ext_hold_elastic_res_arm_ext: available_targets.append('m31_wrist_ext_hold_elastic_res_arm_ext')
        if m32_wrist_and_fingers_ext_flex: available_targets.append('m32_wrist_and_fingers_ext_flex')
        if m33_wrist_and_fingers_ext_flex_w_res: available_targets.append('m33_wrist_and_fingers_ext_flex_w_res')
        if m34_zeros: available_targets.append('m34_zeros')
        if m35_eights: available_targets.append('m35_eights')
        # from 21st June 2018 on 
        if m36_wrist_flex_inhibit_w_paretic_idem: available_targets.append('m36_wrist_flex_inhibit_w_paretic_idem') # central inhibition of flexors
        if m37_push_box_w_paretic_idem: available_targets.append('m37_push_box_w_paretic_idem') #reciprocal inhibition of flexors

        # from 13rd Sept 2018 on
        if m38_pinky_flex_ext: available_targets.append('m38_pinky_flex_ext')
        if m39_ring_flex_ext: available_targets.append('m39_ring_flex_ext')
        if m40_middle_flex_ext: available_targets.append('m40_middle_flex_ext')
        if m41_index_flex_ext: available_targets.append('m41_index_flex_ext')
        # from 7th Nov 2018 on
        if m42_grasp_inhibit_reflex: available_targets.append('m42_grasp_inhibit_reflex')
        
        if m_new_exercise: available_targets.append('m_new_exercise') #trial type to use whenver

        targets = NonInvasiveBase._make_block_repeated_targets(length, available_targets, shuffle=0)

        return targets
    

    # @staticmethod
    # def F1_B2_targets(length=1, red_grasp=0, red_point=0, red_up=0, green_grasp=0, green_point=0, green_up=0, 
    #                     blue_grasp=0, blue_point=0, blue_up=0, shuffle=1):
    #     '''
    #     Generate target sequence for the F1_B2 task.
    #     '''
    #     selected_targets = []        
    #     if red_grasp: selected_targets.append('red_grasp')
    #     if red_point: selected_targets.append('red_point')
    #     if red_up: selected_targets.append('red_up')
    #     if green_grasp: selected_targets.append('green_grasp')
    #     if green_point: selected_targets.append('green_point')
    #     if green_up: selected_targets.append('green_up')
    #     if blue_grasp: selected_targets.append('blue_grasp')
    #     if blue_point: selected_targets.append('blue_point')
    #     if blue_up: selected_targets.append('blue_up')

    #     available_targets = []

    #     for F1_target_1 in selected_targets:
    #         for F1_target_2 in selected_targets:
    #             if F1_target_1 != F1_target_2:
    #                 available_targets.append(F1_target_1 + ' to ' + F1_target_2)

    #     targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=shuffle)

    #     return targets 

    @staticmethod
    def F1_B2_targets(length=1, red_up=0, green_up=0, blue_up=0, red_up2=0, green_up2=0, blue_up2=0, shuffle=1):
        '''
        Generate target sequence for the F1_B2 task.
        '''
        selected_targets = []  
        selected_targets2 = []       
        # if red_grasp: selected_targets.append('red_grasp')
        # if red_point: selected_targets.append('red_point')
        if red_up: selected_targets.append('red_up')
        # if green_grasp: selected_targets.append('green_grasp')
        # if green_point: selected_targets.append('green_point')
        if green_up: selected_targets.append('green_up')
        # if blue_grasp: selected_targets.append('blue_grasp')
        # if blue_point: selected_targets.append('blue_point')
        if blue_up: selected_targets.append('blue_up')

        if red_up2: selected_targets2.append('red_up2')
        # if green_grasp: selected_targets.append('green_grasp')
        # if green_point: selected_targets.append('green_point')
        if green_up2: selected_targets2.append('green_up2')
        # if blue_grasp: selected_targets.append('blue_grasp')
        # if blue_point: selected_targets.append('blue_point')
        if blue_up2: selected_targets2.append('blue_up2')

        available_targets = []

        for F1_target_1 in selected_targets:
            for F1_target_2 in selected_targets2:
                if F1_target_1 != F1_target_2[:-1]:
                    available_targets.append(F1_target_1 + ' to ' + F1_target_2[:-1])

        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=shuffle)

        return targets 

    @staticmethod
    def FreeMov_targets(length=1):
        return length * ['go']      

    @staticmethod
    def double_target(length=10):
        colors = ['blue', 'green', 'red']
        trial_types = []
        for c1 in colors:
            for c2 in colors:
                if c1 != c2:
                    trial_types.append(c1 + ' to ' + c2)
        
        return length * trial_types

    @staticmethod
    def cyclic_targets(length=5, circular=1, linear_blue = 1, linear_brown = 1, linear_green = 1, linear_red = 1, sequence=1, shuffle=0):
        available_targets = []
        if circular: available_targets.append('circular')
        if linear_blue: available_targets.append('linear_blue')
        if linear_brown: available_targets.append('linear_brown')
        if linear_green: available_targets.append('linear_green')
        if linear_red: available_targets.append('linear_red')
        if sequence: available_targets.append('sequence')
        print available_targets
        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=shuffle)
        return targets
   
    @staticmethod
    def targets_linear(length = 2, red=1, green = 1, brown= 1, blue=1, red_to_blue=0, shuffle=0):
        available_targets = []
        if red: available_targets.append('linear_red')
        if green: available_targets.append('linear_green')
        if brown: available_targets.append('linear_brown')
        if blue: available_targets.append('linear_blue')
        if red_to_blue: available_targets.append('linear_red_to_blue')
             
        targets = NonInvasiveBase._make_block_repeated_targets(length, available_targets, shuffle=shuffle)
        return targets

    @staticmethod
    def targets_circular(length=4, circular_clockwise=1, circular_anticlockwise=1):
        available_targets = []
        # if circular_small: available_targets.append('circular_small')
        # if circular_big: available_targets.append('circular_big')
        if circular_clockwise: available_targets.append('circular_clockwise')
        if circular_anticlockwise: available_targets.append('circular_anticlockwise')
        
        targets = NonInvasiveBase._make_block_repeated_targets(length, available_targets, shuffle=0)
        return targets
    
    @staticmethod
    #as sequential goal targets will be used in tasks that include reaching + graspings, we might need to create goal targets with the combined names of the 2 movemetns
    def targets_sequence(length=5, red=1, green = 1, brown= 1, blue=1, shuffle=0):
        available_targets = []
        if red: available_targets.append('red')
        if green: available_targets.append('green')
        if brown: available_targets.append('brown')
        if blue: available_targets.append('blue')
             
        targets = NonInvasiveBase._make_block_repeated_targets(length, available_targets, shuffle=shuffle)
        return targets

    ### GOAL TARGETS ###

    @staticmethod
    def goal_targets_B1(length=1, rest=1, red=1, green = 1, blue=1, shuffle=0):
        available_targets = []
        available_targets.append('rest')
        if red: available_targets.append('red')
        if green: available_targets.append('green')
        if blue: available_targets.append('blue')

        return available_targets
        #define goal targets separately here because they include a rest trial that will not be considered as a trial type in other tasks
    @staticmethod
    def goal_targets_B2(length=1, rest=1, grasp=1, pinch=1, point=1,  up=1, down=1, shuffle=0):
        available_targets = []
        available_targets.append('rest')
        if grasp: available_targets.append('grasp')
        if pinch: available_targets.append('pinch')
        if point: available_targets.append('point')
        if up: available_targets.append('up')
        if down: available_targets.append('down')
        
        return available_targets

    @staticmethod
    def goal_targets_B3(length=1, rest=1, grasp_up=1, grasp_down=1, point_up=1, point_down=1):
        available_targets = ['rest']
        if grasp_up: available_targets.append('grasp_up')
        if grasp_down: available_targets.append('grasp_down')
        if point_up: available_targets.append('point_up')
        if point_down: available_targets.append('point_down')

        return available_targets

    @staticmethod
    def goal_targets_B1_B2(length=1, rest=1, red_grasp=1, red_point=1, red_up=1, red_down=1,
        green_grasp=1, green_point=1, green_up=1, green_down=1,
        blue_grasp=1, blue_point=1, blue_up=1, blue_down=1,
        shuffle=1):
        '''
        Generate target sequence for the B1_B2 task.
        '''
        available_targets = []
        if rest: available_targets.append('rest')
        if red_grasp: available_targets.append('red_grasp')
        if red_point: available_targets.append('red_point')
        if red_up: available_targets.append('red_up')
        if red_down: available_targets.append('red_down')
        if green_grasp: available_targets.append('green_grasp')
        if green_point: available_targets.append('green_point')
        if green_up: available_targets.append('green_up')
        if green_down: available_targets.append('green_down')
        if blue_grasp: available_targets.append('blue_grasp')
        if blue_point: available_targets.append('blue_point')
        if blue_up: available_targets.append('blue_up')
        if blue_down: available_targets.append('blue_down')
  
        return available_targets   

    @staticmethod
    def goal_targets_B1_B3(rest=1, red_grasp_up=1, red_grasp_down=1, red_point_up=1, red_point_down=1,
        green_grasp_up=1, green_grasp_down=1, green_point_up=1, green_point_down=1,
        blue_grasp_up=1, blue_grasp_down=1, blue_point_up=1, blue_point_down=1):
        available_targets = ['rest']
        if red_grasp_up: available_targets.append('red_grasp_up')
        if red_grasp_down: available_targets.append('red_grasp_down')
        if green_grasp_up: available_targets.append('green_grasp_up')
        if green_grasp_down: available_targets.append('green_grasp_down')
        if blue_grasp_up: available_targets.append('blue_grasp_up')
        if blue_grasp_down: available_targets.append('blue_grasp_down')

        if red_point_up: available_targets.append('red_point_up')
        if red_point_down: available_targets.append('red_point_down')
        if green_point_up: available_targets.append('green_point_up')
        if green_point_down: available_targets.append('green_point_down')
        if blue_point_up: available_targets.append('blue_point_up')
        if blue_point_down: available_targets.append('blue_point_down')
        return available_targets   

    @staticmethod
    def goal_targets_cyclic(length=1, rest = 0, circular=1, linear_blue = 1, linear_brown = 1, linear_green = 1, linear_red = 1, sequence=1, shuffle=0):
        available_targets = []
        if rest: available_targets.append('rest')
        if circular: available_targets.append('circular')
        if linear_blue: available_targets.append('linear_blue')
        if linear_brown: available_targets.append('linear_brown')
        if linear_green: available_targets.append('linear_green')
        if linear_red: available_targets.append('linear_red')
        if sequence: available_targets.append('sequence')

        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=shuffle)
        return targets


class NonInvasiveBase(Autostart, Sequence, IsMoreBase):
    '''Abstract base class for noninvasive IsMore tasks (e.g., tasks for
    recording and playing back trajectories). This class defines some 
    common sequence generators for those inheriting classes.
    '''
    language = traits.OptionsList(*languages_list, bmi3d_input_options=languages_list)
    sequence_generators = [
        'B1_targets', #these target lists do not include the rest target
        'B2_targets',
        'F1_targets',
        'B3_targets',
        'B1_B2_targets',
        'F1_B2_targets',
        'single_target',

        'cyclic_targets',
        'FreeMov_targets',
        'double_target',
        'targets_linear',
        'targets_circular',
        'targets_sequence',

        'goal_targets_B1', #these target lists do include the rest target
        'goal_targets_B2',
        'goal_targets_B1_B2',
        'goal_targets_cyclic', 
        'goal_targets_B3',
        'goal_targets_B1_B3',     
        ]


    def __init__(self, *args, **kwargs):
        super(NonInvasiveBase, self).__init__(*args, **kwargs)
        self.add_dtype('plant_vel_filt', 'f8', (len(self.plant_vel),))
        self.add_dtype('plant_pos_filt', 'f8', (len(self.plant_vel),))
        
        pygame.mixer.init()               

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
    def _make_block_rand_targets_new(length, available_targets, shuffle=False):
        targets = {}
        for k in range(length):
            a_ = available_targets
            if shuffle:
                random.shuffle(a_)
            targets += a_
        return targets

    @staticmethod
    def _make_block_repeated_targets(length, available_targets, shuffle=False):
        targets = []
        for k in available_targets:
            block = [k]*length
            targets += block
        return targets



        


class RecordBrainAmpData(Autostart):
    channel_list_name = traits.OptionsList(*channel_list_options, bmi3d_input_options=channel_list_options) #choose always the filtered + raw option!!!!!!

    def __init__(self, *args, **kwargs):
        super(RecordBrainAmpData, self).__init__(*args, **kwargs)
        print self.channel_list_name
        self.brainamp_channels = getattr(brainamp_channel_lists, self.channel_list_name)  

    def init(self):

        from riglib import source
        from ismore.brainamp import rda

        #self.brainamp_source = source.MultiChanDataSource(rda.EMGData, name='brainamp', channels=self.brainamp_channels, brainamp_channels=self.brainamp_channels, send_data_to_sink_manager=True)

        # from riglib import sink
        # sink.sinks.register(self.brainamp_source)

        super(RecordBrainAmpData, self).init() #we may not need this since the parent classes do no have an init() function

        #we can also do it adding this to the parent class and whenever we wanna hide it add:
        #exclude_parent_traits : the name of the emg channels trait.

    def cleanup(self, database, saveid, **kwargs):
        print 'cleaning up in brainampdata'
        # from socket import *
        # self.sock3 = socket(AF_INET, SOCK_STREAM)
        # self.sock3.connect(('192.168.137.1',6700))
        # self.sock3.send("Q")
        # from ismore.brainamp.rda import EMGData
        # # rda.EMGData.cleanup(database, saveid, **kwargs)
        # self.sock2.send("Q")
        
        super(RecordBrainAmpData,self).cleanup(database, saveid, **kwargs)


#####################################################  TASKS   ##################################################

class Disable_System(NonInvasiveBase):
    '''
    Task to disable the motors of the selected plant.
    '''
    def __init__(self, *args, **kwargs):
        super(Disable_System, self).__init__(*args, **kwargs)
        self.plant.disable()
        print 'Motors disabled'

class RecordGoalTargets(NonInvasiveBase):
    '''
    Base class to record goal target points 
    '''

    fps = 20

    status = {
        'wait': {
            'start_trial': 'instruct_trial_type',
            'stop': None},
        # as sometimes we will record several points (sequential) for a trial_type, we include this subtrial_wait in order to accept or discard all the trial_type positions 
        'subtrial_wait': {
            'start_trial': 'instruct_trial_type',
            'stop': None},
        'instruct_trial_type': {
            'end_instruct': 'trial',
            'stop':      None},
        'trial': {
            # 'end_trial': 'wait',
            'accept_trial': 'wait',
            'reject_trial': 'instruct_trial_type',
            'transition_saved': 'subtrial_wait',
            'subtarget_saved': 'subtrial_wait',
            'rest_saved': 'subtrial_wait',
            'stop':      None},    
    }
    state = 'wait'  # initial state

    sequence_generators = ['goal_targets_B1', 'goal_targets_B2','goal_targets_B1_B2',
    'targets_circular', 'targets_linear', 'targets_sequence', 'goal_targets_B3', 
    'goal_targets_B1_B3']

    def __init__(self, *args, **kwargs):
        ## Init the pygame mixer for playing back sounds
        super(RecordGoalTargets, self).__init__(*args, **kwargs)
    
        self.add_dtype('trial_type', np.str_, 40)
        self.add_dtype('trial_accept_reject', np.str_, 10)
        self.plant.disable() 
 

        self.experimenter_acceptance_of_trial = ''
        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)

      
    def _cycle(self):
        '''Runs self.fps times per second.'''

        self.plant_pos_raw[:] = self.plant.get_pos_raw()
        self.plant_pos[:] = self.plant.get_pos() 
        
        self.update_plant_display()
        
        self.plant_vel_raw[:] = self.plant.get_vel_raw()
        self.plant_vel[:] = self.plant.get_vel()
  

        self.task_data['plant_pos']  = self.plant_pos_raw.values
        self.task_data['plant_pos_filt']  = self.plant_pos.values 

        self.task_data['plant_vel']  = self.plant_vel_raw.values
        self.task_data['plant_vel_filt']  = self.plant_vel.values

        self.task_data['trial_type'] = self.trial_type
        self.task_data['ts']         = time.time()
        self.task_data['trial_accept_reject'] = self.experimenter_acceptance_of_trial

        super(RecordGoalTargets, self)._cycle()

    def _test_end_instruct(self, *args, **kwargs):
        return not pygame.mixer.music.get_busy()

    def _start_instruct_trial_type(self):
        self.experimenter_acceptance_of_trial = ''
        self._play_sound(self.sounds_dir, [self.trial_type]) 

        
    def _parse_next_trial(self):
        self.trial_type = self.next_trial

    def _start_trial(self):
        print 'Move the exo to target ', self.trial_type

    def _test_end_trial(self, ts):
        return ts > self.trial_time

    def _test_accept_trial(self, *args, **kwargs):
        return self.experimenter_acceptance_of_trial == 'accept'

    def _test_reject_trial(self, *args, **kwargs):
        return self.experimenter_acceptance_of_trial == 'reject'

    def _test_transition_saved(self, *args, **kwargs):
        return self.experimenter_acceptance_of_trial == 'transition'
    
    def _test_subtarget_saved(self, *args, **kwargs):
        return self.experimenter_acceptance_of_trial == 'subtarget'

    def _test_rest_saved(self, *args, **kwargs):
        return self.experimenter_acceptance_of_trial == 'rest'

class RecordSafetyGrid(RecordGoalTargets):
    def __init__(self, *args, **kwargs):
        super(RecordSafetyGrid, self).__init__(*args, **kwargs)

class GoToTarget(NonInvasiveBase):
    '''
    Drives the exo towards a predefined target position. 
    '''
    fps = 20
    status = {
        'wait': {
            'start_trial': 'instruct_trial_type',
            'stop': None},
        'instruct_trial_type': {
            'end_instruct': 'trial',
            'stop':      None},
        'trial': {
            'end_trial': 'wait',#'instruct_trial_go_to_start'
            'stop':      None},
        }
    
    state = 'wait'  # initial state

    # settable parameters on web interface    
    targets_matrix = traits.DataFile(object,desc='test', bmi3d_query_kwargs=dict(system__name='misc'))
    index_targets_matrix = traits.Int(0, desc=' index of the position in the targets_matrix we want to send the exo to')
    speed = traits.OptionsList(*speed_options, bmi3d_input_options= speed_options)
    debug = False
    DoF_control = traits.OptionsList(*DoF_control_options, bmi3d_input_options=DoF_control_options, desc='DoFs to be taken into account for condition fulfilment')
    DoF_target = traits.OptionsList(*DoF_target_options, bmi3d_input_options=DoF_target_options, desc='DoFs to be moved/controlled, the rest are stopped.')
    # Set the "is_bmi_seed" flag so that the server knows that this is a task which can be used to create a BMI decoder
    is_bmi_seed = True

    sequence_generators = ['goal_targets_B1', 'goal_targets_B2','goal_targets_B1_B2', 'targets_linear', 'targets_circular', 'targets_sequence']

    def __init__(self, *args, **kwargs):
        super(GoToTarget, self).__init__(*args, **kwargs)

        self.add_dtype('trial_type',   np.str_, 40)
        self.add_dtype('command_vel',  'f8',    (len(self.vel_states),))
        self.add_dtype('command_vel_raw',  'f8',    (len(self.vel_states),))
        self.add_dtype('command_pos',  'f8',    (len(self.pos_states),))
        self.add_dtype('goal_pos',      'f8',    (len(self.pos_states),))
        self.add_dtype('speed',   np.str_, 20)
        self.add_dtype('command_vel_final',  'f8',    (len(self.vel_states),))
      
        #area considered as acceptable as rest around the rest position (first position in trial) or the subtask end position
        #self.target_rect = np.array([2., 2., np.deg2rad(10),  np.deg2rad(5), np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)])# for targets during the trial time
        #self.rest_rect = np.array([3., 3., np.deg2rad(10),  np.deg2rad(5), np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)])# for rest position during 'go to start' time
        #more general

        self.target_margin = pd.Series(np.array([1, 1, np.deg2rad(10), np.deg2rad(2),  np.deg2rad(2), np.deg2rad(2), np.deg2rad(2)]), ismore_pos_states)
        self.target_margin = self.target_margin[self.pos_states]

        self.add_dtype('target_margin',      'f8',    (len(self.target_margin),))    

        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)
        self.reached_goal_position = False #If the goal_position is reached then give feedback to patient and start the movement back towards the rest_position

        self.plant.enable() 
        [self.DoF_target_idx_init, self.DoF_target_idx_end, self.DoF_not_control_idx_init, self.DoF_not_control_idx_end] = check_plant_and_DoFs(self.plant_type, self.DoF_control, self.DoF_target)
        self.subgoal_names = self.targets_matrix['subgoal_names']
        
        # self.define_safety()

        # 4th order butterworth filter for command_vel
        fs_synch = self.fps #Frequency at which the task is running
        nyq   = 0.5 * fs_synch
        cuttoff_freq  = 1.5 / nyq
        bpf_kin_coeffs = butter(4, cuttoff_freq, btype='low')

        self.command_lpfs = dict()
        for state in self.vel_states:
            self.command_lpfs[state] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1]) # low-pass filter to smooth out command velocities



    def init(self):
        kwargs = {
            'call_rate': self.fps,
            'xy_cutoff': 2.,
        }
        # self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT[self.plant_type](**kwargs)
        self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT[self.plant_type](speed = self.speed,**kwargs)

        super(GoToTarget, self).init()


    def pos_diff(self, x1, x2):
        '''
        Calculate x1 - x2, but with a specialized definition of "-"
        '''
        x1 = np.array(x1).ravel()
        x2 = np.array(x2).ravel()
        diff = []
        for  i, j in izip(x1, x2):
            diff.append(i-j)
        return np.array(diff)

        
    def _while_trial(self):

        if np.all(np.abs(self.pos_diff(self.goal_position[self.DoF_target_idx_init:self.DoF_target_idx_end],self.plant.get_pos()[self.DoF_target_idx_init:self.DoF_target_idx_end])) < self.target_margin[self.pos_states[self.DoF_target_idx_init:self.DoF_target_idx_end]]):
            self.reached_goal_position = True
            
    
    def move_plant(self):
        command_vel  = pd.Series(0.0, self.vel_states)
        command_vel_raw  = pd.Series(0.0, self.vel_states)
        command_vel_raw[:] = command_vel[:]
        command_vel_final  = pd.Series(0.0, self.vel_states)

        # Command zero velocity if the task is in a non-moving state
        if self.state in ['wait', 'instruct_trial_type']: 
            command_vel[:] = 0

        elif self.state == 'trial':
            current_pos = self.plant_pos[:].ravel()
            current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
            target_state = np.hstack([self.targets_matrix[self.trial_type][self.index_targets_matrix][self.pos_states], np.zeros_like(current_pos),1 ]).reshape(-1,1)

            assist_output = self.assister(current_state, target_state, 1.)
            Bu = np.array(assist_output["x_assist"]).ravel()

            command_vel[:] = Bu[len(current_pos):len(current_pos)*2]
  

        # # #Apply low-pass filter to command velocities
        # for state in self.vel_states:
        #     command_vel[state] = self.command_lpfs[state](command_vel[state])
        #     if np.isnan(command_vel[state]):
        #         command_vel[state] = 0

        self.task_data['command_vel']  = command_vel.values
        self.task_data['command_vel_raw']  = command_vel_raw.values

        # set the velocities of the DoFs that should not be controlled to zero.
        if np.isnan(self.DoF_not_control_idx_init) == False and np.isnan(self.DoF_not_control_idx_end) == False:
            command_vel[self.DoF_not_control_idx_init:self.DoF_not_control_idx_end] = 0
        
        # print "command_vel_before_safety" ,  command_vel
        # self.check_safety(command_vel)
        # print "command_vel_after_safety" ,  command_vel
        
        self.plant.send_vel(command_vel.values) #send velocity command to EXO 

        self.task_data['command_vel_final']  = command_vel.values
    
    def _cycle(self):
        '''Runs self.fps times per second.'''

        # get latest position/velocity information before calling move_plant()
        self.plant_pos_raw[:] = self.plant.get_pos_raw()
        self.plant_pos[:] = self.plant.get_pos() 

        self.plant_vel_raw[:] = self.plant.get_vel_raw()
        self.plant_vel[:] = self.plant.get_vel()

        self.move_plant()

        self.update_plant_display()

        # print self.subtrial_idx
        if not self.state in ['trial']:
            self.task_data['goal_pos'] = np.ones(len(self.pos_states))*np.nan
        else:
            self.task_data['goal_pos'] = self.goal_position

        self.task_data['plant_pos']  = self.plant_pos_raw.values
        self.task_data['plant_pos_filt']  = self.plant_pos.values 
        self.task_data['plant_vel']  = self.plant_vel_raw.values
        self.task_data['plant_vel_filt']  = self.plant_vel.values

        self.task_data['trial_type'] = self.trial_type
        self.task_data['ts']         = time.time()
        self.task_data['target_margin'] = self.target_margin
        self.task_data['speed'] = self.speed
      
        super(GoToTarget, self)._cycle()

    #### STATE AND TEST FUNCTIONS ####
    def _start_wait(self):
        super(GoToTarget, self)._start_wait()

    def _parse_next_trial(self):
        self.trial_type = self.next_trial

    def _test_end_instruct(self, *args, **kwargs):
        return not pygame.mixer.music.get_busy()

    def _start_instruct_trial_type(self):
        self._play_sound(self.sounds_dir, self.subgoal_names[self.trial_type][self.index_targets_matrix]) 
        self.reached_goal_position = False

    def _start_trial(self):
        print self.trial_type
        if self.index_targets_matrix > len(self.targets_matrix[self.trial_type].keys()) -1:
            print "ERROR: Selected index out of bounds. Sending the exo towards first position in targets_matrix"
            self.index_targets_matrix = 0
        self.goal_position = self.targets_matrix[self.trial_type][self.index_targets_matrix][self.pos_states]
        print "goal position: ", self.goal_position

    def _test_end_trial(self, ts):
        return self.reached_goal_position

    def _end_trial(self):
        pass

class EndPointMovement(NonInvasiveBase):
    '''
    Drives the exo towards previously recorded target positions. 
    Class to make a position control based on target configurations / target positions 
    rather than playing back a previously recorded trajectory.
    '''
    fps = 20
    
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
            'end_instruct': 'preparation',
            'stop':      None},
        'preparation': {
            'end_preparation': 'instruct_go',
            'stop':      None},    
        'instruct_go': {
            'end_instruct': 'trial',
            'stop':      None},
        'trial': {
            'end_trial': 'instruct_trial_return',#'instruct_trial_go_to_start'
            'stop':      None},    
        #If we wanna include always a "return trial" to go to the initial position after the target trial then one option would be to add this and use instruct_trial_go_to_start instead of wait at the previous state:
        'instruct_trial_return': {
            'end_instruct': 'trial_return',
            'stop':      None},
        'trial_return': {
            'end_trial': 'wait',
            'stop':      None},    
        }
    
    state = 'wait'  # initial state

    # settable parameters on web interface    
    preparation_interval = traits.Tuple((2., 3.), desc='time to remain in the preparation state.')
    rest_interval  = traits.Tuple((3., 3.), desc='Min and max time to remain in the rest state.')
    targets_matrix = traits.DataFile(object,desc='test', bmi3d_query_kwargs=dict(system__name='misc'))
    give_feedback  = traits.Int((0), desc=' 0 if we do not give feedback, 1 if we give feedback about whether the patient reached the goal position')
    music_feedback = traits.Int((0), desc=' 0 if we do not want to include music, 1 if we want different classical music pieces with increasing intensity to be played')
    speed = traits.OptionsList(*speed_options, bmi3d_input_options= speed_options)
    debug = False
    #trial_end_states = ['rest']
    DoF_control = traits.OptionsList(*DoF_control_options, bmi3d_input_options=DoF_control_options, desc='DoFs to be taken into account for condition fulfilment')
    DoF_target = traits.OptionsList(*DoF_target_options, bmi3d_input_options=DoF_target_options, desc='DoFs to be moved/controlled, the rest are stopped.')
   
    sequence_generators = ['B1_targets', 'B2_targets', 'F1_targets','B1_B2_targets','F1_B2_targets', 'B3_targets']

    # Set the "is_bmi_seed" flag so that the server knows that this is a task which can be used to create a BMI decoder
    is_bmi_seed = True

    def __init__(self, *args, **kwargs):
        super(EndPointMovement, self).__init__(*args, **kwargs)

        self.add_dtype('trial_type',   np.str_, 40)
        self.add_dtype('command_vel',  'f8',    (len(self.vel_states),))
        self.add_dtype('command_vel_raw',  'f8',    (len(self.vel_states),))
        self.add_dtype('command_pos',  'f8',    (len(self.pos_states),))
        self.add_dtype('goal_pos',      'f8',    (len(self.pos_states),))
        #self.add_dtype('audio_feedback_start',      int,    (1,))
        self.add_dtype('goal_idx', float, (1,))
        self.add_dtype('reached_goal_position',bool, (1,))
        self.add_dtype('speed',   np.str_, 20)
        self.add_dtype('command_vel_final',  'f8',    (len(self.vel_states),))


        self.parallel_sound = pygame.mixer.Sound('')
        # self.define_safety() #nerea
        # self.display_safety_area() #nerea
        # print "'safety_area_plant ", self.safety_area_plant
        #area considered as acceptable as rest around the rest position (first position in trial) or the subtask end position
        #self.target_rect = np.array([2., 2., np.deg2rad(10),  np.deg2rad(5), np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)])# for targets during the trial time
        #self.rest_rect = np.array([3., 3., np.deg2rad(10),  np.deg2rad(5), np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)])# for rest position during 'go to start' time
        #more general
        # target margin used for DK calibration sessions
        self.target_margin = pd.Series(np.array([2, 2, np.deg2rad(5), np.deg2rad(3),  np.deg2rad(3), np.deg2rad(3), np.deg2rad(3)]), ismore_pos_states)
        self.target_margin = self.target_margin[self.pos_states]

        self.add_dtype('target_margin',      'f8',    (len(self.target_margin),))    

        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)
        self.sounds_dir_classical = os.path.expandvars('$HOME/code/ismore/sounds/classical')

        self.reached_goal_position = False #If the goal_position is reached then give feedback to patient and start the movement back towards the rest_position

        self.plant.enable() 
       
        [self.DoF_target_idx_init, self.DoF_target_idx_end, self.DoF_not_control_idx_init, self.DoF_not_control_idx_end] = check_plant_and_DoFs(self.plant_type, self.DoF_control, self.DoF_target)
        self.subgoal_names = self.targets_matrix['subgoal_names']
        
        # 4th order butterworth filter for command_vel
        fs_synch = self.fps #Frequency at which the task is running

        nyq   = 0.5 * fs_synch
        cuttoff_freq  = 1.5 / nyq
        bpf_kin_coeffs = butter(4, cuttoff_freq, btype='low')
        self.command_lpfs = dict()
        for state in self.vel_states:
            self.command_lpfs[state] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1]) # low-pass filter to smooth out command velocities


        pygame.mixer.init()
        # import serial

        # self.serial_trigger =serial.Serial(
        #     port='/dev/ttyUSB0',
        #     baudrate=9600,
        #     parity=serial.PARITY_NONE,
        #     stopbits=serial.STOPBITS_ONE,
        #     bytesize=serial.SEVENBITS
        #     )

    def init(self):
        kwargs = {
            'call_rate': self.fps,
            'xy_cutoff': 2.,
        }
        # self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT[self.plant_type](**kwargs)
        self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT[self.plant_type](speed = self.speed,**kwargs)

        super(EndPointMovement, self).init()


    def pos_diff(self, x1, x2):
        '''
        Calculate x1 - x2, but with a specialized definition of "-"
        '''
        # if self.plant_type == 'ArmAssist':
        #     sub_fns = [operator.sub, operator.sub, angle_subtract]
        # elif self.plant_type == 'ReHand':
        #     sub_fns = [angle_subtract, angle_subtract, angle_subtract, angle_subtract]
        # elif self.plant_type == 'IsMore':
        #     sub_fns = [operator.sub, operator.sub, angle_subtract, angle_subtract, angle_subtract, angle_subtract, angle_subtract]

        # x1 = np.array(x1).ravel()
        # x2 = np.array(x2).ravel()
        # diff_ = []
        # for sub_fn, i, j in izip(sub_fns, x1, x2):
        #     diff_.append(sub_fn(i, j))
        # return np.array(diff_)

        x1 = np.array(x1).ravel()
        x2 = np.array(x2).ravel()
        diff = []
        for  i, j in izip(x1, x2):
            diff.append(i-j)
        return np.array(diff)
    
        # return task_type
    # def _set_subgoals(self):

    #     self.goal_position = self.targets_matrix[self.trial_type]

        
    def _while_trial(self):

        #     fb_time = self.feedback_time[self.trial_type][self.subtrial_idx]
        #self.task_data['audio_feedback_start'] = 0
       
        if np.all(np.abs(self.pos_diff(self.goal_position[self.DoF_target_idx_init:self.DoF_target_idx_end],self.plant.get_pos()[self.DoF_target_idx_init:self.DoF_target_idx_end])) < self.target_margin[self.pos_states[self.DoF_target_idx_init:self.DoF_target_idx_end]]):
            #self.task_data['audio_feedback_start'] = 1
            
            # if self.give_feedback:
            #     # self._play_sound(os.path.join(self.sounds_general_dir, 'beep.wav')) # nerea
            #     self._play_sound(self.sounds_general_dir, ['beep'])

            if self.goal_idx < len(self.targets_matrix[self.trial_type].keys())-1:
                pygame.mixer.music.stop()
                self.parallel_sound.stop()
                self.goal_idx +=1

                print 'heading to next subtarget'
                self.goal_position = self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states]
                self._play_sound(self.sounds_dir, self.subgoal_names[self.trial_type][self.goal_idx]) #nerea

                self.parallel_sound = pygame.mixer.Sound(os.path.join(self.sounds_dir_classical, self.subgoal_names[self.trial_type][self.goal_idx][0]+'.wav'))
                self.parallel_sound.play()

              

                # self._play_sound(self.sounds_dir_classical, [self.subgoal_names[self.trial_type][self.goal_idx][0]])  #nerea

            else:
                print 'all subtargets reached'
                self.reached_goal_position = True
                self.task_data['reached_goal_position']  = self.reached_goal_position
        
            

    def _while_trial_return(self):

        #fb_time = self.feedback_time[self.trial_type][self.subtrial_idx]
        #self.task_data['audio_feedback_start'] = 0
           
        if np.all(np.abs(self.pos_diff(self.goal_position[self.DoF_target_idx_init:self.DoF_target_idx_end],self.plant.get_pos()[self.DoF_target_idx_init:self.DoF_target_idx_end])) < self.target_margin[self.pos_states[self.DoF_target_idx_init:self.DoF_target_idx_end]]):
            self.reached_goal_position = True
            self.task_data['reached_goal_position']  = self.reached_goal_position
            
            #self.goal_position = self.rest_position
            #self.task_data['audio_feedback_start'] = 1
            # if self.give_feedback:
            #     # self._play_sound(os.path.join(self.sounds_general_dir, 'beep.wav')) # nerea
            #     self._play_sound(self.sounds_general_dir, ['beep'])
 
    
    def move_plant(self):
        command_vel  = pd.Series(0.0, self.vel_states)
        command_vel_raw  = pd.Series(0.0, self.vel_states)
        command_vel_final  = pd.Series(0.0, self.vel_states)


        #calculate the output of the LQR controller at all states
        current_pos = self.plant_pos[:].ravel()
        current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
        #print self.state
        if self.state in ['wait','rest','rest_return', 'instruct_rest', 'preparation', 'preparation_return', 'instruct_go','instruct_go_return', 'instruct_trial_type', 'instruct_rest_return']:
            #in return state and in the states where the exo does not move the target position is the rest position
            target_state = current_state
  
        elif self.state in ['trial_return', 'instruct_trial_return']:
            #in return state and in the states where the exo does not move the target position is the rest position
            target_state = np.hstack([self.targets_matrix['rest'][0][self.pos_states], np.zeros_like(current_pos),1]).reshape(-1,1)
  
        elif self.state == 'trial':
            target_state = np.hstack([self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states], np.zeros_like(current_pos),1 ]).reshape(-1,1)

        assist_output = self.assister(current_state, target_state, 1.)
        Bu = np.array(assist_output["x_assist"]).ravel()
        # command_vel[:] = Bu[len(current_pos):len(current_pos)*2]
        command_vel_raw[:] = Bu[len(current_pos):len(current_pos)*2]
        #copy the command_vel before fitlering
        # command_vel_raw[:] = command_vel[:]

        #filter command_vel
        for state in self.vel_states:
            # command_vel[state] = self.command_lpfs[state](command_vel[state])
            command_vel[state] = self.command_lpfs[state](command_vel_raw[state])
            if np.isnan(command_vel[state]):
                command_vel[state] = 0

        # Command zero velocity if the task is in a non-moving state
        if self.state in ['wait','rest', 'instruct_rest', 'instruct_trial_type','instruct_trial_return','preparation_return']: 
            command_vel[:] = 0
            # we could also set the raw signal to 0 or just keep the output of the LQR as it is
            # command_vel_raw[:] = 0


        #testing nerea

        # # Command zero velocity if the task is in a non-moving state
        # if self.state in ['wait','rest', 'instruct_rest', 'instruct_trial_type']: 
        #     command_vel[:] = 0

        # elif self.state in ['trial', 'trial_return']: 
        #     current_pos = self.plant_pos[:].ravel()
        #     current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
            
        #     if self.state == 'trial_return':

        #         target_state = np.hstack([self.targets_matrix['rest'][0][self.pos_states], np.zeros_like(current_pos),1]).reshape(-1,1)
  

        #     elif self.state == 'trial':
        #         target_state = np.hstack([self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states], np.zeros_like(current_pos),1 ]).reshape(-1,1)

   
        #     assist_output = self.assister(current_state, target_state, 1.)
        #     Bu = np.array(assist_output["x_assist"]).ravel()

        #     command_vel[:] = Bu[len(current_pos):len(current_pos)*2]
  
        # command_vel_raw[:] = command_vel[:]
        # # #Apply low-pass filter to command velocities
        # for state in self.vel_states:
        #     command_vel[state] = self.command_lpfs[state](command_vel[state])
        #     if np.isnan(command_vel[state]):
        #         command_vel[state] = 0


        # do NOT use this kind of functions in cycle, it causes delays in the reception of position and vel data and the closed-loop control does not work in real time
        # rh_plant.get_enable_state()            

        # motor_res = 1e-3
        # if any(command_vel.values < motor_res) and all(command_vel.values != 0):
        #     print command_vel.values

        self.task_data['command_vel']  = command_vel.values
        self.task_data['command_vel_raw']  = command_vel_raw.values

        # set the velocities of the DoFs that should not be controlled to zero.
        if np.isnan(self.DoF_not_control_idx_init) == False and np.isnan(self.DoF_not_control_idx_end) == False:
            command_vel[self.DoF_not_control_idx_init:self.DoF_not_control_idx_end] = 0

        # self.check_safety(command_vel) #nerea

        self.plant.send_vel(command_vel.values) #send velocity command to EXO 

        self.task_data['command_vel_final']  = command_vel.values

    
    def _cycle(self):
        '''Runs self.fps times per second.'''

        # get latest position/velocity information before calling move_plant()
        self.plant_pos_raw[:] = self.plant.get_pos_raw()
        self.plant_pos[:] = self.plant.get_pos() 

        self.plant_vel_raw[:] = self.plant.get_vel_raw()
        self.plant_vel[:] = self.plant.get_vel()
   
        self.move_plant()

        self.update_plant_display()

        # print self.subtrial_idx
        if not self.state in ['trial','trial_return']:
            #self.task_data['audio_feedback_start'] = 0
            self.task_data['goal_pos'] = np.ones(len(self.pos_states))*np.nan
            self.task_data['goal_idx'] = np.nan
        else:
            self.task_data['goal_pos'] = self.goal_position
            self.task_data['goal_idx'] = self.goal_idx
        
        self.task_data['plant_pos']  = self.plant_pos_raw.values
        self.task_data['plant_pos_filt']  = self.plant_pos.values 
        self.task_data['plant_vel']  = self.plant_vel_raw.values
        self.task_data['plant_vel_filt']  = self.plant_vel.values
        self.task_data['plant_type']  = self.plant_type

        self.task_data['trial_type'] = self.trial_type
        self.task_data['ts']         = time.time()
        self.task_data['target_margin'] = self.target_margin
        self.task_data['reached_goal_position']  = self.reached_goal_position
        self.task_data['speed'] = self.speed
      
        super(EndPointMovement, self)._cycle()

    #### STATE AND TEST FUNCTIONS ####
    def _start_wait(self):
        # try:
        #     self.serial_trigger.setRTS(True)
        # except IOError as e:
        #     print(e)
        # determine the random length of time to stay in the rest state
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time
        min_time, max_time = self.preparation_interval
        self.preparation_time = random.random() * (max_time - min_time) + min_time

        super(EndPointMovement, self)._start_wait()

    def _test_end_rest(self, ts):
        return ts > self.rest_time  

    def _test_end_preparation(self, ts):
        return ts > self.preparation_time  

    def _parse_next_trial(self):
        self.trial_type = self.next_trial

    def _test_end_instruct(self, *args, **kwargs):
        return not pygame.mixer.music.get_busy()

    def _start_instruct_rest(self):
        self.parallel_sound.stop()

        # try:
        #     self.serial_trigger.setRTS(False)
        # except IOError as e:
        #     print(e)
        self._play_sound(self.sounds_dir, ['rest'])
        print 'rest'

    def _start_instruct_trial_type(self):
        self._play_sound(self.sounds_dir, self.subgoal_names[self.trial_type][0])

    def _end_instruct_trial_type(self):
        self.reached_goal_position = False

    def _start_instruct_trial_return(self):
        self._play_sound(self.sounds_dir, ['back'])

    def _end_instruct_trial_return(self):
        self.reached_goal_position = False

    def _start_instruct_go(self):
        self._play_sound(self.sounds_dir, ['go'])
    
    def _start_trial(self):
        self.goal_position = self.targets_matrix[self.trial_type][0][self.pos_states]
        self.goal_idx = 0
        if self.music_feedback:
            self._play_sound(self.sounds_dir_classical, [self.subgoal_names[self.trial_type][self.goal_idx][0]])
       
    def _start_trial_return(self):
        print 'return trial'
        self.goal_position = self.targets_matrix['rest'][0][self.pos_states]
        if self.music_feedback:
            self._play_sound(self.sounds_dir_classical, [self.subgoal_names[self.trial_type][self.goal_idx][0]])

    def _test_end_trial(self, ts):
        return self.reached_goal_position

    def _test_end_trial_return(self,ts):
        return self.reached_goal_position

    def _end_trial(self):
        if self.music_feedback:
            self.parallel_sound.stop()
            pygame.mixer.music.stop()
        else:
            pass

    def _end_trial_return(self):
        if self.music_feedback:
            self.parallel_sound.stop()
            pygame.mixer.music.stop()
        else:
            pass

    # def _test_at_starting_config(self, *args, **kwargs):

    #     start_targ = self.targets_matrix['rest']
    #     diff_to_start = np.abs(self.plant.get_pos() - start_targ[self.pos_states]) 
    #     return np.all(diff_to_start < self.rest_rect[:len(self.pos_states)])


    # def cleanup(self, database, saveid, **kwargs):
    #     self.serial_trigger.close()

class EndPointMovement_testing(NonInvasiveBase):
    '''
    Drives the exo towards previously recorded target positions. 
    Class to make a position control based on target configurations / target positions 
    rather than playing back a previously recorded trajectory.
    '''
    fps = 20
    
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
            'end_instruct': 'preparation',
            'stop':      None},
        'preparation': {
            'end_preparation': 'instruct_go',
            'stop':      None},    
        'instruct_go': {
            'end_instruct': 'trial',
            'stop':      None},
        'trial': {
            # 'end_trial' : 'instruct_rest',
            'end_trial' : 'instruct_trial_type',
            'end_alltrials' : 'wait',
            'stop':      None},    
        }
    
    state = 'wait'  # initial state

    # settable parameters on web interface    
    preparation_time = traits.Float(2, desc='time to remain in the preparation state.')
    rest_interval  = traits.Tuple((3., 4.), desc='Min and max time to remain in the rest state.')
    targets_matrix = traits.DataFile(object,desc='test', bmi3d_query_kwargs=dict(system__name='misc'))
    give_feedback  = traits.Int((0), desc=' 0 if we do not give feedback, 1 if we give feedback about whether the patient reached the goal position')
    music_feedback = traits.Int((1), desc=' 0 if we do not want to include music, 1 if we want different classical music pieces with increasing intensity to be played')
    speed = traits.OptionsList(*speed_options, bmi3d_input_options= speed_options)
    debug = False
    #trial_end_states = ['rest']
    DoF_control = traits.OptionsList(*DoF_control_options, bmi3d_input_options=DoF_control_options, desc='DoFs to be taken into account for condition fulfilment')
    DoF_target = traits.OptionsList(*DoF_target_options, bmi3d_input_options=DoF_target_options, desc='DoFs to be moved/controlled, the rest are stopped.')
   
    sequence_generators = ['B1_targets', 'B2_targets', 'F1_targets','B1_B2_targets','F1_B2_targets', 'B3_targets']

    # Set the "is_bmi_seed" flag so that the server knows that this is a task which can be used to create a BMI decoder
    is_bmi_seed = True

    def __init__(self, *args, **kwargs):
        super(EndPointMovement_testing, self).__init__(*args, **kwargs)

        self.add_dtype('trial_type',   np.str_, 40)
        self.add_dtype('command_vel',  'f8',    (len(self.vel_states),))
        self.add_dtype('command_vel_raw',  'f8',    (len(self.vel_states),))
        self.add_dtype('command_pos',  'f8',    (len(self.pos_states),))
        self.add_dtype('goal_pos',      'f8',    (len(self.pos_states),))
        #self.add_dtype('audio_feedback_start',      int,    (1,))
        self.add_dtype('goal_idx', float, (1,))
        self.add_dtype('reached_goal_position',bool, (1,))
        self.add_dtype('reached_subtarget',bool, (1,))
        self.add_dtype('speed',   np.str_, 20)
        self.add_dtype('command_vel_final',  'f8',    (len(self.vel_states),))


        self.parallel_sound = pygame.mixer.Sound('')
        # self.define_safety() #nerea
        # self.display_safety_area() #nerea
        # print "'safety_area_plant ", self.safety_area_plant
        #area considered as acceptable as rest around the rest position (first position in trial) or the subtask end position
        #self.target_rect = np.array([2., 2., np.deg2rad(10),  np.deg2rad(5), np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)])# for targets during the trial time
        #self.rest_rect = np.array([3., 3., np.deg2rad(10),  np.deg2rad(5), np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)])# for rest position during 'go to start' time
        #more general
        # target margin used for DK calibration sessions
        self.target_margin = pd.Series(np.array([2, 2, np.deg2rad(5), np.deg2rad(3),  np.deg2rad(1), np.deg2rad(1), np.deg2rad(3)]), ismore_pos_states)
        self.target_margin = self.target_margin[self.pos_states]

        self.add_dtype('target_margin',      'f8',    (len(self.target_margin),))    

        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)
        self.sounds_dir_classical = os.path.expandvars('$HOME/code/ismore/sounds/classical')

        self.reached_goal_position = False #If the goal_position is reached then give feedback to patient and start the movement back towards the rest_position
        self.reached_subtarget = False #If the goal_position is reached then give feedback to patient and start the movement back towards the rest_position


        self.plant.enable() 
               
        [self.DoF_target_idx_init, self.DoF_target_idx_end, self.DoF_not_control_idx_init, self.DoF_not_control_idx_end] = check_plant_and_DoFs(self.plant_type, self.DoF_control, self.DoF_target)
        self.subgoal_names = self.targets_matrix['subgoal_names']
        
        # 4th order butterworth filter for command_vel
        fs_synch = self.fps #Frequency at which the task is running

        nyq   = 0.5 * fs_synch
        cuttoff_freq  = 1.5 / nyq
        bpf_kin_coeffs = butter(4, cuttoff_freq, btype='low')
        self.command_lpfs = dict()
        for state in self.vel_states:
            self.command_lpfs[state] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1]) # low-pass filter to smooth out command velocities


        pygame.mixer.init()
        self.goal_idx = 0
        self.trial_type = None
      
    def init(self):
        kwargs = {
            'call_rate': self.fps,
            'xy_cutoff': 2.,
        }
        # self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT[self.plant_type](**kwargs)
        self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT[self.plant_type](speed = self.speed,**kwargs)

        super(EndPointMovement_testing, self).init()


    def pos_diff(self, x1, x2):
        '''
        Calculate x1 - x2, but with a specialized definition of "-"
        '''
        x1 = np.array(x1).ravel()
        x2 = np.array(x2).ravel()
        diff = []
        for  i, j in izip(x1, x2):
            diff.append(i-j)
        return np.array(diff)
    
        
    # def _while_subtrial(self):
    def _while_trial(self):
      
        if np.all(np.abs(self.pos_diff(self.goal_position[self.DoF_target_idx_init:self.DoF_target_idx_end],self.plant.get_pos()[self.DoF_target_idx_init:self.DoF_target_idx_end])) < self.target_margin[self.pos_states[self.DoF_target_idx_init:self.DoF_target_idx_end]]):
            
            if self.give_feedback:
                self._play_sound(self.sounds_general_dir, ['beep'])

            if self.goal_idx < len(self.targets_matrix[self.trial_type].keys())-1:
                self.reached_subtarget = True
                # self.task_data['reached_subtarget']  = self.reached_subtarget
                # print 'heading to next subtarget'

            else:
                # print 'all subtargets reached'
                self.reached_goal_position = True
                # self.task_data['reached_goal_position']  = self.reached_goal_position
        
    
    def move_plant(self):
        command_vel  = pd.Series(0.0, self.vel_states)
        command_vel_raw  = pd.Series(0.0, self.vel_states)
        command_vel_final  = pd.Series(0.0, self.vel_states)


        #calculate the output of the LQR controller at all states
        current_pos = self.plant_pos[:].ravel()
        current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
        #print self.state
        if self.state in ['wait','rest' ,'instruct_rest', 'preparation', 'instruct_go', 'instruct_trial_type']:
            #in return state and in the states where the exo does not move the target position is the rest position
            target_state = current_state
  
        elif self.state == 'trial':
            target_state = np.hstack([self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states], np.zeros_like(current_pos),1 ]).reshape(-1,1)

        assist_output = self.assister(current_state, target_state, 1.)
        Bu = np.array(assist_output["x_assist"]).ravel()

        command_vel_raw[:] = Bu[len(current_pos):len(current_pos)*2]

        #filter command_vel
        for state in self.vel_states:
            # command_vel[state] = self.command_lpfs[state](command_vel[state])
            command_vel[state] = self.command_lpfs[state](command_vel_raw[state])
            if np.isnan(command_vel[state]):
                command_vel[state] = 0

        # Command zero velocity if the task is in a non-moving state
        if self.state in ['wait','rest', 'instruct_rest', 'instruct_trial_type']: 
            command_vel[:] = 0
            # we could also set the raw signal to 0 or just keep the output of the LQR as it is
            # command_vel_raw[:] = 0


        self.task_data['command_vel']  = command_vel.values
        self.task_data['command_vel_raw']  = command_vel_raw.values

        # set the velocities of the DoFs that should not be controlled to zero.
        if np.isnan(self.DoF_not_control_idx_init) == False and np.isnan(self.DoF_not_control_idx_end) == False:
            command_vel[self.DoF_not_control_idx_init:self.DoF_not_control_idx_end] = 0

        # self.check_safety(command_vel) #nerea

        self.plant.send_vel(command_vel.values) #send velocity command to EXO 
        # plants.UDP_PLANT_CLS_DICT['ReHand']().diff_enable([0, 0, 0, 1])
        self.task_data['command_vel_final']  = command_vel.values

    
    def _cycle(self):
        '''Runs self.fps times per second.'''

        # get latest position/velocity information before calling move_plant()
        self.plant_pos_raw[:] = self.plant.get_pos_raw()
        self.plant_pos[:] = self.plant.get_pos() 

        self.plant_vel_raw[:] = self.plant.get_vel_raw()
        self.plant_vel[:] = self.plant.get_vel()
   
        self.move_plant()

        self.update_plant_display()

        # print self.subtrial_idx
        if not self.state in ['trial']:
            #self.task_data['audio_feedback_start'] = 0
            self.task_data['goal_pos'] = np.ones(len(self.pos_states))*np.nan
            self.task_data['goal_idx'] = np.nan
        else:
            self.task_data['goal_pos'] = self.goal_position
            self.task_data['goal_idx'] = self.goal_idx
        
        self.task_data['plant_pos']  = self.plant_pos_raw.values
        self.task_data['plant_pos_filt']  = self.plant_pos.values 
        self.task_data['plant_vel']  = self.plant_vel_raw.values
        self.task_data['plant_vel_filt']  = self.plant_vel.values
        self.task_data['plant_type']  = self.plant_type

        self.task_data['trial_type'] = self.trial_type
        self.task_data['ts']         = time.time()
        self.task_data['target_margin'] = self.target_margin
        self.task_data['reached_goal_position']  = self.reached_goal_position
        self.task_data['reached_subtarget']  = self.reached_subtarget
        self.task_data['speed'] = self.speed
      
        super(EndPointMovement_testing, self)._cycle()

    #### STATE AND TEST FUNCTIONS ####
    def _start_wait(self):
        # determine the random length of time to stay in the rest state
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time

        self.goal_idx = 0
        print "trial type : ", self.trial_type
        super(EndPointMovement_testing, self)._start_wait()

    def _test_end_rest(self, ts):
        return ts > self.rest_time  

    def _test_end_preparation(self, ts):
        return ts > self.preparation_time  

    def _parse_next_trial(self):
        self.trial_type = self.next_trial

    def _test_end_instruct(self, *args, **kwargs):
        return not pygame.mixer.music.get_busy()

    def _start_instruct_rest(self):
        self.parallel_sound.stop()
        self._play_sound(self.sounds_dir, ['rest'])
        print 'rest'

    def _start_instruct_trial_type(self):
        self._play_sound(self.sounds_dir, self.subgoal_names[self.trial_type][self.goal_idx])

    def _end_instruct_trial_type(self):
        self.reached_goal_position = False

    def _start_instruct_go(self):
        self._play_sound(self.sounds_dir, ['go'])
    
    def _test_end_alltrials(self,ts):
        return self.reached_goal_position

    def _end_alltrials(self):
        print 'all trials reached'
        self.task_data['reached_goal_position']  = self.reached_goal_position

    def _start_trial(self):
        print "subtrial : ", self.subgoal_names[self.trial_type][self.goal_idx]
        self.goal_position = self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states]
        if self.music_feedback:
            self._play_sound(self.sounds_dir_classical, [self.subgoal_names[self.trial_type][self.goal_idx][0]])
    
    def _end_trial(self):
        self.reached_subtarget = False
        pygame.mixer.music.stop()
        self.parallel_sound.stop()
        self.goal_idx +=1   
        self.task_data['reached_subtarget']  = self.reached_subtarget
        print 'trial end - heading to next subtarget'

    def _test_end_trial(self, ts):
        return self.reached_subtarget   

    #######################################

    # def _end_trial(self):
    #     return self.reached_goal_position & self.reached_subtarget
    #     # if self.music_feedback:
    #     #     self.parallel_sound.stop()
    #     #     pygame.mixer.music.stop()
    #     # else:
    #     #     pass

class CyclicEndPointMovement(NonInvasiveBase): 
    '''
    Drives the exo towards previously recorded target positions.
    '''
    fps = 20
    
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
            'end_instruct': 'preparation',
            'stop':      None},
        'preparation': {
            'end_preparation': 'instruct_go',
            'stop':      None},    
        'instruct_go': {
            'end_instruct': 'trial',
            'stop':      None},
        'trial': {
            'end_trial': 'wait',#'instruct_trial_go_to_start'
            'stop':      None},    
   
        }

    state = 'wait'  # initial state

    preparation_time = traits.Float(2, desc='time to remain in the preparation state.')
    rest_interval  = traits.Tuple((2., 3.), desc='Min and max time to remain in the rest state.')
    targets_matrix = traits.DataFile(object,desc='test', bmi3d_query_kwargs=dict(system__name='misc'))
    give_feedback  = traits.Int((0,1), desc=' 0 if we do not give feedback, 1 if we give feedback about whether the patient reached the goal position')
    repetitions_cycle  = traits.Int(5, desc='Number of times that the cyclic movement should be repeated')
    speed = traits.OptionsList(*speed_options, bmi3d_input_options= speed_options)
    debug = False
    DoF_control = traits.OptionsList(*DoF_control_options, bmi3d_input_options=DoF_control_options, desc='DoFs to be taken into account for condition fulfilment')
    DoF_target = traits.OptionsList(*DoF_target_options, bmi3d_input_options=DoF_target_options, desc='DoFs to be moved/controlled, the rest are stopped.')
    
    
    sequence_generators = ['targets_circular', 'targets_linear', 'targets_sequence']

    def __init__(self, *args, **kwargs):

        super(CyclicEndPointMovement, self).__init__(*args, **kwargs)

        self.add_dtype('trial_type',   np.str_, 40)
        self.add_dtype('ts',           'f8',    (1,))
        self.add_dtype('command_vel',  'f8',    (len(self.vel_states),))
        self.add_dtype('command_vel_raw',  'f8',    (len(self.vel_states),))
        self.add_dtype('command_pos',  'f8',    (len(self.pos_states),))
        self.add_dtype('goal_pos',      'f8',    (len(self.pos_states),))
        self.add_dtype('goal_vel',      'f8',    (len(self.vel_states),))
        #self.add_dtype('audio_feedback_start',      int,    (1,))
        self.add_dtype('goal_idx', float, (1,))
        self.add_dtype('speed',   np.str_, 20)

        #area considered as acceptable as rest around the rest position (first position in trial) or the subtask end position
        #self.target_rect = np.array([2., 2., np.deg2rad(10),  np.deg2rad(5), np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)])# for targets during the trial time
        #self.rest_rect = np.array([3., 3., np.deg2rad(10),  np.deg2rad(5), np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)])# for rest position during 'go to start' time
        
        #more general
        self.target_margin = pd.Series(np.array([4, 4, np.deg2rad(15), np.deg2rad(2),  np.deg2rad(2), np.deg2rad(2), np.deg2rad(2)]), ismore_pos_states)
        self.target_margin = pd.Series(np.array([2, 2, np.deg2rad(15), np.deg2rad(2),  np.deg2rad(2), np.deg2rad(2), np.deg2rad(2)]), ismore_pos_states)
        self.target_margin = pd.Series(np.array([3, 3, np.deg2rad(15), np.deg2rad(2),  np.deg2rad(2), np.deg2rad(2), np.deg2rad(2)]), ismore_pos_states)
      
        # reduced because proximal and distal points too close to each other and subgoals are reached too fast.

        self.target_margin = self.target_margin[self.pos_states]

        self.subgoal_names = self.targets_matrix['subgoal_names']

        self.add_dtype('target_margin',      'f8',    (len(self.target_margin),))    

        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)
        self.reached_goal_position = False #If the goal_position is reached then give feedback to patient and start the movement back towards the rest_position
        self.plant.enable() 
        [self.DoF_target_idx_init, self.DoF_target_idx_end, self.DoF_not_control_idx_init, self.DoF_not_control_idx_end] = check_plant_and_DoFs(self.plant_type, self.DoF_control, self.DoF_target)

        # 4th order butterworth filter for command_vel
        fs_synch = self.fps #Frequency at which the task is running
        nyq   = 0.5 * fs_synch
        cuttoff_freq  = 1.5 / nyq
        bpf_kin_coeffs = butter(4, cuttoff_freq, btype='low')

        self.command_lpfs = dict()
        for state in self.vel_states:
            self.command_lpfs[state] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1]) # low-pass filter to smooth out command velocities

    def init(self):
        kwargs = {
            'call_rate': self.fps,
            'xy_cutoff': 2.,
        }
        # self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT[self.plant_type](**kwargs)
        self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT[self.plant_type](speed = self.speed,**kwargs)

        super(CyclicEndPointMovement, self).init()


    def pos_diff(self, x1, x2):
        '''
        Calculate x1 - x2, but with a specialized definition of "-"
        '''
        # if self.plant_type == 'ArmAssist':
        #     sub_fns = [operator.sub, operator.sub, angle_subtract]
        # elif self.plant_type == 'ReHand':
        #     sub_fns = [angle_subtract, angle_subtract, angle_subtract, angle_subtract]
        # elif self.plant_type == 'IsMore':
        #     sub_fns = [operator.sub, operator.sub, angle_subtract, angle_subtract, angle_subtract, angle_subtract, angle_subtract]

        # x1 = np.array(x1).ravel()
        # x2 = np.array(x2).ravel()
        # diff_ = []
        # for sub_fn, i, j in izip(sub_fns, x1, x2):
        #     diff_.append(sub_fn(i, j))
        # return np.array(diff_)

        x1 = np.array(x1).ravel()
        x2 = np.array(x2).ravel()
        diff = []
        for  i, j in izip(x1, x2):
            diff.append(i-j)
        return np.array(diff)
        

    def move_plant(self):
       
        command_vel  = pd.Series(0.0, self.vel_states)
        command_vel_raw  = pd.Series(0.0, self.vel_states)

        #calculate the output of the LQR controller at all states
        current_pos = self.plant_pos[:].ravel()
        current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
        
        if self.state in ['wait','rest', 'instruct_rest', 'preparation', 'instruct_go', 'instruct_trial_type']:
            #in return state and in the states where the exo does not move the target position is the rest position
            target_state = current_state
  
        elif self.state == 'trial':
            target_state = np.hstack([self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states + self.vel_states], 1]).reshape(-1,1)
     
        assist_output = self.assister(current_state, target_state, 1.)
        Bu = np.array(assist_output["x_assist"]).ravel()
        # command_vel[:] = Bu[len(current_pos):len(current_pos)*2]
        command_vel_raw[:] = Bu[len(current_pos):len(current_pos)*2]
        #copy the command_vel before fitlering
        # command_vel_raw[:] = command_vel[:]

        #filter command_vel
        for state in self.vel_states:
            # command_vel[state] = self.command_lpfs[state](command_vel[state])
            command_vel[state] = self.command_lpfs[state](command_vel_raw[state])
            if np.isnan(command_vel[state]):
                command_vel[state] = 0

        # Command zero velocity if the task is in a non-moving state
        if self.state in ['wait','rest', 'instruct_rest', 'instruct_trial_type']: 
            command_vel[:] = 0
            # we could also set the raw signal to 0 or just keep the output of the LQR as it is
            # command_vel_raw[:] = 0

        self.plant.send_vel(command_vel.values) #send velocity command to EXO
        #print "vel command:", command_vel.values
        self.task_data['command_vel']  = command_vel.values
        self.task_data['command_vel_raw']  = command_vel_raw.values


    def _while_trial(self):

        #     fb_time = self.feedback_time[self.trial_type][self.subtrial_idx]
        #self.task_data['audio_feedback_start'] = 0
     
        # print 'distance to target', self.pos_diff(self.goal_position[self.pos_states],self.plant.get_pos())
        if np.all(np.abs(self.pos_diff(self.goal_position[self.DoF_target_idx_init:self.DoF_target_idx_end],self.plant.get_pos()[self.DoF_target_idx_init:self.DoF_target_idx_end])) < self.target_margin[self.pos_states[self.DoF_target_idx_init:self.DoF_target_idx_end]]):
            print "approaching goal target"
            #self.task_data['audio_feedback_start'] = 1
            if self.give_feedback:
                # self._play_sound(os.path.join(self.sounds_general_dir, 'beep.wav'))
                self._play_sound(self.sounds_general_dir, ['beep'])
            if self.goal_idx < len(self.targets_matrix[self.trial_type].keys())-1:
                self.goal_idx +=1
                # print "goal position ", self.goal_position
                print 'heading to next subtarget'
                self.goal_position = self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states + self.vel_states]
                print self.goal_position

            elif self.reps < self.repetitions_cycle:
                print 'cycle completed'
                self.reps += 1
                self.goal_position = self.targets_matrix[self.trial_type][0][self.pos_states + self.vel_states]
                self.goal_idx = 0
            else: 
                print 'all cycles completed'
                self.reached_goal_position = True
                self.goal_idx = 0

        
    def _cycle(self):
        '''Runs self.fps times per second.'''

        # get latest position/velocity information before calling move_plant()
        self.plant_pos_raw[:] = self.plant.get_pos_raw()
        self.plant_pos[:] = self.plant.get_pos() 

        self.plant_vel_raw[:] = self.plant.get_vel_raw()
        self.plant_vel[:] = self.plant.get_vel()

        self.move_plant()

        self.update_plant_display()
        

        # print self.subtrial_idx
        if not self.state == 'trial':
            #self.task_data['audio_feedback_start'] = 0
            self.task_data['goal_pos'] = np.ones(len(self.pos_states))*np.nan
            self.task_data['goal_vel'] = np.ones(len(self.vel_states))*np.nan
            self.task_data['goal_idx'] = np.nan
        else:
            self.task_data['goal_pos'] = self.goal_position[self.pos_states]
            self.task_data['goal_idx'] = self.goal_idx
            self.task_data['goal_vel'] = self.goal_position[self.vel_states]

        self.task_data['plant_pos']  = self.plant_pos_raw.values
        self.task_data['plant_pos_filt']  = self.plant_pos.values 

        self.task_data['plant_vel']  = self.plant_vel_raw.values
        self.task_data['plant_vel_filt']  = self.plant_vel.values

        self.task_data['trial_type'] = self.trial_type
        self.task_data['ts']         = time.time()
        self.task_data['target_margin'] = self.target_margin
        self.task_data['speed'] = self.speed
      
        super(CyclicEndPointMovement, self)._cycle()


    #### STATE AND TEST FUNCTIONS ####
    def _start_wait(self):
        # determine the random length of time to stay in the rest state
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time

        super(CyclicEndPointMovement, self)._start_wait()

    def _test_end_rest(self, ts):
        return ts > self.rest_time  

    def _test_end_preparation(self, ts):
        return ts > self.preparation_time  

    def _parse_next_trial(self):
        self.trial_type = self.next_trial

    def _test_end_instruct(self, *args, **kwargs):
        return not pygame.mixer.music.get_busy()

    def _start_instruct_rest(self):
        # self._play_sound(os.path.join(self.sounds_dir, 'rest.wav'))
        self._play_sound(self.sounds_dir, ['rest'])
        print 'rest'

    def _start_instruct_trial_type(self):
        # sound_fname = os.path.join(self.sounds_dir, self.trial_type + '.wav')
        # self._play_sound(sound_fname)
        print "self.subgoal_names[self.trial_type] ", self.subgoal_names[self.trial_type]
        self._play_sound(self.sounds_dir, self.subgoal_names[self.trial_type][0])
        
    def _end_instruct_trial_type(self):
        self.reached_goal_position = False


    def _start_instruct_go(self):
        # self._play_sound(os.path.join(self.sounds_dir, 'go.wav'))
        self._play_sound(self.sounds_dir, ['go'])

    def _start_trial(self):
        #print self.targets_matrix[self.trial_type][0]
        self.goal_position = self.targets_matrix[self.trial_type][0][self.pos_states + self.vel_states]
        self.goal_idx = 0
        self.reps = 1
    

    def _test_end_trial(self, ts):
        return self.reached_goal_position


###############################################################################
########### Tasks based on trajectory recording and playback ##################
###############################################################################

class RecordTrajectoriesBase(NonInvasiveBase):
    '''
    Base class for all tasks involving recording trajectories.
    '''
    
    fps = 20 

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
    # disable_armassist = traits.Int((0,1), desc='0: disable armassist. 1: enable armassist')


    def __init__(self, *args, **kwargs):
        ## Init the pygame mixer for playing back sounds
        super(RecordTrajectoriesBase, self).__init__(*args, **kwargs)

        self.add_dtype('trial_type', np.str_, 40)
        self.add_dtype('trial_accept_reject', np.str_, 10)

        # self.plant.disable() 
   
        # if self.disable_armassist == 0:
        #     aa_plant =  plants.UDP_PLANT_CLS_DICT['ArmAssist']()
        #     aa_plant.enable()

        self.experimenter_acceptance_of_trial = ''
        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)
        
    def _cycle(self):
        '''Runs self.fps times per second.'''
        
        self.plant_pos_raw[:] = self.plant.get_pos_raw()
        self.plant_pos[:] = self.plant.get_pos() 

        self.update_plant_display()
        
        self.plant_vel_raw[:] = self.plant.get_vel_raw()
        self.plant_vel[:] = self.plant.get_vel()

        self.task_data['plant_pos']  = self.plant_pos_raw.values
        self.task_data['plant_pos_filt']  = self.plant_pos.values 

        self.task_data['plant_vel']  = self.plant_vel_raw.values
        self.task_data['plant_vel_filt']  = self.plant_vel.values

        self.task_data['trial_type'] = self.trial_type
        self.task_data['ts']         = time.time()
        self.task_data['trial_accept_reject'] = self.experimenter_acceptance_of_trial

        super(RecordTrajectoriesBase, self)._cycle()

    def _test_end_instruct(self, *args, **kwargs):
        return not pygame.mixer.music.get_busy()

    def _start_instruct_rest(self):
        self._play_sound(os.path.join(self.sounds_dir, 'rest.wav'))
        print 'rest'

    def _start_instruct_trial_type(self):
        sound_fname = os.path.join(self.sounds_dir, self.trial_type + '.wav')
        self._play_sound(sound_fname)

    def _start_trial(self):
        print self.trial_type

    def _test_end_trial(self, ts):
        return self.finished_vel_playback

    def _test_at_starting_config(self, *args, **kwargs):
        traj = self.ref_trajectories[self.trial_type]['traj']
        diff_to_start = np.abs(self.plant.get_pos() - traj[self.pos_states].ix[0].ravel())
        print "diff_to_start: ", diff_to_start
        return np.all(diff_to_start < self.target_rect[:len(self.pos_states)])

    def _end_trial(self):
        pass



###############################################################################
################## Tasks for Trajectory Following Control #####################
###############################################################################

from riglib.bmi.assist import FeedbackControllerAssist
from riglib.bmi.feedback_controllers import FeedbackController
from riglib.bmi.extractor import DummyExtractor
import operator
from itertools import izip

# Initial version of RecordEXG/ not valid for double targets
# class RecordEXG(RecordBrainAmpData,RecordTrajectoriesBase):
#     '''Task class for recording trajectories for the center-out task.'''

#     status = {
#         'wait': {
#             'start_trial': 'rest',
#             'stop': None},
#         'rest': {
#             'end_rest': 'preparation',
#             'stop':      None},
#         'preparation': {
#             'end_preparation': 'trial',
#             'stop':      None},
#         'trial': {
#             'end_trial': 'instruct_trial_return',
#             # 'end_trial': 'rest',
#             'stop':      None},    
#        'instruct_trial_return': {
#            'end_instruct': 'trial_return',
#            'stop':      None},
#        'trial_return': {
#             'end_trial_return': 'wait',
#             'stop':      None},    
#     }
#     state = 'wait'  # initial state
#     # state = 'rest'  # initial state

#     # settable parameters on web interface
#     rest_interval = traits.Tuple((3., 4.), desc='Min and max time to remain in the rest state.')
#     preparation_time    = traits.Float(2,        desc='Time to remain in the ready state.')
#     trial_time    = traits.Float(6,       desc='Time to remain in the trial state.')
#     plant_type = traits.OptionsList('ArmAssist', bmi3d_input_options=plant_type_options, desc='Device connected, data will be acquired from this plant')

#     # def init(self):
#     #     super(RecordB1, self).init()
#     #     import socket
#     #     self.UDP_IP = '192.168.137.3'       
#     #     self.UDP_PORT = 6000
#     #     MESSAGE = "start recording"
#     #     self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     #     self.sock.sendto(MESSAGE, (self.UDP_IP, self.UDP_PORT))
#     #     print "------------------------------------------------------------------start recording"        
#     def init(self):
#         super(RecordEXG, self).init()
#         self.trial_return_time = self.trial_time
#     # def _while_trial(self):
      
#     #     if np.all(np.abs(self.pos_diff(self.goal_position[self.DoF_target_idx_init:self.DoF_target_idx_end],self.plant.get_pos()[self.DoF_target_idx_init:self.DoF_target_idx_end])) < self.target_margin[self.pos_states[self.DoF_target_idx_init:self.DoF_target_idx_end]]):
            
#     #         if self.give_feedback:
#     #             self._play_sound(self.sounds_general_dir, ['beep'])

#     #         if self.goal_idx < len(self.targets_matrix[self.trial_type].keys())-1:
#     #             self.reached_subtarget = True
#     #             # self.task_data['reached_subtarget']  = self.reached_subtarget
#     #             # print 'heading to next subtarget'

#     #         else:
#     #             # print 'all subtargets reached'
#     #             self.reached_goal_position = True
#     #             # self.task_data['reached_goal_position']  = self.reached_goal_position
        
#     def _start_wait(self):
#         # determine the random length of time to stay in the rest state
#         min_time, max_time = self.rest_interval
#         self.rest_time = random.random() * (max_time - min_time) + min_time
#         super(RecordEXG, self)._start_wait()
    
#     #this function runs at the beginning, in the wait state so that the trial_type is already
#     def _parse_next_trial(self): 
#         self.trial_type = self.next_trial

#     def _play_sound(self, fpath,fname):

#         for filename in fname:
#             # print 'filename ', filename
#             if '_' in filename:
#                 filename = filename[:filename.find('_')]
#             sound_fname = os.path.join(fpath, filename + '.wav')
#             pygame.mixer.music.load(sound_fname)
#             pygame.mixer.music.play()

#     def _start_rest(self):
#         print 'rest'
#         # filename = os.path.join(self.sounds_dir, 'rest.wav')
#         # self._play_sound(filename)
#         self._play_sound(self.sounds_dir, ['rest'])

#     def _start_instruct_trial_return(self):
#         print 'back'
#         # filename = os.path.join(self.sounds_dir, 'back.wav')
#         # self._play_sound(filename)
#         self._play_sound(self.sounds_dir, ['back'])

#     # def _end_trial(self):
#     #     # get the next trial type in the sequence
#     #     try:
#     #         self.trial_type = self.gen.next()
#     #         # self.trial_type = self.next_trial
#     #     except StopIteration:
#     #         self.end_task()
        
#     def _test_end_rest(self, ts):
#         return ts > self.rest_time  # and not self.pause -- needed?

#     def _start_preparation(self):
#         print self.trial_type
#         # filename = os.path.join(self.sounds_dir, self.trial_type + '.wav')
#         # self._play_sound(filename)
#         self._play_sound(self.sounds_dir, [self.trial_type]) 

#     def _test_end_preparation(self, ts):
#         return ts > self.preparation_time

#     def _start_trial(self):  
#         print 'go'      
#         filename = os.path.join(self.sounds_dir,'go.wav')
#         # self._play_sound(filename)
#         self._play_sound(self.sounds_dir, ['go'])

#     def _start_trial_return(self):  
#         print 'back'      
#         # filename = os.path.join(self.sounds_dir,'back.wav')
#         # self._play_sound(filename)


#     def _test_end_trial(self, ts):
#         return ts > self.trial_time

#     def _test_end_trial_return(self, ts):        
#         return ts > self.trial_return_time

#     def cleanup(self, database, saveid, **kwargs):
       
#         super(RecordEXG,self).cleanup(database, saveid, **kwargs)


class RecordEXG(RecordBrainAmpData,NonInvasiveBase):
    '''Task class for recording trajectories for the center-out task.'''
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
            'end_instruct': 'preparation',
            'stop':      None},
        'preparation': {
            'end_preparation': 'instruct_go',
            'stop':      None},    
        'instruct_go': {
            'end_instruct': 'trial',
            'stop':      None},
        'trial': {
            # 'end_trial' : 'instruct_rest',
            'end_trial' : 'instruct_trial_type',
            'end_alltrials' : 'wait',
            'stop':      None},    
        }

    state = 'wait'  # initial state
    # state = 'rest'  # initial state

    # settable parameters on web interface
    rest_interval = traits.Tuple((3., 4.), desc='Min and max time to remain in the rest state.')
    preparation_time    = traits.Float(2,        desc='Time to remain in the ready state.')
    subtrial_time    = traits.Float(7,       desc='Time to remain in the trial state.')
    plant_type = traits.OptionsList('ArmAssist', bmi3d_input_options=plant_type_options, desc='Device connected, data will be acquired from this plant')
    targets_matrix = traits.DataFile(object,desc='test', bmi3d_query_kwargs=dict(system__name='misc'))
    music_feedback = traits.Int((1), desc=' 0 if we do not want to include music, 1 if we want different classical music pieces with increasing intensity to be played')
    give_feedback  = traits.Int((0), desc=' 0 if we do not give feedback, 1 if we give feedback about whether the patient reached the goal position')

    sequence_generators = ['B1_targets', 'B2_targets', 'F1_targets','B1_B2_targets','F1_B2_targets', 'B3_targets']
#is_bmi_seed = True
    def __init__(self, *args, **kwargs):
        super(RecordEXG, self).__init__(*args, **kwargs)

        self.add_dtype('trial_type',   np.str_, 40)

        self.add_dtype('ts',           'f8',    (1,))
        
        #self.add_dtype('goal_pos',      'f8',    (len(self.pos_states),))
        #self.add_dtype('audio_feedback_start',      int,    (1,))
        self.add_dtype('goal_idx', float, (1,))
        self.add_dtype('reached_goal_position',bool, (1,))
        self.add_dtype('reached_subtarget',bool, (1,))

        self.parallel_sound = pygame.mixer.Sound('')
        # self.display_safety_area() #nerea
        # print "'safety_area_plant ", self.safety_area_plant
        #area considered as acceptable as rest around the rest position (first position in trial) or the subtask end position
        #self.target_rect = np.array([2., 2., np.deg2rad(10),  np.deg2rad(5), np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)])# for targets during the trial time
        #self.rest_rect = np.array([3., 3., np.deg2rad(10),  np.deg2rad(5), np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)])# for rest position during 'go to start' time
        #more general
        # target margin used for DK calibration sessions
        
        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)
        self.sounds_dir_classical = os.path.expandvars('$HOME/code/ismore/sounds/classical')

        self.reached_goal_position = False #If the goal_position is reached then give feedback to patient and start the movement back towards the rest_position
        self.reached_subtarget = False #If the goal_position is reached then give feedback to patient and start the movement back towards the rest_position

        self.subtrial_time_finished = False
      
        self.subgoal_names = self.targets_matrix['subgoal_names']
        
       

        pygame.mixer.init()
        self.goal_idx = 0
        self.trial_type = None

    def init(self):
        super(RecordEXG, self).init()
        #self.trial_return_time = self.trial_time
    def _while_trial(self):
        
        if self.subtrial_time_finished:
            
            if self.give_feedback:
                self._play_sound(self.sounds_general_dir, ['beep'])

            if self.goal_idx < len(self.targets_matrix[self.trial_type].keys())-1:
                self.reached_subtarget = True
                # self.task_data['reached_subtarget']  = self.reached_subtarget
                # print 'heading to next subtarget'

            else:
                # print 'all subtargets reached'
                self.reached_goal_position = True
                # self.task_data['reached_goal_position']  = self.reached_goal_position
    def _cycle(self):
        '''Runs self.fps times per second.'''

        # get latest position/velocity information before calling move_plant()
       
   

        # print self.subtrial_idx
        if not self.state in ['trial']:
            #self.task_data['audio_feedback_start'] = 0
           # self.task_data['goal_pos'] = np.ones(len(self.pos_states))*np.nan
            self.task_data['goal_idx'] = np.nan
        else:
          #  self.task_data['goal_pos'] = self.goal_position
            self.task_data['goal_idx'] = self.goal_idx
        
        

        self.task_data['trial_type'] = self.trial_type
        self.task_data['ts']         = time.time()
        self.task_data['reached_goal_position']  = self.reached_goal_position
        self.task_data['reached_subtarget']  = self.reached_subtarget
      
        super(RecordEXG, self)._cycle()

    #### STATE AND TEST FUNCTIONS ####
    def _start_wait(self):
        # determine the random length of time to stay in the rest state
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time

        self.goal_idx = 0
        print "trial type : ", self.trial_type
        super(RecordEXG, self)._start_wait()

    def _test_end_rest(self, ts):
        return ts > self.rest_time  

    def _test_end_preparation(self, ts):
        return ts > self.preparation_time  

    def _parse_next_trial(self):
        self.trial_type = self.next_trial

    def _test_end_instruct(self, *args, **kwargs):
        return not pygame.mixer.music.get_busy()

    def _start_instruct_rest(self):
        self.parallel_sound.stop()
        self._play_sound(self.sounds_dir, ['rest'])
        print 'rest'

    def _start_instruct_trial_type(self):
        self._play_sound(self.sounds_dir, self.subgoal_names[self.trial_type][self.goal_idx])

    def _end_instruct_trial_type(self):
        self.reached_goal_position = False

    def _start_instruct_go(self):
        self._play_sound(self.sounds_dir, ['go'])
    
    def _test_end_alltrials(self,ts):
        return self.reached_goal_position

    def _end_alltrials(self):
        print 'all trials reached'
        self.task_data['reached_goal_position']  = self.reached_goal_position

    def _start_trial(self):
        print "subtrial : ", self.subgoal_names[self.trial_type][self.goal_idx]
        #self.goal_position = self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states]
        if self.music_feedback:
            self._play_sound(self.sounds_dir_classical, [self.subgoal_names[self.trial_type][self.goal_idx][0]])
    
    def _end_trial(self):
        self.reached_subtarget = False
        self.subtrial_time_finished = False
        pygame.mixer.music.stop()
        self.parallel_sound.stop()
        self.goal_idx +=1   
        self.task_data['reached_subtarget']  = self.reached_subtarget
        print 'trial end - heading to next subtarget'

    def _test_end_trial(self, ts):
        if ts > self.subtrial_time:
            self.subtrial_time_finished = True
        return self.reached_subtarget   

  

    def _play_sound(self, fpath,fname):

        for filename in fname:
            # print 'filename ', filename
            if '_' in filename:
                filename = filename[:filename.find('_')]
            sound_fname = os.path.join(fpath, filename + '.wav')
            pygame.mixer.music.load(sound_fname)
            pygame.mixer.music.play()


    def cleanup(self, database, saveid, **kwargs):
       
        super(RecordEXG,self).cleanup(database, saveid, **kwargs)



# For cyclic movements
# class RecordEXG(RecordBrainAmpData,RecordTrajectoriesBase):
#     '''Task class for recording trajectories for the center-out task.'''

#     status = {
#         'wait': {
#             'start_trial': 'rest',
#             'stop': None},
#         'rest': {
#             'end_rest': 'preparation',
#             'stop':      None},
#         'preparation': {
#             'end_preparation': 'trial',
#             'stop':      None},
#         'trial': {
#             # 'end_trial': 'instruct_trial_return',
#             'end_trial': 'rest',
#             'stop':      None},    
#        # 'instruct_trial_return': {
#        #     'end_instruct': 'trial_return',
#        #     'stop':      None},
#        # 'trial_return': {
#        #      'end_trial_return': 'wait',
#        #      'stop':      None},    
#     }
#     state = 'wait'  # initial state
#     # state = 'rest'  # initial state

#     # settable parameters on web interface
#     rest_interval = traits.Tuple((3., 4.), desc='Min and max time to remain in the rest state.')
#     preparation_time    = traits.Float(2,        desc='Time to remain in the ready state.')
#     trial_time    = traits.Float(7,       desc='Time to remain in the trial state.')
#     plant_type = traits.OptionsList('ArmAssist', bmi3d_input_options=plant_type_options, desc='Device connected, data will be acquired from this plant')

#     # def init(self):
#     #     super(RecordB1, self).init()
#     #     import socket
#     #     self.UDP_IP = '192.168.137.3'       
#     #     self.UDP_PORT = 6000
#     #     MESSAGE = "start recording"
#     #     self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     #     self.sock.sendto(MESSAGE, (self.UDP_IP, self.UDP_PORT))
#     #     print "------------------------------------------------------------------start recording"        
#     def init(self):
#         super(RecordEXG, self).init()
#         self.trial_return_time = self.trial_time
    
#     def _start_wait(self):
#         # determine the random length of time to stay in the rest state
#         min_time, max_time = self.rest_interval
#         self.rest_time = random.random() * (max_time - min_time) + min_time
#         super(RecordEXG, self)._start_wait()
    
#     #this function runs at the beginning, in the wait state so that the trial_type is already
#     def _parse_next_trial(self): 
#         self.trial_type = self.next_trial

#     def _play_sound(self, fname):
#         pygame.mixer.music.load(fname)
#         pygame.mixer.music.play()

#     def _start_rest(self):
#         print 'rest'
#         filename = os.path.join(self.sounds_dir, 'rest.wav')
#         self._play_sound(filename)

#     def _start_instruct_trial_return(self):
#         print 'back'
#         filename = os.path.join(self.sounds_dir, 'back.wav')
#         self._play_sound(filename)

#     def _end_trial(self):
#         # get the next trial type in the sequence
#         try:
#             self.trial_type = self.gen.next()
#             # self.trial_type = self.next_trial
#         except StopIteration:
#             self.end_task()
        
#     def _test_end_rest(self, ts):
#         return ts > self.rest_time  # and not self.pause -- needed?

#     def _start_preparation(self):
#         print self.trial_type
#         filename = os.path.join(self.sounds_dir, self.trial_type + '.wav')
#         self._play_sound(filename)

#     def _test_end_preparation(self, ts):
#         return ts > self.preparation_time

#     def _start_trial(self):  
#         print 'go'      
#         filename = os.path.join(self.sounds_dir,'go.wav')
#         self._play_sound(filename)

#     # def _start_trial_return(self):  
#     #     print 'back'      
#     #     filename = os.path.join(self.sounds_dir,'back.wav')
#     #     self._play_sound(filename)

#     def _test_end_trial(self, ts):
#         return ts > self.trial_time

#     # def _test_end_trial_return(self, ts):        
#     #     return ts > self.trial_return_time

#     def cleanup(self, database, saveid, **kwargs):
       
#         super(RecordEXG,self).cleanup(database, saveid, **kwargs)

###############################################################################
################## Tasks for Trajectory Following Control #####################
###############################################################################



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

        #for this case import filter from riglib
        from riglib.filter import Filter
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

        
        # return a "fake" next state where the position is not updated (since it's not a tracked variable anyway..)
        ns = current_state.copy()
        ns[3:6,0] = playback_vel.reshape(-1,1)
        return ns


