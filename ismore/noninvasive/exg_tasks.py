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

import serial

from riglib.experiment import traits, Sequence, generate, FSMTable, StateTransitions
from riglib.stereo_opengl.window import WindowDispl2D
from riglib.stereo_opengl.primitives import Circle, Sector, Line
from riglib.bmi import clda, extractor, train
from riglib.bmi.bmi import Decoder, BMISystem, GaussianStateHMM, BMILoop, GaussianState, MachineOnlyFilter

from ismore import plants, settings, ismoretasks
from ismore.common_state_lists import *
from features.bmi_task_features import LinearlyDecreasingAssist, LinearlyDecreasingHalfLife
from features.simulation_features import SimTime #, SimHDF
from ismore import ismore_bmi_lib
from utils.angle_utils import *
from utils.util_fns import *
from utils.constants import *
from ismore.noninvasive.emg_decoding import LinearEMGDecoder
from ismore.noninvasive.eeg_decoding import LinearEEGDecoder 
from ismore.noninvasive.emg_classification import SVM_EMGClassifier
from ismore.invasive.bmi_ismoretasks import PlantControlBase

#from ismore.ismore_tests.eeg_decoding import LinearEEGDecoder #uncomment this if we want to use the SimEEGMovementDecoding class

from ismore.ismoretasks import NonInvasiveBase, RecordBrainAmpData, IsMoreBase, EndPointMovement, EndPointMovement_testing, CyclicEndPointMovement
# from ismore.ismoretasks import PlaybackTrajectories, SimRecordBrainAmpData #uncomment if we want to use them

from ismore.ismoretasks import plant_type_options

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
except:
    from sklearn.lda import LDA

from db.tracker import models


from utils.ringbuffer import RingBuffer
from itertools import izip

from features.generator_features import Autostart

import pygame


from riglib.plants import RefTrajectories
from ismore.filter import Filter
from scipy.signal import butter,lfilter
from ismore import brainamp_channel_lists
from ismore.ismoretasks import check_plant_and_DoFs

from utils.constants import *

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

clda_update_methods = ['RML', 'Smoothbatch', ]
languages_list = ['english', 'deutsch', 'castellano', 'euskara']
speed_options = ['very-low','low', 'medium','high']
channel_list_options = brainamp_channel_lists.channel_list_options
DoF_control_options = ['IsMore','ArmAssist', 'ReHand', 'ReHand-Pronosup', 'ReHand-Pronosup-FingersDisabled']
DoF_target_options  = ['IsMore','ArmAssist', 'ReHand', 'ReHand-Pronosup', 'ReHand-Pronosup-FingersDisabled']
#######################################################################


class EEG_Screening(RecordBrainAmpData, Sequence):
    #needs to inherit from RecordBrainAmpData first to run the init of Autostart before than the init of Sequence
    fps = 20
    #fps = rda.samplingFrequencyBrainamp #would this work????

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
            'end_trial': 'wait',
            'stop':      None},    
    }
    state = 'wait'  # initial state

    rest_interval = traits.Tuple((3., 4.), desc='Min and max time to remain in the rest state.')
    preparation_time    = traits.Float(2,        desc='Time to remain in the preparation state.')
    trial_time    = traits.Float(5,       desc='Time to remain in the trial state.') 
    language = traits.OptionsList(*languages_list, bmi3d_input_options=languages_list)
    sequence_generators = ['OPEN_CLOSE_targets']

    @staticmethod
    def _make_block_rand_targets(length, available_targets, shuffle = False):
        targets = []
        for k in range(length):
            a_ = available_targets[:]
            if shuffle:
                random.shuffle(a_)
            targets += a_
        return targets

    @staticmethod
    def OPEN_CLOSE_targets(length=8, right=1, left=1, relax=1, shuffle = 1):
        available_targets = []
        if right: available_targets.append('right')
        if left: available_targets.append('left')
        if relax: available_targets.append('relax')

        targets = EEG_Screening._make_block_rand_targets(length, available_targets, shuffle = shuffle)
        return targets  

    def _play_sound(self, fname):
        pygame.mixer.music.load(fname)
        pygame.mixer.music.play()

    def show_image(self, image_fname):
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (self.monitors[self.active_monitor].width ,0)

        window = pygame.display.set_mode(self.window_size,pygame.NOFRAME)
     
        img = pygame.image.load(os.path.join(self.image_fname))
        img = pygame.transform.scale(img, self.window_size)

        window.blit(img, (0,0))
        pygame.display.flip()

    def __init__(self, *args, **kwargs):
        ## Init the pygame mixer for playing back sounds
        super(EEG_Screening, self).__init__(*args, **kwargs)
        self.add_dtype('trial_type', np.str_, 40)
        self.add_dtype('ts',         'f8',    (1,))
        pygame.mixer.init()

        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)

        self.image_dir_general = os.path.expandvars('$HOME/code/ismore/images')
        self.image_dir = os.path.join(self.image_dir_general, self.language)
        
        self.channels = [chan + '_filt' for chan in self.brainamp_channels]

        #import here because when importing at the beginning of the script it gives an error if you stop an experiment and run it again without rerunning the server
        from gi.repository import Gdk, Gtk
        window = Gtk.Window() # Replace w with the GtkWindow of your application
        s = window.get_screen() # Get the screen from the GtkWindow

        # collect data about each monitor
        self.monitors = []
        nmons = s.get_n_monitors()
        for m in range(nmons):
            mg = s.get_monitor_geometry(m)
            self.monitors.append(mg)

        # Using the screen of the Window, the monitor it's on can be identified
        self.active_monitor = s.get_monitor_at_window(s.get_active_window())

        if nmons ==2:
            #considering 2 monitors connected
            if (self.active_monitor == 1):
                self.feedback_monitor = 0
            elif (self.active_monitor ==0):
                self.feedback_monitor =1
        else:
            self.feedback_monitor =0
        print "feedback_monitor: ", self.feedback_monitor

        #set the size of the window where the visual stimuli will be presented to the size of the screen
        self.window_size = [self.monitors[self.feedback_monitor].width ,self.monitors[self.feedback_monitor].height ]
        


        #self.window_size = [monitors[active_monitor].width, monitors[active_monitor].height - 50]
        # self.window_size = [monitors[active_monitor].width, monitors[active_monitor].height]

        # self.serial_trigger =serial.Serial(
        #     port='/dev/ttyUSB0',
        #     baudrate=9600,
        #     parity=serial.PARITY_NONE,
        #     stopbits=serial.STOPBITS_ONE,
        #     bytesize=serial.SEVENBITS
        #     )

        

    def _cycle(self):
        '''Runs self.fps times per second.'''
        try:
            self.task_data['trial_type'] = self.trial_type
        except:
            ''
        self.task_data['ts']         = time.time()

        super(EEG_Screening, self)._cycle()

    def _test_end_instruct(self, *args, **kwargs):
        return not pygame.mixer.music.get_busy()

    def _start_instruct_rest(self):
        self._play_sound(os.path.join(self.sounds_dir, 'rest.wav'))

    def _start_rest(self):
        #determine the random length of time to stay in the rest state
        # try:
        #     self.serial_trigger.setRTS(True)
        # except IOError as e:
        #     print(e)
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time

    # def _end_rest(self):
    #     try:
    #         self.serial_trigger.setRTS(False)
    #     except IOError as e:
    #         print(e)


    def _start_instruct_go(self):
        sound_fname = os.path.join(self.sounds_dir,'go.wav')
        self._play_sound(sound_fname)

    # def _start_trial(self): 
    #     try:
    #         self.serial_trigger.setRTS(True)
    #     except IOError as e:
    #         print(e)

    # def _end_trial(self):
    #     try:
    #         self.serial_trigger.setRTS(False)
    #     except IOError as e:
    #         print(e)
    # def _start_preparation(self):
    #     min_time, max_time = self.preparation_time

    # def _while_wait(self):
    #     self.image_fname = os.path.join(self.image_dir_general, 'rest.bmp')
    #     self.show_image(self.image_fname)
    #     time.sleep(3)

    def _test_end_preparation(self, ts):
        return ts > self.preparation_time

    #do we also need an auditory cue for the trial tasks or just visual? for now set to play Go cue
    def _start_instruct_trial_type(self):        
        sound_fname = os.path.join(self.sounds_dir, self.trial_type + '.wav')
        self._play_sound(sound_fname)
    
    def _parse_next_trial(self):
        self.trial_type = self.next_trial

    def _test_end_rest(self, ts):
        return ts > self.rest_time  # and not self.pause -- needed?

    def _start_trial(self):
        print self.trial_type

    def _test_end_trial(self, ts):
        return ts > self.trial_time

    def _while_instruct_rest(self):
        self.image_fname = os.path.join(self.image_dir_general, 'rest.bmp')
        self.show_image(self.image_fname)

    def _while_rest(self):
        self.image_fname = os.path.join(self.image_dir_general, 'rest.bmp')
        self.show_image(self.image_fname)

    def _while_instruct_preparation(self):
        self.image_fname = os.path.join(self.image_dir_general, self.trial_type + '.bmp')
        self.show_image(self.image_fname)

    def _while_preparation(self):
        self.image_fname = os.path.join(self.image_dir_general, self.trial_type  + '.bmp')
        self.show_image(self.image_fname)

    def _while_instruct_trial_type(self):
        self.image_fname = os.path.join(self.image_dir_general, self.trial_type + '.bmp')
        self.show_image(self.image_fname)

    def _while_trial(self):
        self.image_fname = os.path.join(self.image_dir_general, self.trial_type + '.bmp')
        self.show_image(self.image_fname)

class RecordExGData(RecordBrainAmpData, Sequence):
    #needs to inherit from RecordBrainAmpData first to run the init of Autostart before than the init of Sequence
    fps = 20
    #fps = rda.samplingFrequencyBrainamp #would this work????

    status = {
        'wait': {
            'start_trial': 'instruct_rest',
            'stop': None},
        'instruct_rest': {
            'end_instruct': 'rest',
            'stop':      None},            
        'rest': {
            'end_rest': None,
            'stop':      None},
    }
    state = 'wait'  # initial state

    rest_time = traits.Float(300, desc='Min and max time to remain in the rest state.')
    language = traits.OptionsList(*languages_list, bmi3d_input_options=languages_list)
  
    def _play_sound(self, fname):
        pygame.mixer.music.load(fname)
        pygame.mixer.music.play()

    def __init__(self, *args, **kwargs):
        ## Init the pygame mixer for playing back sounds
        super(RecordExGData, self).__init__(*args, **kwargs)
        self.add_dtype('trial_type', np.str_, 40)
        self.add_dtype('ts',         'f8',    (1,))
        pygame.mixer.init()

        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)

        self.image_dir_general = os.path.expandvars('$HOME/code/ismore/images')
        self.image_dir = os.path.join(self.image_dir_general, self.language)
        
        self.channels = [chan + '_filt' for chan in self.brainamp_channels]
        
        #import here because when importing at the beginning of the script it gives an error if you stop an experiment and run it again without rerunning the server
        from gi.repository import Gdk, Gtk
        window = Gtk.Window() # Replace w with the GtkWindow of your application
        s = window.get_screen() # Get the screen from the GtkWindow

        # collect data about each monitor
        monitors = []
        nmons = s.get_n_monitors()
        for m in range(nmons):
            mg = s.get_monitor_geometry(m)
            monitors.append(mg)

        # Using the screen of the Window, the monitor it's on can be identified
        active_monitor = s.get_monitor_at_window(s.get_active_window())

        if nmons ==2:
            #considering 2 monitors connected
            if (active_monitor == 1):
                feedback_monitor = 0
            elif (active_monitor ==0):
                feedback_monitor =1
        else:
            feedback_monitor =0
        print "feedback_monitor: ", feedback_monitor

        #set the size of the window where the visual stimuli will be presented to the size of the screen
        self.window_size = [monitors[feedback_monitor].width, monitors[feedback_monitor].height - 50]
        #self.window_size = [monitors[active_monitor].width, monitors[active_monitor].height ]

        # self.serial_trigger =serial.Serial(
        #     port='/dev/ttyUSB0',
        #     baudrate=9600,
        #     parity=serial.PARITY_NONE,
        #     stopbits=serial.STOPBITS_ONE,
        #     bytesize=serial.SEVENBITS
        #     )

    def _cycle(self):
        '''Runs self.fps times per second.'''
        try:
            self.task_data['trial_type'] = self.trial_type
        except:
            ''
        self.task_data['ts']         = time.time()

        super(RecordExGData, self)._cycle()

    def _test_end_instruct(self, *args, **kwargs):
        return not pygame.mixer.music.get_busy()

    def _start_instruct_rest(self):
        self._play_sound(os.path.join(self.sounds_dir, 'rest.wav'))

    def _test_end_rest(self, ts):
        return ts > self.rest_time  
   
    def _while_instruct_rest(self):
        self.image_fname = os.path.join(self.image_dir_general, 'pos2.bmp')
        self.show_image(self.image_fname)

    def _while_rest(self):
        self.image_fname = os.path.join(self.image_dir_general, 'pos2.bmp')
        self.show_image(self.image_fname)


    def show_image(self, image_fname):
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (1,1)

        window = pygame.display.set_mode(self.window_size)
      
        img = pygame.image.load(os.path.join(self.image_fname))
        img = pygame.transform.scale(img, self.window_size)

        window.blit(img, (0,0))
        pygame.display.flip()

    ##might be useful if we wanna set the size of the screen depending on the screen we are using
    #from gi.repository import Gdk
    #screen_size = Gdk.Screen.get_default()
    #window_size = tuple((screen_size.get_width(), screen_size.get_height()))

    # def cleanup(self, database, saveid, **kwargs):
    #     self.serial_trigger.close()

class EEGMovementDecoding(NonInvasiveBase):
    
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
            'timeout': 'instruct_trial_return',#'instruct_trial_go_to_start'
            'end_trial': 'instruct_trial_return',#'instruct_trial_go_to_start'
            'stop':      None},    
    #If we wanna include always a "return trial" to go to the initial position after the target trial then one option would be to add this and use instruct_trial_go_to_start instead of wait at the previous state:
        'instruct_trial_return': {
            'end_instruct': 'trial_return',
            'stop':      None},
        'trial_return': {
            'timeout': 'wait',#'instruct_rest'
            'end_trial': 'wait',
            'stop':      None},    
        }
    
    state = 'wait'  # initial state

     # settable parameters on web interface    
    eeg_decoder          = traits.InstanceFromDB(LinearEEGDecoder, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='eeg_decoder'))
    rest_interval        = traits.Tuple((5., 6.), desc='Min and max time to remain in the rest state.')
    preparation_time     = traits.Float(2, desc='time to remain in the preparation state.')
    timeout_time         = traits.Float(30, desc='Maximum time given to the patient to accomplish the task before considering it like incomplete and re-starting it from the current position') 
    give_feedback        = traits.Int((0,1), desc=' 0 if we do not give feedback, 1 if we give feedback about whether the patient reached the goal position')
    targets_matrix       = traits.DataFile(object, desc='goal positions for each of the trial types', bmi3d_query_kwargs=dict(system__name='misc'))
    window_size = traits.Tuple((1000, 560), desc='Size of window to display the plant position/angle')
    channel_list_name = traits.OptionsList(*channel_list_options, bmi3d_input_options=channel_list_options) #choose always the filtered + raw option!!!!!!
    speed = traits.OptionsList(*speed_options, bmi3d_input_options= speed_options)
    #neighbour_channels = ???
    debug = False

    # Set the "is_bmi_seed" flag so that the server knows that this is a task which can be used to create a BMI decoder
    #is_bmi_seed = False
    def _play_sound(self, fname):
        pygame.mixer.music.load(fname)
        pygame.mixer.music.play()

    def __init__(self, *args, **kwargs):
        super(EEGMovementDecoding, self).__init__(*args, **kwargs)
        
        self.add_dtype('command_vel',  'f8',    (len(self.vel_states),))
        self.add_dtype('command_vel_raw',  'f8',    (len(self.vel_states),))
        self.add_dtype('trial_type',   np.str_, 40)
        # self.add_dtype('plant_type',   np.str_, 40)
        # self.add_dtype('ts',           'f8',    (1,))
        self.add_dtype('goal_pos',      'f8',    (len(self.pos_states),))
        #self.add_dtype('difference_position','f8', (len(self.pos_states),))
        self.add_dtype('reached_goal_position',bool, (1,))
        self.add_dtype('reached_subtarget',bool, (1,))
        #self.add_dtype('audio_feedback_start',      int,    (1,))
        self.add_dtype('goal_idx', float, (1,))
        self.add_dtype('speed',   np.str_, 20)


      
        # if EEG decoder file was specified, load EEG decoder and create feature extractor 
        
        #if len(self.decoder_file) > 3:
        #    self.eeg_decoder = pickle.load(open(self.decoder_file, 'rb'))

        # create EEG extractor object (its 'source' will be set later in the init method)
      
        eeg_extractor_cls    = self.eeg_decoder.extractor_cls
        eeg_extractor_kwargs = self.eeg_decoder.extractor_kwargs
        self.rest_feature_buffer = self.eeg_decoder.rest_feature_buffer
        self.mov_feature_buffer = self.eeg_decoder.mov_feature_buffer
        #self.channels = eeg_extractor_kwargs['channels']
        #eeg_extractor_kwargs['brainamp_channels'] = getattr(brainamp_channel_lists, self.channel_list_name)  
        try:
            self.channels = self.eeg_extractor_kwargs['eeg_channels']
        except:
            self.channels = self.eeg_extractor_kwargs['channels']
        self.brainamp_channels = getattr(brainamp_channel_lists, self.channel_list_name)  

        self.neighbour_channels = self.eeg_decoder.neighbour_channels
      
        self.eeg_playback = False
        self.fs = eeg_extractor_kwargs['fs']

        self.retrained_decoder = copy.copy(self.eeg_decoder.decoder)
        
        

        self.eeg_extractor = eeg_extractor_cls(source=None, **eeg_extractor_kwargs)
        self.n_features = self.eeg_extractor.n_features
        #self.add_dtype('eeg_features',    'f8', (self.eeg_extractor.n_features,))
        self.add_dtype('eeg_features',    'f8', (self.n_features,))
        self.add_dtype('eeg_features_Z',    'f8', (self.n_features,))
        self.add_dtype('eeg_mean_features',    'f8', (self.n_features,))
        self.add_dtype('eeg_std_features',    'f8', (self.n_features,))
        self.add_dtype('eeg_decoder_coef',    'f8', (self.n_features,))
        self.add_dtype('eeg_decoder_intercept', 'f8', (self.n_features,))
        self.add_dtype('eeg_decoder_means', 'f8', (2,self.n_features))
        self.add_dtype('decoder_output',    'f8', (1,))
        self.add_dtype('decoder_output_probability',    'f8', (1,2))#2 classes
        self.add_dtype('state_decoder',  int, (1,))
        #self.add_dtype('decoder', InstanceFromDB(LinearEEGDecoder))

        
        # for low-pass filtering decoded EEG velocities
        # self.eeg_vel_buffer = RingBuffer(
        #     item_len=len(self.vel_states),
        #     capacity=10,
        # )

        # 4th order butterworth filter for command_vel
        fs_synch = self.fps #Frequency at which the task is running
        nyq   = 0.5 * fs_synch
        cuttoff_freq  = 1.5 / nyq
        bpf_kin_coeffs = butter(2, cuttoff_freq, btype='low')

        self.command_lpfs = dict()
        for state in self.vel_states:
            self.command_lpfs[state] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1]) # low-pass filter to smooth out command velocities



        self.plant.enable() 
        #initialize values for the state of the decoder
        self.consec_mov_outputs = 0
        self.consec_rest_outputs = 0
        self.prev_output = 0
        self.state_decoder = 0


        # if self.plant_type == 'ArmAssist':
        #     self.target_margin = np.array([2, 2, np.deg2rad(10)])
        # elif self.plant_type == 'ReHand':
        #     self.target_margin = np.array([np.deg2rad(2),  np.deg2rad(2), np.deg2rad(2), np.deg2rad(2)])
        # elif self.plant_type == 'IsMore':
        #     self.target_margin = np.array([2, 2, np.deg2rad(10), np.deg2rad(2),  np.deg2rad(2), np.deg2rad(2), np.deg2rad(2)])
          
        self.target_margin = pd.Series(np.array([3, 3, np.deg2rad(15), np.deg2rad(5),  np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)]), ismore_pos_states)
        self.target_margin = self.target_margin[self.pos_states]



        self.add_dtype('target_margin',      'f8',    (len(self.target_margin),))
        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)

        self.image_dir_general = os.path.expandvars('$HOME/code/ismore/images')
        self.image_dir = os.path.join(self.image_dir_general, self.language)
        

        self.reached_goal_position = False #If the goal_position is reached then give feedback to patient and start the movement back towards the rest_position
        self.reached_subtarget = False
        self.init_show_decoder_output()

    def init(self):
        kwargs = {
            'call_rate': self.fps, #kwargs used by the assister
            'xy_cutoff': 2.,#What is this for? Radius of margin?
        }

        from riglib import source
        from ismore.brainamp import rda

        #self.brainamp_source = source.MultiChanDataSource(rda.EMGData, name='brainamp', channels=self.brainamp_channels, brainamp_channels=self.brainamp_channels, send_data_to_sink_manager=True)
        # self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT[self.plant_type](**kwargs)
        self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT[self.plant_type](speed = self.speed,**kwargs)
        #print "before brainamp_source"
        self.eeg_extractor.source = self.brainamp_source
        #print "brainamp_source", self.brainamp_source
        super(EEGMovementDecoding, self).init()

    # def _set_goal_position(self):
    #     self.goal_position = self.targets_matrix[self.trial_type]

        

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

        # if self.give_feedback == 1:
        # #     fb_time = self.feedback_time[self.trial_type][self.subtrial_idx]
        #     self.task_data['audio_feedback_start'] = 0

        # # print "goal position: ", self.goal_position
        # # print "plant position: ", self.plant.get_pos()
        # # print "abs difference: ", np.abs(self.pos_diff(self.goal_position,self.plant.get_pos()))
        # # print "target margin: ", self.target_margin

        # if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin):
        #     self.reached_goal_position = True
        #     if self.give_feedback:
        #         self.task_data['audio_feedback_start'] = 1
        #         self._play_sound(os.path.join(self.sounds_general_dir, 'beep.wav'))
        #         self.task_data['audio_feedback_start'] = 0
        # print 'distance to target', self.pos_diff(self.goal_position[self.pos_states],self.plant.get_pos())


        #self.task_data['audio_feedback_start'] = 0

        if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
            #self.task_data['audio_feedback_start'] = 1
            if self.give_feedback:
                self._play_sound(os.path.join(self.sounds_general_dir, 'beep.wav'))

            if self.goal_idx < len(self.targets_matrix[self.trial_type].keys())-1:
                self.goal_idx +=1
                print 'heading to next subtarget'
                self.reached_subtarget = True
                self.goal_position = self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states]
                #print self.goal_position
            else:
                print 'all subtargets reached'
                self.reached_goal_position = True

        #Show output decoder 
        # if self.state_decoder == 1:
        #     self.image_fname = os.path.join(self.image_dir, 'mov.bmp')
        #     self.show_image(self.image_fname) 
        # else:
        #     self.image_fname = os.path.join(self.image_dir, 'rest.bmp')
        #     self.show_image(self.image_fname) 

    def _while_trial_return(self):

        # if self.give_feedback == 1:
        #     #fb_time = self.feedback_time[self.trial_type][self.subtrial_idx]
        #     self.task_data['audio_feedback_start'] = 0
        # if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin):
        #     self.reached_goal_position = True
        #     #self.goal_position = self.rest_position
        #     if self.give_feedback:
        #         self.task_data['audio_feedback_start'] = 1
        #         self._play_sound(os.path.join(self.sounds_general_dir, 'beep.wav'))




        #self.task_data['audio_feedback_start'] = 0

        if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
            self.reached_goal_position = True
            #self.goal_position = self.rest_position
            #self.task_data['audio_feedback_start'] = 1
            if self.give_feedback:
                self._play_sound(os.path.join(self.sounds_general_dir, 'beep.wav'))
        #Show output decoder  
        # if self.state_decoder == 1:
        #     self.image_fname = os.path.join(self.image_dir, 'mov.bmp')
        #     self.show_image(self.image_fname) 
        # else:
        #     self.image_fname = os.path.join(self.image_dir, 'rest.bmp')
        #     self.show_image(self.image_fname) 
    # def _while_rest(self):
        
    #     #self.mov_data = self.mov_data_buffer.get_all()
    #     #self.rest_data = self.rest_data_buffer.get_all()     
    #     #rest_features, mov_features = self.eeg_extractor.extract_features_2retrain(self.rest_data, self.mov_data)
       
      
    def move_plant(self):
        '''Docstring.'''

        command_vel  = pd.Series(0.0, self.vel_states)
        command_vel_raw  = pd.Series(0.0, self.vel_states)

        # run EEG feature extractor and decoder
        #self.eeg_extractor.source = self.brainamp_source
        eeg_features = self.eeg_extractor() # eeg_features is of type 'dict'
        # print 'eeg_features'
        # print eeg_features
        feat_mov = self.mov_feature_buffer.get_all()
        feat_rest = self.rest_feature_buffer.get_all()
        mean_feat = np.mean(np.hstack([feat_mov, feat_rest]), axis = 1)
        std_feat = np.std(np.hstack([feat_mov, feat_rest]), axis = 1)

        if self.state in ['trial','trial_return']:
            self.mov_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
            #self.mov_feature_buffer.add(eeg_features)
        elif self.state == 'rest':
            self.rest_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
            #self.rest_feature_buffer.add(eeg_features)

        self.task_data['eeg_features'] = eeg_features
        self.task_data['eeg_mean_features'] = mean_feat
        self.task_data['eeg_std_features'] = std_feat
        
        # normalize features
        # eeg_features = (eeg_features - mean_feat.reshape(-1,1))/ std_feat.reshape(-1,1)
        eeg_features = (eeg_features - mean_feat)/ std_feat
        # mean_feat.ravel()
        self.task_data['eeg_features_Z'] = eeg_features
        #print 'eeg_features.shpae'
        eeg_features = np.reshape(eeg_features,(1,eeg_features.shape[0]))
        
        self.decoder_output = self.eeg_decoder(eeg_features) 
        self.probability = self.eeg_decoder.decoder.predict_proba(eeg_features) 
        
        # print self.decoder_output, ' with probability:', self.probability
        # Command zero velocity if the task is in a non-moving state
        if self.state in ['wait','rest', 'instruct_rest', 'instruct_trial_type', 'preparation', 'instruct_go']: 
            command_vel[:] = 0
            self.state_decoder = 0
            
        elif self.state in ['trial', 'trial_return', 'instruct_trial_return']:
            if self.decoder_output == 1 and self.prev_output == 1:
                # we need 5 consecutive outputs of the same type
                self.consec_mov_outputs +=1
                if self.consec_mov_outputs == 5 and self.state_decoder == 0:
                    self.consec_rest_outputs = 0
            elif self.decoder_output == 1 and self.prev_output == 0:
                if self.state_decoder == 1: #if it's moving
                    self.consec_rest_outputs = 0
                else:
                    self.consec_mov_outputs = 1
            elif self.decoder_output == 0 and self.prev_output == 0:
                self.consec_rest_outputs +=1
                if self.consec_rest_outputs == 5 and self.state_decoder == 1:
                    self.consec_mov_outputs = 0
            elif self.decoder_output == 0 and self.prev_output == 1:
                if self.state_decoder == 1: #if it's moving
                    self.consec_rest_outputs = 1
                else:
                    self.consec_mov_outputs = 0
    
            if self.consec_mov_outputs >= 5:

                self.state_decoder = 1
                current_pos = self.plant_pos[:].ravel()
                current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
                target_state = np.hstack([self.goal_position, np.zeros_like(current_pos), 1]).reshape(-1, 1)
                assist_output = self.assister(current_state, target_state, 1)
                           
                Bu = np.array(assist_output["x_assist"]).ravel()
                #Bu = np.array(assist_output['Bu']).ravel()
                command_vel[:] = Bu[len(current_pos):len(current_pos)*2]
                #set all the velocities to a constant value towards the end point
            elif self.consec_rest_outputs >=5:

                self.state_decoder = 0
                command_vel[:] = 0 #set all the velocities to zero
        
        command_vel_raw[:] = command_vel[:]
        command_vel[state] = self.command_lpfs[state](command_vel[state])
        

        self.prev_output = self.decoder_output
        #self.task_data['eeg_features']   = eeg_features[self.eeg_extractor.feature_type]
        
        self.task_data['decoder_output'] = self.decoder_output
        self.task_data['decoder_output_probability'] = self.probability
        self.task_data['state_decoder'] = self.state_decoder
        self.plant.send_vel(command_vel.values) #send velocity command to EXO
        
        self.task_data['command_vel']  = command_vel.values
        self.task_data['command_vel_raw']  = command_vel_raw.values
        
    def _cycle(self):
        '''Runs self.fps times per second.'''

        self.plant_pos_raw[:] = self.plant.get_pos_raw()
        self.plant_pos[:] = self.plant.get_pos() 
        
        self.plant_vel_raw[:] = self.plant.get_vel_raw()
        self.plant_vel[:] = self.plant.get_vel()

        #if self.state in ['trial','go_to_start']:
        # velocity control
        self.move_plant()

        
        self.update_plant_display()
        self.update_decoder_ouput()
        
        

        # print self.subtrial_idx
        if not self.state in ['trial','trial_return']:
            #self.task_data['audio_feedback_start'] = 0
            self.task_data['goal_pos'] = np.ones(len(self.pos_states))*np.nan
            self.task_data['reached_goal_position'] = False
            self.task_data['reached_subtarget'] = False
            self.task_data['goal_idx'] = np.nan
        else:
            self.task_data['goal_pos'] = self.goal_position
            self.task_data['reached_goal_position'] = self.reached_goal_position            
            self.task_data['goal_idx'] = self.goal_idx
            self.task_data['reached_subtarget'] = self.reached_subtarget

        self.task_data['eeg_decoder_coef']  = self.eeg_decoder.decoder.coef_
        self.task_data['eeg_decoder_intercept']  = self.eeg_decoder.decoder.intercept_
        self.task_data['eeg_decoder_means']  = self.eeg_decoder.decoder.means_

        self.task_data['plant_type']  = self.plant_type
        self.task_data['plant_pos']  = self.plant_pos_raw.values
        self.task_data['plant_pos_filt']  = self.plant_pos.values 
        self.task_data['plant_vel']  = self.plant_vel_raw.values
        self.task_data['plant_vel_filt']  = self.plant_vel.values

        self.task_data['trial_type'] = self.trial_type
        self.task_data['ts']         = time.time()
        self.task_data['target_margin'] = self.target_margin
        self.task_data['speed'] = self.speed
        #self.task_data['decoder']    = self.eeg_decoder.decoder

      
        super(EEGMovementDecoding, self)._cycle()


    def init_show_decoder_output(self):

        self.decoder_background_line  = Line(np.array([80, 0]), 100, 3, 1, COLORS['blue'])
        self.add_model(self.decoder_background_line)

        self.decoder_move_perc_line  = Line(np.array([80, 0]), 2, 3, 1, COLORS['red'])
        self.add_model(self.decoder_move_perc_line)

        self.decoder_middle_line = Line(np.array([80, 49]), 0.2, 3, 1, COLORS['white'])
        self.add_model(self.decoder_middle_line)


    def update_decoder_ouput(self):

        #backgroun line in white 
        self.decoder_background_line.color = COLORS['blue']
        self.decoder_background_line.start_pos   = np.array([80, 0])
        self.decoder_background_line.angle = 90*deg_to_rad

        #movement output in green
        self.decoder_move_perc_line.length = self.probability[0,1]*np.int(100)
        self.decoder_move_perc_line.start_pos   = np.array([80, 0])
        self.decoder_move_perc_line.angle = 90*deg_to_rad
        self.decoder_move_perc_line.color = COLORS['red']

        self.decoder_middle_line.color = COLORS['white']
        self.decoder_middle_line.start_pos   = np.array([80, 49])
        self.decoder_middle_line.angle = 90*deg_to_rad

  
    #### STATE AND TEST FUNCTIONS ####   
    def _start_wait(self):
        # determine the random length of time to stay in the rest state
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time
        super(EEGMovementDecoding, self)._start_wait()

    def _test_end_rest(self, ts):
        return ts > self.rest_time  

    def _test_end_preparation(self, ts):
        return ts > self.preparation_time  

    def _parse_next_trial(self):
        self.trial_type = self.next_trial

    def _test_end_instruct(self, *args, **kwargs):
        return not pygame.mixer.music.get_busy()

    def _start_instruct_rest(self):
        self._play_sound(os.path.join(self.sounds_dir, 'rest.wav'))
        #initial_mov_buffer_data = self.mov_data_buffer.get_all()
        print 'rest'

    def _start_instruct_trial_type(self):
        sound_fname = os.path.join(self.sounds_dir, self.trial_type + '.wav')
        self._play_sound(sound_fname)
        
        mov_features = self.mov_feature_buffer.get_all().T
        rest_features = self.rest_feature_buffer.get_all().T
        
        # normalization of features
        self.features = np.vstack([mov_features, rest_features])
        mean_features = np.mean(self.features, axis = 0)
        std_features = np.std(self.features, axis = 0)
        self.features = (self.features - mean_features) / std_features

        self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])
        self.retrained_decoder.fit(self.features, self.labels.ravel())
        print "before updating decoder"
        self.eeg_decoder.decoder = copy.copy(self.retrained_decoder)
        #print 'decoder retrained'
    def _end_instruct_trial_type(self):
        self.reached_goal_position = False
        self.reached_subtarget = False

        self.reached_timeout = False

    def _start_instruct_trial_return(self):
        sound_fname = os.path.join(self.sounds_dir, 'back.wav')
        self._play_sound(sound_fname)

    def _end_instruct_trial_return(self):
        self.reached_goal_position = False
        self.reached_timeout = False
        self.reached_subtarget = False
        # self.consec_mov_outputs = 0
        # self.consec_rest_outputs = 0
        #self.state_decoder = 0


    def _start_instruct_go(self):
        self._play_sound(os.path.join(self.sounds_dir, 'go.wav'))
        self.consec_mov_outputs = 0
        self.consec_rest_outputs = 0
        self.state_decoder = 0

    def _start_trial(self):
        print self.trial_type
        #self.plant.set_pos_control() #to set it to position control during the trial state
        #self._set_task_type()
        #self._set_goal_position()
        self.goal_position = self.targets_matrix[self.trial_type][0][self.pos_states]
        self.goal_idx = 0
        

    def _start_trial_return(self):
        print 'return trial'
        #self.plant.set_pos_control() #to set it to position control during the trial state

        #self._set_task_type()
        self.goal_position = self.targets_matrix['rest'][0][self.pos_states]


    def _test_end_trial(self,ts):
        return (self.reached_goal_position or self.reached_timeout)

    def _test_end_trial_return(self,ts):
        return (self.reached_goal_position or self.reached_timeout)

    def _test_timeout(self, ts):
        if ts > self.timeout_time:
            self.reached_timeout = True
            print 'timeout'
        return self.reached_timeout

    # def _test_at_starting_config(self, *args, **kwargs):
    #     traj = self.ref_trajectories[self.trial_type]['traj']
    #     diff_to_start = np.abs(self.plant.get_pos() - traj[self.pos_states].ix[0].ravel())
    #     #print diff_to_start

    #     return np.all(diff_to_start < self.rest_rect[:len(self.pos_states)])

    

    def cleanup(self, database, saveid, **kwargs):
        #Old way of buffering rest and mov data
        # self.mov_data = self.mov_data_buffer.get_all()
        # self.rest_data = self.rest_data_buffer.get_all()
        #rest_features, mov_features = self.eeg_extractor.extract_features_2retrain(self.rest_data, self.mov_data)
        #self.features = np.vstack([mov_features, rest_features])
        #self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])

        # New method of buffering rest and mov data to retrain decoder
        mov_features = self.mov_feature_buffer.get_all().T
        rest_features = self.rest_feature_buffer.get_all().T

        # normalization of features
        self.features = np.vstack([mov_features, rest_features])
        mean_features = np.mean(self.features, axis = 0)
        std_features = np.std(self.features, axis = 0)
        self.features = (self.features - mean_features) / std_features

        self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])
        self.retrained_decoder.fit(self.features, self.labels.ravel())


        self.eeg_decoder.rest_feature_buffer = self.rest_feature_buffer
        self.eeg_decoder.mov_feature_buffer = self.mov_feature_buffer
        self.eeg_decoder.decoder = copy.copy(self.retrained_decoder)

        #Values just to make it compatible with the task interface (they are nonsense)
        self.eeg_decoder.units  = self.eeg_decoder.channels_2train
        # self.decoder.binlen = # the decoder is updated after the end of each return trial
        # self.decoder.tslice = 

        #save eeg_decder object into a new pkl file. 
        storage_dir = '/storage/decoders'
        eeg_decoder_name = self.eeg_decoder.decoder_name 
        # n = decoder_name[-1]
        # n = int(n)        
        index = eeg_decoder_name.rfind('_') + 1
        #new_decoder_name = decoder_name[0:index] + str(n + 1)
        new_decoder_name = eeg_decoder_name[0:index] + str(saveid) 
        self.eeg_decoder.decoder_name = new_decoder_name
        new_pkl_name = new_decoder_name + '.pkl' 
        pickle.dump(self.eeg_decoder, open(os.path.join(storage_dir, new_pkl_name), 'wb'))

        super(EEGMovementDecoding,self).cleanup(database, saveid, **kwargs)
        # Create a new database record for the decoder object if it doesn't already exist
        dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
        dfs = models.Decoder.objects.filter(name=new_decoder_name)
        if len(dfs) == 0:
            df = models.Decoder()
            df.path = new_pkl_name
            df.name = new_decoder_name
            df.entry = models.TaskEntry.objects.using(dbname).get(id=saveid) 
            df.save()
        elif len(dfs) == 1:
            pass # no new data base record needed
        elif len(dfs) > 1:
            print "More than one decoder with the same name! fix manually!"
# class EEGCyclicMovementDecodingNew(CyclicEndPointMovement):
    # fps = 20
    # status = {
    #     'wait': {
    #         'start_trial': 'instruct_rest',
    #         'stop': None},
    #     'instruct_rest': {
    #         'end_instruct': 'rest',
    #         'stop':      None},            
    #     'rest': {
    #         'end_rest': 'instruct_trial_type',
    #         'stop':      None},
    #     'instruct_trial_type': {
    #         'end_instruct': 'preparation',
    #         'stop':      None},
    #     'preparation': {
    #         'end_preparation': 'instruct_go',
    #         'stop':      None},    
    #     'instruct_go': {
    #         'end_instruct': 'trial',
    #         'stop':      None},
    #     'trial': {
    #         'end_trial': 'wait',#'instruct_trial_go_to_start'
    #         'timeout': 'instruct_trial_type',
    #         'stop':      None},    
   
    #     }

    #  # settable parameters on web interface    
    # eeg_decoder          = traits.InstanceFromDB(LinearEEGDecoder, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='eeg_decoder'))
    # rest_interval        = traits.Tuple((3., 4.), desc='Min and max time to remain in the rest state.')
    # preparation_time     = traits.Float(2, desc='time to remain in the preparation state.')
    # timeout_time         = traits.Float(7, desc='Maximum time given to the patient to accomplish the task before considering it like incomplete and re-starting it from the current position') 
    # give_feedback        = traits.Int((0,1), desc=' 0 if we do not give feedback, 1 if we give feedback about whether the patient reached the goal position')
    # targets_matrix       = traits.DataFile(object, desc='goal positions for each of the trial types', bmi3d_query_kwargs=dict(system__name='misc'))
    # window_size          = traits.Tuple((1000, 560), desc='Size of window to display the plant position/angle')
    # channel_list_name    = traits.OptionsList(*channel_list_options, bmi3d_input_options=channel_list_options)
    # speed                = traits.OptionsList(*speed_options, bmi3d_input_options= speed_options)
    # music_feedback       = traits.Int((1), desc=' 0 if we do not want to include music, 1 if we want different classical music pieces with increasing intensity to be played')
    # #artifact_rejection   = traits.Int(1, desc=' 0 if artifacts are not rejected online, 1 if the artifact rejection is applied in real-time too')
    # #session_length = traits.Float(20, desc='overall time that the block will last') #It shows up by default in the interface
    # #neighbour_channels = ???
    # debug = False
    # DoF_control = traits.OptionsList(*DoF_control_options, bmi3d_input_options=DoF_control_options, desc='DoFs to be taken into account for condition fulfilment')
    # DoF_target = traits.OptionsList(*DoF_target_options, bmi3d_input_options=DoF_target_options, desc='DoFs to be moved/controlled, the rest are stopped.')

    # def _play_sound(self, fpath,fname):

    #     for filename in fname:
    #         # print 'filename ', filename
    #         if '_' in filename:
    #             filename = filename[:filename.find('_')]
    #         sound_fname = os.path.join(fpath, filename + '.wav')
    #         pygame.mixer.music.load(sound_fname)
    #         pygame.mixer.music.play()

    # def __init__(self, *args, **kwargs):
    #     super(EEGCyclicMovementDecodingNew, self).__init__(*args, **kwargs)

    #     self.add_dtype('command_vel_final',  'f8',    (len(self.vel_states),))
    #     #self.add_dtype('plant_type',   np.str_, 40)
    #     #self.add_dtype('difference_position','f8', (len(self.pos_states),))
    #     self.add_dtype('reached_goal_position',bool, (1,))
    #     self.add_dtype('reached_subtarget',bool, (1,))
    #     self.add_dtype('reached_timeout',bool, (1,))
    #     self.add_dtype('simult_reach_and_timeout',bool, (1,))
    #     #self.add_dtype('audio_feedback_start',      int,    (1,))
       

    #     self.parallel_sound = pygame.mixer.Sound('')
      
    #     # if EEG decoder file was specified, load EEG decoder and create feature extractor 
        
    #     #if len(self.decoder_file) > 3:
    #     #    self.eeg_decoder = pickle.load(open(self.decoder_file, 'rb'))

    #     # create EEG extractor object (its 'source' will be set later in the init method)
      
    #     eeg_extractor_cls    = self.eeg_decoder.extractor_cls
    #     self.eeg_decoder_name = self.eeg_decoder.decoder_name
    #     self.eeg_extractor_kwargs = self.eeg_decoder.extractor_kwargs
    #     self.artifact_rejection = self.eeg_extractor_kwargs['artifact_rejection']
        
    #     # Check if chosen decoder is trained with artifact rejection or not. If artifact_rejection = 1 and decoder not designed for that, print an error!

    #     self.TH_lowF = self.eeg_decoder.TH_lowF 
    #     self.TH_highF = self.eeg_decoder.TH_highF 
    #     self.eog_coeffs = self.eeg_decoder.eog_coeffs 


    #     self.rest_feature_buffer = self.eeg_decoder.rest_feature_buffer
    #     #self.trial_hand_side = self.eeg_extractor_kwargs['trial_hand_side']
    #     self.mov_feature_buffer = self.eeg_decoder.mov_feature_buffer
    #     try:
    #         self.channels = self.eeg_extractor_kwargs['eeg_channels']
    #     except:
    #         self.channels = self.eeg_extractor_kwargs['channels']
    #     #self.channels = self.eeg_extractor_kwargs['eeg_channels']
    #     self.brainamp_channels = getattr(brainamp_channel_lists, self.channel_list_name)  

    #     #self.brainamp_channels = self.eeg_extractor_kwargs['brainamp_channels']
    #     self.neighbour_channels = self.eeg_decoder.neighbour_channels
        
    #     self.eeg_playback = False
    #     self.fs = self.eeg_extractor_kwargs['fs']

    #     self.retrained_decoder = copy.copy(self.eeg_decoder.decoder)
        
    #     self.eeg_extractor_kwargs['eog_coeffs'] = self.eog_coeffs 
    #     self.eeg_extractor_kwargs['TH_lowF'] = self.TH_lowF 
    #     self.eeg_extractor_kwargs['TH_highF'] = self.TH_highF 

    #     self.eeg_extractor = eeg_extractor_cls(source=None, **self.eeg_extractor_kwargs)
    #     self.n_features = self.eeg_extractor.n_features
    #     #dtype = np.dtype(['name',       np.str, [len(self.channels),20])
        
    #     #self.add_dtype('eeg_features',    'f8', (self.eeg_extractor.n_features,))
    #     self.add_dtype('eeg_features',    'f8', (self.n_features,))
    #     #self.add_dtype('channels',   np.str_, [len(self.channels),20])
    #     self.add_dtype('eeg_features_mov_buffer',    'f8', (self.n_features,))
    #     self.add_dtype('eeg_features_rest_buffer',    'f8', (self.n_features,))
    #     self.add_dtype('eeg_features_Z',    'f8', (self.n_features,))
    #     self.add_dtype('eeg_mean_features',    'f8', (self.n_features,))
    #     self.add_dtype('eeg_std_features',    'f8', (self.n_features,))
    #     self.add_dtype('eeg_decoder_coef',    'f8', (self.n_features,))
    #     self.add_dtype('eeg_decoder_intercept', 'f8', (self.n_features,))
    #     self.add_dtype('eeg_decoder_means', 'f8', (2,self.n_features))
    #     self.add_dtype('decoder_output',    'f8', (1,))
    #     self.add_dtype('decoder_output_probability',    'f8', (1,2))#2 classes
    #     self.add_dtype('state_decoder',  int, (1,))
    #     self.add_dtype('consec_mov_outputs',  int, (1,))
    #     self.add_dtype('consec_rest_outputs',  int, (1,))
    #     self.add_dtype('rejected_window',  int, (1,))
    #     #self.add_dtype('decoder', InstanceFromDB(LinearEEGDecoder))
    #     self.sounds_dir_classical = os.path.expandvars('$HOME/code/ismore/sounds/classical')

               
    #     # for low-pass filtering decoded EEG velocities
    #     # self.eeg_vel_buffer = RingBuffer(
    #     #     item_len=len(self.vel_states),
    #     #     capacity=10,
    #     # )

    #     self.plant.enable() 
    #     [self.DoF_target_idx_init, self.DoF_target_idx_end, self.DoF_not_control_idx_init, self.DoF_not_control_idx_end] = check_plant_and_DoFs(self.plant_type, self.DoF_control, self.DoF_target)
    #     self.subgoal_names = self.targets_matrix['subgoal_names']


    #     #initialize values for the state of the decoder
    #     self.consec_mov_outputs = 0
    #     self.consec_rest_outputs = 0
    #     self.prev_output = 0
    #     self.state_decoder = 0

    #     # if self.plant_type == 'ArmAssist':
    #     #     self.target_margin = np.array([2, 2, np.deg2rad(10)])
    #     # elif self.plant_type == 'ReHand':
    #     #     self.target_margin = np.array([np.deg2rad(2),  np.deg2rad(2), np.deg2rad(2), np.deg2rad(2)])
    #     # elif self.plant_type == 'IsMore':
    #     #     self.target_margin = np.array([2, 2, np.deg2rad(10), np.deg2rad(2),  np.deg2rad(2), np.deg2rad(2), np.deg2rad(2)])
          
    #     #self.target_margin = pd.Series(np.array([3, 3, np.deg2rad(15), np.deg2rad(5),  np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)]), ismore_pos_states)
    #     # target margin used for DK calibration sessions
   

    #     self.goal_idx = 0
    #     self.trial_number = 0


    #     self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
    #     self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)

    #     self.image_dir_general = os.path.expandvars('$HOME/code/ismore/images')
    #     self.image_dir = os.path.join(self.image_dir_general, self.language)
        

    #     self.reached_goal_position = False #If the goal_position is reached then give feedback to patient and start the movement back towards the rest_position
    #     self.reached_subtarget = False #If the task has more than one target position, this shows when the targets before the last target are reached
    #     self.reached_timeout = False
    #     self.simult_reach_and_timeout = False


    #     self.init_show_decoder_output()     

    # def init(self):
    #     from riglib import source
    #     from ismore.brainamp import rda
    #     #self.brainamp_source = source.MultiChanDataSource(rda.EMGData, name='brainamp', channels=self.brainamp_channels, brainamp_channels=self.brainamp_channels, send_data_to_sink_manager=True)
    #     self.eeg_extractor.source = self.brainamp_source
    #     super(EEGCyclicMovementDecodingNew, self).init()

    # def _while_trial(self):
    #     #print 'reps', self.reps
    #     print 'self.pos_diff(self.goal_position,self.plant.get_pos()))', self.pos_diff(self.goal_position,self.plant.get_pos()) 
    #     print 'self.target_margin[self.pos_states]', self.target_margin[self.pos_states]
    #     if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
    #         #self.task_data['audio_feedback_start'] = 1
    #         print 'goal_dix', self.goal_idx
    #         print 'len', len(self.targets_matrix[self.trial_type].keys())-1
    #         if self.give_feedback:
    #             # self._play_sound(os.path.join(self.sounds_general_dir, 'beep.wav'))
    #             #self._play_sound(self.sounds_general_dir, ['beep']) #nerea
    #             pass
    #         if self.goal_idx < len(self.targets_matrix[self.trial_type].keys())-1:
                
    #             pygame.mixer.music.stop() #nerea
                
    #             self.parallel_sound.stop()
    #             self.goal_idx +=1
    #             print 'heading to next subtarget'
    #             self._play_sound(self.sounds_general_dir, ['beep']) #nerea
    #             self.goal_position = self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states]
               
    #             #self._play_sound(self.sounds_dir, self.subgoal_names[self.trial_type][self.goal_idx]) #nerea
    #             # pygame.mixer.music.queue(os.path.join(self.sounds_dir_classical, self.subgoal_names[self.trial_type][self.goal_idx][0]+'.wav'))
    #             self.parallel_sound = pygame.mixer.Sound(os.path.join(self.sounds_dir_classical, self.subgoal_names[self.trial_type][self.goal_idx][0]+'.wav'))
    #             self.parallel_sound.play()
    #             # self._play_sound(self.sounds_dir_classical, [self.subgoal_names[self.trial_type][self.goal_idx][0]])  #nerea
    #         elif self.reps < self.repetitions_cycle:
    #             print 'cycle completed'
    #             self.reps += 1
    #             self.goal_position = self.targets_matrix[self.trial_type][0][self.pos_states + self.vel_states]
    #             self.goal_idx = 0
    #             self._play_sound(self.sounds_general_dir, ['beep']) #nerea
    #             #self._play_sound(self.sounds_dir, self.subgoal_names[self.trial_type][self.goal_idx]) #nerea
    #             #self.parallel_sound.play()
    #             #self.reached_subtarget = True
    #             # #print self.goal_position
    #         else:
    #             print 'all subtargets reached'
    #             self.reached_goal_position = True
    #             self.goal_idx = 0

    # def move_plant(self):
    #     '''Docstring.'''

    #     command_vel  = pd.Series(0.0, self.vel_states)
    #     command_vel_raw  = pd.Series(0.0, self.vel_states)


    #     #calculate the output of the LQR controller at all states
    #     current_pos = self.plant_pos[:].ravel()
    #     current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
        
        
    #     # run EEG feature extractor and decoder
    #     #self.eeg_extractor.source = self.brainamp_source
    #     if self.artifact_rejection == 1:
    #         eeg_features, rejected_window = self.eeg_extractor()
    #         self.task_data['rejected_window'] = rejected_window
    #     else:
    #         eeg_features = self.eeg_extractor() # eeg_features is of type 'dict'
          
    #     if self.state in ['wait','rest', 'instruct_rest', 'preparation', 'instruct_go', 'instruct_trial_type']:
    #         #in return state and in the states where the exo does not move the target position is the rest position
    #         target_state = current_state
  
    #     elif self.state == 'trial':
    #         target_state = np.hstack([self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states + self.vel_states], 1]).reshape(-1,1)
           
    #     feat_mov = self.mov_feature_buffer.get_all()
    #     feat_rest = self.rest_feature_buffer.get_all()
    #     mean_feat = np.mean(np.hstack([feat_mov, feat_rest]), axis = 1)
    #     std_feat = np.std(np.hstack([feat_mov, feat_rest]), axis = 1)


    #     if self.trial_number > 0:
    #         if self.state in ['trial','trial_return']:
    #             if self.artifact_rejection == 1 & rejected_window == 0:
    #                 self.mov_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
    #             elif self.artifact_rejection == 0:
    #                 self.mov_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
    #             self.task_data['eeg_features_mov_buffer'] = eeg_features
    #             #self.mov_feature_buffer.add(eeg_features)
    #         elif self.state in ['rest','rest_return']:
    #             if self.artifact_rejection == 1 & rejected_window == 0:
    #                 self.rest_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
    #             elif self.artifact_rejection == 0:
    #                 self.rest_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
                    
    #             self.task_data['eeg_features_rest_buffer'] = eeg_features
         
        
    #     self.task_data['eeg_features'] = eeg_features
    #     self.task_data['eeg_mean_features'] = mean_feat
    #     self.task_data['eeg_std_features'] = std_feat
    #     #self.task_data['channels'] = self.channels
    #     # normalize features
    #     # eeg_features = (eeg_features - mean_feat.reshape(-1,1))/ std_feat.reshape(-1,1)
    #     eeg_features = (eeg_features - mean_feat)/ std_feat
    #     # mean_feat.ravel()
    #     self.task_data['eeg_features_Z'] = eeg_features

    #     #print 'eeg_features.shpae'

    #     try:
    #         eeg_features = np.reshape(eeg_features,(1,eeg_features.shape[0]))
    #     except:
    #         pass
    #     #eeg_features(eeg_features == np.inf) = 1
    #     self.decoder_output = self.eeg_decoder(eeg_features)
    #     self.probability = self.eeg_decoder.decoder.predict_proba(eeg_features)
    
    #     # print "dec probability : ", self.probability

    #     #print self.decoder_output, ' with probability:', probability

    #     # Command zero velocity if the task is in a non-moving state
    #     if self.state not in ['trial']:#['wait','rest', 'rest_return','instruct_rest', 'instruct_trial_type', 'preparation', 'instruct_go','instruct_rest_return']: 
    #         command_vel[:] = 0
    #         self.state_decoder = 0
            
    #     else:#if self.state in ['trial', 'trial_return', 'instruct_trial_return']:
    #         if self.decoder_output == 1 and self.prev_output == 1:
    #             # we need 5 consecutive outputs of the same type
    #             self.consec_mov_outputs +=1
    #             if self.consec_mov_outputs == 5 and self.state_decoder == 0:
    #                 self.consec_rest_outputs = 0
    #         elif self.decoder_output == 1 and self.prev_output == 0:
    #             if self.state_decoder == 1: #if it's moving
    #                 self.consec_rest_outputs = 0
    #             else:
    #                 self.consec_mov_outputs = 1
    #         elif self.decoder_output == 0 and self.prev_output == 0:
    #             self.consec_rest_outputs +=1
    #             if self.consec_rest_outputs == 5 and self.state_decoder == 1:
    #                 self.consec_mov_outputs = 0
    #         elif self.decoder_output == 0 and self.prev_output == 1:
    #             if self.state_decoder == 1: #if it's moving
    #                 self.consec_rest_outputs = 1
    #             else:
    #                 self.consec_mov_outputs = 0
    
    #         if self.consec_mov_outputs >= 5:

    #             self.state_decoder = 1
    #             current_pos = self.plant_pos[:].ravel()
    #             current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
    #             target_state = np.hstack([self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states + self.vel_states], 1]).reshape(-1,1)
    #             #target_state = np.hstack([self.goal_position, np.zeros_like(current_pos), 1]).reshape(-1, 1)
    #             assist_output = self.assister(current_state, target_state, 1)
                           
    #             Bu = np.array(assist_output["x_assist"]).ravel()
    #             #Bu = np.array(assist_output['Bu']).ravel()
    #             command_vel[:] = Bu[len(current_pos):len(current_pos)*2]
    #             #print 'command_vel', command_vel
    #             #set all the velocities to a constant value towards the end point
    #         elif self.consec_rest_outputs >=5:

    #             self.state_decoder = 0
    #             command_vel[:] = 0 #set all the velocities to zero
        
    #     command_vel_raw[:] = command_vel[:]
    #     for state in self.vel_states:
    #         command_vel[state] = self.command_lpfs[state](command_vel[state])
        
    #     self.prev_output = self.decoder_output
    #     #self.task_data['eeg_features']   = eeg_features[self.eeg_extractor.feature_type]
        
    #     self.task_data['decoder_output'] = self.decoder_output
    #     self.task_data['decoder_output_probability'] = self.probability
    #     self.task_data['state_decoder'] = self.state_decoder
    #     self.task_data['consec_mov_outputs'] = self.consec_mov_outputs
    #     self.task_data['consec_rest_outputs'] = self.consec_rest_outputs
    #     self.plant.send_vel(command_vel.values) #send velocity command to EXO
        
    #     self.task_data['command_vel']  = command_vel.values
    #     self.task_data['command_vel_raw']  = command_vel_raw.values
    
    # def _cycle(self):
    #     '''Runs self.fps times per second.'''

    #     # get latest position/velocity information before calling move_plant()
    #     self.plant_pos_raw[:] = self.plant.get_pos_raw()
    #     self.plant_pos[:] = self.plant.get_pos() 
        
    #     self.plant_vel_raw[:] = self.plant.get_vel_raw()
    #     self.plant_vel[:] = self.plant.get_vel()

    #     #if self.state in ['trial','go_to_start']:
    #     # velocity control
    #     self.move_plant()

    #     # position control
    #     # self.move_plant_pos_control()

    #     self.update_plant_display()
    #     self.update_decoder_ouput()

    #     # print self.subtrial_idx
    #     if not self.state in ['trial','trial_return']:
    #         #self.task_data['audio_feedback_start'] = 0
    #         self.task_data['goal_pos'] = np.ones(len(self.pos_states))*np.nan            
    #         self.task_data['goal_idx'] = np.nan
            
    #     else:
    #         self.task_data['goal_pos'] = self.goal_position[self.pos_states]
    #         self.task_data['goal_idx'] = self.goal_idx
            

    #     self.task_data['plant_type'] = self.plant_type
    #     self.task_data['eeg_decoder_coef']  = self.eeg_decoder.decoder.coef_
    #     self.task_data['eeg_decoder_intercept']  = self.eeg_decoder.decoder.intercept_
    #     self.task_data['eeg_decoder_means']  = self.eeg_decoder.decoder.means_
    #     self.task_data['reached_goal_position'] = self.reached_goal_position  
    #     self.task_data['reached_subtarget'] = self.reached_subtarget          
            
    #     self.task_data['reached_timeout'] = self.reached_timeout
    #     self.task_data['simult_reach_and_timeout'] = self.simult_reach_and_timeout
        
    #     self.task_data['plant_pos']  = self.plant_pos_raw.values
    #     self.task_data['plant_pos_filt']  = self.plant_pos.values 
    #     self.task_data['plant_vel']  = self.plant_vel_raw.values
    #     self.task_data['plant_vel_filt']  = self.plant_vel.values

    #     self.task_data['trial_type'] = self.trial_type
    #     self.task_data['speed'] = self.speed
    #     self.task_data['ts']         = time.time()
    #     self.task_data['target_margin'] = self.target_margin
    #     #self.task_data['decoder']    = self.eeg_decoder.decoder

        
    #     super(EEGCyclicMovementDecodingNew, self)._cycle()  

    # def init_show_decoder_output(self):

    #     self.decoder_background_line  = Line(np.array([80, 0]), 100, 3, 1, COLORS['blue'])
    #     self.add_model(self.decoder_background_line)

    #     self.decoder_move_perc_line  = Line(np.array([80, 0]), 2, 3, 1, COLORS['red'])
    #     self.add_model(self.decoder_move_perc_line)

    #     self.decoder_middle_line = Line(np.array([80, 49]), 0.2, 3, 1, COLORS['white'])
    #     self.add_model(self.decoder_middle_line)



    # def update_decoder_ouput(self):

    #     #backgroun line in white 
    #     self.decoder_background_line.color = COLORS['blue']
    #     self.decoder_background_line.start_pos   = np.array([80, 0])
    #     self.decoder_background_line.angle = 90*deg_to_rad

    #     #movement output in green
    #     self.decoder_move_perc_line.length = self.probability[0,1]*np.int(100)
    #     self.decoder_move_perc_line.start_pos   = np.array([80, 0])
    #     self.decoder_move_perc_line.angle = 90*deg_to_rad
    #     self.decoder_move_perc_line.color = COLORS['red']

    #     self.decoder_middle_line.color = COLORS['white']
    #     self.decoder_middle_line.start_pos   = np.array([80, 49])
    #     self.decoder_middle_line.angle = 90*deg_to_rad
    # def _start_instruct_rest(self):
    #     self.parallel_sound.stop()
    #     # self._play_sound(os.path.join(self.sounds_dir, 'rest.wav')) #nerea
    #     self._play_sound(self.sounds_dir, ['rest'])
    #     self.goal_idx = 0
    #     self.reps = 1   
    #     #initial_mov_buffer_data = self.mov_data_buffer.get_all()
    #     print 'rest'

    # def _start_instruct_trial_type(self):
    #     #print 'instruct trial type'
    #     # sound_fname = os.path.join(self.sounds_dir, self.trial_type + '.wav') #nerea
    #     # self._play_sound(sound_fname)
    #     # self._play_sound(self.sounds_dir, self.subgoal_names[self.trial_type][0]) 
    #     self._play_sound(self.sounds_dir, self.subgoal_names[self.trial_type][self.goal_idx]) #nerea ??
        
    #     mov_features = self.mov_feature_buffer.get_all().T
    #     rest_features = self.rest_feature_buffer.get_all().T

    #     # normalization of features
    #     self.features = np.vstack([mov_features, rest_features])
    #     mean_features = np.mean(self.features, axis = 0)
    #     std_features = np.std(self.features, axis = 0)
    #     self.features = (self.features - mean_features) / std_features
        
    #     self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])
    #     print 'retraining decoder'
    #     self.retrained_decoder.fit(self.features, self.labels.ravel())
        
    #     # import time
    #     # t0 = time.time()
    #     self.eeg_decoder.decoder = copy.copy(self.retrained_decoder)      

    # def _start_instruct_go(self):
    #     # self._play_sound(os.path.join(self.sounds_dir, 'go.wav'))
    #     self._play_sound(self.sounds_dir, ['go'])
    #     self.consec_mov_outputs = 0
    #     self.consec_rest_outputs = 0
    #     self.reached_goal_position = False
    #     self.reached_subtarget = False
    #     self.reached_timeout = False
    #     self.simult_reach_and_timeout = False
    #     #self.state_decoder = 0

    # def _start_trial(self):
    #     print self.trial_type
    #     #self.plant.set_pos_control() #to set it to position control during the trial state
    #     #self._set_task_type()
    #     #self._set_goal_position()
    #     #self.goal_position = self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states]  
    #     self.goal_position = self.targets_matrix[self.trial_type][0][self.pos_states + self.vel_states]
        
    #     if self.music_feedback:
    #         self._play_sound(self.sounds_dir_classical, [self.subgoal_names[self.trial_type][self.goal_idx][0]])
    

    # def _test_end_trial(self,ts):
       
    #     return (self.reached_goal_position or self.reached_timeout)

    # def _test_timeout(self, ts):
    #     if ts > self.timeout_time:
    #         self.reached_timeout = True
    #         if self.reached_goal_position == True:
    #             #self.reached_timeout = False
    #             self.simult_reach_and_timeout = True
    #         #print 'reached goal position', self.reached_goal_position
    #         print 'timeout'
    #         # if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
    #         #     self.reached_timeout = False
    #     return self.reached_timeout

   
    # def _end_trial(self):
        
    #     self.trial_number +=1
    #     if self.music_feedback:
    #         pygame.mixer.music.stop()
    #         self.parallel_sound.stop()
    #     else:
    #         pass


    # def cleanup_hdf(self):
    #     super(EEGCyclicMovementDecodingNew, self).cleanup_hdf()    

    #     import tables
    #     h5file = tables.openFile(self.h5file.name, mode='a')
    #     h5file.root.task.attrs['eeg_decoder_name'] = self.eeg_decoder_name
    #     #h5file.root.task.attrs['brainamp_channels'] = self.channel_list_name
    #     #compfilt = tables.Filters(complevel=5, complib="zlib", shuffle=True)
    #     eeg_extractor_grp = h5file.createGroup(h5file.root, "eeg_extractor_kwargs", "Parameters for feature extraction")
    #     for key in self.eeg_extractor_kwargs:
    #         if isinstance(self.eeg_extractor_kwargs[key], dict):
    #             if key == 'feature_fn_kwargs':
    #                 for key2 in self.eeg_extractor_kwargs['feature_fn_kwargs']['AR']['freq_bands']:
    #                     if isinstance(self.eeg_extractor_kwargs['feature_fn_kwargs']['AR']['freq_bands'][key2], np.ndarray):
    #                         h5file.createArray(eeg_extractor_grp, 'freq_band_' + key2, self.eeg_extractor_kwargs['feature_fn_kwargs']['AR']['freq_bands'][key2])
    #                     else:
    #                         h5file.createArray(eeg_extractor_grp, 'freq_band_' + key2, np.array([self.eeg_extractor_kwargs['feature_fn_kwargs']['AR']['freq_bands'][key2]]))
    #             else:
    #                 for key2 in self.eeg_extractor_kwargs[key]:
    #                     if isinstance(self.eeg_extractor_kwargs[key][key2], np.ndarray):
    #                         h5file.createArray(eeg_extractor_grp, key + '_' + key2, self.eeg_extractor_kwargs[key][key2])
    #                     else:
    #                         h5file.createArray(eeg_extractor_grp, key + '_' + key2, np.array([self.eeg_extractor_kwargs[key][key2]]))

    #         else:
    #             if isinstance(self.eeg_extractor_kwargs[key], np.ndarray):
    #                 h5file.createArray(eeg_extractor_grp, key, self.eeg_extractor_kwargs[key])
    #             else:
    #                 h5file.createArray(eeg_extractor_grp, key, np.array([self.eeg_extractor_kwargs[key]]))
                

    #     h5file.close()

    # def cleanup(self, database, saveid, **kwargs):
    #     #Old way of buffering rest and mov data
    #     # self.mov_data = self.mov_data_buffer.get_all()
    #     # self.rest_data = self.rest_data_buffer.get_all()
    #     #rest_features, mov_features = self.eeg_extractor.extract_features_2retrain(self.rest_data, self.mov_data)
    #     #self.features = np.vstack([mov_features, rest_features])
    #     #self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])

    #     # New method of buffering rest and mov data to retrain decoder
    #     mov_features = self.mov_feature_buffer.get_all().T
    #     rest_features = self.rest_feature_buffer.get_all().T
    #     # normalization of features
    #     self.features = np.vstack([mov_features, rest_features])
    #     mean_features = np.mean(self.features, axis = 0)
    #     std_features = np.std(self.features, axis = 0)
    #     self.features = (self.features - mean_features) / std_features

        
    #     self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])
    #     self.retrained_decoder.fit(self.features, self.labels.ravel())


    #     self.eeg_decoder.rest_feature_buffer = self.rest_feature_buffer
    #     self.eeg_decoder.mov_feature_buffer = self.mov_feature_buffer
    #     self.eeg_decoder.decoder = copy.copy(self.retrained_decoder)

    #     #Values just to make it compatible with the task interface (they are nonsense)
    #     self.eeg_decoder.units  = self.eeg_decoder.channels_2train
    #     # self.eeg_decoder.binlen = # the decoder is updated after the end of each return trial
    #     # self.eeg_decoder.tslice = 

    #     #save eeg_decder object into a new pkl file. 
    #     storage_dir = '/storage/decoders'
    #     eeg_decoder_name = self.eeg_decoder.decoder_name 
    #     # n = decoder_name[-1]
    #     # n = int(n)        
    #     index = eeg_decoder_name.rfind('_') + 1
    #     #new_decoder_name = decoder_name[0:index] + str(n + 1)
    #     new_decoder_name = eeg_decoder_name[0:index] + str(saveid) 
    #     self.eeg_decoder.decoder_name = new_decoder_name
    #     new_pkl_name = new_decoder_name + '.pkl' 
    #     pickle.dump(self.eeg_decoder, open(os.path.join(storage_dir, new_pkl_name), 'wb'))

    #     super(EEGCyclicMovementDecodingNew,self).cleanup(database, saveid, **kwargs)
    #     # Create a new database record for the decoder object if it doesn't already exist
    #     dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
    #     dfs = models.Decoder.objects.filter(name=new_decoder_name)
    #     if len(dfs) == 0:
    #         df = models.Decoder()
    #         df.path = new_pkl_name
    #         df.name = new_decoder_name
    #         df.entry = models.TaskEntry.objects.using(dbname).get(id=saveid) 
    #         df.save()
    #     elif len(dfs) == 1:
    #         pass # no new data base record needed
    #     elif len(dfs) > 1:
    #         print "More than one decoder with the same name! fix manually!"


class EEGMovementDecodingNew(NonInvasiveBase):
    # Unlike the EEGMovementDecoding task, it keeps going towards the same target until it reaches the target position
    fps = 20
    
    status = {
        'wait': {
            'start_trial': 'instruct_rest',
            'stop': None},
        'instruct_rest': {
            'end_instruct': 'rest',
            'stop':      None},            
        'rest': {
            'late_end_trial': 'instruct_trial_return',
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
            'end_trial': 'instruct_rest_return',#'instruct_trial_go_to_start'
            'timeout': 'instruct_rest',#'instruct_trial_go_to_start'
            'stop':      None},  
        'instruct_rest_return': {
            'end_instruct': 'rest_return',
            'stop':      None},            
        'rest_return': {
            'late_end_trial_return': 'instruct_trial_type',
            'end_rest': 'instruct_trial_return',
            'stop':      None},
        'instruct_trial_return': {
            'end_instruct': 'preparation_return',
            'stop':      None},
        'preparation_return': {
            'end_preparation': 'instruct_go_return',
            'stop':      None},    
        'instruct_go_return': {
            'end_instruct': 'trial_return',
            'stop':      None},  
        'trial_return': {            
            'end_trial': 'wait',
            'timeout': 'instruct_rest_return',#'instruct_rest'
            'stop':      None},    
        }
    
    state = 'wait'  # initial state

     # settable parameters on web interface    
    eeg_decoder          = traits.InstanceFromDB(LinearEEGDecoder, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='eeg_decoder'))
    rest_interval        = traits.Tuple((3., 4.), desc='Min and max time to remain in the rest state.')
    preparation_time     = traits.Float(2, desc='time to remain in the preparation state.')
    timeout_time         = traits.Float(10, desc='Maximum time given to the patient to accomplish the task before considering it like incomplete and re-starting it from the current position') 
    give_feedback        = traits.Int((0,1), desc=' 0 if we do not give feedback, 1 if we give feedback about whether the patient reached the goal position')
    targets_matrix       = traits.DataFile(object, desc='goal positions for each of the trial types', bmi3d_query_kwargs=dict(system__name='misc'))
    window_size          = traits.Tuple((1000, 560), desc='Size of window to display the plant position/angle')
    channel_list_name    = traits.OptionsList(*channel_list_options, bmi3d_input_options=channel_list_options)
    speed                = traits.OptionsList(*speed_options, bmi3d_input_options= speed_options)
    music_feedback       = traits.Int((1), desc=' 0 if we do not want to include music, 1 if we want different classical music pieces with increasing intensity to be played')
    #artifact_rejection   = traits.Int(1, desc=' 0 if artifacts are not rejected online, 1 if the artifact rejection is applied in real-time too')
    #session_length = traits.Float(20, desc='overall time that the block will last') #It shows up by default in the interface
    #neighbour_channels = ???
    debug = False
    DoF_control = traits.OptionsList(*DoF_control_options, bmi3d_input_options=DoF_control_options, desc='DoFs to be taken into account for condition fulfilment')
    DoF_target = traits.OptionsList(*DoF_target_options, bmi3d_input_options=DoF_target_options, desc='DoFs to be moved/controlled, the rest are stopped.')


    # Set the "is_bmi_seed" flag so that the server knows that this is a task which can be used to create a BMI decoder
    #is_bmi_seed = False
    # def _play_sound(self, fname):
    #     pygame.mixer.music.load(fname)
    #     pygame.mixer.music.play()

    def _play_sound(self, fpath,fname):

        for filename in fname:
            # print 'filename ', filename
            if '_' in filename:
                filename = filename[:filename.find('_')]
            sound_fname = os.path.join(fpath, filename + '.wav')
            pygame.mixer.music.load(sound_fname)
            pygame.mixer.music.play()


            # print 'sound_fname ' , sound_fname
            # while pygame.mixer.music.get_busy():
            #     time.sleep(0)
            # else:
            #     pygame.mixer.music.load(sound_fname)
            #     pygame.time.Clock().tick(1)
            #     # print 'clock'
            #     pygame.mixer.music.play()

    def __init__(self, *args, **kwargs):
        super(EEGMovementDecodingNew, self).__init__(*args, **kwargs)

        self.add_dtype('command_vel',  'f8',    (len(self.vel_states),))
        self.add_dtype('command_vel_raw',  'f8',    (len(self.vel_states),))
        self.add_dtype('command_vel_final',  'f8',    (len(self.vel_states),))
        

        self.add_dtype('trial_type',   np.str_, 40)
        #self.add_dtype('plant_type',   np.str_, 40)
        
        # self.add_dtype('ts',           'f8',    (1,)) # it is already saved in IsMoreBase class (basic class)
        self.add_dtype('goal_pos',      'f8',    (len(self.pos_states),))
        #self.add_dtype('difference_position','f8', (len(self.pos_states),))
        self.add_dtype('reached_goal_position',bool, (1,))
        self.add_dtype('reached_subtarget',bool, (1,))
        self.add_dtype('reached_timeout',bool, (1,))
        self.add_dtype('simult_reach_and_timeout',bool, (1,))
        #self.add_dtype('audio_feedback_start',      int,    (1,))
        self.add_dtype('goal_idx', float, (1,))
        self.add_dtype('speed',   np.str_, 20)

        self.parallel_sound = pygame.mixer.Sound('')
      
        # if EEG decoder file was specified, load EEG decoder and create feature extractor 
        
        #if len(self.decoder_file) > 3:
        #    self.eeg_decoder = pickle.load(open(self.decoder_file, 'rb'))

        # create EEG extractor object (its 'source' will be set later in the init method)
      
        eeg_extractor_cls    = self.eeg_decoder.extractor_cls
        self.eeg_decoder_name = self.eeg_decoder.decoder_name
        self.eeg_extractor_kwargs = self.eeg_decoder.extractor_kwargs
        self.artifact_rejection = self.eeg_extractor_kwargs['artifact_rejection']
        
        # Check if chosen decoder is trained with artifact rejection or not. If artifact_rejection = 1 and decoder not designed for that, print an error!

        self.TH_lowF = self.eeg_decoder.TH_lowF 
        self.TH_highF = self.eeg_decoder.TH_highF 
        self.eog_coeffs = self.eeg_decoder.eog_coeffs 


        self.rest_feature_buffer = self.eeg_decoder.rest_feature_buffer
        #self.trial_hand_side = self.eeg_extractor_kwargs['trial_hand_side']
        self.mov_feature_buffer = self.eeg_decoder.mov_feature_buffer
        try:
            self.channels = self.eeg_extractor_kwargs['eeg_channels']
        except:
            self.channels = self.eeg_extractor_kwargs['channels']
        #self.channels = self.eeg_extractor_kwargs['eeg_channels']
        self.brainamp_channels = getattr(brainamp_channel_lists, self.channel_list_name)  

        #self.brainamp_channels = self.eeg_extractor_kwargs['brainamp_channels']
        self.neighbour_channels = self.eeg_decoder.neighbour_channels
        
        self.eeg_playback = False
        self.fs = self.eeg_extractor_kwargs['fs']

        self.retrained_decoder = copy.copy(self.eeg_decoder.decoder)
        
        self.eeg_extractor_kwargs['eog_coeffs'] = self.eog_coeffs 
        self.eeg_extractor_kwargs['TH_lowF'] = self.TH_lowF 
        self.eeg_extractor_kwargs['TH_highF'] = self.TH_highF 

        self.eeg_extractor = eeg_extractor_cls(source=None, **self.eeg_extractor_kwargs)
        self.n_features = self.eeg_extractor.n_features
        #dtype = np.dtype(['name',       np.str, [len(self.channels),20])
        
        #self.add_dtype('eeg_features',    'f8', (self.eeg_extractor.n_features,))
        self.add_dtype('eeg_features',    'f8', (self.n_features,))
        #self.add_dtype('channels',   np.str_, [len(self.channels),20])
        self.add_dtype('eeg_features_mov_buffer',    'f8', (self.n_features,))
        self.add_dtype('eeg_features_rest_buffer',    'f8', (self.n_features,))
        self.add_dtype('eeg_features_Z',    'f8', (self.n_features,))
        self.add_dtype('eeg_mean_features',    'f8', (self.n_features,))
        self.add_dtype('eeg_std_features',    'f8', (self.n_features,))
        self.add_dtype('eeg_decoder_coef',    'f8', (self.n_features,))
        self.add_dtype('eeg_decoder_intercept', 'f8', (self.n_features,))
        self.add_dtype('eeg_decoder_means', 'f8', (2,self.n_features))
        self.add_dtype('decoder_output',    'f8', (1,))
        self.add_dtype('decoder_output_probability',    'f8', (1,2))#2 classes
        self.add_dtype('state_decoder',  int, (1,))
        self.add_dtype('consec_mov_outputs',  int, (1,))
        self.add_dtype('consec_rest_outputs',  int, (1,))
        self.add_dtype('rejected_window',  int, (1,))
        #self.add_dtype('decoder', InstanceFromDB(LinearEEGDecoder))
        self.sounds_dir_classical = os.path.expandvars('$HOME/code/ismore/sounds/classical')

               
        # for low-pass filtering decoded EEG velocities
        # self.eeg_vel_buffer = RingBuffer(
        #     item_len=len(self.vel_states),
        #     capacity=10,
        # )

        self.plant.enable() 
        [self.DoF_target_idx_init, self.DoF_target_idx_end, self.DoF_not_control_idx_init, self.DoF_not_control_idx_end] = check_plant_and_DoFs(self.plant_type, self.DoF_control, self.DoF_target)
        self.subgoal_names = self.targets_matrix['subgoal_names']


        #initialize values for the state of the decoder
        self.consec_mov_outputs = 0
        self.consec_rest_outputs = 0
        self.prev_output = 0
        self.state_decoder = 0

        # if self.plant_type == 'ArmAssist':
        #     self.target_margin = np.array([2, 2, np.deg2rad(10)])
        # elif self.plant_type == 'ReHand':
        #     self.target_margin = np.array([np.deg2rad(2),  np.deg2rad(2), np.deg2rad(2), np.deg2rad(2)])
        # elif self.plant_type == 'IsMore':
        #     self.target_margin = np.array([2, 2, np.deg2rad(10), np.deg2rad(2),  np.deg2rad(2), np.deg2rad(2), np.deg2rad(2)])
          
        #self.target_margin = pd.Series(np.array([3, 3, np.deg2rad(15), np.deg2rad(5),  np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)]), ismore_pos_states)
        # target margin used for DK calibration sessions
        self.target_margin = pd.Series(np.array([2, 2, np.deg2rad(8), np.deg2rad(3),  np.deg2rad(3), np.deg2rad(3), np.deg2rad(5)]), ismore_pos_states)
   

        self.target_margin = self.target_margin[self.pos_states]
        self.goal_idx = 0
        self.trial_number = 0


        self.add_dtype('target_margin',      'f8',    (len(self.target_margin),))
        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)

        self.image_dir_general = os.path.expandvars('$HOME/code/ismore/images')
        self.image_dir = os.path.join(self.image_dir_general, self.language)
        

        self.reached_goal_position = False #If the goal_position is reached then give feedback to patient and start the movement back towards the rest_position
        self.reached_subtarget = False #If the task has more than one target position, this shows when the targets before the last target are reached
        self.reached_timeout = False
        self.simult_reach_and_timeout = False


        # 2nd order butterworth filter for command_vel
        fs_synch = self.fps #Frequency at which the task is running
        nyq   = 0.5 * fs_synch
        cuttoff_freq  = 1.5 / nyq
        bpf_kin_coeffs = butter(2, cuttoff_freq, btype='low')

        self.command_lpfs = dict()
        for state in self.vel_states:
            self.command_lpfs[state] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1]) # low-pass filter to smooth out command velocities


        self.init_show_decoder_output()
        print " DoF_target_idx_init : ", self.DoF_target_idx_init
        print " DoF_target_idx_end : ", self.DoF_target_idx_end
        

    def init(self):
        kwargs = {
            'call_rate': self.fps, #kwargs used by the assister
            'xy_cutoff': 2.,#What is this for? Radius of margin?
        }

        from riglib import source
        from ismore.brainamp import rda
       
        #self.brainamp_source = source.MultiChanDataSource(rda.EMGData, name='brainamp', channels=self.brainamp_channels, brainamp_channels=self.brainamp_channels, send_data_to_sink_manager=True)
        self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT[self.plant_type](speed = self.speed,**kwargs)

        self.eeg_extractor.source = self.brainamp_source
        super(EEGMovementDecodingNew, self).init()

    # def _set_goal_position(self):
    #     self.goal_position = self.targets_matrix[self.trial_type]

        

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

        # if self.give_feedback == 1:
        # #     fb_time = self.feedback_time[self.trial_type][self.subtrial_idx]
        #     self.task_data['audio_feedback_start'] = 0

        # # print "goal position: ", self.goal_position
        # # print "plant position: ", self.plant.get_pos()
        # # print "abs difference: ", np.abs(self.pos_diff(self.goal_position,self.plant.get_pos()))
        # # print "target margin: ", self.target_margin

        # if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin):
        #     self.reached_goal_position = True
        #     if self.give_feedback:
        #         self.task_data['audio_feedback_start'] = 1
        #         self._play_sound(os.path.join(self.sounds_general_dir, 'beep.wav'))
        #         self.task_data['audio_feedback_start'] = 0
        # print 'distance to target', self.pos_diff(self.goal_position[self.pos_states],self.plant.get_pos())

        
        #self.task_data['audio_feedback_start'] = 0

        # if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
        if np.all(np.abs(self.pos_diff(self.goal_position[self.DoF_target_idx_init:self.DoF_target_idx_end],self.plant.get_pos()[self.DoF_target_idx_init:self.DoF_target_idx_end])) < self.target_margin[self.pos_states[self.DoF_target_idx_init:self.DoF_target_idx_end]]):

            #self.task_data['audio_feedback_start'] = 1
            
            if self.give_feedback:
                # self._play_sound(os.path.join(self.sounds_general_dir, 'beep.wav'))
                self._play_sound(self.sounds_general_dir, ['beep']) #nerea

            if self.goal_idx < len(self.targets_matrix[self.trial_type].keys())-1:
                pygame.mixer.music.stop() #nerea
                
                self.parallel_sound.stop()
                self.goal_idx +=1
                print 'heading to next subtarget'
                
                self.goal_position = self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states]
               
                self._play_sound(self.sounds_dir, self.subgoal_names[self.trial_type][self.goal_idx]) #nerea
                # pygame.mixer.music.queue(os.path.join(self.sounds_dir_classical, self.subgoal_names[self.trial_type][self.goal_idx][0]+'.wav'))
                self.parallel_sound = pygame.mixer.Sound(os.path.join(self.sounds_dir_classical, self.subgoal_names[self.trial_type][self.goal_idx][0]+'.wav'))
                self.parallel_sound.play()
                # self._play_sound(self.sounds_dir_classical, [self.subgoal_names[self.trial_type][self.goal_idx][0]])  #nerea

                self.reached_subtarget = True
                # #print self.goal_position
            else:
                print 'all subtargets reached'
                self.reached_goal_position = True
            
        # #Show output decoder 
        # if self.state_decoder == 1:
        #     self.image_fname = os.path.join(self.image_dir, 'mov.bmp')
        #     self.show_image(self.image_fname) 
        # else:
        #     self.image_fname = os.path.join(self.image_dir, 'rest.bmp')
        #     self.show_image(self.image_fname) 

    def _while_trial_return(self):

        # if self.give_feedback == 1:
        #     #fb_time = self.feedback_time[self.trial_type][self.subtrial_idx]
        #     self.task_data['audio_feedback_start'] = 0
        # if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin):
        #     self.reached_goal_position = True
        #     #self.goal_position = self.rest_position
        #     if self.give_feedback:
        #         self.task_data['audio_feedback_start'] = 1
        #         self._play_sound(os.path.join(self.sounds_general_dir, 'beep.wav'))


       #self.task_data['audio_feedback_start'] = 0

      
        # if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
        #     self.reached_goal_position = True
        #     #self.goal_position = self.rest_position
        #     #self.task_data['audio_feedback_start'] = 1
        #     if self.give_feedback:
        #         self._play_sound(os.path.join(self.sounds_general_dir, 'beep.wav'))
        #         # self._play_sound(self.sounds_general_dir, ['beep'])


        if np.all(np.abs(self.pos_diff(self.goal_position[self.DoF_target_idx_init:self.DoF_target_idx_end],self.plant.get_pos()[self.DoF_target_idx_init:self.DoF_target_idx_end])) < self.target_margin[self.pos_states[self.DoF_target_idx_init:self.DoF_target_idx_end]]):
            self.reached_goal_position = True
            # pygame.mixer.music.stop() #nerea
            if self.give_feedback:
                # self._play_sound(os.path.join(self.sounds_general_dir, 'beep.wav'))
                self._play_sound(self.sounds_general_dir, ['beep'])
            

        #Show output decoder  
        # if self.state_decoder == 1:
        #     self.image_fname = os.path.join(self.image_dir, 'mov.bmp')
        #     self.show_image(self.image_fname) 
        # else:
        #     self.image_fname = os.path.join(self.image_dir, 'rest.bmp')
        #     self.show_image(self.image_fname) 
    # def _while_rest(self):
        
    #     #self.mov_data = self.mov_data_buffer.get_all()
    #     #self.rest_data = self.rest_data_buffer.get_all()     
    #     #rest_features, mov_features = self.eeg_extractor.extract_features_2retrain(self.rest_data, self.mov_data)
       
      
    def move_plant(self):
        '''Docstring.'''

        command_vel  = pd.Series(0.0, self.vel_states)
        command_vel_raw  = pd.Series(0.0, self.vel_states)
        command_vel_final  = pd.Series(0.0, self.vel_states)

        # run EEG feature extractor and decoder
        #self.eeg_extractor.source = self.brainamp_source
        if self.artifact_rejection == 1:
            eeg_features, rejected_window = self.eeg_extractor()
            self.task_data['rejected_window'] = rejected_window
        else:
            eeg_features = self.eeg_extractor() # eeg_features is of type 'dict'
            rejected_window = 0

        feat_mov = self.mov_feature_buffer.get_all()
        feat_rest = self.rest_feature_buffer.get_all()
        mean_feat = np.mean(np.hstack([feat_mov, feat_rest]), axis = 1)
        std_feat = np.std(np.hstack([feat_mov, feat_rest]), axis = 1)


        if self.trial_number > 0:
            if self.state in ['trial','trial_return']:
                if self.artifact_rejection == 1 & rejected_window == 0:
                    self.mov_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
                elif self.artifact_rejection == 0:
                    self.mov_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
                self.task_data['eeg_features_mov_buffer'] = eeg_features
                #self.mov_feature_buffer.add(eeg_features)
            elif self.state in ['rest','rest_return']:
                if self.artifact_rejection == 1 & rejected_window == 0:
                    self.rest_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
                elif self.artifact_rejection == 0:
                    self.rest_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
                    
                self.task_data['eeg_features_rest_buffer'] = eeg_features
         
        
        self.task_data['eeg_features'] = eeg_features
        self.task_data['eeg_mean_features'] = mean_feat
        self.task_data['eeg_std_features'] = std_feat
        #self.task_data['channels'] = self.channels
        # normalize features
        # eeg_features = (eeg_features - mean_feat.reshape(-1,1))/ std_feat.reshape(-1,1)
        eeg_features = (eeg_features - mean_feat)/ std_feat
        # mean_feat.ravel()
        self.task_data['eeg_features_Z'] = eeg_features

        #print 'eeg_features.shpae'

        try:
            eeg_features = np.reshape(eeg_features,(1,eeg_features.shape[0]))
        except:
            pass
        #eeg_features(eeg_features == np.inf) = 1
        self.decoder_output = self.eeg_decoder(eeg_features)
        self.probability = self.eeg_decoder.decoder.predict_proba(eeg_features)
    
        # print "dec probability : ", self.probability

        #print self.decoder_output, ' with probability:', probability

        # Command zero velocity if the task is in a non-moving state
        if self.state not in ['trial', 'trial_return','drive_to_start']:#['wait','rest', 'rest_return','instruct_rest', 'instruct_trial_type', 'preparation', 'instruct_go','instruct_rest_return']: 
            command_vel[:] = 0
            self.state_decoder = 0

        
        else:#if self.state in ['trial', 'trial_return', 'instruct_trial_return']:
            if self.decoder_output == 1 and self.prev_output == 1:
                # we need 5 consecutive outputs of the same type
                self.consec_mov_outputs +=1
                if self.consec_mov_outputs == 5 and self.state_decoder == 0:
                    self.consec_rest_outputs = 0
            elif self.decoder_output == 1 and self.prev_output == 0:
                if self.state_decoder == 1: #if it's moving
                    self.consec_rest_outputs = 0
                else:
                    self.consec_mov_outputs = 1
            elif self.decoder_output == 0 and self.prev_output == 0:
                self.consec_rest_outputs +=1
                if self.consec_rest_outputs == 5 and self.state_decoder == 1:
                    self.consec_mov_outputs = 0
            elif self.decoder_output == 0 and self.prev_output == 1:
                if self.state_decoder == 1: #if it's moving
                    self.consec_rest_outputs = 1
                else:
                    self.consec_mov_outputs = 0
    
            if self.consec_mov_outputs >= 5:

                self.state_decoder = 1
                current_pos = self.plant_pos[:].ravel()
                current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
                target_state = np.hstack([self.goal_position, np.zeros_like(current_pos), 1]).reshape(-1, 1)
                assist_output = self.assister(current_state, target_state, 1)
                           
                Bu = np.array(assist_output["x_assist"]).ravel()
                #Bu = np.array(assist_output['Bu']).ravel()
                command_vel[:] = Bu[len(current_pos):len(current_pos)*2]
                #print 'command_vel', command_vel
                #set all the velocities to a constant value towards the end point
            elif self.consec_rest_outputs >=5:

                self.state_decoder = 0
                command_vel[:] = 0 #set all the velocities to zero
        
        if self.state in ['drive_to_start', 'timeout_penalty', 'reward']:
            current_pos = self.plant_pos[:].ravel()
            current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
            self.goal_position = self.targets_matrix['rest'][self.goal_idx][self.pos_states]
            target_state = np.hstack([self.goal_position, np.zeros_like(current_pos), 1]).reshape(-1, 1)
            assist_output = self.assister(current_state, target_state, 1)
                         
            Bu = np.array(assist_output["x_assist"]).ravel()
            command_vel[:] = Bu[len(current_pos):len(current_pos)*2]

        # Command zero velocity if the task is in a non-moving state
        if self.state not in ['trial', 'trial_return','drive_to_start']:#['wait','rest', 'rest_return','instruct_rest', 'instruct_trial_type', 'preparation', 'instruct_go','instruct_rest_return']: 
            command_vel[:] = 0
            self.state_decoder = 0

        if not self.state in ['drive_to_start', 'timeout_penalty', 'reward']:

            command_vel_raw[:] = command_vel[:]
            for state in self.vel_states:
                command_vel[state] = self.command_lpfs[state](command_vel[state])

            
        self.prev_output = self.decoder_output
        #self.task_data['eeg_features']   = eeg_features[self.eeg_extractor.feature_type]
        
        self.task_data['decoder_output'] = self.decoder_output
        self.task_data['decoder_output_probability'] = self.probability
        self.task_data['state_decoder'] = self.state_decoder
        self.task_data['consec_mov_outputs'] = self.consec_mov_outputs
        self.task_data['consec_rest_outputs'] = self.consec_rest_outputs

        # # Before 2017/08/21
        # self.plant.send_vel(command_vel.values) #send velocity command to EXO

        # self.task_data['command_vel']  = command_vel.values
        # self.task_data['command_vel_raw']  = command_vel_raw.values



        # After 2017/08/21 - only control the DoFs selected 
        self.task_data['command_vel']  = command_vel.values
        self.task_data['command_vel_raw']  = command_vel_raw.values

        if np.isnan(self.DoF_not_control_idx_init) == False and np.isnan(self.DoF_not_control_idx_end) == False:
            command_vel[self.DoF_not_control_idx_init:self.DoF_not_control_idx_end] = 0
        
        self.plant.send_vel(command_vel.values) #send velocity command to EXO
        self.task_data['command_vel_final']  = command_vel.values
       

    def _cycle(self):
        '''Runs self.fps times per second.'''

        # get latest position/velocity information before calling move_plant()
        self.plant_pos_raw[:] = self.plant.get_pos_raw()
        self.plant_pos[:] = self.plant.get_pos() 
        
        self.plant_vel_raw[:] = self.plant.get_vel_raw()
        self.plant_vel[:] = self.plant.get_vel()

        #if self.state in ['trial','go_to_start']:
        # velocity control
        self.move_plant()

        # position control
        # self.move_plant_pos_control()

        self.update_plant_display()
        self.update_decoder_ouput()

        # print self.subtrial_idx
        if not self.state in ['trial','trial_return']:
            #self.task_data['audio_feedback_start'] = 0
            self.task_data['goal_pos'] = np.ones(len(self.pos_states))*np.nan            
            self.task_data['goal_idx'] = np.nan
            
        else:
            self.task_data['goal_pos'] = self.goal_position
            self.task_data['goal_idx'] = self.goal_idx
            

        self.task_data['plant_type'] = self.plant_type
        self.task_data['eeg_decoder_coef']  = self.eeg_decoder.decoder.coef_
        self.task_data['eeg_decoder_intercept']  = self.eeg_decoder.decoder.intercept_
        self.task_data['eeg_decoder_means']  = self.eeg_decoder.decoder.means_
        self.task_data['reached_goal_position'] = self.reached_goal_position  
        self.task_data['reached_subtarget'] = self.reached_subtarget          
            
        self.task_data['reached_timeout'] = self.reached_timeout
        self.task_data['simult_reach_and_timeout'] = self.simult_reach_and_timeout
        
        self.task_data['plant_pos']  = self.plant_pos_raw.values
        self.task_data['plant_pos_filt']  = self.plant_pos.values 
        self.task_data['plant_vel']  = self.plant_vel_raw.values
        self.task_data['plant_vel_filt']  = self.plant_vel.values

        self.task_data['trial_type'] = self.trial_type
        self.task_data['speed'] = self.speed
        self.task_data['ts']         = time.time()
        self.task_data['target_margin'] = self.target_margin
        #self.task_data['decoder']    = self.eeg_decoder.decoder

        super(EEGMovementDecodingNew, self)._cycle()

    def init_show_decoder_output(self):

        self.decoder_background_line  = Line(np.array([80, 0]), 100, 3, 1, COLORS['blue'])
        self.add_model(self.decoder_background_line)

        self.decoder_move_perc_line  = Line(np.array([80, 0]), 2, 3, 1, COLORS['red'])
        self.add_model(self.decoder_move_perc_line)

        self.decoder_middle_line = Line(np.array([80, 49]), 0.2, 3, 1, COLORS['white'])
        self.add_model(self.decoder_middle_line)



    def update_decoder_ouput(self):

        #backgroun line in white 
        self.decoder_background_line.color = COLORS['blue']
        self.decoder_background_line.start_pos   = np.array([80, 0])
        self.decoder_background_line.angle = 90*deg_to_rad

        #movement output in green
        self.decoder_move_perc_line.length = self.probability[0,1]*np.int(100)
        self.decoder_move_perc_line.start_pos   = np.array([80, 0])
        self.decoder_move_perc_line.angle = 90*deg_to_rad
        self.decoder_move_perc_line.color = COLORS['red']

        self.decoder_middle_line.color = COLORS['white']
        self.decoder_middle_line.start_pos   = np.array([80, 49])
        self.decoder_middle_line.angle = 90*deg_to_rad

    # def show_image(self, image_fname):

    #     window = pygame.display.set_mode(self.window_size)
    #     img = pygame.image.load(os.path.join(self.image_fname))
    #     img = pygame.transform.scale(img, self.window_size)

    #     window.blit(img, (0,0))
    #     pygame.display.flip()

    #### STATE AND TEST FUNCTIONS ####   
    def _start_wait(self):
        print 'wait'
        # determine the random length of time to stay in the rest state
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time
        super(EEGMovementDecodingNew, self)._start_wait()

    

    def _test_late_end_trial(self, ts):
       
        try:
            if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
                if self.goal_idx > len(self.targets_matrix[self.trial_type].keys())-1:
                    #print 'all subtargets reached in last clock cycle'
                    self.reached_goal_position = True

        except:
            pass
        return ts > self.rest_time  and self.reached_goal_position and self.reached_timeout

    def _test_end_rest(self, ts):
        return ts > self.rest_time  

    def _test_late_end_trial_return(self, ts):
        
        if np.all(np.abs(self.pos_diff(self.targets_matrix['rest'][0][self.pos_states],self.plant.get_pos())) < self.target_margin[self.pos_states]):
            self.reached_goal_position = True
            #print 'rest targets reached in last clock cycle'
            self.trial_type = self.next_trial

        return ts > self.rest_time  and self.reached_goal_position and self.reached_timeout

    def _test_end_rest_return(self, ts):
        return ts > self.rest_time  

    def _test_end_preparation(self, ts):
        return ts > self.preparation_time  

    def _test_end_preparation_return(self, ts):
        return ts > self.preparation_time  

    def _parse_next_trial(self):
        self.trial_type = self.next_trial
        
    def _test_end_instruct(self, *args, **kwargs):
        return not pygame.mixer.music.get_busy()

    def _start_instruct_rest(self):
        self.parallel_sound.stop()
        # self._play_sound(os.path.join(self.sounds_dir, 'rest.wav')) #nerea
        self._play_sound(self.sounds_dir, ['rest'])
        #initial_mov_buffer_data = self.mov_data_buffer.get_all()
        print 'rest'

    def _start_instruct_rest_return(self):
        self.parallel_sound.stop()
        # self._play_sound(os.path.join(self.sounds_dir, 'rest.wav')) #nerea
        self._play_sound(self.sounds_dir, ['rest'])
        #initial_mov_buffer_data = self.mov_data_buffer.get_all()
        print 'rest'

    def _start_instruct_trial_type(self):
        #print 'instruct trial type'
        # sound_fname = os.path.join(self.sounds_dir, self.trial_type + '.wav') #nerea
        # self._play_sound(sound_fname)
        # self._play_sound(self.sounds_dir, self.subgoal_names[self.trial_type][0]) 
        self._play_sound(self.sounds_dir, self.subgoal_names[self.trial_type][self.goal_idx]) #nerea ??
        
        mov_features = self.mov_feature_buffer.get_all().T
        rest_features = self.rest_feature_buffer.get_all().T

        # normalization of features
        self.features = np.vstack([mov_features, rest_features])
        mean_features = np.mean(self.features, axis = 0)
        std_features = np.std(self.features, axis = 0)
        self.features = (self.features - mean_features) / std_features
        
        self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])
        print 'retraining decoder'
        self.retrained_decoder.fit(self.features, self.labels.ravel())
        
        # import time
        # t0 = time.time()
        self.eeg_decoder.decoder = copy.copy(self.retrained_decoder)
        # print 'time2retrain', time.time() - t0
        #print 'decoder retrained'
        
        # self.consec_mov_outputs = 0
        # self.consec_rest_outputs = 0

    def _start_instruct_trial_return(self):
        # sound_fname = os.path.join(self.sounds_dir, 'back.wav')#nerea
        # self._play_sound(sound_fname)
        self._play_sound(self.sounds_dir, ['back'])
        

        mov_features = self.mov_feature_buffer.get_all().T
        rest_features = self.rest_feature_buffer.get_all().T

       # normalization of features
        self.features = np.vstack([mov_features, rest_features])
        mean_features = np.mean(self.features, axis = 0)
        std_features = np.std(self.features, axis = 0)
        self.features = (self.features - mean_features) / std_features

        self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])
        self.retrained_decoder.fit(self.features, self.labels.ravel())
        # import time
        # t0 = time.time()
        self.eeg_decoder.decoder = copy.copy(self.retrained_decoder)
        # print 'time2retrain', time.time() - t0
        #print 'decoder retrained'
        # self.consec_mov_outputs = 0
        # self.consec_rest_outputs = 0
        # self.state_decoder = 0


    def _start_instruct_go(self):
        # self._play_sound(os.path.join(self.sounds_dir, 'go.wav'))
        self._play_sound(self.sounds_dir, ['go'])
        self.consec_mov_outputs = 0
        self.consec_rest_outputs = 0
        self.reached_goal_position = False
        self.reached_subtarget = False
        self.reached_timeout = False
        self.simult_reach_and_timeout = False
        #self.state_decoder = 0

    def _start_instruct_go_return(self):
        # self._play_sound(os.path.join(self.sounds_dir, 'go.wav')) #nerea
        self._play_sound(self.sounds_dir, ['go'])
        self.consec_mov_outputs = 0
        self.consec_rest_outputs = 0
        self.reached_goal_position = False
        self.reached_subtarget = False
        self.reached_timeout = False
        self.simult_reach_and_timeout = False
        #self.state_decoder = 0

    def _start_trial(self):
        print self.trial_type
        #self.plant.set_pos_control() #to set it to position control during the trial state
        #self._set_task_type()
        #self._set_goal_position()
        self.goal_position = self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states]     
        if self.music_feedback:
            self._play_sound(self.sounds_dir_classical, [self.subgoal_names[self.trial_type][self.goal_idx][0]])
    

    def _start_trial_return(self):
        print 'return trial'
        #self.plant.set_pos_control() #to set it to position control during the trial state

        #self._set_task_type()
        
        self.goal_position = self.targets_matrix['rest'][0][self.pos_states]
        self.goal_idx = 0
        if self.music_feedback:
            self._play_sound(self.sounds_dir_classical, [self.subgoal_names[self.trial_type][self.goal_idx][0]])
    

    def _test_end_trial(self,ts):
        # Test if simultaneous timeout and end_trial issue is solved with this
        # if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
        #     if self.goal_idx >= len(self.targets_matrix[self.trial_type].keys())-1:               
        #         self.reached_goal_position = True
        # if ts > self.timeout_time:
        #     self.reached_timeout = True
        #     print 'timeout'
        # if self.reached_timeout == True and np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
        #     self.reached_goal_position = True
        
        return (self.reached_goal_position or self.reached_timeout)

    def _test_end_trial_return(self,ts):
        # Test if simultaneous timeout and end_trial issue is solved with this
        # if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
        #     self.reached_goal_position = True

        return (self.reached_goal_position or self.reached_timeout)

    def _test_timeout(self, ts):
        if ts > self.timeout_time:
            self.reached_timeout = True
            if self.reached_goal_position == True:
                #self.reached_timeout = False
                self.simult_reach_and_timeout = True
            #print 'reached goal position', self.reached_goal_position
            print 'timeout'
            # if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
            #     self.reached_timeout = False
        return self.reached_timeout

    # def _test_at_starting_config(self, *args, **kwargs):
    #     traj = self.ref_trajectories[self.trial_type]['traj']
    #     diff_to_start = np.abs(self.plant.get_pos() - traj[self.pos_states].ix[0].ravel())
    #     #print diff_to_start

    #     return np.all(diff_to_start < self.rest_rect[:len(self.pos_states)])
    def _end_trial(self):
        
        self.trial_number +=1
        if self.music_feedback:
            pygame.mixer.music.stop()
            self.parallel_sound.stop()
        else:
            pass
    def _end_trial_return(self):
    
        if self.music_feedback:
                pygame.mixer.music.stop()
                self.parallel_sound.stop()
        else:
            pass

    def cleanup_hdf(self):
        super(EEGMovementDecodingNew, self).cleanup_hdf()    

        import tables
        h5file = tables.openFile(self.h5file.name, mode='a')
        h5file.root.task.attrs['eeg_decoder_name'] = self.eeg_decoder_name
        #h5file.root.task.attrs['brainamp_channels'] = self.channel_list_name
        #compfilt = tables.Filters(complevel=5, complib="zlib", shuffle=True)
        eeg_extractor_grp = h5file.createGroup(h5file.root, "eeg_extractor_kwargs", "Parameters for feature extraction")
        for key in self.eeg_extractor_kwargs:
            if isinstance(self.eeg_extractor_kwargs[key], dict):
                if key == 'feature_fn_kwargs':
                    for key2 in self.eeg_extractor_kwargs['feature_fn_kwargs']['AR']['freq_bands']:
                        if isinstance(self.eeg_extractor_kwargs['feature_fn_kwargs']['AR']['freq_bands'][key2], np.ndarray):
                            h5file.createArray(eeg_extractor_grp, 'freq_band_' + key2, self.eeg_extractor_kwargs['feature_fn_kwargs']['AR']['freq_bands'][key2])
                        else:
                            h5file.createArray(eeg_extractor_grp, 'freq_band_' + key2, np.array([self.eeg_extractor_kwargs['feature_fn_kwargs']['AR']['freq_bands'][key2]]))
                else:
                    for key2 in self.eeg_extractor_kwargs[key]:
                        if isinstance(self.eeg_extractor_kwargs[key][key2], np.ndarray):
                            h5file.createArray(eeg_extractor_grp, key + '_' + key2, self.eeg_extractor_kwargs[key][key2])
                        else:
                            h5file.createArray(eeg_extractor_grp, key + '_' + key2, np.array([self.eeg_extractor_kwargs[key][key2]]))

            else:
                if isinstance(self.eeg_extractor_kwargs[key], np.ndarray):
                    h5file.createArray(eeg_extractor_grp, key, self.eeg_extractor_kwargs[key])
                else:
                    h5file.createArray(eeg_extractor_grp, key, np.array([self.eeg_extractor_kwargs[key]]))
                

        h5file.close()

    def cleanup(self, database, saveid, **kwargs):
        #Old way of buffering rest and mov data
        # self.mov_data = self.mov_data_buffer.get_all()
        # self.rest_data = self.rest_data_buffer.get_all()
        #rest_features, mov_features = self.eeg_extractor.extract_features_2retrain(self.rest_data, self.mov_data)
        #self.features = np.vstack([mov_features, rest_features])
        #self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])

        # New method of buffering rest and mov data to retrain decoder
        mov_features = self.mov_feature_buffer.get_all().T
        rest_features = self.rest_feature_buffer.get_all().T
        # normalization of features
        self.features = np.vstack([mov_features, rest_features])
        mean_features = np.mean(self.features, axis = 0)
        std_features = np.std(self.features, axis = 0)
        self.features = (self.features - mean_features) / std_features

        
        self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])
        self.retrained_decoder.fit(self.features, self.labels.ravel())


        self.eeg_decoder.rest_feature_buffer = self.rest_feature_buffer
        self.eeg_decoder.mov_feature_buffer = self.mov_feature_buffer
        self.eeg_decoder.decoder = copy.copy(self.retrained_decoder)

        #Values just to make it compatible with the task interface (they are nonsense)
        self.eeg_decoder.units  = self.eeg_decoder.channels_2train
        # self.eeg_decoder.binlen = # the decoder is updated after the end of each return trial
        # self.eeg_decoder.tslice = 

        #save eeg_decder object into a new pkl file. 
        storage_dir = '/storage/decoders'
        eeg_decoder_name = self.eeg_decoder.decoder_name 
        # n = decoder_name[-1]
        # n = int(n)        
        index = eeg_decoder_name.rfind('_') + 1
        #new_decoder_name = decoder_name[0:index] + str(n + 1)
        new_decoder_name = eeg_decoder_name[0:index] + str(saveid) 
        self.eeg_decoder.decoder_name = new_decoder_name
        new_pkl_name = new_decoder_name + '.pkl' 
        pickle.dump(self.eeg_decoder, open(os.path.join(storage_dir, new_pkl_name), 'wb'))

        super(EEGMovementDecodingNew,self).cleanup(database, saveid, **kwargs)
        # Create a new database record for the decoder object if it doesn't already exist
        dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
        dfs = models.Decoder.objects.filter(name=new_decoder_name)
        if len(dfs) == 0:
            df = models.Decoder()
            df.path = new_pkl_name
            df.name = new_decoder_name
            df.entry = models.TaskEntry.objects.using(dbname).get(id=saveid) 
            df.save()
        elif len(dfs) == 1:
            pass # no new data base record needed
        elif len(dfs) > 1:
            print "More than one decoder with the same name! fix manually!"


class EEGCyclicMovementDecodingNew(NonInvasiveBase):
    # Unlike the EEGMovementDecoding task, it keeps going towards the same target until it reaches the target position
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
            'timeout': 'instruct_rest',#'instruct_trial_go_to_start'
            'stop':      None},  
       
        }
    
    state = 'wait'  # initial state

     # settable parameters on web interface    
    eeg_decoder          = traits.InstanceFromDB(LinearEEGDecoder, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='eeg_decoder'))
    rest_interval        = traits.Tuple((3., 4.), desc='Min and max time to remain in the rest state.')
    preparation_time     = traits.Float(2, desc='time to remain in the preparation state.')
    timeout_time         = traits.Float(7, desc='Maximum time given to the patient to accomplish the task before considering it like incomplete and re-starting it from the current position') 
    give_feedback        = traits.Int((0,1), desc=' 0 if we do not give feedback, 1 if we give feedback about whether the patient reached the goal position')
    targets_matrix       = traits.DataFile(object, desc='goal positions for each of the trial types', bmi3d_query_kwargs=dict(system__name='misc'))
    window_size          = traits.Tuple((1000, 560), desc='Size of window to display the plant position/angle')
    channel_list_name    = traits.OptionsList(*channel_list_options, bmi3d_input_options=channel_list_options)
    speed                = traits.OptionsList(*speed_options, bmi3d_input_options= speed_options)
    music_feedback       = traits.Int((1), desc=' 0 if we do not want to include music, 1 if we want different classical music pieces with increasing intensity to be played')
    #artifact_rejection   = traits.Int(1, desc=' 0 if artifacts are not rejected online, 1 if the artifact rejection is applied in real-time too')
    #session_length = traits.Float(20, desc='overall time that the block will last') #It shows up by default in the interface
    #neighbour_channels = ???
    debug = False
    DoF_control = traits.OptionsList(*DoF_control_options, bmi3d_input_options=DoF_control_options, desc='DoFs to be taken into account for condition fulfilment')
    DoF_target = traits.OptionsList(*DoF_target_options, bmi3d_input_options=DoF_target_options, desc='DoFs to be moved/controlled, the rest are stopped.')


    # Set the "is_bmi_seed" flag so that the server knows that this is a task which can be used to create a BMI decoder
    #is_bmi_seed = False
    # def _play_sound(self, fname):
    #     pygame.mixer.music.load(fname)
    #     pygame.mixer.music.play()
    def _play_sound(self, fpath,fname):

        for filename in fname:
            # print 'filename ', filename
            if '_' in filename:
                filename = filename[:filename.find('_')]
            sound_fname = os.path.join(fpath, filename + '.wav')
            pygame.mixer.music.load(sound_fname)
            pygame.mixer.music.play()
            # print 'sound_fname ' , sound_fname
            # while pygame.mixer.music.get_busy():
            #     time.sleep(0)
            # else:
            #     pygame.mixer.music.load(sound_fname)
            #     pygame.time.Clock().tick(1)
            #     # print 'clock'
            #     pygame.mixer.music.play()

    def __init__(self, *args, **kwargs):
        super(EEGCyclicMovementDecodingNew, self).__init__(*args, **kwargs)

        self.add_dtype('command_vel',  'f8',    (len(self.vel_states),))
        self.add_dtype('command_vel_raw',  'f8',    (len(self.vel_states),))
        self.add_dtype('command_vel_final',  'f8',    (len(self.vel_states),))
        self.add_dtype('trial_type',   np.str_, 40)
        #self.add_dtype('plant_type',   np.str_, 40)
        
        self.add_dtype('ts',           'f8',    (1,))
        self.add_dtype('goal_pos',      'f8',    (len(self.pos_states),))
        #self.add_dtype('difference_position','f8', (len(self.pos_states),))
        self.add_dtype('reached_goal_position',bool, (1,))
        self.add_dtype('reached_subtarget',bool, (1,))
        self.add_dtype('reached_timeout',bool, (1,))
        self.add_dtype('simult_reach_and_timeout',bool, (1,))
        #self.add_dtype('audio_feedback_start',      int,    (1,))
        self.add_dtype('goal_idx', float, (1,))
        self.add_dtype('speed',   np.str_, 20)


        self.parallel_sound = pygame.mixer.Sound('')
      
        # if EEG decoder file was specified, load EEG decoder and create feature extractor 
        
        #if len(self.decoder_file) > 3:
        #    self.eeg_decoder = pickle.load(open(self.decoder_file, 'rb'))

        # create EEG extractor object (its 'source' will be set later in the init method)
      
        eeg_extractor_cls    = self.eeg_decoder.extractor_cls
        self.eeg_decoder_name = self.eeg_decoder.decoder_name
        self.eeg_extractor_kwargs = self.eeg_decoder.extractor_kwargs
        self.artifact_rejection = self.eeg_extractor_kwargs['artifact_rejection']
        
        # Check if chosen decoder is trained with artifact rejection or not. If artifact_rejection = 1 and decoder not designed for that, print an error!

        self.TH_lowF = self.eeg_decoder.TH_lowF 
        self.TH_highF = self.eeg_decoder.TH_highF 
        self.eog_coeffs = self.eeg_decoder.eog_coeffs 


        self.rest_feature_buffer = self.eeg_decoder.rest_feature_buffer
        #self.trial_hand_side = self.eeg_extractor_kwargs['trial_hand_side']
        self.mov_feature_buffer = self.eeg_decoder.mov_feature_buffer
        try:
            self.channels = self.eeg_extractor_kwargs['eeg_channels']
        except:
            self.channels = self.eeg_extractor_kwargs['channels']
        #self.channels = self.eeg_extractor_kwargs['eeg_channels']
        self.brainamp_channels = getattr(brainamp_channel_lists, self.channel_list_name)  

        #self.brainamp_channels = self.eeg_extractor_kwargs['brainamp_channels']
        self.neighbour_channels = self.eeg_decoder.neighbour_channels
        
        self.eeg_playback = False
        self.fs = self.eeg_extractor_kwargs['fs']

        self.retrained_decoder = copy.copy(self.eeg_decoder.decoder)
        
        self.eeg_extractor_kwargs['eog_coeffs'] = self.eog_coeffs 
        self.eeg_extractor_kwargs['TH_lowF'] = self.TH_lowF 
        self.eeg_extractor_kwargs['TH_highF'] = self.TH_highF 

        self.eeg_extractor = eeg_extractor_cls(source=None, **self.eeg_extractor_kwargs)
        self.n_features = self.eeg_extractor.n_features
        #dtype = np.dtype(['name',       np.str, [len(self.channels),20])
        
        #self.add_dtype('eeg_features',    'f8', (self.eeg_extractor.n_features,))
        self.add_dtype('eeg_features',    'f8', (self.n_features,))
        #self.add_dtype('channels',   np.str_, [len(self.channels),20])
        self.add_dtype('eeg_features_mov_buffer',    'f8', (self.n_features,))
        self.add_dtype('eeg_features_rest_buffer',    'f8', (self.n_features,))
        self.add_dtype('eeg_features_Z',    'f8', (self.n_features,))
        self.add_dtype('eeg_mean_features',    'f8', (self.n_features,))
        self.add_dtype('eeg_std_features',    'f8', (self.n_features,))
        self.add_dtype('eeg_decoder_coef',    'f8', (self.n_features,))
        self.add_dtype('eeg_decoder_intercept', 'f8', (self.n_features,))
        self.add_dtype('eeg_decoder_means', 'f8', (2,self.n_features))
        self.add_dtype('decoder_output',    'f8', (1,))
        self.add_dtype('decoder_output_probability',    'f8', (1,2))#2 classes
        self.add_dtype('state_decoder',  int, (1,))
        self.add_dtype('consec_mov_outputs',  int, (1,))
        self.add_dtype('consec_rest_outputs',  int, (1,))
        self.add_dtype('rejected_window',  int, (1,))
        #self.add_dtype('decoder', InstanceFromDB(LinearEEGDecoder))
        self.sounds_dir_classical = os.path.expandvars('$HOME/code/ismore/sounds/classical')

               
        # for low-pass filtering decoded EEG velocities
        # self.eeg_vel_buffer = RingBuffer(
        #     item_len=len(self.vel_states),
        #     capacity=10,
        # )

        self.plant.enable() 
        [self.DoF_target_idx_init, self.DoF_target_idx_end, self.DoF_not_control_idx_init, self.DoF_not_control_idx_end] = check_plant_and_DoFs(self.plant_type, self.DoF_control, self.DoF_target)
        self.subgoal_names = self.targets_matrix['subgoal_names']


        #initialize values for the state of the decoder
        self.consec_mov_outputs = 0
        self.consec_rest_outputs = 0
        self.prev_output = 0
        self.state_decoder = 0

        # if self.plant_type == 'ArmAssist':
        #     self.target_margin = np.array([2, 2, np.deg2rad(10)])
        # elif self.plant_type == 'ReHand':
        #     self.target_margin = np.array([np.deg2rad(2),  np.deg2rad(2), np.deg2rad(2), np.deg2rad(2)])
        # elif self.plant_type == 'IsMore':
        #     self.target_margin = np.array([2, 2, np.deg2rad(10), np.deg2rad(2),  np.deg2rad(2), np.deg2rad(2), np.deg2rad(2)])
          
        #self.target_margin = pd.Series(np.array([3, 3, np.deg2rad(15), np.deg2rad(5),  np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)]), ismore_pos_states)
        # target margin used for DK calibration sessions
        self.target_margin = pd.Series(np.array([2, 2, np.deg2rad(8), np.deg2rad(3),  np.deg2rad(3), np.deg2rad(3), np.deg2rad(3)]), ismore_pos_states)
   

        self.target_margin = self.target_margin[self.pos_states]
        self.goal_idx = 0
        self.trial_number = 0


        self.add_dtype('target_margin',      'f8',    (len(self.target_margin),))
        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)

        self.image_dir_general = os.path.expandvars('$HOME/code/ismore/images')
        self.image_dir = os.path.join(self.image_dir_general, self.language)
        

        self.reached_goal_position = False #If the goal_position is reached then give feedback to patient and start the movement back towards the rest_position
        self.reached_subtarget = False #If the task has more than one target position, this shows when the targets before the last target are reached
        self.reached_timeout = False
        self.simult_reach_and_timeout = False


        # 2nd order butterworth filter for command_vel
        fs_synch = self.fps #Frequency at which the task is running
        nyq   = 0.5 * fs_synch
        cuttoff_freq  = 1.5 / nyq
        bpf_kin_coeffs = butter(2, cuttoff_freq, btype='low')

        self.command_lpfs = dict()
        for state in self.vel_states:
            self.command_lpfs[state] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1]) # low-pass filter to smooth out command velocities


        self.init_show_decoder_output()
        

    def init(self):
        kwargs = {
            'call_rate': self.fps, #kwargs used by the assister
            'xy_cutoff': 2.,#What is this for? Radius of margin?
        }

        from riglib import source
        from ismore.brainamp import rda
       
        #self.brainamp_source = source.MultiChanDataSource(rda.EMGData, name='brainamp', channels=self.brainamp_channels, brainamp_channels=self.brainamp_channels, send_data_to_sink_manager=True)
        self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT[self.plant_type](speed = self.speed,**kwargs)
        self.eeg_extractor.source = self.brainamp_source
        super(EEGCyclicMovementDecodingNew, self).init()

    # def _set_goal_position(self):
    #     self.goal_position = self.targets_matrix[self.trial_type]

        

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

        # if self.give_feedback == 1:
        # #     fb_time = self.feedback_time[self.trial_type][self.subtrial_idx]
        #     self.task_data['audio_feedback_start'] = 0

        # # print "goal position: ", self.goal_position
        # # print "plant position: ", self.plant.get_pos()
        # # print "abs difference: ", np.abs(self.pos_diff(self.goal_position,self.plant.get_pos()))
        # # print "target margin: ", self.target_margin

        # if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin):
        #     self.reached_goal_position = True
        #     if self.give_feedback:
        #         self.task_data['audio_feedback_start'] = 1
        #         self._play_sound(os.path.join(self.sounds_general_dir, 'beep.wav'))
        #         self.task_data['audio_feedback_start'] = 0
        # print 'distance to target', self.pos_diff(self.goal_position[self.pos_states],self.plant.get_pos())

        
        #self.task_data['audio_feedback_start'] = 0

        if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
            #self.task_data['audio_feedback_start'] = 1
            
            if self.give_feedback:
                # self._play_sound(os.path.join(self.sounds_general_dir, 'beep.wav'))
                self._play_sound(self.sounds_general_dir, ['beep']) #nerea

            if self.goal_idx < len(self.targets_matrix[self.trial_type].keys())-1:
                pygame.mixer.music.stop() #nerea
                
                self.parallel_sound.stop()
                self.goal_idx +=1
                print 'heading to next subtarget'
                
                self.goal_position = self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states]
               
                # self._play_sound(self.sounds_dir, self.subgoal_names[self.trial_type][self.goal_idx]) #nerea
                self._play_sound(self.sounds_general_dir, ['beep']) #nerea
                # pygame.mixer.music.queue(os.path.join(self.sounds_dir_classical, self.subgoal_names[self.trial_type][self.goal_idx][0]+'.wav'))
                self.parallel_sound = pygame.mixer.Sound(os.path.join(self.sounds_dir_classical, self.subgoal_names[self.trial_type][self.goal_idx][0]+'.wav'))
                self.parallel_sound.play()
                # self._play_sound(self.sounds_dir_classical, [self.subgoal_names[self.trial_type][self.goal_idx][0]])  #nerea

                self.reached_subtarget = True
                # #print self.goal_position
            else:
                print 'all subtargets reached'
                self.reached_goal_position = True
            
        # #Show output decoder 
        # if self.state_decoder == 1:
        #     self.image_fname = os.path.join(self.image_dir, 'mov.bmp')
        #     self.show_image(self.image_fname) 
        # else:
        #     self.image_fname = os.path.join(self.image_dir, 'rest.bmp')
        #     self.show_image(self.image_fname) 

   
    def move_plant(self):
        '''Docstring.'''

        command_vel  = pd.Series(0.0, self.vel_states)
        command_vel_raw  = pd.Series(0.0, self.vel_states)

        # run EEG feature extractor and decoder
        #self.eeg_extractor.source = self.brainamp_source
        if self.artifact_rejection == 1:
            eeg_features, rejected_window = self.eeg_extractor()
            self.task_data['rejected_window'] = rejected_window
        else:
            eeg_features = self.eeg_extractor() # eeg_features is of type 'dict'
          

        feat_mov = self.mov_feature_buffer.get_all()
        feat_rest = self.rest_feature_buffer.get_all()
        mean_feat = np.mean(np.hstack([feat_mov, feat_rest]), axis = 1)
        std_feat = np.std(np.hstack([feat_mov, feat_rest]), axis = 1)


        if self.trial_number > 0:
            if self.state in ['trial']:
                if self.artifact_rejection == 1 & rejected_window == 0:
                    self.mov_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
                elif self.artifact_rejection == 0:
                    self.mov_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
                self.task_data['eeg_features_mov_buffer'] = eeg_features
                #self.mov_feature_buffer.add(eeg_features)
            elif self.state in ['rest']:
                if self.artifact_rejection == 1 & rejected_window == 0:
                    self.rest_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
                elif self.artifact_rejection == 0:
                    self.rest_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
                    
                self.task_data['eeg_features_rest_buffer'] = eeg_features
         
        
        self.task_data['eeg_features'] = eeg_features
        self.task_data['eeg_mean_features'] = mean_feat
        self.task_data['eeg_std_features'] = std_feat
        #self.task_data['channels'] = self.channels
        # normalize features
        # eeg_features = (eeg_features - mean_feat.reshape(-1,1))/ std_feat.reshape(-1,1)
        eeg_features = (eeg_features - mean_feat)/ std_feat
        # mean_feat.ravel()
        self.task_data['eeg_features_Z'] = eeg_features

        #print 'eeg_features.shpae'

        try:
            eeg_features = np.reshape(eeg_features,(1,eeg_features.shape[0]))
        except:
            pass
        #eeg_features(eeg_features == np.inf) = 1
        self.decoder_output = self.eeg_decoder(eeg_features)
        self.probability = self.eeg_decoder.decoder.predict_proba(eeg_features)
    
        # print "dec probability : ", self.probability

        #print self.decoder_output, ' with probability:', probability

        # Command zero velocity if the task is in a non-moving state
        if self.state not in ['trial']:#['wait','rest', 'rest_return','instruct_rest', 'instruct_trial_type', 'preparation', 'instruct_go','instruct_rest_return']: 
            command_vel[:] = 0
            self.state_decoder = 0
            
        else:#if self.state in ['trial', 'trial_return', 'instruct_trial_return']:
            if self.decoder_output == 1 and self.prev_output == 1:
                # we need 5 consecutive outputs of the same type
                self.consec_mov_outputs +=1
                if self.consec_mov_outputs == 5 and self.state_decoder == 0:
                    self.consec_rest_outputs = 0
            elif self.decoder_output == 1 and self.prev_output == 0:
                if self.state_decoder == 1: #if it's moving
                    self.consec_rest_outputs = 0
                else:
                    self.consec_mov_outputs = 1
            elif self.decoder_output == 0 and self.prev_output == 0:
                self.consec_rest_outputs +=1
                if self.consec_rest_outputs == 5 and self.state_decoder == 1:
                    self.consec_mov_outputs = 0
            elif self.decoder_output == 0 and self.prev_output == 1:
                if self.state_decoder == 1: #if it's moving
                    self.consec_rest_outputs = 1
                else:
                    self.consec_mov_outputs = 0
    
            if self.consec_mov_outputs >= 5:

                self.state_decoder = 1
                current_pos = self.plant_pos[:].ravel()
                current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
                target_state = np.hstack([self.goal_position, np.zeros_like(current_pos), 1]).reshape(-1, 1)
                assist_output = self.assister(current_state, target_state, 1)
                           
                Bu = np.array(assist_output["x_assist"]).ravel()
                #Bu = np.array(assist_output['Bu']).ravel()
                command_vel[:] = Bu[len(current_pos):len(current_pos)*2]
                #print 'command_vel', command_vel
                #set all the velocities to a constant value towards the end point
            elif self.consec_rest_outputs >=5:

                self.state_decoder = 0
                command_vel[:] = 0 #set all the velocities to zero
        
        command_vel_raw[:] = command_vel[:]
        for state in self.vel_states:
            command_vel[state] = self.command_lpfs[state](command_vel[state])
        
        self.prev_output = self.decoder_output
        #self.task_data['eeg_features']   = eeg_features[self.eeg_extractor.feature_type]
        
        self.task_data['decoder_output'] = self.decoder_output
        self.task_data['decoder_output_probability'] = self.probability
        self.task_data['state_decoder'] = self.state_decoder
        self.task_data['consec_mov_outputs'] = self.consec_mov_outputs
        self.task_data['consec_rest_outputs'] = self.consec_rest_outputs
        self.plant.send_vel(command_vel.values) #send velocity command to EXO
        
        self.task_data['command_vel']  = command_vel.values
        self.task_data['command_vel_raw']  = command_vel_raw.values
        
        # print "state decoder : ", self.state_decoder

    def _cycle(self):
        '''Runs self.fps times per second.'''

        # get latest position/velocity information before calling move_plant()
        self.plant_pos_raw[:] = self.plant.get_pos_raw()
        self.plant_pos[:] = self.plant.get_pos() 
        
        self.plant_vel_raw[:] = self.plant.get_vel_raw()
        self.plant_vel[:] = self.plant.get_vel()

        #if self.state in ['trial','go_to_start']:
        # velocity control
        self.move_plant()

        # position control
        # self.move_plant_pos_control()

        self.update_plant_display()
        self.update_decoder_ouput()

        # print self.subtrial_idx
        if not self.state in ['trial']:
            #self.task_data['audio_feedback_start'] = 0
            self.task_data['goal_pos'] = np.ones(len(self.pos_states))*np.nan            
            self.task_data['goal_idx'] = np.nan
            
        else:
            self.task_data['goal_pos'] = self.goal_position
            self.task_data['goal_idx'] = self.goal_idx
            

        self.task_data['plant_type'] = self.plant_type
        self.task_data['eeg_decoder_coef']  = self.eeg_decoder.decoder.coef_
        self.task_data['eeg_decoder_intercept']  = self.eeg_decoder.decoder.intercept_
        self.task_data['eeg_decoder_means']  = self.eeg_decoder.decoder.means_
        self.task_data['reached_goal_position'] = self.reached_goal_position  
        self.task_data['reached_subtarget'] = self.reached_subtarget          
            
        self.task_data['reached_timeout'] = self.reached_timeout
        self.task_data['simult_reach_and_timeout'] = self.simult_reach_and_timeout
        
        self.task_data['plant_pos']  = self.plant_pos_raw.values
        self.task_data['plant_pos_filt']  = self.plant_pos.values 
        self.task_data['plant_vel']  = self.plant_vel_raw.values
        self.task_data['plant_vel_filt']  = self.plant_vel.values

        self.task_data['trial_type'] = self.trial_type
        self.task_data['speed'] = self.speed
        self.task_data['ts']         = time.time()
        self.task_data['target_margin'] = self.target_margin
        #self.task_data['decoder']    = self.eeg_decoder.decoder

        
        super(EEGCyclicMovementDecodingNew, self)._cycle()


    def init_show_decoder_output(self):

        self.decoder_background_line  = Line(np.array([80, 0]), 100, 3, 1, COLORS['blue'])
        self.add_model(self.decoder_background_line)

        self.decoder_move_perc_line  = Line(np.array([80, 0]), 2, 3, 1, COLORS['red'])
        self.add_model(self.decoder_move_perc_line)

        self.decoder_middle_line = Line(np.array([80, 49]), 0.2, 3, 1, COLORS['white'])
        self.add_model(self.decoder_middle_line)



    def update_decoder_ouput(self):

        #backgroun line in white 
        self.decoder_background_line.color = COLORS['blue']
        self.decoder_background_line.start_pos   = np.array([80, 0])
        self.decoder_background_line.angle = 90*deg_to_rad

        #movement output in green
        self.decoder_move_perc_line.length = self.probability[0,1]*np.int(100)
        self.decoder_move_perc_line.start_pos   = np.array([80, 0])
        self.decoder_move_perc_line.angle = 90*deg_to_rad
        self.decoder_move_perc_line.color = COLORS['red']

        self.decoder_middle_line.color = COLORS['white']
        self.decoder_middle_line.start_pos   = np.array([80, 49])
        self.decoder_middle_line.angle = 90*deg_to_rad

    # def show_image(self, image_fname):

    #     window = pygame.display.set_mode(self.window_size)
    #     img = pygame.image.load(os.path.join(self.image_fname))
    #     img = pygame.transform.scale(img, self.window_size)

    #     window.blit(img, (0,0))
    #     pygame.display.flip()

    #### STATE AND TEST FUNCTIONS ####   
    def _start_wait(self):
        print 'wait'
        # determine the random length of time to stay in the rest state
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time
        super(EEGCyclicMovementDecodingNew, self)._start_wait()

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
        # self._play_sound(os.path.join(self.sounds_dir, 'rest.wav')) #nerea
        self._play_sound(self.sounds_dir, ['rest'])
        #initial_mov_buffer_data = self.mov_data_buffer.get_all()
        print 'rest'


    def _start_instruct_trial_type(self):
        #print 'instruct trial type'
        # sound_fname = os.path.join(self.sounds_dir, self.trial_type + '.wav') #nerea
        # self._play_sound(sound_fname)
        # self._play_sound(self.sounds_dir, self.subgoal_names[self.trial_type][0]) 
        self._play_sound(self.sounds_dir, self.subgoal_names[self.trial_type][self.goal_idx]) #nerea ??
        
        mov_features = self.mov_feature_buffer.get_all().T
        rest_features = self.rest_feature_buffer.get_all().T

        # normalization of features
        self.features = np.vstack([mov_features, rest_features])
        mean_features = np.mean(self.features, axis = 0)
        std_features = np.std(self.features, axis = 0)
        self.features = (self.features - mean_features) / std_features
        
        self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])
        print 'retraining decoder'
        self.retrained_decoder.fit(self.features, self.labels.ravel())
        
        # import time
        # t0 = time.time()
        self.eeg_decoder.decoder = copy.copy(self.retrained_decoder)
        # print 'time2retrain', time.time() - t0
        #print 'decoder retrained'
        
        # self.consec_mov_outputs = 0
        # self.consec_rest_outputs = 0

    def _start_instruct_go(self):
        # self._play_sound(os.path.join(self.sounds_dir, 'go.wav'))
        self._play_sound(self.sounds_dir, ['go'])
        self.consec_mov_outputs = 0
        self.consec_rest_outputs = 0
        self.reached_goal_position = False
        self.reached_subtarget = False
        self.reached_timeout = False
        self.simult_reach_and_timeout = False
        #self.state_decoder = 0


    def _start_trial(self):
        print self.trial_type
        #self.plant.set_pos_control() #to set it to position control during the trial state
        #self._set_task_type()
        #self._set_goal_position()
        self.goal_position = self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states]     
        if self.music_feedback:
            self._play_sound(self.sounds_dir_classical, [self.subgoal_names[self.trial_type][self.goal_idx][0]])
    

    def _test_end_trial(self,ts):
        # Test if simultaneous timeout and end_trial issue is solved with this
        # if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
        #     if self.goal_idx >= len(self.targets_matrix[self.trial_type].keys())-1:               
        #         self.reached_goal_position = True
        # if ts > self.timeout_time:
        #     self.reached_timeout = True
        #     print 'timeout'
        # if self.reached_timeout == True and np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
        #     self.reached_goal_position = True
        
        return (self.reached_goal_position or self.reached_timeout)



    def _test_timeout(self, ts):
        if ts > self.timeout_time:
            self.reached_timeout = True
            if self.reached_goal_position == True:
                #self.reached_timeout = False
                self.simult_reach_and_timeout = True
            #print 'reached goal position', self.reached_goal_position
            print 'timeout'
            # if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
            #     self.reached_timeout = False
        return self.reached_timeout

    # def _test_at_starting_config(self, *args, **kwargs):
    #     traj = self.ref_trajectories[self.trial_type]['traj']
    #     diff_to_start = np.abs(self.plant.get_pos() - traj[self.pos_states].ix[0].ravel())
    #     #print diff_to_start

    #     return np.all(diff_to_start < self.rest_rect[:len(self.pos_states)])
    def _end_trial(self):
        
        self.trial_number +=1
        if self.music_feedback:
            pygame.mixer.music.stop()
            self.parallel_sound.stop()
        else:
            pass
  

    def cleanup_hdf(self):
        super(EEGCyclicMovementDecodingNew, self).cleanup_hdf()    

        import tables
        h5file = tables.openFile(self.h5file.name, mode='a')
        h5file.root.task.attrs['eeg_decoder_name'] = self.eeg_decoder_name
        #h5file.root.task.attrs['brainamp_channels'] = self.channel_list_name
        #compfilt = tables.Filters(complevel=5, complib="zlib", shuffle=True)
        eeg_extractor_grp = h5file.createGroup(h5file.root, "eeg_extractor_kwargs", "Parameters for feature extraction")
        for key in self.eeg_extractor_kwargs:
            if isinstance(self.eeg_extractor_kwargs[key], dict):
                if key == 'feature_fn_kwargs':
                    for key2 in self.eeg_extractor_kwargs['feature_fn_kwargs']['AR']['freq_bands']:
                        if isinstance(self.eeg_extractor_kwargs['feature_fn_kwargs']['AR']['freq_bands'][key2], np.ndarray):
                            h5file.createArray(eeg_extractor_grp, 'freq_band_' + key2, self.eeg_extractor_kwargs['feature_fn_kwargs']['AR']['freq_bands'][key2])
                        else:
                            h5file.createArray(eeg_extractor_grp, 'freq_band_' + key2, np.array([self.eeg_extractor_kwargs['feature_fn_kwargs']['AR']['freq_bands'][key2]]))
                else:
                    for key2 in self.eeg_extractor_kwargs[key]:
                        if isinstance(self.eeg_extractor_kwargs[key][key2], np.ndarray):
                            h5file.createArray(eeg_extractor_grp, key + '_' + key2, self.eeg_extractor_kwargs[key][key2])
                        else:
                            h5file.createArray(eeg_extractor_grp, key + '_' + key2, np.array([self.eeg_extractor_kwargs[key][key2]]))

            else:
                if isinstance(self.eeg_extractor_kwargs[key], np.ndarray):
                    h5file.createArray(eeg_extractor_grp, key, self.eeg_extractor_kwargs[key])
                else:
                    h5file.createArray(eeg_extractor_grp, key, np.array([self.eeg_extractor_kwargs[key]]))
                

        h5file.close()

    def cleanup(self, database, saveid, **kwargs):
        #Old way of buffering rest and mov data
        # self.mov_data = self.mov_data_buffer.get_all()
        # self.rest_data = self.rest_data_buffer.get_all()
        #rest_features, mov_features = self.eeg_extractor.extract_features_2retrain(self.rest_data, self.mov_data)
        #self.features = np.vstack([mov_features, rest_features])
        #self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])

        # New method of buffering rest and mov data to retrain decoder
        mov_features = self.mov_feature_buffer.get_all().T
        rest_features = self.rest_feature_buffer.get_all().T
        # normalization of features
        self.features = np.vstack([mov_features, rest_features])
        mean_features = np.mean(self.features, axis = 0)
        std_features = np.std(self.features, axis = 0)
        self.features = (self.features - mean_features) / std_features

        
        self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])
        self.retrained_decoder.fit(self.features, self.labels.ravel())


        self.eeg_decoder.rest_feature_buffer = self.rest_feature_buffer
        self.eeg_decoder.mov_feature_buffer = self.mov_feature_buffer
        self.eeg_decoder.decoder = copy.copy(self.retrained_decoder)

        #Values just to make it compatible with the task interface (they are nonsense)
        self.eeg_decoder.units  = self.eeg_decoder.channels_2train
        # self.eeg_decoder.binlen = # the decoder is updated after the end of each return trial
        # self.eeg_decoder.tslice = 

        #save eeg_decder object into a new pkl file. 
        storage_dir = '/storage/decoders'
        eeg_decoder_name = self.eeg_decoder.decoder_name 
        # n = decoder_name[-1]
        # n = int(n)        
        index = eeg_decoder_name.rfind('_') + 1
        #new_decoder_name = decoder_name[0:index] + str(n + 1)
        new_decoder_name = eeg_decoder_name[0:index] + str(saveid) 
        self.eeg_decoder.decoder_name = new_decoder_name
        new_pkl_name = new_decoder_name + '.pkl' 
        pickle.dump(self.eeg_decoder, open(os.path.join(storage_dir, new_pkl_name), 'wb'))

        super(EEGCyclicMovementDecodingNew,self).cleanup(database, saveid, **kwargs)
        # Create a new database record for the decoder object if it doesn't already exist
        dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
        dfs = models.Decoder.objects.filter(name=new_decoder_name)
        if len(dfs) == 0:
            df = models.Decoder()
            df.path = new_pkl_name
            df.name = new_decoder_name
            df.entry = models.TaskEntry.objects.using(dbname).get(id=saveid) 
            df.save()
        elif len(dfs) == 1:
            pass # no new data base record needed
        elif len(dfs) > 1:
            print "More than one decoder with the same name! fix manually!"

class EEGMovementDecodingNew_testing(NonInvasiveBase):
    # Unlike the EEGMovementDecoding task, it keeps going towards the same target until it reaches the target position
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
            'timeout': 'instruct_trial_type',
            'end_trial' : 'instruct_trial_type',
            'end_alltrials' : 'wait',
            'stop':      None},    
        }
    

    
    state = 'wait'  # initial state

     # settable parameters on web interface    
    eeg_decoder          = traits.InstanceFromDB(LinearEEGDecoder, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='eeg_decoder'))
    rest_interval        = traits.Tuple((4., 5.), desc='Min and max time to remain in the rest state.')
    preparation_time     = traits.Float(2, desc='time to remain in the preparation state.')
    timeout_time         = traits.Float(10, desc='Maximum time given to the patient to accomplish the task before considering it like incomplete and re-starting it from the current position') 
    give_feedback        = traits.Int((0,1), desc=' 0 if we do not give feedback, 1 if we give feedback about whether the patient reached the goal position')
    music_feedback = traits.Int((1), desc=' 0 if we do not want to include music, 1 if we want different classical music pieces with increasing intensity to be played')
    targets_matrix       = traits.DataFile(object, desc='goal positions for each of the trial types', bmi3d_query_kwargs=dict(system__name='misc'))
    window_size          = traits.Tuple((1000, 560), desc='Size of window to display the plant position/angle')
    channel_list_name    = traits.OptionsList(*channel_list_options, bmi3d_input_options=channel_list_options)
    speed                = traits.OptionsList(*speed_options, bmi3d_input_options= speed_options)
    #artifact_rejection   = traits.Int(1, desc=' 0 if artifacts are not rejected online, 1 if the artifact rejection is applied in real-time too')
    #session_length = traits.Float(20, desc='overall time that the block will last') #It shows up by default in the interface
    #neighbour_channels = ???
    debug = False
    DoF_control = traits.OptionsList(*DoF_control_options, bmi3d_input_options=DoF_control_options, desc='DoFs to be taken into account for condition fulfilment')
    DoF_target = traits.OptionsList(*DoF_target_options, bmi3d_input_options=DoF_target_options, desc='DoFs to be moved/controlled, the rest are stopped.')


    # Set the "is_bmi_seed" flag so that the server knows that this is a task which can be used to create a BMI decoder
    #is_bmi_seed = False
    # def _play_sound(self, fname):
    #     pygame.mixer.music.load(fname)
    #     pygame.mixer.music.play()
    def _play_sound(self, fpath,fname):

        for filename in fname:
            # print 'filename ', filename
            if '_' in filename:
                filename = filename[:filename.find('_')]
            sound_fname = os.path.join(fpath, filename + '.wav')
            pygame.mixer.music.load(sound_fname)
            pygame.mixer.music.play()
            # print 'sound_fname ' , sound_fname
            # while pygame.mixer.music.get_busy():
            #     time.sleep(0)
            # else:
            #     pygame.mixer.music.load(sound_fname)
            #     pygame.time.Clock().tick(1)
            #     # print 'clock'
            #     pygame.mixer.music.play()


    def __init__(self, *args, **kwargs):
        super(EEGMovementDecodingNew_testing, self).__init__(*args, **kwargs)

        self.add_dtype('command_vel',  'f8',    (len(self.vel_states),))
        self.add_dtype('command_vel_raw',  'f8',    (len(self.vel_states),))
        self.add_dtype('command_vel_final',  'f8',    (len(self.vel_states),))
        self.add_dtype('trial_type',   np.str_, 40)
        #self.add_dtype('plant_type',   np.str_, 40)
        
        # self.add_dtype('ts',           'f8',    (1,)) # it is already saved in IsMoreBase class (basic class)
        self.add_dtype('goal_pos',      'f8',    (len(self.pos_states),))
        #self.add_dtype('difference_position','f8', (len(self.pos_states),))
        self.add_dtype('reached_goal_position',bool, (1,))
        self.add_dtype('reached_subtarget',bool, (1,))
        self.add_dtype('reached_timeout',bool, (1,))
        #self.add_dtype('audio_feedback_start',      int,    (1,))
        self.add_dtype('goal_idx', float, (1,))
        self.add_dtype('speed',   np.str_, 20)


        self.parallel_sound = pygame.mixer.Sound('')
      
        # if EEG decoder file was specified, load EEG decoder and create feature extractor 
        
        #if len(self.decoder_file) > 3:
        #    self.eeg_decoder = pickle.load(open(self.decoder_file, 'rb'))

        # create EEG extractor object (its 'source' will be set later in the init method)
      
        eeg_extractor_cls    = self.eeg_decoder.extractor_cls
        self.eeg_decoder_name = self.eeg_decoder.decoder_name
        self.eeg_extractor_kwargs = self.eeg_decoder.extractor_kwargs
        self.artifact_rejection = self.eeg_extractor_kwargs['artifact_rejection']
        # Check if chosen decoder is trained with artifact rejection or not. If artifact_rejection = 1 and decoder not designed for that, print an error!

        self.TH_lowF = self.eeg_decoder.TH_lowF 
        self.TH_highF = self.eeg_decoder.TH_highF 
        self.eog_coeffs = self.eeg_decoder.eog_coeffs 


        self.rest_feature_buffer = self.eeg_decoder.rest_feature_buffer
        #self.trial_hand_side = self.eeg_extractor_kwargs['trial_hand_side']
        self.mov_feature_buffer = self.eeg_decoder.mov_feature_buffer
        try:
            self.channels = self.eeg_extractor_kwargs['eeg_channels']
        except:
            self.channels = self.eeg_extractor_kwargs['channels']
        #self.channels = self.eeg_extractor_kwargs['eeg_channels']
        self.brainamp_channels = getattr(brainamp_channel_lists, self.channel_list_name)  

        #self.brainamp_channels = self.eeg_extractor_kwargs['brainamp_channels']
        self.neighbour_channels = self.eeg_decoder.neighbour_channels
        
        self.eeg_playback = False
        self.fs = self.eeg_extractor_kwargs['fs']

        self.retrained_decoder = copy.copy(self.eeg_decoder.decoder)
        
        self.eeg_extractor_kwargs['eog_coeffs'] = self.eog_coeffs 
        self.eeg_extractor_kwargs['TH_lowF'] = self.TH_lowF 
        self.eeg_extractor_kwargs['TH_highF'] = self.TH_highF 

        self.eeg_extractor = eeg_extractor_cls(source=None, **self.eeg_extractor_kwargs)
        self.n_features = self.eeg_extractor.n_features
        #dtype = np.dtype(['name',       np.str, [len(self.channels),20])
        
        #self.add_dtype('eeg_features',    'f8', (self.eeg_extractor.n_features,))
        self.add_dtype('eeg_features',    'f8', (self.n_features,))
        #self.add_dtype('channels',   np.str_, [len(self.channels),20])
        self.add_dtype('eeg_features_mov_buffer',    'f8', (self.n_features,))
        self.add_dtype('eeg_features_rest_buffer',    'f8', (self.n_features,))
        self.add_dtype('eeg_features_Z',    'f8', (self.n_features,))
        self.add_dtype('eeg_mean_features',    'f8', (self.n_features,))
        self.add_dtype('eeg_std_features',    'f8', (self.n_features,))
        self.add_dtype('eeg_decoder_coef',    'f8', (self.n_features,))
        self.add_dtype('eeg_decoder_intercept', 'f8', (self.n_features,))
        self.add_dtype('eeg_decoder_means', 'f8', (2,self.n_features))
        self.add_dtype('decoder_output',    'f8', (1,))
        self.add_dtype('decoder_output_probability',    'f8', (1,2))#2 classes
        self.add_dtype('state_decoder',  int, (1,))
        self.add_dtype('consec_mov_outputs',  int, (1,))
        self.add_dtype('consec_rest_outputs',  int, (1,))
        self.add_dtype('rejected_window',  int, (1,))
        #self.add_dtype('decoder', InstanceFromDB(LinearEEGDecoder))

               
        # for low-pass filtering decoded EEG velocities
        # self.eeg_vel_buffer = RingBuffer(
        #     item_len=len(self.vel_states),
        #     capacity=10,
        # )

        self.plant.enable() 
        [self.DoF_target_idx_init, self.DoF_target_idx_end, self.DoF_not_control_idx_init, self.DoF_not_control_idx_end] = check_plant_and_DoFs(self.plant_type, self.DoF_control, self.DoF_target)
        self.subgoal_names = self.targets_matrix['subgoal_names']


        #initialize values for the state of the decoder
        self.consec_mov_outputs = 0
        self.consec_rest_outputs = 0
        self.prev_output = 0
        self.state_decoder = 0

        # if self.plant_type == 'ArmAssist':
        #     self.target_margin = np.array([2, 2, np.deg2rad(10)])
        # elif self.plant_type == 'ReHand':
        #     self.target_margin = np.array([np.deg2rad(2),  np.deg2rad(2), np.deg2rad(2), np.deg2rad(2)])
        # elif self.plant_type == 'IsMore':
        #     self.target_margin = np.array([2, 2, np.deg2rad(10), np.deg2rad(2),  np.deg2rad(2), np.deg2rad(2), np.deg2rad(2)])
          
        #self.target_margin = pd.Series(np.array([3, 3, np.deg2rad(15), np.deg2rad(5),  np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)]), ismore_pos_states)
        # target margin used for DK calibration sessions
        self.target_margin = pd.Series(np.array([2, 2, np.deg2rad(8), np.deg2rad(3),  np.deg2rad(3), np.deg2rad(3), np.deg2rad(3)]), ismore_pos_states)
   

        self.target_margin = self.target_margin[self.pos_states]
        self.goal_idx = 0
        self.trial_number = 0
        self.trial_type = None


        self.add_dtype('target_margin',      'f8',    (len(self.target_margin),))
        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)
        self.sounds_dir_classical = os.path.expandvars('$HOME/code/ismore/sounds/classical')


        self.image_dir_general = os.path.expandvars('$HOME/code/ismore/images')
        self.image_dir = os.path.join(self.image_dir_general, self.language)
        

        self.reached_goal_position = False #If the goal_position is reached then give feedback to patient and start the movement back towards the rest_position
        self.reached_subtarget = False #If the task has more than one target position, this shows when the targets before the last target are reached
        self.reached_timeout = False


        # 2nd order butterworth filter for command_vel
        fs_synch = self.fps #Frequency at which the task is running
        nyq   = 0.5 * fs_synch
        cuttoff_freq  = 1.5 / nyq
        bpf_kin_coeffs = butter(2, cuttoff_freq, btype='low')

        self.command_lpfs = dict()
        for state in self.vel_states:
            self.command_lpfs[state] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1]) # low-pass filter to smooth out command velocities


        self.init_show_decoder_output()
        

    def init(self):
        kwargs = {
            'call_rate': self.fps, #kwargs used by the assister
            'xy_cutoff': 2.,#What is this for? Radius of margin?
        }

        from riglib import source
        from ismore.brainamp import rda
       
        #self.brainamp_source = source.MultiChanDataSource(rda.EMGData, name='brainamp', channels=self.brainamp_channels, brainamp_channels=self.brainamp_channels, send_data_to_sink_manager=True)
        self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT[self.plant_type](speed = self.speed,**kwargs)
        self.eeg_extractor.source = self.brainamp_source
        super(EEGMovementDecodingNew_testing, self).init()

    # def _set_goal_position(self):
    #     self.goal_position = self.targets_matrix[self.trial_type]

        

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

        # if self.give_feedback == 1:
        # #     fb_time = self.feedback_time[self.trial_type][self.subtrial_idx]
        #     self.task_data['audio_feedback_start'] = 0

        # # print "goal position: ", self.goal_position
        # # print "plant position: ", self.plant.get_pos()
        # # print "abs difference: ", np.abs(self.pos_diff(self.goal_position,self.plant.get_pos()))
        # # print "target margin: ", self.target_margin

        # if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin):
        #     self.reached_goal_position = True
        #     if self.give_feedback:
        #         self.task_data['audio_feedback_start'] = 1
        #         self._play_sound(os.path.join(self.sounds_general_dir, 'beep.wav'))
        #         self.task_data['audio_feedback_start'] = 0
        # print 'distance to target', self.pos_diff(self.goal_position[self.pos_states],self.plant.get_pos())


        #self.task_data['audio_feedback_start'] = 0

        if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
            #self.task_data['audio_feedback_start'] = 1
            
            if self.give_feedback:
                # self._play_sound(os.path.join(self.sounds_general_dir, 'beep.wav'))
                self._play_sound(self.sounds_general_dir, ['beep']) #nerea

            if self.goal_idx < len(self.targets_matrix[self.trial_type].keys())-1:
                # pygame.mixer.music.stop() #nerea
                
                # self.parallel_sound.stop()
                #self.goal_idx +=1
                # print 'heading to next subtarget'
                
                #self.goal_position = self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states]
               
               
                # pygame.mixer.music.queue(os.path.join(self.sounds_dir_classical, self.subgoal_names[self.trial_type][self.goal_idx][0]+'.wav'))
                # self.parallel_sound = pygame.mixer.Sound(os.path.join(self.sounds_dir_classical, self.subgoal_names[self.trial_type][self.goal_idx][0]+'.wav'))
                # self.parallel_sound.play()
                # self._play_sound(self.sounds_dir_classical, [self.subgoal_names[self.trial_type][self.goal_idx][0]])  #nerea

                self.reached_subtarget = True
                # #print self.goal_position
            else:
                print 'all subtargets reached'
                self.reached_goal_position = True
            
        #Show output decoder 
        # if self.state_decoder == 1:
        #     self.image_fname = os.path.join(self.image_dir, 'mov.bmp')
        #     self.show_image(self.image_fname) 
        # else:
        #     self.image_fname = os.path.join(self.image_dir, 'rest.bmp')
        #     self.show_image(self.image_fname) 

    
      
    def move_plant(self):
        '''Docstring.'''

        command_vel  = pd.Series(0.0, self.vel_states)
        command_vel_raw  = pd.Series(0.0, self.vel_states)
        command_vel_final  = pd.Series(0.0, self.vel_states)

        # run EEG feature extractor and decoder
        #self.eeg_extractor.source = self.brainamp_source
        if self.artifact_rejection == 1:
            eeg_features, rejected_window = self.eeg_extractor()
            self.task_data['rejected_window'] = rejected_window
        else:
            eeg_features = self.eeg_extractor() # eeg_features is of type 'dict'
            

        feat_mov = self.mov_feature_buffer.get_all()
        feat_rest = self.rest_feature_buffer.get_all()
        mean_feat = np.mean(np.hstack([feat_mov, feat_rest]), axis = 1)
        std_feat = np.std(np.hstack([feat_mov, feat_rest]), axis = 1)
       
        if self.trial_number > 0:
            if self.state in ['trial']:
                if self.artifact_rejection == 1 & rejected_window == 0:
                    self.mov_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
                elif self.artifact_rejection == 0:
                    self.mov_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
                self.task_data['eeg_features_mov_buffer'] = eeg_features
                #self.mov_feature_buffer.add(eeg_features)
            elif self.state in ['rest']:
                if self.artifact_rejection == 1 & rejected_window == 0:
                    self.rest_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
                elif self.artifact_rejection == 0:
                    self.rest_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
                    
                self.task_data['eeg_features_rest_buffer'] = eeg_features
         
        
        self.task_data['eeg_features'] = eeg_features
        self.task_data['eeg_mean_features'] = mean_feat
        self.task_data['eeg_std_features'] = std_feat
        #self.task_data['channels'] = self.channels
        # normalize features
        # eeg_features = (eeg_features - mean_feat.reshape(-1,1))/ std_feat.reshape(-1,1)
        eeg_features = (eeg_features - mean_feat)/ std_feat
        # mean_feat.ravel()
        self.task_data['eeg_features_Z'] = eeg_features

        #print 'eeg_features.shpae'

        try:
            eeg_features = np.reshape(eeg_features,(1,eeg_features.shape[0]))
        except:
            pass
        #eeg_features(eeg_features == np.inf) = 1
        self.decoder_output = self.eeg_decoder(eeg_features)
        self.probability = self.eeg_decoder.decoder.predict_proba(eeg_features)
        

        #print self.decoder_output, ' with probability:', probability

        # Command zero velocity if the task is in a non-moving state
        if self.state not in ['trial']:#['wait','rest', 'rest_return','instruct_rest', 'instruct_trial_type', 'preparation', 'instruct_go','instruct_rest_return']: 
            command_vel[:] = 0
            self.state_decoder = 0
            
        else:#if self.state in ['trial', 'trial_return', 'instruct_trial_return']:
            if self.decoder_output == 1 and self.prev_output == 1:
                # we need 5 consecutive outputs of the same type
                self.consec_mov_outputs +=1
                if self.consec_mov_outputs == 5 and self.state_decoder == 0:
                    self.consec_rest_outputs = 0
            elif self.decoder_output == 1 and self.prev_output == 0:
                if self.state_decoder == 1: #if it's moving
                    self.consec_rest_outputs = 0
                else:
                    self.consec_mov_outputs = 1
            elif self.decoder_output == 0 and self.prev_output == 0:
                self.consec_rest_outputs +=1
                if self.consec_rest_outputs == 5 and self.state_decoder == 1:
                    self.consec_mov_outputs = 0
            elif self.decoder_output == 0 and self.prev_output == 1:
                if self.state_decoder == 1: #if it's moving
                    self.consec_rest_outputs = 1
                else:
                    self.consec_mov_outputs = 0
    
            if self.consec_mov_outputs >= 5:

                self.state_decoder = 1
                current_pos = self.plant_pos[:].ravel()
                current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
                target_state = np.hstack([self.goal_position, np.zeros_like(current_pos), 1]).reshape(-1, 1)
                assist_output = self.assister(current_state, target_state, 1)
                           
                Bu = np.array(assist_output["x_assist"]).ravel()
                #Bu = np.array(assist_output['Bu']).ravel()
                command_vel[:] = Bu[len(current_pos):len(current_pos)*2]
                #print 'command_vel', command_vel
                #set all the velocities to a constant value towards the end point
            elif self.consec_rest_outputs >=5:

                self.state_decoder = 0
                command_vel[:] = 0 #set all the velocities to zero
        
        command_vel_raw[:] = command_vel[:]
        for state in self.vel_states:
            command_vel[state] = self.command_lpfs[state](command_vel[state])
        
        self.prev_output = self.decoder_output
        #self.task_data['eeg_features']   = eeg_features[self.eeg_extractor.feature_type]
        
        self.task_data['decoder_output'] = self.decoder_output
        self.task_data['decoder_output_probability'] = self.probability
        self.task_data['state_decoder'] = self.state_decoder
        self.task_data['consec_mov_outputs'] = self.consec_mov_outputs
        self.task_data['consec_rest_outputs'] = self.consec_rest_outputs
        self.task_data['command_vel']  = command_vel.values
        self.task_data['command_vel_raw']  = command_vel_raw.values

        # set the velocities of the DoFs that should not be controlled to zero.
        if np.isnan(self.DoF_not_control_idx_init) == False and np.isnan(self.DoF_not_control_idx_end) == False:
            command_vel[self.DoF_not_control_idx_init:self.DoF_not_control_idx_end] = 0


        self.plant.send_vel(command_vel.values) #send velocity command to EXO
        self.task_data['command_vel_final']  = command_vel.values
        
        
    def _cycle(self):
        '''Runs self.fps times per second.'''

        # get latest position/velocity information before calling move_plant()
        self.plant_pos_raw[:] = self.plant.get_pos_raw()
        self.plant_pos[:] = self.plant.get_pos() 
        
        self.plant_vel_raw[:] = self.plant.get_vel_raw()
        self.plant_vel[:] = self.plant.get_vel()

        #if self.state in ['trial','go_to_start']:
        # velocity control
        self.move_plant()

        # position control
        # self.move_plant_pos_control()

        self.update_plant_display()
        self.update_decoder_ouput()

        # print self.subtrial_idx
        if not self.state in ['trial']:
            #self.task_data['audio_feedback_start'] = 0
            self.task_data['goal_pos'] = np.ones(len(self.pos_states))*np.nan            
            self.task_data['goal_idx'] = np.nan
            
        else:
            self.task_data['goal_pos'] = self.goal_position
            self.task_data['goal_idx'] = self.goal_idx
            

        self.task_data['plant_type'] = self.plant_type
        self.task_data['eeg_decoder_coef']  = self.eeg_decoder.decoder.coef_
        self.task_data['eeg_decoder_intercept']  = self.eeg_decoder.decoder.intercept_
        self.task_data['eeg_decoder_means']  = self.eeg_decoder.decoder.means_
        self.task_data['reached_goal_position'] = self.reached_goal_position  
        self.task_data['reached_subtarget'] = self.reached_subtarget          
            
        self.task_data['reached_timeout'] = self.reached_timeout
        
        self.task_data['plant_pos']  = self.plant_pos_raw.values
        self.task_data['plant_pos_filt']  = self.plant_pos.values 
        self.task_data['plant_vel']  = self.plant_vel_raw.values
        self.task_data['plant_vel_filt']  = self.plant_vel.values

        self.task_data['trial_type'] = self.trial_type
        self.task_data['speed'] = self.speed
        self.task_data['ts']         = time.time()
        self.task_data['target_margin'] = self.target_margin
        #self.task_data['decoder']    = self.eeg_decoder.decoder

      
        super(EEGMovementDecodingNew_testing, self)._cycle()


    def init_show_decoder_output(self):

        self.decoder_background_line  = Line(np.array([80, 0]), 100, 3, 1, COLORS['blue'])
        self.add_model(self.decoder_background_line)

        self.decoder_move_perc_line  = Line(np.array([80, 0]), 2, 3, 1, COLORS['red'])
        self.add_model(self.decoder_move_perc_line)

        self.decoder_middle_line = Line(np.array([80, 49]), 0.2, 3, 1, COLORS['white'])
        self.add_model(self.decoder_middle_line)



    def update_decoder_ouput(self):

        #backgroun line in white 
        self.decoder_background_line.color = COLORS['blue']
        self.decoder_background_line.start_pos   = np.array([80, 0])
        self.decoder_background_line.angle = 90*deg_to_rad

        #movement output in green
        self.decoder_move_perc_line.length = self.probability[0,1]*np.int(100)
        self.decoder_move_perc_line.start_pos   = np.array([80, 0])
        self.decoder_move_perc_line.angle = 90*deg_to_rad
        self.decoder_move_perc_line.color = COLORS['red']

        self.decoder_middle_line.color = COLORS['white']
        self.decoder_middle_line.start_pos   = np.array([80, 49])
        self.decoder_middle_line.angle = 90*deg_to_rad

    # def show_image(self, image_fname):

    #     window = pygame.display.set_mode(self.window_size)
    #     img = pygame.image.load(os.path.join(self.image_fname))
    #     img = pygame.transform.scale(img, self.window_size)

    #     window.blit(img, (0,0))
    #     pygame.display.flip()

    #### STATE AND TEST FUNCTIONS ####   
    def _start_wait(self):
        print 'wait'
        # determine the random length of time to stay in the rest state
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time
        self.goal_idx = 0
        print "trial type : ", self.trial_type
        super(EEGMovementDecodingNew_testing, self)._start_wait()


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
        # self._play_sound(os.path.join(self.sounds_dir, 'rest.wav')) #nerea
        self._play_sound(self.sounds_dir, ['rest'])
        #initial_mov_buffer_data = self.mov_data_buffer.get_all()
        print 'rest'

    def _start_instruct_trial_type(self):
        #print 'instruct trial type'
        # sound_fname = os.path.join(self.sounds_dir, self.trial_type + '.wav') #nerea
        # self._play_sound(sound_fname)
        # self._play_sound(self.sounds_dir, self.subgoal_names[self.trial_type][0]) 
        print "self.subgoal_names[self.trial_type][self.goal_idx] ", self.subgoal_names[self.trial_type][self.goal_idx]
        self._play_sound(self.sounds_dir, self.subgoal_names[self.trial_type][self.goal_idx]) #nerea ??
        
                
        mov_features = self.mov_feature_buffer.get_all().T
        rest_features = self.rest_feature_buffer.get_all().T

        # normalization of features
        self.features = np.vstack([mov_features, rest_features])
        mean_features = np.mean(self.features, axis = 0)
        std_features = np.std(self.features, axis = 0)
        self.features = (self.features - mean_features) / std_features
        
        self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])
        print 'retraining decoder'
        # import time
        # t0 = time.time()
        self.retrained_decoder.fit(self.features, self.labels.ravel())
        
        # import time
        # t0 = time.time()
        self.eeg_decoder.decoder = copy.copy(self.retrained_decoder)
        print 'decoder trained'
        # print 'time2retrain', time.time() - t0
        #print 'decoder retrained'
        
        # self.consec_mov_outputs = 0
        # self.consec_rest_outputs = 0

    # def _end_instruct_trial_type(self):
    #     self.reached_goal_position = False

    def _start_instruct_go(self):
        # self._play_sound(os.path.join(self.sounds_dir, 'go.wav'))
        self._play_sound(self.sounds_dir, ['go'])
        self.consec_mov_outputs = 0
        self.consec_rest_outputs = 0
        self.reached_goal_position = False
        self.reached_subtarget = False
        self.reached_timeout = False
        #self.state_decoder = 0

   

    def _start_trial(self):
        #print self.trial_type
        print "subtrial : ", self.subgoal_names[self.trial_type][self.goal_idx]
        #self.plant.set_pos_control() #to set it to position control during the trial state
        #self._set_task_type()
        #self._set_goal_position()
        self.goal_position = self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states]     
        if self.music_feedback:
            self._play_sound(self.sounds_dir_classical, [self.subgoal_names[self.trial_type][self.goal_idx][0]])
    
    def _test_end_alltrials(self,ts):
        return self.reached_goal_position
    
    def _test_end_trial(self,ts):
        return self.reached_subtarget 
        #return (self.reached_goal_position or self.reached_timeout)

    def _end_alltrials(self):
        print 'all trials reached'
        self.task_data['reached_goal_position']  = self.reached_goal_position

    def _end_trial(self):    
        self.reached_subtarget = False
        pygame.mixer.music.stop()
        self.parallel_sound.stop()
        if self.reached_timeout == False:
            self.goal_idx +=1   
        self.trial_number +=1
        self.task_data['reached_subtarget']  = self.reached_subtarget
        print 'trial end - heading to next subtarget'
        #

    def _test_timeout(self, ts):
        if ts > self.timeout_time:
            self.reached_timeout = True
            
            #print 'reached goal position', self.reached_goal_position
            print 'timeout'
            # if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
            #     self.reached_timeout = False
        return self.reached_timeout



    def cleanup_hdf(self):
        super(EEGMovementDecodingNew_testing, self).cleanup_hdf()    

        import tables
        h5file = tables.openFile(self.h5file.name, mode='a') 
        h5file.root.task.attrs['eeg_decoder_name'] = self.eeg_decoder_name
        #h5file.root.task.attrs['brainamp_channels'] = self.channel_list_name
        #compfilt = tables.Filters(complevel=5, complib="zlib", shuffle=True)
        eeg_extractor_grp = h5file.createGroup(h5file.root, "eeg_extractor_kwargs", "Parameters for feature extraction")
        for key in self.eeg_extractor_kwargs:
            if isinstance(self.eeg_extractor_kwargs[key], dict):
                if key == 'feature_fn_kwargs':
                    for key2 in self.eeg_extractor_kwargs['feature_fn_kwargs']['AR']['freq_bands']:
                        if isinstance(self.eeg_extractor_kwargs['feature_fn_kwargs']['AR']['freq_bands'][key2], np.ndarray):
                            h5file.createArray(eeg_extractor_grp, 'freq_band_' + key2, self.eeg_extractor_kwargs['feature_fn_kwargs']['AR']['freq_bands'][key2])
                        else:
                            h5file.createArray(eeg_extractor_grp, 'freq_band_' + key2, np.array([self.eeg_extractor_kwargs['feature_fn_kwargs']['AR']['freq_bands'][key2]]))
                else:
                    for key2 in self.eeg_extractor_kwargs[key]:
                        if isinstance(self.eeg_extractor_kwargs[key][key2], np.ndarray):
                            h5file.createArray(eeg_extractor_grp, key + '_' + key2, self.eeg_extractor_kwargs[key][key2])
                        else:
                            h5file.createArray(eeg_extractor_grp, key + '_' + key2, np.array([self.eeg_extractor_kwargs[key][key2]]))

            else:
                if isinstance(self.eeg_extractor_kwargs[key], np.ndarray):
                    h5file.createArray(eeg_extractor_grp, key, self.eeg_extractor_kwargs[key])
                else:
                    h5file.createArray(eeg_extractor_grp, key, np.array([self.eeg_extractor_kwargs[key]]))
                

        h5file.close()

    def cleanup(self, database, saveid, **kwargs):
        #Old way of buffering rest and mov data
        # self.mov_data = self.mov_data_buffer.get_all()
        # self.rest_data = self.rest_data_buffer.get_all()
        #rest_features, mov_features = self.eeg_extractor.extract_features_2retrain(self.rest_data, self.mov_data)
        #self.features = np.vstack([mov_features, rest_features])
        #self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])

        # New method of buffering rest and mov data to retrain decoder
        mov_features = self.mov_feature_buffer.get_all().T
        rest_features = self.rest_feature_buffer.get_all().T
        # normalization of features
        self.features = np.vstack([mov_features, rest_features])
        mean_features = np.mean(self.features, axis = 0)
        std_features = np.std(self.features, axis = 0)
        self.features = (self.features - mean_features) / std_features

        
        self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])
        self.retrained_decoder.fit(self.features, self.labels.ravel())


        self.eeg_decoder.rest_feature_buffer = self.rest_feature_buffer
        self.eeg_decoder.mov_feature_buffer = self.mov_feature_buffer
        self.eeg_decoder.decoder = copy.copy(self.retrained_decoder)

        #Values just to make it compatible with the task interface (they are nonsense)
        self.eeg_decoder.units  = self.eeg_decoder.channels_2train
        # self.eeg_decoder.binlen = # the decoder is updated after the end of each return trial
        # self.eeg_decoder.tslice = 

        #save eeg_decder object into a new pkl file. 
        storage_dir = '/storage/decoders'
        eeg_decoder_name = self.eeg_decoder.decoder_name 
        # n = decoder_name[-1]
        # n = int(n)        
        index = eeg_decoder_name.rfind('_') + 1
        #new_decoder_name = decoder_name[0:index] + str(n + 1)
        new_decoder_name = eeg_decoder_name[0:index] + str(saveid) 
        self.eeg_decoder.decoder_name = new_decoder_name
        new_pkl_name = new_decoder_name + '.pkl' 
        pickle.dump(self.eeg_decoder, open(os.path.join(storage_dir, new_pkl_name), 'wb'))

        super(EEGMovementDecodingNew_testing,self).cleanup(database, saveid, **kwargs)
        # Create a new database record for the decoder object if it doesn't already exist
        dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
        dfs = models.Decoder.objects.filter(name=new_decoder_name)
        if len(dfs) == 0:
            df = models.Decoder()
            df.path = new_pkl_name
            df.name = new_decoder_name
            df.entry = models.TaskEntry.objects.using(dbname).get(id=saveid) 
            df.save()
        elif len(dfs) == 1:
            pass # no new data base record needed
        elif len(dfs) > 1:
            print "More than one decoder with the same name! fix manually!"

class SimEEGMovementDecoding_PK(EEGMovementDecoding):
    def __init__(self, *args, **kwargs):
        super(SimEEGMovementDecoding_PK, self).__init__(*args, **kwargs)
        self.eeg_decoder = kwargs['decoder']
        self.brainamp_channels = kwargs['brainamp_channels']


class SimEEGMovementDecoding(EEGMovementDecoding):
    
    

    # def __init__(self, *args, **kwargs):
    #     super(SimEEGMovementDecoding, self).__init__(*args, **kwargs)

    #     self.rest_data_buffer = self.eeg_decoder.rest_data_buffer
    #     self.mov_data_buffer = self.eeg_decoder.mov_data_buffer
        #self.add_dtype('signal_2test',    'f8', (5,500))
        
    # def init(self):

    #     super(SimEEGMovementDecoding, self).init()
        
    # def _while_rest(self):

    #     self.mov_data = self.mov_data_buffer.get_all()
    #     self.rest_data = self.rest_data_buffer.get_all()
        
        
    #     rest_features, mov_features = self.eeg_extractor.extract_features_2retrain(self.rest_data, self.mov_data)
        
    #     self.features = np.vstack([mov_features, rest_features])
    #     self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])
        
    #     self.retrained_decoder.fit(self.features, self.labels.ravel())
    
    def move_plant(self):
        '''Docstring.'''

        command_vel  = pd.Series(0.0, self.vel_states)

        # run EEG feature extractor and decoder
        #self.eeg_extractor.source = self.brainamp_source

        if self.state in ['wait','rest', 'instruct_rest', 'instruct_trial_type', 'preparation','instruct_go','instruct_rest_return']: 
            #eeg_features, self.signal_2test = self.eeg_extractor.sim_call_rest()
            eeg_features = self.eeg_extractor.sim_call_rest() # eeg_features is of type 'dict'
        elif self.state in ['trial', 'trial_return', 'instruct_trial_return']:
            #eeg_features, self.signal_2test = self.eeg_extractor.sim_call_mov()
            eeg_features = self.eeg_extractor.sim_call_mov()

        if self.state in ['trial','trial_return']:
            self.mov_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
            #self.mov_feature_buffer.add(eeg_features)
        elif self.state == 'rest':
            self.rest_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))

        # print 'eeg_features'
        # print eeg_features
        self.task_data['eeg_features'] = eeg_features
        #print 'eeg_features.shpae'
        eeg_features = np.reshape(eeg_features,(1,eeg_features.shape[0]))
        #print eeg_features.shape
        self.decoder_output = self.eeg_decoder(eeg_features)
        self.probability = self.eeg_decoder.decoder.predict_proba(eeg_features)

        print self.decoder_output, ' with probability:', self.probability
        # Command zero velocity if the task is in a non-moving state
        if self.state in ['wait','rest', 'instruct_rest', 'instruct_trial_type', 'preparation', 'instruct_go']: 
            command_vel[:] = 0
            
        elif self.state in ['trial', 'trial_return', 'instruct_trial_return']:
            if self.decoder_output == 1 and self.prev_output == 1:
                # we need 5 consecutive outputs of the same type
                self.consec_mov_outputs +=1
                if self.consec_mov_outputs == 5 and self.state_decoder == 0:
                    self.consec_rest_outputs = 0
            elif self.decoder_output == 1 and self.prev_output == 0:
                if self.state_decoder == 1: #if it's moving
                    self.consec_rest_outputs = 0
                else:
                    self.consec_mov_outputs = 1
            elif self.decoder_output == 0 and self.prev_output == 0:
                self.consec_rest_outputs +=1
                if self.consec_rest_outputs == 5 and self.state_decoder == 1:
                    self.consec_mov_outputs = 0
            elif self.decoder_output == 0 and self.prev_output == 1:
                if self.state_decoder == 1: #if it's moving
                    self.consec_rest_outputs = 1
                else:
                    self.consec_mov_outputs = 0
    
            if self.consec_mov_outputs >= 5:

                self.state_decoder = 1
                current_pos = self.plant_pos[:].ravel()
                current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
                target_state = np.hstack([self.goal_position, np.zeros_like(current_pos), 1]).reshape(-1, 1)
                assist_output = self.assister(current_state, target_state, 1)
                           
                Bu = np.array(assist_output["x_assist"]).ravel()
                #Bu = np.array(assist_output['Bu']).ravel()
                command_vel[:] = Bu[len(current_pos):len(current_pos)*2]
                #set all the velocities to a constant value towards the end point
            elif self.consec_rest_outputs >=5:

                self.state_decoder = 0
                command_vel[:] = 0 #set all the velocities to zero
        self.prev_output = self.decoder_output
        #self.task_data['eeg_features']   = eeg_features[self.eeg_extractor.feature_type]
        
        self.task_data['decoder_output'] = self.decoder_output
        self.task_data['decoder_output_probability'] = self.probability
        self.task_data['state_decoder'] = self.state_decoder

  
        self.plant.send_vel(command_vel.values) #send velocity command to EXO
        
        self.task_data['command_vel']  = command_vel.values
        #self.task_data['signal_2test'] = self.signal_2test
        
    def _cycle(self):
        '''Runs self.fps times per second.'''

        # get latest position/velocity information before calling move_plant()
        self.plant_pos_raw[:] = self.plant.get_pos_raw()
        self.plant_pos[:] = self.plant.get_pos() 
        
        self.plant_vel_raw[:] = self.plant.get_vel_raw()
        self.plant_vel[:] = self.plant.get_vel()

        #if self.state in ['trial','go_to_start']:
        # velocity control
        self.move_plant()

        # position control
        # self.move_plant_pos_control()

        self.update_plant_display()

        # print self.subtrial_idx
        if not self.state in ['trial','trial_return']:
            #self.task_data['audio_feedback_start'] = 0
            self.task_data['goal_pos'] = np.ones(len(self.pos_states))*np.nan
            self.task_data['reached_goal_position'] = False
            self.task_data['goal_idx'] = np.nan
        else:
            self.task_data['goal_pos'] = self.goal_position
            self.task_data['reached_goal_position'] = self.reached_goal_position
            self.task_data['goal_idx'] = self.goal_idx
        #print 'coefs', self.decoder.decoder.coef_
        self.task_data['eeg_decoder_coef']  = self.eeg_decoder.decoder.coef_
        self.task_data['eeg_decoder_intercept']  = self.eeg_decoder.decoder.intercept_
        self.task_data['eeg_decoder_means']  = self.eeg_decoder.decoder.means_

        
        self.task_data['plant_pos']  = self.plant_pos_raw.values
        self.task_data['plant_pos_filt']  = self.plant_pos.values 
        self.task_data['plant_vel']  = self.plant_vel_raw.values
        self.task_data['plant_vel_filt']  = self.plant_vel.values
        
        self.task_data['trial_type'] = self.trial_type
        self.task_data['ts']         = time.time()
        self.task_data['target_margin'] = self.target_margin
        #self.task_data['decoder']    = self.eeg_decoder.decoder

      
        super(SimEEGMovementDecoding, self)._cycle()

    
class SimEEGMovementDecodingNew(EEGMovementDecodingNew):
    
 
    def move_plant(self):
        '''Docstring.'''

        command_vel  = pd.Series(0.0, self.vel_states)

        # run EEG feature extractor and decoder
        #self.eeg_extractor.source = self.brainamp_source
        if self.state not in ['trial', 'trial_return']:#['wait','rest', 'rest_return','instruct_rest', 'instruct_trial_type', 'preparation','instruct_go']: 
            #eeg_features, self.signal_2test = self.eeg_extractor.sim_call_rest()
            eeg_features = self.eeg_extractor.sim_call_rest() # eeg_features is of type 'dict'
        else:#if self.state in ['trial', 'trial_return']:
            #eeg_features, self.signal_2test = self.eeg_extractor.sim_call_mov()
            eeg_features = self.eeg_extractor.sim_call_mov()

        if self.state in ['trial','trial_return']:
            self.mov_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
            #self.mov_feature_buffer.add(eeg_features)
        elif self.state in ['rest','rest_return']:
            self.rest_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
            #self.rest_feature_buffer.add(eeg_features)
        
        
        
        # print 'eeg_features'
        # print eeg_features
        self.task_data['eeg_features'] = eeg_features
        #print 'eeg_features.shpae'
        eeg_features = np.reshape(eeg_features,(1,eeg_features.shape[0]))
        #print eeg_features.shape
        self.decoder_output = self.eeg_decoder(eeg_features)
        self.probability = self.eeg_decoder.decoder.predict_proba(eeg_features)

        print decoder_output, ' with probability:', probability
        # Command zero velocity if the task is in a non-moving state
        if self.state not in ['trial', 'trial_return']:#['wait','rest', 'instruct_rest', 'instruct_trial_type', 'preparation', 'instruct_go']: 
            command_vel[:] = 0
            self.state_decoder = 0
            
        else:#if self.state in ['trial', 'trial_return']:
            if self.decoder_output == 1 and self.prev_output == 1:
                # we need 5 consecutive outputs of the same type
                self.consec_mov_outputs +=1
                if self.consec_mov_outputs == 5 and self.state_decoder == 0:
                    self.consec_rest_outputs = 0
            elif self.decoder_output == 1 and self.prev_output == 0:
                if self.state_decoder == 1: #if it's moving
                    self.consec_rest_outputs = 0
                else:
                    self.consec_mov_outputs = 1
            elif self.decoder_output == 0 and self.prev_output == 0:
                self.consec_rest_outputs +=1
                if self.consec_rest_outputs == 5 and self.state_decoder == 1:
                    self.consec_mov_outputs = 0
            elif self.decoder_output == 0 and self.prev_output == 1:
                if self.state_decoder == 1: #if it's moving
                    self.consec_rest_outputs = 1
                else:
                    self.consec_mov_outputs = 0
    
            if self.consec_mov_outputs >= 5:

                self.state_decoder = 1
                current_pos = self.plant_pos[:].ravel()
                current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
                target_state = np.hstack([self.goal_position, np.zeros_like(current_pos), 1]).reshape(-1, 1)
                assist_output = self.assister(current_state, target_state, 1)
                           
                Bu = np.array(assist_output["x_assist"]).ravel()
                #Bu = np.array(assist_output['Bu']).ravel()
                command_vel[:] = Bu[len(current_pos):len(current_pos)*2]
                #set all the velocities to a constant value towards the end point
            elif self.consec_rest_outputs >=5:

                self.state_decoder = 0
                command_vel[:] = 0 #set all the velocities to zero
        self.prev_output = self.decoder_output
        #self.task_data['eeg_features']   = eeg_features[self.eeg_extractor.feature_type]
        
        self.task_data['decoder_output'] = self.decoder_output
        self.task_data['decoder_output_probability'] = self.probability
        self.task_data['state_decoder'] = self.state_decoder
  
        self.plant.send_vel(command_vel.values) #send velocity command to EXO
        
        self.task_data['command_vel']  = command_vel.values
        #self.task_data['signal_2test'] = self.signal_2test
        
    def _cycle(self):
        '''Runs self.fps times per second.'''

        # get latest position/velocity information before calling move_plant()
        self.plant_pos_raw[:] = self.plant.get_pos_raw()
        self.plant_pos[:] = self.plant.get_pos() 
        
        self.plant_vel_raw[:] = self.plant.get_vel_raw()
        self.plant_vel[:] = self.plant.get_vel()

        #if self.state in ['trial','go_to_start']:
        # velocity control
        self.move_plant()

        # position control
        # self.move_plant_pos_control()

        self.update_plant_display()



        # print self.subtrial_idx
        if not self.state in ['trial','trial_return']:
            #self.task_data['audio_feedback_start'] = 0
            self.task_data['goal_pos'] = np.ones(len(self.pos_states))*np.nan
            self.task_data['reached_goal_position'] = False            
            self.task_data['goal_idx'] = np.nan
        else:
            self.task_data['goal_pos'] = self.goal_position
            self.task_data['reached_goal_position'] = self.reached_goal_position
            self.task_data['goal_idx'] = self.goal_idx
        #print 'coefs', self.eeg_decoder.decoder.coef_
        self.task_data['eeg_decoder_coef']  = self.eeg_decoder.decoder.coef_
        self.task_data['eeg_decoder_intercept']  = self.eeg_decoder.decoder.intercept_
        self.task_data['eeg_decoder_means']  = self.eeg_decoder.decoder.means_

        
        self.task_data['plant_pos']  = self.plant_pos_raw.values
        self.task_data['plant_pos_filt']  = self.plant_pos.values 
        self.task_data['plant_vel']  = self.plant_vel_raw.values
        self.task_data['plant_vel_filt']  = self.plant_vel.values

        self.task_data['trial_type'] = self.trial_type
        self.task_data['ts']         = time.time()
        self.task_data['target_margin'] = self.target_margin
        #self.task_data['decoder']    = self.eeg_decoder.decoder

      
        super(SimEEGMovementDecodingNew, self)._cycle()
   

class EMGEndPointMovement(RecordBrainAmpData,EndPointMovement):

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


    def __init__(self, *args, **kwargs):
        super(EMGEndPointMovement, self).__init__(*args, **kwargs)



class EXGCyclicEndPointMovement(RecordBrainAmpData,CyclicEndPointMovement):

    def __init__(self, *args, **kwargs):
        super(EXGCyclicEndPointMovement, self).__init__(*args, **kwargs)

class EXGEndPointMovement(RecordBrainAmpData,EndPointMovement):
    # Task to record compliant movements with 100% assistance while recording different data (e.g. EEG, EOG, EMG, etc...). 
    # Preparation and rest periods are included both in the forward and backward phases of the movement.
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
            'end_trial': 'instruct_rest_return',
            'stop':      None},  
        'instruct_rest_return': {
            'end_instruct': 'rest_return',
            'stop':      None},            
        'rest_return': {
            'end_rest': 'instruct_trial_return',
            'stop':      None},
        'instruct_trial_return': {
            'end_instruct': 'preparation_return',
            'stop':      None},
        'preparation_return': {
            'end_preparation': 'instruct_go_return',
            'stop':      None},    
        'instruct_go_return': {
            'end_instruct': 'trial_return',
            'stop':      None},  
        'trial_return': {            
            'end_trial': 'wait',
            'stop':      None},    
        } 
    
    state = 'wait'  # initial state


    def __init__(self, *args, **kwargs):
        super(EXGEndPointMovement, self).__init__(*args, **kwargs)

    def _start_instruct_rest_return(self):
        self._play_sound(self.sounds_dir, ['rest'])

    def _start_instruct_go_return(self):
        self._play_sound(self.sounds_dir, ['go'])


class EXGEndPointMovement_testing(RecordBrainAmpData,EndPointMovement_testing):
    # Task to record compliant movements with 100% assistance while recording different data (e.g. EEG, EOG, EMG, etc...). 
    # Preparation and rest periods are included both in the forward and backward phases of the movement.
    # fps = 20

    # status = {
    #     'wait': {
    #         'start_trial': 'instruct_rest',
    #         'stop': None},
    #     'instruct_rest': {
    #         'end_instruct': 'rest',
    #         'stop':      None},            
    #     'rest': {
    #         'end_rest': 'instruct_trial_type',
    #         'stop':      None},
    #     'instruct_trial_type': {
    #         'end_instruct': 'preparation',
    #         'stop':      None},
    #     'preparation': {
    #         'end_preparation': 'instruct_go',
    #         'stop':      None},    
    #     'instruct_go': {
    #         'end_instruct': 'trial',
    #         'stop':      None},
    #     'trial': {
    #         # 'end_trial' : 'instruct_rest',
    #         'end_trial' : 'instruct_trial_type',
    #         'end_alltrials' : 'wait',
    #         'stop':      None},    
    #     }
    
    # state = 'wait'  # initial state


    def __init__(self, *args, **kwargs):
        super(EXGEndPointMovement_testing, self).__init__(*args, **kwargs)



# class SimEMGEndPointMovement(EndPointMovement):
#     fps = 20
#     status = {
#     'wait': {
#         'start_trial': 'instruct_rest',
#         'stop': None},
#     'instruct_rest': {
#         'end_instruct': 'rest',
#         'stop':      None},            
#     'rest': {
#         'end_rest': 'instruct_trial_type',
#         'stop':      None},
#     'instruct_trial_type': {
#         'end_instruct': 'trial',
#         'stop':      None},
#     'trial': {
#         'end_trial': 'instruct_trial_return',#'instruct_trial_go_to_start'
#         'stop':      None},    
#     'instruct_trial_return': {
#         'end_instruct': 'trial_return',
#         'stop':      None},
#     'trial_return': {
#         'end_trial': 'wait',
#         'stop':      None},    
#     }

#     state = 'wait'  # initial state

#     # settable parameters on web interface    
#     preparation_time = 2.
#     rest_interval  = (2., 3.) 
#     import pickle
#     targets_matrix = pickle.load(open('/home/lab/code/ismore/ismore_tests/targets_matrix_testing_4462_B1.pkl'))
#     give_feedback  = 0 
#     debug = False
#     import brainamp_channel_lists
#     brainamp_channels = brainamp_channel_lists.eog1_raw_filt
    
#     def __init__(self, *args, **kwargs):
#         super(SimEMGEndPointMovement, self).__init__(*args, **kwargs)


class EMGTrajDecodingEndPoint(EMGEndPointMovement):

    gamma             = traits.Float(0.5,   desc='Gamma value for incorporating EMG decoded velocity.')
    emg_decoder       = traits.InstanceFromDB(LinearEMGDecoder, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='emg_decoder'))
    channel_list_name = traits.OptionsList(*channel_list_options, bmi3d_input_options=channel_list_options) #choose always the filtered + raw option!!!!!!

    #use_emg_decoder   = traits.Int((0,1), desc=' 0 if we do not give feedback, 1 if we give feedback')

    def __init__(self, *args, **kwargs):
        super(EMGTrajDecodingEndPoint, self).__init__(*args, **kwargs)
       
        # if EMG decoder file was specified, load EMG decoder and create feature extractor 
        # if len(self.emg_decoder_file) > 3:
        #     self.emg_decoder = pickle.load(open(self.emg_decoder_file, 'rb'))

        emg_extractor_cls    = self.emg_decoder.extractor_cls
        self.emg_extractor_kwargs = self.emg_decoder.extractor_kwargs

        #print [settings.BRAINAMP_CHANNELS[chan] + "_filt" for i, chan in enumerate(settings.BRAINAMP_CHANNELS)]
        # if self.brainamp_channels != channels_filt:
        #     print 'ERROR: The selected channels in the interface do not match those defined in settings to be streamed from the amplifier.'

        # create EMG extractor object (it's 'source' will be set later in the init method)
        if self.emg_decoder.plant_type != self.plant_type:
            print 'Chosen plant_type on the interface does not match the plant type used to train the decoder. Make sure you are selecting the right one'
    
        self.brainamp_channels = getattr(brainamp_channel_lists, self.channel_list_name)
        try:
            self.channels = self.emg_extractor_kwargs['emg_channels']
        except:
            self.channels = self.emg_extractor_kwargs['channels']
        # self.channels_str_2discard = emg_extractor_kwargs['channels_str_2discard']
        # self.channels_str_2keep = emg_extractor_kwargs['channels_str_2keep']
        # self.channels_diag1_1 = emg_extractor_kwargs['channels_diag1_1']
        # self.channels_diag1_2 = emg_extractor_kwargs['channels_diag1_2']
        # self.channels_diag2_1 = emg_extractor_kwargs['channels_diag2_1']
        # self.channels_diag2_2 = emg_extractor_kwargs['channels_diag2_2']
        #self.brainamp_channels = emg_extractor_kwargs['brainamp_channels'] 
        
        # extractor_kwargs['channels_filt'] = list()
        # for i in range(len(extractor_kwargs['channels'])):
        #     extractor_kwargs['channels_filt'] = [extractor_kwargs['channels'][i] + "_filt"]
        #     extractor_kwargs['channels_filt'].append(extractor_kwargs['channels_filt'])
        
        #self.emg_extractor = emg_extractor_cls(source=None, channels = self.brainamp_channels, **extractor_kwargs)
        self.nstates_decoder = len(self.emg_decoder.beta.keys())
        #self.emg_extractor = emg_extractor_cls(source=None, **self.extractor_kwargs)
        #self.emg_extractor = emg_extractor_cls(None, emg_channels = self.emg_extractor_kwargs['emg_channels'], feature_names = self.emg_extractor_kwargs['feature_names'], feature_fn_kwargs = self.emg_extractor_kwargs['feature_fn_kwargs'], win_len=self.emg_extractor_kwargs['win_len'], fs=self.emg_extractor_kwargs['fs'])
        self.emg_extractor = emg_extractor_cls(None, emg_channels = self.channels, feature_names = self.emg_extractor_kwargs['feature_names'], feature_fn_kwargs = self.emg_extractor_kwargs['feature_fn_kwargs'], win_len=self.emg_extractor_kwargs['win_len'], fs=self.emg_extractor_kwargs['fs'])
        self.emg_decoder_name = self.emg_decoder.decoder_name

        self.add_dtype('emg_features',    'f8', (self.emg_extractor.n_features,))
        self.add_dtype('emg_features_Z',  'f8', (self.emg_extractor.n_features,))
        self.add_dtype('emg_features_mean',    'f8', (self.emg_extractor.n_features,))
        self.add_dtype('emg_features_std',    'f8', (self.emg_extractor.n_features,))
        #self.add_dtype('emg_vel',         'f8', (len(self.vel_states),))
        self.add_dtype('emg_vel',         'f8', (self.nstates_decoder,))# to save all the kinematics estimated by the emg decoder even if a less DoF plant is being used online. At least the ones without applying the lpf
        self.add_dtype('predefined_vel',  'f8', (len(self.vel_states),))
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
        
        self.plant.enable() 

        # 4th order butterworth filter for command_vel
        fs_synch = self.fps #Frequency at which the task is running
        nyq   = 0.5 * fs_synch
        cuttoff_freq  = 1.5 / nyq
        bpf_kin_coeffs = butter(2, cuttoff_freq, btype='low')

        self.command_lpfs = dict()
        for state in self.vel_states:
            self.command_lpfs[state] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1]) # low-pass filter to smooth out command velocities

    def init(self):
        #Check if I can call first the init() function and then assign the brainamp source!!!!
        super(EMGTrajDecodingEndPoint, self).init()
        #from ismore.brainamp import rda
        self.emg_extractor.source = self.brainamp_source
        #self.emg_extractor.source = rda.SimEMGData
        
    def move_plant(self):
        command_vel  = pd.Series(0.0, self.vel_states)
        command_vel_raw  = pd.Series(0.0, self.vel_states)
        emg_vel  = pd.Series(0.0, self.vel_states) 
        emg_vel_lpf  = pd.Series(0.0, self.vel_states)
        predefined_vel  = pd.Series(0.0, self.vel_states)
        command_vel_final  = pd.Series(0.0, self.vel_states) 

        # run EMG feature extractor and decoder      
        emg_features = self.emg_extractor() # emg_features is of type 'dict'

        #emg_features[self.emg_extractor.feature_type] = emg_features[self.emg_extractor.feature_type][len(emg_features[self.emg_extractor.feature_type])/2:]
        self.features_buffer.add(emg_features[self.emg_extractor.feature_type])
        #print 'emg_features[self.emg_extractor.feature_type]', emg_features[self.emg_extractor.feature_type].shape
        
        if self.features_buffer.num_items() > 1 * self.fps:#1: #
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
        emg_features_Z = (emg_features[self.emg_extractor.feature_type] - features_mean) / features_std 
        # print 'std',features_std[-1]
        emg_vel = self.emg_decoder(emg_features_Z)
        self.emg_vel_buffer.add(emg_vel[self.vel_states])

        
        # Using a weighted moving avge filter
        n_items = self.emg_vel_buffer.num_items()
        buffer_emg = self.emg_vel_buffer.get(n_items)
        win = min(9,n_items)
        weights = np.arange(1./win, 1 + 1./win, 1./win)
        try:
            emg_vel_lpf = np.sum(weights*buffer_emg[:,n_items-win:n_items+1], axis = 1)/np.sum(weights)
        except:
            pass


        # Using a regular moving avge filter
        # emg_vel_lpf = np.mean(self.emg_vel_buffer.get_all(), axis=1)

        self.task_data['emg_features']   = emg_features[self.emg_extractor.feature_type]
        self.task_data['emg_features_Z'] = emg_features_Z
        self.task_data['emg_features_mean']   = features_mean
        self.task_data['emg_features_std']   = features_std
        self.task_data['emg_vel']        = emg_vel.values
        self.task_data['emg_vel_lpf']    = emg_vel_lpf
                  

        # Command zero velocity if the task is in a non-moving state
        if self.state in ['wait','rest', 'instruct_rest', 'instruct_trial_type']: 
            command_vel[:] = 0

        elif self.state in ['trial', 'trial_return']: 
            current_pos = self.plant_pos[:].ravel()
            current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
            
            if self.state == 'trial_return':

                target_state = np.hstack([self.targets_matrix['rest'][0][self.pos_states], np.zeros_like(current_pos),1]).reshape(-1,1)
  

            elif self.state == 'trial':
                target_state = np.hstack([self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states], np.zeros_like(current_pos),1 ]).reshape(-1,1)
            #print 'diff', target_state - current_state
            #print "calling assister"
            assist_output = self.assister(current_state, target_state, 1.)
            Bu = np.array(assist_output["x_assist"]).ravel()

            predefined_vel[:] = Bu[len(current_pos):len(current_pos)*2]

            # combine EMG decoded velocity and playback velocity into one velocity command
            norm_playback_vel = np.linalg.norm(predefined_vel)
            epsilon = 1e-6
            if (norm_playback_vel < epsilon):
                # if norm of the playback velocity is 0 or close to 0,
                #   then just set command velocity to 0s
                command_vel[:] = 0.0

            else:

                #feedback 1
                term1 = self.gamma * emg_vel_lpf
                #print 'emg_vel_lpf', emg_vel_lpf
                term2 = (1 - self.gamma) * predefined_vel
                command_vel = term1 + term2
                #command_vel = term2
                #print 'predefined_vel', predefined_vel
                # print 'pred_vel', predefined_vel
                
                # print 'command', command_vel
                #feedback 2
                # term1 = self.gamma * ((np.dot(emg_vel_lpf, predefined_vel) / (norm_playback_vel**2)) * predefined_vel)
                # term2 = (1 - self.gamma) * predefined_vel


                #term1 = self.gamma * self.emg_decoder.gamma_coeffs * ((np.dot(emg_vel_lpf, predefined_vel) / (norm_predefined_vel**2)) * predefined_vel)
                #term2 = (1 - self.gamma * self.emg_decoder.gamma_coeffs) * predefined_vel


      

        command_vel_raw[:] = command_vel[:]

        # # # # Apply low-pass filter to command velocities
        for state in self.vel_states:
        #     print command_vel[state]
            command_vel[state] = self.command_lpfs[state](command_vel[state])
            if np.isnan(command_vel[state]):
                command_vel[state] = 0

        self.task_data['command_vel']  = command_vel.values
        self.task_data['command_vel_raw']  = command_vel_raw.values
        self.task_data['predefined_vel'] = predefined_vel.values

        # set the velocities of the DoFs that should not be controlled to zero.
        if np.isnan(self.DoF_not_control_idx_init) == False and np.isnan(self.DoF_not_control_idx_end) == False:
            command_vel[self.DoF_not_control_idx_init:self.DoF_not_control_idx_end] = 0

        self.plant.send_vel(command_vel.values) #send velocity command to EXO
        self.task_data['command_vel_final']  = command_vel.values


    def cleanup_hdf(self):
        super(EMGTrajDecodingEndPoint, self).cleanup_hdf()    

        import tables
        h5file = tables.openFile(self.h5file.name, mode='a')
        h5file.root.task.attrs['emg_decoder_name'] = self.emg_decoder_name
        #h5file.root.task.attrs['brainamp_channels'] = self.channel_list_name
        #compfilt = tables.Filters(complevel=5, complib="zlib", shuffle=True)
        emg_extractor_grp = h5file.createGroup(h5file.root, "emg_extractor_kwargs", "Parameters for feature extraction")
        for key in self.emg_extractor_kwargs:
            if isinstance(self.emg_extractor_kwargs[key], dict):
                if key != 'feature_fn_kwargs':
                    for key2 in self.emg_extractor_kwargs[key]:
                        if isinstance(self.emg_extractor_kwargs[key][key2], np.ndarray):
                            h5file.createArray(emg_extractor_grp, key + '_' + key2, self.emg_extractor_kwargs[key][key2])
                        else:
                            h5file.createArray(emg_extractor_grp, key + '_' + key2, np.array([self.emg_extractor_kwargs[key][key2]]))

            else:
                if isinstance(self.emg_extractor_kwargs[key], np.ndarray):
                    h5file.createArray(emg_extractor_grp, key, self.emg_extractor_kwargs[key])
                else:
                    h5file.createArray(emg_extractor_grp, key, np.array([self.emg_extractor_kwargs[key]]))
                
        h5file.close()

class EMGDecodingMotorLearning_ref(EMGTrajDecodingEndPoint):
    # Timeout option included with respect to EMGEndPointMovement task
    # Music also added
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
            'end_trial': 'instruct_trial_return',#'instruct_trial_go_to_start'
            'timeout': 'instruct_trial_return',
            'stop':      None},    
        'instruct_trial_return': {
            'end_instruct': 'trial_return',
            'stop':      None},
        'trial_return': {
            'end_trial': 'wait',
            'timeout': 'wait',
            'stop':      None},    
        }
    
    state = 'wait'  # initial state

    gamma             = traits.Float(0,   desc='Gamma value for incorporating EMG decoded velocity.')
    emg_decoder       = traits.InstanceFromDB(LinearEMGDecoder, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='emg_decoder'))
    channel_list_name = traits.OptionsList(*channel_list_options, bmi3d_input_options=channel_list_options) #choose always the filtered + raw option!!!!!!
    timeout_time      = traits.Float(30, desc='Maximum time given to the patient to accomplish the task before considering it like incomplete and re-starting it from the current position') 
    
    # #use_emg_decoder   = traits.Int((0,1), desc=' 0 if we do not give feedback, 1 if we give feedback')
    # def _play_sound(self, fname):
    #     pygame.mixer.music.load(fname)
    #     pygame.mixer.music.play()


    def __init__(self, *args, **kwargs):
        super(EMGDecodingMotorLearning_ref, self).__init__(*args, **kwargs)
        self.add_dtype('reached_timeout', bool, (1,))
        self.add_dtype('gamma_used', float, (1,))
        pygame.mixer.init()
        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)
        self.sounds_dir_classical = os.path.expandvars('$HOME/code/ismore/sounds/classical')
        self.reached_timeout = False
        self.reached_goal_position = False
        #self.channels = self.emg_decoder.extractor_kwargs['channels']
        
    # def init(self):
    #     #Check if I can call first the init() function and then assign the brainamp source!!!!
    #     super(EMGDecodingMotorLearning_ref, self).init()
    #     #from ismore.brainamp import rda
    #     self.emg_extractor.source = self.brainamp_source
    
    def _cycle(self):
        '''Runs self.fps times per second.'''


        self.task_data['gamma_used'] = self.gamma
        self.task_data['reached_timeout']  = self.reached_timeout

        super(EMGDecodingMotorLearning_ref, self)._cycle()

    def _end_instruct_trial_type(self):
        self.reached_goal_position = False
        self.reached_timeout =  False

    def _end_instruct_trial_return(self):
        self.reached_timeout =  False
        self.reached_goal_position = False

    def _test_end_trial(self,ts):
        return (self.reached_goal_position or self.reached_timeout)

    def _test_end_trial_return(self,ts):
        return (self.reached_goal_position or self.reached_timeout)

    def _test_timeout(self, ts):
        if ts > self.timeout_time:
            self.reached_timeout = True
            # self.task_data['reached_timeout']  = self.reached_timeout
            print 'timeout'
        return self.reached_timeout

class EMGDecodingMotorLearning(EMGDecodingMotorLearning_ref):


    gamma             = traits.Float(0.7,   desc='Gamma value for incorporating EMG decoded velocity.')
    emg_decoder       = traits.InstanceFromDB(LinearEMGDecoder, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='emg_decoder'))
    channel_list_name = traits.OptionsList(*channel_list_options, bmi3d_input_options=channel_list_options) #choose always the filtered + raw option!!!!!!
    # Same as _ref but in this case the gamma value is changed in two trials: one is assigned to 0.4 and the other one to 0.9. The rest are set to the value chosen from the interface.

    def __init__(self, *args, **kwargs):
        super(EMGDecodingMotorLearning, self).__init__(*args, **kwargs)

        import random
        self.gamma_low_block = random.randint(0,7)
        blocks = [num for num in np.arange(0,8) if num != self.gamma_low_block]
        self.gamma_high_block = random.choice(blocks)
        self.add_dtype('gamma_low_block', int, (1,))
        self.add_dtype('gamma_high_block', int, (1,))
        
        
        self.gamma_chosen = self.gamma
        self.block_number = 0


    def _parse_next_trial(self):
        self.trial_type = self.next_trial

        if self.block_number == self.gamma_low_block:
            self.gamma = 0.4
        elif self.block_number == self.gamma_high_block:
            self.gamma = 0.9
        else:
            self.gamma = self.gamma_chosen

        self.block_number +=1   

    def _cycle(self):
        '''Runs self.fps times per second.'''

        self.task_data['gamma_low_block'] = self.gamma_low_block
        self.task_data['gamma_high_block'] = self.gamma_high_block

   
        super(EMGDecodingMotorLearning, self)._cycle()


class EMGDecodingMotorLearning_question(EMGDecodingMotorLearning_ref):

    fps = 20
    # A additional state (question) is included to ask the subject to rate the difficulty of the trial.
    status = {
        'wait': {
            'start_trial': 'instruct_rest',
            'stop': None},
        'instruct_rest': {
            'end_instruct': 'rest',
            'stop':      None},  
         #None?    #???? Do I have to include this?   
        'rest': {
            'end_rest': 'instruct_trial_type',
            'stop':      None},
        'instruct_trial_type': {
            'end_instruct': 'trial',
            'stop':      None},
        'trial': {
            'end_trial': 'instruct_trial_return',#'instruct_trial_go_to_start'
            'timeout': 'instruct_trial_return',
            'stop':      None},    
        'instruct_trial_return': {
            'end_instruct': 'trial_return',
            'stop':      None},
        'trial_return': {
            'end_trial': 'question',
            'timeout': 'question',
            'stop':      None},  
        'question': {
            'accept_rating': 'wait',
            'reject_rating': 'question', 
            'stop':      None},   
        }

    gamma             = traits.Float(0.7,   desc='Gamma value for incorporating EMG decoded velocity.')
    emg_decoder       = traits.InstanceFromDB(LinearEMGDecoder, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='emg_decoder'))
    channel_list_name = traits.OptionsList(*channel_list_options, bmi3d_input_options=channel_list_options) #choose always the filtered + raw option!!!!!!


    def __init__(self, *args, **kwargs):
        super(EMGDecodingMotorLearning_question, self).__init__(*args, **kwargs)

        import random
        self.gamma_low_block = random.randint(0,7)
        blocks = [num for num in np.arange(0,8) if num != self.gamma_low_block]
        self.gamma_high_block = random.choice(blocks)
        self.add_dtype('gamma_low_block', int, (1,))
        self.add_dtype('gamma_high_block', int, (1,))
        self.add_dtype('rating_difficulty', float, (1,))
        self.add_dtype('experimenter_acceptance_of_rating', np.str_, 10)
        
        self.gamma_chosen = self.gamma
        self.block_number = 0
        self.rating_difficulty = np.nan
        self.experimenter_acceptance_of_rating = ''

    def _parse_next_trial(self):
        self.trial_type = self.next_trial

        if self.block_number == self.gamma_low_block:
            self.gamma = 0.4
        elif self.block_number == self.gamma_high_block:
            self.gamma = 0.9
        else:
            self.gamma = self.gamma_chosen

        self.block_number +=1   
        
        self.rating_difficulty = np.nan

    def _end_instruct_trial_type(self):
        self.reached_goal_position = False
        self.reached_timeout =  False
        self.experimenter_acceptance_of_rating = ''

    def _cycle(self):
        '''Runs self.fps times per second.'''

        self.task_data['gamma_low_block'] = self.gamma_low_block
        self.task_data['gamma_high_block'] = self.gamma_high_block
        self.task_data['rating_difficulty'] = self.rating_difficulty
        self.task_data['experimenter_acceptance_of_rating'] = self.experimenter_acceptance_of_rating

   
        super(EMGDecodingMotorLearning_question, self)._cycle()
    

    def _test_accept_rating(self, *args, **kwargs):
        self.task_data['rating_difficulty'] = self.rating_difficulty
        # self.task_data['experimenter_acceptance_of_rating'] = self.experimenter_acceptance_of_rating
        return self.experimenter_acceptance_of_rating == 'accept'

    def _test_reject_rating(self, *args, **kwargs):
        return self.experimenter_acceptance_of_rating == 'reject'

    def _start_question(self):
        self._play_sound(os.path.join(self.sounds_general_dir, 'beep.wav'))
        print 'Ask the subject to rate the difficulty of the control during the last trial'
        print 'Select the rating and click on Accept to continue with the experiment'

class HybridBCI(EEGMovementDecodingNew):

    # settable parameters on web interface for the EMG decoder  
    music_feedback    = traits.Int((1), desc=' 0 if we do not want to include music, 1 if we want different classical music pieces with increasing intensity to be played')
    gamma             = traits.Float(0.5,   desc='Gamma value for incorporating EMG decoded velocity.')
    emg_decoder       = traits.InstanceFromDB(LinearEMGDecoder, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='emg_decoder'))

    def __init__(self,*args, **kwargs):
        super(HybridBCI, self).__init__(*args,**kwargs)     
       

        emg_extractor_cls    = self.emg_decoder.extractor_cls
        self.emg_extractor_kwargs = self.emg_decoder.extractor_kwargs

        #print [settings.BRAINAMP_CHANNELS[chan] + "_filt" for i, chan in enumerate(settings.BRAINAMP_CHANNELS)]
        # if self.brainamp_channels != channels_filt:
        #     print 'ERROR: The selected channels in the interface do not match those defined in settings to be streamed from the amplifier.'

        # create EMG extractor object (it's 'source' will be set later in the init method)
        if self.emg_decoder.plant_type != self.plant_type:
            print 'Chosen plant_type on the interface does not match the plant type used to train the decoder. Make sure you are selecting the right one'
        try:
            self.emg_channels = self.emg_extractor_kwargs['emg_channels']
        except:
            self.emg_channels = self.emg_extractor_kwargs['channels']
        self.eeg_channels = self.eeg_extractor_kwargs['eeg_channels']
        self.channels = self.emg_channels + self.eeg_channels
        # self.channels_str_2discard = extractor_kwargs['channels_str_2discard']
        # self.channels_str_2keep = extractor_kwargs['channels_str_2keep']
        # self.channels_diag1_1 = extractor_kwargs['channels_diag1_1']
        # self.channels_diag1_2 = extractor_kwargs['channels_diag1_2']
        # self.channels_diag2_1 = extractor_kwarg['channels_diag2_1']
        # self.channels_diag2_2 = extractor_kwargs['channels_diag2_2']
        #self.brainamp_channels = extractor_kwargs['brainamp_channels'] 
        try:
            self.emg_extractor = emg_extractor_cls(None, emg_channels = self.emg_extractor_kwargs['emg_channels'], feature_names = self.emg_extractor_kwargs['feature_names'], feature_fn_kwargs = self.emg_extractor_kwargs['feature_fn_kwargs'], win_len=self.emg_extractor_kwargs['win_len'], fs=self.emg_extractor_kwargs['fs'])
        except:
            self.emg_extractor = emg_extractor_cls(None, emg_channels = self.emg_extractor_kwargs['channels'], feature_names = self.emg_extractor_kwargs['feature_names'], feature_fn_kwargs = self.emg_extractor_kwargs['feature_fn_kwargs'], win_len=self.emg_extractor_kwargs['win_len'], fs=self.emg_extractor_kwargs['fs'])
        self.emg_decoder_name = self.emg_decoder.decoder_name
        self.nstates_decoder = len(self.emg_decoder.beta.keys())

        self.add_dtype('emg_features',    'f8', (self.emg_extractor.n_features,))
        self.add_dtype('emg_features_Z',  'f8', (self.emg_extractor.n_features,))
        self.add_dtype('emg_features_mean',    'f8', (self.emg_extractor.n_features,))
        self.add_dtype('emg_features_std',    'f8', (self.emg_extractor.n_features,))
        #self.add_dtype('emg_vel',         'f8', (len(self.vel_states),))
        self.add_dtype('emg_vel',         'f8', (self.nstates_decoder,))# to save all the kinematics estimated by the emg decoder even if a less DoF plant is being used online. At least the ones without applying the lpf
        self.add_dtype('predefined_vel',  'f8', (len(self.vel_states),))
        self.add_dtype('emg_vel_lpf',     'f8', (len(self.vel_states),))

        #self.add_dtype('command_vel_raw',  'f8',    (len(self.vel_states),))
        #self.add_dtype('command_pos',  'f8',    (len(self.pos_states),))
        self.add_dtype('gamma_used', float, (1,))
        

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
        
        self.sounds_dir_classical = os.path.expandvars('$HOME/code/ismore/sounds/classical')
        pygame.mixer.init()

    def init(self):
        super(HybridBCI,self).init()
        self.emg_extractor.source = self.brainamp_source
        
    def move_plant(self):
        '''Docstring.'''

        command_vel  = pd.Series(0.0, self.vel_states)
        command_vel_raw  = pd.Series(0.0, self.vel_states)
        emg_vel  = pd.Series(0.0, self.vel_states) 
        emg_vel_lpf  = pd.Series(0.0, self.vel_states)
        predefined_vel  = pd.Series(0.0, self.vel_states) 
        command_vel_final  = pd.Series(0.0, self.vel_states)


        # run EEG&EMG feature extractor and decoder
        #self.eeg_extractor.source = self.brainamp_source
        if self.artifact_rejection == 1:
            eeg_features, rejected_window = self.eeg_extractor()
            self.task_data['rejected_window'] = rejected_window
        else:
            eeg_features = self.eeg_extractor() # eeg_features is of type 'dict'
    

        emg_features = self.emg_extractor() # emg_features is of type 'dict'
        self.features_buffer.add(emg_features[self.emg_extractor.feature_type])
        
        # EMG feature extraction and normalization
        if self.features_buffer.num_items() > 1 * self.fps:#1: #
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
        emg_features_Z = (emg_features[self.emg_extractor.feature_type] - features_mean) / features_std 
        # print 'std',features_std[-1]
        emg_vel = self.emg_decoder(emg_features_Z)
        self.emg_vel_buffer.add(emg_vel[self.vel_states])

        n_items = self.emg_vel_buffer.num_items()
        buffer_emg = self.emg_vel_buffer.get(n_items)
        win = min(9,n_items)
        weights = np.arange(1./win, 1 + 1./win, 1./win)
        try:
            emg_vel_lpf = np.sum(weights*buffer_emg[:,n_items-win:n_items+1], axis = 1)/np.sum(weights)
        except:
            pass


        # Using a regular moving avge filter
        # emg_vel_lpf = np.mean(self.emg_vel_buffer.get_all(), axis=1)

        self.task_data['emg_features']   = emg_features[self.emg_extractor.feature_type]
        self.task_data['emg_features_Z'] = emg_features_Z
        self.task_data['emg_features_mean'] = features_mean
        self.task_data['emg_features_std'] = features_std
        self.task_data['emg_vel']        = emg_vel.values
        self.task_data['emg_vel_lpf']    = emg_vel_lpf
                  


        # EEG feature extraction and normalization
        feat_mov = self.mov_feature_buffer.get_all()
        feat_rest = self.rest_feature_buffer.get_all()
        mean_feat = np.mean(np.hstack([feat_mov, feat_rest]), axis = 1)
        std_feat = np.std(np.hstack([feat_mov, feat_rest]), axis = 1)

        if self.trial_number > 0:
            if self.state in ['trial','trial_return']:
                if self.artifact_rejection == 1: 
                    if rejected_window == 0:
                        self.mov_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
                elif self.artifact_rejection == 0:
                    self.mov_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
                self.task_data['eeg_features_mov_buffer'] = eeg_features
                #self.mov_feature_buffer.add(eeg_features)
            elif self.state in ['rest','rest_return']:
                if self.artifact_rejection == 1:
                    if rejected_window == 0:
                        self.rest_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
                elif self.artifact_rejection == 0:
                    self.rest_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
                    
                self.task_data['eeg_features_rest_buffer'] = eeg_features
                #self.rest_feature_buffer.add(eeg_features)

        self.task_data['eeg_features'] = eeg_features
        self.task_data['eeg_mean_features'] = mean_feat
        self.task_data['eeg_std_features'] = std_feat
        
        # normalize features
        # eeg_features = (eeg_features - mean_feat.reshape(-1,1))/ std_feat.reshape(-1,1)
        eeg_features = (eeg_features - mean_feat)/ std_feat
        # mean_feat.ravel()
        self.task_data['eeg_features_Z'] = eeg_features

        #print 'eeg_features.shpae'

        try:
            eeg_features = np.reshape(eeg_features,(1,eeg_features.shape[0]))
        except:
            pass
        #eeg_features(eeg_features == np.inf) = 1
        self.decoder_output = self.eeg_decoder(eeg_features)
        self.probability = self.eeg_decoder.decoder.predict_proba(eeg_features)
        

        # Move plant according to the EEG&EMG decoders' outputs
        # Command zero velocity if the task is in a non-moving state
        if self.state not in ['trial', 'trial_return']:#['wait','rest', 'rest_return','instruct_rest', 'instruct_trial_type', 'preparation', 'instruct_go','instruct_rest_return']: 
            command_vel[:] = 0
            self.state_decoder = 0
            
        else:#if self.state in ['trial', 'trial_return', 'instruct_trial_return']:
            # compute the predefined vel independently of the robot having to move or not. Just to have a measure of how the correct veloctiy would be at all times.
           
            current_pos = self.plant_pos[:].ravel()
            current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
            target_state = np.hstack([self.goal_position, np.zeros_like(current_pos), 1]).reshape(-1, 1)
            assist_output = self.assister(current_state, target_state, 1)
                           
            Bu = np.array(assist_output["x_assist"]).ravel()
            #Bu = np.array(assist_output['Bu']).ravel()
            predefined_vel[:] = Bu[len(current_pos):len(current_pos)*2]

            if self.decoder_output == 1 and self.prev_output == 1:
                # we need 5 consecutive outputs of the same type
                self.consec_mov_outputs +=1
                if self.consec_mov_outputs == 5 and self.state_decoder == 0:
                    self.consec_rest_outputs = 0
            elif self.decoder_output == 1 and self.prev_output == 0:
                if self.state_decoder == 1: #if it's moving
                    self.consec_rest_outputs = 0
                else:
                    self.consec_mov_outputs = 1
            elif self.decoder_output == 0 and self.prev_output == 0:
                self.consec_rest_outputs +=1
                if self.consec_rest_outputs == 5 and self.state_decoder == 1:
                    self.consec_mov_outputs = 0
            elif self.decoder_output == 0 and self.prev_output == 1:
                if self.state_decoder == 1: #if it's moving
                    self.consec_rest_outputs = 1
                else:
                    self.consec_mov_outputs = 0
    
            if self.consec_mov_outputs >= 5:

                self.state_decoder = 1
                # current_pos = self.plant_pos[:].ravel()
                # current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
                # target_state = np.hstack([self.goal_position, np.zeros_like(current_pos), 1]).reshape(-1, 1)
                # assist_output = self.assister(current_state, target_state, 1)
                           
                # Bu = np.array(assist_output["x_assist"]).ravel()
                # #Bu = np.array(assist_output['Bu']).ravel()
                # predefined_vel[:] = Bu[len(current_pos):len(current_pos)*2]

                # combine EMG decoded velocity and playback velocity into one velocity command
                norm_playback_vel = np.linalg.norm(predefined_vel)
                epsilon = 1e-6
                if (norm_playback_vel < epsilon):
                    # if norm of the playback velocity is 0 or close to 0,
                    #   then just set command velocity to 0s
                    command_vel[:] = 0.0

                else:

                    #feedback 1
                    term1 = self.gamma * emg_vel_lpf
                    #print 'emg_vel_lpf', emg_vel_lpf
                    term2 = (1 - self.gamma) * predefined_vel
                    command_vel = term1 + term2

                   
            elif self.consec_rest_outputs >=5:

                self.state_decoder = 0
                command_vel[:] = 0 #set all the velocities to zero
                # command_vel_raw[:] = command_vel[:]
                # for state in self.vel_states:
                #     command_vel[state] = self.command_lpfs[state](command_vel[state])
                #     if np.isnan(command_vel[state]):
                #         command_vel[state] = 0

        command_vel_raw[:] = command_vel[:]
        for state in self.vel_states:
            command_vel[state] = self.command_lpfs[state](command_vel[state])
            if np.isnan(command_vel[state]):
                command_vel[state] = 0
        

        self.prev_output = self.decoder_output
        #self.task_data['eeg_features']   = eeg_features[self.eeg_extractor.feature_type]
        self.task_data['command_vel']  = command_vel.values
        self.task_data['command_vel_raw']  = command_vel_raw.values

        # set the velocities of the DoFs that should not be controlled to zero.
        if np.isnan(self.DoF_not_control_idx_init) == False and np.isnan(self.DoF_not_control_idx_end) == False:
            command_vel[self.DoF_not_control_idx_init:self.DoF_not_control_idx_end] = 0


        self.plant.send_vel(command_vel.values) #send velocity command to EXO

        self.task_data['decoder_output'] = self.decoder_output
        self.task_data['decoder_output_probability'] = self.probability
        self.task_data['state_decoder'] = self.state_decoder
        self.task_data['consec_mov_outputs'] = self.consec_mov_outputs
        self.task_data['consec_rest_outputs'] = self.consec_rest_outputs

        self.task_data['predefined_vel'] = predefined_vel.values
        self.task_data['command_vel_final']  = command_vel.values

    def _cycle(self):
        '''Runs self.fps times per second.'''

        super(HybridBCI, self)._cycle()
        self.task_data['gamma_used'] = self.gamma

    def _start_trial(self):
        print self.trial_type
        self.goal_position = self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states]
        if self.music_feedback:
            # self._play_sound(os.path.join(self.sounds_dir_classical, self.trial_type + '.wav')) #nerea
            self._play_sound(self.sounds_dir_classical, [self.subgoal_names[self.trial_type][self.goal_idx][0]])
       
    def _start_trial_return(self):
        print 'return trial'
        self.goal_position = self.targets_matrix['rest'][0][self.pos_states]
        self.goal_idx = 0
        if self.music_feedback:
            # self._play_sound(os.path.join(self.sounds_dir_classical, self.trial_type + '.wav'))
            self._play_sound(self.sounds_dir_classical, [self.subgoal_names[self.trial_type][self.goal_idx][0]])

    def _end_trial(self):

        if self.music_feedback:
            pygame.mixer.music.stop()
            self.parallel_sound.stop()
        else:
            pass
        self.trial_number +=1

    def cleanup_hdf(self):
        import tables
        h5file = tables.openFile(self.h5file.name, mode='a')
        h5file.root.task.attrs['emg_decoder_name'] = self.emg_decoder_name
        #h5file.root.task.attrs['brainamp_channels'] = self.channel_list_name
        #compfilt = tables.Filters(complevel=5, complib="zlib", shuffle=True)
        emg_extractor_grp = h5file.createGroup(h5file.root, "emg_extractor_kwargs", "Parameters for feature extraction")
        for key in self.emg_extractor_kwargs:
            if isinstance(self.emg_extractor_kwargs[key], dict):
                if key != 'feature_fn_kwargs':
                    for key2 in self.emg_extractor_kwargs[key]:
                        if isinstance(self.emg_extractor_kwargs[key][key2], np.ndarray):
                            h5file.createArray(emg_extractor_grp, key + '_' + key2, self.emg_extractor_kwargs[key][key2])
                        else:
                            h5file.createArray(emg_extractor_grp, key + '_' + key2, np.array([self.emg_extractor_kwargs[key][key2]]))

            else:
                if isinstance(self.emg_extractor_kwargs[key], np.ndarray):
                    h5file.createArray(emg_extractor_grp, key, self.emg_extractor_kwargs[key])
                else:
                    h5file.createArray(emg_extractor_grp, key, np.array([self.emg_extractor_kwargs[key]]))
                
        h5file.close()
        super(HybridBCI, self).cleanup_hdf()


class EMGClassificationEndPoint(EMGEndPointMovement):

    emg_classifier    = traits.InstanceFromDB(SVM_EMGClassifier, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='emg_classifier'))
    channel_list_name = traits.OptionsList(*channel_list_options, bmi3d_input_options=channel_list_options) 

    def __init__(self, *args, **kwargs):
        super(EMGClassificationEndPoint, self).__init__(*args, **kwargs)
       
        #self.target_margin = pd.Series(np.array([1, 1, np.deg2rad(3), np.deg2rad(1),  np.deg2rad(1), np.deg2rad(1), np.deg2rad(1)]), ismore_pos_states)
        #self.target_margin = self.target_margin[self.pos_states]

        # if EMG decoder file was specified, load EMG decoder and create feature extractor 
        # if len(self.emg_decoder_file) > 3:
        #     self.emg_decoder = pickle.load(open(self.emg_decoder_file, 'rb'))
        emg_extractor_cls    = self.emg_classifier.extractor_cls
        self.emg_extractor_kwargs = self.emg_classifier.extractor_kwargs

        #print [settings.BRAINAMP_CHANNELS[chan] + "_filt" for i, chan in enumerate(settings.BRAINAMP_CHANNELS)]
        # if self.brainamp_channels != channels_filt:
        #     print 'ERROR: The selected channels in the interface do not match those defined in settings to be streamed from the amplifier.'

        # create EMG extractor object (it's 'source' will be set later in the init method)
      
        self.brainamp_channels = getattr(brainamp_channel_lists, self.channel_list_name)
        self.channels = self.emg_classifier.channel_names
        #self.brainamp_channels = self.emg_extractor_kwargs['brainamp_channels'] 
        self.brainamp_channels = self.emg_extractor_kwargs['emg_channels'] 
        print "brainamp_channels ", self.brainamp_channels
        #self.channels = [chan + '_filt' for chan in self.brainamp_channels]

        # we have 2 independent classifiers: Mov-NoMov classifier and MultiClass classifier
        self.MovNoMov_classifier = self.emg_classifier.classifier_MovNoMov
        self.MultiClass_classifier = self.emg_classifier.classifier_MultiClass

        self.mov_class_labels = self.MultiClass_classifier.mov_class_labels
        self.output_classes = self.MultiClass_classifier.output_classes
        #self.nstates_decoder = len(self.emg_decoder.beta.keys())
        
        try:
            self.emg_extractor = emg_extractor_cls(None, emg_channels = self.emg_extractor_kwargs['emg_channels'], feature_names = self.emg_extractor_kwargs['feature_names'], feature_fn_kwargs = self.emg_extractor_kwargs['feature_fn_kwargs'], win_len=self.emg_extractor_kwargs['win_len'], fs=self.emg_extractor_kwargs['fs'])
        except:
            self.emg_extractor = emg_extractor_cls(None, emg_channels = self.emg_extractor_kwargs['channels'], feature_names = self.emg_extractor_kwargs['feature_names'], feature_fn_kwargs = self.emg_extractor_kwargs['feature_fn_kwargs'], win_len=self.emg_extractor_kwargs['win_len'], fs=self.emg_extractor_kwargs['fs'])
    
        self.emg_classifier_name = self.emg_classifier.classifier_name

        self.add_dtype('emg_features',    'f8', (self.emg_extractor.n_features,))
        self.add_dtype('emg_features_Z',  'f8', (self.emg_extractor.n_features,))
        self.add_dtype('emg_features_mean',    'f8', (self.emg_extractor.n_features,))
        self.add_dtype('emg_features_std',    'f8', (self.emg_extractor.n_features,))
        
        #these might need to go somewhere else
        self.add_dtype('MovNoMov_classifier_output',  float, (1,))
        self.add_dtype('MultiClass_classifier_output', float, (1,))


        self.add_dtype('mov_class_consec_mov_outputs',  int, (1,))
        self.add_dtype('mov_class_consec_rest_outputs',  int, (1,))
        self.add_dtype('mov_class_state',  int, (1,))
        self.add_dtype('multi_class_state',  int, (1,))
        

        #initialize values for the state of the decoder
        self.mov_class_consec_mov_outputs = 0
        self.mov_class_consec_rest_outputs = 0

        self.mov_class_prev_output = 0
        self.multi_class_prev_output = []

        self.mov_class_state= 0
        self.multi_class_state= 0

        #self.add_dtype('predefined_vel',  'f8', (len(self.vel_states),))
        #self.add_dtype('emg_vel_lpf',     'f8', (len(self.vel_states),))

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

        # for collecting last multi_class_classifier last 10 outputs
        self.muticlass_output_buffer = RingBuffer(
            item_len=1,
            capacity=10,
        )
        
        self.plant.enable() 

        # 4th order butterworth filter for command_vel
        fs_synch = self.fps #Frequency at which the task is running
        nyq   = 0.5 * fs_synch
        cuttoff_freq  = 1.5 / nyq
        bpf_kin_coeffs = butter(2, cuttoff_freq, btype='low')

        self.command_lpfs = dict()
        for state in self.vel_states:
            self.command_lpfs[state] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1]) # low-pass filter to smooth out command velocities

    def init(self):
        #Check if I can call first the init() function and then assign the brainamp source!!!!
        
        from riglib import source
        from ismore.brainamp import rda
        #self.brainamp_source = source.MultiChanDataSource(rda.EMGData, name='brainamp', channels=self.brainamp_channels, brainamp_channels=self.brainamp_channels, send_data_to_sink_manager=True)
        self.emg_extractor.source = self.brainamp_source
        super(EMGClassificationEndPoint, self).init()


    def move_plant(self):
        command_vel  = pd.Series(0.0, self.vel_states)
        command_vel_raw  = pd.Series(0.0, self.vel_states)
        command_vel_final  = pd.Series(0.0, self.vel_states)
        
        # emg_vel  = pd.Series(0.0, self.vel_states) 
        # emg_vel_lpf  = pd.Series(0.0, self.vel_states)
        

        # run EMG feature extractor and decoder      
        emg_features = self.emg_extractor() # emg_features is of type 'dict'
        #emg_features[self.emg_extractor.feature_type] = emg_features[self.emg_extractor.feature_type][len(emg_features[self.emg_extractor.feature_type])/2:]
        self.features_buffer.add(emg_features[self.emg_extractor.feature_type])
        #print 'emg_features[self.emg_extractor.feature_type]', emg_features[self.emg_extractor.feature_type].shape
        
        if self.features_buffer.num_items() > 1 * self.fps:#1: #
                # if we have more than 1 second of recent EMG data, then
                #   calculate mean and std from this data
            recent_features = self.features_buffer.get_all()
            #print 'recent_features', recent_features.shape
            features_mean = np.mean(recent_features, axis=1)
            features_std  = np.std(recent_features, axis=1)
        else:
                # else use mean and std from the EMG data that was used to 
                #   train the decoder
            features_mean = self.MovNoMov_classifier.features_mean
            features_std  = self.MovNoMov_classifier.features_std

        features_std[features_std == 0] = 1

        # z-score the EMG features
        emg_features_Z = (emg_features[self.emg_extractor.feature_type] - features_mean) / features_std 
        # print 'std',features_std[-1]
        
        # classify between Mov and NoMov
        self.MovNoMov_classifier_output = self.MovNoMov_classifier(emg_features_Z)
        self.MultiClass_classifier_output = self.MultiClass_classifier(emg_features_Z)

        # store last 10 outputs of the multiclass classifier
        self.muticlass_output_buffer.add(self.MultiClass_classifier_output)
        #print 'self.muticlass_output_buffer : ', self.muticlass_output_buffer.get_all()[0]

        #emg_vel = self.emg_decoder(emg_features_Z)

        self.task_data['emg_features']   = emg_features[self.emg_extractor.feature_type]
        self.task_data['emg_features_Z'] = emg_features_Z
        self.task_data['emg_features_mean']   = features_mean
        self.task_data['emg_features_std']   = features_std
        
        self.task_data['MovNoMov_classifier_output']        = self.MovNoMov_classifier_output
        self.task_data['MultiClass_classifier_output']    = self.MultiClass_classifier_output
                  
        #print 'MovNoMov_classifier_output : ', self.MovNoMov_classifier_output
        #print 'MultiClass_classifier_output : ', self.MultiClass_classifier_output


        # Command zero velocity if the task is in a non-moving state
        if self.state not in ['trial', 'trial_return']: 
            command_vel[:] = 0
            self.mov_class_state = 0
            self.state_multiclass_classifier = []

        else:
            # movement detected in the present and previous cycles
            if self.MovNoMov_classifier_output == 1 and self.mov_class_prev_output == 1:
                self.mov_class_consec_mov_outputs +=1
                if self.mov_class_consec_mov_outputs == 5 and self.mov_class_state == 0:
                    self.mov_class_consec_rest_outputs = 0
            # movement detected in this cycle but not in previous
            elif self.MovNoMov_classifier_output == 1 and self.mov_class_prev_output == 0:
                #if the movnomov_class was not moving
                if self.mov_class_state == 1: 
                    self.mov_class_consec_rest_outputs = 0
                #if the movnomov_class was moving
                else:
                    self.mov_class_consec_mov_outputs = 1
            # no mov detected in previous and present cycles
            elif self.MovNoMov_classifier_output == 0 and self.mov_class_prev_output == 0:
                self.mov_class_consec_rest_outputs +=1
                if self.mov_class_consec_rest_outputs == 5 and self.mov_class_state == 1:
                    self.mov_class_consec_mov_outputs = 0
            # movement detected in this cycle, not in previous
            elif self.MovNoMov_classifier_output == 0 and self.mov_class_prev_output == 1:
                # it was moving
                if self.mov_class_state == 1: 
                    self.mov_class_consec_rest_outputs = 1
                #it was not moving
                else:
                    self.mov_class_consec_mov_outputs = 0


            if self.mov_class_consec_mov_outputs >= 3:

                self.mov_class_state = 1

                current_pos = self.plant_pos[:].ravel()
                current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
                
                # self.goal_position is the position where the exo should be moving to according to the instructions that the subject is rceiving
                # self.classified_position is the position that is being classified from the emg 
                

                from scipy.stats import mode
                self.classified_position = int(mode(self.muticlass_output_buffer.get_all()[0])[0])
                #print 'self.classified_position : ', self.classified_position
            
                self.classified_position_idx = [i for i, j in enumerate(self.mov_class_labels) if j == self.classified_position]
                #print 'self.classified_position_idx  ', self.classified_position_idx

                self.classified_goal_trial_type  = self.output_classes[self.classified_position_idx[0]]
                #print 'self.classified_goal_target  ', self.classified_goal_trial_type
                    
                self.goal_classified_positon = self.targets_matrix[self.classified_goal_trial_type][self.goal_idx][self.pos_states]

                target_state = np.hstack([self.goal_classified_positon, np.zeros_like(current_pos), 1]).reshape(-1, 1)
                assist_output = self.assister(current_state, target_state, 1)
                           
                Bu = np.array(assist_output["x_assist"]).ravel()
                #Bu = np.array(assist_output['Bu']).ravel()
                command_vel[:] = Bu[len(current_pos):len(current_pos)*2]
                #print 'command_vel', command_vel
                #set all the velocities to a constant value towards the end point
            
            elif self.mov_class_consec_rest_outputs >=5:

                self.mov_class_state = 0
                command_vel[:] = 0 #set all the velocities to zero

        
        if self.mov_class_state == 1:
            print 'mov detected - ' , self.classified_goal_trial_type

        self.mov_class_prev_output = self.MovNoMov_classifier_output

        command_vel_raw[:] = command_vel[:]

        # # # # Apply low-pass filter to command velocities
        for state in self.vel_states:
        #     print command_vel[state]
            command_vel[state] = self.command_lpfs[state](command_vel[state])
            if np.isnan(command_vel[state]):
                command_vel[state] = 0

        self.task_data['command_vel']  = command_vel.values
        self.task_data['command_vel_raw']  = command_vel_raw.values

        # set the velocities of the DoFs that should not be controlled to zero.
        if np.isnan(self.DoF_not_control_idx_init) == False and np.isnan(self.DoF_not_control_idx_end) == False:
            command_vel[self.DoF_not_control_idx_init:self.DoF_not_control_idx_end] = 0

        self.plant.send_vel(command_vel.values) #send velocity command to EXO 

        self.task_data['command_vel_final']  = command_vel.values

       

################## Measurements tasks ##################
class ExG_FM_ARAT_CODA(RecordBrainAmpData, Sequence):
    '''
    Task to record ExG and send triggers to CODA to start recording and trial synchro triggers
    '''
    #needs to inherit from RecordBrainAmpData first to run the init of Autostart before than the init of Sequence
    fps = 20

    status = {
        'wait': {
            'start_trial': 'rest',
            'stop': None},
        'rest': {
            'end_rest' : 'trial',
            'stop':      None},
        'trial': {
            # 'end_trial': 'wait',
            'accept_trial': 'wait',
            'reject_trial': 'rest',
            'stop': None},      
    }
    state = 'wait'  # initial state

    channel_list_name = traits.OptionsList(*channel_list_options, bmi3d_input_options=channel_list_options) #choose always the filtered + raw option!!!!!!
    window_size = traits.Tuple((500, 280), desc='Size of window to display the plant position/angle')
    language = traits.OptionsList(*languages_list, bmi3d_input_options=languages_list)
    # rest_interval = traits.Tuple((2., 4.), desc='Min and max time to remain in the rest state.')
  
    sequence_generators = ['FM_CODA_tasks']
  
    @staticmethod
    def FM_CODA_tasks(length=3, shoulder_flexion=1, hand_extension=1, spherical_grip=1):
        available_targets = []
        if shoulder_flexion: available_targets.append('shoulder_flexion')
        if hand_extension: available_targets.append('hand_extension')
        if spherical_grip: available_targets.append('spherical_grip')

        targets = available_targets*length
        return targets  

    def __init__(self, *args, **kwargs):
        ## Init the pygame mixer for playing back sounds
        super(ExG_FM_ARAT_CODA, self).__init__(*args, **kwargs)
        self.add_dtype('trial_type', np.str_, 40)
        self.add_dtype('ts',         'f8',    (1,))
        self.add_dtype('trial_start_accept_reject', np.str_, 10)


        self.image_dir_general = os.path.expandvars('$HOME/code/ismore/images')
        self.image_dir = os.path.join(self.image_dir_general, self.language)

        self.experimenter_acceptance_of_trial = ''

        self.port = serial.Serial('/dev/arduino_neurosync',baudrate=115200)

    def _cycle(self):
        '''Runs self.fps times per second.'''
        # try:
        #     self.task_data['trial_type'] = self.trial_type
        # except:
        #     ''       
        self.task_data['trial_type'] = self.trial_type

        if (self.experimenter_acceptance_of_trial in ['accept', 'reject', 'start']):
            print self.experimenter_acceptance_of_trial

        # print self.experimenter_acceptance_of_trial
        self.task_data['ts']         = time.time()
        self.task_data['trial_start_accept_reject'] = self.experimenter_acceptance_of_trial
              
        super(ExG_FM_ARAT_CODA, self)._cycle()


    # def _start_wait(self):
    #     # determine the random length of time to stay in the rest state
    #     min_time, max_time = self.rest_interval
    #     self.rest_time = random.random() * (max_time - min_time) + min_time

    #     # if (self.experimenter_acceptance_of_trial in ['accept']):
    #     #     self.port.write('l')
    #     #     print "t sent rest"

    #     super(ExG_FM_ARAT_CODA, self)._start_wait()

    def _parse_next_trial(self):
        self.trial_type = self.next_trial
  
    def _start_trial(self):
        self.experimenter_acceptance_of_trial = ''
        print self.trial_type

    def _start_rest(self):
        self.experimenter_acceptance_of_trial = ''

    def _while_rest(self):
        self.image_fname = os.path.join(self.image_dir_general, 'rest.bmp')
        self.show_image(self.image_fname)
    
    def _while_trial(self):
        self.image_fname = os.path.join(self.image_dir_general, 'mov.bmp')
        self.show_image(self.image_fname)

    def _test_end_rest(self, *args, **kwargs):
        return self.experimenter_acceptance_of_trial == 'start'
        self.experimenter_acceptance_of_trial = ''

    def _test_accept_trial(self, *args, **kwargs):
        return self.experimenter_acceptance_of_trial == 'accept'
        self.experimenter_acceptance_of_trial = ''

    def _test_reject_trial(self, *args, **kwargs):
        return self.experimenter_acceptance_of_trial == 'reject'
        self.experimenter_acceptance_of_trial = ''


    def show_image(self, image_fname):

        window = pygame.display.set_mode(self.window_size)
        img = pygame.image.load(os.path.join(self.image_fname))
        img = pygame.transform.scale(img, self.window_size)

        window.blit(img, (0,0))
        pygame.display.flip()




class ExG_FM_3movs_CODA(RecordBrainAmpData, Sequence):
    '''
    Task to record ExG and send triggers to CODA to start recording and trial synchro triggers
    '''
    #needs to inherit from RecordBrainAmpData first to run the init of Autostart before than the init of Sequence
    fps = 20

    status = {
        'wait': {
            'start_trial': 'instruct_rest',
            'stop': None},
        'instruct_rest': {
            'end_instruct': 'rest',
            'stop':      None},            
        'rest': {
            'starts_trial' : 'instruct_go',
            'stop':      None},
        'instruct_go': {
            'end_instruct': 'trial',
            'stop':      None},
        'trial': {
            # 'end_trial': 'wait',
            'accept_trial': 'wait',
            'reject_trial': 'instruct_rest',
            'stop': None},      
    }
    state = 'wait'  # initial state

    channel_list_name = traits.OptionsList(*channel_list_options, bmi3d_input_options=channel_list_options) #choose always the filtered + raw option!!!!!!
    rest_interval = traits.Tuple((2., 4.), desc='Min and max time to remain in the rest state.')
    #add the windows size trait to be able to modifiy it manually
    window_size = traits.Tuple((1000, 560), desc='Size of window to display the plant position/angle')
    language = traits.OptionsList(*languages_list, bmi3d_input_options=languages_list)

    sequence_generators = ['FM_CODA_tasks']
  
    @staticmethod
    def FM_CODA_tasks(length=3, shoulder_flexion=1, hand_extension=1, spherical_grip=1):
        available_targets = []
        if shoulder_flexion: available_targets.append('shoulder_flexion')
        if hand_extension: available_targets.append('hand_extension')
        if spherical_grip: available_targets.append('spherical_grip')

        targets = available_targets*length
        return targets  

    def _play_sound(self, fname):
        pygame.mixer.music.load(fname)
        pygame.mixer.music.play()

    def __init__(self, *args, **kwargs):
        ## Init the pygame mixer for playing back sounds
        super(ExG_FM_3movs_CODA, self).__init__(*args, **kwargs)
        self.add_dtype('trial_type', np.str_, 40)
        self.add_dtype('ts',         'f8',    (1,))
        self.add_dtype('trial_start_accept_reject', np.str_, 10)
        pygame.mixer.init()

        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)

        self.image_dir_general = os.path.expandvars('$HOME/code/ismore/images')
        self.image_dir = os.path.join(self.image_dir_general, self.language)

        self.experimenter_acceptance_of_trial = ''

        self.port = serial.Serial('/dev/arduino_neurosync',baudrate=115200)
        # self.port.write('l')
        # self.port.write('p')
        # print "l sent init"

    def _cycle(self):
        '''Runs self.fps times per second.'''
        try:
            self.task_data['trial_type'] = self.trial_type
        except:
            ''
        self.task_data['ts']         = time.time()
        self.task_data['trial_start_accept_reject'] = self.experimenter_acceptance_of_trial
        self.experimenter_acceptance_of_trial = ''

        if (self.experimenter_acceptance_of_trial in ['accept', 'reject', 'start']):
            print self.experimenter_acceptance_of_trial

        # self.experimenter_acceptance_of_trial = ''
        super(ExG_FM_3movs_CODA, self)._cycle()



    def _test_end_instruct(self, *args, **kwargs):
        self.experimenter_acceptance_of_trial = ''
        return not pygame.mixer.music.get_busy()

    def _start_instruct_rest(self):
        self._play_sound(os.path.join(self.sounds_dir, 'rest.wav'))

        
        #send 10ms pulse for the end of the trial only if the trial has been accepted or rejected (do not send pulse in the rest period previous to first trial)
        # if (self.experimenter_acceptance_of_trial in ['reject']):
        #     self.port.write('l')
        #     print "l sent rest"
        # self.experimenter_acceptance_of_trial = ''

    def _start_wait(self):
        # determine the random length of time to stay in the rest state
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time

        # if (self.experimenter_acceptance_of_trial in ['accept']):
        #     self.port.write('l')
        #     print "t sent rest"

        super(ExG_FM_3movs_CODA, self)._start_wait()

    def _start_instruct_go(self):
        
        sound_fname = os.path.join(self.sounds_dir,'go.wav')
        self._play_sound(sound_fname)
        #send a 10ms pulse to trial pin
        # self.port.write('t')
        # print "t sent go"

    def _parse_next_trial(self):
        self.trial_type = self.next_trial

  
    def _start_trial(self):
        print self.trial_type

    def _while_instruct_rest(self):
        self.image_fname = os.path.join(self.image_dir_general, 'rest.bmp')
        self.show_image(self.image_fname)

    def _while_rest(self):
        self.image_fname = os.path.join(self.image_dir_general, 'rest.bmp')
        self.show_image(self.image_fname)
    
    def _while_instruct_trial_type(self):
        self.image_fname = os.path.join(self.image_dir_general, 'mov.bmp')
        self.show_image(self.image_fname)

    def _while_trial(self):
        self.image_fname = os.path.join(self.image_dir_general, 'mov.bmp')
        self.show_image(self.image_fname)

    def _test_starts_trial(self, *args, **kwargs):
        return self.experimenter_acceptance_of_trial == 'start'


    def _test_accept_trial(self, *args, **kwargs):
        return self.experimenter_acceptance_of_trial == 'accept'


    def _test_reject_trial(self, *args, **kwargs):
        return self.experimenter_acceptance_of_trial == 'reject'


    def show_image(self, image_fname):

        window = pygame.display.set_mode(self.window_size)
        img = pygame.image.load(os.path.join(self.image_fname))
        img = pygame.transform.scale(img, self.window_size)

        window.blit(img, (0,0))
        pygame.display.flip()

#dictionary with task descriptions for the Fugl-Meyer and ARAT measurement (6 movements)
FM_measurement_tasks = dict()

FM_measurement_tasks['english'] = dict()
FM_measurement_tasks['english']['A'] = 'Raise your arms'
FM_measurement_tasks['english']['B'] = 'Elbow and shoulder rotation'
FM_measurement_tasks['english']['C'] = 'Extension of the elbow'
FM_measurement_tasks['english']['D'] = 'Supination'
FM_measurement_tasks['english']['E'] = 'Wrist extension'
FM_measurement_tasks['english']['F'] = 'Finger extension'
FM_measurement_tasks['english']['rest'] = 'Rest'
FM_measurement_tasks['english']['ready'] = 'Ready'
FM_measurement_tasks['english']['steady'] = 'Steady'
FM_measurement_tasks['english']['go'] = 'Go!'

FM_measurement_tasks['deutsch'] = dict()
FM_measurement_tasks['deutsch']['A'] = 'Anheben des Oberarms'
FM_measurement_tasks['deutsch']['B'] = 'Aussendrehung im Schultergelenk'
FM_measurement_tasks['deutsch']['C'] = 'Streckung im Ellenbogen'
FM_measurement_tasks['deutsch']['D'] = 'Drehung im Unterarm'
FM_measurement_tasks['deutsch']['E'] = 'Anheben im Handgelenk'
FM_measurement_tasks['deutsch']['F'] = 'Fingerstreckung'
FM_measurement_tasks['deutsch']['rest'] = 'Entspannen'
FM_measurement_tasks['deutsch']['ready'] = 'Auf die Plaetze!'
FM_measurement_tasks['deutsch']['steady'] = 'Fertig!'
FM_measurement_tasks['deutsch']['go'] = 'los!'

FM_measurement_tasks['castellano'] = dict()
FM_measurement_tasks['castellano']['A'] = 'Levantar los brazos'
FM_measurement_tasks['castellano']['B'] = 'Rotacion externa de los hombros y codos'
FM_measurement_tasks['castellano']['C'] = 'Extension de los codos'
FM_measurement_tasks['castellano']['D'] = 'Supinacion'
FM_measurement_tasks['castellano']['E'] = 'Extension de las muinecas'
FM_measurement_tasks['castellano']['F'] = 'Extension de los dedos'
FM_measurement_tasks['castellano']['rest'] = 'Pausa'
FM_measurement_tasks['castellano']['ready'] = 'Preparados...'
FM_measurement_tasks['castellano']['steady'] = 'Listos...'
FM_measurement_tasks['castellano']['go'] = 'Ya!'

FM_measurement_tasks['euskara'] = dict()
FM_measurement_tasks['euskara']['A'] = 'Altxatu besoak'
FM_measurement_tasks['euskara']['B'] = 'Biratu sorbaldak eta ukalondoak'
FM_measurement_tasks['euskara']['C'] = 'Luzatu ukalondoak'
FM_measurement_tasks['euskara']['D'] = 'Supinazioa / Esku-azpiak goruntz'
FM_measurement_tasks['euskara']['E'] = 'Luzatu eskumuturrak'
FM_measurement_tasks['euskara']['F'] = 'Luzatu hatzak'
FM_measurement_tasks['euskara']['rest'] = 'Lasai'
FM_measurement_tasks['euskara']['ready'] = 'Adi...'
FM_measurement_tasks['euskara']['steady'] = 'Prest...'
FM_measurement_tasks['euskara']['go'] = 'Hasi!'

class ExG_FM_6movs_CODA(RecordBrainAmpData, Sequence):

    fps = 20

    status = {
        'wait': {
            'start_trial': 'instruct_rest',
            'stop': None},
        'instruct_rest': {
            'end_instruct_rest': 'rest',
            'stop':      None},            
        'rest': {
            'end_rest': 'instruct_trial_type',
            'stop':      None},
        'instruct_trial_type': {
            'end_instruct_trial_type': 'ready',
            'stop':      None},
        'ready': {
            'end_ready': 'steady',
            'stop':      None},
        'steady': {
            'end_steady': 'instruct_go',
            'stop':      None},
        'instruct_go': {
            'end_instruct_go': 'trial',
            'stop':      None},
        'trial': {
            'end_trial': 'wait',
            'stop':      None},    
    }
    state = 'wait'  # initial state

    rest_interval = traits.Tuple((4., 7.), desc='Min and max time to remain in the rest state.') 
    trial_time    = traits.Float(6,       desc='Time to remain in the trial state.') 
    instruct_trial_type_time = traits.Float(3,       desc='Time to remain in the trial state.') 

    language = traits.OptionsList(*languages_list, bmi3d_input_options=languages_list)
  
    sequence_generators = ['FM_6movs']

    
    @staticmethod
    def _make_block_rand_targets(length, available_targets, shuffle = False):
        targets = []
        for k in range(length):
            a_ = available_targets[:]
            if shuffle:
                random.shuffle(a_)
            targets += a_
        return targets

    @staticmethod
    def FM_6movs(length=8, A=1, B=1, C=1, D=1, E=1, F=1, shuffle = 1):
        available_targets = []
        if A: available_targets.append('A')
        if B: available_targets.append('B')
        if C: available_targets.append('C')
        if D: available_targets.append('D')
        if E: available_targets.append('E')
        if F: available_targets.append('F')

        targets = ExG_FM_6movs_CODA._make_block_rand_targets(length, available_targets, shuffle = shuffle)
        return targets  


    def __init__(self, *args, **kwargs):
        ## Init the pygame mixer for playing back sounds
        super(ExG_FM_6movs_CODA, self).__init__(*args, **kwargs)
        self.add_dtype('trial_type', np.str_, 40)
        self.add_dtype('ts',         'f8',    (1,))
        
        pygame.mixer.init(44100, -16, 4, 2048)

        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)

        self.image_dir_general = os.path.expandvars('$HOME/code/ismore/images')

   
        self.instruct_ready_time = 1
        self.instruct_steady_time = 1
        self.instruct_rest_time = 1
        pygame.init()

        #import here because when importing at the beginning of the script it gives an error if you stop an experiment and run it again without rerunning the server
        from gi.repository import Gdk, Gtk
        # Replace w with the GtkWindow of your application
        window = Gtk.Window()
        # Get the screen from the GtkWindow
        s = window.get_screen()

        # collect data about each monitor
        monitors = []
        nmons = s.get_n_monitors()
        for m in range(nmons):
            mg = s.get_monitor_geometry(m)
            monitors.append(mg)

        # Using the screen of the Window, the monitor it's on can be identified
        active_monitor = s.get_monitor_at_window(s.get_active_window())

        #considering 2 monitors connected
        if (active_monitor == 1):
            feedback_monitor = 0
        elif (active_monitor ==0):
            feedback_monitor =1

        #set the size of the window where the visual stimuli will be presented to the size of the screen
        self.window_size = [monitors[feedback_monitor].width, monitors[feedback_monitor].height]

    def _cycle(self):
        '''Runs self.fps times per second.'''
        try:
            self.task_data['trial_type'] = self.trial_type
        except:
            ''
        self.task_data['ts']         = time.time()

        super(ExG_FM_6movs_CODA, self)._cycle()


    def _display_text(self, text_display):
        
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (1,1)

        pygame.init()
        window = pygame.display.set_mode(self.window_size)

        # Fill background
        background = pygame.Surface(window.get_size())
        background = background.convert()
        background.fill((0, 0, 0))

        #add text
        font = pygame.font.Font(None, 48)
        text = font.render(text_display, 1, (255,255,255))
        textpos = text.get_rect()
        textpos.centerx = background.get_rect().centerx
        textpos.centery = background.get_rect().centery -200
        background.blit(text, textpos)
        window.blit(background, (0, 0))
        pygame.display.flip()


    def show_image_and_text(self, trial_type,language):

        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (1,1)
        window = pygame.display.set_mode(self.window_size)

        # Fill background
        background = pygame.Surface(window.get_size())
        background = background.convert()
        background.fill((0, 0, 0))

        #add text
        font = pygame.font.Font(None, 48)
        text_display = FM_measurement_tasks[language][trial_type]
        text = font.render(text_display, 1, (255,255,255))
        textpos = text.get_rect()
        textpos.centerx = background.get_rect().centerx
        textpos.centery = background.get_rect().centery -200       
        background.blit(text, textpos)
        window.blit(background, (0, 0))

        #add image
        image_fname1 = os.path.join(self.image_dir_general, self.trial_type + '1.bmp')
        image_fname2 = os.path.join(self.image_dir_general, self.trial_type + '2.bmp')
        image_fname3 = os.path.join(self.image_dir_general, self.trial_type + '3.bmp')
        img1 = pygame.image.load(os.path.join(image_fname1))
        img2 = pygame.image.load(os.path.join(image_fname2))
        img3 = pygame.image.load(os.path.join(image_fname3))

        new_x = int(window.get_size()[0]/3)
        new_y = int(window.get_size()[1]/3)


        img1 = pygame.transform.scale(img1, [new_x,new_y])
        img2 = pygame.transform.scale(img2, [new_x,new_y])
        img3 = pygame.transform.scale(img3, [new_x,new_y])

        window.blit(img1, (0,background.get_rect().centery))
        window.blit(img2, (new_x,background.get_rect().centery))
        window.blit(img3, (new_x*2,background.get_rect().centery))

        pygame.display.flip()


    def _test_end_instruct_rest(self, *args, **kwargs):
        return not self.chan_rest.get_busy()

    def _start_instruct_rest(self):
        rest_sound = pygame.mixer.Sound(os.path.join(self.sounds_dir, 'rest.wav'))
        self.chan_rest = pygame.mixer.find_channel()
        self.chan_rest.play(rest_sound)


    def _while_instruct_rest(self):
        text = FM_measurement_tasks[self.language]['rest']
        self._display_text(text)

    def _start_rest(self):
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time

    def _while_rest(self):
        text = FM_measurement_tasks[self.language][self.state]
        self._display_text('')

    def _test_end_rest(self, ts):
        return ts > self.rest_time  # and not self.pause -- needed?

    def _start_instruct_trial_type(self):
        background_sound = pygame.mixer.Sound(os.path.join(self.sounds_general_dir, self.trial_type +'.wav'))
        self.chan_background = pygame.mixer.find_channel()
        self.chan_background.play(background_sound)

    def _while_instruct_trial_type(self):
        self.show_image_and_text(self.trial_type,self.language)

    def _test_end_instruct_trial_type(self, ts):
        return ts > self.instruct_trial_type_time  

    def _while_ready(self):
        text = FM_measurement_tasks[self.language][self.state]
        self._display_text(text)

    def _test_end_ready(self, ts):
        return ts > self.instruct_ready_time  

    def _while_steady(self):
        text = FM_measurement_tasks[self.language][self.state]
        self._display_text(text)

    def _test_end_steady(self, ts):
        return ts > self.instruct_steady_time  

    def _start_instruct_go(self):

        go_sound = pygame.mixer.Sound(os.path.join(self.sounds_dir,'go.wav'))
        self.chan_go = pygame.mixer.find_channel()
        self.chan_go.play(go_sound)
        text = FM_measurement_tasks[self.language]['go']
        self._display_text(text)

    def _test_end_instruct_go(self, *args, **kwargs):
        return not self.chan_go.get_busy()

    def _parse_next_trial(self):
        self.trial_type = self.next_trial

    def _start_trial(self):
        print self.trial_type

    def _test_end_trial(self, ts):
        return ts > self.trial_time

    def _while_trial(self):
        self.show_image_and_text(self.trial_type,self.language)

class EMG_SynergiesTasks(RecordBrainAmpData, Sequence):
    '''
    Task to record EMG data
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
            'starts_trial' : 'instruct_go',
            'stop':      None},
        'instruct_go': {
            'end_instruct': 'trial',
            'stop':      None},
        'trial': {
            # 'end_trial': 'wait',
            'accept_trial': 'wait',
            'reject_trial': 'instruct_rest',
            'stop': None},      
    }
    state = 'wait'  # initial state

    channel_list_name = traits.OptionsList(*channel_list_options, bmi3d_input_options=channel_list_options) #choose always the filtered + raw option!!!!!!
    rest_interval = traits.Tuple((2., 4.), desc='Min and max time to remain in the rest state.')
    window_size = traits.Tuple((1000, 560), desc='Size of window to display the plant position/angle')
    language = traits.OptionsList(*languages_list, bmi3d_input_options=languages_list)

    sequence_generators = ['Synergies_hand_objects']
  
    @staticmethod
    def Synergies_hand_objects(length=3, name=1, bottle=1, cup=1, plate=1, pencil=1, neddle=1):
        available_targets = []
        if name: available_targets.append('name')
        if bottle: available_targets.append('bottle')
        if cup: available_targets.append('cup')
        if plate: available_targets.append('plate')
        if pencil: available_targets.append('pencil')
        if neddle: available_targets.append('neddle')

        targets = available_targets*length
        return targets  

    def _play_sound(self, fname):
        pygame.mixer.music.load(fname)
        pygame.mixer.music.play()

    def __init__(self, *args, **kwargs):
        ## Init the pygame mixer for playing back sounds
        super(EMG_SynergiesTasks, self).__init__(*args, **kwargs)
        self.add_dtype('trial_type', np.str_, 40)
        self.add_dtype('ts',         'f8',    (1,))
        self.add_dtype('trial_start_accept_reject_grasp', np.str_, 10)
        pygame.mixer.init()

        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)

        self.image_dir_general = os.path.expandvars('$HOME/code/ismore/images')
        self.image_dir = os.path.join(self.image_dir_general, self.language)

        self.experimenter_acceptance_of_trial = ''

       

    def _cycle(self):
        '''Runs self.fps times per second.'''
        try:
            self.task_data['trial_type'] = self.trial_type
        except:
            ''
        self.task_data['ts']         = time.time()
        self.task_data['trial_start_accept_reject_grasp'] = self.experimenter_acceptance_of_trial
        
        if (self.experimenter_acceptance_of_trial in ['accept', 'reject', 'start','grasp']):
            print self.experimenter_acceptance_of_trial

            
        super(EMG_SynergiesTasks, self)._cycle()



    def _test_end_instruct(self, *args, **kwargs):
        self.experimenter_acceptance_of_trial = ''
        return not pygame.mixer.music.get_busy()

    def _start_instruct_rest(self):
        self._play_sound(os.path.join(self.sounds_dir, 'rest.wav'))

        
        #send 10ms pulse for the end of the trial only if the trial has been accepted or rejected (do not send pulse in the rest period previous to first trial)
        # if (self.experimenter_acceptance_of_trial in ['reject']):
        #     self.port.write('l')
        #     print "l sent rest"
        # self.experimenter_acceptance_of_trial = ''


    def _start_wait(self):
        # determine the random length of time to stay in the rest state
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time

        # if (self.experimenter_acceptance_of_trial in ['accept']):
        #     self.port.write('l')
        #     print "t sent rest"

        super(EMG_SynergiesTasks, self)._start_wait()

    def _start_instruct_go(self):
        
        sound_fname = os.path.join(self.sounds_dir,'go.wav')
        self._play_sound(sound_fname)
        #send a 10ms pulse to trial pin
        # self.port.write('t')
        # print "t sent go"

    def _parse_next_trial(self):
        self.trial_type = self.next_trial

  
    def _start_trial(self):
        print self.trial_type

    def _while_instruct_rest(self):
        self.image_fname = os.path.join(self.image_dir_general, 'rest.bmp')
        self.show_image(self.image_fname)

    def _while_rest(self):
        self.image_fname = os.path.join(self.image_dir_general, 'rest.bmp')
        self.show_image(self.image_fname)
    
    def _while_instruct_trial_type(self):
        self.image_fname = os.path.join(self.image_dir_general, 'mov.bmp')
        self.show_image(self.image_fname)

    def _while_trial(self):
        self.image_fname = os.path.join(self.image_dir_general, 'mov.bmp')
        self.show_image(self.image_fname)

    def _test_starts_trial(self, *args, **kwargs):
        return self.experimenter_acceptance_of_trial == 'start'


    def _test_accept_trial(self, *args, **kwargs):
        return self.experimenter_acceptance_of_trial == 'accept'


    def _test_reject_trial(self, *args, **kwargs):
        return self.experimenter_acceptance_of_trial == 'reject'



    def show_image(self, image_fname):

        window = pygame.display.set_mode(self.window_size)
        img = pygame.image.load(os.path.join(self.image_fname))
        img = pygame.transform.scale(img, self.window_size)

        window.blit(img, (0,0))
        pygame.display.flip()


class Active_Movements(PlantControlBase):
    '''Record neural and exg data, plant replaced by dummy plant'''
    
    
    exclude_parent_traits = ['plant_type','max_attempt_per_targ','reward_time', 'rand_start', 'simulate', 'show_environment', 'arm_side']
    plant_type = 'DummyPlant'
    max_attempt_per_targ = 1
    
    is_bmi_seed = True
    def __init__(self, *args, **kwargs):
        super(Active_Movements, self).__init__(*args, **kwargs)
        self.experimenter_acceptance_of_trial = ''
        self.add_dtype('trial_accept_reject', np.str_, 10)
        print 'Active Movements recording'

    def move_plant(self):
        '''Do nothing here -- plant is moved manually.'''
        pass

    def verify_plant_data_arrival(self,n_secs):
        pass

    def _cycle(self):
        self.task_data['trial_accept_reject'] = self.experimenter_acceptance_of_trial
        super(Active_Movements, self)._cycle()



class Mirror_Therapy_Movements(Active_Movements):

    max_attempt_per_targ = 1
    sequence_generators = ['mirror_therapy_movements_blk_new']
    exclude_parent_traits = ['decoder','blocking_opts', 'ignore_correctness_jts', 'safety_grid_file', 'target_radius_x', 'target_radius_y', 'targets_matrix', 'tol_deg_fing','tol_deg_pron','tol_deg_psi','tol_deg_thumb','hold_time','plant_type','max_attempt_per_targ','reward_time', 'rand_start', 'simulate', 'show_environment', 'arm_side']

    def __init__(self, *args, **kwargs):
        super(Mirror_Therapy_Movements, self).__init__(*args, **kwargs)
        print 'Mirror therapy Movements recording'

    def _parse_next_trial(self):
        self.trial_type = self.next_trial
        self.chain_length = 1
        print 'target index : ', self.target_index

    def _start_instruct_trial_type(self):
        self._play_sound(self.sounds_dir, ['go'])
        self.target_index += 1

    def _while_instruct_trial_type(self):
        pass

    def _test_end_instruct(self, *args, **kwargs):
        return not pygame.mixer.music.get_busy()




