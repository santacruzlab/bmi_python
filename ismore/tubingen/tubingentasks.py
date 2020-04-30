''' Basic and Exo-related tasks specific for Tubingen experiments '''

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
from scipy.stats import threshold



from utils.constants import *

from ismore.ismoretasks import * #nerea: to delete once everything extrictly needed is imported
# from ismore.ismoretasks import NonInvasiveBase, RecordBrainAmpData, RecordTrajectoriesBase

from ismore.tubingen import brainamp_channel_lists

# new for sleep task#
from ismore.brainamp import rda

from ismore.invasive.bmi_ismoretasks import *

#nerea: check if this is properly imported
# from ismore.noninvasive.exg_tasks import EEGMovementDecodingNew
from ismore.tubingen.noninvasive_tubingen.exg_tasks_tubingen import EEGMovementDecodingNew
from ismore.invasive.bmi_ismoretasks import PlantControlBase
from ismore.noninvasive.exg_tasks import EMGEndPointMovement
from ismore.noninvasive.exg_tasks import EXGEndPointMovement

from ismore.tubingen.noninvasive_tubingen.emg_decoding import LinearEMGDecoder

###### Options to select in interface ######
plant_type_options  = ['IsMore','ArmAssist', 'ReHand']
# nerea : are you using these 2 variables? What DoFs are controlled with BMI-EMG? How is the reward given? What DoFs need to be reached to accomplish the task?
DoF_control_options = ['IsMore','ArmAssist', 'ReHand', 'ReHand-Pronosup', 'ReHand-Pronosup-FingersDisabled']
DoF_target_options  = ['IsMore','ArmAssist', 'ReHand', 'ReHand-Pronosup', 'ReHand-Pronosup-FingersDisabled']

arm_side_options = ['left','right']
languages_list = ['english', 'deutsch', 'castellano', 'euskara'] # nerea: do you need all these?
speed_options = ['very-low','low', 'medium','high'] #nerea: what speed are you using in the compliant calibration sessions?
channel_list_options = brainamp_channel_lists.channel_list_options #nerea: we should try to create a separate channels_list in tubingen folder


####################################   DEFINITIONS   ---------------------------------------------- 

###### Colors ######
# nerea: you might not need to define the colors for specific trial types.
COLORS = {
    'black': (0, 0, 0, 1),
    'red':   (1, 0, 0, 1),
    'grasp':   (1, 0, 0, 1),
    'pinch':   (1, 0, 0, 1),
    'green': (0, 1, 0, 1),
    'point': (0, 1, 0, 1),
    'blue':  (0, 0, 1, 1),
    'up':  (0, 0, 1, 1),
    'rest': (1, 1, 1, 1),
    'white': (1, 1, 1, 1),
    'magenta': (0, 1, 0, 0), 
    'brown': (.36, .2, .09, 1),
    'yellow': (0, 0, 1, 0),
    'down':   (1, 0, 0, 1),
    'alert_state_ok': (.4, .8, 0, 1),
    'alert_state_failed': (1, 0, 0, 1)
}

# nerea: is this task working there? not here. are you using it?
class RecordB1(RecordBrainAmpData,RecordTrajectoriesBase):
    '''Task class for recording trajectories for the center-out task.'''

    status = {
        'wait': {
            'start_trial': 'rest',
            'stop': None},
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
    state = 'wait'  # initial state
    # state = 'rest'  # initial state

    # settable parameters on web interface
    rest_interval = traits.Tuple((2., 4.), desc='Min and max time to remain in the rest state.')
    ready_time    = traits.Float(2,        desc='Time to remain in the ready state.')
    trial_time    = traits.Float(10,       desc='Time to remain in the trial state.')
    plant_type = traits.OptionsList('ArmAssist', bmi3d_input_options=plant_type_options, desc='Device connected, data will be acquired from this plant')

    # def init(self):
    #     super(RecordB1, self).init()
    #     import socket
    #     self.UDP_IP = '192.168.137.3'       
    #     self.UDP_PORT = 6000
    #     MESSAGE = "start recording"
    #     self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #     self.sock.sendto(MESSAGE, (self.UDP_IP, self.UDP_PORT))
    #     print "------------------------------------------------------------------start recording"        

    def _start_wait(self):
        import socket
        self.UDP_IP = '192.168.137.3'       
        self.UDP_PORT = 6000
        MESSAGE = "start recording in wait"
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.sendto(MESSAGE, (self.UDP_IP, self.UDP_PORT))
        print "------------------------------------------------------------------start recording in WAIT state"        
        
        # determine the random length of time to stay in the rest state
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time

        super(RecordB1, self)._start_wait()
    
    #this function runs at the beginning, in the wait state so that the trial_type is already
    def _parse_next_trial(self): 
        self.trial_type = self.next_trial

    def _play_sound(self, fname):
        pygame.mixer.music.load(fname)
        pygame.mixer.music.play()

    def _start_rest(self):
        filename = os.path.join(self.sounds_dir, 'rest.wav')
        self._play_sound(filename)

    def _end_trial(self):
        # get the next trial type in the sequence
        try:
            self.trial_type = self.gen.next()
            # self.trial_type = self.next_trial

        except StopIteration:
            self.end_task()
        
    def _test_end_rest(self, ts):
        return ts > self.rest_time  # and not self.pause -- needed?

    def _start_ready(self):
        filename = os.path.join(self.sounds_dir, self.trial_type + '.wav')
        self._play_sound(filename)

    def _test_end_ready(self, ts):
        return ts > self.ready_time

    def _start_trial(self):
        print self.trial_type
        filename = os.path.join(self.sounds_dir,'go.wav')
        self._play_sound(filename)



    def _test_end_trial(self, ts):
        return ts > self.trial_time

    def cleanup(self, database, saveid, **kwargs):
        MESSAGE = "stop recording"
        self.sock.sendto(MESSAGE, (self.UDP_IP, self.UDP_PORT))
        print "--------------------------------------------------------------------stop recording"
        super(RecordB1,self).cleanup(database, saveid, **kwargs)


# nerea: is this task running using a plant different from AA? not here.
class Record_MuscleFatigue_PassiveBase(RecordBrainAmpData,RecordTrajectoriesBase):
    '''Task class for recording trajectories for the center-out task.'''

    status = {
        'wait': {
            'start_trial': 'trial',
            'stop': None},
        'trial': {
            'end_trial': 'trial_return',
            'stop':      None},
        'trial_return': {
            'end_trial_return': 'wait',
            'stop':      None},
    }
    state = 'wait'  # initial state


    # settable parameters on web interface
    trial_time    = traits.Float(2,       desc='Time to remain in the trial state.')
    trial_return_time    = traits.Float(3,       desc='Time to remain in the trial_return state.')
    plant_type = traits.OptionsList('ArmAssist', bmi3d_input_options=plant_type_options, desc='Device connected, data will be acquired from this plant')
    
    #this function runs at the beginning, in the wait state so that the trial_type is already
    def _parse_next_trial(self): 
        self.trial_type = self.next_trial

    def _play_sound(self, fname):
        pygame.mixer.music.load(fname)
        pygame.mixer.music.play()

    def _end_trial(self):
        # get the next trial type in the sequence
        try:
            self.trial_type = self.gen.next()
            # self.trial_type = self.next_trial

        except StopIteration:
            self.end_task()
        
    def _start_trial(self):
        print self.trial_type
        filename = os.path.join(self.sounds_dir,'go.wav')
        self._play_sound(filename)

    def _test_end_trial(self, ts):
        return ts > self.trial_time

    def _start_trial_return(self):
        print self.trial_type
        filename = os.path.join(self.sounds_dir,'back.wav')
        self._play_sound(filename)

    def _test_end_trial_return(self, ts):
        return ts > self.trial_return_time



# nerea: works OK (apart from FES part that we have not connected here)
class Sleep_1D_EEG_BCI_exo_FES_task(EEGMovementDecodingNew): 
    status = {
        'wait': {
            'start_trial': 'instruct_to_start',
            'stop': None},
        'instruct_to_start':{
            'end_instruct' : 'drive_to_start',
            'stop' : None},
        'drive_to_start': {
            'at_start' : 'instruct_rest',
            'stop': None},
        'instruct_rest': {
            'end_instruct': 'rest'},
        'rest': {
            'end_rest': 'instruct_go',
            'stop': None},   
        'instruct_go': {
            'end_instruct': 'trial'},
        'trial': {
            'trial_complete': 'reward',
            'timeout': 'timeout_penalty',
            'stop': None},  
        'reward' : {
            'reward_end': 'drive_to_start',
            'stop': None},
        'timeout_penalty': {
            'timeout_penalty_end': 'drive_to_start',
            'stop': None},
        }
    state = 'wait'
    sequence_generators = ['sleep_task_gen', "B1_targets_sleep"]
    music_feedback = 0
    exclude_parent_traits = ['give_feedback', 'music_feedback', 'preparation_time', 'rand_start', 'show_environment', 'session_length', 'show_FB_window']

    @staticmethod
    def sleep_task_gen(length=100, green=1):
        return ['green']*length

    @staticmethod
    def B1_targets_sleep(length=5, red=1, green = 1, blue=1, shuffle=1):
        available_targets = []
        if red: available_targets.append('red')
        if green: available_targets.append('green')
        if blue: available_targets.append('blue')

        targets = NonInvasiveBase._make_block_rand_targets(length, available_targets, shuffle=shuffle)
        return targets

    def __init__(self, *args, **kwargs):
        super(Sleep_1D_EEG_BCI_exo_FES_task, self).__init__(*args, **kwargs)

        # Create serial port for FES communication
        import glob
        import serial
        portList = glob.glob("/dev/ttyUSB*")

        if len(portList) == 1:
            portNameFES = portList[0]
        elif len(portList) > 1:
            print "Several USB devices connected besides FES system "
            raise Exception
        elif len(portList) == 0:
            print "FES deviced is NOT connected "
            raise Exception

        print " FES port name :  ", portNameFES
        self.ser = serial.Serial(portNameFES, baudrate=921600)

        if not self.ser.isOpen():
            self.ser.open()


    def init(self):
        super(Sleep_1D_EEG_BCI_exo_FES_task, self).init()

    def _cycle(self):
        super(Sleep_1D_EEG_BCI_exo_FES_task, self)._cycle()
        
        self.fes_stimulation()

    def fes_stimulation(self):
        if self.state  == 'trial':
            # FES stimulation
            if self.state_decoder == 1:
                self.ser.write(">S<")
            elif self.state_decoder == 0:
                self.ser.write(">R<")

    def _start_instruct_to_start(self):
        self._play_sound(self.sounds_dir, ['back'])

    def _test_end_instruct_to_start(self, ts):
        return not pygame.mixer.music.get_busy() 

    def _start_drive_to_start(self):
        print 'start drive to start'
        self.goal_position = self.targets_matrix['rest'][self.goal_idx][self.pos_states]
        
    def _while_drive_to_start(self):        
        self.goal_position = self.targets_matrix['rest'][self.goal_idx][self.pos_states]

    def _test_at_start(self,ts):
        in_target  = np.all(np.abs(self.pos_diff(self.goal_position[self.DoF_target_idx_init:self.DoF_target_idx_end],self.plant.get_pos()[self.DoF_target_idx_init:self.DoF_target_idx_end])) < self.target_margin[self.pos_states[self.DoF_target_idx_init:self.DoF_target_idx_end]])
        return in_target

    def _test_trial_complete(self,ts):
        return self.reached_goal_position 

    def _start_reward(self):
        self.ser.write(">R<") # stop FES stimulation
        self._play_sound(self.sounds_general_dir, ['beep'])

    def _test_reward_end(self,ts):
        return not pygame.mixer.music.get_busy() 

    def _start_timeout_penalty(self):
        self.ser.write(">R<") # stop FES stimulation
        self._play_sound(self.sounds_general_dir, ['beep4'])

    def _test_timeout_penalty_end(self,ts):
        return not pygame.mixer.music.get_busy() 

    # In order to show the simplified exo visualization, comment init_plant_display, update_plant_display and display_targets functions here
#    def init_plant_display(self):
#       pass
#
#    def update_plant_display(self):
#        pass
#
#    def display_targets(self):
#        pass
#
#    # In order to show the output of the EEG classifier, comment the init_show_decoder_output and update_decoder_ouput
#    def init_show_decoder_output(self):
#        pass
#
#    def update_decoder_ouput(self):
#        pass


# nerea: works OK
class EEG_Screening_Tue(RecordBrainAmpData, Sequence):
    ''' Represents the EEG only screening for the hybrid BMI experiments in 2018.
    The patient is instructed to (try to) open one hand actively and then relax.
    The "go" cue is eliminated from the task.'''
    # needs to inherit from RecordBrainAmpData first to run the init of Autostart before than the init of Sequence
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
            'end_preparation': 'trial',
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
    def OPEN_CLOSE_targets(length=8, right=1, left=1, close_eyes=0, relax=1, shuffle = 1):
        available_targets = []
        if right: available_targets.append('right')
        if left: available_targets.append('left')
        if relax: available_targets.append('relax')
        if close_eyes: available_targets.append('close_eyes')

        targets = EEG_Screening_Tue._make_block_rand_targets(length, available_targets, shuffle = shuffle)
        return targets

    def _play_sound(self, fname):
        pygame.mixer.music.load(fname)
        pygame.mixer.music.play()

    def show_image(self, image_fname):
        ''' 
        Handles image loading, resizing and displaying the image.
        '''
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (self.monitor_offset)

        window = pygame.display.set_mode(self.window_size,pygame.NOFRAME)
        
        img = pygame.image.load(os.path.join(self.image_fname))
        img = pygame.transform.scale(img, (320,240))

        # Place image at the center of the screen
        window.blit(img, (0.5*self.window_size[0]-160,0.5*self.window_size[1]-120))
        pygame.display.flip()

    def __init__(self, *args, **kwargs):
        # Init the pygame mixer for playing back sounds
        super(EEG_Screening_Tue, self).__init__(*args, **kwargs)
        self.add_dtype('trial_type', np.str_, 40)
        self.add_dtype('ts',         'f8',    (1,))
        pygame.mixer.init()

        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)

        self.image_dir_general = os.path.expandvars('$HOME/code/ismore/images')
        self.image_dir = os.path.join(self.image_dir_general, self.language)
        
        self.channels = [chan + '_filt' for chan in self.brainamp_channels]

        # Get information on the connected screens
        # (The following code is only for INFORMATION on the screens. The display of visual cues is actually performed using pygame)
        # Import here because when importing at the beginning of the script it gives an error if you stop an experiment and run it again without rerunning the server
        from gi.repository import Gdk, Gtk
        window = Gtk.Window()
        s = window.get_screen() # Get the screen from the GtkWindow

        # Collect data about each monitor
        self.monitors = []
        nmons = s.get_n_monitors()
        for m in range(nmons):
            mg = s.get_monitor_geometry(m)
            self.monitors.append(mg)

        # Using the screen of the Window, the monitor it is on can be identified
        self.active_monitor = s.get_monitor_at_window(s.get_active_window())

        # Find out the feedback monitor depending on the number of monitors and set the offset for displaying the visual cues
        self.monitor_offset = (0,0)
        if nmons == 2:
            # Considering 2 monitors connected
            if (self.active_monitor == 1):
                self.feedback_monitor = 0
            elif (self.active_monitor == 0):
                self.feedback_monitor = 1
                self.monitor_offset = (self.monitors[1].width, 0)
        elif nmons == 3:
            # Specifically for the setup in Tuebingen
            self.feedback_monitor = 2
            self.monitor_offset = (self.monitors[0].width + self.monitors[1].width, 0)
        else:
            self.feedback_monitor = 0
        print "Monitor used for feedback: ", self.feedback_monitor

        # Set the size of the window where the visual stimuli will be presented to the size of the screen
        self.window_size = [self.monitors[self.feedback_monitor].width ,self.monitors[self.feedback_monitor].height]  

    def _cycle(self):
        '''Runs self.fps times per second.'''
        try:
            self.task_data['trial_type'] = self.trial_type
        except:
            ''
        self.task_data['ts']         = time.time()

        super(EEG_Screening_Tue, self)._cycle()

    def _test_end_instruct(self, *args, **kwargs):
        return not pygame.mixer.music.get_busy()

    def _start_instruct_rest(self):
        self._play_sound(os.path.join(self.sounds_dir, 'relax.wav'))

    def _start_rest(self):
        # Determine the random length of time to stay in the rest state
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time

    def _test_end_preparation(self, ts):
        return ts > self.preparation_time

    #do we also need an auditory cue for the trial tasks or just visual? for now set to play Go cue
    def _start_instruct_trial_type(self):
        # Replace "left" and "right" by "open"; Replace "relax" by "rest"
        if self.trial_type == 'left':
            auditoryCue = 'open'
        elif self.trial_type == 'right':
            auditoryCue = 'open'
        elif self.trial_type == 'relax':
            auditoryCue = 'rest'
        else:
            auditoryCue = self.trial_type
        sound_fname = os.path.join(self.sounds_dir, auditoryCue + '.wav')
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


class HybridBCI(EEGMovementDecodingNew):
    '''
    verify_plant_data_arrival is added to this class the gipsy way.
    '''


    # Settable parameters on web interface for the EMG decoder  
    music_feedback    = traits.Int((0), desc=' 0 if we do not want to include music, 1 if we want different classical music pieces with increasing intensity to be played')
    gamma             = traits.Float(0.5,   desc='Gamma value for incorporating EMG decoded velocity.')
    emg_decoder       = traits.InstanceFromDB(LinearEMGDecoder, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='emg_decoder'))
    beep_on_fail      = traits.Int((0), desc=' 0 if a beep should be played when no ArmAssist or ReHand data has arrived.')

    def __init__(self,*args, **kwargs):

        super(HybridBCI, self).__init__(*args,**kwargs)     

        # Create EMG extractor object (its 'source' will be set later in the init method)
        if self.emg_decoder.plant_type != self.plant_type:
            print 'Chosen plant_type on the interface does not match the plant type used to train the decoder. Make sure you are selecting the right one'
        
        self.emg_channels = self.emg_decoder.extractor_kwargs['emg_channels']
        self.eeg_channels = self.eeg_extractor_kwargs['eeg_channels']
        self.channels = self.emg_channels + self.eeg_channels

        self.emg_decoder_extractor = self.emg_decoder.extractor_cls(None, 
            emg_channels = self.emg_decoder.extractor_kwargs['emg_channels'],
            feature_names = self.emg_decoder.extractor_kwargs['feature_names'],
            feature_fn_kwargs = self.emg_decoder.extractor_kwargs['feature_fn_kwargs'],
            win_len=self.emg_decoder.extractor_kwargs['win_len'],
            fs=self.emg_decoder.extractor_kwargs['fs'])

        self.emg_decoder_name = self.emg_decoder.decoder_name

        self.nstates_decoder = len(self.emg_decoder.beta.keys())

        self.add_dtype('emg_decoder_features',    'f8', (self.emg_decoder_extractor.n_features,))
        self.add_dtype('emg_decoder_features_Z',  'f8', (self.emg_decoder_extractor.n_features,))
        self.add_dtype('emg_decoder_features_mn','f8', (self.emg_decoder_extractor.n_features,))
        self.add_dtype('emg_decoder_features_std','f8', (self.emg_decoder_extractor.n_features,))
        # to save all the kinematics estimated by the emg decoder even if a less DoF plant is being used online. At least the ones without applying the lpf
        self.add_dtype('emg_vel',         'f8', (self.nstates_decoder,))
        self.add_dtype('predefined_vel',  'f8', (len(self.vel_states),))
        self.add_dtype('emg_vel_lpf',     'f8', (len(self.vel_states),))
        self.add_dtype('gamma_used', float, (1,))
        
        # Create buffer that is used for calculating/updating the mean and the std of EMG features online
        self.features_buffer = RingBuffer(
            item_len=self.emg_decoder_extractor.n_features,
            capacity=60*self.fps
        )

        # for low-pass filtering decoded EMG velocities
        self.emg_vel_buffer = RingBuffer(
            item_len=len(self.vel_states),
            capacity=10
        )
        
        # Check if task went to a state during which no data from the robot was received
        if self.plant_type == 'IsMore':
            self.data_not_arrived_state = {'ArmAssist': False, 'ReHand': False}

        self.sounds_dir_classical = os.path.expandvars('$HOME/code/ismore/sounds/classical')
        pygame.mixer.init()

    def init(self):
        super(HybridBCI,self).init()
        self.emg_decoder_extractor.source = self.brainamp_source
        
        self.alert_state_indicators = {}
        self.alert_state_indicators['ArmAssist'] = Circle(np.array([0, 0]), 3, COLORS['alert_state_ok'])
        self.alert_state_indicators['ReHand'] = Line(np.array([5, 0]), 5, 10, 0, COLORS['alert_state_ok'])
        
        self.add_model(self.alert_state_indicators['ArmAssist'])
        self.add_model(self.alert_state_indicators['ReHand'])

    def move_plant(self):
        '''Decodes EEG and EMG and sends velocity commands to the robot.
        Usually called in every cycle.'''

        command_vel  = pd.Series(0.0, self.vel_states)
        command_vel_raw  = pd.Series(0.0, self.vel_states)
        emg_vel  = pd.Series(0.0, self.vel_states) 
        emg_vel_lpf  = pd.Series(0.0, self.vel_states)
        predefined_vel  = pd.Series(0.0, self.vel_states) 
        command_vel_final  = pd.Series(0.0, self.vel_states)
        #TODO Replace the following line by the real subject randomization value
        subject_group     = 1

        # ??? What are vel_states?

        #
        # EMG feature extraction and normalization
        #

        emg_decoder_features = self.emg_decoder_extractor() # emg_features is of type 'dict'
        self.features_buffer.add(emg_decoder_features[self.emg_decoder_extractor.feature_type])
        
        if self.features_buffer.num_items() > 60 * self.fps:
            # If there is more than 60 seconds of recent EMG data, calculate mean and std from this data
            recent_features = self.features_buffer.get_all()
            self.features_mean_emg_dec = np.mean(recent_features, axis=1)

            if hasattr(self.emg_decoder, 'fixed_var_scalar') and self.emg_decoder.fixed_var_scalar:
                self.features_std_emg_dec  = self.emg_decoder.recent_features_std
            else:
                self.features_std_emg_dec = np.std(recent_features, axis=1)
        else:
            # ??? Why try/except?
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
            # Prevent standard deviation from being 0 (?)
            self.features_std_emg_dec[self.features_std_emg_dec == 0] = 1
        except:
            pass

        # Compute z-score of the EMG features
        emg_decoder_features_Z = (emg_decoder_features[self.emg_decoder_extractor.feature_type] - self.features_mean_emg_dec) / self.features_std_emg_dec 
        # Call EMG decoder using the z-score of the EMG features as input
        emg_vel = self.emg_decoder(emg_decoder_features_Z)

        # ???
        self.emg_vel_buffer.add(emg_vel[self.vel_states])

        n_items = self.emg_vel_buffer.num_items()
        buffer_emg = self.emg_vel_buffer.get(n_items)
        win = min(9,n_items)
        weights = np.arange(1./win, 1 + 1./win, 1./win)
        # ??? What is emg_vel_lpf? --> low-pass filtered?
        try:
            emg_vel_lpf = np.sum(weights*buffer_emg[:,n_items-win:n_items+1], axis = 1)/np.sum(weights)
        except:
            pass

        # Store EMG features and velocities in task data
        self.task_data['emg_decoder_features'] = emg_decoder_features[self.emg_decoder_extractor.feature_type]
        self.task_data['emg_decoder_features_Z'] = emg_decoder_features_Z
        self.task_data['emg_decoder_features_mn'] = self.features_mean_emg_dec
        self.task_data['emg_decoder_features_std'] = self.features_std_emg_dec
        self.task_data['emg_vel'] = emg_vel.values
        self.task_data['emg_vel_lpf'] = emg_vel_lpf
                  
        #
        # EEG feature extraction and normalization
        #

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
            if self.state in ['trial','trial_return']:
                if self.artifact_rejection == 1: 
                    if rejected_window == 0:
                        self.mov_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
                elif self.artifact_rejection == 0:
                    self.mov_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
                self.task_data['eeg_features_mov_buffer'] = eeg_features
            elif self.state in ['rest','rest_return']:
                if self.artifact_rejection == 1:
                    if rejected_window == 0:
                        self.rest_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
                elif self.artifact_rejection == 0:
                    self.rest_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
                    
                self.task_data['eeg_features_rest_buffer'] = eeg_features

        self.task_data['eeg_features'] = eeg_features
        self.task_data['eeg_mean_features'] = mean_feat
        self.task_data['eeg_std_features'] = std_feat
        
        # Normalize EEG features
        # eeg_features = (eeg_features - mean_feat.reshape(-1,1))/ std_feat.reshape(-1,1)
        eeg_features = (eeg_features - mean_feat)/ std_feat
        # mean_feat.ravel()
        self.task_data['eeg_features_Z'] = eeg_features

        # ??? Why try/except
        try:
            eeg_features = np.reshape(eeg_features,(1,eeg_features.shape[0]))
        except:
            pass
        
        # Call EEG decoder using the z-score of the EMG features as input
        self.decoder_output = self.eeg_decoder(eeg_features)
        # ??? Why is there a field and a method called eeg_decoder()?
        self.probability = self.eeg_decoder.decoder.predict_proba(eeg_features)
        
        #
        # Move plant according to the output of the EEG & EMG decoder
        #

        # Set velocity to zero if the task is in a non-moving state
        if self.state not in ['trial', 'trial_return']:
            command_vel[:] = 0
            self.state_decoder = 0
            
        # Compute the predefined velocity independently of the robot having to move or not. 
        # Just to have a measure of how the correct veloctiy would be at all times.
        # ??? Is this a todo or is it down here? Does not make sense.
        else:
            current_pos = self.plant_pos[:].ravel()
            current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
            target_state = np.hstack([self.goal_position, np.zeros_like(current_pos), 1]).reshape(-1, 1)
            assist_output = self.assister(current_state, target_state, 1)

            # ??? What is assister? What is Bu?                           
            Bu = np.array(assist_output["x_assist"]).ravel()
            #Bu = np.array(assist_output['Bu']).ravel()
            predefined_vel[:] = Bu[len(current_pos):len(current_pos)*2]

            # EEG gating

            # Check if current and previous output of the EEG decoder are "move"
            if self.decoder_output == 1 and self.prev_output == 1:
                # Increment "move" counter by 1
                self.consec_mov_outputs +=1
                # ??? What is state_decoder?
                if self.consec_mov_outputs == 5 and self.state_decoder == 0:
                    self.consec_rest_outputs = 0
            # If the current output of the EEG decoder is "move" and the previous output was "rest"
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
    
            # If the output of the decoder was "move" for five or more times
            if self.consec_mov_outputs >= 5:

                self.state_decoder = 1
                # Combine the velocity of the EMG decoder and the playback velocity into one velocity command
                norm_playback_vel = np.linalg.norm(predefined_vel)
                epsilon = 1e-6
                # If the norm of the playback velocity is 0 or close to 0, set velocity command to 0                
                if (norm_playback_vel < epsilon):
                    command_vel[:] = 0.0
                # If the norm of the playback velocity is not 0 or close to 0, set actual velocity command
                else:
                    # Improved assistance for the hand movements:
                    # We define a vector of gammas, assigning the value provided in the platform to the 3 DoFs of the base, and a value of 0.2 lower to the hand DoFs
                    weights_vec = np.array([self.gamma, self.gamma, self.gamma, self.gamma-0.2, self.gamma-0.2, self.gamma-0.2, self.gamma-0.2])
                    # Set values below zero to zero
                    weights_vec = threshold(weights_vec,0.0)

                    # Randomization and blinding:
                    # We keep the weight values for subjects of the hybrid group, whereas they are multiplied by 0 for the subjects of the EEG-only group
                    weights_vec = weights_vec * subject_group

                    # Compute final velocity command
                    term1 = weights_vec * emg_vel_lpf
                    term2 = (1 - weights_vec) * predefined_vel
                    command_vel = term1 + term2
                   
            # If the output of the decoder was "rest" for five or more times
            elif self.consec_rest_outputs >=5:
                # Set the decoder state to "rest"
                self.state_decoder = 0
                # Set the velocities of all DoFs to zero
                command_vel[:] = 0

        # ??? What is that doing?
        # ??? What is the difference between command_vel and command_vel_raw?
        command_vel_raw[:] = command_vel[:]
        for state in self.vel_states:
            command_vel[state] = self.command_lpfs[state](command_vel[state])
            if np.isnan(command_vel[state]):
                command_vel[state] = 0

        # Store current output of the EEG decoder as previous output
        self.prev_output = self.decoder_output

        self.task_data['command_vel']  = command_vel.values
        self.task_data['command_vel_raw']  = command_vel_raw.values

        # Set the velocities of the DoFs that should not be controlled to zero.
        if np.isnan(self.DoF_not_control_idx_init) == False and np.isnan(self.DoF_not_control_idx_end) == False:
            command_vel[self.DoF_not_control_idx_init:self.DoF_not_control_idx_end] = 0

        # Send velocity command to the EXO
        self.plant.send_vel(command_vel.values)

        self.task_data['decoder_output'] = self.decoder_output
        self.task_data['decoder_output_probability'] = self.probability
        self.task_data['state_decoder'] = self.state_decoder
        self.task_data['consec_mov_outputs'] = self.consec_mov_outputs
        self.task_data['consec_rest_outputs'] = self.consec_rest_outputs

        self.task_data['predefined_vel'] = predefined_vel.values
        self.task_data['command_vel_final']  = command_vel.values

    def verify_plant_data_arrival(self, n_secs):
        # Check if data has arrived from ArmAssist and ReHand and if not handle
        time_since_started = time.time() - self.plant.ts_start_data
        last_ts_arrival = self.plant.last_data_ts_arrival()

        if self.plant_type in ['ArmAssist', 'ReHand']:
            print(time_since_started)
            print(n_secs)
            if time_since_started > n_secs:
                if last_ts_arrival == 0:
                    print 'No %s data has arrived at all' % self.plant_type
                else:
                    t_elapsed = time.time() - last_ts_arrival
                    if t_elapsed > n_secs:
                        print 'No %s data in the last %.1f s' % (self.plant_type, t_elapsed)
        
        elif self.plant_type == 'IsMore':
            if time_since_started > n_secs:
                for key in last_ts_arrival.keys():
                    # Check if server is in "no data arrived state"
                    if last_ts_arrival[key] == 0:
                        # No data has arrived at all
                        print('No {} data has arrived at all'.format(key))
                        
                        if not(self.data_not_arrived_state[key]):
                            self.alert_state_indicators[key].color = COLORS['alert_state_failed']
                            if self.beep_on_fail:
                                try:
                                    subprocess.Popen(["/bin/bash","code/ismore/tubingen/scripts/play_alarm","500","300","25","25","2","1"])
                                except:
                                    print("{} DISCONNECTED Could not play alarm sound.".format(key))
                        
                        self.data_not_arrived_state[key] = True
                    else:
                        t_elapsed = time.time() - last_ts_arrival[key]
                        if t_elapsed > n_secs:
                            # There has been missing data in the last seconds
                            print('No {} data in the last {} s').format(key,t_elapsed)
                            self.alert_state_indicators[key].color = COLORS['alert_state_failed']
                            
                            if not(self.data_not_arrived_state[key]):
                                if self.beep_on_fail:
                                    try:
                                        subprocess.Popen(["/bin/bash","code/ismore/tubingen/scripts/play_alarm","500","300","25","25","2","1"])
                                    except:
                                        print("{} DISCONNECTED Could not play alarm sound.".format(key))
                            
                            self.data_not_arrived_state[key] = True
                        else:
                            # Data is arriving
                            if self.data_not_arrived_state[key]:
                                self.alert_state_indicators[key].color = COLORS['alert_state_ok']
                                if self.beep_on_fail:
                                    try:
                                        subprocess.Popen(["/bin/bash","code/ismore/tubingen/scripts/play_alarm","400","800","25","25","1","1"])
                                    except:
                                        print("{} CONNECTED Could not play confirmation sound.".format(key))

                                self.data_not_arrived_state[key] = False

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
        for key in self.emg_decoder_extractor_kwargs:
            if isinstance(self.emg_decoder_extractor_kwargs[key], dict):
                if key != 'feature_fn_kwargs':
                    for key2 in self.emg_decoder_extractor_kwargs[key]:
                        if isinstance(self.emg_decoder_extractor_kwargs[key][key2], np.ndarray):
                            h5file.createArray(emg_extractor_grp, key + '_' + key2, self.emg_decoder_extractor_kwargs[key][key2])
                        else:
                            h5file.createArray(emg_extractor_grp, key + '_' + key2, np.array([self.emg_decoder_extractor_kwargs[key][key2]]))

            else:
                if isinstance(self.emg_decoder_extractor_kwargs[key], np.ndarray):
                    h5file.createArray(emg_extractor_grp, key, self.emg_decoder_extractor_kwargs[key])
                else:
                    h5file.createArray(emg_extractor_grp, key, np.array([self.emg_decoder_extractor_kwargs[key]]))
                
        h5file.close()
        super(HybridBCI, self).cleanup_hdf()

class hybrid_GoToTarget(GoToTarget):
    '''
    Drives the exo towards a predefined target position. 



    verify_plant_data_arrival is added to this class the gipsy way.
    '''

    beep_on_fail      = traits.Int((0), desc=' 0 if a beep should be played when no ArmAssist or ReHand data has arrived.')

    def __init__(self, *args, **kwargs):
        super(hybrid_GoToTarget, self).__init__(*args, **kwargs)

        # Check if task went to a state during which no data from the robot was received
        if self.plant_type == 'IsMore':
            self.data_not_arrived_state = {'ArmAssist': False, 'ReHand': False}

    def init(self):
        super(hybrid_GoToTarget,self).init()

        self.alert_state_indicators = {}
        self.alert_state_indicators['ArmAssist'] = Circle(np.array([0, 0]), 3, COLORS['alert_state_ok'])
        self.alert_state_indicators['ReHand'] = Line(np.array([5, 0]), 5, 10, 0, COLORS['alert_state_ok'])
        
        self.add_model(self.alert_state_indicators['ArmAssist'])
        self.add_model(self.alert_state_indicators['ReHand'])

    def verify_plant_data_arrival(self, n_secs):
        # Check if data has arrived from ArmAssist and ReHand and if not handle
        time_since_started = time.time() - self.plant.ts_start_data
        last_ts_arrival = self.plant.last_data_ts_arrival()

        if self.plant_type in ['ArmAssist', 'ReHand']:
            print(time_since_started)
            print(n_secs)
            if time_since_started > n_secs:
                if last_ts_arrival == 0:
                    print 'No %s data has arrived at all' % self.plant_type
                else:
                    t_elapsed = time.time() - last_ts_arrival
                    if t_elapsed > n_secs:
                        print 'No %s data in the last %.1f s' % (self.plant_type, t_elapsed)
        
        elif self.plant_type == 'IsMore':
            if time_since_started > n_secs:
                for key in last_ts_arrival.keys():
                    # Check if server is in "no data arrived state"
                    if last_ts_arrival[key] == 0:
                        # No data has arrived at all
                        print('No {} data has arrived at all'.format(key))
                        
                        if not(self.data_not_arrived_state[key]):
                            self.alert_state_indicators[key].color = COLORS['alert_state_failed']
                            if self.beep_on_fail:
                                try:
                                    subprocess.Popen(["/bin/bash","code/ismore/tubingen/scripts/play_alarm","500","300","25","25","2","1"])
                                except:
                                    print("{} DISCONNECTED Could not play alarm sound.".format(key))
                        
                        self.data_not_arrived_state[key] = True
                    else:
                        t_elapsed = time.time() - last_ts_arrival[key]
                        if t_elapsed > n_secs:
                            # There has been missing data in the last seconds
                            print('No {} data in the last {} s').format(key,t_elapsed)
                            self.alert_state_indicators[key].color = COLORS['alert_state_failed']
                            
                            if not(self.data_not_arrived_state[key]):
                                if self.beep_on_fail:
                                    try:
                                        subprocess.Popen(["/bin/bash","code/ismore/tubingen/scripts/play_alarm","500","300","25","25","2","1"])
                                    except:
                                        print("{} DISCONNECTED Could not play alarm sound.".format(key))
                            
                            self.data_not_arrived_state[key] = True
                        else:
                            # Data is arriving
                            if self.data_not_arrived_state[key]:
                                self.alert_state_indicators[key].color = COLORS['alert_state_ok']
                                if self.beep_on_fail:
                                    try:
                                        subprocess.Popen(["/bin/bash","code/ismore/tubingen/scripts/play_alarm","400","800","25","25","1","1"])
                                    except:
                                        print("{} CONNECTED Could not play confirmation sound.".format(key))

                                self.data_not_arrived_state[key] = False

class hybrid_EXGEndPointMovement(EXGEndPointMovement):
    '''

    The gipsy way of adding verify_plant_data_arrival.

    '''
    
    beep_on_fail      = traits.Int((0), desc=' 0 if a beep should be played when no ArmAssist or ReHand data has arrived.')

    def __init__(self, *args, **kwargs):
        super(hybrid_EXGEndPointMovement, self).__init__(*args, **kwargs)

        # Check if task went to a state during which no data from the robot was received
        if self.plant_type == 'IsMore':
            self.data_not_arrived_state = {'ArmAssist': False, 'ReHand': False}

    def init(self):
        super(hybrid_EXGEndPointMovement,self).init()

        self.alert_state_indicators = {}
        self.alert_state_indicators['ArmAssist'] = Circle(np.array([0, 0]), 3, COLORS['alert_state_ok'])
        self.alert_state_indicators['ReHand'] = Line(np.array([5, 0]), 5, 10, 0, COLORS['alert_state_ok'])
        
        self.add_model(self.alert_state_indicators['ArmAssist'])
        self.add_model(self.alert_state_indicators['ReHand'])

    def verify_plant_data_arrival(self, n_secs):
        # Check if data has arrived from ArmAssist and ReHand and if not handle
        time_since_started = time.time() - self.plant.ts_start_data
        last_ts_arrival = self.plant.last_data_ts_arrival()

        if self.plant_type in ['ArmAssist', 'ReHand']:
            print(time_since_started)
            print(n_secs)
            if time_since_started > n_secs:
                if last_ts_arrival == 0:
                    print 'No %s data has arrived at all' % self.plant_type
                else:
                    t_elapsed = time.time() - last_ts_arrival
                    if t_elapsed > n_secs:
                        print 'No %s data in the last %.1f s' % (self.plant_type, t_elapsed)
        
        elif self.plant_type == 'IsMore':
            if time_since_started > n_secs:
                for key in last_ts_arrival.keys():
                    # Check if server is in "no data arrived state"
                    if last_ts_arrival[key] == 0:
                        # No data has arrived at all
                        print('No {} data has arrived at all'.format(key))
                        
                        if not(self.data_not_arrived_state[key]):
                            self.alert_state_indicators[key].color = COLORS['alert_state_failed']
                            if self.beep_on_fail:
                                try:
                                    subprocess.Popen(["/bin/bash","code/ismore/tubingen/scripts/play_alarm","500","300","25","25","2","1"])
                                except:
                                    print("{} DISCONNECTED Could not play alarm sound.".format(key))
                        
                        self.data_not_arrived_state[key] = True
                    else:
                        t_elapsed = time.time() - last_ts_arrival[key]
                        if t_elapsed > n_secs:
                            # There has been missing data in the last seconds
                            print('No {} data in the last {} s').format(key,t_elapsed)
                            self.alert_state_indicators[key].color = COLORS['alert_state_failed']
                            
                            if not(self.data_not_arrived_state[key]):
                                if self.beep_on_fail:
                                    try:
                                        subprocess.Popen(["/bin/bash","code/ismore/tubingen/scripts/play_alarm","500","300","25","25","2","1"])
                                    except:
                                        print("{} DISCONNECTED Could not play alarm sound.".format(key))
                            
                            self.data_not_arrived_state[key] = True
                        else:
                            # Data is arriving
                            if self.data_not_arrived_state[key]:
                                self.alert_state_indicators[key].color = COLORS['alert_state_ok']
                                if self.beep_on_fail:
                                    try:
                                        subprocess.Popen(["/bin/bash","code/ismore/tubingen/scripts/play_alarm","400","800","25","25","1","1"])
                                    except:
                                        print("{} CONNECTED Could not play confirmation sound.".format(key))

                                self.data_not_arrived_state[key] = False


class EMG_Only_Control_Decoder(EMGEndPointMovement):
    '''
    verify_plant_data_arrival is added to this class the gipsy way.
    '''
  
    # Settable parameters on web interface for the EMG decoder  
    #music_feedback    = traits.Int((0), desc=' 0 if we do not want to include music, 1 if we want different classical music pieces with increasing intensity to be played')
    gamma             = traits.Float(0.5,   desc='Gamma value for incorporating EMG decoded velocity.')
    emg_decoder       = traits.InstanceFromDB(LinearEMGDecoder, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='emg_decoder'))
    beep_on_fail      = traits.Int((0), desc=' 0 if a beep should be played when no ArmAssist or ReHand data has arrived.')
    channel_list_name = traits.OptionsList(*channel_list_options, bmi3d_input_options=channel_list_options) #choose always the filtered + raw option!!!!!!

    def __init__(self,*args, **kwargs):

        super(EMG_Only_Control_Decoder, self).__init__(*args,**kwargs)     

        # Create EMG extractor object (its 'source' will be set later in the init method)
        if self.emg_decoder.plant_type != self.plant_type:
            print 'Chosen plant_type on the interface does not match the plant type used to train the decoder. Make sure you are selecting the right one'
        
        self.emg_channels = self.emg_decoder.extractor_kwargs['emg_channels']
        self.channels = self.emg_channels

        self.emg_decoder_extractor = self.emg_decoder.extractor_cls(None, 
            emg_channels = self.emg_decoder.extractor_kwargs['emg_channels'],
            feature_names = self.emg_decoder.extractor_kwargs['feature_names'],
            feature_fn_kwargs = self.emg_decoder.extractor_kwargs['feature_fn_kwargs'],
            win_len=self.emg_decoder.extractor_kwargs['win_len'],
            fs=self.emg_decoder.extractor_kwargs['fs'])

        self.emg_decoder_name = self.emg_decoder.decoder_name

        self.nstates_decoder = len(self.emg_decoder.beta.keys())

        self.add_dtype('emg_decoder_features',    'f8', (self.emg_decoder_extractor.n_features,))
        self.add_dtype('emg_decoder_features_Z',  'f8', (self.emg_decoder_extractor.n_features,))
        self.add_dtype('emg_decoder_features_mn','f8', (self.emg_decoder_extractor.n_features,))
        self.add_dtype('emg_decoder_features_std','f8', (self.emg_decoder_extractor.n_features,))
        # to save all the kinematics estimated by the emg decoder even if a less DoF plant is being used online. At least the ones without applying the lpf
        self.add_dtype('emg_vel',         'f8', (self.nstates_decoder,))
        self.add_dtype('predefined_vel',  'f8', (len(self.vel_states),))
        self.add_dtype('emg_vel_lpf',     'f8', (len(self.vel_states),))
        
        # Create buffer that is used for calculating/updating the mean and the std of EMG features online
        self.features_buffer = RingBuffer(
            item_len=self.emg_decoder_extractor.n_features,
            capacity=60*self.fps
        )

        # for low-pass filtering decoded EMG velocities
        self.emg_vel_buffer = RingBuffer(
            item_len=len(self.vel_states),
            capacity=10
        )
        
        # Check if task went to a state during which no data from the robot was received
        if self.plant_type == 'IsMore':
            self.data_not_arrived_state = {'ArmAssist': False, 'ReHand': False}

        self.sounds_dir_classical = os.path.expandvars('$HOME/code/ismore/sounds/classical')
        pygame.mixer.init()

    def init(self):
        super(EMG_Only_Control_Decoder,self).init()
        self.emg_decoder_extractor.source = self.brainamp_source
        
        self.alert_state_indicators = {}
        self.alert_state_indicators['ArmAssist'] = Circle(np.array([0, 0]), 3, COLORS['alert_state_ok'])
        self.alert_state_indicators['ReHand'] = Line(np.array([5, 0]), 5, 10, 0, COLORS['alert_state_ok'])
        
        self.add_model(self.alert_state_indicators['ArmAssist'])
        self.add_model(self.alert_state_indicators['ReHand'])

    def move_plant(self):
        '''Decodes EEG and EMG and sends velocity commands to the robot.
        Usually called in every cycle.'''

        command_vel  = pd.Series(0.0, self.vel_states)
        command_vel_raw  = pd.Series(0.0, self.vel_states)
        emg_vel  = pd.Series(0.0, self.vel_states) 
        emg_vel_lpf  = pd.Series(0.0, self.vel_states)
        predefined_vel  = pd.Series(0.0, self.vel_states) 
        command_vel_final  = pd.Series(0.0, self.vel_states)
        
        #TODO Replace the following line by the real subject randomization value
        subject_group     = 1

        #
        # EMG feature extraction and normalization
        #

        emg_decoder_features = self.emg_decoder_extractor() # emg_features is of type 'dict'
        self.features_buffer.add(emg_decoder_features[self.emg_decoder_extractor.feature_type])
        
        if self.features_buffer.num_items() > 60 * self.fps:
            # If there is more than 60 seconds of recent EMG data, calculate mean and std from this data
            recent_features = self.features_buffer.get_all()
            self.features_mean_emg_dec = np.mean(recent_features, axis=1)

            if hasattr(self.emg_decoder, 'fixed_var_scalar') and self.emg_decoder.fixed_var_scalar:
                self.features_std_emg_dec  = self.emg_decoder.recent_features_std
            else:
                self.features_std_emg_dec = np.std(recent_features, axis=1)
        else:
            # ??? Why try/except?
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
            # Prevent standard deviation from being 0 (?)
            self.features_std_emg_dec[self.features_std_emg_dec == 0] = 1
        except:
            pass

        # Compute z-score of the EMG features
        emg_decoder_features_Z = (emg_decoder_features[self.emg_decoder_extractor.feature_type] - self.features_mean_emg_dec) / self.features_std_emg_dec 
        # Call EMG decoder using the z-score of the EMG features as input
        emg_vel = self.emg_decoder(emg_decoder_features_Z)

        # ???
        self.emg_vel_buffer.add(emg_vel[self.vel_states])

        n_items = self.emg_vel_buffer.num_items()
        buffer_emg = self.emg_vel_buffer.get(n_items)
        win = min(9,n_items)
        weights = np.arange(1./win, 1 + 1./win, 1./win)
        # ??? What is emg_vel_lpf? --> low-pass filtered?
        try:
            emg_vel_lpf = np.sum(weights*buffer_emg[:,n_items-win:n_items+1], axis = 1)/np.sum(weights)
        except:
            pass

        # Store EMG features and velocities in task data
        self.task_data['emg_decoder_features'] = emg_decoder_features[self.emg_decoder_extractor.feature_type]
        self.task_data['emg_decoder_features_Z'] = emg_decoder_features_Z
        self.task_data['emg_decoder_features_mn'] = self.features_mean_emg_dec
        self.task_data['emg_decoder_features_std'] = self.features_std_emg_dec
        self.task_data['emg_vel'] = emg_vel.values
        self.task_data['emg_vel_lpf'] = emg_vel_lpf
                  
        #
        # Move plant according to the output of the EMG decoder
        #

        # Set velocity to zero if the task is in a non-moving state
        if self.state not in ['trial', 'trial_return']:
            command_vel[:] = 0
            self.state_decoder = 0
            
        # Compute the predefined velocity independently of the robot having to move or not. 
        # Just to have a measure of how the correct veloctiy would be at all times.
        # ??? Is this a todo or is it down here? Does not make sense.
        else:
            current_pos = self.plant_pos[:].ravel()
            current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
            target_state = np.hstack([self.goal_position, np.zeros_like(current_pos), 1]).reshape(-1, 1)
            assist_output = self.assister(current_state, target_state, 1)

            # ??? What is assister? What is Bu?                           
            Bu = np.array(assist_output["x_assist"]).ravel()
            #Bu = np.array(assist_output['Bu']).ravel()
            predefined_vel[:] = Bu[len(current_pos):len(current_pos)*2]

    
            self.state_decoder = 1
            # Combine the velocity of the EMG decoder and the playback velocity into one velocity command
            norm_playback_vel = np.linalg.norm(predefined_vel)
            epsilon = 1e-6
            # If the norm of the playback velocity is 0 or close to 0, set velocity command to 0                
            if (norm_playback_vel < epsilon):
                command_vel[:] = 0.0
            # If the norm of the playback velocity is not 0 or close to 0, set actual velocity command
            else:
                # Improved assistance for the hand movements:
                # We define a vector of gammas, assigning the value provided in the platform to the 3 DoFs of the base, and a value of 0.2 lower to the hand DoFs
                weights_vec = np.array([self.gamma, self.gamma, self.gamma, self.gamma-0.2, self.gamma-0.2, self.gamma-0.2, self.gamma-0.2])
                # Set values below zero to zero
                weights_vec = threshold(weights_vec,0.0)

                # Randomization and blinding:
                # We keep the weight values for subjects of the hybrid group, whereas they are multiplied by 0 for the subjects of the EEG-only group
                weights_vec = weights_vec * subject_group

                # Compute final velocity command
                term1 = weights_vec * emg_vel_lpf
                term2 = (1 - weights_vec) * predefined_vel
                command_vel = term1 + term2

        # ??? What is that doing? Low-pass filtering the velocity commands?!
        # ??? What is the difference between command_vel and command_vel_raw?
        command_vel_raw[:] = command_vel[:]
        for state in self.vel_states:
            command_vel[state] = self.command_lpfs[state](command_vel[state])
            if np.isnan(command_vel[state]):
                command_vel[state] = 0

        self.task_data['command_vel']  = command_vel.values
        self.task_data['command_vel_raw']  = command_vel_raw.values
        self.task_data['predefined_vel'] = predefined_vel.values

        # Set the velocities of the DoFs that should not be controlled to zero.
        if np.isnan(self.DoF_not_control_idx_init) == False and np.isnan(self.DoF_not_control_idx_end) == False:
            command_vel[self.DoF_not_control_idx_init:self.DoF_not_control_idx_end] = 0

        # Send velocity command to the EXO
        self.plant.send_vel(command_vel.values)

        self.task_data['command_vel_final']  = command_vel.values

    def verify_plant_data_arrival(self, n_secs):
        # Check if data has arrived from ArmAssist and ReHand and if not handle
        time_since_started = time.time() - self.plant.ts_start_data
        last_ts_arrival = self.plant.last_data_ts_arrival()

        if self.plant_type in ['ArmAssist', 'ReHand']:
            print(time_since_started)
            print(n_secs)
            if time_since_started > n_secs:
                if last_ts_arrival == 0:
                    print 'No %s data has arrived at all' % self.plant_type
                else:
                    t_elapsed = time.time() - last_ts_arrival
                    if t_elapsed > n_secs:
                        print 'No %s data in the last %.1f s' % (self.plant_type, t_elapsed)
        
        elif self.plant_type == 'IsMore':
            if time_since_started > n_secs:
                for key in last_ts_arrival.keys():
                    # Check if server is in "no data arrived state"
                    if last_ts_arrival[key] == 0:
                        # No data has arrived at all
                        print('No {} data has arrived at all'.format(key))
                        
                        if not(self.data_not_arrived_state[key]):
                            self.alert_state_indicators[key].color = COLORS['alert_state_failed']
                            if self.beep_on_fail:
                                try:
                                    subprocess.Popen(["/bin/bash","code/ismore/tubingen/scripts/play_alarm","500","300","25","25","2","1"])
                                except:
                                    print("{} DISCONNECTED Could not play alarm sound.".format(key))
                        
                        self.data_not_arrived_state[key] = True
                    else:
                        t_elapsed = time.time() - last_ts_arrival[key]
                        if t_elapsed > n_secs:
                            # There has been missing data in the last seconds
                            print('No {} data in the last {} s').format(key,t_elapsed)
                            self.alert_state_indicators[key].color = COLORS['alert_state_failed']
                            
                            if not(self.data_not_arrived_state[key]):
                                if self.beep_on_fail:
                                    try:
                                        subprocess.Popen(["/bin/bash","code/ismore/tubingen/scripts/play_alarm","500","300","25","25","2","1"])
                                    except:
                                        print("{} DISCONNECTED Could not play alarm sound.".format(key))
                            
                            self.data_not_arrived_state[key] = True
                        else:
                            # Data is arriving
                            if self.data_not_arrived_state[key]:
                                self.alert_state_indicators[key].color = COLORS['alert_state_ok']
                                if self.beep_on_fail:
                                    try:
                                        subprocess.Popen(["/bin/bash","code/ismore/tubingen/scripts/play_alarm","400","800","25","25","1","1"])
                                    except:
                                        print("{} CONNECTED Could not play confirmation sound.".format(key))

                                self.data_not_arrived_state[key] = False

    def cleanup_hdf(self):
        h5file = tables.openFile(self.h5file.name, mode='a')
        h5file.root.task.attrs['emg_decoder_name'] = self.emg_decoder_name
        emg_extractor_grp = h5file.createGroup(h5file.root, "emg_extractor_kwargs", "Parameters for feature extraction")
        for key in self.emg_decoder.extractor_kwargs:
            if isinstance(self.emg_decoder.extractor_kwargs[key], dict):
                if key != 'feature_fn_kwargs':
                    for key2 in self.emg_decoder.extractor_kwargs[key]:
                        if isinstance(self.emg_decoder.extractor_kwargs[key][key2], np.ndarray):
                            h5file.createArray(emg_extractor_grp, key + '_' + key2, self.emg_decoder.extractor_kwargs[key][key2])
                        else:
                            h5file.createArray(emg_extractor_grp, key + '_' + key2, np.array([self.emg_decoder.extractor_kwargs[key][key2]]))

            else:
                if isinstance(self.emg_decoder.extractor_kwargs[key], np.ndarray):
                    h5file.createArray(emg_extractor_grp, key, self.emg_decoder.extractor_kwargs[key])
                else:
                    h5file.createArray(emg_extractor_grp, key, np.array([self.emg_decoder.extractor_kwargs[key]]))
                
        h5file.close()
        super(EMG_Only_Control_Decoder, self).cleanup_hdf()

class EMG_Only_Control_Reference(EMG_Only_Control_Decoder):
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
    timeout_time      = traits.Float(20, desc='Maximum time given to the patient to accomplish the task before considering it like incomplete and re-starting it from the current position') 
    
    def __init__(self, *args, **kwargs):
        super(EMG_Only_Control_Reference, self).__init__(*args, **kwargs)
        self.add_dtype('reached_timeout', bool, (1,))
        self.add_dtype('gamma_used', float, (1,))
        pygame.mixer.init()
        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)
        self.sounds_dir_classical = os.path.expandvars('$HOME/code/ismore/sounds/classical')
        self.reached_timeout = False
        self.reached_goal_position = False
    
    def _cycle(self):
        '''Runs self.fps times per second.'''

        self.task_data['gamma_used'] = self.gamma
        self.task_data['reached_timeout']  = self.reached_timeout

        super(EMG_Only_Control_Reference, self)._cycle()

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

class EMG_Only_Control_Evaluation(EMG_Only_Control_Reference):

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
            'end_trial': 'instruct_rest_return',#'instruct_trial_go_to_start'
            'timeout': 'instruct_rest_return',
            'stop':      None},
        'instruct_rest_return': {
            'end_instruct': 'rest_return',
            'stop':      None},            
        'rest_return': {
            'end_rest': 'instruct_trial_return',
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

    music_feedback    = traits.Int((0), desc=' 0 if we do not want to include music, 1 if we want different classical music pieces with increasing intensity to be played')
    gamma             = traits.Float(0.5,   desc='Gamma value for incorporating EMG decoded velocity.')
    emg_decoder       = traits.InstanceFromDB(LinearEMGDecoder, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='emg_decoder'))
    channel_list_name = traits.OptionsList(*channel_list_options, bmi3d_input_options=channel_list_options) #choose always the filtered + raw option!!!!!!

    def __init__(self, *args, **kwargs):
        super(EMG_Only_Control_Evaluation, self).__init__(*args, **kwargs)

        import random
        self.gamma_low_block = random.randint(0,8)
        blocks = [num for num in np.arange(0,9) if num != self.gamma_low_block]
        self.gamma_high_block = random.choice(blocks)
        self.add_dtype('gamma_low_block', int, (1,))
        self.add_dtype('gamma_high_block', int, (1,))
        self.add_dtype('rating_difficulty', float, (1,))
        self.add_dtype('experimenter_acceptance_of_rating', np.str_, 10)
        
        self.gamma_chosen = self.gamma
        self.block_number = 0
        self.rating_difficulty = np.nan
        self.experimenter_acceptance_of_rating = ''

    def _cycle(self):
        '''Runs self.fps times per second.'''

        self.task_data['gamma_used'] = self.gamma
        self.task_data['reached_timeout']  = self.reached_timeout

        super(EMG_Only_Control_Evaluation, self)._cycle()

    def _parse_next_trial(self):
        ''' Set the new gamma value for the next trial.
        '''
        self.trial_type = self.next_trial

        if self.block_number == self.gamma_low_block:
            self.gamma -= 0.2
        elif self.block_number == self.gamma_high_block:
            self.gamma += 0.2
        else:
            self.gamma = self.gamma_chosen

        self.block_number +=1   
        
        self.rating_difficulty = np.nan

        print("Gamma value is {}".format(self.gamma))

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

        super(EMG_Only_Control_Evaluation, self)._cycle()
    

    def _test_accept_rating(self, *args, **kwargs):
        self.task_data['rating_difficulty'] = self.rating_difficulty
        return self.experimenter_acceptance_of_rating == 'accept'

    def _test_reject_rating(self, *args, **kwargs):
        return self.experimenter_acceptance_of_rating == 'reject'

    def _start_question(self):
        if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
            self._play_sound(self.sounds_general_dir, ['beep'])
        else:
            self._play_sound(self.sounds_dir, ['rest'])

        print 'Ask the subject to rate the difficulty of the control during the last trial'
        print 'Select the rating and click on Accept to continue with the experiment'

    def _start_instruct_rest_return(self):
        print 'safd'
        if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
            self._play_sound(self.sounds_general_dir, ['beep'])
        else:
            self._play_sound(self.sounds_dir, ['rest'])