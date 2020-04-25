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
#from ismore.ismore_tests.eeg_decoding import LinearEEGDecoder #uncomment this if we want to use the SimEEGMovementDecoding class
from ismore.ismoretasks import PlaybackTrajectories, NonInvasiveBase, RecordBrainAmpData, SimRecordBrainAmpData, IsMoreBase, EndPointMovement
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
from riglib.filter import Filter

from ismore import brainamp_channel_lists

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


targetsB1 =  ['blue', 'brown', 'green', 'red']
targetsB2 = [ 'up', 'down', 'grasp', 'pinch', 'point']
targetsF1_F2 = [
        'green to brown', 
        'green to blue', 
        'brown to red',
        'brown to green',
        'red to green',
        'red to brown',
        'red to blue',
        'green to red',
        'brown to blue',
        'blue to red',
        'blue to green',
        'blue to brown'
    ] 
targets_cyclic = ['circular', 'linear_blue', 'linear_brown', 'linear_green', 'linear_red']

plant_type_options = ['ArmAssist', 'ReHand', 'IsMore']
clda_update_methods = ['RML', 'Smoothbatch', ]
languages_list = ['english', 'deutsch', 'castellano', 'euskara']
#######################################################################


def device_to_use(trial_type):
    '''Return 'ArmAssist' or 'ReHand' depending on whether xy position
    or ReHand angular positions should be used for the given trial_type
    for identifying the current point on trajectory playback.'''
    
    if (trial_type in targetsB1):
        return 'ArmAssist'
        
    elif (trial_type in targetsB2):
        return 'ReHand'

    elif (trial_type in targetsF1_F2):
        return 'IsMore'

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

    rest_interval = traits.Tuple((5., 7.), desc='Min and max time to remain in the rest state.')
    preparation_time    = traits.Float(2,        desc='Time to remain in the preparation state.')
    trial_time    = traits.Float(5,       desc='Time to remain in the trial state.') 

    #add the windows size trait to be able to modifiy it manually
    window_size = traits.Tuple((1000, 560), desc='Size of window to display the plant position/angle')

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
        # from ismore import brainamp_channel_lists
        # self.channels = brainamp_channel_lists.eog4_filt
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

    def _play_sound(self, fname):
        pygame.mixer.music.load(fname)
        pygame.mixer.music.play()

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
        self.image_fname = os.path.join(self.image_dir, 'rest.bmp')
        self.show_image(self.image_fname)

    def _while_rest(self):
        self.image_fname = os.path.join(self.image_dir, 'rest.bmp')
        self.show_image(self.image_fname)

    def _while_instruct_preparation(self):
        self.image_fname = os.path.join(self.image_dir, self.trial_type + '.bmp')
        self.show_image(self.image_fname)

    def _while_preparation(self):
        self.image_fname = os.path.join(self.image_dir, self.trial_type  + '.bmp')
        self.show_image(self.image_fname)

    def _while_instruct_trial_type(self):
        self.image_fname = os.path.join(self.image_dir, self.trial_type + '.bmp')
        self.show_image(self.image_fname)

    def _while_trial(self):
        self.image_fname = os.path.join(self.image_dir, self.trial_type + '.bmp')
        self.show_image(self.image_fname)



    def show_image(self, image_fname):

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

channel_list_options = brainamp_channel_lists.channel_list_options

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
    decoder              = traits.InstanceFromDB(LinearEEGDecoder, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='eeg_decoder'))
    rest_interval        = traits.Tuple((5., 6.), desc='Min and max time to remain in the rest state.')
    preparation_time     = traits.Float(2, desc='time to remain in the preparation state.')
    timeout_time         = traits.Float(30, desc='Maximum time given to the patient to accomplish the task before considering it like incomplete and re-starting it from the current position') 
    give_feedback        = traits.Int((0,1), desc=' 0 if we do not give feedback, 1 if we give feedback about whether the patient reached the goal position')
    targets_matrix       = traits.DataFile(object, desc='goal positions for each of the trial types', bmi3d_query_kwargs=dict(system__name='misc'))
    window_size = traits.Tuple((1000, 560), desc='Size of window to display the plant position/angle')
    channel_list_name = traits.OptionsList(*channel_list_options, bmi3d_input_options=channel_list_options) #choose always the filtered + raw option!!!!!!

    #neighbour_channels = ???
    debug = False

    # Set the "is_bmi_seed" flag so that the server knows that this is a task which can be used to create a BMI decoder
    #is_bmi_seed = False

    def __init__(self, *args, **kwargs):
        super(EEGMovementDecoding, self).__init__(*args, **kwargs)

        self.add_dtype('command_vel',  'f8',    (len(self.vel_states),))
        #self.add_dtype('command_vel_raw',  'f8',    (len(self.vel_states),))
        self.add_dtype('trial_type',   np.str_, 40)
        self.add_dtype('ts',           'f8',    (1,))
        self.add_dtype('goal_pos',      'f8',    (len(self.pos_states),))
        #self.add_dtype('difference_position','f8', (len(self.pos_states),))
        self.add_dtype('reached_goal_position',bool, (1,))
        self.add_dtype('audio_feedback_start',      int,    (1,))
        self.add_dtype('goal_idx', float, (1,))


      
        # if EEG decoder file was specified, load EEG decoder and create feature extractor 
        
        #if len(self.decoder_file) > 3:
        #    self.decoder = pickle.load(open(self.decoder_file, 'rb'))

        # create EEG extractor object (its 'source' will be set later in the init method)
      
        extractor_cls    = self.decoder.extractor_cls
        extractor_kwargs = self.decoder.extractor_kwargs
        self.rest_feature_buffer = self.decoder.rest_feature_buffer
        self.mov_feature_buffer = self.decoder.mov_feature_buffer
        self.channels = extractor_kwargs['channels']
        #extractor_kwargs['brainamp_channels'] = getattr(brainamp_channel_lists, self.channel_list_name)  
        
        self.channels = extractor_kwargs['channels']
        self.brainamp_channels = getattr(brainamp_channel_lists, self.channel_list_name)  

        self.neighbour_channels = self.decoder.neighbour_channels
        self.time_trial = 5 #secs equal to the trial time better than hardcoding it here just in case we change the trial/rest periods time
        # trial time is variable! Think of another way of doing it!
        self.eeg_playback = False
        self.fs = extractor_kwargs['fs']

        self.retrained_decoder = copy.copy(self.decoder.decoder)
        
        

        self.eeg_extractor = extractor_cls(source=None, **extractor_kwargs)
        self.n_features = self.eeg_extractor.n_features
        #self.add_dtype('eeg_features',    'f8', (self.eeg_extractor.n_features,))
        self.add_dtype('eeg_features',    'f8', (self.n_features,))
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
          
        self.target_margin = pd.Series(np.array([3, 3, np.deg2rad(15), np.deg2rad(2),  np.deg2rad(2), np.deg2rad(2), np.deg2rad(2)]), ismore_pos_states)
        self.target_margin = self.target_margin[self.pos_states]



        self.add_dtype('target_margin',      'f8',    (len(self.target_margin),))
        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)

        self.image_dir_general = os.path.expandvars('$HOME/code/ismore/images')
        self.image_dir = os.path.join(self.image_dir_general, self.language)
        

        self.reached_goal_position = False #If the goal_position is reached then give feedback to patient and start the movement back towards the rest_position
        self.init_show_decoder_output()

    def init(self):
        kwargs = {
            'call_rate': self.fps, #kwargs used by the assister
            'xy_cutoff': 2.,#What is this for? Radius of margin?
        }

        from riglib import source
        from ismore.brainamp import rda

        #self.brainamp_source = source.MultiChanDataSource(rda.EMGData, name='brainamp', channels=self.brainamp_channels, brainamp_channels=self.brainamp_channels, send_data_to_sink_manager=True)
        self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT[self.plant_type](**kwargs)
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
        #         self._play_sound(os.path.join(self.sounds_dir, 'beep.wav'))
        #         self.task_data['audio_feedback_start'] = 0
        # print 'distance to target', self.pos_diff(self.goal_position[self.pos_states],self.plant.get_pos())


        self.task_data['audio_feedback_start'] = 0

        if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
            self.task_data['audio_feedback_start'] = 1
            if self.give_feedback:
                self._play_sound(os.path.join(self.sounds_dir, 'beep.wav'))
            if self.goal_idx < len(self.targets_matrix[self.trial_type].keys())-1:
                self.goal_idx +=1
                print 'heading to next subtarget'
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
        #         self._play_sound(os.path.join(self.sounds_dir, 'beep.wav'))




        self.task_data['audio_feedback_start'] = 0

        if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
            self.reached_goal_position = True
            #self.goal_position = self.rest_position
            self.task_data['audio_feedback_start'] = 1
            if self.give_feedback:
                self._play_sound(os.path.join(self.sounds_dir, 'beep.wav'))
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

        # run EEG feature extractor and decoder
        #self.eeg_extractor.source = self.brainamp_source
        eeg_features = self.eeg_extractor() # eeg_features is of type 'dict'
        # print 'eeg_features'
        # print eeg_features
        if self.state in ['trial','trial_return']:
            self.mov_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
            #self.mov_feature_buffer.add(eeg_features)
        elif self.state == 'rest':
            self.rest_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
            #self.rest_feature_buffer.add(eeg_features)

        self.task_data['eeg_features'] = eeg_features
        #print 'eeg_features.shpae'
        eeg_features = np.reshape(eeg_features,(1,eeg_features.shape[0]))
        
        self.decoder_output = self.decoder(eeg_features) #nerea
        self.probability = self.decoder.decoder.predict_proba(eeg_features) #nerea
        
        print self.decoder_output, ' with probability:', self.probability
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
        self.prev_output = self.decoder_output
        #self.task_data['eeg_features']   = eeg_features[self.eeg_extractor.feature_type]
        
        self.task_data['decoder_output'] = self.decoder_output
        self.task_data['decoder_output_probability'] = self.probability
        self.task_data['state_decoder'] = self.state_decoder
        self.plant.send_vel(command_vel.values) #send velocity command to EXO
        
        self.task_data['command_vel']  = command_vel.values
        
    def _cycle(self):
        '''Runs self.fps times per second.'''

        # get latest position/velocity information before calling move_plant()
        self.plant_pos[:] = self.plant.get_pos()
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
            self.task_data['audio_feedback_start'] = 0
            self.task_data['goal_pos'] = np.ones(len(self.pos_states))*np.nan
            self.task_data['reached_goal_position'] = False
            self.task_data['goal_idx'] = np.nan
        else:
            self.task_data['goal_pos'] = self.goal_position
            self.task_data['reached_goal_position'] = self.reached_goal_position            
            self.task_data['goal_idx'] = self.goal_idx

        self.task_data['eeg_decoder_coef']  = self.decoder.decoder.coef_
        self.task_data['eeg_decoder_intercept']  = self.decoder.decoder.intercept_
        self.task_data['eeg_decoder_means']  = self.decoder.decoder.means_

        
        self.task_data['plant_pos']  = self.plant_pos.values
        self.task_data['plant_vel']  = self.plant_vel.values
        self.task_data['trial_type'] = self.trial_type
        self.task_data['ts']         = time.time()
        self.task_data['target_margin'] = self.target_margin
        #self.task_data['decoder']    = self.decoder.decoder

      
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

        #maybe this is not needed
        device = device_to_use(self.trial_type)
        if device == 'ArmAssist':
            self.states = aa_xy_states
        elif device == 'ReHand':
            self.states = rh_pos_states

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
        
        self.features = np.vstack([mov_features, rest_features])
        self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])
        self.retrained_decoder.fit(self.features, self.labels.ravel())
        #print 'shape', self.eeg_extractor.source.get(self.time_trial*self.fs, self.channels)['data'].shape
        #self.rest_data_buffer.add_multiple_values(self.eeg_extractor.source.get(self.time_trial*self.fs, self.channels)['data'])
        print "before updating decoder"
        self.decoder.decoder = copy.copy(self.retrained_decoder)
        #print 'decoder retrained'
        self.reached_goal_position = False
        self.reached_timeout = False

    def _start_instruct_trial_return(self):
        sound_fname = os.path.join(self.sounds_dir, 'back.wav')
        self._play_sound(sound_fname)
        self.reached_goal_position = False
        self.reached_timeout = False
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

    # def _end_trial(self):
    #     self.mov_data_buffer.add_multiple_values(self.eeg_extractor.source.get(self.time_trial/2*self.fs, self.channels)['data'])


    # def _end_trial_return(self):
        
    #     self.mov_data_buffer.add_multiple_values(self.eeg_extractor.source.get(self.time_trial/2*self.fs, self.channels)['data'])
        

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
        self.features = np.vstack([mov_features, rest_features])
        self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])
        self.retrained_decoder.fit(self.features, self.labels.ravel())


        self.decoder.rest_feature_buffer = self.rest_feature_buffer
        self.decoder.mov_feature_buffer = self.mov_feature_buffer
        self.decoder.decoder = copy.copy(self.retrained_decoder)

        #Values just to make it compatible with the task interface (they are nonsense)
        self.decoder.units  = self.decoder.channels_2train
        # self.decoder.binlen = # the decoder is updated after the end of each return trial
        # self.decoder.tslice = 

        #save eeg_decder object into a new pkl file. 
        storage_dir = '/storage/decoders'
        decoder_name = self.decoder.decoder_name 
        # n = decoder_name[-1]
        # n = int(n)        
        index = decoder_name.rfind('_') + 1
        #new_decoder_name = decoder_name[0:index] + str(n + 1)
        new_decoder_name = decoder_name[0:index] + str(saveid) 
        self.decoder.decoder_name = new_decoder_name
        new_pkl_name = new_decoder_name + '.pkl' 
        pickle.dump(self.decoder, open(os.path.join(storage_dir, new_pkl_name), 'wb'))

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

class EEGMovementDecodingNew(NonInvasiveBase):
    
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
            'end_trial': 'instruct_rest_return',#'instruct_trial_go_to_start'
            'timeout': 'instruct_rest',#'instruct_trial_go_to_start'
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
            'timeout': 'instruct_rest_return',#'instruct_rest'
            'stop':      None},    
        }
    
    state = 'wait'  # initial state

     # settable parameters on web interface    
    decoder              = traits.InstanceFromDB(LinearEEGDecoder, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='eeg_decoder'))
    rest_interval        = traits.Tuple((4., 5.), desc='Min and max time to remain in the rest state.')
    preparation_time     = traits.Float(2, desc='time to remain in the preparation state.')
    timeout_time         = traits.Float(7, desc='Maximum time given to the patient to accomplish the task before considering it like incomplete and re-starting it from the current position') 
    give_feedback        = traits.Int((0,1), desc=' 0 if we do not give feedback, 1 if we give feedback about whether the patient reached the goal position')
    targets_matrix       = traits.DataFile(object, desc='goal positions for each of the trial types', bmi3d_query_kwargs=dict(system__name='misc'))
    window_size = traits.Tuple((1000, 560), desc='Size of window to display the plant position/angle')
    channel_list_name = traits.OptionsList(*channel_list_options, bmi3d_input_options=channel_list_options)
    #neighbour_channels = ???
    debug = False

    # Set the "is_bmi_seed" flag so that the server knows that this is a task which can be used to create a BMI decoder
    #is_bmi_seed = False

    def __init__(self, *args, **kwargs):
        super(EEGMovementDecodingNew, self).__init__(*args, **kwargs)

        self.add_dtype('command_vel',  'f8',    (len(self.vel_states),))
        #self.add_dtype('command_vel_raw',  'f8',    (len(self.vel_states),))
        self.add_dtype('trial_type',   np.str_, 40)
        self.add_dtype('ts',           'f8',    (1,))
        self.add_dtype('goal_pos',      'f8',    (len(self.pos_states),))
        #self.add_dtype('difference_position','f8', (len(self.pos_states),))
        self.add_dtype('reached_goal_position',bool, (1,))
        self.add_dtype('reached_timeout',bool, (1,))
        self.add_dtype('simult_reach_and_timeout',bool, (1,))
        self.add_dtype('audio_feedback_start',      int,    (1,))
        self.add_dtype('goal_idx', float, (1,))


      
        # if EEG decoder file was specified, load EEG decoder and create feature extractor 
        
        #if len(self.decoder_file) > 3:
        #    self.decoder = pickle.load(open(self.decoder_file, 'rb'))

        # create EEG extractor object (its 'source' will be set later in the init method)
      
        extractor_cls    = self.decoder.extractor_cls
        
        extractor_kwargs = self.decoder.extractor_kwargs
        
        self.rest_feature_buffer = self.decoder.rest_feature_buffer
        #self.trial_hand_side = extractor_kwargs['trial_hand_side']
        self.mov_feature_buffer = self.decoder.mov_feature_buffer
        self.channels = extractor_kwargs['channels']
        self.brainamp_channels = getattr(brainamp_channel_lists, self.channel_list_name)  

        #self.brainamp_channels = extractor_kwargs['brainamp_channels']
        self.neighbour_channels = self.decoder.neighbour_channels
        self.time_trial = 5 #secs equal to the trial time better than hardcoding it here just in case we change the trial/rest periods time
        # trial time is variable! Think of another way of doing it!
        self.eeg_playback = False
        self.fs = extractor_kwargs['fs']

        self.retrained_decoder = copy.copy(self.decoder.decoder)
        
        

        self.eeg_extractor = extractor_cls(source=None, **extractor_kwargs)
        self.n_features = self.eeg_extractor.n_features
        
        #self.add_dtype('eeg_features',    'f8', (self.eeg_extractor.n_features,))
        self.add_dtype('eeg_features',    'f8', (self.n_features,))
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
          
        self.target_margin = pd.Series(np.array([3, 3, np.deg2rad(15), np.deg2rad(20),  np.deg2rad(2), np.deg2rad(2), np.deg2rad(2)]), ismore_pos_states)
        self.target_margin = self.target_margin[self.pos_states]
        self.goal_idx = 0


        self.add_dtype('target_margin',      'f8',    (len(self.target_margin),))
        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)

        self.image_dir_general = os.path.expandvars('$HOME/code/ismore/images')
        self.image_dir = os.path.join(self.image_dir_general, self.language)
        

        self.reached_goal_position = False #If the goal_position is reached then give feedback to patient and start the movement back towards the rest_position
        self.reached_timeout = False
        self.simult_reach_and_timeout = False

        self.init_show_decoder_output()

    def init(self):
        kwargs = {
            'call_rate': self.fps, #kwargs used by the assister
            'xy_cutoff': 2.,#What is this for? Radius of margin?
        }

        from riglib import source
        from ismore.brainamp import rda

        #self.brainamp_source = source.MultiChanDataSource(rda.EMGData, name='brainamp', channels=self.brainamp_channels, brainamp_channels=self.brainamp_channels, send_data_to_sink_manager=True)
        self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT[self.plant_type](**kwargs)
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
        #         self._play_sound(os.path.join(self.sounds_dir, 'beep.wav'))
        #         self.task_data['audio_feedback_start'] = 0
        # print 'distance to target', self.pos_diff(self.goal_position[self.pos_states],self.plant.get_pos())


        self.task_data['audio_feedback_start'] = 0

        if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
            self.task_data['audio_feedback_start'] = 1
            if self.give_feedback:
                self._play_sound(os.path.join(self.sounds_dir, 'beep.wav'))
            if self.goal_idx < len(self.targets_matrix[self.trial_type].keys())-1:
                self.goal_idx +=1
                print 'heading to next subtarget'
            
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
        #         self._play_sound(os.path.join(self.sounds_dir, 'beep.wav'))




        self.task_data['audio_feedback_start'] = 0

        if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
            self.reached_goal_position = True
            #self.goal_position = self.rest_position
            self.task_data['audio_feedback_start'] = 1
            if self.give_feedback:
                self._play_sound(os.path.join(self.sounds_dir, 'beep.wav'))
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

        # run EEG feature extractor and decoder
        #self.eeg_extractor.source = self.brainamp_source
        eeg_features = self.eeg_extractor() # eeg_features is of type 'dict'
        # print 'eeg_features'
        # print eeg_features
        if self.state in ['trial','trial_return']:
            self.mov_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
            #self.mov_feature_buffer.add(eeg_features)
        elif self.state in ['rest','rest_return']:
            self.rest_feature_buffer.add_multiple_values(eeg_features.reshape(-1,1))
            #self.rest_feature_buffer.add(eeg_features)

        self.task_data['eeg_features'] = eeg_features
        #print 'eeg_features.shpae'
        try:
            eeg_features = np.reshape(eeg_features,(1,eeg_features.shape[0]))
        except:
            pass
        #eeg_features(eeg_features == np.inf) = 1
        self.decoder_output = self.decoder(eeg_features)
        self.probability = self.decoder.decoder.predict_proba(eeg_features)
        
        #print self.decoder_output, ' with probability:', probability
        # Command zero velocity if the task is in a non-moving state
        if self.state not in ['trial', 'trial_return']:#['wait','rest', 'rest_return','instruct_rest', 'instruct_trial_type', 'preparation', 'instruct_go','instruct_rest_return']: 
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
        self.prev_output = self.decoder_output
        #self.task_data['eeg_features']   = eeg_features[self.eeg_extractor.feature_type]
        
        self.task_data['decoder_output'] = self.decoder_output
        self.task_data['decoder_output_probability'] = self.probability
        self.task_data['state_decoder'] = self.state_decoder
        self.plant.send_vel(command_vel.values) #send velocity command to EXO
        
        self.task_data['command_vel']  = command_vel.values
        
    def _cycle(self):
        '''Runs self.fps times per second.'''

        # get latest position/velocity information before calling move_plant()
        self.plant_pos[:] = self.plant.get_pos()
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
            self.task_data['audio_feedback_start'] = 0
            self.task_data['goal_pos'] = np.ones(len(self.pos_states))*np.nan            
            self.task_data['goal_idx'] = np.nan
            
        else:
            self.task_data['goal_pos'] = self.goal_position
            self.task_data['goal_idx'] = self.goal_idx
            


        self.task_data['eeg_decoder_coef']  = self.decoder.decoder.coef_
        self.task_data['eeg_decoder_intercept']  = self.decoder.decoder.intercept_
        self.task_data['eeg_decoder_means']  = self.decoder.decoder.means_
        self.task_data['reached_goal_position'] = self.reached_goal_position            
            
        self.task_data['reached_timeout'] = self.reached_timeout
        self.task_data['simult_reach_and_timeout'] = self.simult_reach_and_timeout
        
        self.task_data['plant_pos']  = self.plant_pos.values
        self.task_data['plant_vel']  = self.plant_vel.values
        self.task_data['trial_type'] = self.trial_type
        self.task_data['ts']         = time.time()
        self.task_data['target_margin'] = self.target_margin
        #self.task_data['decoder']    = self.decoder.decoder

      
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

    def _test_end_rest(self, ts):
        return ts > self.rest_time  

    def _test_end_rest_return(self, ts):
        return ts > self.rest_time  

    def _test_end_preparation(self, ts):
        return ts > self.preparation_time  

    def _test_end_preparation_return(self, ts):
        return ts > self.preparation_time  

    def _parse_next_trial(self):
        self.trial_type = self.next_trial

        #maybe this is not needed
        device = device_to_use(self.trial_type)
        if device == 'ArmAssist':
            self.states = aa_xy_states
        elif device == 'ReHand':
            self.states = rh_pos_states

    def _test_end_instruct(self, *args, **kwargs):
        return not pygame.mixer.music.get_busy()

    def _start_instruct_rest(self):
        self._play_sound(os.path.join(self.sounds_dir, 'rest.wav'))
        #initial_mov_buffer_data = self.mov_data_buffer.get_all()
        print 'rest'

    def _start_instruct_rest_return(self):
        self._play_sound(os.path.join(self.sounds_dir, 'rest.wav'))
        #initial_mov_buffer_data = self.mov_data_buffer.get_all()
        print 'rest'

    def _start_instruct_trial_type(self):
        print 'instruct trial type'
        sound_fname = os.path.join(self.sounds_dir, self.trial_type + '.wav')
        self._play_sound(sound_fname)
        
        mov_features = self.mov_feature_buffer.get_all().T
        rest_features = self.rest_feature_buffer.get_all().T
        
        self.features = np.vstack([mov_features, rest_features])
        self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])
        self.retrained_decoder.fit(self.features, self.labels.ravel())
        #print 'shape', self.eeg_extractor.source.get(self.time_trial*self.fs, self.channels)['data'].shape
        #self.rest_data_buffer.add_multiple_values(self.eeg_extractor.source.get(self.time_trial*self.fs, self.channels)['data'])
        
        self.decoder.decoder = copy.copy(self.retrained_decoder)
        #print 'decoder retrained'
        self.reached_goal_position = False
        self.reached_timeout = False
        self.simult_reach_and_timeout = False
        self.consec_mov_outputs = 0
        self.consec_rest_outputs = 0

    def _start_instruct_trial_return(self):
        sound_fname = os.path.join(self.sounds_dir, 'back.wav')
        self._play_sound(sound_fname)
        self.reached_goal_position = False
        self.reached_timeout = False
        self.simult_reach_and_timeout = False

        mov_features = self.mov_feature_buffer.get_all().T
        rest_features = self.rest_feature_buffer.get_all().T
        
        self.features = np.vstack([mov_features, rest_features])
        self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])
        self.retrained_decoder.fit(self.features, self.labels.ravel())
        #print 'shape', self.eeg_extractor.source.get(self.time_trial*self.fs, self.channels)['data'].shape
        #self.rest_data_buffer.add_multiple_values(self.eeg_extractor.source.get(self.time_trial*self.fs, self.channels)['data'])
        
        self.decoder.decoder = copy.copy(self.retrained_decoder)
        #print 'decoder retrained'
        self.consec_mov_outputs = 0
        self.consec_rest_outputs = 0
        # self.state_decoder = 0


    def _start_instruct_go(self):
        self._play_sound(os.path.join(self.sounds_dir, 'go.wav'))
        self.consec_mov_outputs = 0
        self.consec_rest_outputs = 0
        #self.state_decoder = 0

    def _start_instruct_go_return(self):
        self._play_sound(os.path.join(self.sounds_dir, 'go.wav'))
        self.consec_mov_outputs = 0
        self.consec_rest_outputs = 0
        #self.state_decoder = 0

    def _start_trial(self):
        print self.trial_type
        #self.plant.set_pos_control() #to set it to position control during the trial state
        #self._set_task_type()
        #self._set_goal_position()
        self.goal_position = self.targets_matrix[self.trial_type][self.goal_idx][self.pos_states]
        
        

    def _start_trial_return(self):
        print 'return trial'
        #self.plant.set_pos_control() #to set it to position control during the trial state

        #self._set_task_type()
        self.goal_position = self.targets_matrix['rest'][0][self.pos_states]
        self.goal_idx = 0

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
                self.reached_timeout = False
                self.simult_reach_and_timeout = True
            print 'reached goal position', self.reached_goal_position
            print 'timeout'
            # if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
            #     self.reached_timeout = False
        return self.reached_timeout

    # def _test_at_starting_config(self, *args, **kwargs):
    #     traj = self.ref_trajectories[self.trial_type]['traj']
    #     diff_to_start = np.abs(self.plant.get_pos() - traj[self.pos_states].ix[0].ravel())
    #     #print diff_to_start

    #     return np.all(diff_to_start < self.rest_rect[:len(self.pos_states)])

    # def _end_trial(self):
    #     self.mov_data_buffer.add_multiple_values(self.eeg_extractor.source.get(self.time_trial/2*self.fs, self.channels)['data'])


    # def _end_trial_return(self):
        
    #     self.mov_data_buffer.add_multiple_values(self.eeg_extractor.source.get(self.time_trial/2*self.fs, self.channels)['data'])
        

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
        self.features = np.vstack([mov_features, rest_features])
        self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])
        self.retrained_decoder.fit(self.features, self.labels.ravel())


        self.decoder.rest_feature_buffer = self.rest_feature_buffer
        self.decoder.mov_feature_buffer = self.mov_feature_buffer
        self.decoder.decoder = copy.copy(self.retrained_decoder)

        #Values just to make it compatible with the task interface (they are nonsense)
        self.decoder.units  = self.decoder.channels_2train
        # self.decoder.binlen = # the decoder is updated after the end of each return trial
        # self.decoder.tslice = 

        #save eeg_decder object into a new pkl file. 
        storage_dir = '/storage/decoders'
        decoder_name = self.decoder.decoder_name 
        # n = decoder_name[-1]
        # n = int(n)        
        index = decoder_name.rfind('_') + 1
        #new_decoder_name = decoder_name[0:index] + str(n + 1)
        new_decoder_name = decoder_name[0:index] + str(saveid) 
        self.decoder.decoder_name = new_decoder_name
        new_pkl_name = new_decoder_name + '.pkl' 
        pickle.dump(self.decoder, open(os.path.join(storage_dir, new_pkl_name), 'wb'))

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

class SimEEGMovementDecoding_PK(EEGMovementDecoding):
    def __init__(self, *args, **kwargs):
        super(SimEEGMovementDecoding_PK, self).__init__(*args, **kwargs)
        self.decoder = kwargs['decoder']
        self.brainamp_channels = kwargs['brainamp_channels']


class SimEEGMovementDecoding(EEGMovementDecoding):
    
    

    # def __init__(self, *args, **kwargs):
    #     super(SimEEGMovementDecoding, self).__init__(*args, **kwargs)

    #     self.rest_data_buffer = self.decoder.rest_data_buffer
    #     self.mov_data_buffer = self.decoder.mov_data_buffer
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
        self.decoder_output = self.decoder(eeg_features)
        self.probability = self.decoder.decoder.predict_proba(eeg_features)

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
        self.plant_pos[:] = self.plant.get_pos()
        self.plant_vel[:] = self.plant.get_vel()

        #if self.state in ['trial','go_to_start']:
        # velocity control
        self.move_plant()

        # position control
        # self.move_plant_pos_control()

        self.update_plant_display()

        # print self.subtrial_idx
        if not self.state in ['trial','trial_return']:
            self.task_data['audio_feedback_start'] = 0
            self.task_data['goal_pos'] = np.ones(len(self.pos_states))*np.nan
            self.task_data['reached_goal_position'] = False
            self.task_data['goal_idx'] = np.nan
        else:
            self.task_data['goal_pos'] = self.goal_position
            self.task_data['reached_goal_position'] = self.reached_goal_position
            self.task_data['goal_idx'] = self.goal_idx
        #print 'coefs', self.decoder.decoder.coef_
        self.task_data['eeg_decoder_coef']  = self.decoder.decoder.coef_
        self.task_data['eeg_decoder_intercept']  = self.decoder.decoder.intercept_
        self.task_data['eeg_decoder_means']  = self.decoder.decoder.means_

        
        self.task_data['plant_pos']  = self.plant_pos.values
        self.task_data['plant_vel']  = self.plant_vel.values
        self.task_data['trial_type'] = self.trial_type
        self.task_data['ts']         = time.time()
        self.task_data['target_margin'] = self.target_margin
        #self.task_data['decoder']    = self.decoder.decoder

      
        super(SimEEGMovementDecoding, self)._cycle()


    # def show_image(self, image_fname):

    #     window = pygame.display.set_mode(self.window_size)
    #     img = pygame.image.load(os.path.join(self.image_fname))
    #     img = pygame.transform.scale(img, self.window_size)

    #     window.blit(img, (0,0))
    #     pygame.display.flip()

    #### STATE AND TEST FUNCTIONS ####   
    

    # def _start_instruct_trial_type(self):
    #     sound_fname = os.path.join(self.sounds_dir, self.trial_type + '.wav')
    #     self._play_sound(sound_fname)
    #     data = self.eeg_extractor.source.get(self.time_trial*self.fs, self.channels)['data']
        
   
    #     len_data = len(data[0])
    #     fsample = 1000.00 #Sample frequency in Hz
    #     f = 10 # in Hz
    #     rest_amp = 10
    #     cnt = 1
    #     cnt_noise = 1

    #     for k, n in enumerate(self.channels): #for loop on number of electrodes
    #         if k in ['chan8_filt', 'chan9_filt', 'chan13_filt', 'chan14_filt', 'chan18_filt', 'chan19_filt']:
    #         #if k in ['chan13_filt', 'chan14_filt', 'chan18_filt', 'chan19_filt']:
    #             rest_noise = rest_amp*0.1*np.random.randn(len_data) #10% of signal amplitude
    #             rest_signal = np.zeros(len_data)
                
    #             for i in np.arange(len_data):
    #                 rest_signal[i] = (rest_amp+cnt) * math.sin((f+cnt-1)*2*math.pi*t[i]) + rest_noise[i] #rest sinusoidal signal
    #             cnt += 1

    #         else:
    #             rest_signal = (rest_amp + cnt_noise)*0.1*np.random.randn(len_data) #10% of signal amplitude. only noise                 move_signal = rest_amp*0.1*cnt_noise*np.random.randn(self.n_win_pts) #10% of signal amplitude
    #             cnt_noise += 1  
         
    #         data[k] = rest_signal.copy()

    #     self.rest_data_buffer.add_multiple_values(data)
    #     self.decoder.decoder = copy.copy(self.retrained_decoder)
    #     self.reached_goal_position = False
    #     self.reached_timeout = False


    # def _end_trial(self):
    #     data = self.eeg_extractor.source.get(self.time_trial/2*self.fs, self.channels)['data']
    #     len_data = len(data[0])
    #     fsample = 1000.00 #Sample frequency in Hz
    #     f = 10 # in Hz
    #     rest_amp = 10
    #     move_amp = 5; #mov state amplitude 
    #     cnt = 1
    #     cnt_noise = 1
    #     samples = dict()
    #     for k, n in enumerate(self.channels): #for loop on number of electrodes
    #         if k in ['chan8_filt', 'chan9_filt', 'chan13_filt', 'chan14_filt', 'chan18_filt', 'chan19_filt']:
    #         #if k in ['chan13_filt', 'chan14_filt', 'chan18_filt', 'chan19_filt']:
                
    #             move_noise = move_amp*0.1*np.random.randn(len_data) #10% of signal amplitude
    #             move_signal = np.zeros(len_data)
    #             for i in np.arange(len_data):
                    
    #                 move_signal[i] = (move_amp+cnt) * math.sin((f+cnt-1)*2*math.pi*t[i]) + move_noise[i]
    #             cnt += 1

    #         else:
                
    #             move_signal = (rest_amp + cnt_noise)*0.1*np.random.randn(len_data) #10% of signal amplitude
    #             cnt_noise += 1

    #         data[k] = move_signal.copy()
         
    #     self.mov_data_buffer.add_multiple_values(data)

        

    # def _end_trial_return(self):
    #     data = self.eeg_extractor.source.get(self.time_trial/2*self.fs, self.channels)['data']
    #     len_data = len(data[0])
    #     fsample = 1000.00 #Sample frequency in Hz
    #     f = 10 # in Hz
    #     rest_amp = 10
    #     move_amp = 5; #mov state amplitude 
    #     cnt = 1
    #     cnt_noise = 1
    #     samples = dict()

    #     for k, n in enumerate(self.channels): #for loop on number of electrodes

            
    #         if k in ['chan13_filt', 'chan14_filt', 'chan18_filt', 'chan19_filt']:
                
    #             move_noise = move_amp*0.1*np.random.randn(len_data) #10% of signal amplitude
    #             move_signal = np.zeros(len_data)
    #             for i in np.arange(len_data):
                    
    #                 move_signal[i] = move_amp*cnt * math.sin((f+cnt-1)*2*math.pi*t[i]) + move_noise[i]
    #             cnt += 1

    #         else:
                
    #             move_signal = rest_amp*0.1*cnt_noise*np.random.randn(len_data) #10% of signal amplitude
    #             cnt_noise += 1
                
    #         data[k] = move_signal.copy()
    #     self.mov_data_buffer.add_multiple_values(data)
    
    # def cleanup(self, database, saveid, **kwargs):
    #     self.mov_data = self.mov_data_buffer.get_all()
    #     self.rest_data = self.rest_data_buffer.get_all()
    #     rest_features, mov_features = self.eeg_extractor.extract_features_2retrain(self.rest_data, self.mov_data)
    #     self.features = np.vstack([mov_features, rest_features])
    #     self.labels = np.vstack([np.ones([mov_features.shape[0],1]), np.zeros([rest_features.shape[0],1])])
    #     # self.retrained_decoder = LDA()# default values are being used: solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001)
    #     self.retrained_decoder.fit(self.features, self.labels.ravel())


    #     self.decoder.rest_data_buffer = self.rest_data_buffer
    #     self.decoder.mov_data_buffer = self.mov_data_buffer
    #     self.decoder.decoder = copy.copy(self.retrained_decoder)

    #     #Values just to make it compatible with the task interface (they are nonsense)
    #     self.decoder.units  = self.decoder.channels_2train
    #     # self.decoder.binlen = # the decoder is updated after the end of each return trial
    #     # self.decoder.tslice = 

    #     #save eeg_decder object into a new pkl file. 
    #     storage_dir = '/storage/decoders'
    #     decoder_name = self.decoder.decoder_name 
    #     # n = decoder_name[-1]
    #     # n = int(n)        
    #     index = decoder_name.rfind('_') + 1
    #     #new_decoder_name = decoder_name[0:index] + str(n + 1)
    #     new_decoder_name = decoder_name[0:index] + str(saveid) 
    #     self.decoder.decoder_name = new_decoder_name
    #     new_pkl_name = new_decoder_name + '.pkl' 
    #     pickle.dump(self.decoder, open(os.path.join(storage_dir, new_pkl_name), 'wb'))

    #     super(SimEEGMovementDecoding,self).cleanup(database, saveid, **kwargs)
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
        self.decoder_output = self.decoder(eeg_features)
        self.probability = self.decoder.decoder.predict_proba(eeg_features)

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
        self.plant_pos[:] = self.plant.get_pos()
        self.plant_vel[:] = self.plant.get_vel()

        #if self.state in ['trial','go_to_start']:
        # velocity control
        self.move_plant()

        # position control
        # self.move_plant_pos_control()

        self.update_plant_display()

        # print self.subtrial_idx
        if not self.state in ['trial','trial_return']:
            self.task_data['audio_feedback_start'] = 0
            self.task_data['goal_pos'] = np.ones(len(self.pos_states))*np.nan
            self.task_data['reached_goal_position'] = False            
            self.task_data['goal_idx'] = np.nan
        else:
            self.task_data['goal_pos'] = self.goal_position
            self.task_data['reached_goal_position'] = self.reached_goal_position
            self.task_data['goal_idx'] = self.goal_idx
        #print 'coefs', self.decoder.decoder.coef_
        self.task_data['eeg_decoder_coef']  = self.decoder.decoder.coef_
        self.task_data['eeg_decoder_intercept']  = self.decoder.decoder.intercept_
        self.task_data['eeg_decoder_means']  = self.decoder.decoder.means_

        
        self.task_data['plant_pos']  = self.plant_pos.values
        self.task_data['plant_vel']  = self.plant_vel.values
        self.task_data['trial_type'] = self.trial_type
        self.task_data['ts']         = time.time()
        self.task_data['target_margin'] = self.target_margin
        #self.task_data['decoder']    = self.decoder.decoder

      
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

        extractor_cls    = self.emg_decoder.extractor_cls
        extractor_kwargs = self.emg_decoder.extractor_kwargs

        #print [settings.BRAINAMP_CHANNELS[chan] + "_filt" for i, chan in enumerate(settings.BRAINAMP_CHANNELS)]
        # if self.brainamp_channels != channels_filt:
        #     print 'ERROR: The selected channels in the interface do not match those defined in settings to be streamed from the amplifier.'

        # create EMG extractor object (it's 'source' will be set later in the init method)
        if self.emg_decoder.plant_type != self.plant_type:
            print 'Chosen plant_type on the interface does not match the plant type used to train the decoder. Make sure you are selecting the right one'
    
        self.brainamp_channels = getattr(brainamp_channel_lists, self.channel_list_name)
        self.channels = extractor_kwargs['channels']
        # self.channels_2discard = extractor_kwargs['channels_2discard']
        # self.channels_2keep = extractor_kwargs['channels_2keep']
        # self.channels_diag1_1 = extractor_kwargs['channels_diag1_1']
        # self.channels_diag1_2 = extractor_kwargs['channels_diag1_2']
        # self.channels_diag2_1 = extractor_kwarg['channels_diag2_1']
        # self.channels_diag2_2 = extractor_kwargs['channels_diag2_2']
        #self.brainamp_channels = extractor_kwargs['brainamp_channels'] 
        
        # extractor_kwargs['channels_filt'] = list()
        # for i in range(len(extractor_kwargs['channels'])):
        #     extractor_kwargs['channels_filt'] = [extractor_kwargs['channels'][i] + "_filt"]
        #     extractor_kwargs['channels_filt'].append(extractor_kwargs['channels_filt'])
        
        #self.emg_extractor = extractor_cls(source=None, channels = self.brainamp_channels, **extractor_kwargs)
       
        #self.emg_extractor = extractor_cls(source=None, **extractor_kwargs)
        self.emg_extractor = extractor_cls(None, channels = extractor_kwargs['channels'], feature_names = extractor_kwargs['feature_names'], feature_fn_kwargs = extractor_kwargs['feature_fn_kwargs'], win_len=extractor_kwargs['win_len'], fs=extractor_kwargs['fs'])

        self.add_dtype('emg_features',    'f8', (self.emg_extractor.n_features,))
        self.add_dtype('emg_features_Z',  'f8', (self.emg_extractor.n_features,))
        self.add_dtype('emg_vel',         'f8', (len(self.vel_states),))
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

        self.command_lpfs = dict()
        for state in self.vel_states:
            self.command_lpfs[state] = Filter(b=[1., -0.3], a=[1.]) # low-pass filter to smooth out command velocities
    
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


                #term1 = self.gamma * self.emg_decoder.lambda_coeffs * ((np.dot(emg_vel_lpf, predefined_vel) / (norm_predefined_vel**2)) * predefined_vel)
                #term2 = (1 - self.gamma * self.emg_decoder.lambda_coeffs) * predefined_vel


            if (device_to_use(self.trial_type) == 'ReHand' and self.plant_type == 'IsMore') :
                command_vel[aa_vel_states] = 0.0
        

        command_vel_raw[:] = command_vel[:]

        # # # # Apply low-pass filter to command velocities
        for state in self.vel_states:
        #     print command_vel[state]
            command_vel[state] = self.command_lpfs[state](command_vel[state])
            if np.isnan(command_vel[state]):
                command_vel[state] = 0

        self.plant.send_vel(command_vel.values) #send velocity command to EXO
        
        self.task_data['command_vel']  = command_vel.values
        self.task_data['command_vel_raw']  = command_vel_raw.values
        self.task_data['predefined_vel'] = predefined_vel.values


class EMGTrajectoryDecoding(PlaybackTrajectories):


    # settable parameters on web interface    
    gamma             = traits.Float(0.5,   desc='Gamma value for incorporating EMG decoded velocity.')
    emg_playback_file = traits.String('',   desc='Full path to recorded EMG data file.')
    emg_decoder       = traits.InstanceFromDB(LinearEMGDecoder, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='emg_decoder'))
    use_emg_decoder   = traits.Int((0,1), desc=' 0 if we do not give feedback, 1 if we give feedback')


    def __init__(self, *args, **kwargs):
        super(EMGTrajectoryDecoding, self).__init__(*args, **kwargs)
       

        #self.channels_filt = brainamp_channel_lists.emg14_filt

        # if EMG decoder file was specified, load EMG decoder and create feature extractor 
        # if len(self.emg_decoder_file) > 3:
        #     self.emg_decoder = pickle.load(open(self.emg_decoder_file, 'rb'))

        # print settings.BRAINAMP_CHANNELS
        # channels_filt = []
        # for k in range(len(settings.BRAINAMP_CHANNELS)):
        #     channels_filt.append(settings.BRAINAMP_CHANNELS[k] + "_filt")

        extractor_cls    = self.emg_decoder.extractor_cls
        extractor_kwargs = self.emg_decoder.extractor_kwargs

        #print [settings.BRAINAMP_CHANNELS[chan] + "_filt" for i, chan in enumerate(settings.BRAINAMP_CHANNELS)]
        # if self.brainamp_channels != channels_filt:
        #     print 'ERROR: The selected channels in the interface do not match those defined in settings to be streamed from the amplifier.'

        # create EMG extractor object (it's 'source' will be set later in the init method)
        if self.emg_decoder.plant_type != self.plant_type:
            print 'Chosen plant_type on the interface does not match the plant type used to train the decoder. Make sure you are selecting the right one'

        

        self.brainamp_channels = extractor_kwargs['brainamp_channels'] 
        
        # extractor_kwargs['channels_filt'] = list()
        # for i in range(len(extractor_kwargs['channels'])):
        #     extractor_kwargs['channels_filt'] = [extractor_kwargs['channels'][i] + "_filt"]
        #     extractor_kwargs['channels_filt'].append(extractor_kwargs['channels_filt'])

        self.emg_playback = False
        
        #self.emg_extractor = extractor_cls(source=None, channels = self.brainamp_channels, **extractor_kwargs)

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
        
        self.plant.enable() 

        self.command_lpfs = dict()
        for state in self.vel_states:
            self.command_lpfs[state] = Filter(b=[1., -0.3], a=[1.]) # low-pass filter to smooth out command velocities

    def init(self):
        
        from riglib import source
        from ismore.brainamp import rda

        self.brainamp_source = source.MultiChanDataSource(rda.EMGData, name='brainamp', channels=self.brainamp_channels, brainamp_channels=self.brainamp_channels, send_data_to_sink_manager=True)
        self.emg_extractor.source = self.brainamp_source
        #self.emg_extractor.channels = self.brainamp_channels
        super(EMGTrajectoryDecoding, self).init()

    def move_plant(self):
        '''Docstring.'''

        #playback_vel = pd.Series(0.0, self.vel_states)
        command_vel  = pd.Series(0.0, self.vel_states)
        command_vel_raw  = pd.Series(0.0, self.vel_states)
        #aim_pos      = pd.Series(0.0, self.pos_states)
        emg_vel      = pd.Series(0.0, self.vel_states) #nerea
        
        # run EMG feature extractor and decoder

        #self.emg_extractor.source = self.brainamp_source

        emg_features = self.emg_extractor() # emg_features is of type 'dict'

        #emg_features[self.emg_extractor.feature_type] = emg_features[self.emg_extractor.feature_type][len(emg_features[self.emg_extractor.feature_type])/2:]
        self.features_buffer.add(emg_features[self.emg_extractor.feature_type])
        #print 'emg_features[self.emg_extractor.feature_type]', emg_features[self.emg_extractor.feature_type]
        if 1: #self.features_buffer.num_items() > 1 * self.fps:
                # if we have more than 1 second of recent EMG data, then
                #   calculate mean and std from this data
            recent_features = self.features_buffer.get_all()
            #print 'recent_features', recent_features
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
        emg_vel = self.emg_decoder(emg_features_Z)

        self.emg_vel_buffer.add(emg_vel[self.vel_states])

            #print 'any zeros in std vector?:', any(features_std == 0.0)
        

        emg_vel_lpf = np.mean(self.emg_vel_buffer.get_all(), axis=1)

        self.task_data['emg_features']   = emg_features[self.emg_extractor.feature_type]
        self.task_data['emg_features_Z'] = emg_features_Z
        self.task_data['emg_vel']        = emg_vel
        self.task_data['emg_vel_lpf']    = emg_vel_lpf
                  

        # combine EMG decoded velocity and playback velocity into one velocity command
        norm_playback_vel = np.linalg.norm(self.playback_vel)
        epsilon = 1e-6
        if (norm_playback_vel < epsilon):
                # if norm of the playback velocity is 0 or close to 0,
                #   then just set command velocity to 0s
            command_vel[:] = 0.0

        else:

            #feedback 1
            term1 = self.gamma * emg_vel_lpf
            term2 = (1 - self.gamma) * self.playback_vel

                #feedback 2
                # term1 = self.gamma * ((np.dot(emg_vel_lpf, playback_vel) / (norm_playback_vel**2)) * playback_vel)
                # term2 = (1 - self.gamma) * playback_vel


                #term1 = self.gamma * self.emg_decoder.lambda_coeffs * ((np.dot(emg_vel_lpf, playback_vel) / (norm_playback_vel**2)) * playback_vel)
                #term2 = (1 - self.gamma * self.emg_decoder.lambda_coeffs) * playback_vel
                

            command_vel = term1 + term2


            if (device_to_use(self.trial_type) == 'ReHand' and self.plant_type == 'IsMore') :
                command_vel[aa_vel_states] = 0.0
        


        command_vel_raw[:] = command_vel[:]

        # # # # Apply low-pass filter to command velocities
        for state in self.vel_states:
        #     print command_vel[state]
            command_vel[state] = self.command_lpfs[state](command_vel[state])
            if np.isnan(command_vel[state]):
                command_vel[state] = 0

            #command_vel[state][np.isnan(command_vel[state][:])]
              
        # iterate over actual State objects, not state names
        # for state in self.ssm.states:
        #     if state.name in self.vel_states:
        #         command_vel[state.name] = bound(command_vel[state.name], state.min_val, state.max_val)

        # Command zero velocity if the task is in a non-moving state
        if self.state in ['wait','rest', 'instruct_rest', 'instruct_trial_type']: 
            command_vel[:] = 0

        
        self.plant.send_vel(command_vel.values)
        

        
        self.task_data['command_vel']  = command_vel.values
        self.task_data['command_vel_raw']  = command_vel_raw.values

class SimEMGTrajectoryDecoding(EMGTrajectoryDecoding):
    '''
    Same as above, but only for debugging purposes, so uses an old HDF file for EMG data instead of live streaming data
    '''
    emg_playback_file = traits.String('', desc='file from which to replay old EMG data. Leave blank to stream EMG data from the brainamp system')
