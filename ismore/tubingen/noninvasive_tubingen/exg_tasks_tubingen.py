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
from ismore.tubingen.noninvasive_tubingen.emg_decoding import LinearEMGDecoder
from ismore.tubingen.noninvasive_tubingen.eeg_decoding import LinearEEGDecoder 
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
    timeout_attempts      = traits.Float(4, desc='Maximum number of attempts given to the patient to accomplish the task before moving to the next trial') 
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
            #if '_' in filename:
            #    filename = filename[:filename.find('_')]
            sound_fname = os.path.join(fpath, filename + '.wav')
            
            print sound_fname

            if pygame.mixer.music.get_busy():
                pygame.mixer.music.queue(sound_fname)
                print "queued sound: "+sound_fname
            else:
                pygame.mixer.music.load(sound_fname)
                pygame.mixer.music.play()


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

        self.attempt = 0

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
        
        if self.reached_goal_position == True:
            self.attempt = 0

        if self.attempt == self.timeout_attempts - 1:
            if ts > (self.timeout_time-0.2):  
            #no so elegant. Otherwise, the last trial will finish for the time out and there will be a quick trial afterwards
            #cause this state is being refreshed after finishing the trial by the timeout. 
                self.reached_goal_position = True
                print 'max number of attempts consumed'
                self.attempt = 0

        return (self.reached_goal_position or self.reached_timeout)

    def _test_end_trial_return(self,ts):
        # Test if simultaneous timeout and end_trial issue is solved with this
        # if np.all(np.abs(self.pos_diff(self.goal_position,self.plant.get_pos())) < self.target_margin[self.pos_states]):
        #     self.reached_goal_position = True
        
        if self.reached_goal_position == True:
            self.attempt = 0

        if self.attempt == self.timeout_attempts - 1:
            if ts > (self.timeout_time-0.2):
            #no so elegant. Otherwise, the last trial will finish for the time out and there will be a quick trial afterwards
            #cause this state is being refreshed after finishing the trial by the timeout.     
                self.reached_goal_position = True
                print 'max number of attempts consumed'
                self.attempt = 0


        return (self.reached_goal_position or self.reached_timeout)

    def _test_timeout(self, ts):
        if ts > self.timeout_time:
            self.reached_timeout = True
            print 'timeout'

            self.attempt += 1

            if self.reached_goal_position == True:
                #self.reached_timeout = False
                self.simult_reach_and_timeout = True
            #print 'reached goal position', self.reached_goal_position
            
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