from __future__ import division
import numpy as np
from collections import OrderedDict
import time

from riglib import reward
from riglib.experiment import traits, Sequence

from riglib.stereo_opengl.window import Window, FPScontrol, WindowDispl2D
from riglib.stereo_opengl.primitives import Cylinder, Plane, Sphere
from riglib.stereo_opengl.models import FlatMesh, Group
from riglib.stereo_opengl.textures import Texture, TexModel
from riglib.stereo_opengl.render import stereo, Renderer
from riglib.stereo_opengl.utils import cloudy_tex

from .plantlist import plantlist

from riglib.stereo_opengl import ik

import math
import traceback

import random
import serial, glob

sec_per_min = 60.0
target_colors = {"blue":(0,0,1,0.5),
"yellow": (1,1,0,0.5),
"hibiscus":(0.859,0.439,0.576,0.5),
"magenta": (1,0,1,0.5),
"purple":(0.608,0.188,1,0.5),
"lightsteelblue":(0.690,0.769,0.901,0.5),
"dodgerblue": (0.118,0.565,1,0.5),
"teal":(0,0.502,0.502,0.5),
"aquamarine":(0.498,1,0.831,0.5),
"olive":(0.420,0.557,0.137,0.5),
"chiffonlemon": (0.933,0.914,0.749,0.5),
"juicyorange": (1,0.502,0,0.5),
"salmon":(1,0.549,0.384,0.5),
"wood": (0.259,0.149,0.071,0.5),
"elephant":(0.409,0.409,0.409,0.5)}


class ManualControlFreeChoice(Sequence, Window):
    '''
    This is for a free-choice task with two targets (left and right) presented to choose from.  
    The position of the targets may change along the x-axis, according to the target generator, 
    and each target has a varying probability of reward, also according to the target generator.
    The code as it is written is for a joystick.  
    Notes: want target_index to only write once per trial.  if so, can make instructed trials random.  else, make new state for instructed trial.
    '''

    background = (0,0,0,1)
    shoulder_anchor = np.array([2., 0., -15.]) # Coordinates of shoulder anchor point on screen
    
    arm_visible = traits.Bool(True, desc='Specifies whether entire arm is displayed or just endpoint')
    
    cursor_radius = traits.Float(.5, desc="Radius of cursor")
    cursor_color = (.5,0,.5,1)

    joystick_method = traits.Float(1,desc="1: Normal velocity, 0: Position control")
    joystick_speed = traits.Float(20, desc="Speed of cursor")

    #plant_type_options = plantlist.keys()   ### original one
    plant_type_options = list(plantlist.keys()) ###############################
    #plant_type = traits.OptionsList(*plantlist, bmi3d_input_options=plantlist.keys())   ## original one
    plant_type = traits.OptionsList(*plantlist, bmi3d_input_options=list(plantlist.keys()))
    starting_pos = (5, 0, 5)
    # window_size = (1280*2, 1024)
    window_size = traits.Tuple((1920*2, 1080), desc='window size')
    percentage_instructed_trials = traits.Int(10,desc='Percentage of instructed trials: multiples of 10')
    #stimulation_frequency = traits.Float(200, desc="Stimulation frequency")
    #stimulation_pulse_length = traits.Float(200, desc="Stimulation pulse duration in us")

    status = dict(
        #wait = dict(start_trial="target", stop=None),
        wait = dict(start_trial="center", stop=None),
        center = dict(enter_center="hold_center", timeout="timeout_penalty", stop=None),
        hold_center = dict(leave_early_center = "hold_penalty",hold_center_complete="target", timeout="timeout_penalty", stop=None),
        target = dict(enter_targetL="hold_targetL", enter_targetH = "hold_targetH", timeout="timeout_penalty", stop=None),
        hold_targetH = dict(leve_early_H="hold_penalty", hold_complete="targ_transition"),
        hold_targetL = dict(leave_early_L="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="check_reward",trial_abort="wait", trial_incomplete="center"),
        check_reward = dict(yes_reward="reward",no_reward="wait"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )
    #
    target_color = (.5,1,.5,0)

    #initial state
    state = "wait"

    #create settable traits
    reward_time = traits.Float(.5, desc="Length of juice reward")
    reward_time_small = traits.Float(.2, desc="Length of juice for small reward if Small/Large schedule is used")
    reward_time_large = traits.Float(.8, desc="Length of juice for large reward if Small/Large schedule is used")
    reward_SmallLarge = traits.Float(0, desc="Use Small/Large juice rewards rather than No/Yes rewards")
    target_radius = traits.Float(1.5, desc="Radius of targets in cm")
    block_length = traits.Float(100, desc="Number of trials per block")  
    # target_location_offset_allowed should be Float not Bool to simplify it's use in math operations in the code    
    location_offset_allowed = traits.Float(1, desc="If targets may be at asymmetric distances from center, this is 1. If always equidistant, this is 0.")
    hold_time = traits.Float(.2, desc="Length of hold required at targets")
    hold_penalty_time = traits.Float(1, desc="Length of penalty time for target hold error")
    timeout_time = traits.Float(10, desc="Time allowed to go between targets")
    timeout_penalty_time = traits.Float(1, desc="Length of penalty time for timeout error")
    max_attempts = traits.Int(10, desc='The number of attempts at a target before\
        skipping to the next one')
    session_length = traits.Float(0, desc="Time until task automatically stops. Length of 0 means no auto stop.")
    marker_num = traits.Int(14, desc="The index of the motiontracker marker to use for cursor position")
    # NOTE!!! The marker on the hand was changed from #0 to #14 on
    # 5/19/13 after LED #0 broke. All data files saved before this date
    # have LED #0 controlling the cursor.

    arm_hide_rate = traits.Float(0.0, desc='If the arm is visible, specifies a percentage of trials where it will be hidden')
    target_index = 0 # Helper variable to keep track of whether trial is instructed (1 = 1 choice) or free-choice (2 = 2 choices)
    target_selected = 'L'   # Helper variable to indicate which target was selected
    tries = 0 # Helper variable to keep track of the number of failed attempts at a given trial.
    timedout = False    # Helper variable to keep track if transitioning from timeout_penalty
    reward_counter = 0.0
    cursor_visible = False # Determines when to hide the cursor.
    no_data_count = 0 # Counter for number of missing data frames in a row
    scale_factor = 3.0 #scale factor for converting hand movement to screen movement (1cm hand movement = 3.5cm cursor movement)
    starting_dist = 10.0 # starting distance from center target
    #color_targets = np.random.randint(2)
    color_targets = 1   # 0: yellow low, blue high; 1: blue low, yellow high
    stopped_center_hold = False   #keep track if center hold was released early
    rewardH = 0   # initially no reward schedule for high value target
    rewardL = 0   # initialy no reward sheduled for low value target

    limit2d = 1

    color1 = target_colors['purple']#target_colors['purple'] 'magenta' olive# high color
    color2 = target_colors['blue']#target_colors['lightsteelblue'] 'teal' juicyorange #low color

    sequence_generators = ['colored_targets_with_probabilistic_reward','block_probabilistic_reward','colored_targets_with_randomwalk_reward','randomwalk_probabilistic_reward']
    
    def __init__(self, *args, **kwargs):
        super(ManualControlFreeChoice, self).__init__(*args, **kwargs)
        self.cursor_visible = True

        # Add graphics models for the plant and targets to the window

        self.plant = plantlist[self.plant_type]
        self.plant_vis_prev = True

        # Add graphics models for the plant and targets to the window
        if hasattr(self.plant, 'graphics_models'):
            for model in self.plant.graphics_models:
                self.add_model(model)

        self.current_pt=np.zeros([3]) #keep track of current pt
        self.last_pt=np.zeros([3]) #kee
        ## Declare cursor
        #self.dtype.append(('cursor', 'f8', (3,)))
        if 0: #hasattr(self.arm, 'endpt_cursor'):
            self.cursor = self.arm.endpt_cursor
        else:
            self.cursor = Sphere(radius=self.cursor_radius, color=self.cursor_color)
            self.add_model(self.cursor)
            self.cursor.translate(*self.get_arm_endpoint(), reset=True) 

        ## Instantiate the targets. Target 1 is center target, Target H is target with high probability of reward, Target L is target with low probability of reward.
        self.target1 = Sphere(radius=self.target_radius, color=self.target_color)           # center target
        self.add_model(self.target1)
        self.targetH = Sphere(radius=self.target_radius, color=self.target_color)           # left target
        self.add_model(self.targetH)
        self.targetL = Sphere(radius=self.target_radius, color=self.target_color)           # right target
        self.add_model(self.targetL)

        # Initialize target location variable. Target locations will be updated throughout task.
        self.target_location1 = np.array([0,0,0])
        self.target_locationH = np.array([-self.starting_dist,0,0])
        self.target_locationL = np.array([self.starting_dist,0,0])

        self.target1.translate(*self.target_location1, reset=True)
        self.targetH.translate(*self.target_locationH, reset=True)
        self.targetL.translate(*self.target_locationL, reset=True)

        # Initialize colors for high probability and low probability target.  Color will not change.
        #self.targetH.color = self.color_targets*(1,1,0,.5) + (1 - self.color_targets)*(0,0,1,.5)     # high probability target red if color_targets = 1
        #self.targetL.color = (1 - self.color_targets)*(1,1,0,.5) + self.color_targets*(0,0,1,.5)    # low probability target blue if color_targets = 1
        self.targetH.color = self.color_targets*(1,0,1,0.5) + (1 - self.color_targets)*(1,0.502,0,0.5)  # high is magenta if color_targets = 1, juicyorange otherwise
        self.targetL.color = (1 - self.color_targets)*(1,0,1,0.5) + self.color_targets*(1,0.502,0,0.5)
        '''
        self.targets = [self.targetH, self.targetL]
        for target in self.targets:
            for model in target.graphics_models:
                self.add_model(model)
        '''
        # Initialize target location variable
        self.target_location = np.array([0, 0, 0])

        # Declare any plant attributes which must be saved to the HDF file at the _cycle rate
        for attr in self.plant.hdf_attrs:
            self.add_dtype(*attr)  


    def init(self):
        self.add_dtype('targetH', 'f8', (3,))
        self.add_dtype('targetL','f8', (3,))
        self.add_dtype('reward_scheduleH','f8', (1,))
        self.add_dtype('reward_scheduleL','f8', (1,)) 
        self.add_dtype('target_index', 'i', (1,))
        super(ManualControlFreeChoice, self).init()
        self.trial_allocation = np.zeros(1000)
        for i in range(1,11):
            define_instructed_trials = random.sample(range(100*(i-1),100*(i-1) + 100),self.percentage_instructed_trials)
            self.trial_allocation[define_instructed_trials] = 1

    def _cycle(self):
        ''' Calls any update functions necessary and redraws screen. Runs 60x per second. '''

        ## Run graphics commands to show/hide the arm if the visibility has changed
        if self.plant_type != 'cursor_14x14':
            if self.arm_visible != self.arm_vis_prev:
                self.arm_vis_prev = self.arm_visible
                self.show_object(self.arm, show=self.arm_visible)

        self.move_arm()
        #self.move_plant()

        ## Save plant status to HDF file
        plant_data = self.plant.get_data_to_save()
        for key in plant_data:
            self.task_data[key] = plant_data[key]

        self.update_cursor()
        #if self.cursor_visible:
            #self.task_data['cursor'] = self.cursor.xfm.move.copy()
        #else:
            #if the cursor is not visible, write NaNs into cursor location saved in file
            #self.task_data['cursor'] = np.array([np.nan, np.nan, np.nan])

        if self.plant_type != 'cursor_14x14':
            self.task_data['joint_angles'] = self.get_arm_joints()

            #if self.arm_visible:
                #self.task_data['arm_visible'] = 1
            #else:
                #self.task_data['arm_visible'] = 0

        super(ManualControlFreeChoice, self)._cycle()
        
    ## Plant functions
    def get_cursor_location(self):
        # arm returns it's position as if it was anchored at the origin, so have to translate it to the correct place
        return self.get_arm_endpoint()

    def get_arm_endpoint(self):
        return self.plant.get_endpoint_pos() 

    def set_arm_endpoint(self, pt, **kwargs):
        self.plant.set_endpoint_pos(pt, **kwargs) 

    def set_arm_joints(self, angles):
        self.arm.set_intrinsic_coordinates(angles)

    def get_arm_joints(self):
        return self.arm.get_intrinsic_coordinates()

    def update_cursor(self):
        '''
        Update the cursor's location and visibility status.
        '''
        pt = self.get_cursor_location()
        self.update_cursor_visibility()
        if pt is not None:
            self.move_cursor(pt)

    def move_cursor(self, pt):
        ''' Move the cursor object to the specified 3D location. '''
        # if not hasattr(self.arm, 'endpt_cursor'):
        self.cursor.translate(*pt[:3],reset=True)

    ##    


    ##### HELPER AND UPDATE FUNCTIONS ####

    def move_arm(self):
        ''' Returns the 3D coordinates of the cursor. For manual control, uses
        joystick data. If no joystick data available, returns None'''

        pt = self.joystick.get()
        if len(pt) > 0:
            pt = pt[-1][0]
            #pt[0]=1-pt[0]; #Switch L / R axes
            calib = [0.33,0.5] #Sometimes zero point is subject to drift this is the value of the incoming joystick when at 'rest' 

            if self.joystick_method==0:                
                pos = np.array([(pt[0]-calib[0]), 0, calib[1]-pt[1]])
                pos[0] = pos[0]*36
                pos[2] = pos[2]*24
                self.current_pt = pos

            elif self.joystick_method==1:
                vel=np.array([(pt[0]-calib[0]), 0, calib[1]-pt[1]])
                epsilon = 2*(10**-2) #Define epsilon to stabilize cursor movement
                if sum((vel)**2) > epsilon:
                    self.current_pt=self.last_pt+self.joystick_speed*vel*(1/60) #60 Hz update rate, dt = 1/60
                else:
                    self.current_pt = self.last_pt

                if self.current_pt[0] < -25: self.current_pt[0] = -25
                if self.current_pt[0] > 25: self.current_pt[0] = 25
                if self.current_pt[-1] < -14: self.current_pt[-1] = -14
                if self.current_pt[-1] > 14: self.current_pt[-1] = 14

            self.set_arm_endpoint(self.current_pt)
            self.last_pt = self.current_pt.copy()

    def convert_to_cm(self, val):
        return val/10.0

    def update_cursor_visibility(self):
        ''' Update cursor visible flag to hide cursor if there has been no good data for more than 3 frames in a row'''
        prev = self.cursor_visible
        if self.no_data_count < 3:
            self.cursor_visible = True
            if prev != self.cursor_visible:
            	self.show_object(self.cursor, show=True)
            	self.requeue()
        else:
            self.cursor_visible = False
            if prev != self.cursor_visible:
            	self.show_object(self.cursor, show=False)
            	self.requeue()

    def calc_n_successfultrials(self):
        trialendtimes = np.array([state[1] for state in self.state_log if state[0]=='check reward'])
        return len(trialendtimes)

    def calc_n_rewards(self):
        rewardtimes = np.array([state[1] for state in self.state_log if state[0]=='reward'])
        return len(rewardtimes)

    def calc_trial_num(self):
        '''Calculates the current trial count: completed + aborted trials'''
        trialtimes = [state[1] for state in self.state_log if state[0] in ['wait']]
        return len(trialtimes)-1

    def calc_targetH_num(self):
        '''Calculates the total number of times the high-value target was selected'''
        trialtimes = [state[1] for state in self.state_log if state[0] in ['hold_targetH']]
        return len(trialtimes) - 1

    def calc_rewards_per_min(self, window):
        '''Calculates the Rewards/min for the most recent window of specified number of seconds in the past'''
        rewardtimes = np.array([state[1] for state in self.state_log if state[0]=='reward'])
        if (self.get_time() - self.task_start_time) < window:
            divideby = (self.get_time() - self.task_start_time)/sec_per_min
        else:
            divideby = window/sec_per_min
        return np.sum(rewardtimes >= (self.get_time() - window))/divideby

    def calc_success_rate(self, window):
        '''Calculates the rewarded trials/initiated trials for the most recent window of specified length in sec'''
        trialtimes = np.array([state[1] for state in self.state_log if state[0] in ['reward', 'timeout_penalty', 'hold_penalty']])
        rewardtimes = np.array([state[1] for state in self.state_log if state[0]=='reward'])
        if len(trialtimes) == 0:
            return 0.0
        else:
            return float(np.sum(rewardtimes >= (self.get_time() - window)))/np.sum(trialtimes >= (self.get_time() - window))

    def update_report_stats(self):
        '''Function to update any relevant report stats for the task. Values are saved in self.reportstats,
        an ordered dictionary. Keys are strings that will be displayed as the label for the stat in the web interface,
        values can be numbers or strings. Called every time task state changes.'''
        super(ManualControlFreeChoice, self).update_report_stats()
        self.reportstats['Trial #'] = self.calc_trial_num()
        self.reportstats['Reward/min'] = np.round(self.calc_rewards_per_min(120),decimals=2)
        self.reportstats['High-value target selections'] = self.calc_targetH_num()
        #self.reportstats['Success rate'] = str(np.round(self.calc_success_rate(120)*100.0,decimals=2)) + '%'
        start_time = self.state_log[0][1]
        rewardtimes=np.array([state[1] for state in self.state_log if state[0]=='reward'])
        if len(rewardtimes):
            rt = rewardtimes[-1]-start_time
        else:
            rt= np.float64("0.0")

        sec = str(np.int(np.mod(rt,60)))
        if len(sec) < 2:
            sec = '0'+sec
        self.reportstats['Time Of Last Reward'] = str(np.int(np.floor(rt/60))) + ':' + sec



    #### TEST FUNCTIONS ####
    def _test_enter_center(self, ts):
        #return true if the distance between center of cursor and target is smaller than the cursor radius

        d = np.sqrt((self.cursor.xfm.move[0]-self.target_location1[0])**2 + (self.cursor.xfm.move[1]-self.target_location1[1])**2 + (self.cursor.xfm.move[2]-self.target_location1[2])**2)
        #print 'TARGET SELECTED', self.target_selected
        return d <= self.target_radius - self.cursor_radius

    def _test_enter_targetL(self, ts):
        if self.target_index == 1 and self.LH_target_on[0]==0:
            #return false if instructed trial and this target is not on
            return False
        else:
            #return true if the distance between center of cursor and target is smaller than the cursor radius
            d = np.sqrt((self.cursor.xfm.move[0]-self.target_locationL[0])**2 + (self.cursor.xfm.move[1]-self.target_locationL[1])**2 + (self.cursor.xfm.move[2]-self.target_locationL[2])**2)
            self.target_selected = 'L'
            #print 'TARGET SELECTED', self.target_selected
            return d <= self.target_radius - self.cursor_radius

    def _test_enter_targetH(self, ts):
        if self.target_index ==1 and self.LH_target_on[1]==0:
            return False
        else:
            #return true if the distance between center of cursor and target is smaller than the cursor radius
            d = np.sqrt((self.cursor.xfm.move[0]-self.target_locationH[0])**2 + (self.cursor.xfm.move[1]-self.target_locationH[1])**2 + (self.cursor.xfm.move[2]-self.target_locationH[2])**2)
            self.target_selected = 'H'
            #print 'TARGET SELECTED', self.target_selected
            return d <= self.target_radius - self.cursor_radius
    def _test_leave_early_center(self, ts):
        # return true if cursor moves outside the exit radius (gives a bit of slack around the edge of target once cursor is inside)
        d = np.sqrt((self.cursor.xfm.move[0]-self.target_location1[0])**2 + (self.cursor.xfm.move[1]-self.target_location1[1])**2 + (self.cursor.xfm.move[2]-self.target_location1[2])**2)
        rad = self.target_radius - self.cursor_radius
        return d > rad

    def _test_leave_early_L(self, ts):
        # return true if cursor moves outside the exit radius (gives a bit of slack around the edge of target once cursor is inside)
        d = np.sqrt((self.cursor.xfm.move[0]-self.target_locationL[0])**2 + (self.cursor.xfm.move[1]-self.target_locationL[1])**2 + (self.cursor.xfm.move[2]-self.target_locationL[2])**2)
        rad = self.target_radius - self.cursor_radius
        return d > rad

    def _test_leave_early_H(self, ts):
        # return true if cursor moves outside the exit radius (gives a bit of slack around the edge of target once cursor is inside)
        d = np.sqrt((self.cursor.xfm.move[0]-self.target_locationH[0])**2 + (self.cursor.xfm.move[1]-self.target_locationH[1])**2 + (self.cursor.xfm.move[2]-self.target_locationH[2])**2)
        rad = self.target_radius - self.cursor_radius
        return d > rad

    def _test_hold_center_complete(self, ts):
        return ts>=self.hold_time
    
    def _test_hold_complete(self, ts):
        return ts>=self.hold_time

    def _test_timeout(self, ts):
        return ts>self.timeout_time

    def _test_timeout_penalty_end(self, ts):
        return ts>self.timeout_penalty_time

    def _test_hold_penalty_end(self, ts):
        return ts>self.hold_penalty_time

    def _test_trial_complete(self, ts):
        #return self.target_index==self.chain_length-1
        return not self.timedout

    def _test_trial_incomplete(self, ts):
        return (not self._test_trial_complete(ts)) and (self.tries<self.max_attempts)

    def _test_trial_abort(self, ts):
        return (not self._test_trial_complete(ts)) and (self.tries==self.max_attempts)

    def _test_yes_reward(self,ts):
        if self.target_selected == 'H':
            #reward_assigned = self.targs[0,1]
            reward_assigned = self.rewardH
        else:
            #reward_assigned = self.targs[1,1]
            reward_assigned = self.rewardL
        if self.reward_SmallLarge==1:
            self.reward_time = reward_assigned*self.reward_time_large + (1 - reward_assigned)*self.reward_time_small   # update reward time if using Small/large schedule
            reward_assigned = 1    # always rewarded
        return bool(reward_assigned)

    def _test_no_reward(self,ts):
        if self.target_selected == 'H':
            #reward_assigned = self.targs[0,1]
            reward_assigned = self.rewardH
        else:
            #reward_assigned = self.targs[1,1]
            reward_assigned = self.rewardL
        if self.reward_SmallLarge==True:
            self.reward_time = reward_assigned*self.reward_time_large + (1 - reward_assigned)*self.reward_time_small   # update reward time if using Small/large schedule
            reward_assigned = 1    # always rewarded
        return bool(not reward_assigned)

    def _test_reward_end(self, ts):
        time_ended = (ts > self.reward_time)
        self.reward_counter = self.reward_counter + 1
        return time_ended

    def _test_stop(self, ts):
        if self.session_length > 0 and (time.time() - self.task_start_time) > self.session_length:
            self.end_task()
        return self.stop

    #### STATE FUNCTIONS ####

    def show_object(self, obj, show=False):
        '''
        Show or hide an object
        '''
        if show:
            obj.attach()
        else:
            obj.detach()
        self.requeue()


    def _start_wait(self):
        super(ManualControlFreeChoice, self)._start_wait()
        self.tries = 0
        self.target_index = 0     # indicator for instructed or free-choice trial
        #hide targets
        self.show_object(self.target1, False)
        self.show_object(self.targetL, False)
        self.show_object(self.targetH, False)


        #get target positions and reward assignments for this trial
        self.targs = self.next_trial
        if self.plant_type != 'cursor_14x14' and np.random.rand() < self.arm_hide_rate:
            self.arm_visible = False
        else:
            self.arm_visible = True
        #self.chain_length = self.targs.shape[0] #Number of sequential targets in a single trial

        #self.task_data['target'] = self.target_locationH.copy()
        assign_reward = np.random.randint(0,100,size=2)
        self.rewardH = np.greater(self.targs[0,1],assign_reward[0])
        #print 'high value target reward prob', self.targs[0,1]
        self.rewardL = np.greater(self.targs[1,1],assign_reward[1])

        
        #print 'TARGET GENERATOR', self.targs[0,]
        self.task_data['targetH'] = self.targs[0,].copy()
        self.task_data['reward_scheduleH'] = self.rewardH.copy()
        self.task_data['targetL'] = self.targs[1,].copy()
        self.task_data['reward_scheduleL'] = self.rewardL.copy()
        
        self.requeue()

    def _start_center(self):

        #self.target_index += 1

        #set target colors 
        self.target1.color = (1,0,0,.5)      # center target red
        #self.targetH.color = self.color_targets*(1,1,0,.5) + (1 - self.color_targets)*(0,0,1,.5)     # high probability target red if color_targets = 1
        #self.targetL.color = (1 - self.color_targets)*(1,1,0,.5) + self.color_targets*(0,0,1,.5)    # low probability target blue if color_targets = 1
        self.targetH.color = self.color_targets*self.color1 + (1 - self.color_targets)*self.color2 # high is magenta if color_targets = 1, juicyorange otherwise
        self.targetL.color = (1 - self.color_targets)*self.color1 + self.color_targets*self.color2
        self.show_object(self.target1, True)
        self.show_object(self.cursor, True)
        
        # Third argument in self.targs determines if target is on left or right
        # First argument in self.targs determines if location is offset to farther distances
        offsetH = (2*self.targs[0,2] - 1)*(self.starting_dist + self.location_offset_allowed*self.targs[0,0]*4.0)
        moveH = np.array([offsetH,0,0]) 
        offsetL = (2*self.targs[1,2] - 1)*(self.starting_dist + self.location_offset_allowed*self.targs[1,0]*4.0)
        moveL = np.array([offsetL,0,0])

        self.targetL.translate(*moveL, reset=True) 
        #self.targetL.move_to_position(*moveL, reset=True)           
        ##self.targetL.translate(*self.targs[self.target_index], reset=True)
        self.show_object(self.targetL, True)
        self.target_locationL = self.targetL.xfm.move

        self.targetH.translate(*moveH, reset=True)
        #self.targetR.move_to_position(*moveR, reset=True)
        ##self.targetR.translate(*self.targs[self.target_index], reset=True)
        self.show_object(self.targetH, True)
        self.target_locationH = self.targetH.xfm.move


        # Insert instructed trials within free choice trials
        if self.trial_allocation[self.calc_trial_num()] == 1:
        #if (self.calc_trial_num() % 10) < (self.percentage_instructed_trials/10):
            self.target_index = 1    # instructed trial
            leftright_coinflip = np.random.randint(0,2)
            if leftright_coinflip == 0:
                self.show_object(self.targetL, False)
                self.LH_target_on = (0, 1)
            else:
                self.show_object(self.targetH, False)
                self.LR_coinflip = 0
                self.LH_target_on = (1, 0)
        else:
            self.target_index = 2   # free-choice trial

        self.cursor_visible = True
        self.task_data['target_index'] = self.target_index
        self.requeue()

    def _start_target(self):

    	#self.target_index += 1

        #move targets to current location and set location attribute.  Target1 (center target) position does not change.                    
        
        self.show_object(self.target1, False)
        #self.target_location1 = self.target1.xfm.move
        self.show_object(self.cursor, True)
       
        self.update_cursor()
        self.requeue()

    def _start_hold_center(self):
        self.show_object(self.target1, True)
        self.timedout = False
        self.requeue()

    def _start_hold_targetL(self):
        #make next target visible unless this is the final target in the trial
        #if 1 < self.chain_length:
            #self.targetL.translate(*self.targs[self.target_index+1], reset=True)
         #   self.show_object(self.targetL, True)
         #   self.requeue()
        self.show_object(self.targetL, True)
        self.timedout = False
        self.requeue()

    def _start_hold_targetH(self):
        #make next target visible unless this is the final target in the trial
        #if 1 < self.chain_length:
            #self.targetR.translate(*self.targs[self.target_index+1], reset=True)
         #   self.show_object(self.targetR, True)
          #  self.requeue()
        self.show_object(self.targetH, True)
        self.timedout = False
        self.requeue()

    def _end_hold_center(self):
        self.target1.radius = 0.7*self.target_radius # color target green
    
    def _end_hold_targetL(self):
        self.targetL.color = (0,1,0,0.5)    # color target green

    def _end_hold_targetH(self):
        self.targetH.color = (0,1,0,0.5)    # color target green

    def _start_hold_penalty(self):
    	#hide targets
        #self.targetH.color = self.color_targets*(1,1,0,.5) + (1 - self.color_targets)*(0,0,1,.5)     # high probability target red if color_targets = 1
        #self.targetL.color = (1 - self.color_targets)*(1,1,0,.5) + self.color_targets*(0,0,1,.5)    # low probability target blue if color_targets = 1
        self.targetH.color = self.color_targets*self.color1 + (1 - self.color_targets)*self.color2  # high is magenta if color_targets = 1, juicyorange otherwise
        self.targetL.color = (1 - self.color_targets)*self.color1 + self.color_targets*self.color2
        self.show_object(self.target1, False)
        self.show_object(self.targetL, False)
        self.show_object(self.targetH, False)
        self.timedout = True
        self.requeue()
        self.tries += 1
        #self.target_index = -1
    
    def _start_timeout_penalty(self):
    	#hide targets
        self.show_object(self.target1, False)
        self.show_object(self.targetL, False)
        self.show_object(self.targetH, False)
        self.timedout = True
        self.requeue()
        self.tries += 1
        #self.target_index = -1


    def _start_targ_transition(self):
        #hide targets

        self.show_object(self.target1, False)
        self.show_object(self.targetL, False)
        self.show_object(self.targetH, False)
        self.requeue()

    def _start_check_reward(self):
        #hide targets
        self.show_object(self.target1, False)
        self.show_object(self.targetL, False)
        self.show_object(self.targetH, False)
        self.requeue()

    def _start_reward(self):
        #super(ManualControlFreeChoice, self)._start_reward()
        if self.target_selected == 'L':
            self.show_object(self.targetL, True)  
            #reward_assigned = self.targs[1,1]
        else:
            self.show_object(self.targetH, True)
            #reward_assigned = self.targs[0,1]
        #self.reward_counter = self.reward_counter + float(reward_assigned)
        self.requeue()

    @staticmethod
    def colored_targets_with_probabilistic_reward(length=2000, boundaries=(-18,18,-10,10,-15,15),reward_high_prob=80,reward_low_prob=40): #! 20210102 HMS changed length from 1000 to 2000

        """
        Generator should return array of ntrials x 2 x 3. The second dimension is for each target.
        For example, first is the target with high probability of reward, and the second 
        entry is for the target with low probability of reward.  The third dimension holds three variables indicating 
        position offset (yes/no), reward probability (fixed in this case), and location (binary returned where the
        ouput indicates either left or right).
        UPDATE: CHANGED SO THAT THE SECOND DIMENSION CARRIES THE REWARD PROBABILITY RATHER THAN THE REWARD SCHEDULE
        """

        position_offsetH = np.random.randint(2,size=(1,length))
        position_offsetL = np.random.randint(2,size=(1,length))
        location_int = np.random.randint(2,size=(1,length))

        # coin flips for reward schedules, want this to be elementwise comparison
        #assign_rewardH = np.random.randint(0,100,size=(1,length))
        #assign_rewardL = np.random.randint(0,100,size=(1,length))
        high_prob = reward_high_prob*np.ones((1,length))
        low_prob = reward_low_prob*np.ones((1,length))
        
        #reward_high = np.greater(high_prob,assign_rewardH)
        #reward_low = np.greater(low_prob,assign_rewardL)

        pairs = np.zeros([length,2,3])
        pairs[:,0,0] = position_offsetH
        #pairs[:,0,1] = reward_high
        pairs[:,0,1] = high_prob
        pairs[:,0,2] = location_int

        pairs[:,1,0] = position_offsetL
        #pairs[:,1,1] = reward_low
        pairs[:,1,1] = low_prob
        pairs[:,1,2] = 1 - location_int

        return pairs

    @staticmethod
    def block_probabilistic_reward(length=1000, boundaries=(-18,18,-10,10,-15,15),reward_high_prob=80,reward_low_prob=40):
        pairs = colored_targets_with_probabilistic_reward(length=length, boundaries=boundaries,reward_high_prob=reward_high_prob,reward_low_prob=reward_low_prob)
        return pairs

    @staticmethod
    def colored_targets_with_randomwalk_reward(length=1000,reward_high_prob=80,reward_low_prob=40,reward_high_span = 20, reward_low_span = 20,step_size_mean = 0, step_size_var = 1):

        """
        Generator should return array of ntrials x 2 x 3. The second dimension is for each target.
        For example, first is the target with high probability of reward, and the second 
        entry is for the target with low probability of reward.  The third dimension holds three variables indicating 
        position offset (yes/no), reward probability, and location (binary returned where the
        ouput indicates either left or right).  The variables reward_high_span and reward_low_span indicate the width
        of the range that the high or low reward probability are allowed to span respectively, e.g. if reward_high_prob
        is 80 and reward_high_span is 20, then the reward probability for the high value target will be bounded
        between 60 and 100 percent.
        """

        position_offsetH = np.random.randint(2,size=(1,length))
        position_offsetL = np.random.randint(2,size=(1,length))
        location_int = np.random.randint(2,size=(1,length))

        # define variables for increments: amount of increment and in which direction (i.e. increasing or decreasing)
        assign_rewardH = np.random.randn(1,length)
        assign_rewardL = np.random.randn(1,length)
        assign_rewardH_direction = np.random.randn(1,length)
        assign_rewardL_direction = np.random.randn(1,length)

        r_0_high = reward_high_prob
        r_0_low = reward_low_prob
        r_lowerbound_high = r_0_high - (reward_high_span/2)
        r_upperbound_high = r_0_high + (reward_high_span/2)
        r_lowerbound_low = r_0_low - (reward_low_span/2)
        r_upperbound_low = r_0_low + (reward_low_span/2)
        
        reward_high = np.zeros(length)
        reward_low = np.zeros(length)
        reward_high[0] = r_0_high
        reward_low[0] = r_0_low

        eps_high = assign_rewardH*step_size_mean + [2*(assign_rewardH_direction > 0) - 1]*step_size_var
        eps_low = assign_rewardL*step_size_mean + [2*(assign_rewardL_direction > 0) - 1]*step_size_var

        eps_high = eps_high.ravel()
        eps_low = eps_low.ravel()

        for i in range(1,length):
            '''
            assign_rewardH_direction = np.random.randn(1)
            assign_rewardL_direction = np.random.randn(1)
            assign_rewardH = np.random.randn(1)
            if assign_rewardH_direction[i-1,] < 0:
                eps_high = step_size_mean*assign_rewardH[i-1] - step_size_var
            else:
                eps_high = step_size_mean*assign_rewardH[i-1] + step_size_var
            if assign_rewardL_direction[i] < 0:
                eps_low = step_size_mean*assign_rewardL[i] - step_size_var
            else:
                eps_low = step_size_mean*assign_rewardL[i] + step_size_var
            '''
            reward_high[i] = reward_high[i-1] + eps_high[i-1]
            reward_low[i] = reward_low[i-1] + eps_low[i-1]

            reward_high[i] = (r_lowerbound_high < reward_high[i] < r_upperbound_high)*reward_high[i] + (r_lowerbound_high > reward_high[i])*(r_lowerbound_high+ eps_high[i-1]) + (r_upperbound_high < reward_high[i])*(r_upperbound_high - eps_high[i-1])
            reward_low[i] = (r_lowerbound_low < reward_low[i] < r_upperbound_low)*reward_low[i] + (r_lowerbound_low > reward_low[i])*(r_lowerbound_low+ eps_low[i-1]) + (r_upperbound_low < reward_low[i])*(r_upperbound_low - eps_low[i-1])

        pairs = np.zeros([length,2,3])
        pairs[:,0,0] = position_offsetH
        pairs[:,0,1] = reward_high
        pairs[:,0,2] = location_int

        pairs[:,1,0] = position_offsetL
        pairs[:,1,1] = reward_low
        pairs[:,1,2] = 1 - location_int

        return pairs

    @staticmethod
    def randomwalk_probabilistic_reward(length=1000,reward_high_prob=80,reward_low_prob=40,reward_high_span = 20, reward_low_span = 20,step_size_mean = 0, step_size_var = 1):
        pairs = colored_targets_with_randomwalk_reward(length=length,reward_high_prob=reward_high_prob,reward_low_prob=reward_low_prob,reward_high_span = reward_high_span, reward_low_span = reward_low_span,step_size_mean = step_size_mean, step_size_var = step_size_var)
        return pairs

class FreeChoicePilotTask(ManualControlFreeChoice):
    '''
    This is the free-choice probabilistic reward task broken into three phases.
    Phase 1: Directed choice of target with variable position, every 5th target is free choice. 100 trials.
    Phase 2: Directed choice of target with variable position, stimulation applied for lower probability target.  100 trials.
    Phase 3: Free choice of target with variable position, every 5th target is directed choice.  100 trials.
    '''
    #stimulation_pulse_length = traits.Float(200, desc="Pulse length in us")
    #stimulation_frequency = traits.Float(200, desc="Stimulation frequency")
    
    instructed_target = 0
    intervention_trial = False  # indicate if this is a trial with stimulation or not
    #com = comedi.comedi_open('/dev/comedi0')
    pulse_count = 0  #initializing number of pulses that have occured
    hold_time = traits.Float(.2, desc="Length of hold required at targets")
    percentage_instructed_trials = traits.Int(10,desc='Percentage of instructed trials')

#HMS & YZ - added settable traits and sequence generators for MCFC_Pseudo
    background = (0,0,0,1)
    shoulder_anchor = np.array([2., 0., -15.]) # Coordinates of shoulder anchor point on screen
    
    arm_visible = traits.Bool(True, desc='Specifies whether entire arm is displayed or just endpoint')
    
    cursor_radius = traits.Float(.5, desc="Radius of cursor")
    cursor_color = (.5,0,.5,1)

    joystick_method = traits.Float(1,desc="1: Normal velocity, 0: Position control")
    joystick_speed = traits.Float(20, desc="Speed of cursor")

    plant_type_options = list(plantlist.keys())
    plant_type = traits.OptionsList(*plantlist, bmi3d_input_options=list(plantlist.keys()))
    starting_pos = (5, 0, 5)
    # window_size = (1280*2, 1024)
    window_size = traits.Tuple((1920*2, 1080), desc='window size')
    percentage_instructed_trials = traits.Int(10,desc='Percentage of instructed trials: multiples of 10')
    print('in task')

    status = dict(
        #wait = dict(start_trial="target", stop=None),
        wait = dict(start_trial="center", stop=None),
        #center = dict(enter_center="hold_center", timeout="timeout_penalty", enter_center_and_stimulate = "pulse",stop=None),
        center = dict(enter_center="hold_center", timeout="timeout_penalty", enter_center_and_stimulate = "hold_center_and_stimulate",stop=None),
        hold_center = dict(leave_early_center = "hold_penalty",hold_center_complete="target", timeout="timeout_penalty", stop=None),
        hold_center_and_stimulate = dict(leave_early_center = "hold_penalty",hold_center_complete="target", timeout="timeout_penalty", stop=None),
        #pulse = dict(pulse_end="interpulse_period", stop=None),
        #interpulse_period = dict(another_pulse="pulse", pulse_train_complete = "target", stop=None),
        target = dict(enter_targetL="hold_targetL", enter_targetH = "hold_targetH", timeout="timeout_penalty", stop=None),
        hold_targetH = dict(leave_early_H="hold_penalty", hold_complete="targ_transition"),
        hold_targetL = dict(leave_early_L="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="check_reward",trial_abort="wait", trial_incomplete="center"),
        check_reward = dict(yes_reward="reward",no_reward="wait"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )

    sequence_generators = ['colored_targets_with_probabilistic_reward','block_probabilistic_reward','colored_targets_with_randomwalk_reward','randomwalk_probabilistic_reward']
    
    def __init__(self, *args, **kwargs):
        super(FreeChoicePilotTask, self).__init__(*args, **kwargs)
        #print "num pulses", self.number_of_pulses
        # random assignment of instructed trials by doing sampling without replacement over a block of 100 trials
        self.trial_allocation = np.zeros(1000)
        #! self.port = serial.Serial(glob.glob("/dev/ttyACM*")[0], baudrate=115200) #original line of code
        #self.port = serial.Serial(glob.glob("/dev/arduino_neurosync', baudrate=9600)"))
        for i in range(1,11):
            define_instructed_trials = random.sample(range(100*(i-1),100*(i-1) + 100),self.percentage_instructed_trials)
            self.trial_allocation[define_instructed_trials] = 1

    def _start_center(self):

        #self.target_index += 1
        self.pulse_count = 0    # resert count of number of stimulation pulses
        #set target colors 
        self.target1.color = (1,0,0,.5)      # center target red
        #self.targetH.color = self.color_targets*(1,1,0,.5) + (1 - self.color_targets)*(0,0,1,.5)     # high probability target red if color_targets = 1
        #self.targetL.color = (1 - self.color_targets)*(1,1,0,.5) + self.color_targets*(0,0,1,.5)    # low probability target blue if color_targets = 1
        self.targetH.color = self.color_targets*self.color1 + (1 - self.color_targets)*self.color2  # high is magenta if color_targets = 1, juicyorange otherwise
        self.targetL.color = (1 - self.color_targets)*self.color1 + self.color_targets*self.color2      
        
        self.show_object(self.target1, True)
        self.show_object(self.targetH, False)
        self.show_object(self.targetL, False)
        self.show_object(self.cursor, True)
        #self.target_location1 = self.target1.xfm.move
        #print 'TARGET GENERATOR', self.targs[0]
        # offset targets from original location according to random generator
        '''
        if 100 < self.calc_trial_num() < 201:
            self.instructed_target = np.random.randint(0,2)
            if self.instructed_target == 1:
                self.intervention_trial = True
            else:
                self.intervention_trial = False
        else:
            self.intervention_trial = False
        '''
        # Third argument in self.targs determines if target is on left or right
        # First argument in self.targs determines if location is offset to farther distances
        offsetH = (2*self.targs[0,2] - 1)*(self.starting_dist + self.location_offset_allowed*self.targs[0,0]*4.0)
        moveH = np.array([offsetH,0,0]) 
        offsetL = (2*self.targs[1,2] - 1)*(self.starting_dist + self.location_offset_allowed*self.targs[1,0]*4.0)
        moveL = np.array([offsetL,0,0])

        self.targetL.translate(*moveL, reset=True) 
        #self.targetL.move_to_position(*moveL, reset=True)           
        ##self.targetL.translate(*self.targs[self.target_index], reset=True)
        self.show_object(self.targetL, True)
        self.target_locationL = self.targetL.xfm.move

        self.targetH.translate(*moveH, reset=True)
        #self.targetR.move_to_position(*moveR, reset=True)
        ##self.targetR.translate(*self.targs[self.target_index], reset=True)
        self.show_object(self.targetH, True)
        self.target_locationH = self.targetH.xfm.move

        if self.calc_trial_num() < 101:  
            # use commented code if we're able to keep track of instructed versus free choice trials
            #check_if_instructed = np.random.randint(0,100)
            #if check_if_instructed < self.percentage_instructed_trials:
            self.intervention_trial = False
            if self.trial_allocation[self.calc_trial_num()] == 1:
            #if (self.calc_trial_num() % 10) < (self.percentage_instructed_trials/10):
                self.target_index = 1
                
                leftright_coinflip = np.random.randint(0,2)
                if leftright_coinflip == 0:
                    self.show_object(self.targetL, False)
                    self.LH_target_on = (0, 1)
                else:
                    self.show_object(self.targetH, False)
                    self.LH_target_on = (1, 0)
            else:
                self.target_index = 2
                self.LH_target_on = (1, 1)

        elif self.calc_trial_num() < 201:

            # All instructed trials during phase II (intervention phase)
            self.instructed_target = np.random.randint(0,2)    # choose whether high or low value target is shown
            self.target_index = 1
            if self.instructed_target == 0:
                self.show_object(self.targetL, False)
                self.LH_target_on = (0, 1)
                self.intervention_trial = False    # when high value target is shown, no stimulation is performed
            else:
                self.show_object(self.targetH, False)
                self.LH_target_on = (1, 0)
                self.intervention_trial = True     # when low value target is shown, stimulation is performed
                #print 'stim delivered', self.intervention_trial
        else:
            # All free-choice trials during phase III (test phase)
            if self.trial_allocation[self.calc_trial_num()] == 1:
            #if (self.calc_trial_num() % 10) < (self.percentage_instructed_trials/10):
                self.target_index = 1
                leftright_coinflip = 1  #always low value target show
                #leftright_coinflip = np.random.randint(0,2)
                if leftright_coinflip == 0:
                    self.show_object(self.targetL, False)
                    self.LH_target_on = (0, 1)
                    self.intervention_trial = False
                else:
                    self.show_object(self.targetH, False)
                    self.LH_target_on = (1, 0)
                    self.intervention_trial = True
            else:
                self.target_index = 2
                self.LH_target_on = (1, 1)
                self.intervention_trial = False
            '''
            self.intervention_trial = False
            self.target_index = 2
            self.LH_target_on = (1, 1)
            '''       
        self.task_data['target_index'] = self.target_index
        '''
        
        import comedi
        com = comedi.comedi_open('/dev/comedi0')
        subdevice = 0
        write_mask = 0x800000
        val = 0x000000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)
        '''

        self.requeue()
      

    def _test_enter_center(self, ts):
        #return true if the distance between center of cursor and target is smaller than the cursor radius

        d = np.sqrt((self.cursor.xfm.move[0]-self.target_location1[0])**2 + (self.cursor.xfm.move[1]-self.target_location1[1])**2 + (self.cursor.xfm.move[2]-self.target_location1[2])**2)
        #print 'TARGET SELECTED', self.target_selected
        return (d <= self.target_radius - self.cursor_radius) and (not self.intervention_trial)

    def _test_enter_center_and_stimulate(self, ts):
        #return true if the distance between center of cursor and target is smaller than the cursor radius

        d = np.sqrt((self.cursor.xfm.move[0]-self.target_location1[0])**2 + (self.cursor.xfm.move[1]-self.target_location1[1])**2 + (self.cursor.xfm.move[2]-self.target_location1[2])**2)
        #print 'TARGET SELECTED', self.target_selected
        return (d <= self.target_radius - self.cursor_radius) and (self.intervention_trial)

    def _start_hold_center_and_stimulate(self):
        self.show_object(self.target1, True)
        self.timedout = False
        '''
        At the start of the stimulation state, send TTL pulse
        '''
        self.port.write(b'x')
        '''
        import comedi
        com = comedi.comedi_open('/dev/comedi0')
        subdevice = 0
        write_mask = 0x800000
        val = 0x800000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)
        self.pulse_count = self.pulse_count + 1
        '''
        self.requeue()
    
    def _start_target(self):

        #self.target_index += 1
        self.port.write(b'o')
        '''
        import comedi
        com = comedi.comedi_open('/dev/comedi0')
        subdevice = 0
        write_mask = 0x800000
        val = 0x000000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)
        '''
        #set target colors 
        self.target1.color = (1,0,0,.5)      # center target red
        self.target_index = 2                # free choice trial = 2, instructed = 1
        #move targets to current location and set location attribute.  Target1 (center target) position does not change.                    
        
        self.show_object(self.target1, False)
        #self.target_location1 = self.target1.xfm.move
        self.show_object(self.cursor, True)
        

        self.cursor_visible = True
        self.update_cursor()
        self.requeue()

    def _start_hold_penalty(self):
        #hide targets
        self.show_object(self.target1, False)
        self.show_object(self.targetL, False)
        self.show_object(self.targetH, False)
        self.timedout = True
        self.requeue()
        self.tries += 1
        
        self.port.write(b'o')
        '''
        import comedi
        com = comedi.comedi_open('/dev/comedi0')
        subdevice = 0
        write_mask = 0x800000
        val = 0x000000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)
        '''

    def _start_timeout_penalty(self):
        #hide targets
        self.show_object(self.target1, False)
        self.show_object(self.targetL, False)
        self.show_object(self.targetH, False)
        self.timedout = True
        self.requeue()
        self.tries += 1

        '''
        import comedi
        com = comedi.comedi_open('/dev/comedi0')
        subdevice = 0
        write_mask = 0x800000
        val = 0x000000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)
        '''
    @staticmethod
    def colored_targets_with_block10_probabilistic_reward(length=1000, boundaries=(-18,18,-10,10,-15,15),reward_high_prob=80,reward_low_prob=40):
        
        position_offsetH = np.random.randint(2,size=(1,length))
        position_offsetL = np.random.randint(2,size=(1,length))
        location_int = np.random.randint(2,size=(1,length))
        reward_low = np.zeros((1,length))
        reward_low = np.ravel(reward_low)
        reward_high = np.zeros((1,length))
        reward_high = np.ravel(reward_high)
        num_blocks = int(length/10)
        num_low = int(reward_low_prob/10)
        num_high = int(reward_high_prob/10)
        for i in range(0,num_blocks):
            low_rewarded_trials = random.sample(range(10*i,10*i + 10),num_low)
            high_rewarded_trials = random.sample(range(10*i,10*i + 10),num_high)
            reward_low[low_rewarded_trials] = 1
            reward_high[high_rewarded_trials] = 1
        
        pairs = np.zeros([length,2,3])
        pairs[:,0,0] = position_offsetH
        pairs[:,0,1] = reward_high
        #pairs[:,0,1] = high_prob
        pairs[:,0,2] = location_int

        pairs[:,1,0] = position_offsetL
        pairs[:,1,1] = reward_low
        #pairs[:,1,1] = low_prob
        pairs[:,1,2] = 1 - location_int

        return pairs
    
    @staticmethod
    def block10_probabilistic_reward(boundaries=(-18,18,-10,10,-15,15),length=1000,reward_high_prob=80,reward_low_prob=40):
        pairs = colored_targets_with_block10_probabilistic_reward(length=length, boundaries=boundaries,reward_high_prob=reward_high_prob,reward_low_prob=reward_low_prob)
        return pairs

class DelayedReachTask(ManualControlFreeChoice):
    '''
    This is a delayed reach task. The task begins with an initial hold at a center target, followed with
    the presentation of the peripheral target. Stimulation is delivered at a random point after the peripheral
    target is shown. 
    '''
    
    instructed_target = 0
    intervention_trial = False  # indicate if this is a trial with stimulation or not
    pulse_count = 0  #initializing number of pulses that have occured
    center_hold_time = traits.Float(.2, desc="Length of hold required at targets")
    min_preparatory_hold = traits.Float(.2, desc="Min time the peripheral target is shown")
    max_preparatory_hold = traits.Float(.4, desc="Max time the peripheral target is shown")
    stim_duration = traits.Float(0.067, desc="Length of stimulation delivered during hold")
    delay_stim = traits.Float(0.075, desc="Min delay after peripheral target is shown before stim")
    peripheral_hold_time = traits.Float(.2, desc="Length of hold required at peripheral targets")
    percentage_stim_trials = traits.Float(50, desc="Percentage of trials with stimulation")
    
    status = dict(
        wait = dict(start_trial="center", stop=None),
        center = dict(enter_center="hold_center", timeout="timeout_penalty", stop=None),
        hold_center = dict(leave_early_center = "hold_penalty",hold_center_complete="first_prep_hold", timeout="timeout_penalty", stop=None),
        first_prep_hold = dict(leave_early_center = "hold_penalty",hold_center_complete="stim_hold", timeout="timeout_penalty", stop=None),
        stim_hold = dict(leave_early_center = "hold_penalty",hold_center_complete="second_prep_hold", timeout="timeout_penalty", stop=None),
        second_prep_hold = dict(leave_early_center = "hold_penalty",hold_center_complete="target", timeout="timeout_penalty", stop=None),
        target = dict(enter_targetH = "hold_targetH", timeout="timeout_penalty", stop=None),
        hold_targetH = dict(leave_early_H="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="reward",trial_abort="wait", trial_incomplete="center"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )
    
    def __init__(self, *args, **kwargs):
        super(DelayedReachTask, self).__init__(*args, **kwargs)
        #print "num pulses", self.number_of_pulses
        # random assignment of instructed trials by doing sampling without replacement over a block of 100 trials
        self.trial_allocation = np.zeros(1000)
        self.port = serial.Serial("/dev/arduino_neurosync", baudrate=115200)
        self.preparatory_hold = self.hold_time
        self.stim_start = self.hold_time

    def _start_center(self):

        # Set hold time to the initial hold time
        self.hold_time = self.center_hold_time
        print("Center hold:", self.hold_time)

        self.target1.color = (1,0,0,.5)      # center target red
        #self.targetH.color = self.color_targets*self.color1 + (1 - self.color_targets)*self.color2  # high is magenta if color_targets = 1, juicyorange otherwise
        self.targetH.color = (1,0,0,0.5)

        self.show_object(self.target1, True)
        self.show_object(self.targetH, False)
        #self.show_object(self.targetL, False)
        self.show_object(self.cursor, True)
        
        # Third argument in self.targs determines if target is on left or right
        # First argument in self.targs determines if location is offset to farther distances
        offsetH = (2*self.targs[0,2] - 1)*(self.starting_dist + self.location_offset_allowed*self.targs[0,0]*4.0)
        moveH = np.array([offsetH,0,0]) 
        
        self.targetH.translate(*moveH, reset=True)
        self.show_object(self.targetH, False)
        self.target_locationH = self.targetH.xfm.move

        self.LH_target_on = (0, 1)
        check_if_stim = np.random.randint(0,100)
        if check_if_stim < self.percentage_stim_trials:  
            self.intervention_trial = True
        else:
            self.intervention_trial = False

        self.task_data['target_index'] = self.target_index
        self.requeue()

    def _start_first_prep_hold(self):

        # Set hold time to first hold period
        self.preparatory_hold = self.min_preparatory_hold + 0.001*np.random.randint(0,1000*(self.max_preparatory_hold - self.min_preparatory_hold))
        self.stim_start = 0.001*np.random.randint(0, 1000*(self.preparatory_hold - self.delay_stim - self.stim_duration))
        self.hold_time = self.delay_stim + self.stim_start
        print("First hold:", self.hold_time)

        self.target1.color = (1,0,0,.5)      # center target red
        #self.targetH.color = self.color_targets*self.color1 + (1 - self.color_targets)*self.color2  # high is magenta if color_targets = 1, juicyorange otherwise
        self.targetH.color = (1,0,0,0.5)

        self.show_object(self.target1, True)
        self.show_object(self.targetH, True)
        self.show_object(self.cursor, True)
        
        '''
        # Third argument in self.targs determines if target is on left or right
        # First argument in self.targs determines if location is offset to farther distances
        offsetH = (2*self.targs[0,2] - 1)*(self.starting_dist + self.location_offset_allowed*self.targs[0,0]*4.0)
        moveH = np.array([offsetH,0,0]) 
        
        self.targetH.translate(*moveH, reset=True)
        self.show_object(self.targetH, True)
        self.target_locationH = self.targetH.xfm.move
        self.LH_target_on = (0, 1)
        check_if_stim = np.random.randint(0,100)
        if check_if_stim < self.percentage_stim_trials:  
            self.intervention_trial = True
        else:
            self.intervention_trial = False
        '''
        self.task_data['target_index'] = self.target_index
        self.requeue()

    def _start_stim_hold(self):

        # Set hold time to first hold period
        #preparatory_hold = self.min_preparatory_hold + 0.001*np.random.randint(0,1000*(self.max_preparatory_hold - self.min_preparatory_hold))
        #stim_start = 0.001*np.random.randint(0, 1000*(preparatory_hold - self.delay_stim - self.stim_duration))
        self.hold_time = self.stim_duration
        print("Stim hold:", self.hold_time)

        self.target1.color = (1,0,0,.5)      # center target red
        #self.targetH.color = self.color_targets*self.color1 + (1 - self.color_targets)*self.color2  # high is magenta if color_targets = 1, juicyorange otherwise
        self.targetH.color = (1,0,0,0.5)

        self.show_object(self.target1, True)
        self.show_object(self.targetH, True)
        self.show_object(self.cursor, True)

        '''
        At the start of the stimulation state, send TTL pulse
        '''
        self.port.write('x')
        
        '''
        # Third argument in self.targs determines if target is on left or right
        # First argument in self.targs determines if location is offset to farther distances
        offsetH = (2*self.targs[0,2] - 1)*(self.starting_dist + self.location_offset_allowed*self.targs[0,0]*4.0)
        moveH = np.array([offsetH,0,0]) 
        
        self.targetH.translate(*moveH, reset=True)
        self.show_object(self.targetH, True)
        self.target_locationH = self.targetH.xfm.move
        self.LH_target_on = (0, 1)
        check_if_stim = np.random.randint(0,100)
        if check_if_stim < self.percentage_stim_trials:  
            self.intervention_trial = True
        else:
            self.intervention_trial = False
        '''
        self.task_data['target_index'] = self.target_index
        self.requeue()

    def _start_second_prep_hold(self):

        # Set hold time to first hold period
        #preparatory_hold = self.min_preparatory_hold + 0.001*np.random.randint(0,1000*(self.max_preparatory_hold - self.min_preparatory_hold))
        #stim_start = 0.001*np.random.randint(0, 1000*(preparatory_hold - self.delay_stim - self.stim_hold))
        self.hold_time = self.preparatory_hold - self.delay_stim - self.stim_start - self.stim_duration
        print("Second after-stim hold:", self.hold_time)

        self.target1.color = (1,0,0,.5)      # center target red
        #self.targetH.color = self.color_targets*self.color1 + (1 - self.color_targets)*self.color2  # high is magenta if color_targets = 1, juicyorange otherwise
        self.targetH.color = (1,0,0,0.5)

        self.show_object(self.target1, True)
        self.show_object(self.targetH, True)
        self.show_object(self.cursor, True)

        '''
        At the end of the stimulation state, send TTL pulse
        '''
        self.port.write('o')
        
        '''
        # Third argument in self.targs determines if target is on left or right
        # First argument in self.targs determines if location is offset to farther distances
        offsetH = (2*self.targs[0,2] - 1)*(self.starting_dist + self.location_offset_allowed*self.targs[0,0]*4.0)
        moveH = np.array([offsetH,0,0]) 
        
        self.targetH.translate(*moveH, reset=True)
        self.show_object(self.targetH, True)
        self.target_locationH = self.targetH.xfm.move
        self.LH_target_on = (0, 1)
        check_if_stim = np.random.randint(0,100)
        if check_if_stim < self.percentage_stim_trials:  
            self.intervention_trial = True
        else:
            self.intervention_trial = False
        '''
        self.task_data['target_index'] = self.target_index
        self.requeue()

    def _test_enter_center(self, ts):
        #return true if the distance between center of cursor and target is smaller than the cursor radius

        d = np.sqrt((self.cursor.xfm.move[0]-self.target_location1[0])**2 + (self.cursor.xfm.move[1]-self.target_location1[1])**2 + (self.cursor.xfm.move[2]-self.target_location1[2])**2)
        #print 'TARGET SELECTED', self.target_selected
        return (d <= self.target_radius - self.cursor_radius)

    
    def _start_target(self):

        #self.target_index += 1
        self.port.write('o')

        #set target colors 
        self.target1.color = (1,0,0,.5)      # center target red
        self.target_index = 2                # free choice trial = 2, instructed = 1
        #move targets to current location and set location attribute.  Target1 (center target) position does not change.                    
        
        self.show_object(self.target1, False)
        #self.target_location1 = self.target1.xfm.move
        self.show_object(self.cursor, True)
        

        self.cursor_visible = True
        self.update_cursor()
        self.requeue()

    def _start_hold_penalty(self):
        #hide targets
        self.show_object(self.target1, False)
        self.show_object(self.targetH, False)
        self.timedout = True
        self.requeue()
        self.tries += 1
        
        self.port.write('o')


    def _start_timeout_penalty(self):
        #hide targets
        self.show_object(self.target1, False)
        self.show_object(self.targetH, False)
        self.timedout = True
        self.requeue()
        self.tries += 1


class InstructedChoicePilotTask(ManualControlFreeChoice):
    
    '''This is the forced-choice probabilistic reward task.'''
    
    
    hold_time = traits.Float(.2, desc="Length of hold required at targets")

    status = dict(
        #wait = dict(start_trial="target", stop=None),
        wait = dict(start_trial="center", stop=None),
        center = dict(enter_center="hold_center", timeout="timeout_penalty", enter_center_and_stimulate = "pulse",stop=None),
        hold_center = dict(leave_early_center = "hold_penalty",hold_center_complete="target", timeout="timeout_penalty", stop=None),
        pulse = dict(leave_early_center = "hold_penalty", pulse_end="interpulse_period", stop=None),
        interpulse_period = dict(leave_early_center = "hold_penalty", another_pulse="pulse", pulse_train_complete = "target", stop=None),
        target = dict(enter_targetH = "hold_targetH", timeout="timeout_penalty", stop=None),
        hold_targetH = dict(leave_early_H="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="check_reward",trial_abort="wait", trial_incomplete="center"),
        check_reward = dict(yes_reward="reward",no_reward="wait"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )
    
    def __init__(self, *args, **kwargs):
        super(InstructedChoicePilotTask, self).__init__(*args, **kwargs)
    
    def init(self):
        super(InstructedChoicePilotTask, self).init()

    def _cycle(self):
        '''Calls any update functions necessary and redraws screen. Runs 60x per second. '''
        #self.task_data['target'] = self.target_locationH.copy()
        #self.task_data['target_index'] = self.target_index

        ## Run graphics commands to show/hide the arm if the visibility has changed
        if self.plant_type != 'cursor_14x14':
            if self.arm_visible != self.arm_vis_prev:
                self.arm_vis_prev = self.arm_visible
                self.show_object(self.arm, show=self.arm_visible)

        self.move_arm()
        #self.move_plant()

        ## Save plant status to HDF file
        plant_data = self.plant.get_data_to_save()
        for key in plant_data:
            self.task_data[key] = plant_data[key]

        self.update_cursor()
        #if self.cursor_visible:
            #self.task_data['cursor'] = self.cursor.xfm.move.copy()
        #else:
            #if the cursor is not visible, write NaNs into cursor location saved in file
            #self.task_data['cursor'] = np.array([np.nan, np.nan, np.nan])

        if self.plant_type != 'cursor_14x14':
            self.task_data['joint_angles'] = self.get_arm_joints()

            #if self.arm_visible:
                #self.task_data['arm_visible'] = 1
            #else:
                #self.task_data['arm_visible'] = 0

        super(InstructedChoicePilotTask, self)._cycle()

    def _test_yes_reward(self,ts):
        reward_assigned = self.rewardH
        return bool(reward_assigned)

    def _test_no_reward(self,ts):
        reward_assigned = self.rewardH
        return bool(not reward_assigned)

    def _start_wait(self):
        super(InstructedChoicePilotTask, self)._start_wait()
        self.tries = 0
        self.target_index = 0     # indicator for instructed or free-choice trial
        #hide targets
        self.show_object(self.target1, False)
        self.show_object(self.targetL, False)
        self.show_object(self.targetH, False)
        
        #get target positions and reward assignments for this trial
        self.targs = self.next_trial

        assign_reward = np.random.randint(0,100)
        self.rewardH = np.greater(self.targs[0,1],assign_reward)
        self.rewardL  = self.rewardH

        if self.plant_type != 'cursor_14x14' and np.random.rand() < self.arm_hide_rate:
            self.arm_visible = False
        else:
            self.arm_visible = True
        #self.chain_length = self.targs.shape[0] #Number of sequential targets in a single trial
        self.task_data['targetH'] = self.targs[0,].copy()

        self.requeue()

    def _start_center(self):

        #self.target_index += 1
        self.pulse_count = 0    # resert count of number of stimulation pulses
        #set target colors 
        self.target1.color = (1,0,0,.5)      # center target red
        self.targetH.color = (0,0.5,0.5,.5)      # peripheral target also red

        # Third argument in self.targs determines if target is on left or right
        # First argument in self.targs determines if location is offset to farther distances
        offsetH = (2*self.targs[0,2] - 1)*(self.starting_dist + self.location_offset_allowed*self.targs[0,0]*4.0)
        moveH = np.array([offsetH,0,0]) 
       

        self.targetH.translate(*moveH, reset=True)
        #self.targetR.move_to_position(*moveR, reset=True)
        ##self.targetR.translate(*self.targs[self.target_index], reset=True)
        self.show_object(self.targetH, True)
        self.target_locationH = self.targetH.xfm.move
 
        self.show_object(self.target1, True)
        self.show_object(self.targetH, True)
        self.show_object(self.cursor, True)

        self.LH_target_on = (0, 1)
        #self.target_location1 = self.target1.xfm.move
        #print 'TARGET GENERATOR', self.targs[0]
        # offset targets from original location according to random generator
        
        #self.task_data['target_index'] = self.target_index
        self.requeue()
      

    def _start_target(self):

        self.target_index = 1

        #set target colors 
        self.target1.color = (1,0,0,.5)      # center target red
        #move targets to current location and set location attribute.  Target1 (center target) position does not change.                    
        
        self.show_object(self.target1, False)
        self.show_object(self.targetH, True)
        self.show_object(self.cursor, True)
        

        self.cursor_visible = True
        self.update_cursor()
        self.requeue()

    def _start_hold_penalty(self):
        #hide targets
        self.targetH.color = self.color_targets*(1,1,0,.5) + (1 - self.color_targets)*(0,0,1,.5)     # high probability target red if color_targets = 1
        self.show_object(self.target1, False)
        self.show_object(self.targetH, False)
        self.timedout = True
        self.requeue()
        self.tries += 1
        #self.target_index = -1
    
    def _start_targ_transition(self):
        #hide targets

        self.show_object(self.target1, False)
        self.show_object(self.targetH, False)
        self.requeue()

    def _start_check_reward(self):
        #hide targets
        self.show_object(self.target1, False)
        self.show_object(self.targetH, False)
        self.requeue()

    def _start_reward(self):
        super(InstructedChoicePilotTask, self)._start_reward()
        self.show_object(self.targetH, True)
        #reward_assigned = self.targs[0,1]
        #self.reward_counter = self.reward_counter + float(reward_assigned)
        self.requeue()

class ManualControlFreeChoice_withReversal(ManualControlFreeChoice):
    # create settable traits
    reversal_trial_num_begin = traits.Float(150, desc="Trial number when reversal should begin")
    reversal_trial_num_end = traits.Float(300, desc="Trial number when reversal should end")

    background = (0,0,0,1)
    shoulder_anchor = np.array([2., 0., -15.]) # Coordinates of shoulder anchor point on screen
    
    arm_visible = traits.Bool(True, desc='Specifies whether entire arm is displayed or just endpoint')
    
    cursor_radius = traits.Float(.5, desc="Radius of cursor")
    cursor_color = (.5,0,.5,1)

    joystick_method = traits.Float(1,desc="1: Normal velocity, 0: Position control")
    joystick_speed = traits.Float(20, desc="Speed of cursor")

    plant_type_options = list(plantlist.keys())
    plant_type = traits.OptionsList(*plantlist, bmi3d_input_options=list(plantlist.keys()))
    starting_pos = (5, 0, 5)
    # window_size = (1280*2, 1024)
    window_size = traits.Tuple((1920*2, 1080), desc='window size')
    percentage_instructed_trials = traits.Int(10,desc='Percentage of instructed trials: multiples of 10')
    print('in task')
    
    status = dict(
        #wait = dict(start_trial="target", stop=None),
        wait = dict(start_trial="center", stop=None),
        center = dict(enter_center="hold_center", timeout="timeout_penalty", stop=None),
        hold_center = dict(leave_early_center = "hold_penalty",hold_center_complete="target", timeout="timeout_penalty", stop=None),
        target = dict(enter_targetL="hold_targetL", enter_targetH = "hold_targetH", timeout="timeout_penalty", stop=None),
        hold_targetH = dict(leave_early_H="hold_penalty", hold_complete="targ_transition"),
        hold_targetL = dict(leave_early_L="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="check_reward",trial_abort="wait", trial_incomplete="center"),
        check_reward = dict(yes_reward="reward",no_reward="wait"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )
    ### added by HmS YZ 12/18/2020
    sequence_generators = ['colored_targets_with_probabilistic_reward','block_probabilistic_reward','colored_targets_with_randomwalk_reward','randomwalk_probabilistic_reward']
    
    def __init__(self, *args, **kwargs):
        super(ManualControlFreeChoice_withReversal, self).__init__(*args, **kwargs)
    #    self.port = serial.Serial(glob.glob("/dev/ttyACM*")[0], baudrate=115200)
    
   # def init(self):
   #     super(ManualControlFreeChoice_withReversal, self).init()

    def _start_wait(self):
        super(ManualControlFreeChoice_withReversal, self)._start_wait()
        self.tries = 0
        self.target_index = 0     # indicator for instructed or free-choice trial
        #hide targets
        self.show_object(self.target1, False)
        self.show_object(self.targetL, False)
        self.show_object(self.targetH, False)


        #get target positions and reward assignments for this trial
        self.targs = self.next_trial
        if self.plant_type != 'cursor_14x14' and np.random.rand() < self.arm_hide_rate:
            self.arm_visible = False
        else:
            self.arm_visible = True
        #self.chain_length = self.targs.shape[0] #Number of sequential targets in a single trial

        #self.task_data['target'] = self.target_locationH.copy()
        assign_reward = np.random.randint(0,100,size=2)
        self.rewardHigh = np.greater(self.targs[0,1],assign_reward[0])
        self.rewardLow = np.greater(self.targs[1,1],assign_reward[1])
        self.rewardH = ((self.calc_trial_num() < self.reversal_trial_num_begin) or (self.calc_trial_num() >= self.reversal_trial_num_end))*self.rewardHigh + (self.reversal_trial_num_begin <= self.calc_trial_num() < self.reversal_trial_num_end)*self.rewardLow
        #print 'high value target reward prob', self.targs[0,1]
        self.rewardL = ((self.calc_trial_num() < self.reversal_trial_num_begin) or (self.calc_trial_num() >= self.reversal_trial_num_end))*self.rewardLow + (self.reversal_trial_num_begin <= self.calc_trial_num() < self.reversal_trial_num_end)*self.rewardHigh

        
        #print 'TARGET GENERATOR', self.targs[0,]
        self.task_data['targetH'] = self.targs[0,].copy()
        self.task_data['reward_scheduleH'] = self.rewardH.copy()
        self.task_data['targetL'] = self.targs[1,].copy()
        self.task_data['reward_scheduleL'] = self.rewardL.copy()
        
        self.requeue()

    @staticmethod
    def colored_targets_with_block10_probabilistic_reward(length=1000, boundaries=(-18,18,-10,10,-15,15),reward_high_prob=80,reward_low_prob=40):
        
        position_offsetH = np.random.randint(2,size=(1,length))
        position_offsetL = np.random.randint(2,size=(1,length))
        location_int = np.random.randint(2,size=(1,length))
        reward_low = np.zeros((1,length))
        reward_low = np.ravel(reward_low)
        reward_high = np.zeros((1,length))
        reward_high = np.ravel(reward_high)
        num_blocks = int(length/10)
        num_low = int(reward_low_prob/10)
        num_high = int(reward_high_prob/10)
        for i in range(0,num_blocks):
            low_rewarded_trials = random.sample(range(10*i,10*i + 10),num_low)
            high_rewarded_trials = random.sample(range(10*i,10*i + 10),num_high)
            reward_low[low_rewarded_trials] = 1
            reward_high[high_rewarded_trials] = 1
        
        pairs = np.zeros([length,2,3])
        pairs[:,0,0] = position_offsetH
        pairs[:,0,1] = reward_high
        #pairs[:,0,1] = high_prob
        pairs[:,0,2] = location_int

        pairs[:,1,0] = position_offsetL
        pairs[:,1,1] = reward_low
        #pairs[:,1,1] = low_prob
        pairs[:,1,2] = 1 - location_int

        return pairs
    
    @staticmethod
    def block10_probabilistic_reward(boundaries=(-18,18,-10,10,-15,15),length=1000,reward_high_prob=80,reward_low_prob=40):
        pairs = colored_targets_with_block10_probabilistic_reward(length=length, boundaries=boundaries,reward_high_prob=reward_high_prob,reward_low_prob=reward_low_prob)
        return pairs
    '''   
    def _test_yes_reward(self,ts):
        if self.target_selected == 'H':
            reward_assigned = ((self.calc_trial_num() < self.reversal_trial_num_begin) or (self.calc_trial_num() >= self.reversal_trial_num_end))*self.rewardH + (self.reversal_trial_num_begin <= self.calc_trial_num() < self.reversal_trial_num_end)*self.rewardL
        else:
            reward_assigned = ((self.calc_trial_num() < self.reversal_trial_num_begin) or (self.calc_trial_num() >= self.reversal_trial_num_end))*self.rewardL + (self.reversal_trial_num_begin <= self.calc_trial_num() < self.reversal_trial_num_end)*self.rewardH
        #print "reward assigned", reward_assigned
        return bool(reward_assigned)
    def _test_no_reward(self,ts):
        if self.target_selected == 'H':
            reward_assigned = ((self.calc_trial_num() < self.reversal_trial_num_begin) or (self.calc_trial_num() >= self.reversal_trial_num_end))*self.rewardH + (self.reversal_trial_num_begin <= self.calc_trial_num() < self.reversal_trial_num_end)*self.rewardL
        else:
            reward_assigned = ((self.calc_trial_num() < self.reversal_trial_num_begin) or (self.calc_trial_num() >= self.reversal_trial_num_end))*self.rewardL + (self.reversal_trial_num_begin <= self.calc_trial_num() < self.reversal_trial_num_end)*self.rewardH
        #print "reward assigned", reward_assigned
        return bool(not reward_assigned)
    '''

class ManualControlFreeChoice_PseudoRandom(ManualControlFreeChoice):
    # create settable traits
    background = (0,0,0,1)
    shoulder_anchor = np.array([2., 0., -15.]) # Coordinates of shoulder anchor point on screen
    
    arm_visible = traits.Bool(True, desc='Specifies whether entire arm is displayed or just endpoint')
    
    cursor_radius = traits.Float(.5, desc="Radius of cursor")
    cursor_color = (.5,0,.5,1)

    joystick_method = traits.Float(1,desc="1: Normal velocity, 0: Position control")
    joystick_speed = traits.Float(20, desc="Speed of cursor")

    plant_type_options = list(plantlist.keys())
    plant_type = traits.OptionsList(*plantlist, bmi3d_input_options=list(plantlist.keys()))
    starting_pos = (5, 0, 5)
    # window_size = (1280*2, 1024)
    window_size = traits.Tuple((1920*2, 1080), desc='window size')
    percentage_instructed_trials = traits.Int(10,desc='Percentage of instructed trials: multiples of 10')
    print('in task')
    status = dict(
        #wait = dict(start_trial="target", stop=None),
        wait = dict(start_trial="center", stop=None),
        center = dict(enter_center="hold_center", timeout="timeout_penalty", stop=None),
        hold_center = dict(leave_early_center = "hold_penalty",hold_center_complete="target", timeout="timeout_penalty", stop=None),
        target = dict(enter_targetL="hold_targetL", enter_targetH = "hold_targetH", timeout="timeout_penalty", stop=None),
        hold_targetH = dict(leave_early_H="hold_penalty", hold_complete="targ_transition"),
        hold_targetL = dict(leave_early_L="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="check_reward",trial_abort="wait", trial_incomplete="center"),
        check_reward = dict(yes_reward="reward",no_reward="wait"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )
    
    sequence_generators = ['colored_targets_with_block10_probabilistic_reward','block10_probabilistic_reward']
    #sequence_generators = ['colored_targets_with_block10_probabilistic_reward']

    def __init__(self, *args, **kwargs):
        super(ManualControlFreeChoice_PseudoRandom, self).__init__(*args, **kwargs)

    def _start_wait(self):
        super(ManualControlFreeChoice_PseudoRandom, self)._start_wait()
        self.tries = 0
        self.target_index = 0     # indicator for instructed or free-choice trial
        #hide targets
        self.show_object(self.target1, False)
        self.show_object(self.targetL, False)
        self.show_object(self.targetH, False)


        #get target positions and reward assignments for this trial
        self.targs = self.next_trial
        if self.plant_type != 'cursor_14x14' and np.random.rand() < self.arm_hide_rate:
            self.arm_visible = False
        else:
            self.arm_visible = True
        #self.chain_length = self.targs.shape[0] #Number of sequential targets in a single trial
        #print 'before reward'
        self.rewardH = self.targs[0,1]
        self.rewardL = self.targs[1,1]
        #self.rewardH = 1
        #self.rewardL = 0
        #print 'high reward', self.rewardH
        
        #print 'TARGET GENERATOR', self.targs[0,]
        self.task_data['targetH'] = self.targs[0,].copy()
        self.task_data['reward_scheduleH'] = self.rewardH.copy()
        self.task_data['targetL'] = self.targs[1,].copy()
        self.task_data['reward_scheduleL'] = self.rewardL.copy()
        
        self.requeue()

    @staticmethod
    def colored_targets_with_block10_probabilistic_reward(length=1000, boundaries=(-18,18,-10,10,-15,15),reward_high_prob=80,reward_low_prob=40):
        
        position_offsetH = np.random.randint(2,size=(1,length))
        position_offsetL = np.random.randint(2,size=(1,length))
        location_int = np.random.randint(2,size=(1,length))
        reward_low = np.zeros((1,length))
        reward_low = np.ravel(reward_low)
        reward_high = np.zeros((1,length))
        reward_high = np.ravel(reward_high)
        num_blocks = int(length/10)
        num_low = int(reward_low_prob/10)
        num_high = int(reward_high_prob/10)
        for i in range(0,num_blocks):
            low_rewarded_trials = random.sample(range(10*i,10*i + 10),num_low)
            high_rewarded_trials = random.sample(range(10*i,10*i + 10),num_high)
            reward_low[low_rewarded_trials] = 1
            reward_high[high_rewarded_trials] = 1
        
        pairs = np.zeros([length,2,3])
        pairs[:,0,0] = position_offsetH
        pairs[:,0,1] = reward_high
        #pairs[:,0,1] = high_prob
        pairs[:,0,2] = location_int

        pairs[:,1,0] = position_offsetL
        pairs[:,1,1] = reward_low
        #pairs[:,1,1] = low_prob
        pairs[:,1,2] = 1 - location_int

        return pairs
    
    @staticmethod
    def block10_probabilistic_reward(boundaries=(-18,18,-10,10,-15,15),length=1000,reward_high_prob=80,reward_low_prob=40):
        pairs = colored_targets_with_block10_probabilistic_reward(length=length, boundaries=boundaries,reward_high_prob=reward_high_prob,reward_low_prob=reward_low_prob)
        return pairs

class ManualControlFreeChoice_PseudoRandom_StressCatchTrials(ManualControlFreeChoice_PseudoRandom):
    # create settable traits
    background = (0,0,0,1)
    shoulder_anchor = np.array([2., 0., -15.]) # Coordinates of shoulder anchor point on screen
    
    arm_visible = traits.Bool(True, desc='Specifies whether entire arm is displayed or just endpoint')
    audio_cue = traits.Bool(False, desc='ON or OFF for audio cue on stress trials')
    
    cursor_radius = traits.Float(.5, desc="Radius of cursor")
    stress_trial_begin = traits.Float(40, desc="When the first stress trial occurs")
    cursor_color = (.5,0,.5,1)
    reward_time = traits.Float(.5, desc="Length of juice reward")
    joystick_method = traits.Float(1,desc="1: Normal velocity, 0: Position control")
    joystick_speed = traits.Float(20, desc="Speed of cursor")
    #arm_class = traits.Enum('RobotArm2J2D', 'RobotArm2D', 'CursorPlant', 'RobotArmGen2D','Arm3D')
    plant_type_options = plantlist.keys()
    #plant_type = traits.Enum(*plantlist)
    plant_type = traits.OptionsList(*plantlist, desc='', bmi3d_input_options=plantlist.keys())
    starting_pos = (5, 0, 5)
    # window_size = (1280*2, 1024)
    window_size = traits.Tuple((1920*2, 1080), desc='window size')
    percentage_instructed_trials = traits.Int(10,desc='Percentage of instructed trials: multiples of 10')
    hold_penalty_time = traits.Float(1, desc="Length of penalty time for target hold error")
    timeout_time = traits.Float(10, desc="Time allowed to go between targets")
    stress_timeout_time = traits.Float(1.5, desc="Time allowed to go between targets on stress trials")
    print('in task')
    status = dict(
        #wait = dict(start_trial="target", stop=None),
        wait = dict(start_trial="center", stop=None),
        center = dict(enter_center="hold_center", timeout="timeout_penalty", stop=None),
        hold_center = dict(leave_early_center = "hold_penalty",hold_center_complete="target", timeout="timeout_penalty", stop=None),
        target = dict(enter_targetL="hold_targetL", enter_targetH = "hold_targetH", timeout="timeout_penalty", stop=None),
        hold_targetH = dict(leave_early_H="hold_penalty", hold_complete="targ_transition"),
        hold_targetL = dict(leave_early_L="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="check_reward",trial_abort="wait", trial_incomplete="center"),
        check_reward = dict(yes_reward="reward",no_reward="wait"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )
    
    sequence_generators = ['colored_targets_with_block10_probabilistic_reward','block10_probabilistic_reward']
    #sequence_generators = ['colored_targets_with_block10_probabilistic_reward']

    def __init__(self, *args, **kwargs):
        super(ManualControlFreeChoice_PseudoRandom_StressCatchTrials, self).__init__(*args, **kwargs)
        self.original_timeout_time = self.timeout_time
        self.original_reward_time = self.reward_time

    def init(self):
        self.add_dtype('stress_trial', 'i', (1,))
        super(ManualControlFreeChoice_PseudoRandom_StressCatchTrials, self).init()

    def _start_wait(self):
        super(ManualControlFreeChoice_PseudoRandom_StressCatchTrials, self)._start_wait()
        self.tries = 0
        self.target_index = 0     # indicator for instructed or free-choice trial
        self.stress_trial = 0     # indicator for stress trial
        #hide targets
        self.show_object(self.target1, False)
        self.show_object(self.targetL, False)
        self.show_object(self.targetH, False)


        #get target positions and reward assignments for this trial
        self.targs = self.next_trial
        if self.plant_type != 'cursor_14x14' and np.random.rand() < self.arm_hide_rate:
            self.arm_visible = False
        else:
            self.arm_visible = True
        #self.chain_length = self.targs.shape[0] #Number of sequential targets in a single trial
        #print 'before reward'
        self.rewardH = self.targs[0,1]
        self.rewardL = self.targs[1,1]
        #self.rewardH = 1
        #self.rewardL = 0
        #print 'high reward', self.rewardH

        """
        # Original task: learning block - 100 trials, short alternating blocks of 10 stress/regular trials
        if self.calc_trial_num() > 100:  
            if self.calc_trial_num() % 20 < 10:
                self.timeout_time = self.stress_timeout_time
                self.stress_trial = 1
            else:
                self.timeout_time = self.original_timeout_time
                self.stress_trial = 0
        """
        if self.calc_trial_num() > (self.stress_trial_begin - 1):  
            self.timeout_time = self.stress_timeout_time
            self.stress_trial = 1
            self.reward_time = 2*self.original_reward_time
        else:
            self.reward_time = self.original_reward_time
        
        #print 'TARGET GENERATOR', self.targs[0,]
        self.task_data['targetH'] = self.targs[0,].copy()
        self.task_data['reward_scheduleH'] = self.rewardH.copy()
        self.task_data['targetL'] = self.targs[1,].copy()
        self.task_data['reward_scheduleL'] = self.rewardL.copy()
        self.task_data['stress_trial'] = self.stress_trial
        
        self.requeue()

    def _start_center(self):

        #self.target_index += 1

        ## turn off stim if it was on
        #self.port.write('o')

        #set target colors 
        self.target1.color = (1,0,0,.5)      # center target red
        #self.targetH.color = self.color_targets*(1,1,0,.5) + (1 - self.color_targets)*(0,0,1,.5)     # high probability target red if color_targets = 1
        #self.targetL.color = (1 - self.color_targets)*(1,1,0,.5) + self.color_targets*(0,0,1,.5)    # low probability target blue if color_targets = 1
        self.targetH.color = self.color_targets*self.color1 + (1 - self.color_targets)*self.color2 # high is magenta if color_targets = 1, juicyorange otherwise
        self.targetL.color = (1 - self.color_targets)*self.color1 + self.color_targets*self.color2
        self.show_object(self.target1, True)
        self.show_object(self.cursor, True)
        
        # Third argument in self.targs determines if target is on left or right
        # First argument in self.targs determines if location is offset to farther distances
        offsetH = (2*self.targs[0,2] - 1)*(self.starting_dist + self.location_offset_allowed*self.targs[0,0]*4.0)
        moveH = np.array([offsetH,0,0]) 
        offsetL = (2*self.targs[1,2] - 1)*(self.starting_dist + self.location_offset_allowed*self.targs[1,0]*4.0)
        moveL = np.array([offsetL,0,0])

        self.targetL.translate(*moveL, reset=True) 
        #self.targetL.move_to_position(*moveL, reset=True)           
        ##self.targetL.translate(*self.targs[self.target_index], reset=True)
        self.show_object(self.targetL, True)
        self.target_locationL = self.targetL.xfm.move

        self.targetH.translate(*moveH, reset=True)
        #self.targetR.move_to_position(*moveR, reset=True)
        ##self.targetR.translate(*self.targs[self.target_index], reset=True)
        self.show_object(self.targetH, True)
        self.target_locationH = self.targetH.xfm.move


        # Insert instructed trials within free choice trials
        if self.trial_allocation[self.calc_trial_num()] == 1:
        #if (self.calc_trial_num() % 10) < (self.percentage_instructed_trials/10):
            self.target_index = 1    # instructed trial
            leftright_coinflip = np.random.randint(0,2)
            if leftright_coinflip == 0:
                self.show_object(self.targetL, False)
                self.LH_target_on = (0, 1)
            else:
                self.show_object(self.targetH, False)
                self.LR_coinflip = 0
                self.LH_target_on = (1, 0)
        else:
            self.target_index = 2   # free-choice trial

        if (self.stress_trial==1)&(self.audio_cue==True): 
            self.port.write('a')

        self.cursor_visible = True
        self.task_data['target_index'] = self.target_index
        self.requeue()

    def _start_timeout_penalty(self):
        #hide targets
        self.show_object(self.target1, False)
        self.show_object(self.targetL, False)
        self.show_object(self.targetH, False)

        #start stim
        if (self.stress_trial==1):
            self.port.write('x')
        else:
            self.port.write('o')

        self.timedout = True
        self.tries += 1
        self.requeue()
        
        

    @staticmethod
    def colored_targets_with_block10_probabilistic_reward(length=1000, boundaries=(-18,18,-10,10,-15,15),reward_high_prob=80,reward_low_prob=40):
        
        position_offsetH = np.random.randint(2,size=(1,length))
        position_offsetL = np.random.randint(2,size=(1,length))
        location_int = np.random.randint(2,size=(1,length))
        reward_low = np.zeros((1,length))
        reward_low = np.ravel(reward_low)
        reward_high = np.zeros((1,length))
        reward_high = np.ravel(reward_high)
        num_blocks = int(length/10)
        num_low = int(reward_low_prob/10)
        num_high = int(reward_high_prob/10)
        for i in range(0,num_blocks):
            low_rewarded_trials = random.sample(range(10*i,10*i + 10),num_low)
            high_rewarded_trials = random.sample(range(10*i,10*i + 10),num_high)
            reward_low[low_rewarded_trials] = 1
            reward_high[high_rewarded_trials] = 1
        
        pairs = np.zeros([length,2,3])
        pairs[:,0,0] = position_offsetH
        pairs[:,0,1] = reward_high
        #pairs[:,0,1] = high_prob
        pairs[:,0,2] = location_int

        pairs[:,1,0] = position_offsetL
        pairs[:,1,1] = reward_low
        #pairs[:,1,1] = low_prob
        pairs[:,1,2] = 1 - location_int

        return pairs
    
    @staticmethod
    def block10_probabilistic_reward(boundaries=(-18,18,-10,10,-15,15),length=1000,reward_high_prob=80,reward_low_prob=40):
        pairs = colored_targets_with_block10_probabilistic_reward(length=length, boundaries=boundaries,reward_high_prob=reward_high_prob,reward_low_prob=reward_low_prob)
        return pairs

class ManualControlFreeChoice_StressCatchTrials_OLStim(ManualControlFreeChoice_PseudoRandom_StressCatchTrials):
    """
    This is the ManualControlFreeChoice_PseudoRandom_StressCatchTrials task but with an additional digital output
    indicating when a center hold is complete and the targets are being shown. When used with the right TDT program, 
    this can trigger stimulation pulse trains if physiological markers indicate the subject is in a stress state.
    """

    # create settable traits
    background = (0,0,0,1)
    shoulder_anchor = np.array([2., 0., -15.]) # Coordinates of shoulder anchor point on screen
    
    arm_visible = traits.Bool(True, desc='Specifies whether entire arm is displayed or just endpoint')
    audio_cue = traits.Bool(False, desc='ON or OFF for audio cue on stress trials')
    
    cursor_radius = traits.Float(.5, desc="Radius of cursor")
    stress_trial_begin = traits.Float(40, desc="When the first stress trial occurs")
    cursor_color = (.5,0,.5,1)
    reward_time = traits.Float(.5, desc="Length of juice reward")
    joystick_method = traits.Float(1,desc="1: Normal velocity, 0: Position control")
    joystick_speed = traits.Float(20, desc="Speed of cursor")
    #arm_class = traits.Enum('RobotArm2J2D', 'RobotArm2D', 'CursorPlant', 'RobotArmGen2D','Arm3D')
    plant_type_options = plantlist.keys()
    #plant_type = traits.Enum(*plantlist)
    plant_type = traits.OptionsList(*plantlist, desc='', bmi3d_input_options=plantlist.keys())
    starting_pos = (5, 0, 5)
    # window_size = (1280*2, 1024)
    window_size = traits.Tuple((1920*2, 1080), desc='window size')
    percentage_instructed_trials = traits.Int(10,desc='Percentage of instructed trials: multiples of 10')
    hold_penalty_time = traits.Float(1, desc="Length of penalty time for target hold error")
    timeout_time = traits.Float(10, desc="Time allowed to go between targets")
    stress_timeout_time = traits.Float(1.5, desc="Time allowed to go between targets on stress trials")
    print('in task')
    status = dict(
        #wait = dict(start_trial="target", stop=None),
        wait = dict(start_trial="center", stop=None),
        center = dict(enter_center="hold_center", timeout="timeout_penalty", stop=None),
        hold_center = dict(leave_early_center = "hold_penalty",hold_center_complete="target", timeout="timeout_penalty", stop=None),
        target = dict(enter_targetL="hold_targetL", enter_targetH = "hold_targetH", timeout="timeout_penalty", stop=None),
        hold_targetH = dict(leave_early_H="hold_penalty", hold_complete="targ_transition"),
        hold_targetL = dict(leave_early_L="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="check_reward",trial_abort="wait", trial_incomplete="center"),
        check_reward = dict(yes_reward="reward",no_reward="wait"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )
    
    sequence_generators = ['colored_targets_with_block10_probabilistic_reward','block10_probabilistic_reward']
    #sequence_generators = ['colored_targets_with_block10_probabilistic_reward']

    def __init__(self, *args, **kwargs):
        super(ManualControlFreeChoice_StressCatchTrials_OLStim, self).__init__(*args, **kwargs)
        self.original_timeout_time = self.timeout_time
        self.original_reward_time = self.reward_time

    def init(self):
        super(ManualControlFreeChoice_StressCatchTrials_OLStim, self).init()

    def _start_target(self):

        #self.target_index += 1

        #move targets to current location and set location attribute.  Target1 (center target) position does not change.                    
        
        self.show_object(self.target1, False)
        #self.target_location1 = self.target1.xfm.move
        self.show_object(self.cursor, True)
       
        self.update_cursor()

        #send cue that center hold has ended and subject has transitioned

        if (self.stress_trial==1):
            self.port.write('h')
        
        self.requeue()
        

    @staticmethod
    def colored_targets_with_block10_probabilistic_reward(length=1000, boundaries=(-18,18,-10,10,-15,15),reward_high_prob=80,reward_low_prob=40):
        
        position_offsetH = np.random.randint(2,size=(1,length))
        position_offsetL = np.random.randint(2,size=(1,length))
        location_int = np.random.randint(2,size=(1,length))
        reward_low = np.zeros((1,length))
        reward_low = np.ravel(reward_low)
        reward_high = np.zeros((1,length))
        reward_high = np.ravel(reward_high)
        num_blocks = int(length/10)
        num_low = int(reward_low_prob/10)
        num_high = int(reward_high_prob/10)
        for i in range(0,num_blocks):
            low_rewarded_trials = random.sample(range(10*i,10*i + 10),num_low)
            high_rewarded_trials = random.sample(range(10*i,10*i + 10),num_high)
            reward_low[low_rewarded_trials] = 1
            reward_high[high_rewarded_trials] = 1
        
        pairs = np.zeros([length,2,3])
        pairs[:,0,0] = position_offsetH
        pairs[:,0,1] = reward_high
        #pairs[:,0,1] = high_prob
        pairs[:,0,2] = location_int

        pairs[:,1,0] = position_offsetL
        pairs[:,1,1] = reward_low
        #pairs[:,1,1] = low_prob
        pairs[:,1,2] = 1 - location_int

        return pairs
    
    @staticmethod
    def block10_probabilistic_reward(boundaries=(-18,18,-10,10,-15,15),length=1000,reward_high_prob=80,reward_low_prob=40):
        pairs = colored_targets_with_block10_probabilistic_reward(length=length, boundaries=boundaries,reward_high_prob=reward_high_prob,reward_low_prob=reward_low_prob)
        return pairs

class FreeChoicePilotTask_PseudoRandom(ManualControlFreeChoice_PseudoRandom):
    '''
    This is the free-choice probabilistic reward task broken into three phases.
    Phase 1: Directed choice of target with variable position, every 5th target is free choice. 100 trials.
    Phase 2: Directed choice of target with variable position, stimulation applied for lower probability target.  100 trials.
    Phase 3: Free choice of target with variable position, every 5th target is directed choice.  100 trials.
    '''
    #stimulation_pulse_length = traits.Float(200, desc="Pulse length in us")
    #stimulation_frequency = traits.Float(200, desc="Stimulation frequency")
    
    instructed_target = 0
    intervention_trial = False  # indicate if this is a trial with stimulation or not
    #com = comedi.comedi_open('/dev/comedi0')
    pulse_count = 0  #initializing number of pulses that have occured
    hold_time = traits.Float(.2, desc="Length of hold required at targets")
    percentage_instructed_trials = traits.Int(10,desc='Percentage of instructed trials')

    status = dict(
        #wait = dict(start_trial="target", stop=None),
        wait = dict(start_trial="center", stop=None),
        #center = dict(enter_center="hold_center", timeout="timeout_penalty", enter_center_and_stimulate = "pulse",stop=None),
        center = dict(enter_center="hold_center", timeout="timeout_penalty", enter_center_and_stimulate = "hold_center_and_stimulate",stop=None),
        hold_center = dict(leave_early_center = "hold_penalty",hold_center_complete="target", timeout="timeout_penalty", stop=None),
        hold_center_and_stimulate = dict(leave_early_center = "hold_penalty",hold_center_complete="target", timeout="timeout_penalty", stop=None),
        #pulse = dict(pulse_end="interpulse_period", stop=None),
        #interpulse_period = dict(another_pulse="pulse", pulse_train_complete = "target", stop=None),
        target = dict(enter_targetL="hold_targetL", enter_targetH = "hold_targetH", timeout="timeout_penalty", stop=None),
        hold_targetH = dict(leave_early_H="hold_penalty", hold_complete="targ_transition"),
        hold_targetL = dict(leave_early_L="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="check_reward",trial_abort="wait", trial_incomplete="center"),
        check_reward = dict(yes_reward="reward",no_reward="wait"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )
    
    def __init__(self, *args, **kwargs):
        super(FreeChoicePilotTask_PseudoRandom, self).__init__(*args, **kwargs)
        #print "num pulses", self.number_of_pulses
        # random assignment of instructed trials by doing sampling without replacement over a block of 100 trials
        self.trial_allocation = np.zeros(1000)
        self.port.write('o')
        for i in range(1,11):
            define_instructed_trials = random.sample(range(100*(i-1),100*(i-1) + 100),self.percentage_instructed_trials)
            self.trial_allocation[define_instructed_trials] = 1

    def _start_center(self):

        #self.target_index += 1
        self.pulse_count = 0    # resert count of number of stimulation pulses
        #set target colors 
        self.target1.color = (1,0,0,.5)      # center target red
        self.targetH.color = self.color_targets*self.color1 + (1 - self.color_targets)*self.color2    # high probability target red if color_targets = 1
        self.targetL.color = (1 - self.color_targets)*self.color1 + self.color_targets*self.color2    # low probability target blue if color_targets = 1
               
        self.show_object(self.target1, True)
        self.show_object(self.targetH, False)
        self.show_object(self.targetL, False)
        self.show_object(self.cursor, True)
        #self.target_location1 = self.target1.xfm.move
        #print 'TARGET GENERATOR', self.targs[0]
        # offset targets from original location according to random generator
        '''
        if 100 < self.calc_trial_num() < 201:
            self.instructed_target = np.random.randint(0,2)
            if self.instructed_target == 1:
                self.intervention_trial = True
            else:
                self.intervention_trial = False
        else:
            self.intervention_trial = False
        '''
        # Third argument in self.targs determines if target is on left or right
        # First argument in self.targs determines if location is offset to farther distances
        offsetH = (2*self.targs[0,2] - 1)*(self.starting_dist + self.location_offset_allowed*self.targs[0,0]*4.0)
        moveH = np.array([offsetH,0,0]) 
        offsetL = (2*self.targs[1,2] - 1)*(self.starting_dist + self.location_offset_allowed*self.targs[1,0]*4.0)
        moveL = np.array([offsetL,0,0])

        self.targetL.translate(*moveL, reset=True) 
        #self.targetL.move_to_position(*moveL, reset=True)           
        ##self.targetL.translate(*self.targs[self.target_index], reset=True)
        self.show_object(self.targetL, True)
        self.target_locationL = self.targetL.xfm.move

        self.targetH.translate(*moveH, reset=True)
        #self.targetR.move_to_position(*moveR, reset=True)
        ##self.targetR.translate(*self.targs[self.target_index], reset=True)
        self.show_object(self.targetH, True)
        self.target_locationH = self.targetH.xfm.move

        if self.calc_trial_num() < 101:  
            # use commented code if we're able to keep track of instructed versus free choice trials
            #check_if_instructed = np.random.randint(0,100)
            #if check_if_instructed < self.percentage_instructed_trials:
            self.intervention_trial = False
            if self.trial_allocation[self.calc_trial_num()] == 1:
            #if (self.calc_trial_num() % 10) < (self.percentage_instructed_trials/10):
                self.target_index = 1
                
                leftright_coinflip = np.random.randint(0,2)
                if leftright_coinflip == 0:
                    self.show_object(self.targetL, False)
                    self.LH_target_on = (0, 1)
                else:
                    self.show_object(self.targetH, False)
                    self.LH_target_on = (1, 0)
            else:
                self.target_index = 2
                self.LH_target_on = (1, 1)

        elif self.calc_trial_num() < 201:

            # All instructed trials during phase II (intervention phase)
            self.instructed_target = np.random.randint(0,2)    # choose whether high or low value target is shown
            self.target_index = 1
            if self.instructed_target == 0:
                self.show_object(self.targetL, False)
                self.LH_target_on = (0, 1)
                self.intervention_trial = False    # when high value target is shown, no stimulation is performed
            else:
                self.show_object(self.targetH, False)
                self.LH_target_on = (1, 0)
                self.intervention_trial = True     # when low value target is shown, stimulation is performed
                #print 'stim delivered', self.intervention_trial
        else:
            # All free-choice trials during phase III (test phase)
            if self.trial_allocation[self.calc_trial_num()] == 1:
            #if (self.calc_trial_num() % 10) < (self.percentage_instructed_trials/10):
                self.target_index = 1
                leftright_coinflip = 1  #always low value target show
                #leftright_coinflip = np.random.randint(0,2)
                if leftright_coinflip == 0:
                    self.show_object(self.targetL, False)
                    self.LH_target_on = (0, 1)
                    self.intervention_trial = False
                else:
                    self.show_object(self.targetH, False)
                    self.LH_target_on = (1, 0)
                    self.intervention_trial = True
            else:
                self.target_index = 2
                self.LH_target_on = (1, 1)
                self.intervention_trial = False
            '''
            self.intervention_trial = False
            self.target_index = 2
            self.LH_target_on = (1, 1)
            '''       
        '''
        import comedi
        com = comedi.comedi_open('/dev/comedi0')
        
        subdevice = 0
        write_mask = 0x800000
        val = 0x000000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)
        '''
        self.task_data['target_index'] = self.target_index
        self.requeue()
      

    def _test_enter_center(self, ts):
        #return true if the distance between center of cursor and target is smaller than the cursor radius

        d = np.sqrt((self.cursor.xfm.move[0]-self.target_location1[0])**2 + (self.cursor.xfm.move[1]-self.target_location1[1])**2 + (self.cursor.xfm.move[2]-self.target_location1[2])**2)
        #print 'TARGET SELECTED', self.target_selected
        return (d <= self.target_radius - self.cursor_radius) and (not self.intervention_trial)

    def _test_enter_center_and_stimulate(self, ts):
        #return true if the distance between center of cursor and target is smaller than the cursor radius

        d = np.sqrt((self.cursor.xfm.move[0]-self.target_location1[0])**2 + (self.cursor.xfm.move[1]-self.target_location1[1])**2 + (self.cursor.xfm.move[2]-self.target_location1[2])**2)
        #print 'TARGET SELECTED', self.target_selected
        return (d <= self.target_radius - self.cursor_radius) and (self.intervention_trial)

    def _start_hold_center_and_stimulate(self):
        self.show_object(self.target1, True)
        self.timedout = False
        '''
        At the start of the stimulation state, send TTL pulse
        '''
        self.port.write('x')
        '''
        import comedi
        com = comedi.comedi_open('/dev/comedi0')
        subdevice = 0
        write_mask = 0x800000
        val = 0x800000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)
        self.pulse_count = self.pulse_count + 1
        '''
        self.requeue()

    def _start_target(self):
        self.port.write('o')
        #self.target_index += 1
        '''
        import comedi
        com = comedi.comedi_open('/dev/comedi0')
        subdevice = 0
        write_mask = 0x800000
        val = 0x000000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)
        '''
        #set target colors 
        self.target1.color = (1,0,0,.5)      # center target red
        self.target_index = 2                # free choice trial = 2, instructed = 1
        #move targets to current location and set location attribute.  Target1 (center target) position does not change.                    
        
        self.show_object(self.target1, False)
        #self.target_location1 = self.target1.xfm.move
        self.show_object(self.cursor, True)
        

        self.cursor_visible = True
        self.update_cursor()
        self.requeue()

    def _start_hold_penalty(self):
        #hide targets
        self.show_object(self.target1, False)
        self.show_object(self.targetL, False)
        self.show_object(self.targetH, False)
        self.timedout = True
        self.requeue()
        self.tries += 1
        
        self.port.write('o')
        '''
        subdevice = 0
        write_mask = 0x800000
        val = 0x000000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)
        '''

    def _start_timeout_penalty(self):
        #hide targets
        self.show_object(self.target1, False)
        self.show_object(self.targetL, False)
        self.show_object(self.targetH, False)
        self.timedout = True
        self.requeue()
        self.tries += 1

        '''
        import comedi
        com = comedi.comedi_open('/dev/comedi0')
        subdevice = 0
        write_mask = 0x800000
        val = 0x000000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)
        '''

class FreeChoicePilotTask_ControlHighValueStim_PseudoRandom(ManualControlFreeChoice_PseudoRandom):
    '''
    This is the free-choice probabilistic reward task broken into three phases.
    Phase 1: Free choice with interleaved instructed trials. 100 trials.
    Phase 2: All instructed. Stimulation administered during high-value target trials.  100 trials.
    Phase 3: Free choice with interleaved instructed trials to high-value target.  100+ trials.
    '''
    #stimulation_pulse_length = traits.Float(200, desc="Pulse length in us")
    #stimulation_frequency = traits.Float(200, desc="Stimulation frequency")
    
    instructed_target = 0
    intervention_trial = False  # indicate if this is a trial with stimulation or not
    #com = comedi.comedi_open('/dev/comedi0')
    pulse_count = 0  #initializing number of pulses that have occured
    hold_time = traits.Float(.2, desc="Length of hold required at targets")
    percentage_instructed_trials = traits.Int(10,desc='Percentage of instructed trials')

    status = dict(
        #wait = dict(start_trial="target", stop=None),
        wait = dict(start_trial="center", stop=None),
        #center = dict(enter_center="hold_center", timeout="timeout_penalty", enter_center_and_stimulate = "pulse",stop=None),
        center = dict(enter_center="hold_center", timeout="timeout_penalty", enter_center_and_stimulate = "hold_center_and_stimulate",stop=None),
        hold_center = dict(leave_early_center = "hold_penalty",hold_center_complete="target", timeout="timeout_penalty", stop=None),
        hold_center_and_stimulate = dict(leave_early_center = "hold_penalty",hold_center_complete="target", timeout="timeout_penalty", stop=None),
        #pulse = dict(pulse_end="interpulse_period", stop=None),
        #interpulse_period = dict(another_pulse="pulse", pulse_train_complete = "target", stop=None),
        target = dict(enter_targetL="hold_targetL", enter_targetH = "hold_targetH", timeout="timeout_penalty", stop=None),
        hold_targetH = dict(leave_early_H="hold_penalty", hold_complete="targ_transition"),
        hold_targetL = dict(leave_early_L="hold_penalty", hold_complete="targ_transition"),
        targ_transition = dict(trial_complete="check_reward",trial_abort="wait", trial_incomplete="center"),
        check_reward = dict(yes_reward="reward",no_reward="wait"),
        timeout_penalty = dict(timeout_penalty_end="targ_transition"),
        hold_penalty = dict(hold_penalty_end="targ_transition"),
        reward = dict(reward_end="wait")
    )
    
    def __init__(self, *args, **kwargs):
        super(FreeChoicePilotTask_ControlHighValueStim_PseudoRandom, self).__init__(*args, **kwargs)
        #print "num pulses", self.number_of_pulses
        # random assignment of instructed trials by doing sampling without replacement over a block of 100 trials
        self.trial_allocation = np.zeros(1000)
        for i in range(1,11):
            define_instructed_trials = random.sample(range(100*(i-1),100*(i-1) + 100),self.percentage_instructed_trials)
            self.trial_allocation[define_instructed_trials] = 1

    def _start_center(self):

        #self.target_index += 1
        self.pulse_count = 0    # resert count of number of stimulation pulses
        #set target colors 
        self.target1.color = (1,0,0,.5)      # center target red
        self.targetH.color = self.color_targets*self.color1 + (1 - self.color_targets)*self.color2    # high probability target red if color_targets = 1
        self.targetL.color = (1 - self.color_targets)*self.color1 + self.color_targets*self.color2    # low probability target blue if color_targets = 1
               
        self.show_object(self.target1, True)
        self.show_object(self.targetH, False)
        self.show_object(self.targetL, False)
        self.show_object(self.cursor, True)
        #self.target_location1 = self.target1.xfm.move
        #print 'TARGET GENERATOR', self.targs[0]
        # offset targets from original location according to random generator
        '''
        if 100 < self.calc_trial_num() < 201:
            self.instructed_target = np.random.randint(0,2)
            if self.instructed_target == 1:
                self.intervention_trial = True
            else:
                self.intervention_trial = False
        else:
            self.intervention_trial = False
        '''
        # Third argument in self.targs determines if target is on left or right
        # First argument in self.targs determines if location is offset to farther distances
        offsetH = (2*self.targs[0,2] - 1)*(self.starting_dist + self.location_offset_allowed*self.targs[0,0]*4.0)
        moveH = np.array([offsetH,0,0]) 
        offsetL = (2*self.targs[1,2] - 1)*(self.starting_dist + self.location_offset_allowed*self.targs[1,0]*4.0)
        moveL = np.array([offsetL,0,0])

        self.targetL.translate(*moveL, reset=True) 
        #self.targetL.move_to_position(*moveL, reset=True)           
        ##self.targetL.translate(*self.targs[self.target_index], reset=True)
        self.show_object(self.targetL, True)
        self.target_locationL = self.targetL.xfm.move

        self.targetH.translate(*moveH, reset=True)
        #self.targetR.move_to_position(*moveR, reset=True)
        ##self.targetR.translate(*self.targs[self.target_index], reset=True)
        self.show_object(self.targetH, True)
        self.target_locationH = self.targetH.xfm.move

        if self.calc_trial_num() < 101:  
            # use commented code if we're able to keep track of instructed versus free choice trials
            #check_if_instructed = np.random.randint(0,100)
            #if check_if_instructed < self.percentage_instructed_trials:
            self.intervention_trial = False
            if self.trial_allocation[self.calc_trial_num()] == 1:
            #if (self.calc_trial_num() % 10) < (self.percentage_instructed_trials/10):
                self.target_index = 1
                
                leftright_coinflip = np.random.randint(0,2)
                if leftright_coinflip == 0:
                    self.show_object(self.targetL, False)
                    self.LH_target_on = (0, 1)
                else:
                    self.show_object(self.targetH, False)
                    self.LH_target_on = (1, 0)
            else:
                self.target_index = 2
                self.LH_target_on = (1, 1)

        elif self.calc_trial_num() < 201:

            # All instructed trials during phase II (intervention phase)
            self.instructed_target = np.random.randint(0,2)    # choose whether high or low value target is shown
            self.target_index = 1
            if self.instructed_target == 0:
                self.show_object(self.targetL, False)
                self.LH_target_on = (0, 1)
                self.intervention_trial = True    # when high value target is shown, stimulation is performed
            else:
                self.show_object(self.targetH, False)
                self.LH_target_on = (1, 0)
                self.intervention_trial = False     # when low value target is shown, stimulation is not performed
                #print 'stim delivered', self.intervention_trial
        else:
            # All free-choice trials during phase III (test phase)
            if self.trial_allocation[self.calc_trial_num()] == 1:
            #if (self.calc_trial_num() % 10) < (self.percentage_instructed_trials/10):
                self.target_index = 1
                leftright_coinflip = 0  #always high value target show
                #leftright_coinflip = np.random.randint(0,2)
                if leftright_coinflip == 0:
                    self.show_object(self.targetL, False)
                    self.LH_target_on = (0, 1)
                    self.intervention_trial = True
                else:
                    self.show_object(self.targetH, False)
                    self.LH_target_on = (1, 0)
                    self.intervention_trial = False
            else:
                self.target_index = 2
                self.LH_target_on = (1, 1)
                self.intervention_trial = False
            '''
            self.intervention_trial = False
            self.target_index = 2
            self.LH_target_on = (1, 1)
            '''       
        '''
        import comedi
        com = comedi.comedi_open('/dev/comedi0')
        
        subdevice = 0
        write_mask = 0x800000
        val = 0x000000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)
        '''
        self.task_data['target_index'] = self.target_index
        self.requeue()
      

    def _test_enter_center(self, ts):
        #return true if the distance between center of cursor and target is smaller than the cursor radius

        d = np.sqrt((self.cursor.xfm.move[0]-self.target_location1[0])**2 + (self.cursor.xfm.move[1]-self.target_location1[1])**2 + (self.cursor.xfm.move[2]-self.target_location1[2])**2)
        #print 'TARGET SELECTED', self.target_selected
        return (d <= self.target_radius - self.cursor_radius) and (not self.intervention_trial)

    def _test_enter_center_and_stimulate(self, ts):
        #return true if the distance between center of cursor and target is smaller than the cursor radius

        d = np.sqrt((self.cursor.xfm.move[0]-self.target_location1[0])**2 + (self.cursor.xfm.move[1]-self.target_location1[1])**2 + (self.cursor.xfm.move[2]-self.target_location1[2])**2)
        #print 'TARGET SELECTED', self.target_selected
        return (d <= self.target_radius - self.cursor_radius) and (self.intervention_trial)

    def _start_hold_center_and_stimulate(self):
        self.show_object(self.target1, True)
        self.timedout = False
        '''
        At the start of the stimulation state, send TTL pulse
        '''
        self.port.write('x')
        '''
        import comedi
        com = comedi.comedi_open('/dev/comedi0')
        subdevice = 0
        write_mask = 0x800000
        val = 0x800000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)
        self.pulse_count = self.pulse_count + 1
        '''
        self.requeue()

    def _start_target(self):
        self.port.write('o')
        #self.target_index += 1
        '''
        import comedi
        com = comedi.comedi_open('/dev/comedi0')
        subdevice = 0
        write_mask = 0x800000
        val = 0x000000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)
        '''
        #set target colors 
        self.target1.color = (1,0,0,.5)      # center target red
        self.target_index = 2                # free choice trial = 2, instructed = 1
        #move targets to current location and set location attribute.  Target1 (center target) position does not change.                    
        
        self.show_object(self.target1, False)
        #self.target_location1 = self.target1.xfm.move
        self.show_object(self.cursor, True)
        

        self.cursor_visible = True
        self.update_cursor()
        self.requeue()

    def _start_hold_penalty(self):
        #hide targets
        self.show_object(self.target1, False)
        self.show_object(self.targetL, False)
        self.show_object(self.targetH, False)
        self.timedout = True
        self.requeue()
        self.tries += 1
        
        self.port.write('o')
        '''
        subdevice = 0
        write_mask = 0x800000
        val = 0x000000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)
        '''

    def _start_timeout_penalty(self):
        #hide targets
        self.show_object(self.target1, False)
        self.show_object(self.targetL, False)
        self.show_object(self.targetH, False)
        self.timedout = True
        self.requeue()
        self.tries += 1

        '''
        import comedi
        com = comedi.comedi_open('/dev/comedi0')
        subdevice = 0
        write_mask = 0x800000
        val = 0x000000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)
        '''