'''
Base tasks for generic point-to-point reaching
'''
from __future__ import division
import numpy as np
from collections import OrderedDict
import time

from riglib import reward
from riglib.experiment import traits, Sequence

from riglib.stereo_opengl.window import Window, FPScontrol, WindowDispl2D
from riglib.stereo_opengl.primitives import Cylinder, Plane, Sphere, Cube
from riglib.stereo_opengl.models import FlatMesh, Group
from riglib.stereo_opengl.textures import Texture, TexModel
from riglib.stereo_opengl.render import stereo, Renderer
from riglib.stereo_opengl.utils import cloudy_tex

from .plantlist import plantlist

from riglib.stereo_opengl import ik

import math
import traceback

from riglib.bmi import bmi
#from built_in_tasks.bmimultitasks import BMILoop
from riglib.bmi.bmi import BMILoop
from built_in_tasks.manualcontrolmultitasks import ManualControlMulti, VirtualCircularTarget

from riglib.bmi import onedim_lfp_decoder 

####### CONSTANTS
sec_per_min = 60.0
RED = (1,0,0,.5)
GREEN = (0,1,0,0.5)
mm_per_cm = 1./10

class SquareTarget(object):
    def __init__(self, target_radius=2, target_color=(1, 0, 0, .5), starting_pos=np.zeros(3)):
        self.target_rad = target_radius
        self.target_color = target_color
        self.position = starting_pos
        self.int_position = starting_pos
        self._pickle_init()

    def _pickle_init(self):
        self.cube = Cube(side_len=self.target_rad, color=self.target_color)
        self.graphics_models = [self.cube]
        self.cube.translate(*self.position)

    def move_to_position(self, new_pos):
        self.int_position = new_pos
        self.drive_to_new_pos()

    def drive_to_new_pos(self):
        raise NotImplementedError 


class VirtualSquareTarget(SquareTarget):
    def drive_to_new_pos(self):
        self.position = self.int_position
        self.cube.translate(*self.position, reset=True)

    def hide(self):
        self.cube.detach()

    def show(self):
        self.cube.attach()

    def cue_trial_start(self):
        self.cube.color = self.target_color
        self.show()

    def cue_trial_end_success(self):
        self.cube.color = GREEN

    def cue_trial_end_failure(self):
        self.cube.color = RED
        self.hide()
        # self.sphere.color = GREEN

    def idle(self):
        self.cube.color = self.target_color
        self.hide()


class LFP_Mod(BMILoop, Sequence, Window):

    background = (0,0,0,1)
    
    plant_visible = traits.Bool(True, desc='Specifies whether entire plant is displayed or just endpoint')
    
    lfp_cursor_rad = traits.Float(.5, desc="length of LFP cursor")
    lfp_cursor_color = (.5,0,.5,.75)  
     
    lfp_plant_type_options = list(plantlist.keys())
    lfp_plant_type = traits.OptionsList(*plantlist, bmi3d_input_options=list(plantlist.keys()))

    window_size = traits.Tuple((1920*2, 1080), desc='window size')

    lfp_frac_lims = traits.Tuple((0., 0.35), desc='fraction limits')
    xlfp_frac_lims = traits.Tuple((-.7, 1.7), desc = 'x dir fraction limits')
    lfp_control_band = traits.Tuple((25, 40), desc='beta power band limits')
    lfp_totalpw_band = traits.Tuple((1, 100), desc='total power band limits')
    xlfp_control_band = traits.Tuple((0, 5), desc = 'x direction band limits')
    n_steps = traits.Int(2, desc='moving average for decoder')

    is_bmi_seed = True

    powercap = traits.Float(1, desc="Timeout for total power above this")

    zboundaries=(-12,12)

    status = dict(
        wait = dict(start_trial="lfp_target", stop=None),
        lfp_target = dict(enter_lfp_target="lfp_hold", powercap_penalty="powercap_penalty", stop=None),
        lfp_hold = dict(leave_early="lfp_target", lfp_hold_complete="reward", powercap_penalty="powercap_penalty"),
        powercap_penalty = dict(powercap_penalty_end="lfp_target"),
        reward = dict(reward_end="wait")
        )

    static_states = [] # states in which the decoder is not run
    trial_end_states = ['reward']
    lfp_cursor_on = ['lfp_target', 'lfp_hold']

    #initial state
    state = "wait"

    #create settable traits
    reward_time = traits.Float(.5, desc="Length of juice reward")

    lfp_target_rad = traits.Float(3.6, desc="Length of targets in cm")
    
    lfp_hold_time = traits.Float(.2, desc="Length of hold required at lfp targets")
    lfp_hold_var = traits.Float(.05, desc="Length of hold variance required at lfp targets")

    hold_penalty_time = traits.Float(1, desc="Length of penalty time for target hold error")
    
    powercap_penalty_time = traits.Float(1, desc="Length of penalty time for timeout error")

    # max_attempts = traits.Int(10, desc='The number of attempts at a target before\
    #     skipping to the next one')

    session_length = traits.Float(0, desc="Time until task automatically stops. Length of 0 means no auto stop.")

    #plant_hide_rate = traits.Float(0.0, desc='If the plant is visible, specifies a percentage of trials where it will be hidden')
    lfp_target_color = (123/256.,22/256.,201/256.,.5)
    mc_target_color = (1,0,0,.5)

    target_index = -1 # Helper variable to keep track of which target to display within a trial
    #tries = 0 # Helper variable to keep track of the number of failed attempts at a given trial.
    
    cursor_visible = False # Determines when to hide the cursor.
    no_data_count = 0 # Counter for number of missing data frames in a row
    
    sequence_generators = ['lfp_mod_4targ']
    
    def __init__(self, *args, **kwargs):
        super(LFP_Mod, self).__init__(*args, **kwargs)
        self.cursor_visible = True

        print('INIT FRAC LIMS: ', self.lfp_frac_lims)
        
        dec_params = dict(lfp_frac_lims = self.lfp_frac_lims,
                          xlfp_frac_lims = self.xlfp_frac_lims,
                          powercap = self.powercap,
                          zboundaries = self.zboundaries,
                          lfp_control_band = self.lfp_control_band,
                          lfp_totalpw_band = self.lfp_totalpw_band,
                          xlfp_control_band = self.xlfp_control_band,
                          n_steps = self.n_steps)

        self.decoder.filt.init_from_task(**dec_params)
        self.decoder.init_from_task(**dec_params)

        self.lfp_plant = plantlist[self.lfp_plant_type]
        if self.lfp_plant_type == 'inv_cursor_onedimLFP':
            print('MAKE SURE INVERSE GENERATOR IS ON')
            
        self.plant_vis_prev = True

        self.current_assist_level = 0
        self.learn_flag = False

        if hasattr(self.lfp_plant, 'graphics_models'):
            for model in self.lfp_plant.graphics_models:
                self.add_model(model)

        # Instantiate the targets
        ''' 
        height and width on kinarm machine are 2.4. Here we make it 2.4/8*12 = 3.6
        '''
        lfp_target = VirtualSquareTarget(target_radius=self.lfp_target_rad, target_color=self.lfp_target_color)
        self.targets = [lfp_target]
        
        # Initialize target location variable
        self.target_location_lfp = np.array([-100, -100, -100])

        # Declare any plant attributes which must be saved to the HDF file at the _cycle rate
        for attr in self.lfp_plant.hdf_attrs:
            self.add_dtype(*attr) 

    def init(self):
        self.plant = DummyPlant()
        self.add_dtype('lfp_target', 'f8', (3,)) 
        self.add_dtype('target_index', 'i', (1,))
        self.add_dtype('powercap_flag', 'i',(1,))

        for target in self.targets:
            for model in target.graphics_models:
                self.add_model(model)

        super(LFP_Mod, self).init()

    def _cycle(self):
        '''
        Calls any update functions necessary and redraws screen. Runs 60x per second.
        '''
        self.task_data['loop_time'] = self.iter_time()
        self.task_data['lfp_target'] = self.target_location_lfp.copy()
        self.task_data['target_index'] = self.target_index
        #self.task_data['internal_decoder_state'] = self.decoder.filt.current_lfp_pos
        self.task_data['powercap_flag'] = self.decoder.filt.current_powercap_flag

        self.move_plant()

        ## Save plant status to HDF file, ###ADD BACK
        lfp_plant_data = self.lfp_plant.get_data_to_save()
        for key in lfp_plant_data:
            self.task_data[key] = lfp_plant_data[key]

        super(LFP_Mod, self)._cycle()

    def move_plant(self):
        feature_data = self.get_features()

        # Save the "neural features" (e.g. spike counts vector) to HDF file
        for key, val in feature_data.items():
            self.task_data[key] = val
        Bu = None
        assist_weight = 0
        target_state = np.zeros([self.decoder.n_states, self.decoder.n_subbins])

        ## Run the decoder
        if self.state not in self.static_states:
            neural_features = feature_data[self.extractor.feature_type]
            self.call_decoder(neural_features, target_state, Bu=Bu, assist_level=assist_weight, feature_type=self.extractor.feature_type)

        ## Drive the plant to the decoded state, if permitted by the constraints of the plant
        self.lfp_plant.drive(self.decoder)
        self.task_data['decoder_state'] = decoder_state = self.decoder.get_state(shape=(-1,1))
        return decoder_state     

    def run(self):
        '''
        See experiment.Experiment.run for documentation. 
        '''
        # Fire up the plant. For virtual/simulation plants, this does little/nothing.
        self.lfp_plant.start()
        try:
            super(LFP_Mod, self).run()
        finally:
            self.lfp_plant.stop()

    ##### HELPER AND UPDATE FUNCTIONS ####
    def update_cursor_visibility(self):
        ''' Update cursor visible flag to hide cursor if there has been no good data for more than 3 frames in a row'''
        prev = self.cursor_visible
        if self.no_data_count < 3:
            self.cursor_visible = True
            if prev != self.cursor_visible:
                self.show_object(self.cursor, show=True)
        else:
            self.cursor_visible = False
            if prev != self.cursor_visible:
                self.show_object(self.cursor, show=False)

    def update_report_stats(self):
        '''
        see experiment.Experiment.update_report_stats for docs
        '''
        super(LFP_Mod, self).update_report_stats()
        self.reportstats['Trial #'] = self.calc_trial_num()
        self.reportstats['Reward/min'] = np.round(self.calc_events_per_min('reward', 120), decimals=2)

    #### TEST FUNCTIONS ####
    def _test_powercap_penalty(self, ts):
        if self.decoder.filt.current_powercap_flag:
            #Turn off power cap flag:
            self.decoder.filt.current_powercap_flag = 0
            return True
        else:
            return False


    def _test_enter_lfp_target(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius in the x and z axis only
        '''
        cursor_pos = self.lfp_plant.get_endpoint_pos()
        dx = np.linalg.norm(cursor_pos[0] - self.target_location_lfp[0])
        dz = np.linalg.norm(cursor_pos[2] - self.target_location_lfp[2])
        in_targ = False
        if dx<= (self.lfp_target_rad/2.) and dz<= (self.lfp_target_rad/2.):
            in_targ = True

        return in_targ

        # #return d <= (self.lfp_target_rad - self.lfp_cursor_rad)

        # #If center of cursor enters target at all: 
        # return d <= (self.lfp_target_rad/2.)

        # #New version: 
        # cursor_pos = self.lfp_plant.get_endpoint_pos()
        # d = np.linalg.norm(cursor_pos[2] - self.target_location_lfp[2])
        # d <= (self.lfp_target_rad - self.lfp_cursor_rad)
        
    def _test_leave_early(self, ts):
        '''
        return true if cursor moves outside the exit radius
        '''
        cursor_pos = self.lfp_plant.get_endpoint_pos()
        dx = np.linalg.norm(cursor_pos[0] - self.target_location_lfp[0])
        dz = np.linalg.norm(cursor_pos[2] - self.target_location_lfp[2])
        out_of_targ = False
        if dx > (self.lfp_target_rad/2.) or dz > (self.lfp_target_rad/2.):
            out_of_targ = True
        #rad = self.lfp_target_rad - self.lfp_cursor_rad
        #return d > rad
        return out_of_targ

    def _test_lfp_hold_complete(self, ts):
        return ts>=self.lfp_hold_time_plus_var

    # def _test_lfp_timeout(self, ts):
    #     return ts>self.timeout_time

    def _test_powercap_penalty_end(self, ts):
        if ts>self.powercap_penalty_time:
            self.lfp_plant.turn_on()

        return ts>self.powercap_penalty_time

    def _test_reward_end(self, ts):
        return ts>self.reward_time

    def _test_stop(self, ts):
        if self.session_length > 0 and (self.get_time() - self.task_start_time) > self.session_length:
            self.end_task()
        return self.stop

    #### STATE FUNCTIONS ####
    def _parse_next_trial(self):
        self.targs = self.next_trial
        
    def _start_wait(self):
        super(LFP_Mod, self)._start_wait()
        self.tries = 0
        self.target_index = -1
        #hide targets
        for target in self.targets:
            target.hide()

        #get target locations for this trial
        self._parse_next_trial()
        self.chain_length = 1
        self.lfp_hold_time_plus_var = self.lfp_hold_time + np.random.uniform(low=-1,high=1)*self.lfp_hold_var

    def _start_lfp_target(self):
        self.target_index += 1
        self.target_index = 0

        #only 1 target: 
        target = self.targets[0]
        self.target_location_lfp = self.targs #Just one target. 
        
        target.move_to_position(self.target_location_lfp)
        target.cue_trial_start()

    def _start_lfp_hold(self):
        #make next target visible unless this is the final target in the trial
        idx = (self.target_index + 1)
        if idx < self.chain_length: 
            target = self.targets[idx % 2]
            target.move_to_position(self.targs[idx])
    
    def _end_lfp_hold(self):
        # change current target color to green
        self.targets[self.target_index % 2].cue_trial_end_success()
    
    def _start_timeout_penalty(self):
        #hide targets
        for target in self.targets:
            target.hide()

        self.tries += 1
        self.target_index = -1

    def _start_reward(self):
        #super(LFP_Mod, self)._start_reward()
        self.targets[self.target_index % 2].show()

    def _start_powercap_penalty(self):
        for target in self.targets:
            target.hide()
        self.lfp_plant.turn_off()

    @staticmethod
    def lfp_mod_4targ(nblocks=100, boundaries=(-18,18,-12,12), xaxis=-8):
        '''Mimics beta modulation task from Kinarm Rig:

        In Kinarm rig, the following linear transformations happen: 
            1. LFP cursor is calculated
            2. mapped from fraction limits [0, .35] to [-1, 1] (unit_coordinates)
            3. udp sent to kinarm machine and multiplied by 8
            4. translated upward in the Y direction by + 2.5

        This means, our targets which are at -8, [-0.75, 2.5, 5.75, 9.0]
        must be translated down by 2.5 to: -8, [-3.25,  0.  ,  3.25,  6.5]
        then divided by 8: -1, [-0.40625,  0.     ,  0.40625,  0.8125 ] in unit_coordinates

        The radius is 1.2, which is 0.15 in unit_coordinates

        Now, we map this to a new system: 
        - new_zero: (y1+y2) / 2
        - new_scale: (y2 - y1) / 2

         (([-0.40625,  0.     ,  0.40625,  0.8125 ]) * new_scale ) + new_zero
        
        new_zero = 0
        new_scale = 12

        12 * [-0.40625,  0.     ,  0.40625,  0.8125 ] 

        = array([-4.875,  0.   ,  4.875,  9.75 ])

        '''

        new_zero = (boundaries[3]+boundaries[2]) / 2.
        new_scale = (boundaries[3] - boundaries[2]) / 2.

        kin_targs = np.array([-0.40625,  0.     ,  0.40625,  0.8125 ])

        lfp_targ_y = (new_scale*kin_targs) + new_zero

        for i in range(nblocks):
            temp = lfp_targ_y.copy()
            np.random.shuffle(temp)
            if i==0:
                z = temp.copy()
            else:
                z = np.hstack((z, temp))

        #Fixed X axis: 
        x = np.tile(xaxis,(nblocks*4))
        y = np.zeros(nblocks*4)
        
        pairs = np.vstack([x, y, z]).T
        return pairs

class LFP_Mod_plus_MC_hold(LFP_Mod):

    mc_cursor_radius = traits.Float(.5, desc="Radius of cursor")
    mc_target_radius = traits.Float(3, desc="Radius of MC target")
    mc_cursor_color = (.5,0,.5,1)
    mc_plant_type_options = plantlist.keys()
    mc_plant_type = traits.OptionsList(*plantlist, bmi3d_input_options=plantlist.keys())
    origin_hold_time = traits.Float(.2, desc="Hold time in center")
    exclude_parent_traits = ['goal_cache_block'] #redefine this to NOT include marker_num, marker_count
    marker_num = traits.Int(14,desc='Index')
    marker_count = traits.Int(16,desc='Num of markers')
    joystick_method = traits.Float(1,desc="1: Normal velocity, 0: Position control")
    joystick_speed = traits.Float(20, desc="Radius of cursor")
    move_while_in_center = traits.Float(1, desc="1 = update plant while in lfp_target, lfp_hold, 0 = don't update in these states")
    scale_factor = 3.0 #scale factor for converting hand movement to screen movement (1cm hand movement = 3.5cm cursor movement)
    wait_flag = 1
    # NOTE!!! The marker on the hand was changed from #0 to #14 on
    # 5/19/13 after LED #0 broke. All data files saved before this date
    # have LED #0 controlling the cursor.
    limit2d = 1

    status = dict(
        wait = dict(start_trial="origin", stop=None),
        origin = dict(enter_origin="origin_hold", stop=None),
        origin_hold = dict(origin_hold_complete="lfp_target",leave_origin="hold_penalty", stop=None),
        lfp_target = dict(enter_lfp_target="lfp_hold", leave_origin="hold_penalty", powercap_penalty="powercap_penalty", stop=None),
        lfp_hold = dict(leave_early="lfp_target", lfp_hold_complete="reward", leave_origin="hold_penalty", powercap_penalty="powercap_penalty",stop=None),
        powercap_penalty = dict(powercap_penalty_end="origin"),
        hold_penalty = dict(hold_penalty_end="origin",stop=None),
        reward = dict(reward_end="wait")
    )

    static_states = ['origin'] # states in which the decoder is not run
    trial_end_states = ['reward']
    lfp_cursor_on = ['lfp_target', 'lfp_hold', 'reward']

    sequence_generators = ['lfp_mod_4targ_plus_mc_orig']


    def __init__(self, *args, **kwargs):
        super(LFP_Mod_plus_MC_hold, self).__init__(*args, **kwargs)
        if self.move_while_in_center>0:
            self.no_plant_update_states = []
        else:
            self.no_plant_update_states = ['lfp_target', 'lfp_hold']

        mc_origin = VirtualCircularTarget(target_radius=self.mc_target_radius, target_color=RED)
        lfp_target = VirtualSquareTarget(target_radius=self.lfp_target_rad, target_color=self.lfp_target_color)

        self.targets = [lfp_target, mc_origin]

        self.mc_plant = plantlist[self.mc_plant_type]
        if hasattr(self.mc_plant, 'graphics_models'):
            for model in self.mc_plant.graphics_models:
                self.add_model(model)

        # Declare any plant attributes which must be saved to the HDF file at the _cycle rate
        for attr in self.mc_plant.hdf_attrs:
            self.add_dtype(*attr) 

        self.target_location_mc = np.array([-100, -100, -100])
        self.manual_control_type = None

        self.current_pt=np.zeros([3]) #keep track of current pt
        self.last_pt=np.zeros([3])

    def init(self):
        self.add_dtype('mc_targ', 'f8', (3,)) ###ADD BACK
        super(LFP_Mod_plus_MC_hold, self).init()


    def _cycle(self):
        '''
        Calls any update functions necessary and redraws screen. Runs 60x per second.
        '''
        self.task_data['mc_targ'] = self.target_location_mc.copy()


        mc_plant_data = self.mc_plant.get_data_to_save()
        for key in mc_plant_data:
            self.task_data[key] = mc_plant_data[key]

        super(LFP_Mod_plus_MC_hold, self)._cycle()


    def _parse_next_trial(self):
        t = self.next_trial
        self.lfp_targ = t['lfp']
        self.mc_targ_orig = t['origin']

    def _start_origin(self):
        if self.wait_flag:
            self.origin_hold_time_store = self.origin_hold_time
            self.origin_hold_time = 3
            self.wait_flag = 0
        else:
            self.origin_hold_time = self.origin_hold_time_store
        #only 1 target: 
        target = self.targets[1] #Origin
        self.target_location_mc = self.mc_targ_orig #Origin 
        
        target.move_to_position(self.target_location_mc)
        target.cue_trial_start()

        #Turn off lfp things
        self.lfp_plant.turn_off()
        self.targets[0].hide()

    def _start_lfp_target(self):
        #only 1 target: 
        target = self.targets[0] #LFP target
        self.target_location_lfp = self.lfp_targ #LFP target
        
        target.move_to_position(self.target_location_lfp)
        target.cue_trial_start()

        self.lfp_plant.turn_on()

    def _start_lfp_hold(self):
        #make next target visible unless this is the final target in the trial
        pass

    def _start_hold_penalty(self):
        #hide targets
        for target in self.targets:
            target.hide()

        self.tries += 1
        self.target_index = -1

        #Turn off lfp things
        self.lfp_plant.turn_off()
        self.targets[0].hide()

    def _end_origin(self):
        self.targets[1].cue_trial_end_success()

    def _test_enter_origin(self, ts):
        cursor_pos = self.mc_plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location_mc)
        return d <= (self.mc_target_radius - self.mc_cursor_radius)

    # def _test_origin_timeout(self, ts):
    #     return ts>self.timeout_time

    def _test_leave_origin(self, ts):
        if self.manual_control_type == 'joystick':
            if hasattr(self,'touch'):
                if self.touch <0.5:
                    self.last_touch_zero_event = time.time()
                    return True

        cursor_pos = self.mc_plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location_mc)
        return d > (self.mc_target_radius - self.mc_cursor_radius)

    def _test_origin_hold_complete(self,ts):
        return ts>=self.origin_hold_time

    # def _test_enter_lfp_target(self, ts):
    #     '''
    #     return true if the distance between center of cursor and target is smaller than the cursor radius
    #     '''
    #     cursor_pos = self.lfp_plant.get_endpoint_pos()
    #     cursor_pos = [cursor_pos[0], cursor_pos[2]]
    #     targ_loc = np.array([self.target_location_lfp[0], self.target_location_lfp[2]])


    #     d = np.linalg.norm(cursor_pos - targ_loc)
    #     return d <= (self.lfp_target_rad - self.lfp_cursor_rad)

    # def _test_leave_early(self, ts):
    #     '''
    #     return true if cursor moves outside the exit radius
    #     '''
    #     cursor_pos = self.lfp_plant.get_endpoint_pos()
    #     d = np.linalg.norm(cursor_pos - self.target_location_lfp)
    #     rad = self.lfp_target_rad - self.lfp_cursor_rad
    #     return d > rad

    def _test_hold_penalty_end(self, ts):
        return ts>self.hold_penalty_time

    def _end_lfp_hold(self):
        # change current target color to green
        self.targets[0].cue_trial_end_success()


    def move_plant(self):
        if self.state in self.lfp_cursor_on:
            feature_data = self.get_features()


            # Save the "neural features" (e.g. spike counts vector) to HDF file
            for key, val in feature_data.items():
                self.task_data[key] = val
            
            Bu = None
            assist_weight = 0
            target_state = np.zeros([self.decoder.n_states, self.decoder.n_subbins])

            ## Run the decoder
            neural_features = feature_data[self.extractor.feature_type]

            self.call_decoder(neural_features, target_state, Bu=Bu, assist_level=assist_weight, feature_type=self.extractor.feature_type)

           
            ## Drive the plant to the decoded state, if permitted by the constraints of the plant
            self.lfp_plant.drive(self.decoder)
            self.task_data['decoder_state'] = decoder_state = self.decoder.get_state(shape=(-1,1))
            #return decoder_state
           

        #Sets the plant configuration based on motiontracker data. For manual control, uses
        #motiontracker data. If no motiontracker data available, returns None'''
        
        #get data from motion tracker- take average of all data points since last poll
        if self.state in self.no_plant_update_states:
            pt = np.array([0, 0, 0])
            print('no update')
        else:
            if self.manual_control_type == 'motiondata':
                pt = self.motiondata.get()
                if len(pt) > 0:
                    pt = pt[:, self.marker_num, :]
                    conds = pt[:, 3]
                    inds = np.nonzero((conds>=0) & (conds!=4))[0]
                    if len(inds) > 0:
                        pt = pt[inds,:3]

                        #scale actual movement to desired amount of screen movement
                        pt = pt.mean(0) * self.scale_factor
                        #Set y coordinate to 0 for 2D tasks
                        if self.limit2d: 
                            #pt[1] = 0

                            pt[2] = pt[1].copy()
                            pt[1] = 0


                        pt[1] = pt[1]*2
                        # Return cursor location
                        self.no_data_count = 0
                        pt = pt * mm_per_cm #self.convert_to_cm(pt)
                    else: #if no usable data
                        self.no_data_count += 1
                        pt = None
                else: #if no new data
                    self.no_data_count +=1
                    pt = None
            
            elif self.manual_control_type == 'joystick':
                pt = self.joystick.get()
                #if touch sensor on: 
                try: 
                    self.touch = pt[-1][0][2]
                except:
                    pass

                if len(pt) > 0:
                    pt = pt[-1][0]
                    pt[0]=1-pt[0]; #Switch L / R axes
                    calib = [0.497,0.517] #Sometimes zero point is subject to drift this is the value of the incoming joystick when at 'rest' 
                    if self.joystick_method==0:
                        #pt = pt[-1][0]
                        #pt[0]=1-pt[0]; #Switch L / R axes
                        #calib = [0.497,0.517] #Sometimes zero point is subject to drift this is the value of the incoming joystick when at 'rest' 
                        # calib = [ 0.487,  0.   ]
                        
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
                    pt = self.current_pt

                #self.plant.set_endpoint_pos(self.current_pt)
                self.last_pt = self.current_pt.copy()
            
            elif self.manual_control_type == None:
                pt = None
                try: 
                    pt0 = self.motiondata.get()
                    self.manual_control_type='motiondata'
                except:
                    print('not motiondata')

                try:
                    pt0 = self.joystick.get()
                    self.manual_control_type = 'joystick'
                
                except:
                    print('not joystick data')

        # Set the plant's endpoint to the position determined by the motiontracker, unless there is no data available
        if self.manual_control_type is not None:
            if pt is not None and len(pt)>0:
                self.mc_plant.set_endpoint_pos(pt)   

    @staticmethod
    def lfp_mod_4targ_plus_mc_orig(nblocks=100, boundaries=(-18,18,-12,12), xaxis=-8):
        '''
        See lfp_mod_4targ for lfp target explanation 

        '''
        new_zero = (boundaries[3]+boundaries[2]) / 2.
        new_scale = (boundaries[3] - boundaries[2]) / 2.
        kin_targs = np.array([-0.40625,  0.     ,  0.40625,  0.8125 ])
        lfp_targ_y = (new_scale*kin_targs) + new_zero

        for i in range(nblocks):
            temp = lfp_targ_y.copy()
            np.random.shuffle(temp)
            if i==0:
                z = temp.copy()
            else:
                z = np.hstack((z, temp))

        #Fixed X axis: 
        x = np.tile(xaxis,(nblocks*4))
        y = np.zeros(nblocks*4)
                
        lfp = np.vstack([x, y, z]).T
        origin = np.zeros(( lfp.shape ))

        it = iter([dict(lfp=lfp[i,:], origin=origin[i,:]) for i in range(lfp.shape[0])])
        return it

class LFP_Mod_plus_MC_reach(LFP_Mod_plus_MC_hold):
    mc_cursor_radius = traits.Float(.5, desc="Radius of cursor")
    mc_target_radius = traits.Float(3, desc="Radius of MC target")
    mc_cursor_color = (.5,0,.5,1)
    mc_plant_type_options = plantlist.keys()
    mc_plant_type = traits.OptionsList(*plantlist, bmi3d_input_options=plantlist.keys())
    origin_hold_time = traits.Float(.2, desc="Hold time in center")
    mc_periph_holdtime = traits.Float(.2, desc="Hold time in center")
    mc_timeout_time = traits.Float(10, desc="Time allowed to go between targets")
    exclude_parent_traits = ['goal_cache_block'] #redefine this to NOT include marker_num, marker_count
    marker_num = traits.Int(14,desc='Index')
    marker_count = traits.Int(16,desc='Num of markers')

    scale_factor = 3.0 #scale factor for converting hand movement to screen movement (1cm hand movement = 3.5cm cursor movement)
    wait_flag = 1
    # NOTE!!! The marker on the hand was changed from #0 to #14 on
    # 5/19/13 after LED #0 broke. All data files saved before this date
    # have LED #0 controlling the cursor.
    limit2d = 1
    #   state_file = open("/home/helene/preeya/tot_pw.txt","w")
    state_cnt = 0

    status = dict(
        wait = dict(start_trial="origin", stop=None),
        origin = dict(enter_origin="origin_hold", stop=None),
        origin_hold = dict(origin_hold_complete="lfp_target",leave_origin="hold_penalty", stop=None),
        lfp_target = dict(enter_lfp_target="lfp_hold", leave_origin="hold_penalty", powercap_penalty="powercap_penalty", stop=None),
        lfp_hold = dict(leave_early="lfp_target", lfp_hold_complete="mc_target", leave_origin="hold_penalty",powercap_penalty="powercap_penalty"),
        mc_target = dict(enter_mc_target='mc_hold',mc_timeout="timeout_penalty", stop=None),
        mc_hold = dict(leave_periph_early='hold_penalty',mc_hold_complete="reward"),
        powercap_penalty = dict(powercap_penalty_end="origin"),
        timeout_penalty = dict(timeout_penalty_end="wait"),
        hold_penalty = dict(hold_penalty_end="origin"),
        reward = dict(reward_end="wait"),
    )

    
    static_states = ['origin'] # states in which the decoder is not run
    trial_end_states = ['reward', 'timeout_penalty']
    lfp_cursor_on = ['lfp_target', 'lfp_hold', 'reward']

    sequence_generators = ['lfp_mod_plus_MC_reach', 'lfp_mod_plus_MC_reach_INV']

    def __init__(self, *args, **kwargs):
        # import pickle
        # decoder = pickle.load(open('/storage/decoders/cart20141216_03_cart_new2015_2.pkl'))
        # self.decoder = decoder
        super(LFP_Mod_plus_MC_reach, self).__init__(*args, **kwargs)

        mc_origin = VirtualCircularTarget(target_radius=self.mc_target_radius, target_color=RED)
        mc_periph = VirtualCircularTarget(target_radius=self.mc_target_radius, target_color=RED)
        lfp_target = VirtualSquareTarget(target_radius=self.lfp_target_rad, target_color=self.lfp_target_color)

        self.targets = [lfp_target, mc_origin, mc_periph]

        # #Should be unnecessary: 
        # for target in self.targets:
        #     for model in target.graphics_models:
        #         self.add_model(model)

        # self.lfp_plant = plantlist[self.lfp_plant_type] 
        # if hasattr(self.lfp_plant, 'graphics_models'):
        #     for model in self.lfp_plant.graphics_models:
        #         self.add_model(model)

        # self.mc_plant = plantlist[self.mc_plant_type]
        # if hasattr(self.mc_plant, 'graphics_models'):
        #     for model in self.mc_plant.graphics_models:
        #         self.add_model(model)

    def _parse_next_trial(self):
        t = self.next_trial
        self.lfp_targ = t['lfp']
        self.mc_targ_orig = t['origin']
        self.mc_targ_periph = t['periph']

    def _start_mc_target(self):
        #Turn off LFP things
        self.lfp_plant.turn_off()
        self.targets[0].hide()
        self.targets[1].hide()

        target = self.targets[2] #MC target
        self.target_location_mc = self.mc_targ_periph
        
        target.move_to_position(self.target_location_mc)
        target.cue_trial_start()

    def _test_enter_mc_target(self,ts):
        cursor_pos = self.mc_plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location_mc)
        return d <= (self.mc_target_radius - self.mc_cursor_radius)

    def _test_mc_timeout(self, ts):
        return ts>self.mc_timeout_time

    def _test_leave_periph_early(self, ts):
        cursor_pos = self.mc_plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location_mc)
        rad = self.mc_target_radius - self.mc_cursor_radius
        return d > rad

    def _test_mc_hold_complete(self, ts):
        return ts>self.mc_periph_holdtime

    def _timeout_penalty_end(self, ts):
        print('timeout', ts)
        #return ts > 1.
        return True

    def _end_mc_hold(self):
        self.targets[2].cue_trial_end_success()

    # def _cycle(self):
    #     if self.state_cnt < 3600*3:
    #         self.state_cnt +=1
    #         s = "%s\n" % self.state
    #         self.state_file.write(str(s))

    #     if self.state_cnt == 3600*3:
    #         self.state_file.close()

    #     super(LFP_Mod_plus_MC_reach, self)._cycle()

    def _start_reward(self):
        super(LFP_Mod_plus_MC_reach, self)._start_reward()
        lfp_targ = self.targets[0]
        mc_orig = self.targets[1]
        lfp_targ.hide()
        mc_orig.hide()

    @staticmethod
    def lfp_mod_plus_MC_reach(nblocks=100, boundaries=(-18,18,-12,12), xaxis=-8, target_distance=6, n_mc_targets=4, mc_target_angle_offset=0,**kwargs):
        new_zero = (boundaries[3]+boundaries[2]) / 2.
        new_scale = (boundaries[3] - boundaries[2]) / 2.
        kin_targs = np.array([-0.40625,  0.     ,  0.40625,  0.8125 ])
        lfp_targ_y = (new_scale*kin_targs) + new_zero

        for i in range(nblocks):
            temp = lfp_targ_y.copy()
            np.random.shuffle(temp)
            if i==0:
                z = temp.copy()
            else:
                z = np.hstack((z, temp))

        #Fixed X axis: 
        x = np.tile(xaxis,(nblocks*4))
        y = np.zeros(nblocks*4)
        lfp = np.vstack([x, y, z]).T
        origin = np.zeros(( lfp.shape ))

        theta = []
        for i in range(nblocks*4):
            temp = np.arange(0, 2*np.pi, 2*np.pi/float(n_mc_targets))
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)
        theta = theta + (mc_target_angle_offset*(np.pi/180.))
        x = target_distance*np.cos(theta)
        y = np.zeros(len(theta))
        z = target_distance*np.sin(theta)
        periph = np.vstack([x, y, z]).T
        it = iter([dict(lfp=lfp[i,:], origin=origin[i,:], periph=periph[i,:]) for i in range(lfp.shape[0])])
        
        if ('return_arrays' in kwargs.keys()) and kwargs['return_arrays']==True:
            return lfp, origin, periph
        else:
            return it

    @staticmethod
    def lfp_mod_plus_MC_reach_INV(nblocks=100, boundaries=(-18,18,-12,12), xaxis=-8, target_distance=6, n_mc_targets=4, mc_target_angle_offset=0):
        kw = dict(return_arrays=True)
        lfp, origin, periph = LFP_Mod_plus_MC_reach.lfp_mod_plus_MC_reach(nblocks=nblocks, boundaries=boundaries, xaxis=xaxis, target_distance=target_distance, 
            n_mc_targets=n_mc_targets, mc_target_angle_offset=mc_target_angle_offset,**kw)

        #Invert LFP:
        lfp[:,2] = -1.0*lfp[:,2] 
        it = iter([dict(lfp=lfp[i,:], origin=origin[i,:], periph=periph[i,:]) for i in range(lfp.shape[0])])
        return it

    



class DummyPlant(object):
    def __init__(self,*args,**kwargs):
        self.desc = 'dummy_plant object'

    def get_intrinsic_coordinates(self):
        return None