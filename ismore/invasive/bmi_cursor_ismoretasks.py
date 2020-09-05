from riglib.stereo_opengl.window import WindowDispl2D
from riglib.stereo_opengl.primitives import Circle
from ismore import plants
from ismoretasks import IsMoreBase
from bmi_ismoretasks import PlantControlBase, BMICursorControl
from riglib.bmi.assist import Assister

COLORS = {
    'black': (0, 0, 0, 1),
    'red':   (1, 0, 0, 1),
    'green': (0, 1, 0, 1),
    'blue':  (0, 0, 1, 1),
    'white': (1, 1, 1, 1),
}

class CursorBase(IsMoreBase):
    '''
    A base class for all Cursor / Virtual IsMore tasks. Creates the plant objects
    and updates the display of the plant at every iteration of the task.
    '''

    # settable parameters on web interface
    plant_type = traits.OptionsList(*plants.CURSOR_PLANT_LIST.keys(), bmi3d_input_options=plants.CURSOR_PLANT_LIST.keys())
    cursor_radius = traits.Float(0.4, desc='Radius of cursor')

    def _instantiate_plant(self):
        self.plant = plants.CURSOR_PLANT_LIST[self.plant_type]
        self.add_dtype('plant_pos', 'f8', (len(self.plant.get_pos() ),))

    def _cycle(self):
        # Note: All classes that inherit from this class should probably call
        # the following code at some point during their _cycle methods
            # self.plant_pos[:] = self.plant.get_pos()
            # self.plant_vel[:] = self.plant.get_vel()
            # self.update_plant_display()
            # self.task_data['plant_pos'] = self.plant_pos.values
            # self.task_data['plant_vel'] = self.plant_vel.values
        pass
    def verify_plant_data_arrival(self, n_secs):
        pass 

    def init_plant_display(self):
        self.cursor = Circle(np.array([0, 0]), self.cursor_radius, COLORS['white'])
        self.add_model(self.cursor)

    def update_plant_display(self):
        self.cursor.center_pos = self.plant_pos.values

class CursorControlBase(CursorBase, PlantControlBase):
    '''Abstract base class for controlling cursors through a sequence of targets.
       Note: Analog to Manual Control Multi in bmi3d_input_options
       '''
    
    def __init__(self, *args, **kwargs):
        super(PlantControlBase, self).__init__(*args, **kwargs)
        self.command_vel = pd.Series(0.0, self.vel_states)
        self.target_pos  = pd.Series(0.0, self.pos_states)

        self.add_dtype('command_vel', 'f8', (len(self.command_vel),))
        self.add_dtype('target_pos',  'f8', (len(self.target_pos),))
        self.add_dtype('target_index', 'i', (1,))

        self.init_target_display()

    def init_target_display(self):
        self.target1  = Circle(np.array([0, 0]), self.target_radius, COLORS['green'], False)
        self.target2  = Circle(np.array([0, 0]), self.target_radius, COLORS['green'], False)
        self.add_model(self.target1)
        self.add_model(self.target2)

    def hide_targets(self):
        self.target1.visible        = False
        self.target2.visible        = False

    def _test_enter_target(self, ts):
        cursor_pos = self.plant.get_pos()
        d = np.linalg.norm(cursor_pos - self.target_pos)
        return d <= (self.target_radius - self.cursor_radius)
        
    #### STATE FUNCTIONS ####
    def _start_target(self):
        self.target_index += 1
        self.target_pos = self.targets[self.target_index]
        self.target1.color = COLORS['red']
        self.target2.color = COLORS['red']
        if self.target_index % 2 == 0:
            target = self.target1        
        else:
            target = self.target2

        target.center_pos        = self.target_pos
        target.visible           = True
        
    def _start_hold(self):
    # make next target visible unless this is the final target in the trial
        if (self.target_index + 1) < self.chain_length:
            next_target_pos = self.targets[self.target_index+1]

            if self.target_index % 2 == 0:
                target = self.target2
            else:
                target = self.target1
                
            target.center_pos        = np.array([x, y])
            target.visible           = True
    
    def _end_hold(self):
        if self.target_index % 2 == 0:            
            self.target1.color = COLORS['green']
        else:            
            self.target2.color = COLORS['green']

    def _start_reward(self):
        if self.target_index % 2 == 0:
            self.target1.visible        = True
        else:
            self.target2.visible        = True
        
class BMICursorControl(CursorControlBase, BMIControl):

    def create_assister(self):
        #Assist level set in 'LinearlyDecreasingAssister'
        start_level, end_level = self.assist_level
        kwargs = dict(decoder_binlen=self.decoder.binlen, target_radius=self.target_radius)
        

class SimpleCursorAssister(Assister):
    '''
    Constant velocity toward the target if the cursor is outside the target. If the
    cursor is inside the target, the speed becomes the distance to the center of the
    target divided by 2.
    '''

    def init_assister(self, *args, **kwargs):
        '''    Docstring    '''
        self.decoder_binlen = kwargs.pop('decoder_binlen', 0.1)
        self.assist_speed = kwargs.pop('assist_speed', 5.)
        self.target_radius = kwargs.pop('target_radius', 2.)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''    Docstring    '''
        assist_weight = 0.

        if assist_level > 0:
            cursor_pos = np.array(current_state[0:3,0]).ravel()
            target_pos = np.array(target_state[0:3,0]).ravel()
            decoder_binlen = self.decoder_binlen
            speed = self.assist_speed * decoder_binlen
            target_radius = self.target_radius
            Bu = self.endpoint_assist_simple(cursor_pos, target_pos, decoder_binlen, speed, target_radius, assist_level)
            assist_weight = assist_level 

        # return Bu, assist_weight
        return dict(x_assist=Bu, assist_level=assist_weight)

    @staticmethod 
    def endpoint_assist_simple(cursor_pos, target_pos, decoder_binlen=0.1, speed=0.5, target_radius=2., assist_level=0.):
        '''
        Estimate the next state using a constant velocity estimate moving toward the specified target

        Parameters
        ----------
        cursor_pos: np.ndarray of shape (3,)
            Current position of the cursor
        target_pos: np.ndarray of shape (3,)
            Specified target position
        decoder_binlen: float
            Time between iterations of the decoder
        speed: float
            Speed of the machine-assisted cursor
        target_radius: float
            Radius of the target. When the cursor is inside the target, the machine assisted cursor speed decreases.
        assist_level: float
            Scalar between (0, 1) where 1 indicates full machine control and 0 indicates full neural control.

        Returns
        -------
        x_assist : np.ndarray of shape (7, 1)
            Control vector to add onto the state vector to assist control.
        '''
        diff_vec = target_pos - cursor_pos 
        dist_to_target = np.linalg.norm(diff_vec)
        dir_to_target = diff_vec / (np.spacing(1) + dist_to_target)
        
        if dist_to_target > target_radius:
            assist_cursor_pos = cursor_pos + speed*dir_to_target
        else:
            assist_cursor_pos = cursor_pos + speed*diff_vec/2

        assist_cursor_vel = (assist_cursor_pos-cursor_pos)/decoder_binlen
        x_assist = np.hstack([assist_cursor_pos, assist_cursor_vel, 1])
        x_assist = np.mat(x_assist.reshape(-1,1))
        return x_assist


