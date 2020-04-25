import pygame
from riglib.stereo_opengl.window import WindowDispl2D
from riglib.stereo_opengl.primitives import Circle, Sector, Line
from config import config
import numpy as np
import os
from riglib.stereo_opengl.models import Group

class PatientDisp(WindowDispl2D):

    def __init__(self, *args, **kwargs):
        # Hard code window size -- full size of monitor for patient screen and exp display
        self.exp_window_size = (500, 200) #Size of window to display to the experimenter
        self.pat_window_size = (1000, 560) #Size of window to display to the patient
        self.txt_pos = tuple(1/4.*np.array(self.pat_window_size))
        self.extended_window_size = tuple((self.exp_window_size[0] + self.pat_window_size[0],
            np.max([self.exp_window_size[1], self.pat_window_size[1]])))

        self.exp_wind_coord = (0, 0)
        self.pat_wind_coord = (self.exp_window_size[0], 0)

        self.pat_background_color = (100, 100, 100, 1)
        self.init_pat_display_done = 0
        self.seq = kwargs.pop('seq', None)
        self.seq_params = kwargs.pop('sequ_params', None)


        super(PatientDisp, self).__init__(*args, **kwargs)

    def screen_init(self):

        os.environ['SDL_VIDEO_WINDOW_POS'] = config.display_start_pos
        os.environ['SDL_VIDEO_X11_WMCLASS'] = "iBMI"
        pygame.init()
        self.font = pygame.font.Font(None, 36)
        self.clock = pygame.time.Clock()

        flags = pygame.NOFRAME
        self._set_workspace_size()

        self.workspace_x_len = self.workspace_top_right[0] - self.workspace_bottom_left[0]
        self.workspace_y_len = self.workspace_top_right[1] - self.workspace_bottom_left[1]

        self.display_border = 10
        
        self.exp_size = np.array(self.exp_window_size, dtype=np.float64)
        self.screen = pygame.display.set_mode(self.extended_window_size, flags)

        self.screen_background_exp = pygame.Surface(self.exp_window_size).convert()
        self.screen_background_pat = pygame.Surface(self.pat_window_size).convert()
        
        self.screen_background_exp.fill(self.background)
        self.screen_background_pat.fill(self.pat_background_color)

        x1, y1 = self.workspace_top_right
        x0, y0 = self.workspace_bottom_left

        self.normalize = np.array(np.diag([1./(x1-x0), 1./(y1-y0), 1]))
        self.center_xform = np.array([[1., 0, -x0], 
                                      [0, 1., -y0],
                                      [0, 0, 1]])
        self.norm_to_screen = np.array(np.diag(np.hstack([self.exp_size, 1])))

        # the y-coordinate in pixel space has to be swapped for some graphics convention reason
        self.flip_y_coord = np.array([[1, 0, 0],
                                      [0, -1, self.exp_size[1]],
                                      [0, 0, 1]])

        self.pos_space_to_pixel_space = np.dot(self.flip_y_coord, np.dot(self.norm_to_screen, np.dot(self.normalize, self.center_xform)))

        self.world = Group(self.models)
        # Dont 'init' self.world in this Window. Just allocates a bunch of OpenGL stuff which is not necessary (and may not work in some cases)
        # self.world.init()

        #initialize surfaces for translucent markers

        self.neutral_arm = pygame.image.load(os.path.expandvars('$ISMORE/invasive/display_pngs/neutral.png'))
        self.neutral_arm = self.neutral_arm.convert()
        
        TRANSPARENT = (255,0,255)

        #Gets value at (0,0) to make background of arm image transparent
        arm_TRANSPARENT = self.neutral_arm.get_at((0,0))

        #Surface ['0'] is the cursor surface
        #Surface [1] is the target surface
        #Surface [2] is the patient display surface
        #Surface [3] is the arm image surface

        self.surf={}
        self.surf['0'] = pygame.Surface(self.exp_size)
        self.surf['0'].fill(TRANSPARENT)
        self.surf['0'].set_colorkey(TRANSPARENT)

        self.surf['1'] = pygame.Surface(self.exp_size)
        self.surf['1'].fill(TRANSPARENT)
        self.surf['1'].set_colorkey(TRANSPARENT)        

        self.surf['2'] = pygame.Surface(self.pat_window_size)
        self.surf['2'].fill(TRANSPARENT)
        self.surf['2'].set_colorkey(TRANSPARENT)

        self.surf['3'] = pygame.Surface(self.pat_window_size)
        self.surf['3'].fill(arm_TRANSPARENT)
        self.surf['3'].set_colorkey(arm_TRANSPARENT)    

         #values of alpha: higher = less translucent
        self.surf['0'].set_alpha(170) #Cursor
        self.surf['1'].set_alpha(130) #Targets
        self.surf['2'].set_alpha(130)
        self.surf['3'].set_alpha(200)
        
        self.exp_surf_background = pygame.Surface(self.exp_window_size).convert()
        self.exp_surf_background.fill(TRANSPARENT)

        self.pat_surf_background = pygame.Surface(self.pat_window_size).convert()
        self.pat_surf_background.fill(TRANSPARENT)

        self.arm_surf_background = pygame.Surface(self.pat_window_size).convert()
        self.arm_surf_background.fill(arm_TRANSPARENT)

        self.i = 0


    def draw_world(self):
        #Refreshes the screen with original background
        self.screen.blit(self.screen_background_exp, self.exp_wind_coord)
        self.screen.blit(self.screen_background_pat, self.pat_wind_coord)

        self.surf['0'].blit(self.exp_surf_background,(0, 0))
        self.surf['1'].blit(self.exp_surf_background,(0, 0))
        self.surf['2'].blit(self.pat_surf_background,(0, 0))
        self.surf['3'].blit(self.arm_surf_background,(0, 0))
        
        # surface index
        self.i = 0

        for model in self.world.models:
            self.draw_model(model)
            self.i += 1

        self.update_pat_display()

        #Renders the new surfaces
        self.screen.blit(self.surf['0'], self.exp_wind_coord)
        self.screen.blit(self.surf['1'], self.exp_wind_coord)
        self.screen.blit(self.surf['2'], self.pat_wind_coord)
        self.screen.blit(self.surf['3'], self.pat_wind_coord)

        pygame.display.update()

    def update_pat_display(self):
        '''Example code: To be overwritten in child classes'''
        if self.state == 'target':
            text = self.font.render("Target Number:"+str(self.trial_ix_print), 1, (1, 1, 1))
            self.surf['2'].blit(text, self.txt_pos)
            self.surf['3'].blit(self.neutral_arm, (0,0))

