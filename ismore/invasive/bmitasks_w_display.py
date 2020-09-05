from invasive import bmi_ismoretasks
from invasive import patient_display
from ismore import settings
from ismore import ismoretasks
import pygame
import os
import numpy as np

class VisualFeedbackWithDisplay(patient_display.PatientDisp, bmi_ismoretasks.VisualFeedback):
    sequence_generators = ['armassist_w_disp']
    image_dict = {}

    def test_(self):
        print 'testing ! '

    def _helper_start_wait(self):
        self.n_attempts = 0
        self.target_index = -1
        self.hide_targets()
        # get target locations for this trial
        self.targets = self.next_trial[0]
        # number of sequential targets in a single trial
        self.chain_length = self.targets.shape[0] 

    def _parse_next_trial(self):
        print self.next_trial
        #target locations: 
        self.trial_type = self.next_trial[0]
        self.image_paths = self.next_trial[1]
        self.image_names = self.next_trial[2]

        for i_ix, i_nm in enumerate(self.image_names):
            if i_nm not in self.image_dict:
                image = pygame.image.load(self.image_paths[i_ix])
                self.image_dict[i_nm] = image.convert()

        self.print_text = self.next_trial[3]
        self.print_text_loc = self.next_trial[4]
        self.print_text_col = self.next_trial[5]
        if np.max(self.print_text_col) > 1.:
            self.print_text_col = self.print_text_col/255.

    ''' These generators have 4 items per itertion: 
        INDEX 0 : np.array of origin and peripheral target location (2 x ndim )
        INDEX 1 : np.array of image file names ( 2 x 1 )
        INDEX 2: np.array of image name (e.g. 'origin', 'blue', etc.)
        INDEX 2 : np.array of text to blit on screen (2 x 1)
        INDEX 3 : np.array of text location
        INDEX 4: np.array of text color
    '''

    def update_pat_display(self):
        '''Example code: To be overwritten in child classes'''
        if self.state == 'target':
            ix = self.target_index
            text = self.font.render(self.print_text[ix], 1, self.print_text_col[ix])
            self.surf['2'].blit(text, self.print_text_loc[ix])
            self.surf['3'].blit(self.image_dict[self.image_names[ix]], (0,0))

    @staticmethod
    def armassist_w_disp(nblocks=10):
        aa_pos_states = ['aa_px', 'aa_py', 'aa_ppsi']
        start = settings.starting_pos[aa_pos_states].values

        # Get colored targ coordinates
        target = {}
        coord_dict = settings.COLOR_TARGETS_TO_COORD_OFFS
        for k in coord_dict:
            target[k] = start+coord_dict[k]

        text_color_dict = ismoretasks.COLORS
        neutral_file = os.path.expandvars('$ISMORE/invasive/display_pngs/neutral.png')

        gen = []
        label_y =  50

        for i in range(nblocks):
            for targ in target:
                tg = np.vstack((start, target[targ]))
                nm = np.array([neutral_file]*2)
                nm2 = np.array(['Start', targ])
                txt = np.array(['Return to Origin', 'Move to '+targ+ ' Target'])
                txt_loc = np.vstack(( np.array([start[0], label_y]), np.array([target[targ][0], label_y])))
                txt_col = np.vstack((text_color_dict['black'], text_color_dict[targ]))
                gen.append([tg, nm, nm2, txt, txt_loc, txt_col])

        return gen

    @staticmethod
    def rehand_w_disp(nblocks=10):
        rh_pos_states = ['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono']
        start = settings.starting_pos[rh_pos_states].values

        # Get target coordinates
        fingers = ['forefinger', 'middlefinger', 'smallfinger']
        rh_ix = np.array([1, 2, [3, 4]])
        motion = ['ext', 'flex']

        grasp_coordinates = {}
        for i_f, f in enumerate(finger):
            for m in motion:
                grasp_coordinates[f+'_'+m] = np.zeros((4, 1))
                grasp_coordinates[f+'_'+m][0] = -10
                for ix in rh_ix[i_f]:
                    if m=='ext':
                        grasp_coordinates[f+'_'+m][ix] = 10
                    elif m == 'flex':
                        grasp_coordinates[f+'_'+m][ix] = -10
        
        target = {}
        coord_dict = settings.COLOR_TARGETS_TO_COORD_OFFS
        for k in coord_dict:
            target[k] = start+coord_dict[k]

        neutral_file = os.path.expandvars('$ISMORE/invasive/display_pngs/neutral_grasp.png')

        gen = []
        label_y =  50

        for i in range(nblocks):
            for targ in target:
                tg = np.vstack((start, target[targ]))
                nm = np.array([neutral_file]*2)
                nm2 = np.array(['Start', targ])
                txt = np.array(['Return to Origin', 'Move to '+targ+ ' Target'])
                txt_loc = np.vstack(( np.array([start[0], label_y]), np.array([target[targ][0], label_y])))
                txt_col = np.vstack((text_color_dict['black'], text_color_dict[targ]))
                gen.append([tg, nm, nm2, txt, txt_loc, txt_col])

        return gen

