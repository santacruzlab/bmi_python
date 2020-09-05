''' Simple task to run in command line that combines features: 
    a) relay_blackrockrowbyte
    b) brainampdata
    c) simple state structure
    d) video recordings
'''

'''
Test script to run the visual feedback task from the command line
'''

from riglib import experiment
from riglib.experiment import traits, Sequence

from features.hdf_features import SaveHDF
from features.arduino_features import BlackrockSerialDIORowByte
from ismore.start_video_features import StartVideo
from ismore.brainamp_features import BrainAmpData #, SimBrainAmpData
from ismore import brainamp_channel_lists

import numpy as np
import time
from collections import OrderedDict
from db.tracker import dbq
import os
import pygame
import random

sleep_tsk_id = 83
sleep_seq_id = 1830

def run_task(session_length=300, subject='hud1'):
    # feats = [SaveHDF, BlackrockSerialDIORowByte, StartVideo, BrainAmpData]
    feats = [SaveHDF, BlackrockSerialDIORowByte, BrainAmpData]
    
    # for testing
    # feats = [SaveHDF, BlackrockSerialDIORowByte, SimBrainAmpData]

    # Task = experiment.make(SleepTask, feats)
    # targets = SleepTask.sleep_dummy()

    Task = experiment.make(SleepTask_w_reactivation, feats)
    targets = SleepTask_w_reactivation.sleep_dummy()

    from db.tracker import models
    sleep_task = models.Task.objects.get(pk=sleep_tsk_id)

    # Setup DB : 
    sleep_tsk_entry = models.TaskEntry.objects.get(pk=10091)
    if subject == 'hud1':
        subj_id = 8
    elif subject == 'testing':
        subj_id = 1

    entry = models.TaskEntry(subject_id=subj_id, task=sleep_task)
    seq = models.Sequence.objects.get(pk=sleep_seq_id)
    entry.sequence = seq

    params = dict(session_length=session_length) #can we add stim_name and interstim_time as params here so that they get saved once only as params?    

    entry.params = '{}' #str(params) #Parameters.from_html({})

    entry.save()
    Task.pre_init(saveid=entry.id)
    import time
    time.sleep(3.)
    print 'START RECORDING'
    #params.trait_norm(Exp.class_traits())
    task = Task(targets, **params)
    task.run_sync()
    cleanup_successful = task.cleanup(dbq, entry.id, subject=subject)
    return cleanup_successful

class SleepTask(Sequence):
    status = dict(
        wait = dict(start_sleep="sleep", stop=None),
        sleep = dict(stop=None),    
    )

    sequence_generators = ['sleep_dummy']
    brainamp_channels = brainamp_channel_lists.sleep_mont_new
    
    def __init__(self,*args, **kwargs):
        super(SleepTask, self).__init__(*args, **kwargs)

    def init(self):
        self.add_dtype('time_IML', 'f8', (1,))
        super(SleepTask, self).init()

    def _test_start_sleep(self, ts):
        return True

    def _start_sleep(self):
        pass

    def _cycle(self):
        self.task_data['time_IML']  = time.time()
        super(SleepTask, self)._cycle()

    @staticmethod
    def sleep_dummy(n=100):
        return np.zeros((n, ))


class SleepTask_w_reactivation(SleepTask):
    status = dict(
        wait = dict(start_sleep='stim_off', stop=None),
        stim_off = dict(stim_turned_on='stim_playing', stop_recording=None, stop=None),  
        stim_playing = dict(end_playing = 'interstim_period',stop=None),      
        interstim_period = dict(interstim_finished='stim_playing', stim_turned_off='stim_off', 
            stop_recording=None, stop= None),
    )

    stim_played = 0

    def __init__(self,*args, **kwargs):
        super(SleepTask_w_reactivation, self).__init__(*args, **kwargs)
        self.stim_on_off= 'F'
        self.rec_stop_flag = 'F'
        self.stim_list = ['jingle4.wav']
        self.stim_file_name = ''
                
        # Pre-load list of stim to play randomly (but in blocks of 10 so 
        # not totally random)
        self.stim_index_to_play = []
        tmp = np.array([0]*10)
        for i in range(100): 
            ix = np.random.permutation(10)
            self.stim_index_to_play.append(tmp[ix])
        self.stim_index_to_play = np.hstack((self.stim_index_to_play ))
        self.stim_index_cnt = 0

        pygame.mixer.init()
        
    def init(self):
        self.add_dtype('stim_on_off', np.str_, 1)
        self.add_dtype('rec_stop_flag', np.str_, 1)
        self.add_dtype('stim_played', 'i', (1,))
        self.add_dtype('stim_vol_scale', 'f8', (1, ))
        self.add_dtype('stim_file_name', np.str_, 20)
        
        self.path_stim_txt = os.path.expandvars('$HOME/code/ismore/invasive/stim_on_off.txt')
        self.path_rec_txt = os.path.expandvars('$HOME/code/ismore/invasive/rec_stop.txt')
        self.path_stim_vol = os.path.expandvars('$HOME/code/ismore/invasive/stim_vol.txt')

        #make sure that initally these variables are set to False
        stim_on_off_txt = open(self.path_stim_txt,'w+')
        stim_on_off_txt.write('F')
        stim_on_off_txt.close()

        rec_stop_txt = open(self.path_rec_txt,'w+')
        rec_stop_txt.write('F') 
        rec_stop_txt.close()

        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds/sleep')
        super(SleepTask_w_reactivation, self).init()

    def _cycle(self):
        super(SleepTask_w_reactivation, self)._cycle()
        #read txt file for stopping recording
        rec_stop_txt = open(self.path_rec_txt,'r')        
        self.rec_stop_flag = rec_stop_txt.read()
        rec_stop_txt.close()

        #read txt file of stimulus ON/OFF
        stim_on_off_txt = open(self.path_stim_txt,'r')
        self.stim_on_off = stim_on_off_txt.read()
        stim_on_off_txt.close()

        #read txt file of stimulus volume
        stim_vol_txt= open(self.path_stim_vol,'r')
        self.stim_vol_scale = stim_vol_txt.read()
        stim_vol_txt.close()


        self.task_data['stim_on_off']  = self.stim_on_off
        self.task_data['rec_stop_flag']  = self.rec_stop_flag
        self.task_data['stim_played'] = self.stim_played
        self.task_data['stim_vol_scale'] = self.stim_vol_scale
        self.task_data['stim_file_name'] = self.stim_file_name
        
    def _test_stim_turned_on(self,ts):
        return self.stim_on_off == 'T' 

    def _test_stim_turned_off(self,ts):
        return self.stim_on_off == 'F' 

    def _start_stim_playing(self):
        # choose which stimulation sound to play
        stim_idx = self.stim_index_to_play[self.stim_index_cnt]
        self.stim_index_cnt += 1
        self.stim_file_name = self.stim_list[stim_idx]
        self.stim_sound = pygame.mixer.Sound(os.path.join(self.sounds_general_dir, self.stim_file_name)) # replace by appropriate jingle + save played jingle name
        self.stim_sound.set_volume(float(self.stim_vol_scale))
        self.stim_played = 1
        self.stim_sound.play()
        self.interstim_interval = [4,6]
        min_time, max_time = self.interstim_interval
        self.interstim_time = random.random() * (max_time - min_time) + min_time

    def _test_end_playing(self, ts):
        return not pygame.mixer.get_busy()

    def _start_interstim_period(self):
        self.stim_played = 0
        self.stim_file_name = ''

    def _test_interstim_finished(self,ts):
        return ts > self.interstim_time

    def _test_stop_recording(self,ts):
        return self.rec_stop_flag == 'T'




if __name__ == '__main__':
    import sys
    cleanup_successful = run_task(session_length = int(float(sys.argv[1])))
    print 'cleanup done'

