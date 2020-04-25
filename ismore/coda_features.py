'''
Features to include when using the Arduino board to remotely start neural 
recording for Plexon/Blackrock/TDT systems and also synchronize data between the task and the neural recordings
'''
import traceback
import numpy as np
import struct
import datetime
import serial
import time

baudrate = 115200 #9600

class CodaSync(object):
    '''
    Sends the trial data from task to CODA system using Arduino
    '''
    def __init__(self, *args, **kwargs):
        '''
        Constructor for CodaSync

        Parameters
        ----------
        None 

        Returns
        -------
        CodaSync instance
        '''
        # self.trial_count = -1
        # self.target_count = -1

        self.trial_count = 0
        self.target_count = 0

        self.port = serial.Serial('/dev/arduino_neurosync', baudrate=baudrate)
        super(CodaSync, self).__init__(*args, **kwargs)

    def set_state(self, condition, **kwargs):
        '''
        Extension of riglib.experiment.Experiment.set_state. Send the trial count when 
        'set state' transitions to 'start_trial' or 'reward' (end of trial). 

        Parameters
        ----------
        condition : string
            Name of new state.
        **kwargs : dict 
            Passed to 'super' set_state function

        Returns
        -------
        None
        '''
        # if condition == 'instruct_go':
        #     print "target count" , self.target_count
        #     self.target_count += 1

        #     #only trigger start on reaches to external target (i.e. not during return to center target): 
        #     if self.target_count % 2:  
        #         self.trial_count += 1
        #         self.send_coda_msg(stop=False)

        if condition == 'trial':
            print "trial count" , self.trial_count
            self.trial_count += 1
            self.send_coda_msg(stop=False)

        elif condition in [ 'wait', 'rest']:
            self.send_coda_msg(stop=True)

        super(CodaSync, self).set_state(condition, **kwargs)

    def send_coda_msg(self, stop=False):
        if stop:
            word = 0
        else:
            word = self.trial_count % 8
            
        #send a 'c' for 'coda' to arduino:
        word_str = 'c' + struct.pack('<H', word)
        self.port.write(word_str)

    @classmethod 
    def pre_init(cls, saveid=None):
        '''
        Run prior to starting the task to remotely start recording from the coda system
        '''

        port = serial.Serial('/dev/arduino_neurosync',baudrate=baudrate)
        port.write('g') #make sure that the CODA pin to start data acquisition is set to 1 (default value)
        time.sleep(0.5)

        port.write('h') # the CODA system is by default set to 1 and starts recording when the pin is set to 0, so here!
        time.sleep(2)

        port.close()
        super(CodaSync, cls).pre_init(saveid=saveid)
