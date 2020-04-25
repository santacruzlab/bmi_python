'''Client-side code to receive feedback data from the ArmAssist and ReHand. 
See ArmAssist and ReHand command guides for more details on protocol of what 
data is sent over UDP.
'''

import sys
import time
import socket
import select
import numpy as np

from ismore import settings
from utils.constants import *
from riglib.plants import FeedbackData


class ArmAssistData(FeedbackData):
    '''Client code for use with a DataSource in order to acquire feedback data over UDP from the 
    ArmAssist application.
    '''

    update_freq = 25.
    address     = settings.ARMASSIST_UDP_CLIENT_ADDR
    #feedback_filename = 'armassist_feedback.txt'

    state_names = ['aa_px', 'aa_py', 'aa_ppsi']

    sub_dtype_data     = np.dtype([(name, np.float64) for name in state_names])


    # sub_dtype_data_aux = np.dtype([(name, np.float64) for name in ['force', 'bar_angle']])

    sub_dtype_data_aux = np.dtype([(name, np.float64) for name in ['force', 'bar_angle','load_cell_R', 'load_cell_L']]) 
    
    sub_dtype_data_enc =  np.dtype([(name, np.float64) for name in ['wheel_v1', 'wheel_v2','wheel_v3', 'wheel_t1','wheel_t2','wheel_t3', 'enc_vx', 'enc_vy','enc_vpsi', 'enc_tx','enc_ty','enc_tpsi']])

    dtype = np.dtype([('data',       sub_dtype_data),
                      ('data_filt',  sub_dtype_data),  #keep the same dtype as the raw data! we define a new  name of the subtype of data that we will put in the source,(data_filt) but we keep the same names for the states (pos and vel states names)
                      ('ts',         np.float64),
                      ('ts_arrival', np.float64),
                      ('freq',       np.float64),
                      ('data_aux',   sub_dtype_data_aux),
                      ('ts_aux',     np.float64),
                      ('data_enc',   sub_dtype_data_enc)])

    from scipy.signal import butter, lfilter
    from ismore.filter import Filter
    
    # calculate coefficients for a 2nd-order Butterworth LPF at 1.5 Hz for kinematic data received from the exo
    fs_synch = update_freq #Frequency at which emg and kin data are synchronized
    nyq   = 0.5 * fs_synch
    cuttoff_freq  = 1.5 / nyq
    bpf_kin_coeffs = butter(2, cuttoff_freq, btype='low')
    n_dof             = 3
    pos_filter = [None] * n_dof
    
    for k in range(n_dof):
        pos_filter[k] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1])   
     
    n_getpos_iter = 0

    data_pos_prev = np.zeros(n_dof)
    ts_prev = 0

    def process_received_feedback(self, feedback, ts_arrival):
        '''Process feedback strings of the form:
            "Status ArmAssist freq px py ppsi ts force bar_angle ts_aux\r"
        '''

        items = feedback.rstrip('\r').split(' ')
        
        cmd_id      = items[0]
        dev_id      = items[1]
        data_fields = items[2:]
        
        assert cmd_id == 'Status'
        assert dev_id == 'ArmAssist'

        assert len(data_fields) == 22 

        freq = float(data_fields[0])                    # Hz

        # position data
        px   = float(data_fields[1]) * mm_to_cm         # cm
        py   = float(data_fields[2]) * mm_to_cm         # cm
        ppsi = float(data_fields[3]) * deg_to_rad       # rad
        ts   = int(data_fields[4])   * us_to_s          # sec
        
        # auxiliary data
        force     = float(data_fields[5])               # kg
        bar_angle = float(data_fields[6]) * deg_to_rad  # rad
        ts_aux    = int(data_fields[7])   * us_to_s     # sec

        load_cell_R = float(data_fields[8])            # Kg    
        load_cell_L = float(data_fields[9])            # Kg


        # 2018.08.29
        # vel and torques in wheels reference frame (in the translation direction/rotation axis of wheel)
        wheel_v1 = float(data_fields[10])            # m/s (already translated to linear velocity of the wheel)
        wheel_v2 = float(data_fields[11])            # m/s (already translated to linear velocity of the wheel)
        wheel_v3 = float(data_fields[12])            # m/s (already translated to linear velocity of the wheel)

        wheel_t1 = float(data_fields[13])            # N/m 
        wheel_t2 = float(data_fields[14])            # N/m  
        wheel_t3 = float(data_fields[15])            # N/m 

        # vel and torques converted to X,Y,psi reference frame
        enc_vx = float(data_fields[16])              # m/s
        enc_vy = float(data_fields[17])              # m/s
        enc_vpsi = float(data_fields[18])            # rad/s

        enc_tx = float(data_fields[19])              # N
        enc_ty = float(data_fields[20])              # N
        enc_tpsi = float(data_fields[21])            # N/s


        data_pos     = (px, py, ppsi)
        # data_aux = (force, bar_angle)

        data_aux = (force, bar_angle, load_cell_R, load_cell_L)  # nerea -- to uncomment when load cell implemented

        data_enc = (wheel_v1, wheel_v2, wheel_v3, wheel_t1, wheel_t2, wheel_t3, enc_vx, enc_vy, enc_vpsi, enc_tx, enc_ty, enc_tpsi)
        

        #we don't take the first value of the pos because it is always NaN and if a NaN is introduced in the filter, all the following filtered values will be also NaNs
        if (self.n_getpos_iter <=  1) :            
            self.n_getpos_iter = self.n_getpos_iter +1
            data_pos_filt = data_pos
          
        else: #after 2 points of data -- filter data
            data_pos_filt = np.array([self.pos_filter[k](np.array(data_pos)[k]) for k in range(self.n_dof)]).ravel()

        self.data_prev = data_pos
        self.ts_prev = ts

            
        data = data_pos
      
        data_filt = data_pos_filt

        return np.array([(data,
                          data_filt,
                          ts,
                          ts_arrival,
                          freq,
                          data_aux,
                          ts_aux,
                          data_enc)],
                        dtype=self.dtype)


class ReHandData(FeedbackData):
    '''Client code for use with a DataSource in order to acquire feedback data over UDP from the 
    ReHand application.
    '''

    update_freq = 200.
    address     = settings.REHAND_UDP_CLIENT_ADDR
    #feedback_filename = 'rehand_feedback.txt'

    state_names = ['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono', 
                   'rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']

    sub_dtype_data   = np.dtype([(name, np.float64) for name in state_names])

    
    sub_dtype_torque = np.dtype([(name, np.float64) for name in ['thumb', 'index', 'fing3', 'prono']])
    
    dtype = np.dtype([('data',       sub_dtype_data),
                      ('data_filt',  sub_dtype_data),   #keep the same dtype as the raw data! we define a new  name of the subtype of data that we will put in the source,(data_filt) but we keep the same names for the states (pos and vel states names)
                      ('ts',         np.float64),
                      ('ts_arrival', np.float64),
                      ('freq',       np.float64),
                      ('torque',     sub_dtype_torque)])

    
    ## ----> the ReHand raw data from the encoders is already good, no need to filter it
    from scipy.signal import butter,lfilter
    from ismore.filter import Filter
    # calculate coefficients for a 4th-order Butterworth LPF at 1.5 Hz for kinematic data received from the exo
    fs_synch = update_freq #Frequency at which emg and kin data are synchronized
    nyq   = 0.5 * fs_synch
    cuttoff_freq  = 1.5 / nyq
    bpf_kin_coeffs = butter(2, cuttoff_freq, btype='low')
    pos_vel_filt = [None] * len(state_names)

    for k in range(len(state_names)):
        pos_vel_filt[k] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1])   



    def process_received_feedback(self, feedback, ts_arrival):
        '''Process feedback strings of the form:
            "ReHand Status freq vthumb pthumb tthumb ... tprono ts\r"
        '''

        items = feedback.rstrip('\r').split(' ')
                
        # feedback packet starts with "ReHand Status ...", as opposed 
        #   to "Status ArmAssist ... " for ArmAssist feedback packets
        dev_id      = items[0]
        cmd_id      = items[1]
        data_fields = items[2:]

        assert dev_id == 'ReHand'
        assert cmd_id == 'Status'
        assert len(data_fields) == 14

        freq = float(data_fields[0])

        #display data before being converted to radians

        # print "thumb float:", float(data_fields[1])
        # print "thumb :", (data_fields[1])
        # print "index float:", float(data_fields[4])
        # print "index:", float(data_fields[4])
        # print "3fing float:", float(data_fields[7])
        # print "3fing :", float(data_fields[7])
        # print "prono float:", float(data_fields[10])
        # print "prono:", float(data_fields[10])


        # velocity, position, and torque for the 4 ReHand joints
        vthumb = float(data_fields[1])  * deg_to_rad  # rad
        pthumb = float(data_fields[2])  * deg_to_rad  # rad
        tthumb = float(data_fields[3])                # mNm
        vindex = float(data_fields[4])  * deg_to_rad  # rad
        pindex = float(data_fields[5])  * deg_to_rad  # rad
        tindex = float(data_fields[6])                # mNm
        vfing3 = float(data_fields[7])  * deg_to_rad  # rad
        pfing3 = float(data_fields[8])  * deg_to_rad  # rad
        tfing3 = float(data_fields[9])                # mNm
        vprono = float(data_fields[10]) * deg_to_rad  # rad
        pprono = float(data_fields[11]) * deg_to_rad  # rad
        tprono = float(data_fields[12])               # mNm



        ts = int(data_fields[13]) * us_to_s           # secs

        data   = (pthumb, pindex, pfing3, pprono,
                  vthumb, vindex, vfing3, vprono)
        # data_filt   = (pthumb, pindex, pfing3, pprono,
                  # vthumb, vindex, vfing3, vprono)
        torque = (tthumb, tindex, tfing3, tprono)

        data_filt = np.array([self.pos_vel_filt[k](np.array(data)[k]) for k in range(len(self.state_names))]).ravel()

        return np.array([(data,
                          data_filt,
                          ts, 
                          ts_arrival, 
                          freq,
                          torque)],
                        dtype=self.dtype)
