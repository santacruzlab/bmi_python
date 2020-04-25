'''See the shared Google Drive documentation for an inheritance diagram that
shows the relationships between the classes defined in this file.
'''

import numpy as np
import socket
import time

from riglib import source
from ismore import settings, udp_feedback_client
import ismore_bmi_lib
from utils.constants import *

#import armassist
#import rehand


from riglib.filter import Filter
from riglib.plants import Plant

import os

class BasePlantUDP(Plant):
    '''
    Common UDP interface for the ArmAssist/ReHand
    '''
    debug = 0
    sensor_data_timeout = 1 # seconds. if this number of seconds has passed since sensor data was received, velocity commands will not be sent
    lpf_vel = 0
    # define in subclasses!
    ssm_cls           = None
    addr              = None
    feedback_data_cls = None
    data_source_name  = None
    n_dof             = None
    blocking_joints   = None
    safety_grid       = None
    feedback_str      = ''

    def __init__(self, *args, **kwargs):
        self.source = source.DataSource(self.feedback_data_cls, bufferlen=5, name=self.data_source_name)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # used only for sending

        ssm = self.ssm_cls()
        self.pos_state_names = [s.name for s in ssm.states if s.order == 0]
        self.vel_state_names = [s.name for s in ssm.states if s.order == 1]
        self.aa_xy_ix = [i for i, j in enumerate(ssm.states) if j.name in ['aa_px', 'aa_py']]
        self.aa_psi_ix = [i for i, j in enumerate(ssm.states) if j.name == 'aa_ppsi']
        self.rh_pron_ix = [i for i, j in enumerate(ssm.states) if j.name == 'rh_pprono']
        self.rh_pfings = [(i, j.name) for i, j in enumerate(ssm.states) if j.name in ['rh_pthumb', 'rh_pindex', 'rh_pfing3']]

        self.drive_velocity_raw  = np.zeros((len(self.vel_state_names),))
        self.drive_velocity_raw_fb_gain = np.zeros((len(self.vel_state_names),))
        self.drive_velocity_sent = np.zeros((len(self.vel_state_names),))
        self.drive_velocity_sent_pre_safety = np.zeros((len(self.vel_state_names),))
        self.pre_drive_state = np.zeros((len(self.vel_state_names), ))
        # low-pass filters to smooth out command velocities
        # from scipy.signal import butter
        # b, a = butter(5, 0.1) # fifth order, 2 Hz bandpass (assuming 10 Hz update rate)

        #omega, H = signal.freqz(b, a)
        #plt.figure()
        #plt.plot(omega/np.pi, np.abs(H))

        # self.vel_command_lpfs = [None] * self.n_dof
        # for k in range(self.n_dof):
        #     self.vel_command_lpfs[k] = Filter(b=b, a=a) 

        # self.last_sent_vel = np.ones(self.n_dof) * np.nan

       # calculate coefficients for a 4th-order Butterworth LPF at 1.5 Hz for kinematic data received from the exo
        # fs_synch = 20 #Frequency at which emg and kin data are synchronized
        # nyq   = 0.5 * fs_synch
        # cuttoff_freq  = 1.5 / nyq
        # bpf_kin_coeffs = butter(4, cuttoff_freq, btype='low')

        # self.pos_filt = [None] * self.n_dof
        # for k in range(self.n_dof):
        #     self.pos_filt[k] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1])   
        

    def init(self):
        from riglib import sink
        sink.sinks.register(self.source)

    def start(self):
        # only start this DataSource after it has been registered with 
        # the SinkManager singleton (sink.sinks) in the call to init()
        self.source.start()
        self.ts_start_data = time.time()

    def stop(self):
        # send a zero-velocity command
        self._send_command('SetSpeed %s %s\r' % (self.plant_type, self.pack_vel(np.zeros(self.n_dof))))
        self.source.stop()
        self.feedback_file.close()

    def last_data_ts_arrival(self):
        return self.source.read(n_pts=1)['ts_arrival'][0]

    def _send_command(self, command):
        self.sock.sendto(command, self.addr)

    def pack_vel(self, vel):
        format_str = "%f " * self.n_dof
        return format_str % tuple(vel)


    def send_vel(self, vel):
        assert len(vel) == self.n_dof

        vel = vel.copy()
        vel *= self.vel_gain # change the units of the velocity, if necessary

        self.last_sent_vel = vel
        #command_vel is already fitlered at the task level, no need to filter it again.
        #self.last_sent_vel = filt_vel = np.array([self.vel_command_lpfs[k](vel[k]) for k in range(self.n_dof)]).ravel()
        

        if all(v <= 0.00000001 for v in abs(self.last_sent_vel)):
            print 'last sent vel'
            print self.last_sent_vel 


        if (self.last_data_ts_arrival() == 0) or ((self.last_data_ts_arrival() - time.time()) > self.sensor_data_timeout):
            print "sensor data not received for %s recently enough, not sending velocity command!" % self.plant_type
            return 

        
        # squash any velocities which would take joints outside of the rectangular bounding box
        current_pos = self.get_pos() * self.vel_gain
        projected_pos = current_pos + vel * 0.1
        max_reached, = np.nonzero((projected_pos > self.max_pos_vals) * (vel > 0))
        min_reached, = np.nonzero((projected_pos < self.min_pos_vals) * (vel < 0))

        vel[max_reached] = 0
        vel[min_reached] = 0
        

        self._send_command('SetSpeed %s %s\r' % (self.plant_type, self.pack_vel(vel)))

        # set max speed limts
        faster_than_max_speed, = np.nonzero(np.abs(vel) > self.max_speed)
        vel[faster_than_max_speed] = self.max_speed[faster_than_max_speed] * np.sign(vel[faster_than_max_speed])

        #if we wanna define some limit values for the rehand use the filt_vel. Otherwise use vel
        #self._send_command('SetSpeed %s %s\r' % (self.plant_type, self.pack_vel(filt_vel)))
        self._send_command('SetSpeed %s %s\r' % (self.plant_type, self.pack_vel(vel)))
        # set max speed limts
        faster_than_max_speed, = np.nonzero(np.abs(vel) > self.max_speed)
        vel[faster_than_max_speed] = self.max_speed[faster_than_max_speed] * np.sign(vel[faster_than_max_speed])



        if self.debug:
            print "input vel"
            print vel
            print "vel sent to %s" % self.plant_type
            print vel
            print "current_pos"
            print current_pos
            print "projected_pos"  
            print projected_pos
            print "actual velocity"
            print self.get_vel()


        if self.lpf_vel:
            # squash any velocities which would take joints outside of the rectangular bounding box
            current_pos = self.get_pos() * self.vel_gain
            projected_pos = current_pos + vel * (1.0/20)
            max_reached, = np.nonzero((projected_pos > self.max_pos_vals) * (vel > 0))
            min_reached, = np.nonzero((projected_pos < self.min_pos_vals) * (vel < 0))

            vel[max_reached] = 0
            vel[min_reached] = 0
            
            # set max speed limts
            faster_than_max_speed, = np.nonzero(np.abs(vel) > self.max_speed)
            vel[faster_than_max_speed] = self.max_speed[faster_than_max_speed] * np.sign(vel[faster_than_max_speed])

            if faster_than_max_speed > 0:
                print 'faster_than_max_speed'
                print faster_than_max_speed

            if self.debug:
                print "input vel"
                print vel
                print "vel sent to %s" % self.plant_type
                print vel
                #print "current_pos"
                #print current_pos
                #print "projected_pos"  
                #print projected_pos
                #print "actual velocity"
                #print self.get_vel()

            self._send_command('SetSpeed %s %s\r' % (self.plant_type, self.pack_vel(vel)))
        else:
            self._send_command('SetSpeed %s %s\r' % (self.plant_type, self.pack_vel(vel)))


    # def get_pos(self):
    #     # udp_feedback_client takes care of converting sensor data to cm or rad, as appropriate for the DOF
    #     return np.array(tuple(self.source.read(n_pts=1)['data'][self.pos_state_names][0]))

    def drive(self, decoder):
        vel = decoder['qdot']
        vel_bl = vel.copy()
        feedback_str = ''
        if self.blocking_joints is not None:
            vel_bl[self.blocking_joints] = 0

        if self.safety_grid is not None:

            #If the next position is outside of safety then damp velocity to only go to limit: 
            pos_pred = decoder['q'] + 0.1*vel_bl

            #Make sure predicted AA PX, AA PY within bounds: 
            xy_change = True

            if len(self.aa_xy_ix) > 0:
                if self.safety_grid.is_valid_pos(pos_pred[self.aa_xy_ix]) is False:

                    #If not, make their velocity zero:
                    vel_bl[self.aa_xy_ix] = 0
                    xy_change = False
                    feedback_str = feedback_str+ ' stopping xy from moving'
            else:
                xy_change = False

            # Make sure AA Psi within bounds: 
            if len(self.aa_psi_ix) > 0:

                # If X/Y ok
                if xy_change:
                    mn, mx = self.safety_grid.get_minmax_psi(pos_pred[self.aa_xy_ix])

                # If x/y not ok: 
                else:
                    mn, mx = self.safety_grid.get_minmax_psi(decoder['q'][self.aa_xy_ix])

                # Set psi velocity : 
                if np.logical_and(pos_pred[self.aa_psi_ix] >= mn, pos_pred[self.aa_psi_ix] <= mx):
                    pass
                else:
                    vel_bl[self.aa_psi_ix] = 0
                    feedback_str = feedback_str+ 'stopping psi'

            # Make sure RH Prono within bounds (if SSM is only RH, use settings.starting_pos for AAPX, AAPY)
            if len(self.rh_pron_ix) > 0:

                # If X/Y ok
                if xy_change:
                    mn, mx = self.safety_grid.get_minmax_prono(pos_pred[self.aa_xy_ix])

                # If x/y not ok or not moving bc not part of state pace : 
                else:
                    if len(self.aa_xy_ix) > 0:
                        mn, mx = self.safety_grid.get_minmax_prono(decoder['q'][self.aa_xy_ix])
                    else:
                        mn, mx = self.safety_grid.get_minmax_prono(settings.starting_pos['aa_px'], settings.starting_pos['aa_py'])

                # Set prono velocity : 
                if np.logical_and(pos_pred[self.rh_pron_ix] >= mn, pos_pred[self.rh_pron_ix] <= mx):
                    pass
                else:
                    vel_bl[self.rh_pron_ix] = 0
                    feedback_str = feedback_str+ 'stopping prono'

            # Assure RH fingers are within range: 
            if len(self.rh_pfings) > 0:
                for i, (ix, nm) in enumerate(self.rh_pfings):
                    mn, mx = self.safety_grid.get_rh_minmax(nm)
                    if np.logical_and(pos_pred[ix] >= mn, pos_pred[ix] <= mx):
                        pass
                    else:
                        vel_bl[ix] = 0
                        feedback_str = feedback_str+ 'stopping rh fings'

        self.feedback_str = feedback_str
        self.drive_velocity = vel_bl
        self.send_vel(vel_bl)
        decoder['q'] = self.get_pos()

    def write_feedback(self):
        pos_vel = [str(i) for i in np.hstack(( self.get_pos(), self.get_vel() )) ]
        #self.feedback_file.write(','.join(pos_vel)+'\n')
        if self.feedback_str != '':
            self.feedback_file.write(self.feedback_str+ time.ctime() + '\n')

class ArmAssistPlantUDP(BasePlantUDP):
    '''Sends velocity commands and receives feedback over UDP. Can be used
    with either the real or simulated ArmAssist.
    '''

    ssm_cls           = ismore_bmi_lib.StateSpaceArmAssist
    addr              = settings.ARMASSIST_UDP_SERVER_ADDR
    feedback_data_cls = udp_feedback_client.ArmAssistData
    data_source_name  = 'armassist'
    n_dof             = 3
    plant_type        = 'ArmAssist'
    vel_gain          = np.array([cm_to_mm, cm_to_mm, rad_to_deg]) # convert units to: [mm/s, mm/s, deg/s]
    max_pos_vals      = np.array([np.inf, np.inf, np.inf])
    min_pos_vals      = np.array([-np.inf, -np.inf, -np.inf])
    max_speed         = np.array([np.inf, np.inf, np.inf])
    feedback_file     = open(os.path.expandvars('$HOME/code/bmi3d/log/armassist.txt'), 'w')
    #max_speed         = np.array([40, 60, 20]) # in mm/s and deg/s
    #max_speed         = np.array([60, 80, 50]) # in mm/s and deg/s

    #parameters for kinematics low-pass filtering
    from scipy.signal import butter, lfilter
    from ismore.filter import Filter
    fs_synch = 25 #Frequency at which emg and kin data are synchronized
    nyq   = 0.5 * fs_synch
    cuttoff_freq  = 1.5 / nyq
    bpf_kin_coeffs = butter(2, cuttoff_freq, btype='low')
    n_dof             = 3
    vel_filter = [None] * n_dof
    for k in range(n_dof):
        vel_filter[k] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1])   

    n_getpos_iter= 0

    def __init__(self, *args, **kwargs):
        super(ArmAssistPlantUDP, self).__init__(*args, **kwargs)

    def set_pos_control(self): # position control with global reference system
        self._send_command('SetControlMode ArmAssist Position')

    def set_global_control(self): #velocity control with global reference system
        self._send_command('SetControlMode ArmAssist Global')

    def set_trajectory_control(self): #trajectory control with global reference system
        self._send_command('SetControlMode ArmAssist Trajectory')

    def send_vel(self, vel):
        vel = vel.copy()

        # units of vel should be: [cm/s, cm/s, rad/s]
        assert len(vel) == self.n_dof

        # convert units to: [mm/s, mm/s, deg/s] to send them through UDP to the ArmAssist application
        vel[0] *= cm_to_mm
        vel[1] *= cm_to_mm
        vel[2] *= rad_to_deg

        # set max speed limts
        faster_than_max_speed, = np.nonzero(np.abs(vel) > self.max_speed)
        vel[faster_than_max_speed] = self.max_speed[faster_than_max_speed] * np.sign(vel[faster_than_max_speed])
 

        self.debug = True
        if self.debug:
            # print "vel sent to armassist"
            # print vel

            if faster_than_max_speed.any() > 0:
                print 'faster_than_max_speed'
                print faster_than_max_speed
                print "speed set to: "
                print vel


        self._send_command('SetSpeed ArmAssist %f %f %f\r' % tuple(vel))


    # get raw position
    def get_pos_raw(self):
        # udp_feedback_client takes care of converting sensor data to cm or rad, as appropriate for the DOF
        #get the last  poitns of data of the armassist and low-pass filter
        return  np.array(tuple(self.source.read(n_pts=1)['data'][self.pos_state_names][0]))
   

    # get filtered position
    def get_pos(self):
        return  np.array(tuple(self.source.read(n_pts=1)['data_filt'][self.pos_state_names][0]))
              
    # calculate vel from raw position
    def get_vel_raw(self):    
        recent_pos_data = self.source.read(n_pts=2)
        pos = recent_pos_data['data'][self.pos_state_names]
        ts = recent_pos_data['ts']

        delta_pos = np.array(tuple(pos[1])) - np.array(tuple(pos[0]))
        delta_ts  = ts[1] - ts[0]
        vel = delta_pos / delta_ts

        #filt_vel = np.array([self.vel_command_lpfs[k](vel[k]) for k in range(self.n_dof)]).ravel() #nerea --> to test!

        if ts[0] != 0 and any(np.isnan(v) for v in vel):
            print "WARNING -- delta_ts = 0 in AA vel calculation:", vel
            for i in range(3):
                if np.isnan(vel[i]):
                    vel[i] = 0

        return vel

    #calculate vel from raw position and filter
    def get_vel(self):
        recent_pos_data = self.source.read(n_pts=2)
        pos = recent_pos_data['data'][self.pos_state_names]
        ts = recent_pos_data['ts']

        delta_pos = np.array(tuple(pos[1])) - np.array(tuple(pos[0]))
        delta_ts  = ts[1] - ts[0]
        
        vel = delta_pos / delta_ts

        if ts[0] != 0 and any(np.isnan(v) for v in vel):
            print "WARNING -- delta_ts = 0 in AA vel calculation:", vel
            for i in range(3):
                if np.isnan(vel[i]):
                    vel[i] = 0

        # the first value of the pos because it is always NaN and if a NaN is introduced in the filter, all the following filtered values will be also NaNs
        if np.any(np.isnan(vel)):
            self.n_getpos_iter = self.n_getpos_iter +1
            vel_filt = vel
        else:
            vel_filt = np.array([self.vel_filter[k](vel[k]) for k in range(self.n_dof)]).ravel()

        return vel_filt

    def send_pos(self, pos, time):
        pos = pos.copy()
        
        # units of vel should be: [cm/s, cm/s, rad/s]
        assert len(pos) == 3
              

        # convert units to: [mm/s, mm/s, deg/s]
        pos[0] *= cm_to_mm
        pos[1] *= cm_to_mm
        pos[2] *= rad_to_deg


        # mode 1: the forearm angle (psi) stays the same as it is. mode 2: psi will move according to the determined value
        mode = 2

        pos_command = np.zeros(5)
        pos_command[0] = pos[0]
        pos_command[1] = pos[1]
        pos_command[2] = pos[2]
        pos_command[3] = time
        pos_command[4] = mode


        print "pos"
        print pos
        print "time"
        print time
        self._send_command('SetPosition ArmAssist %f %f %f %f %f\r' % tuple(pos_command))



    def enable(self):
        self._send_command('SetControlMode ArmAssist Global\r')

    def disable(self):
        self._send_command('SetControlMode ArmAssist Disable\r')

    def enable_watchdog(self, timeout_ms):
        print 'ArmAssist watchdog not enabled, doing nothing'


    def send_traj(self, pos_vel):
        pos_vel = pos_vel.copy()
       
        # units of vel should be: [cm/s, cm/s, rad/s]
        assert len(pos_vel) == 6
     
        # units to are alread in [mm/s, mm/s, rad/s] 
        # convert values to integers to reduce noise

        #pos_vel_int = np.rint(pos_vel)
       
        pos_vel_int = pos_vel


        print "trajectory sent to AA"
        print "x     y  psi     vx      vy     vpsi"
        print pos_vel_int

        traj_command = np.zeros(6)
        traj_command[0] = pos_vel_int[0]
        traj_command[1] = pos_vel_int[1]
        traj_command[2] = pos_vel_int[2]
        traj_command[3] = pos_vel_int[3]
        traj_command[4] = pos_vel_int[4]
        traj_command[5] = pos_vel_int[5]


        self._send_command('SetTrajectory ArmAssist %d %d %d %d %d %d\r' % tuple(traj_command))

class DummyPlantUDP(object):

    drive_velocity_raw = np.array([0,0,0])
    drive_velocity_sent = np.array([0,0,0])
    drive_velocity_sent_pre_safety = np.array([0,0,0])
    pre_drive_state = np.array([0, 0, 0])

    def init(self):
        pass

    def enable(self):
        pass
    def start(self):
        pass

    def stop(self):
        pass

    def write_feedback(self):
        pass

    def get_pos_raw(self):
        return np.array([0,0,0])

    def get_pos(self):
        return np.array([0,0,0])

    def get_vel_raw(self):
        return np.array([0,0,0])

    def get_vel(self):
        return np.array([0,0,0])

class ReHandPlantUDP(BasePlantUDP):
    '''Sends velocity commands and receives feedback over UDP. Can be used
    with either the real or simulated ReHand.
    '''

    ssm_cls           = ismore_bmi_lib.StateSpaceReHand
    addr              = settings.REHAND_UDP_SERVER_ADDR
    feedback_data_cls = udp_feedback_client.ReHandData
    data_source_name  = 'rehand'
    n_dof             = 4
    plant_type        = 'ReHand'
    vel_gain          = np.array([rad_to_deg, rad_to_deg, rad_to_deg, rad_to_deg])
    max_pos_vals      = np.array([60, 60, 60, 90], dtype=np.float64) # degrees
    min_pos_vals      = np.array([25, 25, 25, 25], dtype=np.float64) # degrees
    max_speed         = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float64) # degrees/sec
    #max_speed         = np.array([15., 15., 15., 15.], dtype=np.float64) # degrees/sec
    feedback_file     = open(os.path.expandvars('$HOME/code/bmi3d/log/rehand.txt'), 'w')

    def send_vel(self, vel):
        vel = vel.copy()

        # units of vel should be: [rad/s, rad/s, rad/s, rad/s]
        assert len(vel) == self.n_dof
        
        # convert units to: [deg/s, deg/s, deg/s, deg/s]
        vel *= rad_to_deg
      
        #filt_vel = np.array([self.vel_command_lpfs[k](vel[k]) for k in range(self.n_dof)]).ravel()

        # set max speed limts
        faster_than_max_speed, = np.nonzero(np.abs(vel) > self.max_speed)
        vel[faster_than_max_speed] = self.max_speed[faster_than_max_speed] * np.sign(vel[faster_than_max_speed])
 

        self.debug = True
        if self.debug:
            # print 'filt_vel in plants in degrees'
            # print filt_vel #*np.array([deg_to_rad, deg_to_rad, deg_to_rad, deg_to_rad])

            if faster_than_max_speed.any() > 0:
                print 'faster_than_max_speed'
                print faster_than_max_speed
                print "speed set to: "
                print vel


        # self.plant.enable() #when we send vel commands always enable the rehand motors 
        # self._send_command('SystemEnable ReHand\r')
        self._send_command('SetSpeed ReHand %f %f %f %f\r' % tuple(vel))

      
    def get_vel_raw(self):
        return np.array(tuple(self.source.read(n_pts=1)['data'][self.vel_state_names][0]))

    def get_vel(self):
        return np.array(tuple(self.source.read(n_pts=1)['data_filt'][self.vel_state_names][0]))

    def enable(self):
        self._send_command('SystemEnable ReHand\r')

    def disable(self):
        self._send_command('SystemDisable ReHand\r')

    def diff_enable(self,DoFs):        
        self._send_command('DiffEnable ReHand %i %i %i %i\r' % tuple(DoFs))

    def get_enable_state(self):
        self._send_command('GetEnableState ReHand\r')

    def enable_watchdog(self, timeout_ms):
        self._send_command('WatchDogEnable ReHand %d\r' % timeout_ms)

    def get_pos_raw(self):
        # udp_feedback_client takes care of converting sensor data to cm or rad, as appropriate for the DOF
        return np.array(tuple(self.source.read(n_pts=1)['data'][self.pos_state_names][0]))

    #get pos filtered
    def get_pos(self):  
        return np.array(tuple(self.source.read(n_pts=1)['data_filt'][self.pos_state_names][0]))


################################################ 


class BasePlantIsMore(Plant):

    # define in subclasses!
    aa_plant_cls = None
    rh_plant_cls = None
    safety_grid = None
    both_feedback_str = ''

    def __init__(self, *args, **kwargs):
        self.aa_plant = self.aa_plant_cls()
        self.rh_plant = self.rh_plant_cls()
        
        self.drive_velocity_raw = np.zeros((7,))
        self.drive_velocity_sent= np.zeros((7,))
        self.drive_velocity_sent_pre_safety = np.zeros((7, ))
        self.pre_drive_state = np.zeros((7, ))
        self.prev_vel_bl_aa = np.zeros((3, ))*np.NaN
        self.prev_vel_bl_rh = np.zeros((4, ))*np.NaN

        self.accel_lim_armassist = np.inf #0.8  
        self.accel_lim_psi = np.inf #0.16
        self.accel_lim_rehand = np.inf #0.16


    def init(self):
        self.aa_plant.init()
        self.rh_plant.init()

    def start(self):
        self.aa_plant.start()
        self.rh_plant.start()
        self.ts_start_data = time.time()

    def stop(self):
        self.aa_plant.stop()
        self.rh_plant.stop()

    def last_data_ts_arrival(self):
        return {
            'ArmAssist': self.aa_plant.last_data_ts_arrival(), 
            'ReHand':    self.rh_plant.last_data_ts_arrival(),
        }

    def send_vel(self, vel):
        self.aa_plant.send_vel(vel[0:3])
        self.rh_plant.send_vel(vel[3:7])

    def get_pos_raw(self):
        aa_pos = self.aa_plant.get_pos_raw()
        rh_pos = self.rh_plant.get_pos_raw()
        return np.hstack([aa_pos, rh_pos])

    def get_pos(self):
        aa_pos = self.aa_plant.get_pos()
        rh_pos = self.rh_plant.get_pos()
        return np.hstack([aa_pos, rh_pos])

    def get_vel_raw(self):
        aa_vel = self.aa_plant.get_vel_raw()
        rh_vel = self.rh_plant.get_vel_raw()
        return np.hstack([aa_vel, rh_vel])

    def get_vel(self):
        aa_vel = self.aa_plant.get_vel()
        rh_vel = self.rh_plant.get_vel()
        return np.hstack([aa_vel, rh_vel])

    def enable(self):
        self.aa_plant.enable()
        self.rh_plant.enable()

    def disable(self):
        self.aa_plant.disable()
        self.rh_plant.disable()

    def drive(self, decoder):
        # print self.aa_plant.aa_xy_ix: [0, 1]
        # print self.aa_plant.aa_psi_ix: [2]
        # print self.rh_plant.rh_pfings: [0, 1, 2]
        # print self.rh_plant.rh_pron_ix: [3]

        vel = decoder['qdot']
        vel_bl = vel.copy()
        current_state = self.get_pos()
        self.pre_drive_state = current_state.copy()
        self.drive_velocity_raw = vel_bl.copy()
        
        if self.blocking_joints is not None:
            vel_bl[self.blocking_joints] = 0

        vel_bl_aa0 = vel_bl[0:3].copy()
        vel_bl_rh0 = vel_bl[3:7].copy()

        ### Accel Limit Velocitites ###
        # if not np.all(np.isnan(np.hstack((self.prev_vel_bl_aa, self.prev_vel_bl_rh)))):
        #     aa_output_accel = vel_bl_aa - self.prev_vel_bl_aa
        #     rh_output_accel = vel_bl_rh - self.prev_vel_bl_rh

        #     ### AA XY ###
        #     for i in np.arange(2):
        #         if aa_output_accel[i] > self.accel_lim_armassist:
        #             vel_bl_aa[i] = self.prev_vel_bl_aa[i] + self.accel_lim_armassist
        #         elif aa_output_accel[i] < -1*self.accel_lim_armassist:
        #             vel_bl_aa[i] = self.prev_vel_bl_aa[i] - self.accel_lim_armassist
            
        #     ### AA PSI ###
        #     if aa_output_accel[2] > self.accel_lim_psi:
        #         vel_bl_aa[2] = self.prev_vel_bl_aa[2] + self.accel_lim_psi
        #     elif aa_output_accel[2] < -1*self.accel_lim_psi:
        #         vel_bl_aa[2] = self.prev_vel_bl_aa[2] - self.accel_lim_psi

        #     ### RH All ###
        #     for i in np.arange(4):
        #         if rh_output_accel[i] > self.accel_lim_rehand:
        #             vel_bl_rh[i] = self.prev_vel_bl_rh[i] + self.accel_lim_rehand
        #         elif rh_output_accel[i] < -1*self.accel_lim_rehand:
        #             vel_bl_rh[i] = self.prev_vel_bl_rh[i] - self.accel_lim_rehand


        ### Add Attractor ###
        if self.safety_grid is not None:
            attractor_point_aa = self.safety_grid.attractor_point[:3]
            attractor_point_rh = self.safety_grid.attractor_point[3:]
            vel_bl_aa_pull = self.attractor_speed_const*(attractor_point_aa - current_state[:3])/0.05
            vel_bl_rh_pull = self.attractor_speed_const*(attractor_point_rh - current_state[3:])/0.05

            vel_bl_aa = vel_bl_aa0 + vel_bl_aa_pull.copy()
            vel_bl_rh = vel_bl_rh0 + vel_bl_rh_pull.copy()
        else:
            vel_bl_aa = vel_bl_aa0
            vel_bl_rh = vel_bl_rh0 

        ### LPF Filter Velocities ###
        for s, state in enumerate(['aa_vx', 'aa_vy', 'aa_vpsi']):
            vel_bl_aa[s] = self.command_lpfs[state](vel_bl_aa[s])
            if np.isnan(vel_bl_aa[s]):
                vel_bl_aa[s] = 0

        for s, state in enumerate(['rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']):
            vel_bl_rh[s] = self.command_lpfs[state](vel_bl_rh[s])
            if np.isnan(vel_bl_rh[s]):
                vel_bl_rh[s] = 0

        self.drive_velocity_sent_pre_safety = np.hstack(( vel_bl_aa.copy(), vel_bl_rh.copy()))

        #If the next position is outside of safety then damp velocity to only go to limit: 
        pos_pred = current_state + 0.05*np.hstack((vel_bl_aa, vel_bl_rh))
        pos_pred_aa = pos_pred[0:3]
        pos_pred_rh = pos_pred[3:7]
        both_feedback_str = ''

        if self.safety_grid is not None:

            if len(self.aa_plant.aa_xy_ix) > 0:
                x_tmp = self.safety_grid.is_valid_pos(pos_pred_aa[self.aa_plant.aa_xy_ix])
                if x_tmp == False:
                    
                    current_pos = current_state[self.aa_plant.aa_xy_ix]
                    pos_valid = attractor_point_aa[self.aa_plant.aa_xy_ix]

                    #d_to_valid, pos_valid = self.safety_grid.dist_to_valid_point(current_pos)
                    vel_bl_aa[self.aa_plant.aa_xy_ix] = self.attractor_speed*(pos_valid - current_pos)/0.05
                    pos_pred_aa[self.aa_plant.aa_xy_ix] = current_pos + 0.05*vel_bl_aa[self.aa_plant.aa_xy_ix]
                    
                    #print 'plant adjust: ', vel_bl_aa[self.aa_plant.aa_xy_ix], pos_pred_aa[self.aa_plant.aa_xy_ix]
                    xy_change = True


            # Make sure AA Psi within bounds: 
            if len(self.aa_plant.aa_psi_ix) > 0:

                mn, mx = self.safety_grid.get_minmax_psi(pos_pred_aa[self.aa_plant.aa_xy_ix])
                predx, predy= pos_pred_aa[[0, 1]]

                # Set psi velocity : 
                psi_ok = False
                if np.logical_and(pos_pred_aa[self.aa_plant.aa_psi_ix] >= mn, pos_pred_aa[self.aa_plant.aa_psi_ix] <= mx):
                    # Test if globally ok: 
                    global_ok = self.safety_grid.global_hull.hull3d.find_simplex(np.array([predx, predy, pos_pred_aa[2]])) >=0
                    if global_ok:
                        psi_ok = True

                if psi_ok == False:
                    # Move psi back to attractor pos: 
                    psi_neutral = attractor_point_aa[self.aa_plant.aa_psi_ix]
                    vel_bl_aa[self.aa_plant.aa_psi_ix] = self.attractor_speed*(psi_neutral-current_state[self.aa_plant.aa_psi_ix])/0.05

            # Make sure RH Prono within bounds (if SSM is only RH, use settings.starting_pos for AAPX, AAPY)
            if len(self.rh_plant.rh_pron_ix) > 0:

                mn, mx = self.safety_grid.get_minmax_prono(pos_pred_aa[self.aa_plant.aa_xy_ix])

                # Set prono velocity : 
                if np.logical_and(pos_pred_rh[self.rh_plant.rh_pron_ix] >= mn, pos_pred_rh[self.rh_plant.rh_pron_ix] <= mx):
                    pass
                
                else:
                    tmp_pos = pos_pred_rh[self.rh_plant.rh_pron_ix]
                    prono_neutral = attractor_point_rh[self.rh_plant.rh_pron_ix]
                    vel_bl_rh[self.rh_plant.rh_pron_ix] = self.attractor_speed*(prono_neutral-tmp_pos)/0.05

            # Assure RH fingers are within range: 
            if len(self.rh_plant.rh_pfings) > 0:
                for i, (ix, nm) in enumerate(self.rh_plant.rh_pfings):
                    mn, mx = self.safety_grid.get_rh_minmax(nm)
                    if np.logical_and(pos_pred_rh[ix] >= mn, pos_pred_rh[ix] <= mx):
                        pass
                    else:
                        tmp_ = pos_pred_rh[ix]
                        neutral = attractor_point_rh[ix]
                        vel_bl_rh[ix] = self.attractor_speed*(neutral - tmp_)/0.05

        
        # If in the rest state -- block the arm: 
        if self.task_state in ['rest', 'prep', 'baseline_check']:
            vel_bl_aa[:] = 0
            vel_bl_rh[:] = 0
            
        elif self.task_state == 'emg_rest':
            scaling = self.rest_emg_output
            
            if scaling <= 0.5:
                scaling = 0
            else:
                scaling = 0.5*scaling

            vel_bl_aa = scaling*vel_bl_aa
            vel_bl_rh = scaling*vel_bl_rh

        max_vel_xy = 10.
        vel_bl_aa[vel_bl_aa>max_vel_xy] = max_vel_xy
        vel_bl_aa[vel_bl_aa<-1*max_vel_xy] = -1*max_vel_xy
        
        max_vel_ang = 2.
        if vel_bl_aa[2] > max_vel_ang:
            vel_bl_aa[2] = max_vel_ang
        elif vel_bl_aa[2] < -1*max_vel_ang:
            vel_bl_aa[2] = -1*max_vel_ang

        vel_bl_rh[vel_bl_rh>max_vel_ang] = max_vel_ang
        vel_bl_rh[vel_bl_rh<-1*max_vel_ang] = -1*max_vel_ang

        if self.blocking_joints is not None:
            for j in [0, 1, 2]:
                if j in self.blocking_joints:
                    vel_bl_aa[j] = 0
                    #print 'blocking vel_bl_aa: ', j
            for j in [3, 4, 5, 6]:
                if j in self.blocking_joints:
                    vel_bl_rh[j-3] = 0
                    #print 'blocking vel_bl_rh: ', j-3
                    
        self.both_feedback_str = both_feedback_str
        self.aa_plant.send_vel(vel_bl_aa)
        self.rh_plant.send_vel(vel_bl_rh)

        self.prev_vel_bl_aa = vel_bl_aa.copy()
        self.prev_vel_bl_rh = vel_bl_rh.copy()

        self.drive_velocity_sent = np.hstack(( vel_bl_aa.copy(), vel_bl_rh.copy()))
        decoder['q'] = self.get_pos()

class IsMorePlantUDP(BasePlantIsMore):
    '''Sends velocity commands and receives feedback over UDP. Can be used
    with either the real or simulated ArmAssist+ReHand.
    '''
    aa_plant_cls = ArmAssistPlantUDP
    rh_plant_cls = ReHandPlantUDP

    def write_feedback(self):
        self.aa_plant.feedback_str = self.both_feedback_str
        self.aa_plant.write_feedback()
        #self.rh_plant.write_feedback()

class IsMorePlantEMGControl(IsMorePlantUDP): # Plant used for the pure EMG control task

    def drive(self):

        
        vel_bl = self.drive_velocity_raw
        current_state = self.get_pos()
        self.pre_drive_state = current_state.copy()
       
        
        if self.blocking_joints is not None:
            vel_bl[self.blocking_joints] = 0

        vel_bl_aa0 = vel_bl[0:3].copy()
        vel_bl_rh0 = vel_bl[3:7].copy()

        ### Accel Limit Velocitites ###
        # if not np.all(np.isnan(np.hstack((self.prev_vel_bl_aa, self.prev_vel_bl_rh)))):
        #     aa_output_accel = vel_bl_aa - self.prev_vel_bl_aa
        #     rh_output_accel = vel_bl_rh - self.prev_vel_bl_rh

        #     ### AA XY ###
        #     for i in np.arange(2):
        #         if aa_output_accel[i] > self.accel_lim_armassist:
        #             vel_bl_aa[i] = self.prev_vel_bl_aa[i] + self.accel_lim_armassist
        #         elif aa_output_accel[i] < -1*self.accel_lim_armassist:
        #             vel_bl_aa[i] = self.prev_vel_bl_aa[i] - self.accel_lim_armassist
            
        #     ### AA PSI ###
        #     if aa_output_accel[2] > self.accel_lim_psi:
        #         vel_bl_aa[2] = self.prev_vel_bl_aa[2] + self.accel_lim_psi
        #     elif aa_output_accel[2] < -1*self.accel_lim_psi:
        #         vel_bl_aa[2] = self.prev_vel_bl_aa[2] - self.accel_lim_psi

        #     ### RH All ###
        #     for i in np.arange(4):
        #         if rh_output_accel[i] > self.accel_lim_rehand:
        #             vel_bl_rh[i] = self.prev_vel_bl_rh[i] + self.accel_lim_rehand
        #         elif rh_output_accel[i] < -1*self.accel_lim_rehand:
        #             vel_bl_rh[i] = self.prev_vel_bl_rh[i] - self.accel_lim_rehand


        ### Add Attractor ###
        attractor_point_aa = self.safety_grid.attractor_point[:3]
        attractor_point_rh = self.safety_grid.attractor_point[3:]
        vel_bl_aa_pull = self.attractor_speed_const*(attractor_point_aa - current_state[:3])/0.05
        vel_bl_rh_pull = self.attractor_speed_const*(attractor_point_rh - current_state[3:])/0.05

        vel_bl_aa = vel_bl_aa0 + vel_bl_aa_pull.copy()
        vel_bl_rh = vel_bl_rh0 + vel_bl_rh_pull.copy()

        ### LPF Filter Velocities ###
        for s, state in enumerate(['aa_vx', 'aa_vy', 'aa_vpsi']):
            vel_bl_aa[s] = self.command_lpfs[state](vel_bl_aa[s])
            if np.isnan(vel_bl_aa[s]):
                vel_bl_aa[s] = 0

        for s, state in enumerate(['rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']):
            vel_bl_rh[s] = self.command_lpfs[state](vel_bl_rh[s])
            if np.isnan(vel_bl_rh[s]):
                vel_bl_rh[s] = 0

        self.drive_velocity_sent_pre_safety = np.hstack(( vel_bl_aa.copy(), vel_bl_rh.copy()))

        #If the next position is outside of safety then damp velocity to only go to limit: 
        pos_pred = current_state + 0.05*np.hstack((vel_bl_aa, vel_bl_rh))
        pos_pred_aa = pos_pred[0:3]
        pos_pred_rh = pos_pred[3:7]
        both_feedback_str = ''

        if self.safety_grid is not None:

            if len(self.aa_plant.aa_xy_ix) > 0:
                x_tmp = self.safety_grid.is_valid_pos(pos_pred_aa[self.aa_plant.aa_xy_ix])
                if x_tmp == False:
                    
                    current_pos = current_state[self.aa_plant.aa_xy_ix]
                    pos_valid = attractor_point_aa[self.aa_plant.aa_xy_ix]

                    #d_to_valid, pos_valid = self.safety_grid.dist_to_valid_point(current_pos)
                    vel_bl_aa[self.aa_plant.aa_xy_ix] = self.attractor_speed*(pos_valid - current_pos)/0.05
                    pos_pred_aa[self.aa_plant.aa_xy_ix] = current_pos + 0.05*vel_bl_aa[self.aa_plant.aa_xy_ix]
                    
                    #print 'plant adjust: ', vel_bl_aa[self.aa_plant.aa_xy_ix], pos_pred_aa[self.aa_plant.aa_xy_ix]
                    xy_change = True


            # Make sure AA Psi within bounds: 
            if len(self.aa_plant.aa_psi_ix) > 0:

                mn, mx = self.safety_grid.get_minmax_psi(pos_pred_aa[self.aa_plant.aa_xy_ix])
                predx, predy= pos_pred_aa[[0, 1]]

                # Set psi velocity : 
                psi_ok = False
                if np.logical_and(pos_pred_aa[self.aa_plant.aa_psi_ix] >= mn, pos_pred_aa[self.aa_plant.aa_psi_ix] <= mx):
                    # Test if globally ok: 
                    global_ok = self.safety_grid.global_hull.hull3d.find_simplex(np.array([predx, predy, pos_pred_aa[2]])) >=0
                    if global_ok:
                        psi_ok = True

                if psi_ok == False:
                    # Move psi back to attractor pos: 
                    psi_neutral = attractor_point_aa[self.aa_plant.aa_psi_ix]
                    vel_bl_aa[self.aa_plant.aa_psi_ix] = self.attractor_speed*(psi_neutral-current_state[self.aa_plant.aa_psi_ix])/0.05

            # Make sure RH Prono within bounds (if SSM is only RH, use settings.starting_pos for AAPX, AAPY)
            if len(self.rh_plant.rh_pron_ix) > 0:

                mn, mx = self.safety_grid.get_minmax_prono(pos_pred_aa[self.aa_plant.aa_xy_ix])

                # Set prono velocity : 
                if np.logical_and(pos_pred_rh[self.rh_plant.rh_pron_ix] >= mn, pos_pred_rh[self.rh_plant.rh_pron_ix] <= mx):
                    pass
                
                else:
                    tmp_pos = pos_pred_rh[self.rh_plant.rh_pron_ix]
                    prono_neutral = attractor_point_rh[self.rh_plant.rh_pron_ix]
                    vel_bl_rh[self.rh_plant.rh_pron_ix] = self.attractor_speed*(prono_neutral-tmp_pos)/0.05

            # Assure RH fingers are within range: 
            if len(self.rh_plant.rh_pfings) > 0:
                for i, (ix, nm) in enumerate(self.rh_plant.rh_pfings):
                    mn, mx = self.safety_grid.get_rh_minmax(nm)
                    if np.logical_and(pos_pred_rh[ix] >= mn, pos_pred_rh[ix] <= mx):
                        pass
                    else:
                        tmp_ = pos_pred_rh[ix]
                        neutral = attractor_point_rh[ix]
                        vel_bl_rh[ix] = self.attractor_speed*(neutral - tmp_)/0.05

        # If in the rest state -- block the arm: 
        if self.task_state in ['rest', 'prep']:
            vel_bl_aa[:] = 0
            vel_bl_rh[:] = 0
            
        elif self.task_state == 'emg_rest':
            scaling = self.rest_emg_output
            
            if scaling <= 0.5:
                scaling = 0
            else:
                scaling = 0.5*scaling

            vel_bl_aa = scaling*vel_bl_aa
            vel_bl_rh = scaling*vel_bl_rh

        max_vel_xy = 10.
        vel_bl_aa[vel_bl_aa>max_vel_xy] = max_vel_xy
        vel_bl_aa[vel_bl_aa<-1*max_vel_xy] = -1*max_vel_xy
        
        max_vel_ang = 2.
        if vel_bl_aa[2] > max_vel_ang:
            vel_bl_aa[2] = max_vel_ang
        elif vel_bl_aa[2] < -1*max_vel_ang:
            vel_bl_aa[2] = -1*max_vel_ang

        vel_bl_rh[vel_bl_rh>max_vel_ang] = max_vel_ang
        vel_bl_rh[vel_bl_rh<-1*max_vel_ang] = -1*max_vel_ang

        if self.blocking_joints is not None:
            for j in [0, 1, 2]:
                if j in self.blocking_joints:
                    vel_bl_aa[j] = 0
                    #print 'blocking vel_bl_aa: ', j
            for j in [3, 4, 5, 6]:
                if j in self.blocking_joints:
                    vel_bl_rh[j-3] = 0
                    #print 'blocking vel_bl_rh: ', j-3
                    
        self.both_feedback_str = both_feedback_str
        self.aa_plant.send_vel(vel_bl_aa)
        self.rh_plant.send_vel(vel_bl_rh)

        self.prev_vel_bl_aa = vel_bl_aa.copy()
        self.prev_vel_bl_rh = vel_bl_rh.copy()

        self.drive_velocity_sent = np.hstack(( vel_bl_aa.copy(), vel_bl_rh.copy()))
        
class IsMorePlantHybridBMI(IsMorePlantUDP): # Plant used for the hybrid (EMG + brain) BMI task.

    def __init__(self, *args, **kwargs):
        self.drive_velocity_raw_brain = np.zeros((7,))
        self.emg_vel_raw_scaled = np.zeros((7,))
        super(IsMorePlantHybridBMI, self).__init__(*args, **kwargs)

    def drive(self, decoder):

        vel = decoder['qdot']
        vel_brain = vel.copy()
        vel_brain_aa = vel_brain[[0, 1, 2]]
        vel_brain_fingers = vel_brain[[3, 4, 5]]
        vel_brain_prono = vel_brain[[6]]
        
        self.drive_velocity_raw_brain = vel_brain.copy()

        # Use EMG scaled array to scale the output:
        vel_emg = self.emg_vel.copy()
        vel_emg_scaled = []

        for i in range(7):
            vel_emg_scaled.append(vel_emg[i]*self.scale_emg_pred_arr[i])
        vel_emg_scaled = np.hstack((vel_emg_scaled))
        self.emg_vel_raw_scaled = vel_emg_scaled.copy()
        vel_emg_aa = vel_emg_scaled[[0, 1, 2]]
        vel_emg_fingers = vel_emg_scaled[[3, 4, 5]]
        vel_emg_prono = vel_emg_scaled[[6]]

        vel_bl_aa = vel_emg_aa*self.emg_weight_aa + vel_brain_aa*(1-self.emg_weight_aa)
        vel_bl_fingers = vel_emg_fingers*self.emg_weight_fingers + vel_brain_fingers*(1-self.emg_weight_fingers)
        vel_bl_prono = vel_emg_prono*self.emg_weight_prono + vel_brain_prono*(1-self.emg_weight_prono)

        vel_bl = np.hstack((vel_bl_aa, vel_bl_fingers, vel_bl_prono))
        # Fuse velocities from EMG and neural decoders
        #vel_bl = vel_emg*self.emg_weight + vel_brain*(1-self.emg_weight)
        self.drive_velocity_raw = vel_bl.copy()

        vel_bl_fb_gain = []
        for i in range(7):
            vel_bl_fb_gain.append(vel_bl[i]*self.fb_vel_gain[i])
        vel_bl_fb_gain = np.hstack((vel_bl_fb_gain))
        self.drive_velocity_raw_fb_gain = vel_bl_fb_gain.copy()

        current_state = self.get_pos()
        self.pre_drive_state = current_state.copy()
        
        if self.blocking_joints is not None:
            print 'self.blocking_joints  --> ', self.blocking_joints
            vel_bl_fb_gain[self.blocking_joints] = 0

        vel_bl_aa0 = vel_bl_fb_gain[0:3].copy()
        vel_bl_rh0 = vel_bl_fb_gain[3:7].copy()

        ### Accel Limit Velocitites ###
        # if not np.all(np.isnan(np.hstack((self.prev_vel_bl_aa, self.prev_vel_bl_rh)))):
        #     aa_output_accel = vel_bl_aa - self.prev_vel_bl_aa
        #     rh_output_accel = vel_bl_rh - self.prev_vel_bl_rh

        #     ### AA XY ###
        #     for i in np.arange(2):
        #         if aa_output_accel[i] > self.accel_lim_armassist:
        #             vel_bl_aa[i] = self.prev_vel_bl_aa[i] + self.accel_lim_armassist
        #         elif aa_output_accel[i] < -1*self.accel_lim_armassist:
        #             vel_bl_aa[i] = self.prev_vel_bl_aa[i] - self.accel_lim_armassist
            
        #     ### AA PSI ###
        #     if aa_output_accel[2] > self.accel_lim_psi:
        #         vel_bl_aa[2] = self.prev_vel_bl_aa[2] + self.accel_lim_psi
        #     elif aa_output_accel[2] < -1*self.accel_lim_psi:
        #         vel_bl_aa[2] = self.prev_vel_bl_aa[2] - self.accel_lim_psi

        #     ### RH All ###
        #     for i in np.arange(4):
        #         if rh_output_accel[i] > self.accel_lim_rehand:
        #             vel_bl_rh[i] = self.prev_vel_bl_rh[i] + self.accel_lim_rehand
        #         elif rh_output_accel[i] < -1*self.accel_lim_rehand:
        #             vel_bl_rh[i] = self.prev_vel_bl_rh[i] - self.accel_lim_rehand


        ### Add Attractor ###
        attractor_point_aa = self.safety_grid.attractor_point[:3]
        attractor_point_rh = self.safety_grid.attractor_point[3:]
        vel_bl_aa_pull = self.attractor_speed_const*(attractor_point_aa - current_state[:3])/0.05
        vel_bl_rh_pull = self.attractor_speed_const*(attractor_point_rh - current_state[3:])/0.05

        vel_bl_aa = vel_bl_aa0 + vel_bl_aa_pull.copy()
        vel_bl_rh = vel_bl_rh0 + vel_bl_rh_pull.copy()

        ### LPF Filter Velocities ###
        for s, state in enumerate(['aa_vx', 'aa_vy', 'aa_vpsi']):
            vel_bl_aa[s] = self.command_lpfs[state](vel_bl_aa[s])
            if np.isnan(vel_bl_aa[s]):
                vel_bl_aa[s] = 0

        for s, state in enumerate(['rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']):
            vel_bl_rh[s] = self.command_lpfs[state](vel_bl_rh[s])
            if np.isnan(vel_bl_rh[s]):
                vel_bl_rh[s] = 0

        self.drive_velocity_sent_pre_safety = np.hstack(( vel_bl_aa.copy(), vel_bl_rh.copy()))

        #If the next position is outside of safety then damp velocity to only go to limit: 
        pos_pred = current_state + 0.05*np.hstack((vel_bl_aa, vel_bl_rh))
        pos_pred_aa = pos_pred[0:3]
        pos_pred_rh = pos_pred[3:7]
        both_feedback_str = ''

        if self.safety_grid is not None:
            if len(self.aa_plant.aa_xy_ix) > 0:
                x_tmp = self.safety_grid.is_valid_pos(pos_pred_aa[self.aa_plant.aa_xy_ix])
                if x_tmp == False:
                    print 'false position'
                    current_pos = current_state[self.aa_plant.aa_xy_ix]
                    pos_valid = attractor_point_aa[self.aa_plant.aa_xy_ix]

                    #d_to_valid, pos_valid = self.safety_grid.dist_to_valid_point(current_pos)
                    vel_bl_aa[self.aa_plant.aa_xy_ix] = self.attractor_speed*(pos_valid - current_pos)/0.05
                    pos_pred_aa[self.aa_plant.aa_xy_ix] = current_pos + 0.05*vel_bl_aa[self.aa_plant.aa_xy_ix]
                    
                    #print 'plant adjust: ', vel_bl_aa[self.aa_plant.aa_xy_ix], pos_pred_aa[self.aa_plant.aa_xy_ix]
                    xy_change = True


            # Make sure AA Psi within bounds: 
            if len(self.aa_plant.aa_psi_ix) > 0:

                mn, mx = self.safety_grid.get_minmax_psi(pos_pred_aa[self.aa_plant.aa_xy_ix])
                predx, predy= pos_pred_aa[[0, 1]]

                # Set psi velocity : 
                psi_ok = False
                if np.logical_and(pos_pred_aa[self.aa_plant.aa_psi_ix] >= mn, pos_pred_aa[self.aa_plant.aa_psi_ix] <= mx):
                    # Test if globally ok: 
                    #global_ok = self.safety_grid.global_hull.hull3d.find_simplex(np.array([predx, predy, pos_pred_aa[2]])) >=0
                    global_ok = True
                    if global_ok:
                        psi_ok = True

                if psi_ok == False:
                    # Move psi back to attractor pos: 
                    psi_neutral = attractor_point_aa[self.aa_plant.aa_psi_ix]
                    vel_bl_aa[self.aa_plant.aa_psi_ix] = self.attractor_speed*(psi_neutral-current_state[self.aa_plant.aa_psi_ix])/0.05

            # Make sure RH Prono within bounds (if SSM is only RH, use settings.starting_pos for AAPX, AAPY)
            if len(self.rh_plant.rh_pron_ix) > 0:

                mn, mx = self.safety_grid.get_minmax_prono(pos_pred_aa[self.aa_plant.aa_xy_ix])

                # Set prono velocity : 
                if np.logical_and(pos_pred_rh[self.rh_plant.rh_pron_ix] >= mn, pos_pred_rh[self.rh_plant.rh_pron_ix] <= mx):
                    pass
                
                else:
                    tmp_pos = pos_pred_rh[self.rh_plant.rh_pron_ix]
                    prono_neutral = attractor_point_rh[self.rh_plant.rh_pron_ix]
                    vel_bl_rh[self.rh_plant.rh_pron_ix] = self.attractor_speed*(prono_neutral-tmp_pos)/0.05

            # Assure RH fingers are within range: 
            if len(self.rh_plant.rh_pfings) > 0:
                for i, (ix, nm) in enumerate(self.rh_plant.rh_pfings):
                    mn, mx = self.safety_grid.get_rh_minmax(nm)
                    if np.logical_and(pos_pred_rh[ix] >= mn, pos_pred_rh[ix] <= mx):
                        pass
                    else:
                        tmp_ = pos_pred_rh[ix]
                        neutral = attractor_point_rh[ix]
                        vel_bl_rh[ix] = self.attractor_speed*(neutral - tmp_)/0.05
                        # print 'safely adjusting fingers! ', nm, 'min: ', mn, ' max: ', mx, ' pred: ', pos_pred_rh[ix]

        # If in the rest state -- block the arm: 
        if self.task_state in ['rest', 'prep', 'baseline_check']:
            vel_bl_aa[:] = 0
            vel_bl_rh[:] = 0
            
        elif self.task_state == 'emg_rest':
            scaling = self.rest_emg_output
            
            if scaling <= 0.5:
                scaling = 0
            else:
                scaling = 0.5*scaling

            vel_bl_aa = scaling*vel_bl_aa
            vel_bl_rh = scaling*vel_bl_rh

        elif self.task_state == 'rest_back':
            vel_bl_aa = vel_bl_aa_pull/self.attractor_speed_const*self.rest_back_attractor_speed
            vel_bl_rh = vel_bl_rh_pull/self.attractor_speed_const*self.rest_back_attractor_speed
        
        elif self.task_state in ['drive_to_start', 'drive_to_rest']:
            vel_bl_aa = self.back_to_target_speed*(self.drive_to_start_target[:3] - current_state[:3])/0.05
            vel_bl_rh = self.back_to_target_speed*(self.drive_to_start_target[3:] - current_state[3:])/0.05
 
        max_vel_xy = 10.
        vel_bl_aa[vel_bl_aa>max_vel_xy] = max_vel_xy
        vel_bl_aa[vel_bl_aa<-1*max_vel_xy] = -1*max_vel_xy
        
        max_vel_ang = 2.
        if vel_bl_aa[2] > max_vel_ang:
            vel_bl_aa[2] = max_vel_ang
        elif vel_bl_aa[2] < -1*max_vel_ang:
            vel_bl_aa[2] = -1*max_vel_ang

        vel_bl_rh[vel_bl_rh>max_vel_ang] = max_vel_ang
        vel_bl_rh[vel_bl_rh<-1*max_vel_ang] = -1*max_vel_ang
        if self.blocking_joints is not None:
            for j in [0, 1, 2]:
                if j in self.blocking_joints:
                    vel_bl_aa[j] = 0
                    #print 'blocking vel_bl_aa: ', j
            for j in [3, 4, 5, 6]:
                if j in self.blocking_joints:
                    vel_bl_rh[j-3] = 0
                    #print 'blocking vel_bl_rh: ', j-3
                    
        self.both_feedback_str = both_feedback_str
        self.aa_plant.send_vel(vel_bl_aa)
        self.rh_plant.send_vel(vel_bl_rh)

        self.prev_vel_bl_aa = vel_bl_aa.copy()
        self.prev_vel_bl_rh = vel_bl_rh.copy()

        self.drive_velocity_sent = np.hstack(( vel_bl_aa.copy(), vel_bl_rh.copy()))
        decoder['q'] = self.get_pos()

class IsMorePlantHybridBMISoftSafety(IsMorePlantHybridBMI):
    
    def drive(self, decoder):

        vel = decoder['qdot']
        vel_brain = vel.copy()
        vel_brain_aa = vel_brain[[0, 1, 2]]
        vel_brain_fingers = vel_brain[[3, 4, 5]]
        vel_brain_prono = vel_brain[[6]]
        
        self.drive_velocity_raw_brain = vel_brain.copy()

        # Use EMG scaled array to scale the output:
        vel_emg = self.emg_vel.copy()
        vel_emg_scaled = []

        for i in range(7):
            vel_emg_scaled.append(vel_emg[i]*self.scale_emg_pred_arr[i])
        vel_emg_scaled = np.hstack((vel_emg_scaled))
        self.emg_vel_raw_scaled = vel_emg_scaled.copy()
        vel_emg_aa = vel_emg_scaled[[0, 1, 2]]
        vel_emg_fingers = vel_emg_scaled[[3, 4, 5]]
        vel_emg_prono = vel_emg_scaled[[6]]

        vel_bl_aa = vel_emg_aa*self.emg_weight_aa + vel_brain_aa*(1-self.emg_weight_aa)
        vel_bl_fingers = vel_emg_fingers*self.emg_weight_fingers + vel_brain_fingers*(1-self.emg_weight_fingers)
        vel_bl_prono = vel_emg_prono*self.emg_weight_prono + vel_brain_prono*(1-self.emg_weight_prono)

        vel_bl = np.hstack((vel_bl_aa, vel_bl_fingers, vel_bl_prono))

        # Fuse velocities from EMG and neural decoders
        #vel_bl = vel_emg*self.emg_weight + vel_brain*(1-self.emg_weight)
        self.drive_velocity_raw = vel_bl.copy()

        vel_bl_fb_gain = []
        for i in range(7):
            vel_bl_fb_gain.append(vel_bl[i]*self.fb_vel_gain[i])
        vel_bl_fb_gain = np.hstack((vel_bl_fb_gain))
        self.drive_velocity_raw_fb_gain = vel_bl_fb_gain.copy()

        current_state = self.get_pos()
        self.pre_drive_state = current_state.copy()
        
        if self.blocking_joints is not None:
            vel_bl_fb_gain[self.blocking_joints] = 0

        vel_bl_aa0 = vel_bl_fb_gain[0:3].copy()
        vel_bl_rh0 = vel_bl_fb_gain[3:7].copy()

        ### Accel Limit Velocitites ###
        # if not np.all(np.isnan(np.hstack((self.prev_vel_bl_aa, self.prev_vel_bl_rh)))):
        #     aa_output_accel = vel_bl_aa - self.prev_vel_bl_aa
        #     rh_output_accel = vel_bl_rh - self.prev_vel_bl_rh

        #     ### AA XY ###
        #     for i in np.arange(2):
        #         if aa_output_accel[i] > self.accel_lim_armassist:
        #             vel_bl_aa[i] = self.prev_vel_bl_aa[i] + self.accel_lim_armassist
        #         elif aa_output_accel[i] < -1*self.accel_lim_armassist:
        #             vel_bl_aa[i] = self.prev_vel_bl_aa[i] - self.accel_lim_armassist
            
        #     ### AA PSI ###
        #     if aa_output_accel[2] > self.accel_lim_psi:
        #         vel_bl_aa[2] = self.prev_vel_bl_aa[2] + self.accel_lim_psi
        #     elif aa_output_accel[2] < -1*self.accel_lim_psi:
        #         vel_bl_aa[2] = self.prev_vel_bl_aa[2] - self.accel_lim_psi

        #     ### RH All ###
        #     for i in np.arange(4):
        #         if rh_output_accel[i] > self.accel_lim_rehand:
        #             vel_bl_rh[i] = self.prev_vel_bl_rh[i] + self.accel_lim_rehand
        #         elif rh_output_accel[i] < -1*self.accel_lim_rehand:
        #             vel_bl_rh[i] = self.prev_vel_bl_rh[i] - self.accel_lim_rehand


        ### Add Attractor ###
        attractor_point_aa = self.safety_grid.attractor_point[:3]
        attractor_point_rh = self.safety_grid.attractor_point[3:]
        vel_bl_aa_pull = self.attractor_speed_const*(attractor_point_aa - current_state[:3])/0.05
        vel_bl_rh_pull = self.attractor_speed_const*(attractor_point_rh - current_state[3:])/0.05

        vel_bl_aa = vel_bl_aa0 + vel_bl_aa_pull.copy()
        vel_bl_rh = vel_bl_rh0 + vel_bl_rh_pull.copy()

        ### LPF Filter Velocities ###
        for s, state in enumerate(['aa_vx', 'aa_vy', 'aa_vpsi']):
            vel_bl_aa[s] = self.command_lpfs[state](vel_bl_aa[s])
            if np.isnan(vel_bl_aa[s]):
                vel_bl_aa[s] = 0

        for s, state in enumerate(['rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']):
            vel_bl_rh[s] = self.command_lpfs[state](vel_bl_rh[s])
            if np.isnan(vel_bl_rh[s]):
                vel_bl_rh[s] = 0

        self.drive_velocity_sent_pre_safety = np.hstack(( vel_bl_aa.copy(), vel_bl_rh.copy()))

        #If the next position is outside of safety then damp velocity to only go to limit: 
        pos_pred = current_state + 0.05*np.hstack((vel_bl_aa, vel_bl_rh))
        pos_pred_aa = pos_pred[0:3]
        pos_pred_rh = pos_pred[3:7]
        both_feedback_str = ''

        if self.safety_grid is not None:
            if len(self.aa_plant.aa_xy_ix) > 0:
                x_tmp = self.safety_grid.is_valid_pos(pos_pred_aa[self.aa_plant.aa_xy_ix])
                if x_tmp == False:
                    # Find the closest point on the boundary of the safety grid and set velocity in same
                    # direction, but at 90% of way to get to the edge of the safety grid: 
                    current_pos = current_state[self.aa_plant.aa_xy_ix]
                    
                    ### loop through percentages of velocity and check validity of point:
                    valid_scale = False
                    scale = 1.0
                    while valid_scale is False:
                        scale -= 0.05
                        pos_pred_xy  = current_pos + 0.05*(vel_bl_aa[self.aa_plant.aa_xy_ix]*scale)
                        valid_scale = self.safety_grid.is_valid_pos(pos_pred_xy)
                        if scale < -1.0:
                            scale = 0.0
                            break

                    #d_to_valid, pos_valid = self.safety_grid.dist_to_valid_point(current_pos)
                    vel_bl_aa[self.aa_plant.aa_xy_ix] = vel_bl_aa[self.aa_plant.aa_xy_ix]*scale
                    pos_pred_aa[self.aa_plant.aa_xy_ix] = current_pos + 0.05*vel_bl_aa[self.aa_plant.aa_xy_ix]
                    
                    #print 'plant adjust: ', vel_bl_aa[self.aa_plant.aa_xy_ix], pos_pred_aa[self.aa_plant.aa_xy_ix]
                    xy_change = True


            # Make sure AA Psi within bounds: 
            if len(self.aa_plant.aa_psi_ix) > 0:

                mn, mx = self.safety_grid.get_minmax_psi(pos_pred_aa[self.aa_plant.aa_xy_ix])
                predx, predy= pos_pred_aa[[0, 1]]

                # Set psi velocity : 
                psi_ok = False
                if np.logical_and(pos_pred_aa[self.aa_plant.aa_psi_ix] >= mn, pos_pred_aa[self.aa_plant.aa_psi_ix] <= mx):
                    # Test if globally ok: 
                    #global_ok = self.safety_grid.global_hull.hull3d.find_simplex(np.array([predx, predy, pos_pred_aa[2]])) >=0
                    global_ok = True
                    if global_ok:
                        psi_ok = True

                if psi_ok == False:
                    valid_scale_psi = False
                    scale = 1.0
                    while valid_scale_psi is False:
                        scale -= 0.05
                        psi_pred = current_state[self.aa_plant.aa_psi_ix] + 0.05*(scale*vel_bl_aa[self.aa_plant.aa_psi_ix])

                        if np.logical_and(psi_pred >= mn, psi_pred <= mx):
                            valid_scale_psi = True
                        if scale < -1.0:
                            scale = 0.0
                            break


                    vel_bl_aa[self.aa_plant.aa_psi_ix] = scale*vel_bl_aa[self.aa_plant.aa_psi_ix]

            # Make sure RH Prono within bounds (if SSM is only RH, use settings.starting_pos for AAPX, AAPY)
            if len(self.rh_plant.rh_pron_ix) > 0:

                mn, mx = self.safety_grid.get_minmax_prono(pos_pred_aa[self.aa_plant.aa_xy_ix])

                # Set prono velocity : 
                if np.logical_and(pos_pred_rh[self.rh_plant.rh_pron_ix] >= mn, pos_pred_rh[self.rh_plant.rh_pron_ix] <= mx):
                    pass
                
                else:
                    valid_scale_prono = False
                    scale = 1.0
                    while valid_scale_prono is False:
                        scale -= 0.05
                        pron_pred = pos_pred_rh[self.rh_plant.rh_pron_ix] + 0.05*(scale*vel_bl_rh[self.rh_plant.rh_pron_ix])
                        if np.logical_and(pron_pred >= mn, pron_pred <= mx):
                            valid_scale_prono = True
                        if scale < -1.0:
                            scale = 0.
                            break

                    vel_bl_rh[self.rh_plant.rh_pron_ix] = scale*vel_bl_rh[self.rh_plant.rh_pron_ix]


            # Assure RH fingers are within range: 
            if len(self.rh_plant.rh_pfings) > 0:
                for i, (ix, nm) in enumerate(self.rh_plant.rh_pfings):
                    mn, mx = self.safety_grid.get_rh_minmax(nm)
                    if np.logical_and(pos_pred_rh[ix] >= mn, pos_pred_rh[ix] <= mx):
                        pass
                    else:
                        finger_scale = False
                        scale = 1.0
                        while finger_scale is False:
                            scale -= 0.05
                            fing_pred = pos_pred_rh[ix] + 0.05*(scale*vel_bl_rh[ix])
                            if np.logical_and(fing_pred >= mn, fing_pred<= mx):
                                finger_scale = True
                            if scale < -1.0:
                                scale = 0.0
                                break
                        vel_bl_rh[ix] = scale*vel_bl_rh[ix]
                        
        # If in the rest state -- block the arm: 
        if self.task_state in ['rest', 'prep', 'baseline_check', 'wait']:
            vel_bl_aa[:] = 0
            vel_bl_rh[:] = 0
            
        elif self.task_state == 'emg_rest':
            scaling = self.rest_emg_output
            
            if scaling <= 0.5:
                scaling = 0
            else:
                scaling = 0.5*scaling

            vel_bl_aa = scaling*vel_bl_aa
            vel_bl_rh = scaling*vel_bl_rh

        elif self.task_state == 'rest_back':
            vel_bl_aa = vel_bl_aa_pull/self.attractor_speed_const*self.rest_back_attractor_speed
            vel_bl_rh = vel_bl_rh_pull/self.attractor_speed_const*self.rest_back_attractor_speed
        
        elif self.task_state in ['drive_to_start', 'drive_to_rest']:
            vel_bl_aa = self.back_to_target_speed*(self.drive_to_start_target[:3] - current_state[:3])/0.05
            vel_bl_rh = self.back_to_target_speed*(self.drive_to_start_target[3:] - current_state[3:])/0.05
 
        max_vel_xy = 10.
        vel_bl_aa[vel_bl_aa>max_vel_xy] = max_vel_xy
        vel_bl_aa[vel_bl_aa<-1*max_vel_xy] = -1*max_vel_xy
        
        max_vel_ang = 2.
        if vel_bl_aa[2] > max_vel_ang:
            vel_bl_aa[2] = max_vel_ang
        elif vel_bl_aa[2] < -1*max_vel_ang:
            vel_bl_aa[2] = -1*max_vel_ang

        vel_bl_rh[vel_bl_rh>max_vel_ang] = max_vel_ang
        vel_bl_rh[vel_bl_rh<-1*max_vel_ang] = -1*max_vel_ang
        if self.blocking_joints is not None:
            for j in [0, 1, 2]:
                if j in self.blocking_joints:
                    vel_bl_aa[j] = 0
                    #print 'blocking vel_bl_aa: ', j
            for j in [3, 4, 5, 6]:
                if j in self.blocking_joints:
                    vel_bl_rh[j-3] = 0
                    #print 'blocking vel_bl_rh: ', j-3
                    
        self.both_feedback_str = both_feedback_str
        self.aa_plant.send_vel(vel_bl_aa)
        self.rh_plant.send_vel(vel_bl_rh)

        self.prev_vel_bl_aa = vel_bl_aa.copy()
        self.prev_vel_bl_rh = vel_bl_rh.copy()

        self.drive_velocity_sent = np.hstack(( vel_bl_aa.copy(), vel_bl_rh.copy()))
        decoder['q'] = self.get_pos()

UDP_PLANT_CLS_DICT = {
    'ArmAssist': ArmAssistPlantUDP,
    'ReHand':    ReHandPlantUDP,
    'IsMore':    IsMorePlantUDP,
    'IsMoreEMGControl': IsMorePlantEMGControl,
    'IsMoreHybridControl': IsMorePlantHybridBMI,
    'IsMorePlantHybridBMISoftSafety': IsMorePlantHybridBMISoftSafety,
    'DummyPlant':   DummyPlantUDP,
}



###########################
##### Deprecated code #####
###########################
class BasePlant(object):
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Implement in subclasses!')

    def init(self):
        raise NotImplementedError('Implement in subclasses!')

    def start(self):
        raise NotImplementedError('Implement in subclasses!')

    def stop(self):
        raise NotImplementedError('Implement in subclasses!')

    def last_data_ts_arrival(self):
        raise NotImplementedError('Implement in subclasses!')

    def send_vel(self, vel):
        raise NotImplementedError('Implement in subclasses!')

    def get_pos(self):
        raise NotImplementedError('Implement in subclasses!')

    def get_vel(self):
        raise NotImplementedError('Implement in subclasses!')

    def enable(self):
        '''Disable the device's motor drivers.'''
        raise NotImplementedError('Implement in subclasses!')

    def disable(self):
        '''Disable the device's motor drivers.'''
        raise NotImplementedError('Implement in subclasses!')

    def enable_watchdog(self, timeout_ms):
        raise NotImplementedError('Implement in subclasses!')

    def get_intrinsic_coordinates(self):
        return self.get_pos()
