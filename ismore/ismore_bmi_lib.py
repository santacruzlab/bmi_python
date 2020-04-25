'''
BMI code specific to the ISMORE project
'''
import numpy as np 
from riglib.stereo_opengl import ik
from riglib.bmi import feedback_controllers
from riglib.bmi.state_space_models import State, StateSpace, offset_state, _gen_A
from riglib.bmi.assist import Assister
from riglib.bmi.clda import Learner, FeedbackControllerLearner
import pickle

from utils.angle_utils import *
from utils.constants import *


######################
## State-space models 
######################
class StateSpaceArmAssist(StateSpace):
    def __init__(self):
        max_vel = 2  # cm/s
        max_ang_vel = 7.5 * deg_to_rad
        super(StateSpaceArmAssist, self).__init__(
            State('aa_px',   stochastic=False, drives_obs=False, order=0, min_val=0., max_val=42.),
            State('aa_py',   stochastic=False, drives_obs=False, order=0, min_val=0., max_val=30.),
            State('aa_ppsi', stochastic=False, drives_obs=False, order=0),
            State('aa_vx',   stochastic=True,  drives_obs=True,  order=1, min_val=-max_vel, max_val=max_vel),
            State('aa_vy',   stochastic=True,  drives_obs=True,  order=1, min_val=-max_vel, max_val=max_vel),
            State('aa_vpsi', stochastic=True,  drives_obs=True,  order=1, min_val=-max_ang_vel, max_val=max_ang_vel),
            offset_state
        )

    def get_ssm_matrices(self, update_rate=0.1):
        # for now, just use a fixed vel_decay for A and vel_var for W
        #   regardless of the value of update_rate
        vel_decay = 1.
        A = _gen_A(1, update_rate, 0, vel_decay, 1, ndim=3)
        A[5,5] = .8
        vel_var = 7.
        W = _gen_A(0,           0, 0,   vel_var, 0, ndim=3)  # there is no separate _gen_W function
        W[5,5] = .1

        # Control input matrix for SSM for control inputs
        I = np.mat(np.eye(3))
        B = np.vstack([0*I, update_rate*1000 * I, np.zeros([1, 3])])
        return A, B, W


class StateSpaceReHand(StateSpace):
    def __init__(self):
        max_ang_vel = 7.5 * deg_to_rad
        super(StateSpaceReHand, self).__init__(
            State('rh_pthumb', stochastic=False, drives_obs=False, order=0),
            State('rh_pindex', stochastic=False, drives_obs=False, order=0),
            State('rh_pfing3', stochastic=False, drives_obs=False, order=0),
            State('rh_pprono', stochastic=False, drives_obs=False, order=0),
            State('rh_vthumb', stochastic=True,  drives_obs=True,  order=1, min_val=-max_ang_vel, max_val=max_ang_vel),
            State('rh_vindex', stochastic=True,  drives_obs=True,  order=1, min_val=-max_ang_vel, max_val=max_ang_vel),
            State('rh_vfing3', stochastic=True,  drives_obs=True,  order=1, min_val=-max_ang_vel, max_val=max_ang_vel),
            State('rh_vprono', stochastic=True,  drives_obs=True,  order=1, min_val=-max_ang_vel, max_val=max_ang_vel),
            offset_state
        )

    def get_ssm_matrices(self, update_rate=0.1):
        # for now, just use a fixed vel_decay for A and vel_var for W
        #   regardless of the value of update_rate
        vel_decay = .8
        A = _gen_A(1, update_rate, 0, vel_decay, 1, ndim=4) 
        vel_var = .1
        W = _gen_A(0,           0, 0,   vel_var, 0, ndim=4)  # there is no separate _gen_W function

        # Control input matrix for SSM for control inputs
        I = np.mat(np.eye(4))
        B = np.vstack([0*I, update_rate*1000 * I, np.zeros([1, 4])])
        return A, B, W


class StateSpaceIsMore(StateSpace):
    def __init__(self):
        max_vel = 2  # cm/s
        max_ang_vel = 7.5 * deg_to_rad
        super(StateSpaceIsMore, self).__init__(
            # position states
            State('aa_px',     stochastic=False, drives_obs=False, order=0, min_val=0., max_val=42.),
            State('aa_py',     stochastic=False, drives_obs=False, order=0, min_val=0., max_val=30.),
            State('aa_ppsi',   stochastic=False, drives_obs=False, order=0),
            State('rh_pthumb', stochastic=False, drives_obs=False, order=0),
            State('rh_pindex', stochastic=False, drives_obs=False, order=0),
            State('rh_pfing3', stochastic=False, drives_obs=False, order=0),
            State('rh_pprono', stochastic=False, drives_obs=False, order=0),

            # velocity states
            State('aa_vx',     stochastic=True,  drives_obs=True,  order=1, min_val=-max_vel, max_val=max_vel),
            State('aa_vy',     stochastic=True,  drives_obs=True,  order=1, min_val=-max_vel, max_val=max_vel),
            State('aa_vpsi',   stochastic=True,  drives_obs=True,  order=1, min_val=-max_ang_vel, max_val=max_ang_vel),
            State('rh_vthumb', stochastic=True,  drives_obs=True,  order=1, min_val=-max_ang_vel, max_val=max_ang_vel),
            State('rh_vindex', stochastic=True,  drives_obs=True,  order=1, min_val=-max_ang_vel, max_val=max_ang_vel),
            State('rh_vfing3', stochastic=True,  drives_obs=True,  order=1, min_val=-max_ang_vel, max_val=max_ang_vel),
            State('rh_vprono', stochastic=True,  drives_obs=True,  order=1, min_val=-max_ang_vel, max_val=max_ang_vel),

            # offset state
            offset_state
        )

    def get_ssm_matrices(self, update_rate=0.1):
        # for now, just use a fixed vel_decay for A and vel_var for W
        #   regardless of the value of update_rate
        vel_decay = .8
        A = _gen_A(1, update_rate, 0, vel_decay, 1, ndim=7) 
        A[7,7] = 1.
        A[8, 8] =1.

        #vel_var = .1
        vel_var = 0.005
        W = _gen_A(0,           0, 0,   vel_var, 0, ndim=7)  # there is no separate _gen_W function
        
        # W[7,7] = 7
        # W[8, 8] = 7
        W[7, 7] = 0.5
        W[8, 8] = 0.5
        W[9, 9] = 0.01

        # Control input matrix for SSM for control inputs
        I = np.mat(np.eye(7))
        B = np.vstack([0*I, update_rate*1000 * I, np.zeros([1, 7])])
        return A, B, W


class StateSpaceDummy(StateSpace):
    def __init__(self):
        max_vel = 2  # cm/s
        max_ang_vel = 7.5 * deg_to_rad
        super(StateSpaceDummy, self).__init__(
            State('aa_px',   stochastic=False, drives_obs=False, order=0, min_val=0., max_val=42.),
            State('aa_py',   stochastic=False, drives_obs=False, order=0, min_val=0., max_val=30.),
            State('aa_ppsi', stochastic=False, drives_obs=False, order=0),
            State('aa_vx',   stochastic=True,  drives_obs=True,  order=1, min_val=-max_vel, max_val=max_vel),
            State('aa_vy',   stochastic=True,  drives_obs=True,  order=1, min_val=-max_vel, max_val=max_vel),
            State('aa_vpsi', stochastic=True,  drives_obs=True,  order=1, min_val=-max_ang_vel, max_val=max_ang_vel),
            offset_state
        )

    def get_ssm_matrices(self, update_rate=0.1):
        # for now, just use a fixed vel_decay for A and vel_var for W
        #   regardless of the value of update_rate
        vel_decay = 1.
        A = _gen_A(1, update_rate, 0, vel_decay, 1, ndim=3)
        A[5,5] = .8
        vel_var = 7.
        W = _gen_A(0,           0, 0,   vel_var, 0, ndim=3)  # there is no separate _gen_W function
        W[5,5] = .1

        # Control input matrix for SSM for control inputs
        I = np.mat(np.eye(3))
        B = np.vstack([0*I, update_rate*1000 * I, np.zeros([1, 3])])
        return A, B, W


#######################################################################
##### Assisters and Learners based on formal feedback controllers #####
#######################################################################
from riglib.bmi.feedback_controllers import LQRController
from riglib.bmi.assist import FeedbackControllerAssist, FeedbackControllerAssist_StateSpecAssistLevels

class LQRController_accel_limit_armassist(LQRController):
    def __init__(self, *args, **kwargs):
        self.prev_assister_output = np.nan
        self.accel_lim_armassist = .1 
        self.accel_lim_psi = .02
        print 'LQRController'        
        super(LQRController_accel_limit_armassist, self).__init__(*args, **kwargs)

    def calc_next_state(self, current_state, target_state, mode=None):
        current_state = np.mat(current_state).reshape(-1,1)
        target_state = np.mat(target_state).reshape(-1,1)
        assister_output = self.A * current_state + self.B * self.F * (target_state - current_state)
        

        if np.sum(assister_output[3:6]) != 0:
            if np.any(np.isnan(self.prev_assister_output)):
                self.prev_assister_output = np.zeros_like(assister_output)

            assister_output_accel = assister_output - self.prev_assister_output

            for i in np.arange(3,5):
                if assister_output_accel[i, 0] > self.accel_lim_armassist:
                    assister_output[i, 0] = self.prev_assister_output[i, 0] + self.accel_lim_armassist
                elif assister_output_accel[i, 0] < -1*self.accel_lim_armassist:
                    assister_output[i, 0] = self.prev_assister_output[i, 0] - self.accel_lim_armassist

            for i in [5]:
                if assister_output_accel[i, 0] > self.accel_lim_psi:
                    assister_output[i, 0] = self.prev_assister_output[i, 0] + self.accel_lim_psi
                elif assister_output_accel[i, 0] < -1*self.accel_lim_psi:
                    assister_output[i, 0] = self.prev_assister_output[i, 0] - self.accel_lim_psi


        self.prev_assister_output = assister_output
        return assister_output

class LQRController_accel_limit_rehand(LQRController):
    def __init__(self, *args, **kwargs):
        self.prev_assister_output = np.nan
        self.accel_lim = .02
        print 'LQRController'
        super(LQRController_accel_limit_rehand, self).__init__(*args, **kwargs)

    def calc_next_state(self, current_state, target_state, mode=None):
        current_state = np.mat(current_state).reshape(-1,1)
        target_state = np.mat(target_state).reshape(-1,1)
        assister_output = self.A * current_state + self.B * self.F * (target_state - current_state)
        if np.sum(assister_output[4:8]) != 0:
            if np.any(np.isnan(self.prev_assister_output)):
                self.prev_assister_output = np.zeros_like(assister_output)

            assister_output_accel = assister_output - self.prev_assister_output
           # print 'assister output accel', np.squeeze(assister_output_accel)
            for i in np.arange(4,8):
                if assister_output_accel[i, 0] > self.accel_lim:
                    assister_output[i, 0] = self.prev_assister_output[i, 0] + self.accel_lim
                elif assister_output_accel[i, 0] < -1*self.accel_lim:
                    assister_output[i, 0] = self.prev_assister_output[i, 0] - self.accel_lim
        self.prev_assister_output = assister_output
        return assister_output

class LQRController_accel_limit_ismore(LQRController):
    def __init__(self, *args, **kwargs):
        self.prev_assister_output = np.nan        
        self.accel_lim_armassist = .5#.1 
        self.accel_lim_psi = .5#.02
        self.accel_lim_rehand = .5#.02

        # # nerea -- acceleration limit removed  
        # self.accel_lim_armassist = np.inf
        # self.accel_lim_psi = np.inf
        # self.accel_lim_rehand = np.inf
        # print 'LQRController'
        super(LQRController_accel_limit_ismore, self).__init__(*args, **kwargs)

    def calc_next_state(self, current_state, target_state, mode=None):

        current_state = np.mat(current_state).reshape(-1,1)
        target_state = np.mat(target_state).reshape(-1,1)
        assister_output = self.B * self.F * (target_state - current_state)
        if np.sum(assister_output[7:14]) != 0:
            if np.any(np.isnan(self.prev_assister_output)):
                self.prev_assister_output = np.zeros_like(assister_output)
            #print 'assister_output', assister_output[7:14]
            assister_output_accel = assister_output - self.prev_assister_output
            #print 'assister output accel', np.squeeze(assister_output_accel)
            for i in np.arange(7,9):
                if assister_output_accel[i, 0] > self.accel_lim_armassist:
                    assister_output[i, 0] = self.prev_assister_output[i, 0] + self.accel_lim_armassist
                elif assister_output_accel[i, 0] < -1*self.accel_lim_armassist:
                    assister_output[i, 0] = self.prev_assister_output[i, 0] - self.accel_lim_armassist
            for i in [9]:
                if assister_output_accel[i, 0] > self.accel_lim_psi:
                    assister_output[i, 0] = self.prev_assister_output[i, 0] + self.accel_lim_psi
                elif assister_output_accel[i, 0] < -1*self.accel_lim_psi:
                    assister_output[i, 0] = self.prev_assister_output[i, 0] - self.accel_lim_psi
            for i in np.arange(10,14):
                #print "accel rh", assister_output_accel[i, 0]
                if assister_output_accel[i, 0] > self.accel_lim_rehand:
                    print "excedeed rh accel limit in DoF ", i
                    assister_output[i, 0] = self.prev_assister_output[i, 0] + self.accel_lim_rehand
                elif assister_output_accel[i, 0] < -1*self.accel_lim_rehand:
                    print "excedeed rh accel limit"
                    assister_output[i, 0] = self.prev_assister_output[i, 0] - self.accel_lim_rehand
        #print 'assister_output_accel after',  assister_output - self.prev_assister_output
        self.prev_assister_output = assister_output
        return assister_output

class LQRController_accel_limit_only_base_ismore(LQRController_accel_limit_ismore):
    def __init__(self, *args, **kwargs):
        super(LQRController_accel_limit_only_base_ismore, self).__init__(*args, **kwargs)
        self.prev_assister_output = np.nan        
        self.accel_lim_armassist = np.inf 
        self.accel_lim_psi = np.inf
        self.accel_lim_rehand = np.inf


class LQRController_ismore_w_rest(LQRController):
    
    def calc_next_state(self, current_state, target_state, mode=None):
        if mode in ['rest', 'emg_rest']:
            ts = current_state.copy()
        else:
            ts = target_state.copy()

        current_state = np.mat(current_state).reshape(-1,1)
        target_state = np.mat(ts).reshape(-1,1)
        ns = self.A * current_state + self.B * self.F * (target_state - current_state)
        return ns

ssm = StateSpaceArmAssist()
A, B, _ = ssm.get_ssm_matrices()
#Q = np.mat(np.diag([10., 10., 10., 5, 5, 5, 0]))
Q = np.mat(np.diag([1., 1., 1., 5, 5, 5, 0]))
R = 1e6 * np.mat(np.diag([1., 1., 1.]))
arm_assist_controller = LQRController(A, B, Q, R)

ssm = StateSpaceReHand()
A, B, _ = ssm.get_ssm_matrices()
Q = 0.1*np.mat(np.diag([1., 1., 1., 1., 10., 10., 10., 10., 0]))
R = 1e5 * np.mat(np.diag([1., 1., 1., 1.])) #for bmi: 1e7
rehand_controller = LQRController(A, B, Q, R)

ssm = StateSpaceIsMore()
A, B, _ = ssm.get_ssm_matrices()
Q = 0.1*np.mat(np.diag([10., 10., 10., 1., 1., 1., 1., 5, 5, 5, 10., 10., 10., 10., 0]))
R = 1e5 * np.mat(np.diag([1., 1., 1.,1., 1., 1., 1.])) #for bmi =1e6 ;  to make it faster, changed from 1e6 to 1e5
ismore_controller = LQRController(A, B, Q, R)

ismore_controller_w_rest = LQRController_ismore_w_rest(A, B, Q, R)

##################### For BMI SIMS #####################
ssm = StateSpaceArmAssist()
A, B, _ = ssm.get_ssm_matrices()
Q = np.mat(np.diag([10., 10., 1., 5, 5, .5, 0]))
#Q = np.mat(np.diag([1., 1., 1., 5, 5, 5, 0]))
R = 1e6 * np.mat(np.diag([1., 1., 1.]))
arm_assist_controller_bmi_sims = LQRController(A, B, Q, R)

ssm = StateSpaceReHand()
A, B, _ = ssm.get_ssm_matrices()
#Q = np.mat(np.diag([1., 1., 1., 1., .5, .5, .5, .5, 0]))
Q = np.mat(np.diag([1., 1., 1., 1., .5, .5, .5, .5, 0]))
R = 1e6 * np.mat(np.diag([1., 1., 1., 1.])) #for bmi: 1e7
rehand_controller_bmi_sims = LQRController(A, B, Q, R)

# ssm = StateSpaceIsMore()
# A, B, _ = ssm.get_ssm_matrices()
# Q = 0.1*np.mat(np.diag([10., 10., 10., 1., 1., 1., 1., 5, 5, 5, .5, .5, .5, .5, 0]))
# R = 1e6 * np.mat(np.diag(np.ones(7,))) #for bmi =1e6 ;  to make it faster, changed from 1e6 to 1e5
# ismore_controller = LQRController(A, B, Q, R)


class iBMIAssister(Assister):
    '''
    Summary: IBMI Assister modifies Assister output to bound velocities 
    '''
    def __init__(self, ssm, *args, **kwargs):
        super(iBMIAssister, self).__init__(*args, **kwargs)
        self.ssm = ssm
        self.min_motor_vel = 1e-3

        #states that drive obs: 
        self.vel_states = np.nonzero(ssm.state_order==1)[0]

    def __call__(self, *args, **kwargs):
        asst = super(iBMIAssister, self).__call__(*args, **kwargs)
        return self.bound_vel_states(asst)
    
    def bound_vel_states(self, asst):
        if 'Bu' in asst:
            asst_in = asst['Bu'].copy()
            asst_ky = 'Bu'
        elif 'x_assist' in asst:
            asst_in = asst['x_assist'].copy()
            asst_ky = 'x_assist'

        asst_in[self.vel_states, 0] = self._bnd(asst_in[self.vel_states, 0])
        asst[asst_ky] = asst_in
        return asst

    def _bnd(self, vals):
        val_bnd = []
        vals_arr = np.squeeze(np.array(vals))
        for iv, vl in enumerate(vals_arr):
            if vl < 0:
                vl_gain=-1
            else:
                vl_gain = 1

            vl_adj = vl_gain*np.max([np.abs(vl), self.min_motor_vel])
            val_bnd.append(vl_adj)
        return np.array(val_bnd)


###############################################
################ LFC Assisters #################
################################################


class ArmAssistLFCAssister(iBMIAssister, FeedbackControllerAssist):
    '''
    Assister for ArmAssist which uses an infinite-horizon LQR controller
    '''
    def __init__(self, *args, **kwargs):
        super(ArmAssistLFCAssister, self).__init__(StateSpaceArmAssist(), arm_assist_controller, style='mixing')

class ReHandLFCAssister(iBMIAssister, FeedbackControllerAssist):
    def __init__(self, *args, **kwargs):
        super(ReHandLFCAssister, self).__init__(StateSpaceReHand(), rehand_controller, style='mixing')        

#class IsMoreLFCAssister(Assister): #it was like this before, ask Suraj
class IsMoreLFCAssister(iBMIAssister, FeedbackControllerAssist): 
    def __init__(self, *args, **kwargs):
        super(IsMoreLFCAssister, self).__init__(StateSpaceIsMore(), ismore_controller, style='mixing')       


###############################################
####### Orthogonal Damping Assisters ##########
################################################

class OrthoAssist(iBMIAssister):
    def __init__(self, *args, **kwargs):
        super(OrthoAssist, self).__init__(*args, **kwargs)

    def calc_assisted_BMI_state(self, *args, **kwargs):
        assist_kw = super(OrthoAssist, self).calc_assisted_BMI_state(*args, **kwargs)
        assist_kw['ortho_damp_assist'] = True
        return assist_kw

class ArmAssistLFCOrthoDampAssister(ArmAssistLFCAssister, OrthoAssist):
    def __init__(self, *args, **kwargs):
        super(ArmAssistLFCOrthoDampAssister, self).__init__(*args, **kwargs)

class ReHandLFCOrthoDampAssister(ReHandLFCAssister, OrthoAssist):
    def __init__(self, *args, **kwargs):
        super(ReHandLFCOrthoDampAssister, self).__init__(*args, **kwargs)

class IsMoreLFCOrthoDampAssister(IsMoreLFCAssister, OrthoAssist):
    def __init__(self, *args, **kwargs):
        super(IsMoreLFCOrthoDampAssister, self).__init__(*args, **kwargs)

class IsMoreLFCOrthoDamp_diff_assist(IsMoreLFCAssister, OrthoAssist, FeedbackControllerAssist_StateSpecAssistLevels):
    def __init__(self, *args, **kwargs):
        super(IsMoreLFCOrthoDamp_diff_assist, self).__init__(*args, **kwargs)

###############################################
################ OFC Assisters #################
################################################


class ArmAssistOFCLearner(FeedbackControllerLearner):
    def __init__(self, batch_size, *args, **kwargs):
        super(ArmAssistOFCLearner, self).__init__(batch_size, arm_assist_controller)


class ReHandOFCLearner(FeedbackControllerLearner):
    def __init__(self, batch_size, *args, **kwargs):
        super(ReHandOFCLearner, self).__init__(batch_size, rehand_controller)


class IsMoreOFCLearner(FeedbackControllerLearner):
    def __init__(self, batch_size, *args, **kwargs):
        super(IsMoreOFCLearner, self).__init__(batch_size, ismore_controller)

class IsMoreOFCLearning_w_Rest(FeedbackControllerLearner):
    def __init__(self, batch_size, *args, **kwargs):
        super(IsMoreOFCLearning_w_Rest, self).__init__(batch_size, ismore_controller_w_rest)

###############################################################
##### Assisters and Learners based on more ad-hoc methods #####
##############################################################3

class ArmAssistAssister(Assister):
    '''Simple assister that moves ArmAssist position towards the xy target 
    at a constant speed, and towards the psi target at a constant angular 
    speed. When within a certain xy distance or angular distance of the 
    target, these speeds are reduced.'''

    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.call_rate  = kwargs.pop('call_rate',  10)             # secs
        self.xy_speed   = kwargs.pop('xy_speed',   2.)             # cm/s
        self.xy_cutoff  = kwargs.pop('xy_cutoff',  2.)             # cm
        self.psi_speed  = kwargs.pop('psi_speed',  8.*deg_to_rad)  # rad/s
        self.psi_cutoff = kwargs.pop('psi_cutoff', 5.*deg_to_rad)  # rad

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if assist_level > 0:
            xy_pos  = np.array(current_state[0:2, 0]).ravel()
            psi_pos = np.array(current_state[  2, 0]).ravel()
            target_xy_pos  = np.array(target_state[0:2, 0]).ravel()
            target_psi_pos = np.array(target_state[  2, 0]).ravel()
            assist_xy_pos, assist_xy_vel = self._xy_assist(xy_pos, target_xy_pos)
            assist_psi_pos, assist_psi_vel = self._psi_assist(psi_pos, target_psi_pos)

            # if mode == 'hold':
            #     print 'task state is "hold", setting assist vels to 0'
            #     assist_xy_vel[:] = 0.
            #     assist_psi_vel[:] = 0.

            Bu = assist_level * np.hstack([assist_xy_pos, 
                                           assist_psi_pos,
                                           assist_xy_vel,
                                           assist_psi_vel,
                                           1])
            Bu = np.mat(Bu.reshape(-1, 1))

            assist_weight = assist_level
        else:
            Bu = None
            assist_weight = 0.
        return dict(Bu=Bu, assist_level=assist_weight)
        # return Bu, assist_weight

    def _xy_assist(self, xy_pos, target_xy_pos):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        diff_vec = target_xy_pos - xy_pos
        dist_to_target = np.linalg.norm(diff_vec)
        dir_to_target = diff_vec / (np.spacing(1) + dist_to_target)

        # if xy distance is below xy_cutoff (e.g., target radius), use smaller speed
        if dist_to_target < self.xy_cutoff:
            frac = 0.5 * dist_to_target / self.xy_cutoff
            assist_xy_vel = frac * self.xy_speed * dir_to_target
        else:
            assist_xy_vel = self.xy_speed * dir_to_target

        assist_xy_pos = xy_pos + assist_xy_vel/self.call_rate

        return assist_xy_pos, assist_xy_vel

    def _psi_assist(self, psi_pos, target_psi_pos):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        psi_diff = angle_subtract(target_psi_pos, psi_pos)

        # if angular distance is below psi_cutoff, use smaller speed
        if abs(psi_diff) < self.psi_cutoff:
            assist_psi_vel = 0.5 * (psi_diff / self.psi_cutoff) * self.psi_speed
        else:
            assist_psi_vel = np.sign(psi_diff) * self.psi_speed

        assist_psi_pos = psi_pos + assist_psi_vel/self.call_rate

        return assist_psi_pos, assist_psi_vel


class ReHandAssister(Assister):
    '''Simple assister that moves ReHand joint angles towards their angular
    targets at a constant angular speed. When angles are close to the target
    angles, these speeds are reduced.'''

    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.call_rate  = kwargs.pop('call_rate' , 10)             # secs
        self.ang_speed  = kwargs.pop('ang_speed',  8.*deg_to_rad)  # rad/s
        self.ang_cutoff = kwargs.pop('ang_cutoff', 5.*deg_to_rad)  # rad

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if assist_level > 0:
            assist_rh_pos = np.zeros((0, 1))
            assist_rh_vel = np.zeros((0, 1))

            for i in range(4):
                rh_i_pos = np.array(current_state[i, 0]).ravel()
                target_rh_i_pos = np.array(target_state[i, 0]).ravel()
                assist_rh_i_pos, assist_rh_i_vel = self._angle_assist(rh_i_pos, target_rh_i_pos)
                assist_rh_pos = np.vstack([assist_rh_pos, assist_rh_i_pos])
                assist_rh_vel = np.vstack([assist_rh_vel, assist_rh_i_vel])

            # if mode == 'hold':
            #     print 'task state is "hold", setting assist vels to 0'
            #     assist_rh_vel[:] = 0.

            Bu = assist_level * np.vstack([assist_rh_pos,
                                           assist_rh_vel,
                                           1])
            Bu = np.mat(Bu.reshape(-1, 1))

            assist_weight = assist_level
        else:
            Bu = None
            assist_weight = 0.
        return dict(Bu=Bu, assist_level=assist_weight)
        # return Bu, assist_weight

    def _angle_assist(self, ang_pos, target_ang_pos):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        ang_diff = angle_subtract(target_ang_pos, ang_pos)
        if abs(ang_diff) > self.ang_cutoff:
            assist_ang_vel = np.sign(ang_diff) * self.ang_speed
        else:
            assist_ang_vel = 0.5 * (ang_diff / self.ang_cutoff) * self.ang_speed

        assist_ang_pos = ang_pos + assist_ang_vel/self.call_rate

        return assist_ang_pos, assist_ang_vel


class IsMoreAssister(Assister):
    '''Combines an ArmAssistAssister and a ReHandAssister.'''

    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.aa_assister = ArmAssistAssister(*args, **kwargs)
        self.rh_assister = ReHandAssister(*args, **kwargs)

    def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if assist_level > 0:
            aa_current_state = np.vstack([current_state[0:3], current_state[7:10], 1])
            aa_target_state  = np.vstack([target_state[0:3], target_state[7:10], 1])
            aa_Bu = self.aa_assister.calc_assisted_BMI_state(aa_current_state,
                                                             aa_target_state,
                                                             assist_level,
                                                             mode=mode,
                                                             **kwargs)['Bu']

            rh_current_state = np.vstack([current_state[3:7], current_state[10:14], 1])
            rh_target_state  = np.vstack([target_state[3:7], target_state[10:14], 1])
            rh_Bu = self.rh_assister.calc_assisted_BMI_state(rh_current_state,
                                                             rh_target_state,
                                                             assist_level,
                                                             mode=mode,
                                                             **kwargs)['Bu']

            Bu = np.vstack([aa_Bu[0:3],
                            rh_Bu[0:4],
                            aa_Bu[3:6],
                            rh_Bu[4:8],
                            assist_level * 1])
            Bu = np.mat(Bu.reshape(-1, 1))

            assist_weight = assist_level
        else:
            Bu = None
            assist_weight = 0.
        return dict(Bu=Bu, assist_level=assist_weight)
        # return Bu, assist_weight


# LFC iBMI assisters
# not inheriting from LinearFeedbackControllerAssist/SSMLFCAssister because:
# - use of "special angle subtraction" when doing target_state - current_state
# - meant to be used with 'weighted_avg_lfc'=True decoder kwarg, and thus 
#   assist_weight is set to assist_level, not to 0

#  increasing the values in the 'R' matrix you're more heavily weighting the velocity input in the cost function.

from riglib.bmi.feedback_controllers import LQRController
from riglib.bmi.assist import FeedbackControllerAssist

ssm = StateSpaceArmAssist()
A, B, _ = ssm.get_ssm_matrices()
Q = np.mat(np.diag([10., 10., 10., 5, 5, 5, 0]))
R = 1e8 * np.mat(np.diag([1., 1., 1.]))
arm_assist_controller2 = LQRController(A, B, Q, R)

ssm = StateSpaceReHand()
A, B, _ = ssm.get_ssm_matrices()
Q = np.mat(np.diag([1., 1., 1., 1., 0.5, 0.5, 0.5, 0.5, 0]))
R = 1e6 * np.mat(np.diag([1., 1., 1., 1.]))
rehand_controller2 = LQRController(A, B, Q, R)

##-------------------------------------------------##

ssm = StateSpaceArmAssist()
A, B, _ = ssm.get_ssm_matrices()
# Same LQR controller but with different values to use them in the go_to_start phase of the playbacktrajectories task
#Q = np.mat(np.diag([10., 10., 10., 5, 5, 5, 0])) # original values
# Q = np.mat(np.diag([30., 30., 30., 10, 10, 10, 0]))
# Q = np.mat(np.diag([50., 50., 50., 10, 10, 10, 0]))
# R = 1e6 * np.mat(np.diag([1., 1., 1.])) # original values
R = 1e6 * np.mat(np.diag([1., 1., 1.]))
#arm_assist_controller_go_to_start = LQRController(A, B, Q, R)
# arm_assist_controller_go_to_start = LQRController_accel_limit_armassist(A, B, Q, R)

ssm = StateSpaceReHand()
A, B, _ = ssm.get_ssm_matrices()
# Same LQR controller but with different values to use them in the go_to_start phase of the playbacktrajectories task
#Q = np.mat(np.diag([1., 1., 1., 1., 0.5, 0.5, 0.5, 0.5, 0])) #original values
# Q = np.mat(np.diag([30., 30., 30.,30., 10, 10, 10, 10, 0]))
R = 1e6 * np.mat(np.diag([1., 1., 1., 1.]))
#rehand_controller_go_to_start = LQRController(A, B, Q, R)
# rehand_controller_go_to_start = LQRController_accel_limit_rehand(A, B, Q, R)

ssm = StateSpaceIsMore()
A, B, _ = ssm.get_ssm_matrices()
#Q = np.mat(np.diag([15., 15., 15.,15., 15., 15.,15., 5, 5, 5, 5, 5, 5, 5, 0])) 
Q = np.mat(np.diag([30., 30., 30.,30., 30., 30.,30., 10, 10, 10, 10, 10, 10, 10, 0])) 
# Q = np.mat(np.diag([50., 50., 50.,50., 50., 50.,50., 10, 10, 10, 10, 10, 10, 10, 0])) 
# Q = np.mat(np.diag([70., 70., 70.,70., 70., 70.,70., 10., 10., 10.,10., 10., 10.,10., 0])) 

# Q matrix for cyclic movements
# Q = np.mat(np.diag([80., 80., 80.,80., 80., 80.,80., 1., 1., 1., 1., 1., 1., 1., 0])) 
#Q = np.mat(np.diag([80., 80., 80.,80., 80., 80.,80., 10., 10., 10., 10., 10., 10., 10., 0])) 

R = 1e6 * np.mat(np.diag([1., 1., 1., 1., 1., 1., 1.])) #original: 1e6
# ismore_controller_go_to_start = LQRController_accel_limit_ismore(A, B, Q, R)


##-------------------------------------------------##



# class ArmAssistLFCAssister(FeedbackControllerAssist):
#     '''
#     Assister for ArmAssist which uses an infinite-horizon LQR controller
#     '''
#     def __init__(self, *args, **kwargs):
#         super(ArmAssistLFCAssister, self).__init__(arm_assist_controller, style='mixing')

        #self.A = A
        #self.B = B
        #self.Q = Q
        #self.R = R
        #self.F = feedback_controllers.LQRController.dlqr(A, B, Q, R)

    # def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
    #     '''TODO.'''

    #     diff = target_state - current_state
    #     diff[2] = angle_subtract(target_state[2], current_state[2])

    #     Bu = assist_level * self.B*self.F*diff
    #     assist_weight = assist_level
    #     return dict(Bu=Bu, assist_level=assist_weight)
    #     # return Bu, assist_weight


class ReHandLFCAssister(FeedbackControllerAssist):
    def __init__(self, *args, **kwargs):
        super(ReHandLFCAssister, self).__init__(rehand_controller2, style='mixing')        
        # ssm = StateSpaceReHand()
        # A, B, _ = ssm.get_ssm_matrices()
        # Q = np.mat(np.diag([1., 1., 1., 1., 0, 0, 0, 0, 0]))
        # R = 1e6 * np.mat(np.diag([1., 1., 1., 1.]))
        
        # self.A = A
        # self.B = B
        # self.Q = Q
        # self.R = R
        # self.F = feedback_controllers.LQRController.dlqr(A, B, Q, R)

    # def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
    #     '''TODO.'''

    #     diff = target_state - current_state
    #     for i in range(4):
    #         diff[i] = angle_subtract(target_state[i], current_state[i])

    #     Bu = assist_level * self.B*self.F*diff
    #     assist_weight = assist_level
    #     return dict(Bu=Bu, assist_level=assist_weight)
    #     # return Bu, assist_weight


# class IsMoreLFCAssister(Assister):
#     '''
#     Docstring

#     Parameters
#     ----------

#     Returns
#     -------
#     '''
#     def __init__(self, *args, **kwargs):
#         '''
#         Docstring

#         Parameters
#         ----------

#         Returns
#         -------
#         '''
#         ssm = StateSpaceIsMore()
#         A, B, _ = ssm.get_ssm_matrices()        
#         Q = np.mat(np.diag([7., 7., 7., 7., 7., 7., 7., 0, 0, 0, 0, 0, 0, 0, 0]))
#         R = 1e6 * np.mat(np.diag([1., 1., 1., 1., 1., 1., 1.]))

#         self.A = A
#         self.B = B
#         self.Q = Q
#         self.R = R
#         self.F = feedback_controllers.LQRController.dlqr(A, B, Q, R)

#     def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
#         '''TODO.'''
#         diff = target_state - current_state
#         for i in range(2, 7):
#             diff[i] = angle_subtract(target_state[i], current_state[i])

#         Bu = assist_level * self.B*self.F*diff
#         assist_weight = assist_level
#         return dict(Bu=Bu, assist_level=assist_weight)
#         # return Bu, assist_weight

######################################################################################################################
##### Same LFCAssisters but with different values to use them in the go_to_start phase of the playbacktrajectories task
######################################################################################################################

class ArmAssistLFCAssisterGoToStart(FeedbackControllerAssist):
    '''
    Assister for ArmAssist which uses an infinite-horizon LQR controller
    '''
    def __init__(self, *args, **kwargs):
        ssm = StateSpaceArmAssist()
        A, B, _ = ssm.get_ssm_matrices()
        # R = 1e6 * np.mat(np.diag([1., 1., 1.])) # original values
        R = 1e6 * np.mat(np.diag([1., 1., 1.]))
        Q0 = np.mat(np.diag([10., 10., 10., 10, 10, 5,  0])) 
        Q1 = np.mat(np.diag([30., 30., 30., 10, 10, 10,  0])) 
        Q2 = np.mat(np.diag([50., 50., 40., 10, 10, 10,  0])) 
        Q3 = np.mat(np.diag([70., 70., 50., 10.,10.,10., 0])) 

        if kwargs['speed'] == 'very-low':
            # Q = np.mat(np.diag([30., 30., 30., 10, 10, 10, 0])) 
            arm_assist_controller_go_to_start = LQRController_accel_limit_armassist(A, B, Q0, R)
        elif kwargs['speed'] == 'low':
            # Q = np.mat(np.diag([30., 30., 30., 10, 10, 10, 0])) 
            arm_assist_controller_go_to_start = LQRController_accel_limit_armassist(A, B, Q1, R)
        elif kwargs['speed'] == 'medium':
            # Q = np.mat(np.diag([50., 50., 50., 10, 10, 10, 0])) 
            arm_assist_controller_go_to_start = LQRController_accel_limit_armassist(A, B, Q2, R)
        elif kwargs['speed'] == 'high':
            # print "before Q", Q
            # Q = np.mat(np.diag([70., 70., 70., 10., 10.,10., 0])) 
            arm_assist_controller_go_to_start = LQRController_accel_limit_armassist(A, B, Q3, R)

        super(ArmAssistLFCAssisterGoToStart, self).__init__(arm_assist_controller_go_to_start, style='mixing')
        

class ReHandLFCAssisterGoToStart(FeedbackControllerAssist):
    def __init__(self, *args, **kwargs):
        ssm = StateSpaceReHand()
        A, B, _ = ssm.get_ssm_matrices()
        R = 1e6 * np.mat(np.diag([1., 1., 1., 1.]))
        Q0 = np.mat(np.diag([10., 10., 10.,10., 10, 10, 10, 10, 0])) 
        Q1 = np.mat(np.diag([30., 30., 30.,30., 10, 10, 10, 10, 0])) 
        Q2 = np.mat(np.diag([50., 50., 50.,50., 10, 10, 10, 10, 0])) 
        Q3 = np.mat(np.diag([70., 70., 70.,70., 10.,10.,10.,10.,0])) 

        if kwargs['speed'] == 'very-low':
            rehand_controller_go_to_start = LQRController_accel_limit_rehand(A, B, Q0, R)
        elif kwargs['speed'] == 'low':
            rehand_controller_go_to_start = LQRController_accel_limit_rehand(A, B, Q1, R)
        elif kwargs['speed'] == 'medium':
            rehand_controller_go_to_start = LQRController_accel_limit_rehand(A, B, Q2, R)
        elif kwargs['speed'] == 'high':
            rehand_controller_go_to_start = LQRController_accel_limit_rehand(A, B, Q3, R)

        super(ReHandLFCAssisterGoToStart, self).__init__(rehand_controller_go_to_start, style='mixing')        
       
class IsMoreLFCAssisterGoToStart(FeedbackControllerAssist):
    def __init__(self, *args, **kwargs):
        ssm = StateSpaceIsMore()
        A, B, _ = ssm.get_ssm_matrices()
        R = 1e6 * np.mat(np.diag([1., 1., 1., 1., 1., 1., 1.])) #original: 1e6
        Q0 = np.mat(np.diag([10., 10., 10.,10., 10., 10.,10., 10, 10, 1, 10, 10, 10, 10, 0])) 
        Q1 = np.mat(np.diag([30., 30., 30.,30., 30., 30.,30., 10, 10, 1, 10, 10, 10, 10, 0])) 
        Q2 = np.mat(np.diag([50., 50., 40.,50., 50., 50.,50., 10, 10, 10, 10, 10, 10, 10, 0])) 
        Q3 = np.mat(np.diag([70., 70., 50.,70., 70., 70.,70., 10.,10.,10.,10.,10.,10.,10.,0]))
        Q4 = np.mat(np.diag([100., 100., 100.,100., 100., 100.,100., 10.,10.,10.,10.,10.,10.,10.,0])) 
        Q5 = np.mat(np.diag([150., 150., 150.,150., 150., 150.,150., 10.,10.,10.,10.,10.,10.,10.,0])) 
        
        if kwargs['speed'] == 'very-low':
            ismore_controller_go_to_start = LQRController_accel_limit_ismore(A, B, Q0, R)
        if kwargs['speed'] == 'low':
            ismore_controller_go_to_start = LQRController_accel_limit_ismore(A, B, Q1, R)
        elif kwargs['speed'] == 'medium':
            ismore_controller_go_to_start = LQRController_accel_limit_ismore(A, B, Q2, R)
        elif kwargs['speed'] == 'high':
            ismore_controller_go_to_start = LQRController_accel_limit_ismore(A, B, Q3, R)
        elif kwargs['speed'] == 'very-high':
            ismore_controller_go_to_start = LQRController_accel_limit_ismore(A, B, Q4, R)
        elif kwargs['speed'] == 'super-high':
            ismore_controller_go_to_start = LQRController_accel_limit_ismore(A, B, Q5, R)
            

        super(IsMoreLFCAssisterGoToStart, self).__init__(ismore_controller_go_to_start, style='mixing')        
     
class IsMoreLFCAssister_diff_assist(FeedbackControllerAssist_StateSpecAssistLevels):
    def __init__(self, *args, **kwargs):
        ssm = StateSpaceIsMore()
        A, B, _ = ssm.get_ssm_matrices()
        R = 1e6 * np.mat(np.diag([1., 1., 1., 1., 1., 1., 1.])) #original: 1e6
        Q0 = np.mat(np.diag([10., 10., 10.,10., 10., 10.,10., 10, 10, 1, 10, 10, 10, 10, 0])) 
        Q1 = np.mat(np.diag([30., 30., 30.,30., 30., 30.,30., 10, 10, 1, 10, 10, 10, 10, 0])) 
        Q2 = np.mat(np.diag([50., 50., 40.,50., 50., 50.,50., 10, 10, 10, 10, 10, 10, 10, 0])) 
        Q3 = np.mat(np.diag([70., 70., 50.,70., 70., 70.,70., 10.,10.,10.,10.,10.,10.,10.,0])) 
        Q4 = np.mat(np.diag([100., 100., 100.,100., 100., 100.,100., 10.,10.,10.,10.,10.,10.,10.,0])) 
        Q5 = np.mat(np.diag([150., 150., 150.,150., 150., 150.,150., 10.,10.,10.,10.,10.,10.,10.,0])) 

        if kwargs['speed'] == 'very-low':
            ismore_controller_go_to_start = LQRController_accel_limit_ismore(A, B, Q0, R)
        if kwargs['speed'] == 'low':
            ismore_controller_go_to_start = LQRController_accel_limit_ismore(A, B, Q1, R)
        elif kwargs['speed'] == 'medium':
            ismore_controller_go_to_start = LQRController_accel_limit_ismore(A, B, Q2, R)
        elif kwargs['speed'] == 'high':
            ismore_controller_go_to_start = LQRController_accel_limit_ismore(A, B, Q3, R)
        elif kwargs['speed'] == 'very-high':
            ismore_controller_go_to_start = LQRController_accel_limit_ismore(A, B, Q4, R)
        elif kwargs['speed'] == 'super-high':
            ismore_controller_go_to_start = LQRController_accel_limit_ismore(A, B, Q5, R)

        super(IsMoreLFCAssister_diff_assist, self).__init__(ismore_controller_go_to_start, style='mixing')        

class IsMoreLFCAssister_diff_assist_high_accel_hand(FeedbackControllerAssist_StateSpecAssistLevels):
    def __init__(self, *args, **kwargs):
        ssm = StateSpaceIsMore()
        A, B, _ = ssm.get_ssm_matrices()
        R = 1e6 * np.mat(np.diag([1., 1., 1., 1., 1., 1., 1.])) #original: 1e6
        Q0 = np.mat(np.diag([10., 10., 10.,10., 10., 10.,10., 10, 10, 1, 10, 10, 10, 10, 0])) 
        Q1 = np.mat(np.diag([30., 30., 30.,30., 30., 30.,30., 10, 10, 1, 10, 10, 10, 10, 0])) 
        Q2 = np.mat(np.diag([50., 50., 40.,50., 50., 50.,50., 10, 10, 10, 10, 10, 10, 10, 0])) 
        Q3 = np.mat(np.diag([70., 70., 50.,70., 70., 70.,70., 10.,10.,10.,10.,10.,10.,10.,0])) 
        Q4 = np.mat(np.diag([100., 100., 100.,100., 100., 100.,100., 10.,10.,10.,10.,10.,10.,10.,0])) 
        Q5 = np.mat(np.diag([150., 150., 150.,150., 150., 150.,150., 10.,10.,10.,10.,10.,10.,10.,0])) 

        if kwargs['speed'] == 'very-low':
            ismore_controller_go_to_start = LQRController_accel_limit_only_base_ismore(A, B, Q0, R)
        if kwargs['speed'] == 'low':
            ismore_controller_go_to_start = LQRController_accel_limit_only_base_ismore(A, B, Q1, R)
        elif kwargs['speed'] == 'medium':
            ismore_controller_go_to_start = LQRController_accel_limit_only_base_ismore(A, B, Q2, R)
        elif kwargs['speed'] == 'high':
            ismore_controller_go_to_start = LQRController_accel_limit_only_base_ismore(A, B, Q3, R)
        elif kwargs['speed'] == 'very-high':
            ismore_controller_go_to_start = LQRController_accel_limit_only_base_ismore(A, B, Q4, R)
        elif kwargs['speed'] == 'super-high':
            ismore_controller_go_to_start = LQRController_accel_limit_only_base_ismore(A, B, Q5, R)


        super(IsMoreLFCAssister_diff_assist_high_accel_hand, self).__init__(ismore_controller_go_to_start, style='mixing')        


# class IsMoreLFCAssisterGoToStart(Assister): 
#     '''
#     Docstring

#     Parameters
#     ----------

#     Returns
#     -------
#     '''
#     def __init__(self, *args, **kwargs):
#         '''
#         Docstring

#         Parameters
#         ----------

#         Returns
#         -------
#         '''
#         ssm = StateSpaceIsMore()
#         A, B, _ = ssm.get_ssm_matrices()  
#         self.prev_assister_output = np.nan
#         self.accel_lim_armassist = 100#.9
#         self.accel_lim_rehand = 10#.02      
#         #Q = np.mat(np.diag([7., 7., 7., 7., 7., 7., 7., 0, 0, 0, 0, 0, 0, 0, 0])) #original values
#         Q = np.mat(np.diag([15., 15., 15.,15., 15., 15.,15., 5, 5, 5, 5, 5, 5, 5, 0])) 
#         R = 1e6 * np.mat(np.diag([1., 1., 1., 1., 1., 1., 1.]))

#         #nerea
#         # Q = np.mat(np.diag([5., 5., 5.,5., 5., 5.,5., 0, 0, 0, 0, 0, 0, 0, 0])) 
#         # R = 1e6 * np.mat(np.diag([1., 1., 1., 1., 1., 1., 1.]))

#         self.A = A
#         self.B = B
#         self.Q = Q
#         self.R = R
#         self.F = feedback_controllers.LQRController.dlqr(A, B, Q, R)

#     def calc_assisted_BMI_state(self, current_state, target_state, assist_level, mode=None, **kwargs):
#         '''TODO.'''

#         diff = target_state - current_state
#         for i in range(2, 7):
#             diff[i] = angle_subtract(target_state[i], current_state[i])

#         Bu = assist_level * self.B*self.F*diff
#         # limit acceleration
#         if np.sum(Bu[8:16]) != 0:
#             if np.any(np.isnan(self.prev_assister_output)):
#                 self.prev_assister_output = np.zeros_like(Bu)

#             assister_output_accel = Bu - self.prev_assister_output
#             #print 'dBU output', np.squeeze(assister_output_accel)
#             for i in np.arange(7,10):
#                 if assister_output_accel[i, 0] > self.accel_lim_armassist:
#                     Bu[i, 0] = self.prev_assister_output[i, 0] + self.accel_lim_armassist
#                 elif assister_output_accel[i, 0] < -1*self.accel_lim_armassist:
#                     Bu[i, 0] = self.prev_assister_output[i, 0] - self.accel_lim_armassist
#             for i in np.arange(10,14):
#                 if assister_output_accel[i, 0] > self.accel_lim_rehand:
#                     Bu[i, 0] = self.prev_assister_output[i, 0] + self.accel_lim_rehand
#                 elif assister_output_accel[i, 0] < -1*self.accel_lim_rehand:
#                     Bu[i, 0] = self.prev_assister_output[i, 0] - self.accel_lim_rehand
        

#         print 'Bu', Bu
#         self.prev_assister_output = Bu


        
#         assist_weight = assist_level
#         return dict(x_assist=Bu, assist_level=assist_weight)
#         # return Bu, assist_weight


    


###################
## iBMI learners ##
###################

# simple iBMI learners that just use an "assister" object

class ArmAssistLearner(Learner):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        decoder_binlen = kwargs.pop('decoder_binlen', 0.1)
        assist_speed   = kwargs.pop('assist_speed', 2.)
        target_radius  = kwargs.pop('target_radius', 2.)
        assister_kwargs = dict(decoder_binlen=decoder_binlen, target_radius=target_radius, assist_speed=assist_speed)
        self.assister = assist.ArmAssistAssister(**assister_kwargs)

        super(ArmAssistLearner, self).__init__(*args, **kwargs)

        self.input_state_index = -1

    def calc_int_kin(self, decoder_state, target_state, decoder_output, task_state, state_order=None):
        """Calculate/estimate the intended ArmAssist kinematics."""
        current_state = decoder_state[:, None]  # assister expects shape to be (7, 1)
        target_state  = target_state[:, None]   # assister expects shape to be (7, 1)
        intended_state = self.assister(current_state, target_state, 1)[0]

        return intended_state

    def __call__(self, neural_features, decoder_state, target_state, decoder_output, task_state, state_order=None):
        '''Calculate the intended kinematics and pair with the neural data.'''
        super(ArmAssistLearner, self).__call__(neural_features, decoder_state, target_state, decoder_output, task_state, state_order=state_order)


class ReHandLearner(Learner):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        decoder_binlen = kwargs.pop('decoder_binlen', 0.1)
        assist_speed   = kwargs.pop('assist_speed', 2.)
        target_radius  = kwargs.pop('target_radius', 2.)
        assister_kwargs = dict(decoder_binlen=decoder_binlen, target_radius=target_radius, assist_speed=assist_speed)
        self.assister = assist.ReHandAssister(**assister_kwargs)

        super(ReHandLearner, self).__init__(*args, **kwargs)

        self.input_state_index = -1

    def calc_int_kin(self, decoder_state, target_state, decoder_output, task_state, state_order=None):
        """Calculate/estimate the intended ReHand kinematics."""
        current_state = decoder_state[:, None]  # assister expects shape to be (9, 1)
        target_state  = target_state[:, None]   # assister expects shape to be (9, 1)
        intended_state = self.assister(current_state, target_state, 1)[0]

        return intended_state

    def __call__(self, neural_features, decoder_state, target_state, decoder_output, task_state, state_order=None):
        '''Calculate the intended kinematics and pair with the neural data.'''
        super(ReHandLearner, self).__call__(neural_features, decoder_state, target_state, decoder_output, task_state, state_order=state_order)


class IsMoreLearner(Learner):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        decoder_binlen = kwargs.pop('decoder_binlen', 0.1)
        assist_speed   = kwargs.pop('assist_speed', 2.)
        target_radius  = kwargs.pop('target_radius', 2.)
        assister_kwargs = dict(decoder_binlen=decoder_binlen, target_radius=target_radius, assist_speed=assist_speed)
        self.assister = assist.IsMoreAssister(**assister_kwargs)

        super(IsMoreLearner, self).__init__(*args, **kwargs)

        self.input_state_index = -1

    def calc_int_kin(self, decoder_state, target_state, decoder_output, task_state, state_order=None):
        """Calculate/estimate the intended ArmAssist+ReHand kinematics."""
        current_state = decoder_state[:, None]  # assister expects shape to be (15, 1)
        target_state  = target_state[:, None]   # assister expects shape to be (15, 1)
        intended_state = self.assister(current_state, target_state, 1)[0]

        return intended_state

    def __call__(self, neural_features, decoder_state, target_state, decoder_output, task_state, state_order=None):
        '''Calculate the intended kinematics and pair with the neural data.'''
        super(IsMoreLearner, self).__call__(neural_features, decoder_state, target_state, decoder_output, task_state, state_order=state_order)



    


########################################

# Define some dictionaries below so that we don't have to always write:
#     if self.plant_type == 'ArmAssist':
#         ...
#     elif self.plant_type == 'ReHand':
#         ...

SSM_CLS_DICT = {
    'ArmAssist': StateSpaceArmAssist,
    'ReHand':    StateSpaceReHand,
    'IsMore':    StateSpaceIsMore,
    'DummyPlant': StateSpaceDummy,
    'IsMoreEMGControl': StateSpaceIsMore,
    'IsMoreHybridControl': StateSpaceIsMore,
    'IsMorePlantHybridBMISoftSafety': StateSpaceIsMore,
}

ASSISTER_CLS_DICT = {
    'ArmAssist': ArmAssistAssister,
    'ReHand':    ReHandAssister,
    'IsMore':    IsMoreAssister,
}

LFC_ASSISTER_CLS_DICT = {
    'ArmAssist': ArmAssistLFCAssister,
    'ReHand':    ReHandLFCAssister,
    'IsMore':    IsMoreLFCAssister,
}

LFC_GO_TO_START_ASSISTER_CLS_DICT_OG = {
    'ArmAssist': ArmAssistLFCAssisterGoToStart,
    'ReHand':    ReHandLFCAssisterGoToStart,
    'IsMore':    IsMoreLFCAssisterGoToStart,
    'IsMoreHybridControl': IsMoreLFCAssisterGoToStart,
    'IsMoreEMGControl': IsMoreLFCAssisterGoToStart,
    'IsMorePlantHybridBMISoftSafety': IsMoreLFCAssisterGoToStart,
    }
    
LFC_GO_TO_START_ASSISTER_CLS_DICT = {
    'IsMore':   IsMoreLFCAssister_diff_assist,
    'IsMoreHybridControl': IsMoreLFCAssister_diff_assist,
    'IsMorePlantHybridBMISoftSafety': IsMoreLFCAssister_diff_assist_high_accel_hand,
}

# GOAL_CALCULATOR_CLS_DICT = {
#     'ArmAssist': ArmAssistControlGoal,
#     'ReHand':    ReHandControlGoal,
#     'IsMore':    IsMoreControlGoal,
# }


LEARNER_CLS_DICT = {
    'ArmAssist': ArmAssistLearner,
    'ReHand':    ReHandLearner,
    'IsMore':    IsMoreLearner,
}

OFC_LEARNER_CLS_DICT = {
    'ArmAssist': ArmAssistOFCLearner,
    'ReHand':    ReHandOFCLearner,
    'IsMore':    IsMoreOFCLearner,
}

OFC_LEARNER_CLS_DICT_w_REST = {
    'IsMore':    IsMoreOFCLearning_w_Rest,
    'IsMoreHybridControl': IsMoreOFCLearning_w_Rest
}

ORTHO_DAMP_ASSIST_CLS_DICT = {
    'ArmAssist': ArmAssistLFCOrthoDampAssister, 
    'ReHand':   ReHandLFCOrthoDampAssister,
    #'IsMore': IsMoreLFCOrthoDampAssister,
    'IsMore': IsMoreLFCOrthoDamp_diff_assist,
}
