#Copied from bmi_tasks_analysis (same filename) - HS 20211026
from riglib.bmi import kfdecoder
import numpy as np
from riglib.experiment import traits
from .bmimultitasks import SimpleEndpointAssister

class CurlFieldKalmanFilter(kfdecoder.KalmanFilter):
    def _calc_kalman_gain(self, P):
        '''
        see KalmanFilter._calc_kalman_gain
        '''
        K = super(CurlFieldKalmanFilter, self)._calc_kalman_gain(P)
        v_norm = np.linalg.norm(np.array(self.state.mean[3:6,0]))
        rot_angle_deg = self.rot_factor*v_norm
        theta = np.deg2rad(rot_angle_deg)

        R = np.mat([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        K[[0,2],:] = R * K[[0,2],:]
        K[[3,5],:] = R * K[[3,5],:]
        return K

class VisRotKalmanFilter(kfdecoder.KalmanFilter):
    def _calc_kalman_gain(self, P):
        '''
        see KalmanFilter._calc_kalman_gain
        '''
        K = super(VisRotKalmanFilter, self)._calc_kalman_gain(P)
        theta = np.deg2rad(self.rot_angle_deg)

        R = np.mat([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        K[[0,2],:] = R * K[[0,2],:]
        K[[3,5],:] = R * K[[3,5],:]
        return K

from .bmimultitasks import BMIControlMulti, BMIResetting
class BMICursorKinematicCurlField(BMIResetting):
    rot_factor = traits.Float(10., desc='scaling factor from speed to rotation angle in degrees')
    def load_decoder(self):
        super(BMICursorKinematicCurlField, self).load_decoder()
        # Conver the KF to a curl-field generating KF
        dec = self.decoder
        filt = CurlFieldKalmanFilter(A=self.decoder.filt.A, W=self.decoder.filt.W, C=dec.filt.C, Q=dec.filt.Q, is_stochastic=dec.filt.is_stochastic)
        filt.C_xpose_Q_inv = dec.filt.C_xpose_Q_inv
        filt.C_xpose_Q_inv_C = dec.filt.C_xpose_Q_inv_C
        filt.rot_factor = self.rot_factor
        self.decoder.filt = filt


from riglib import plants
class CursorErrorClampPlant(plants.CursorPlant):
    def __init__(self, *args, **kwargs):
        super(CursorErrorClampPlant, self).__init__(*args, **kwargs)
        self.axis = None

    def set_intrinsic_coordinates(self, pt):
        if self.axis is None:
            self.position = pt
        else:
            origin, terminus = self.axis
            vec_to_pt = pt - origin
            vec_to_targ = terminus - origin

            scale = np.dot(vec_to_targ, vec_to_pt) / np.dot(vec_to_targ, vec_to_targ)
            self.position = scale*vec_to_targ + origin
        self.draw()

class CursorErrorClamp(object):
    exclude_parent_traits = ['plant_type']
    sequence_generators = ['center_out_error_clamp_infrequent']
    def __init__(self, *args, **kwargs):
        self.plant = CursorErrorClampPlant(endpt_bounds=(-25, 25, 0., 0., -14, 14))
        super(CursorErrorClamp, self).__init__(*args, **kwargs)

    def _parse_next_trial(self):
        if isinstance(self.next_trial, dict):
            for key in self.next_trial:
                setattr(self, '_gen_%s' % key, self.next_trial[key])

        self.targs = self._gen_targs
        if self._gen_error_clamp:
            self.plant.axis = (self._gen_targs[0], self._gen_targs[1])
        else:
            self.plant.axis = None

        self.reportstats['Perturbation'] = str(self._gen_curl)
        self.reportstats['Error clamp'] = str(self._gen_error_clamp)

    def init(self):
        self.add_dtype('error_clamp', 'i', (1,))
        self.add_dtype('pert', 'i', (1,))
        super(CursorErrorClamp, self).init()

    def _cycle(self):
        self.task_data['error_clamp'] = self._gen_error_clamp
        self.task_data['pert'] = self._gen_curl
        super(CursorErrorClamp, self)._cycle()

    @staticmethod 
    def center_out_error_clamp_infrequent(ntargets=8, distance=10, n_baseline_blocks=8, n_pert_learning_blocks=8, n_pert_err_clamp_blocks=8, n_washout_blocks=16):
        # need to return: 1) target sequences, 2) baseline or perturbation, 3) error clamp axis

        error_clamp_trials_per_block = 1

        n_meta_blocks = int(ntargets/error_clamp_trials_per_block)
        n_baseline_metablocks = int(n_baseline_blocks/n_meta_blocks)
        n_pert_learning_metablocks = int(n_pert_learning_blocks/n_meta_blocks)
        n_pert_err_clamp_metablocks = int(n_pert_err_clamp_blocks/n_meta_blocks)
        n_washout_metablocks = int(n_washout_blocks/n_meta_blocks)
        

        target_angles = np.arange(-np.pi, np.pi, 2*np.pi/ntargets)
        targs = np.vstack([np.cos(target_angles), np.zeros_like(target_angles), np.sin(target_angles)]).T * distance
        targ_seqs = [dict(targs=np.vstack([np.zeros(3), targ]), error_clamp=False, curl=False) for targ in targs]

        metablock = []
        for k in range(n_meta_blocks):
            metablock.append([x.copy() for x in targ_seqs])

        for k in range(ntargets):
            metablock[k][k]['error_clamp'] = True

        trials = []
        import copy
        from random import shuffle

        for _ in range(n_baseline_metablocks):
            _metablock = copy.deepcopy(metablock)
            for row in _metablock:
                shuffle(row)
            shuffle(_metablock)
            for row in _metablock:
                trials += row

        for _ in range(n_pert_learning_metablocks):
            _metablock = copy.deepcopy(metablock)
            for row in _metablock:
                shuffle(row)
                for tr in row:
                    tr['curl'] = True
                    tr['error_clamp'] = False

            shuffle(_metablock)
            for row in _metablock:
                trials += row

        for _ in range(n_pert_err_clamp_metablocks):
            _metablock = copy.deepcopy(metablock)
            for row in _metablock:
                shuffle(row)
                for tr in row:
                    tr['curl'] = True

            shuffle(_metablock)
            for row in _metablock:
                trials += row

        for _ in range(n_washout_metablocks):
            _metablock = copy.deepcopy(metablock)
            for row in _metablock:
                shuffle(row)
                for tr in row:
                    tr['curl'] = False
                    tr['error_clamp'] = False

            shuffle(_metablock)
            for row in _metablock:
                trials += row
        print(np.shape(trials))
        return trials

class BMICursorKinematicCurlErrorClamp(BMICursorKinematicCurlField, CursorErrorClamp):
    exclude_parent_traits = ['plant_type']
    sequence_generators = ['center_out_error_clamp_infrequent']    
    def _parse_next_trial(self):
        super(BMICursorKinematicCurlErrorClamp, self)._parse_next_trial()
        self.decoder.filt.rot_factor = self.rot_factor * int(self._gen_curl)

class BMICursorVisRotErrorClamp(CursorErrorClamp, BMIResetting):
    #fps = 60. # YZ the original fps is set to 60, not sure why here change into 20
    background = (0,0,0,1)
    exclude_parent_traits = ['plant_type', 'timeout_penalty_time', 'marker_num', 'plant_hide_rate', 'plant_visible', 'cursor_radius', 'show_environment', 'rand_start', 'hold_penalty_time']
    sequence_generators = ['center_out_error_clamp_infrequent']    
    rot_angle_deg = traits.Float(10., desc='scaling factor from speed to rotation angle in degrees')    
    def _parse_next_trial(self):
        super(BMICursorVisRotErrorClamp, self)._parse_next_trial()
        self.decoder.filt.rot_angle_deg = self.rot_angle_deg * int(self._gen_curl)

    def create_assister(self):
        kwargs = dict(decoder_binlen=self.decoder.binlen, target_radius=self.target_radius)
        if hasattr(self, 'assist_speed'):
            kwargs['assist_speed'] = self.assist_speed    
        self.assister = SimpleEndpointAssister(**kwargs)

    def load_decoder(self):
        super(BMICursorVisRotErrorClamp, self).load_decoder()
        # Convert the KF to a curl-field generating KF
        dec = self.decoder
        filt = VisRotKalmanFilter(A=dec.filt.A, W=dec.filt.W, C=dec.filt.C, Q=dec.filt.Q, is_stochastic=dec.filt.is_stochastic)
        filt.C_xpose_Q_inv = dec.filt.C_xpose_Q_inv
        filt.C_xpose_Q_inv_C = dec.filt.C_xpose_Q_inv_C
        filt.rot_angle_deg = self.rot_angle_deg
        self.decoder.filt = filt        
