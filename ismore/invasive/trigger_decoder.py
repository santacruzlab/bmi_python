from riglib.bmi import bmi
import numpy as np

class State(object):
    '''For compatibility with other BMI decoding implementations, 
    literally just holds the state'''
    def __init__(self, mean, *args, **kwargs):
        self.mean = mean

class RidgeDecoder(bmi.Decoder):
    def __init__(self, *args, **kwargs):
        super(RidgeDecoder, self).__init__(args[0], args[1], args[2],
            binlen=kwargs['binlen'], call_rate=20.)

        self.extractor_cls = args[3]
        self.extractor_kwargs = args[4]
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.filt.state.mean[key, 0]
        elif key == 'q':
            pos_states, = np.nonzero(self.ssm.state_order == 0)
            return np.array([self.__getitem__(k) for k in pos_states])
        elif key == 'qdot':
            vel_states, = np.nonzero(self.ssm.state_order == 1)
            return np.array([self.__getitem__(k) for k in vel_states]) 
        else:
            return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self,key,value)

    def predict(self, neural_obs, assist_level=0.0, weighted_avg_lfc=False, **kwargs):
        return self.filt(neural_obs, assist_level, **kwargs)

    def init_from_task(self,**kwargs):
        pass


class RidgeFilt(object):
    def __init__(self, ridge_regression_model):
        self.ridge_model = ridge_regression_model
        self.model_attrs = []
        self.thumb_assist = tuple((False, np.nan))
        self.last_state = None

    def get_mean(self):
        return np.array(self.state.mean).ravel()

    def _init_state(self, init_state=None,**kwargs):
        if init_state is None:
            init_state = np.zeros((15, 1)) 
        self.state = State(init_state)

        if hasattr(self, 'prev_drift_corr'):
            self.drift_corr = self.prev_drift_corr.copy()
            print 'prev drift corr', np.mean(self.prev_drift_corr)
        else:
            #Number of states: 
            self.drift_corr = np.mat(np.zeros_like(self.state.mean))
            self.prev_drift_corr = np.mat(np.zeros_like(self.state.mean)) 

        self.vel_ix = np.arange(7, 14)

    def __call__(self, obs, assist_level, **kwargs):

        x = self.ridge_model.predict(obs.T)
        self.state.mean = np.vstack((x.T, np.array([[1]])))
        
        # Calculate new drift if needed: 
        self.drift_corr[self.vel_ix] = self.drift_corr[self.vel_ix]*self.drift_rho + self.state.mean[self.vel_ix]*(1 - self.drift_rho)
        self.prev_drift_corr = self.drift_corr.copy()

        # Correct for drift: 
        decoded_vel = self.state.mean.copy()
        self.state.mean[self.vel_ix] = decoded_vel[self.vel_ix] - self.drift_corr[self.vel_ix]

        if self.task_state == 'prep':
            if self.last_state != 'prep':
                # Start a new buffer for the task state: 
                self.buff = [self.state.mean]
            else:
                self.buff.append(self.state.mean)

        elif self.task_state == 'target':            
            # Trigger velocity is the mean of the previous buffer 
            self.state.mean = self.scale_factor*np.mean(np.hstack((self.buff)), axis=1)[:, np.newaxis]

        # update last-state for next iteration: 
        self.last_state = self.task_state
        
        # add assist: 
        if np.any(assist_level) > 0:
            x_assist = kwargs.pop('x_assist')
            thumb = [self.state.mean[10], x_assist[10]]
            if type(assist_level) is np.ndarray:
                tmp = np.zeros((len(self.state.mean), 1))
                assist_level_ix = kwargs['assist_level_ix']
                for ia, al in enumerate(assist_level):
                    tmp[assist_level_ix[ia]] = (1.-float(al))*self.state.mean[assist_level_ix[ia]] + float(al)*x_assist[assist_level_ix[ia]]
                    self.state.mean[assist_level_ix[ia]] = np.mat(tmp[assist_level_ix[ia]])                    
            else:
                self.state.mean = (1-assist_level)*self.state.mean + assist_level * x_assist
        else:
            try:
                x_assist = kwargs.pop('x_assist')
                thumb = [self.state.mean[10], x_assist[10]]
            except:
                thumb = [self.state.mean[10], 0.]

        # Set thumb assist differently if needed: 
        if self.thumb_assist[0]:
            self.state.mean[10] = (1.-float(self.thumb_assist[1]))*thumb[0] + (float(self.thumb_assist[1])*thumb[1])
        return self.state.mean

    def _pickle_init(self):
        pass