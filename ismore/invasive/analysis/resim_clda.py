''' Re-simulate CLDA with different intention estimators'''

from ismore.ismore_bmi_lib import LQRController_accel_limit_ismore, LQRController_accel_limit_ismore_rester, StateSpaceIsMore
from ismore import ismore_bmi_lib
from riglib.bmi.clda import FeedbackControllerLearner
import numpy as np

ssm = StateSpaceIsMore()
A, B, _ = ssm.get_ssm_matrices()
R = 1e6 * np.mat(np.diag([1., 1., 1., 1., 1., 1., 1.])) #original: 1e6
Q3 = np.mat(np.diag([70., 70., 50.,70., 70., 70.,70., 10.,10.,10.,10.,10.,10.,10.,0])) 
ismore_controller_fast = LQRController_accel_limit_ismore(A, B, Q3, R)
ismore_controller_fast_rest = LQRController_accel_limit_ismore_rester(A, B, Q3, R)


class fast_assist_learner(FeedbackControllerLearner):
    def __init__(self, batch_size, *args, **kwargs):
        super(fast_assist_learner, self).__init__(batch_size, ismore_controller_fast)

class fast_assist_learner_rest(FeedbackControllerLearner):
    def __init__(self, batch_size, *args, **kwargs):
        super(fast_assist_learner_rest, self).__init__(batch_size, ismore_controller_fast_rest)




def verify_ik_estimates(hdf, batch_time=.1):

	#learner1 = ismore_bmi_lib.OFC_LEARNER_CLS_DICT['IsMore'](batch_time/.1)
	learner1 = fast_assist_learner(batch_time/.1)
	learner2 = fast_assist_learner_rest(batch_time/.1)

	sc = hdf.root.task[:]['spike_counts']
	T, nUnits, _ = sc.shape

	target = np.hstack(( hdf.root.task[:]['target_pos'], np.ones((T, 1))))
	plant_state = np.hstack((hdf.root.task[:]['plant_pos'], hdf.root.task[:]['plant_vel']))
	decoder_output = hdf.root.task[:]['decoder_state']
	task_state = 'wait'
	msgs = hdf.root.task_msgs[:]

	ik_est1 = []
	ik_est2 = []

	accum_spks = np.zeros((nUnits, ))

	T = len(hdf.root.task)
	for t in np.arange(T):
		if t in msgs['time']:
			ix = np.nonzero(msgs['time']==t)[0]
			task_state = msgs[ix[0]]['msg']

		accum_spks += sc[t, :, 0]

		if t in np.arange(1, T, 2):
			learner1(accum_spks.reshape(-1, 1), decoder_output[t-1, :, 0], target[t, :], 
				decoder_output[t-1, :, 0], task_state)

			learner2(accum_spks.reshape(-1, 1), decoder_output[t-1, :, 0], target[t, :], 
				decoder_output[t-1, :, 0], task_state)

			accum_spks = np.zeros((nUnits, ))



			batch_data1 = learner1.get_batch()
			batch_data2 = learner2.get_batch()

			ik_est1.append(batch_data1['intended_kin'])
			ik_est2.append(batch_data2['intended_kin'])
	return np.array(np.hstack((ik_est1))), np.array(np.hstack((ik_est2)))
