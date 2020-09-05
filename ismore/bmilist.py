'''
Lookup table for features, generators and tasks for experiments
'''

import numpy as np
from riglib import calibrations, bmi
from riglib.bmi.bmi import BMI, Decoder
from riglib.bmi import state_space_models
from riglib.bmi import train

bmi_algorithms = dict(
    KFDecoder=bmi.train.train_KFDecoder,
    KFDecoderDrift = bmi.train.train_KFDecoderDrift,
    #PPFDecoder=bmi.train.train_PPFDecoder,
)

bmi_training_pos_vars = [
    'plant_pos',  # used for ibmi tasks,
    'cursor'
]

#################################
##### State-space models for BMIs
#################################
from ismore.ismore_bmi_lib import StateSpaceArmAssist, StateSpaceReHand, StateSpaceIsMore
armassist_state_space = StateSpaceArmAssist()
rehand_state_space = StateSpaceReHand()
ismore_state_space = StateSpaceIsMore()

from riglib.bmi.state_space_models import offset_state, State
endpt_2D_states = [State('hand_px', stochastic=False, drives_obs=False, min_val = -25, max_val=25, order=0),
                   State('hand_py', stochastic=False, drives_obs=False, min_val = 0, max_val=0, order=0),
                   State('hand_pz', stochastic=False, drives_obs=False, min_val = -14, max_val=14, order=0),
                   State('hand_vx', stochastic=True, drives_obs=True, order=1),
                   State('hand_vy', stochastic=False, drives_obs=False, order=1),
                   State('hand_vz', stochastic=True, drives_obs=True, order=1),
                   offset_state]

endpt_2D_state_space = state_space_models.LinearVelocityStateSpace(endpt_2D_states)

bmi_state_space_models=dict(
    Armassist=armassist_state_space,
    Rehand=rehand_state_space,
    ISMORE=ismore_state_space,
    Endpt2D=state_space_models.LinearVelocityStateSpace(endpt_2D_states)
)

extractors = dict(
    spikecounts = bmi.extractor.BinnedSpikeCountsExtractor,
    LFPpowerMTM = bmi.extractor.LFPMTMPowerExtractor,
    LFPpowerBPF = bmi.extractor.LFPButterBPFPowerExtractor,
)

default_extractor = "spikecounts"

bmi_update_rates = [10, 20, 30, 60, 120, 180]

kin_extractors = dict(
	pos_vel=train.get_plant_pos_vel,
)

zscores = ['True', 'False']