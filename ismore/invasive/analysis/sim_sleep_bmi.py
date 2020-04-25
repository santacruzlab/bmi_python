import ismore.invasive.bmi_ismoretasks as bmi_ismoretasks
from riglib import experiment
from features.hdf_features import SaveHDF
from ismore.brainamp_features import SimBrainAmpData
import datetime
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

targets = bmi_ismoretasks.SimBMIControlReplayFile.B1_targets(length=100, green=1, red=0, blue=0, brown=0)
plant_type = 'IsMore'

kwargs=dict(assist_level_time=400., assist_level=(1.,1.),session_length=20,
    timeout_time=15., replay_te =11585)
Task = experiment.make(bmi_ismoretasks.SimBMIControl, [SaveHDF])
task = Task(targets, plant_type=plant_type, **kwargs)
task.run_sync()
#    fnm = save_dec_enc(task, pref='aa_')