import ismore.invasive.bmi_ismoretasks as bmi_ismoretasks
import ismore.invasive.patient_display as patient_display
import ismore.invasive.bmitasks_w_display as bmitasks_w_display
from riglib import experiment
from features.hdf_features import SaveHDF
import tables
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import time
#import seaborn
from ismore_tests import test_clda

Task = experiment.make(bmitasks_w_display.VisualFeedbackWithDisplay, [SaveHDF])
targets = bmitasks_w_display.VisualFeedbackWithDisplay.armassist_w_disp()

plant_type = 'ArmAssist'
fnm = '/Users/preeyakhanna/ismore/ismore_tests/sim_data/aa_20160405_15_12_03.pkl'

kwargs=dict(assist_level_time=50, assist_level=(1.,1.),session_length=100,
    timeout_time=60., enc_path=fnm, dec_path=fnm)
#dec = pickle.load(open(fnm))
task_is = Task(targets, plant_type=plant_type, **kwargs)
#task_is.decoder = dec.corresp_dec
task_is.run_sync()
pnm = test_clda.save_dec_enc(task_is, pref='is_')




# Task = experiment.make(patient_display.vfb_w_disp, [SaveHDF])
# targets = bmi_ismoretasks.SimBMIControl.armassist_simple(length=100)
# targets = patient_display.vfb_w_disp.armassist_w_word()
# plant_type = 'ArmAssist'
# fnm = '/Users/preeyakhanna/ismore/ismore_tests/sim_data/aa_20160405_15_12_03.pkl'
# kwargs=dict(assist_level_time=50, assist_level=(1.,1.),session_length=100,
#     timeout_time=60., enc_path=fnm, dec_path=fnm)
# #dec = pickle.load(open(fnm))
# task_is = Task(targets, plant_type=plant_type, **kwargs)
# #task_is.decoder = dec.corresp_dec
# task_is.run_sync()
# pnm = test_clda.save_dec_enc(task_is, pref='is_')



  
# Task = experiment.make(bmi_ismoretasks.SimBMIControlwRating, [SaveHDF])
# targets = bmi_ismoretasks.SimBMIControl.armassist_simple(length=100)
# plant_type = 'ArmAssist'
# fnm = '/Users/preeyakhanna/ismore/ismore_tests/sim_data/aa_20160405_15_12_03.pkl'
# kwargs=dict(assist_level_time=50, assist_level=(1.,1.),session_length=100,
#     timeout_time=60., enc_path=fnm, dec_path=fnm)

# task_is = Task(targets, plant_type=plant_type, **kwargs)
# task_is.run_sync()



Task = experiment.make(bmi_ismoretasks.SimBMIControl, [SaveHDF])
targets = bmi_ismoretasks.SimBMIControl.armassist_simple(length=100)
plant_type = 'ArmAssist'
fnm = '/Users/preeyakhanna/ismore/ismore_tests/sim_data/aa_20160405_15_12_03.pkl'
kwargs=dict(assist_level_time=50, assist_level=(1.,0.),session_length=100,
    timeout_time=60., enc_path=fnm, dec_path=fnm)

task_is = Task(targets, plant_type=plant_type, **kwargs)
task_is.run_sync()
pnm = test_clda.save_dec_enc(task_is, pref='is_')



Task = experiment.make(bmi_ismoretasks.BMIControl_with_Patient, [SaveHDF])
targets = bmi_ismoretasks.SimBMIControl.armassist_simple(length=100)
plant_type = 'ArmAssist'
fnm = '/Users/preeyakhanna/ismore/ismore_tests/sim_data/aa_20160405_15_12_03.pkl'
kwargs=dict(assist_level_time=50, assist_level=(1.,0.),session_length=100,
    timeout_time=60., enc_path=fnm, dec_path=fnm)

task_is = Task(targets, plant_type=plant_type, **kwargs)
task_is.run_sync()
pnm = test_clda.save_dec_enc(task_is, pref='is_')







