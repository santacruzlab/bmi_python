from ismore import ismoretasks
from riglib import experiment
from features.hdf_features import SaveHDF
from ismore.invasive import bmi_ismoretasks


#Playback trajectories
Task = experiment.make(ismoretasks.PlaybackTrajectories, [SaveHDF])
targets = bmi_ismoretasks.SimBMIControl.armassist_simple(length=1)
task = Task(targets, plant_type="ArmAssist", assist_level=(1., 1.))
task.run_sync()

#Endpoint to endpoint movement: No armassist data has arrived at all
Task = experiment.make(ismoretasks.EndPointMovement, [SaveHDF])
targets = bmi_ismoretasks.SimBMIControl.armassist_simple(length=1)
task = Task(targets, plant_type="ArmAssist", assist_level=(1., 1.))

#Visual Feedback: 
#In separate process run armassist_app.py
Task = experiment.make(bmi_ismoretasks.VisualFeedback, [SaveHDF])
targets = bmi_ismoretasks.SimBMIControl.armassist_simple(length=10)
task = Task(targets, plant_type="ArmAssist", assist_level=(1., 1.))
task.run_sync()


# CLDA control
Task = experiment.make(bmi_ismoretasks.SimCLDAControl, [SaveHDF])
targets = bmi_ismoretasks.SimBMIControl.armassist_simple(length=100)
plant_type = 'ArmAssist'
kwargs=dict(assist_level_time=400., assist_level=(1.,0.),session_length=20.,
        half_life=(20., 120), half_life_time = 400., timeout_time=60.)
task = Task(targets, plant_type=plant_type, **kwargs)
task.run_sync()
