from features.arduino_features import BlackrockSerialDIORowByte
from features.hdf_features import SaveHDF
from riglib import experiment
from ismore import bmi_ismoretasks 

Task = experiment.make(bmi_ismoretasks.ManualControl, [BlackrockSerialDIORowByte, SaveHDF])
targets = bmi_ismoretasks.ManualControl.ismore_simple()
task = Task(targets, plant_type = 'IsMore', session_length=10.)

task.init()
task.run()