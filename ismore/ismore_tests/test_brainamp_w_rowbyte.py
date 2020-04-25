from ismore.noninvasive.exg_tasks import SimEMGEndPointMovement
from features.hdf_features import SaveHDF
from ismore.brainamp_features import SimBrainAmpData
from features.arduino_features import PlexonSerialDIORowByte
from riglib import experiment
from ismore import ismoretasks
import pandas as pd
from ismore import bmi_ismoretasks
import numpy as np

def yield_task():
    Task = experiment.make(SimEMGEndPointMovement, [SaveHDF, PlexonSerialDIORowByte, SimBrainAmpData ])
    Task.pre_init()
    targets =  ismoretasks.NonInvasiveBase.B1_targets()
    targets = targets[:4]
    plant_type = 'IsMore'
    task = Task(targets, plant_type=plant_type)
    return task


def test_task():
    Task = experiment.make(SimEMGEndPointMovement, [SaveHDF, PlexonSerialDIORowByte, SimBrainAmpData])
    Task.pre_init()
    targets =  ismoretasks.NonInvasiveBase.B1_targets()
    targets = targets[:1]
    plant_type = 'IsMore'
    task = Task(targets, plant_type=plant_type)
    task.init()
    task.run()
    return task

def save_task(task):
    import datetime
    ct = datetime.datetime.now()
    #pnm = '/Users/preeyakhanna/ismore/ismore_tests/sim_data/'+pref + ct.strftime("%Y%m%d_%H_%M_%S") + '.pkl'
    new_hdf = '/home/lab/code/ismore/ismore_tests/sim_data/'+ 'brain-amp-test-' + ct.strftime("%m%d%y_%H%M") + '.hdf'
    
    import shutil
    f = open(task.h5file.name)
    f.close()

    #Wait 
    import time
    time.sleep(1.)

    #Wait after HDF cleaned up
    task.cleanup_hdf()
    import time
    time.sleep(1.)

    #Copy temp file to actual desired location
    shutil.copy(task.h5file.name, new_hdf)
    f = open(new_hdf)
    f.close()

def parse_plx(plx_fname):
    from plexon import plexfile
    plx = plexfile.openFile(plx_fname)
    events = plx.events[:].data

    from riglib.dio import parse
    reg = parse.registrations(events)
    rowbyte_data = parse.rowbyte(events)
    return events, reg, rowbyte_data
