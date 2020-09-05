import ismore.bmi_ismoretasks as bmi_ismoretasks
from riglib import experiment
from features.hdf_features import SaveHDF
from features.arduino_features import PlexonSerialDIORowByte
import tables
import matplotlib.pyplot as plt
import numpy as np

def test_arduino_dio():
    #Run SimBMIControl -- make sure doesn't inherit from SimClockTime or anything that speeds up BMI loop times
    Task = experiment.make(bmi_ismoretasks.SimBMIControl, [SaveHDF, PlexonSerialDIORowByte])
    targets = bmi_ismoretasks.SimBMIControl.ismore_simple(length=5)

    kwargs=dict(assist_level_time=400., assist_level=(1.,1.),session_length=600.,
        half_life=(20., 120), half_life_time = 400., timeout_time=60.)

    task = Task(targets, plant_type="IsMore", **kwargs)
    task.pre_init(saveid=1)
    task.run_sync()
    hdf_name = save_hdf(task)

    parse_reg(task.data_files,  hdf_name )

def save_hdf(task, pref='is_'):
    import datetime 
    ct = datetime.datetime.now()
    pnm = '/home/lab/code/ismore/ismore_tests/sim_data/'+pref + ct.strftime("%m%d%y_%H%M") + '.pkl'
    new_hdf = pnm[:-4]+'.hdf'

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
    return new_hdf


from riglib.dio import parse
from plexon import plexfile


def parse_reg(plx_fname, hdf_name):
    plx = plexfile.openFile(plx_fname)
    hdf = tables.openFile(hdf_name)

    # Get the list of all the systems registered in the neural data file
    events = plx.events[:].data
    reg = parse.registrations(events)
    rows = parse.rowbyte(events)

    syskey = None
    f, ax= plt.subplots(nrows=len(reg.items()))

    for i_s, system in enumerate(reg.items()):
        plx_timestamps = np.diff(rows[system[0]][:,0])
        tab = hdf.getNode('/'+system[1][0])

        if system[1][0]=='task':
            hdf_ts = tab[:]['loop_time']
        else:
            hdf_ts = np.diff(np.squeeze(tab[:]['ts']))

        ax[i_s].plot(plx_timestamps, label='plx')
        ax[i_s].plot(hdf_ts, label='hdf')
        ax[i_s].set_title(system[1][0])
        ax[i_s].legend()
