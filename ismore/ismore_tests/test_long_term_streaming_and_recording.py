''' 
Method to test Armassist sampling rates, ReHand sampling rates, 
Blackrock Sampling Rates, and HDF file accuracty of all of the above
'''

from features.arduino_features import BlackrockSerialDIORowByte
from features.blackrock_features import BlackrockBMI
from features.hdf_features import SaveHDF
from riglib import experiment
from ismore import bmi_ismoretasks 
from ismore import settings
from ismore.brainamp_features import SimBrainAmpData
from ismore.ismore_tests import test_clda

import os, pickle
from db.tracker import models
from riglib.dio import parse
import tables
import numpy as np
import matplotlib.pyplot as plt


def run_task(ntargs):
    start_pos = settings.starting_pos
    targets = np.zeros((ntargs, 7, 2)) + start_pos[np.newaxis, :, np.newaxis]
    Task = experiment.make(bmi_ismoretasks.BMIControl, [SaveHDF, BlackrockSerialDIORowByte, BlackrockBMI, SimBrainAmpData])
    Task.pre_init()
    kwargs=dict(assist_level_time=400., assist_level=(1.,1.), timeout_time=60.)
    task = Task(targets, plant_type=plant_type, **kwargs)
    task.decoder = pickle.load(open('/storage/decoders/'))
    task.run_sync()
    pnm = test_clda.save_dec_enc(task, pref='enc_')
    return pnm, task.blackrock_file, decoder.units


def compare_BR_and_HDF_file(hdf_file, blackrock_files, decoder_file):
    # Open HDF File
    hdf = tables.openFile(hdf_file)

    # Open Blackrock file / Parse DIO
    brf = [b for b in blackrock_files if '.nev' in b]
    try:
        nev_hdf = tables.openFile(brf[0]+'.hdf')
        make_nev_hdf = False
    except:
        make_nev_hdf = True

    if make_nev_hdf:
        try:
            length, units = models.parse_blackrock_file(brf[0], 0, None)
        except:
            pass
        nev_hdf = tables.openFile(brf[0]+'.hdf')
    
    path = 'channel/digital00001/digital_set'
    ts = nev_hdf.root.channel.digital0001.digital_set[:]['TimeStamp']
    msgs = nev_hdf.root.channel.digital0001.digital_set[:]['Value']
    msgtype = np.right_shift(np.bitwise_and(msgs, parse.msgtype_mask), 8).astype(np.uint8)
    
    # auxdata = np.right_shift(np.bitwise_and(msgs, auxdata_mask), 8).astype(np.uint8)
    auxdata = np.right_shift(np.bitwise_and(msgs, parse.auxdata_mask), 8+3).astype(np.uint8)
    rawdata = np.bitwise_and(msgs, parse.rawdata_mask)
    
    # data is an N x 4 matrix that will be the argument to parse.registrations()
    dio_data = np.vstack([ts, msgtype, auxdata, rawdata]).T
    
    # get system registrations
    dio_registrations = parse.registrations(dio_data)
    dio_rows = parse.rowbyte(dio_data)

    # Make sure there are same amount of HDF rows and blackrock rows for: 
    # Task, Armassist, rehand
    for key in ['task', 'armassist', 'rehand']:
        task_sys_key = [k for i , (k, v) in enumerate(dio_registrations.items()) if v[0] == key]
        if len(task_sys_key) > 0:
            n_hdf_task = len(getattr(hdf.root, key))
            n_nev_task = dio_rows[task_sys_key[0]].shape[0]
            print('%s: # of HDF file rows: %d, Number of NEV file rows: %d' %(key, n_hdf_task, n_nev_task))
        else:
            print ' no key: ', key

    # Make sure Spike counts in in task table matches spike counts in blackrock file
    #for iu, un in enumerate(units):
    
    # Make sure accumulated spike counts in in task table matches spike counts in blackrock file
    if hasattr(hdf.root.task, 'spike_counts'):
        task_spike_counts = hdf.root.task[:]['spike_counts']
        nev_spike_counts = np.zeros_like(task_spike_counts)

        n_hdf_rows = np.min([2400, len(hdf.root.task)])
        task_sys_key = [k for i , (k, v) in enumerate(dio_registrations.items()) if v[0] == 'task']
        nev_task = dio_rows[task_sys_key[0]]

        decoder = pickle.load(open(decoder_file))
        channels = np.unique(decoder.units[:, 0])
        units = {}

        for c in [channels[0]]:
            ic = np.nonzero(decoder.units[:, 0]==c)[0]
            units[c] = [decoder.units[ic, 1], ic]

            c_str = 'channel'+str(c).zfill(5)
            tb = getattr(nev_hdf.root.channel, c_str)
            un = tb.spike_set[:]['Unit']

            for i, (iu, idx) in enumerate(zip(units[c][0], units[c][1])):
                if i == 0:
                    iu = units[c][0][i]
                    idx = units[c][1][i]
                    ts = tb.spike_set[:]['TimeStamp']
                    print 'idx: ', idx
                    for n, ni in enumerate(range(0, n_hdf_rows-1)):
                        if ni % 10000 == 0:
                            print 'row: ', n
                        assert nev_task[ni, 1] == ni%256
                        
                        ts0 = nev_task[ni-2, 0]
                        ts1 = nev_task[ni-1, 0]
                        tr = np.logical_and.reduce((ts <= ts1, ts >= ts0, un == iu))
                        nev_spike_counts[ni, idx, 0] = np.sum(tr)
        # Plot: 

    else:
        print 'not a BMI task, spike counts are not stored ' 
        
def test_BR_file_for_consistent_100Hz_AINP(blackrock_nsx_file):
    hdf_nsx = blackrock_nsx_file+'.hdf'
    if not os.path.isfile(hdf_nsx):
        _, _ = models.parse_blackrock_file(None, [blackrock_nsx_file], None, nsx_chan=129)
    hdf = tables.openFile(hdf_nsx)

    # analog input is channel 129: 
    ts = hdf.root.channel.TimeStamp[:]
    ainp = hdf.root.channel.channel00129.Value[:]
    d_ainp = np.diff(ainp)
    del ainp
    ts_ix = np.nonzero(np.abs(d_ainp) > 500)[0]
    del d_ainp

    ts_pulse_5ms = ts[ts_ix]
    d_ts = np.diff(ts_pulse_5ms)
    del ts_pulse_5ms

    # Remove points that are not > 2ms
    ix = np.nonzero(d_ts > 0.002)[0]
    
    #plt.plot(ts[ts_ix[ix]]/60., d_ts[ix])
    plt.hist(d_ts[ix])
    N, BINS, _ = plt.hist(d_ts[ix])
    #plt.xlabel(' Time, min.')
    plt.xlabel(' Time between each pulse')
    plt.show()

    ### See when you get non-0.005 events ###
    x = np.digitize(d_ts[ix], BINS)
    x_, _ = scipy.stats.mode(x)
    
    t = ts[ts_ix[ix]]/60.
    f, ax = plt.subplots()
    ax.plot(t, x)
    ax.set_xlabel(' Time, min.')
    ax.set_ylabel(' Bin # ')
    ax.set_title('Common Bin: '+str(x_) + ' = ' +str(BINS[x_-1]) + ' - ' + str(BINS[x_]))
    print BINS

def units_from_NEV_HDF_file(hdf):
    units = []
    for c in range(1, 97):
        c_str = 'channel'+str(c).zfill(5)
        tb = getattr(hdf.root.channel, c_str)
        u = np.unique(tb.spike_set[:]['Unit'])

        for iu in u:
            units.append([c, iu])
    print 'Number of Units in NEV file: ', len(units)


