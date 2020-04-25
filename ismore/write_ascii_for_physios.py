''' Install astropy for python 2.7: 
    sudo pip install astropy==2.0.x, (lastest was 2.0.5)
'''
from __future__ import print_function
import astropy
from db import dbfunctions as dbfn
import tables
import numpy as np
from astropy.io import ascii

# Change this to the place you'd like to save / store data
ascii_directory = '/Users/preeyakhanna/Dropbox/Carmena_Lab/Projects/SpainBMI/ismore_analysis/data/ascii/'

def write_ascii_from_te(te_id):
    ''' 
    Method: 
        - take data from supp_HDF and write to ascii file
        - include task data too
    '''
    # Open task entry file 
    tsk = dbfn.TaskEntry(te_id)

    # Supp hdf: 
    supp_hdf = tables.openFile('/storage/supp_hdf/'+tsk.name+'.supp.hdf')
    try:
        ard_hdf = tsk.hdf.root.arduino_imu
        convert_arduino = True
    except:
        convert_arduino = False

    #################################
    ## MAKE SUPP HDF FILE ###
    #################################
    
    # Create new np array w/ just one entry for ts_arrival: 
    channel_names = supp_hdf.root.brainamp.colnames
    dt = [(chan, '<f8') for chan in ['Time'] + channel_names]
    dt = [('Frame', np.uint64)] + dt

    skinny_brainamp = np.array(np.zeros((len(supp_hdf.root.brainamp))), dtype=np.dtype(dt))
    skinny_brainamp[:]['Frame'] = np.arange(len(supp_hdf.root.brainamp))

    # Fix ts_arrival to match: 
    ts = np.arange(len(supp_hdf.root.brainamp))/1000.
    ts0 = supp_hdf.root.brainamp[:][channel_names[1]]['ts_arrival']
    print('inferred number of secs: %d' %ts[-1])
    print(', actual: %d' % int(ts0[-1] - ts0[0]))
    assert np.abs(ts[-1]-(ts0[-1]-ts0[0])) < 1
    skinny_brainamp[:]['Time'] = ts
    
    for chan in channel_names:
        skinny_brainamp[:][chan] = supp_hdf.root.brainamp[:][chan]['data']

    #Write data to ASCII: 
    emg_filename = ascii_directory+tsk.name+'_EMG.emt'
    ascii.write(skinny_brainamp, emg_filename, delimiter=chr(9))

    # Ok, now insert the stuff at the top of the file: 
    add_formated_header(1000, len(skinny_brainamp), 'Emg tracks', 'mV', 14, emg_filename)

    #################################
    ### MAKE ARDUINO ACC DATAFILE ###
    #################################
    if convert_arduino:
        acc_data = np.hstack(( np.arange(len(ard_hdf))[:, np.newaxis],
            ard_hdf[:]['sensors'][:, -1][:, np.newaxis], ard_hdf[:]['sensors'][:, [0, 1, 2]]))

        acc_data[:, 1] = acc_data[:, 1] - acc_data[0, 1]
        labels = ['Time', 'NewAcc3D.X','NewAcc3D.Y','NewAcc3D.Z'] #,'gx (deg/sec)','gy (deg/sec)','gz (deg/sec)']
        dt = [(chan, '<f8') for chan in labels]
        dt = [('Frame', np.uint64)] + dt
        labels = ['Frame']+labels
        acc_dat_table = np.array(np.zeros((len(acc_data))), dtype=np.dtype(dt))

        for i, lab in enumerate(labels):
            acc_dat_table[:][lab] = acc_data[:, i]
        
        #Write data to ASCII: 
        ascii.write(acc_dat_table, ascii_directory+tsk.name+'_ACC.emt', delimiter=chr(9))
        add_formated_header(16, len(acc_dat_table), '3D acceleration tracks', 'm/s^2', 1, ascii_directory+tsk.name+'_ACC.emt')

        ### MAKE ARDUINO GYRO DATA ###
        acc_data = np.hstack(( np.arange(len(ard_hdf))[:, np.newaxis],
            ard_hdf[:]['sensors'][:, -1][:, np.newaxis], ard_hdf[:]['sensors'][:, [3, 4, 5]]))

        acc_data[:, 1] = acc_data[:, 1] - acc_data[0, 1]
        labels = ['Time', 'NewAcc3D.X','NewAcc3D.Y','NewAcc3D.Z'] #,'gx (deg/sec)','gy (deg/sec)','gz (deg/sec)']
        dt = [(chan, '<f8') for chan in labels]
        dt = [('Frame', np.uint64)] + dt
        labels = ['Frame']+labels
        
        acc_dat_table = np.array(np.zeros((len(acc_data))), dtype=np.dtype(dt))

        for i, lab in enumerate(labels):
            acc_dat_table[:][lab] = acc_data[:, i]
        
        #Write data to ASCII: 
        ascii.write(acc_dat_table, ascii_directory+tsk.name+'_GYRO.emt', delimiter=chr(9))
        add_formated_header(16, len(acc_dat_table), '3D acceleration tracks', 'm/s^2', 1, ascii_directory+tsk.name+'_GYRO.emt')

def add_formated_header(freq, frames, type0, unit, tracks, emg_filename):
    header_lines = [np.array([66, 84, 83, 32, 65, 83, 67, 73, 73, 32, 102, 111, 114, 109, 97, 116, 13, 10]),
    np.array([13, 10]),
    np.hstack(( [84, 121, 112, 101,  58,  32,  32,  32,  32,  32,  32,  32,  32, 32,   9],  [ord(i) for i in type0], [13, 10] )), # track name
    np.hstack(( [77, 101,  97, 115, 117, 114, 101,  32, 117, 110, 105, 116,  58, 32, 9], [ord(i) for i in unit], [13, 10] )), # measure unit
    np.array([13, 10]),
    np.hstack(( [84, 114, 97, 99, 107, 115, 58, 32, 32, 32, 32, 32, 32, 32, 9],  [ord(i) for i in str(tracks)], [13, 10] )) , # number of tracks
    np.hstack(( [70, 114, 101, 113, 117, 101, 110,  99, 121,  58,  32,  32,  32, 32,   9],  [ord(i) for i in str(freq)],  [32,  72, 122,  13,  10] )), # frequency in hz
    np.hstack(( [70, 114,  97, 109, 101, 115,  58,  32,  32,  32,  32,  32,  32, 32,   9],  [ord(i) for i in str(frames)],  [13,  10] )), # number of frames
    np.array([83, 116,  97, 114, 116,  32, 116, 105, 109, 101,  58,  32,  32, 32,   9,  48,  46,  48,  48,  48,  13,  10]), # start time = 0
    np.array([13, 10]) ]

    header = np.hstack((header_lines))
    header_string = [chr(i) for i in header]
    header_string = ''.join(header_string)

    with open(emg_filename, 'r+') as fh:
        lines = fh.readlines()
        fh.seek(0)
        lines.insert(0, header_string)
        fh.writelines(lines)

    # Now go through each line and add \t\r\n to the end of each line: 
    with open(emg_filename, 'r+') as fh:
        lines = fh.readlines()
        fh.seek(0)
        for i in range(10, len(lines)):
            lines[i] = lines[i].replace('\n','\t\r\n')
        fh.writelines(lines)

def test_files():
    file1 = '/Users/preeyakhanna/Dropbox/Carmena_Lab/Projects/SpainBMI/ismore_analysis/data/ascii/hud120180409_09_te12980_EMG.emt'
    file2 = '/Users/preeyakhanna/Downloads/rawEMGtracks.emt'
    B2 = []
    B1 = []

    with open(file1, 'r') as F:
        lines1 = F.readlines()
        for line in lines1:
            B1.append([ord(str(i)) for i in line])
    
    with open(file2, 'r') as F:
        lines2 = F.readlines()
        for line in lines2:
            B2.append([ord(str(i)) for i in line])

