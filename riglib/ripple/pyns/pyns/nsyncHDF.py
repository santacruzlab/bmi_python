# Created on June 3, 2021
# @author: Samantha R. Santacruz
"""Class related to extracting HDF row numbers, timing information, and other DIO events from .ns5 files.
"""
from os import path as ospath
import numpy as np
import scipy as sp
from scipy import io
from riglib.ripple.pyns.pyns.nsexceptions import NeuroshareError, NSReturnTypes
#import riglib.ripple.pyns.pyns.nsparser
from riglib.ripple.pyns.pyns.nsparser import ParserFactory
from riglib.ripple.pyns.pyns.nsentity import AnalogEntity, SegmentEntity, EntityType, EventEntity, NeuralEntity
from riglib.blackrock.brpylib import NsxFile


class nsyncHDF:
    """General class used to extract non-neural information from Ripple files
    to synchronize with behavioral data saved in linked HDF files.  
    """
    def __init__(self, filename):
        """Initialize new File instance.
        Parameters:
        filename -- relative path to wanted .ns5 file
        """
        self.name = ospath.basename(filename)[:-4]
        self.path = ospath.dirname(filename)

        # Analogsignals for digital events
        # Naming convention
        # 0 - 3  : SMA 1 - 4
        # 4 - 27 : Pin 1 - 24
        # 28 - 29: Audio 1 - 2 
        # Here we use Pin 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19 (based on Arduino setup)
        self.pins_util = np.array([1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19]) + 3

        if filename[-4:]=='.ns5':
            self.nsfile = NsxFile(filename)
            self.output = self.nsfile.getdata() 
        else:
            raise Exception('Error: Not an .ns5 file')


    def extract_row_numbers(self):
        """Extract message type and row numbers (mod 256) saved in Ripple .ns5 file.
        Parameters:
        Return:
        MSGTYPE -- array of ints containing message types
        ROWNUMB -- array of ints containing row numbers mod 256
        """

        signals = self.output['data']
        msgtype = signals[self.pins_util[8:], :]
        rownum = signals[self.pins_util[:8], :]
        # Convert to 0 or 1 integers (0 ~ 5000 mV from the recordings)
        msgtype = np.flip((msgtype > 2500).astype(int), axis = 0)
        rownum = np.flip((rownum > 2500).astype(int), axis = 0)

        # Convert the binary digits into arrays
        MSGTYPE = np.zeros(msgtype.shape[1])
        ROWNUMB = np.zeros(rownum.shape[1])
        for tp in range(MSGTYPE.shape[0]):
            MSGTYPE[tp] = int(''.join(str(i) for i in msgtype[:,tp]), 2)
            ROWNUMB[tp] = int(''.join(str(i) for i in rownum[:,tp]), 2)

        return MSGTYPE, ROWNUMB


    def make_syncHDF_file(self):
        """Create .mat synchronization file for synchronizing Ripple and behavioral data (saved in .hdf file).

        Parameters:
        Return:
        """

        # Create dictionary to store synchronization data
        hdf_times = dict()
        hdf_times['row_number'] = []          # PyTable row number
        hdf_times['ripple_samplenumber'] = []    # Corresponding Ripple sample number
        hdf_times['ripple_dio_samplerate'] = []  # Sampling frequency of DIO signal recorded by Ripple system
        hdf_times['ripple_recording_start'] = [] # Ripple sample number when behavior recording begins

        signals = self.output['data']
        fs = self.output['samp_per_s']

        rstart = (signals[22 + 3,:] > 2500).astype(int)
        strobe = (signals[20 + 3,:] > 2500).astype(int)

        MSGTYPE, ROWNUMB = self.extract_row_numbers()

        find_recording_start = np.ravel(np.nonzero(strobe))[0]
        find_data_rows = np.logical_and(np.ravel(np.equal(MSGTYPE,13)),np.ravel(np.greater(strobe,0))) 	
        find_data_rows_ind = np.ravel(np.nonzero(find_data_rows))

        rows = ROWNUMB[find_data_rows_ind]	  # row numbers (mod 256)

        prev_row = rows[0] 	# placeholder variable for previous row number
        counter = 0 		# counter for number of cycles (i.e. number of times we wrap around from 255 to 0) in hdf row numbers

        for ind in range(1,len(rows)):
            row = rows[ind]
            cycle = (row < prev_row) # row counter has cycled when the current row number is less than the previous
            counter += cycle
            rows[ind] = counter*256 + row
            prev_row = row    

        # Load data into dictionary
        hdf_times['row_number'] = rows
        hdf_times['ripple_samplenumber'] = find_data_rows_ind
        hdf_times['ripple_recording_start'] = find_recording_start
        hdf_times['ripple_dio_samplerate'] = fs

        # Save syncing data as .mat file
        mat_filename = self.path + '/' + self.name + '_syncHDF.mat'
        print(mat_filename)
        sp.io.savemat(mat_filename,hdf_times)

        return hdf_times