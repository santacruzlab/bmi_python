# Created on Apr 23, 2012
# @author: Elliott L. Barcikowski
'''pyns.nsparser - parser header and data for .nev, .nsx2.1 and .nsx2.2 files.

The pyns.nsparser module provides the functionality for all the unlying 
reading of data for the pyns packet.  Any use of the pyns packet will make
use of these classes behind the scenes.  Advanced uses may find using these
classes directly as an easy way to get data out of .nev and .nsx files 
easily.

The heavy lifting is done with the classes NevParser, Nsx21Parser, and
Nsx22Parser.  These maybe interfaced with through the factory 
ParserFactory.

The header formats and sizes are found for all headers and data packets.
These are found in the top of the files as ????????_FORMAT and 
????????_SIZE constants.  namedtuples are provided to return the header
and packet data.
''' 
from collections import namedtuple
import struct
import os
import sys
from datetime import datetime
try:
    import numpy
except ImportError as msg:
    sys.stderr.write("Could not find numpy module.  Required for pyns."\
                     "Easily installed at http://www.pythonxy.com\n")
    raise ImportError(*msg)

from nsexceptions import NeuroshareError, NSReturnTypes

#def get_bits(byte, nbytes=8):
#    """utility fucntion that returns a list of True and False for all 
#    the non-zeros bits.  The list will have the same number of elements 
#    as nbytes.  This function is useful for the "Packet Insertion Reason" 
#    in the digital event packets 
#    """
#    flagged_bits = []
#    for ibit in xrange(0, nbytes):
#        flagged_bits.append(byte&(1<<ibit)!=0)
#    return flagged_bits

# header and extended header formats for the
# .nev files
NEURALEV_FORMAT = "<8s2BH4I8H32s256sI"
NEUEVWAV_FORMAT = "<8sH2B2H2h2Bf6s"
NEUEVLBL_FORMAT = "<8sh16s6s"
NEUEVFLT_FORMAT = "<8sH2IHIIH2s"
DIGLABEL_FORMAT = "<8s16sB7s"
NSASEXEV_FORMAT = "<8s2BhBhBhBhBhBh6s"
# sizes of headers for .nev files.
# NEURALEV size is 336 bytes
# All extended headers are 32 bytes long
NEURALEV_SIZE = struct.calcsize(NEURALEV_FORMAT)
NEV_EXT_HEADER_SIZE = struct.calcsize(NEUEVWAV_FORMAT)
# structs for headers and data packets for NEURALEV files
NEURALEV = namedtuple("NEURALEV", 
                      "header_type file_rev_major file_rev_minor " \
                      "file_flags bytes_headers bytes_data_packet timestamp_resolution " \
                      "sample_resolution time_origin " \
                      "application comment n_ext_headers")
NEUEVWAV = namedtuple("NEUEVWAV",
                      "header_type packet_id phys_conn conn_pin dig_factor " \
                      "energy_thres high_thres low_thres number_sorted_units " \
                      "bytes_per_waveform stim_amp_dig_factor")
NEUEVLBL = namedtuple("NEUEVLBL",
                      "header_type packet_id label")
NEUEVFLT = namedtuple("NEUEVFLT", "header_type packet_id " \
                      "high_freq_corner high_freq_order high_filter_type " \
                      "low_freq_corner low_freq_order low_filter_type")
DIGLABEL = namedtuple("DIGLABEL", "header_type label mode")
# struct for digital events in NEV file data packets
# digital events are identified by packed_id == 0
NEVEvent = namedtuple("NEVEvent",
                      "timestamp packet_id reason reserved digital_input " \
                      "input1 input2 input3 input4 input5")
# struct Spike events found in NEV file data packets
# digital events are identified by packet_id > 0
NEVSegment = namedtuple("Segment",
                     "timestamp packet_id unit_class reserved waveform")
# namedtuples for NEURALSG (NSx2.1 files) 
# These files only have one variable length basic header.  "chanel_count" 
# says how many electrodes are taking data.  channel_id will be an array 
# with length channel_count 
# NSx2.1 files have no extended header information 
NEURALSG = namedtuple("NEURALSG", "header_type label period channel_count channel_id")

# namedtuples for NEURALCD files (NSx2.2 files)
# NEURALCD is the basic header for NSx2.2 files
NEURALCD = namedtuple("NEURALCD", "header_type maj_revision min_revision bytes_headers label "\
                      "comment period timestamp_resolution time_origin "\
                      "channel_count")
NEURALCD_FORMAT = "<8s2BI16s256s2I8HI"
NEURALCD_SIZE = struct.calcsize(NEURALCD_FORMAT)
# CC is the extended header for NSx2.2 files
CC = namedtuple("CC", "header_type electrode_id electrode_label phys_conn conn_pin "\
                "min_dig_value max_dig_value min_analog_value max_analog_value "\
                "units high_freq_corner high_freq_order high_filter_type "\
                "low_freq_corner low_freq_order low_filter_type")
CC_FORMAT = "<2sH16s2B4h16s2IH2IH"
CC_SIZE = struct.calcsize(CC_FORMAT)
Nsx22DataPacket = namedtuple("Nsx22DataPacket", "header timestamp n_data_points data_points")
     
def _proc_timestamp_struct(tup):
    """A utility function to convert 8 integers corresponding to Windows 
    SYSTEMTIME to Python datetime class Neuroshare file headers often contain 
    Windows SYSTEMTIME structs
    input: tup holding eight integers
    returns: datetime instance
    """
    return datetime(tup[0], tup[1], tup[3], tup[4],
        tup[5], tup[6], tup[7]*1000)

def ParserFactory(filename):
    """ParserFactory provides the interface to the Parser classes listed
    below and handles opening of and checking the type of the files.  Based
    on the string found in the first few bytes, it returns correct class 
    """
    try:
        fid = open(filename, "rb")
    except:
        raise NeuroshareError(NSReturnTypes.NS_BADFILE,
                              "failed to open {0:s}\n".format(filename))
    file_type = fid.read(8)
    if file_type == "NEURALEV":
        return NevParser(fid)
    elif file_type == "NEURALSG":
        return Nsx21Parser(fid)
    elif file_type == "NEURALCD" or file_type == "NEUCDFLT":
        return Nsx22Parser(fid)
    # failed to find valid file header
    raise NeuroshareError(NSReturnTypes.NS_BADFILE, 
                          "invalid or corrupt file: {0:s}".format(filename)) 
        
class NevParser:
    """Interface to .nev files.  Also for easy reading of nev files and functions to
    read known basic and extended headers, as well as both segment data packets and
    event data packets.  Member data is held to simply reading of headers and data packets.
    """  
    def __init__(self, fid):
        """Open file and store some data that is needed to easily read extended headers
        and data packets.  Note, this class will hold and own the file instance here.  
        I.e., it will close the file when deleted. 
        Parameter:
            fid -- valid file pointer
        """
        self.fid = fid
        # Read the whole NEURALEV header and store a few pieces of data that 
        # are useful for parsing extended headers and data packets
        header = self.get_basic_header()
        # Skip to end of file to determine the size
        self.fid.seek(0, os.SEEK_END)
        self.size = self.fid.tell()
        # return the file pointer to the start of file
        self.fid.seek(0, os.SEEK_SET)
        # how many NEUEVWAV, NEUEVLBL, DIGLABEL, etc. type headers
        self.n_ext_headers = header.n_ext_headers
        # size of NEURALEV header + all extended headers
        self.bytes_headers = header.bytes_headers
        # Length of each data packet, determines the length of spike waveforms
        self.bytes_data_packet = header.bytes_data_packet
        # The number of data packets is just the size of the file minus the size of
        # all the headers divided by the size of one data packet
        self.n_data_packets = (self.size - self.bytes_headers) / self.bytes_data_packet
        # sample resolution is used in a variety of quantities and is often requested
        # for this reason we will store it here so it doesn't have to be looked up repeatedly
        self.timestamp_resolution =  header.sample_resolution
        # based on n_data_packets we can calculate the size of data packets
        # the number of waveform bins is (bytes_data_packet - 8) / 2.  We assume
        # all waveforms will be of type int16
        self.sample_count = (self.bytes_data_packet - 8)/2
        self.data_packet_form = "<IH2B{0:d}h".format(self.sample_count)
        self.data_packet_size = struct.calcsize(self.data_packet_form)
        
    def __del__(self):
        """close the file when we're done with this instance"""
        self.fid.close()
        
    @property
    def file_type(self):
        """Static functiont to return the 8 byte header associated with this
        file.
        """
        return "NEURALEV"
            
    def get_basic_header(self):
        """Read the NEURALEV header from a nev file.
        input: fid id of NEURAL  
        returns: struct containing nev header or packet data or None on failure
        """
        self.fid.seek(0, os.SEEK_SET)
        try: 
            buf = self.fid.read(NEURALEV_SIZE)
            tup = struct.unpack(NEURALEV_FORMAT, buf) 
        except:
            raise NeuroshareError(NSReturnTypes.NS_BADFILE,
                                  "failed reading file")
        if tup[0] != "NEURALEV":
            raise NeuroshareError(NSReturnTypes.NS_BADFILE,
                                  "cannot find NEURALEV header\n")
        # NEURALEV files contains Windows SYSTEMTIME struct.  We want to store
        # this as a Python datetime class
        timestamp = _proc_timestamp_struct(tup[8:16])
        return NEURALEV._make(tup[:8] + (timestamp,) + tup[16:])
    
    def get_extended_headers(self):
        """Generator to loop over all extended headers.  Makes use of 
        the get_extended_header function
        """
        self.fid.seek(NEURALEV_SIZE, os.SEEK_SET)
        for _ in range(0, self.n_ext_headers):
            header = self.get_extended_header()
            yield header
            
    def get_extended_header(self, header_index=None):
        """Return extended header for nev file from current position in
        file.  This should be modified to be position independent 
        with possibly a generator
        """
        # if header_index is specified, just to that absolute position in the
        # file.  If not we try to read the header from the current position.
        # This allows the user to skip to the header they want, but when just
        # looping through all headers we don't need to move the file pointer 
        # each iteration
        if header_index != None:
            if header_index >= self.n_ext_headers or header_index < 0:
                raise NeuroshareError(NSReturnTypes.NS_BADINDEX,
                                      "invalid header index {0:d}".format(header_index))
            position = NEURALEV_SIZE + header_index*NEV_EXT_HEADER_SIZE
            self.fid.seek(position, os.SEEK_SET)
        try:
            buf = self.fid.read(NEV_EXT_HEADER_SIZE)
        except:
            raise NeuroshareError(NSReturnTypes.NS_BADFILE,
                                  "failed on file read")
        header_type = buf[0:8]
        if header_type == "NEUEVWAV":
            data = struct.unpack(NEUEVWAV_FORMAT, buf)
            return NEUEVWAV._make(data[:-1])
        elif header_type == "NEUEVLBL":
            data = struct.unpack(NEUEVLBL_FORMAT, buf)            
            return NEUEVLBL._make(data[:-1])
        elif header_type == "DIGLABEL":
            data = struct.unpack(DIGLABEL_FORMAT, buf)            
            return DIGLABEL._make(data[:-1])
        elif header_type == "NEUEVFLT":
            data = struct.unpack(NEUEVFLT_FORMAT, buf)            
            return NEUEVFLT._make(data[:-1])
        else:
            raise NeuroshareError(NSReturnTypes.NS_BADFILE,
                                  "unknown extended header: {0:s}".format(header_type))
        
    def get_packet_headers(self):
        """Defines an iterator that will return only the timestamps, electrode_id, 
        and unit or reason for spike and digital channels respectively.  This 
        function will read in chunks in hopes to more efficiently (in terms of time)
        open NEV files.
        """
        # TODO: Start here.
        # read a maximum of 1024 packets at once
        
        # move file pointer to the start of data packets
        self.fid.seek(self.bytes_headers, os.SEEK_SET)
        
        max_packet_read = 1024
        remaining_packets = self.n_data_packets
        while remaining_packets > 0:
            # read maximum or to the end of the file
            packets_to_read = min(max_packet_read, remaining_packets)
            bytes_to_read = packets_to_read*self.bytes_data_packet
            # TODO: check for valid read
            buf = self.fid.read(bytes_to_read)
            for packet_index in xrange(0, packets_to_read):
                # Find where the current packet starts in the buffer that
                # that we have read.
                buf_pos = packet_index*self.bytes_data_packet
                # In the case of packet_id == 0 (digital events), unit below is 
                # actually the reason for the digital event to be stored
                (timestamp, packet_id, unit) = struct.unpack("<IHB", buf[buf_pos:buf_pos+7])
                
                yield timestamp, packet_id, unit
                
            remaining_packets -= packets_to_read
            
    def get_data_packets(self):
        """Generator to loop over all data packets.  Makes use the 
        get_data_packets function 
        """
        # skip to the start of the data packets
        self.fid.seek(self.bytes_headers, os.SEEK_SET)
        # loop through packets
        for _ in range(0, self.n_data_packets):
            packet = self.get_data_packet()
            yield packet
            
    def get_data_packet(self, packet_index=None):
        """Return the desired data packet.
        Parameter:
            packet_index -- index of desired data packet, default=None.
                if packet_index: seek to absolute path in file and return packet.
                else: return packet from current point in file. 
        Returns:
            NEVEvent or NEVSegment (depending on type of packet) containing data,
            None on failure. 
        """
        # The length of these packets is constant in a given file 
        # (though my vary file to file).  The first 5 pieces of data
        # are part of a data packet header, giving timestamp and other 
        # values.  The rest is either data from the digital event or
        # the spike waveform
        # Note: Here we are assuming each entry in the waveform is a int16
        
        # If packet_index is specified we seek to the wanted packet from
        # the start of the NEV file.
        if packet_index != None:
            # check bounds of packet_index
            if packet_index >= self.n_data_packets or packet_index < 0:
                raise NeuroshareError(NSReturnTypes.NS_BADINDEX,
                                      "invalid packet index {0:d}".format(packet_index))
            position = self.bytes_headers + packet_index*self.bytes_data_packet
            self.fid.seek(position, os.SEEK_SET)
        # Read current data packet
        try:
            buf = self.fid.read(self.data_packet_size)
            packet_tup = struct.unpack(self.data_packet_form, buf)
        except:
            raise NeuroshareError(NSReturnTypes.NS_BADFILE,
                                  "failed on file read")            
        # We use the packet_id to see which type of class we return
        packet_id = packet_tup[1]
        # We found a digital event, return NEVEvent struct
        # The case of a digital event only the first 6 data elements are relevant.
        # The rest should be zero.
        if packet_id == 0:
            return NEVEvent._make(packet_tup[:10])
        # spike waveform found.  Process waveform and return NEVSegment
        # store the waveform data as 16 bit integers.
        waveform = numpy.array(packet_tup[4:], dtype=numpy.int16)
        
        return NEVSegment._make(packet_tup[:4] + (waveform,))

class Nsx21Parser:
    """Interface to Nsx2.1 files.
    
    Uses the Python struct module to read all the binary data found in the 
    Nsx21 style files.  Holds as member variables a file object and a small 
    amount of header data to allow for easy retrieval. 
    """
    
    def __init__(self, fid):
        """Initialize Nsx21Parser. Some internal data is store from
        basic header to facilitate the reading of data packets.  Note, this 
        class will hold and own the file instance here. I.e., it will close 
        the file when deleted.  
        
        Parameter:
            fid -- valid file pointer
        """  
        self.fid = fid
        self.fid.seek(24, os.SEEK_SET)
        # there are no 2.1 float streams, but the AnalogEntity class will look for this number
        self.is_float = False
        buf = self.fid.read(8)
        (self.period, self.channel_count) = struct.unpack("II", buf)

        # calculate the header_format and header size
        self.header_format = "8s16s2I{0:d}I".format(self.channel_count)
        self.header_size = struct.calcsize(self.header_format)
        
        # skip to the end of the file to find the file size
        self.fid.seek(0, os.SEEK_END)
        self.size = self.fid.tell()
        
        # calculate the total number of bins for each analog entity
        # done just by using the total file size, header size, and 
        # number of electrodes.  Note:  here we assume that the data
        # is found as int16s
        self.n_data_points = (self.size - self.header_size) / 2 / self.channel_count

        # return the file pointer to the start to not confuse other
        # functions and reads of the file
        self.fid.seek(0, os.SEEK_SET)
        
    def __del__(self):
        """close thie file when we're done with this instance"""
        self.fid.close()
                
    @property
    def timestamp_resolution(self):
        """Return timestamp_resolution.  For Nsx2.1 this quantity is not 
        stored.  It will always be 3000.0.
        """
        return 30000.0
    
    @property
    def time_span(self):
        """Return time_span of data in this file.  Calculated from number
        of data points, period, and the clock speed.
        """
        return float(self.n_data_points*self.period) / self.timestamp_resolution
    
    @property
    def file_type(self):
        """Return 8 byte header for this file."""
        return "NEURALSG"
    
    @property
    def scale(self):
        """Returns the scale for Nsx2.1 files.  As far as I can tell this 
        is always 1 for NSx2.1"""
        return 1.0
    
    def get_basic_header(self):
        """Return basic header for Nsx2.1 file.
        Returns NEURALSG instance or None with failure
        """
        # full format depends on the channel_count field so we 
        # cannot read the whole header in one go 
        # ensure that we are at the start of the file
        self.fid.seek(0, os.SEEK_SET)
        try:
            buf = self.fid.read(self.header_size)
            header_tup = struct.unpack(self.header_format, buf)
        except:
            raise NeuroshareError(NSReturnTypes.NS_BADFILE,
                                  "failed reading file")
        channel_ids = numpy.array(header_tup[4:])
        return NEURALSG(header_tup[0], header_tup[1], header_tup[2], 
                        header_tup[3], channel_ids)

    # TODO: implement a "fast reader" for this function as done for 2.2    
    def get_analog_data(self, channel, start_index, index_count): 
        """Return the analog waveform for Nsx2.1 files.  Returns data starting at the 
        start_index bin and the next index_count bins.   If the end of the file is reached 
        before index_count return a waveform with as many bins as are found.  
        scale provides the conversion from ADC counts to physical numbers.
        Parameters:
            channel - index of the wanted electrode data
            start_index - first bin of electrode data to return
            index - how many bins of the waveform to return
        """          
        # if index count is not provided we read to the end of the file
        if index_count == None:
            index_count = self.n_data_points - start_index
        # Calculate the size of a single data packet
        packet_size = self.channel_count * 2
        # Find the position of the first wanted data point from the start
        # of the data packets
        offset = start_index*packet_size + 2*channel
        # after a we read an entry, we skip through one set of data packets
        # to the next data point that we want
        skip_size = packet_size - 2
        # skip to the first data point that we want
        self.fid.seek(self.header_size + offset, os.SEEK_SET)
        # setup waveform to return
        waveform = numpy.zeros(index_count, dtype=numpy.double)
        bin_count = 0
        for iBin in xrange(0, index_count):
            # get the wanted data point
            buf = self.fid.read(2)
            # if we've reached the end of the file stop
            if len(buf) < 2:
                sys.stderr.write("warning: file ended\n")
                break
            waveform[iBin] = struct.unpack("h", buf)[0]
            bin_count += 1
            # advance to the start of the next wanted data point            
            self.fid.seek(skip_size, os.SEEK_CUR)
        # if didn't make it to index_count resize the resulting array
        waveform = numpy.resize(waveform, bin_count)
        # return pointer?
        # self.fid.seek(0, os.SEEK_SET)
        return waveform        
        
class Nsx22Parser:
    """Interface to Nsx2.2 files.
    
    Uses the Python struct module to read all the binary data found in the 
    Nsx21 style files.  Holds as member variables a file object and a small 
    amount of header data to allow for easy retrieval. """

    def __init__(self, fid):
        """Initialize Nsx22Parser. Some internal data is store from
        basic header to facilitate the reading of data packets.  Note: This
        class will own and hold the file instance provided in the constructor.
        I.e, it will close the file when deleted.
        
        Parameter:
            fid -- valid file pointer
        """        
        self.fid = fid
        # self.is_float = self.fid.name.endswith('.nf3')
        self.bytes_per_point = 2
        # find the file size simply by skipping to the end of the file
        self.fid.seek(0, os.SEEK_SET)
        # Get file header so we may check that the data file is of type float or int
        header_type = struct.unpack("<8s", self.fid.read(8))[0]
        self.is_float = False
        if header_type == "NEUCDFLT":
            self.is_float = True
        if self.is_float:
            self.bytes_per_point = 4
        # calculate size of the file
        self.fid.seek(0, os.SEEK_END)
        self.size = self.fid.tell()
        
        header = self.get_basic_header()
        # number of electrodes producing analog data.  This is also the number
        # of extended headers (CC headers) found in this file
        self.channel_count = header.channel_count
        self.bytes_headers = header.bytes_headers
        # store the number of data packets (i.e. pauses).  This list will
        # hold a tuple of (timestamps, data points until end of file or next pause)
        self.data_packet_list = []
        self.fid.seek(self.bytes_headers, os.SEEK_SET)
        while True:
            buf = self.fid.read(9)
            if len(buf) != 9:
                break
            (head, ts, n_data_points) = struct.unpack('<B2I', buf)
            if not head == 1:
                raise NeuroshareError(NSReturnTypes.NS_BADFILE,
                                      "Invalid data packet header: {0}".format(head))
            self.data_packet_list.append((ts, n_data_points))
            # skip past all the data point is this pause period to get the header of the next data packet
            if self.is_float:
                # in the case of float streams we have 4 bytes per channel
                self.fid.seek(self.channel_count*n_data_points * 4, os.SEEK_CUR)
            else:
                # with normal NSX2.2 we have shorts for each point
                self.fid.seek(self.channel_count*n_data_points * 2, os.SEEK_CUR)
            
        # now that we know channel count we can calculate the format and size of one data packet
        if self.is_float:
            # in the case of float stream each point is a float
            self.data_packet_form = "<B2I{0:d}f".format(self.channel_count)
        else:
            # normal data point have shorts as data points
            self.data_packet_form = "<B2I{0:d}h".format(self.channel_count)
        self.data_packet_size = struct.calcsize(self.data_packet_form)
        # record the conversion between ADC and physical values.  This is will be needed
        # when we read the analog waveforms 
#        self.scale = float(header.max_analog_value - header.min_analog_value)
#        self.scale = self.scale / (header.max_dig_value - header.min_dig_value)
        self.timestamp_resolution = header.timestamp_resolution
        self.period = header.period
        
    @property
    def n_data_points(self):
        """Return the number of data points from each pause section"""
        n_data_points = 0
        for points in self.data_packet_list:
            n_data_points += points[1]
        return n_data_points
        
    def __del__(self):
        """close the file when we're done with this instance"""
        self.fid.close()
        
    @property
    def time_span(self):
        """Return time_span of data in this file.  Calculated from number of data
        points, period, and the clock speed
        """        
        return float(self.n_data_points * self.period) / self.timestamp_resolution
            
    def get_basic_header(self):
        """return the basic NEURALCD file header using the NEURALCD struct defined above."""
        # ensure we start at the start of the file.
        self.fid.seek(0, os.SEEK_SET)
        buf = self.fid.read(NEURALCD_SIZE)
        tup =  struct.unpack(NEURALCD_FORMAT, buf)
        if not(tup[0] == "NEURALCD" or tup[0] == "NEUCDFLT"):
            raise NeuroshareError(NSReturnTypes.NS_BADFILE,
                                  "cannot find NEURALCD header\n")
        timestamp = _proc_timestamp_struct(tup[8:16])
        return NEURALCD._make(tup[0:8] + (timestamp,) + tup[16:])

    @property
    def file_type(self):
        """Return 8 byte header for this file.  Always NEURALCD"""
        if self.is_float:
            return "NEUCDFLT"
        return "NEURALCD"
    
    def get_extended_headers(self):
        """generator to loop through extended headers.  Makes use of the
        get_extended_header function.  
        """
        # skip to the start of the data packets
        self.fid.seek(NEURALCD_SIZE, os.SEEK_SET)
        for _ in xrange(0, self.channel_count):
            header = self.get_extended_header()
            yield header
            
    def get_extended_header(self, header_index=None):
        """Get the desired extended (CC) header.  If header_index == None,
        than the next header is read from the current position of the file.
        If the header_index is >= 0 skip to that position in the file and
        return that CC header 
        """
        # if header_index is provided we skip to the desired extended
        # header.  Otherwise, we just read from the current file position
        if header_index != None:
            if header_index > self.channel_count or header_index < 0:
                raise NeuroshareError(NSReturnTypes.NS_BADINDEX,
                                      "invalid header index: {0}".format(header_index))
            position = NEURALCD_SIZE + CC_SIZE * header_index
            self.fid.seek(position, os.SEEK_SET)
        buf = self.fid.read(CC_SIZE)
        return CC._make(struct.unpack(CC_FORMAT, buf))
    
    def get_analog_packet(self, channel_index, start_index, index_count): 
        """Generator to to read a file in large chunks but return 
        only the wanted channel and data from a time slice. 
        Parameters:
            channel_index - if specified, only return data for the wanted.  If 
                unspecified, returns all the data for a single time-slice
            packet_index - return the wanted packet.  If unspecified, returns
                as a yield.  Otherwise, 
        """
        # We chunks of 1024 points to ensure that we don't slow down running by excessive 
        # fread calls, but still we don't crash the machine trying to read 10G at once
        max_read_points = 1024
        # max_read_bytes = 2*1024*self.channel_count
        # To support pausing in nsx2.2, we must keep track of the the number of 
        # data points found before the next pause, this is stored in 
        # self.data_packet_list.
        self.fid.seek(self.bytes_headers, os.SEEK_SET)
        # skip to the wanted start index
        remaining_skipped_points = start_index
        skipped_packets = 0
        
        for data_packet in self.data_packet_list:
            self.fid.seek(9, os.SEEK_CUR)
            if remaining_skipped_points <= data_packet[1]:
                self.fid.seek(remaining_skipped_points*self.channel_count*2, os.SEEK_CUR)
                break
            skipped_packets += 1
            remaining_skipped_points -= data_packet[1]
            self.fid.seek(data_packet[1], os.SEEK_CUR)
        points_read = 0
        while points_read < index_count:
            # Loop over all the pause points as each comes with a header
            # hold the number of data points in each pause
            for ipacket, data_packet in enumerate(self.data_packet_list[skipped_packets:]):
                # TODO: do we need to store all the pauses? check how this is used 
                # in DLL and Matlab
                # store the number of data points until the next pause
                 
                remaining_points = data_packet[1] - remaining_skipped_points
                # skip first 3 fields of data packet (B2I)
                if ipacket > 0:
                    self.fid.seek(9, os.SEEK_CUR)
                # read through this pause section
                while remaining_points > 0 and points_read < index_count:
                    # read 1024 points or the end of the section
                    read_points = min(max_read_points, remaining_points)
                    
                    read_bytes = read_points * self.bytes_per_point * self.channel_count
                    read_buffer = self.fid.read(read_bytes)
                    # Throw exception or error on failure
                    if len(read_buffer) != read_bytes:
                        # TODO: throw exception on failed read
                        msg = 'should have read {0}, but read {1}'.format(read_bytes, len(read_buffer))
                        sys.stderr.write('warning: {0}\n'.format(msg))
                        # raise NeuroshareError(msg)
                        read_points = len(read_buffer) / self.bytes_per_point / self.channel_count
                    # Get the wanted data point and yield it
                    for point in range(0, read_points):
                        data_string = '<h'
                        if self.is_float:
                            data_string = '<f'
                        byte = self.bytes_per_point * (channel_index + point * self.channel_count)
                        buf = read_buffer[byte:byte + self.bytes_per_point]
                        data = struct.unpack(data_string, buf)[0]
                        points_read += 1
                        yield data 
                        if points_read == index_count:
                            return
                    # update the remaining points 
                    remaining_points -= read_points
            return

    def get_analog_data(self, channel_index, start_index, index_count):
        """Return the analog waveform for Nsx2.2 files.  Returns data 
        starting at the start_index bin and the next index_count bins. If the end 
        of the file is reached before index_count return a waveform with 
        as many bins as are found.   
        Parameters:
            channel - index of the wanted electrode data
            start_index - first bin of electrode data to return
            index - how many bins of the waveform to return
        Returns:
            Requested waveform as a numpy.array
        """
        if index_count <= 0 or \
            index_count + start_index > self.n_data_points:
            NeuroshareError(NSReturnTypes.NS_BADINDEX, 
                            'invalid index count')
        if start_index < 0:
            NeuroshareError(NSReturnTypes.NS_BADINDEX,
                            'invalid start index')
        if index_count == None:
            index_count = self.n_data_points - start_index
        
        # initialize an array to return
        waveform = numpy.zeros(index_count)
        # Use generator defined above to get analog data efficiently
        data_gen = self.get_analog_packet(channel_index, start_index, index_count)
        # Copy data from generator to waveform 
        for index, data_point in enumerate(data_gen): 
            waveform[index] = data_point

        return waveform

    def get_data_packet_format(self, index):
        """calculate the number format for the index'th data packet using
        the number of data points and the channel count
        """
        n_data_packets = self.data_packet_list[index][1]
        form = "<B2I{0:d}h".format(self.channel_count*n_data_packets)
        return form
    
    def get_data_packet(self, packet_index=None):
        """Return one full data packet.  This will contain an array of all the
        digitized data for one moment in time.  If packet_index is not specified
        we will read from the current moment in the file.  Otherwise, we skip
        to the desired packet
        """
        if packet_index != None:
            if packet_index < 0 or packet_index >= self.n_data_points:
                raise NeuroshareError(NSReturnTypes.NS_BADINDEX, 
                                      "invalid packet_index: {0:d}".format(packet_index))
        
            self.fid.seek(self.bytes_headers, os.SEEK_SET)
            for ipacket in range(0, packet_index - 1):
                size = struct.calcsize(self.get_data_packet_format(ipacket))
                self.fid.seek(size, os.SEEK_CUR)
        
        form = self.get_data_packet_format(packet_index) 
        size = struct.calcsize(form)
        tup = struct.unpack(form, self.fid.read(size))
        #data_points = numpy.array(tup[3:], dtype=numpy.int16)
        
        return tup
    
# Debugging section
if __name__ == "__main__":
    #infile = "/home/elliottb/ripple/test_data/datafile0001.nev"
    infile = "C:/Users/elliottb/Desktop/stim_tests/datafile0006.nev"
    parser = ParserFactory(infile)
    for header in parser.get_extended_headers():
        print header
