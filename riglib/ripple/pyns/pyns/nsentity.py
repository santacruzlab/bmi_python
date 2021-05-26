# Created on Apr 23, 2012
# @author: Elliott L. Barcikowski
'''Collection of classes containing data and functions to hold entity information.
These classes facility the reading of nev and nsx (2.1 and 2.2) files to access
info functions as well as getting data, waveforms, and timestamps.

All of the Neuroshare API info and data functions are included in the following
classes, but these classes extend this and may enable users more flexibility than
that API.
'''
from collections import namedtuple
import sys
import numpy
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

# The below namedtuple's EntityInfo, EventInfo, SegmentInfo, SegSourceInfo, 
# AnalogInfo, NeuralInfo correspond to the ns_ENTITYINFO, ns_EVENTINFO,
# ns_SEGMENTINFO, ns_SEGSOURCEINFO, ns_ANALOGINFO, ns_NEURALINFO (respectively)
# Each of the classes derived from the Entity base class return the respective
# file info containers to produce the results found in the Neuroshare API
EntityInfo = namedtuple("EntityInfo", "label entity_type item_count")
EventInfo = namedtuple("EventInfo", "event_type min_data_length max_data_length "\
                      "csv_desc")
SegmentInfo = namedtuple("SegmentInfo", "source_count min_sample_count max_sample_count " \
                         "sample_rate units")
SegSourceInfo = namedtuple("SegSourceInfo", "min_val max_val resolution subsample_shift "\
                        "location_x location_y location_z location_user high_freq_corner "\
                        "high_freq_order high_filter_type low_freq_corner low_freq_order "\
                        "low_filter_type probe_info")
AnalogInfo = namedtuple("AnalogInfo", "sample_rate min_val max_val units resolution "\
                        "location_x location_y location_z location_user high_freq_corner "\
                        "high_freq_order high_filter_type low_freq_corner low_freq_order "\
                        "low_filter_type probe_info")
NeuralInfo = namedtuple("NeuralInfo", "source_entity_id source_unit_id probe_info")

class EntityType:
    """static class that will to interface with entity types and
    provide a couple of utility functions like getting the equivalent 
    strings
    """
    unknown = 0
    event = 1
    analog = 2
    segment = 3
    neural = 4
    
    @classmethod
    def get_entity_string(cls, entity_type):
        """return string corresponding to entity type"""
        if entity_type == cls.unknown:
            return "unknown"
        elif entity_type == cls.event:
            return "event"
        elif entity_type == cls.analog:
            return "analog"        
        elif entity_type == cls.segment:
            return "segment"
        elif entity_type == cls.neural:
            return "neural"
        else:
            sys.stderr.write("invalid entity type: {0:d}\n".format(entity_type))
            return ""
            
    @classmethod            
    def get_entity_id(cls, entity_string):
        """Return enum value from entity string, one of "analog",
        "segment", "neural", "event", or unknown
        """
        # don't care if upper or lower case is used
        entity_string = entity_string.lower()
        
        if entity_string == "unknown":
            return cls.unknown
        elif entity_string == "event":
            return cls.event
        elif entity_string == "analog":
            return cls.analog
        elif entity_string == "segment":        
            return cls.segment
        elif entity_string == "neural": 
            return cls.neural
        else:
            sys.stderr.write("invalid entity string: {0:s}\n".format(entity_string))
            return -1

class EventType:
    """Static class to interface with digital event types"""
    text  = 1
    csv   = 2
    byte = 3
    word = 4
    dword = 5
        
# PacketData namedtuple is used to store timestamp and packet_index
# This makes plotting spikes easy and makes it so we can easily 
# find data associated with an entity easy
# PacketData = namedtuple("PacketData", "timestamp packet_index")
class Entity: 
    """Base class for Neuroshare Entities.  This is an abstract class that 
    holds data and functions common to all neuroshare entities.  Actual, 
    instances will of one of the below derived classes (SegmentEntity, 
    AnalogEntity, EventEntity, or NeuralEntity)
    """
    # allocate space for 10k NEV data at a time
    PACKET_DATA_ALLOC = 10 * 1024
    
    def __init__(self, parser, electrode_id):
        """Initialize base entity class.  This class should not be called
        directly, but by all derived classes.
        
        Parameters:
            parser -- Parser class from pyns.parser module
            electrode_id -- Id number for this entity            
        """
        # store a reference to the file parser so that we may 
        # get to data and an info in the file through the
        # entity classes
        self.parser = parser
        self.electrode_id = electrode_id
        self.item_count = 0
        
        # packet_data list holds PacketData namedtuples to save timestamps and
        # locatations of data packets.  There will be one entry in this
        # list for each "item" associated with the entity  
        # self.packet_data = []
        self.packet_data = numpy.array([], dtype=numpy.uint32)
    def get_entity_info(self):
        """return the entity info for this entity"""
        return EntityInfo(self.label, self.entity_type, self.item_count)
         
    def add_packet_data(self, timestamp, packet_index):
        """Add packet data to list.  The length of this list should be the
        same as item_count
        """
        # if we have run out of space, reallocate
        length = self.packet_data.shape[0]
        if self.item_count >= length:
            self.packet_data = numpy.resize(self.packet_data, [length + self.PACKET_DATA_ALLOC, 2])
        # self.packet_data.append(PacketData(timestamp, packet_index))
        self.packet_data[self.item_count] = [timestamp, packet_index]
        
        self.item_count += 1

    def resize_packet_data(self):
        """Reset packet data to the number of items found"""
        self.packet_data = numpy.resize(self.packet_data, [self.item_count, 2])
        
    def get_time_by_index(self, index):
        """Equivalent to the Neuroshare function ns_GetTimeByIndex.  Returns
        the time in seconds from the start of data taking for the time
        corresponding to index.  Note: This is overridden by each 
        derived Entity class
        
        Parameters:
            index - index of wanted item
            
        Returns:
            time in seconds since the start of data taking
        """
        pass

    def get_index_by_time(self, time, flag=0):
        """Equivalent to the Neuroshare function ns_GetIndexByTime.
        Return the segment index to the segment best matched by time.
        Note: This is overridden by each derived Entity class
        
        Parameters:
            time - time in seconds from the start of data taking
            flag - flag to describe how to match files:
                -1 - Return the data entry occurring before
                     or inclusive of time.
                 0 - Return the data entry occurring at or
                     closest to time.
                 1 - Return the data entry occurring after or
                     inclusive of time. (default)
                     
        Returns:
            best matched index
        """
        pass            
    @property
    def label(self):
        """Return default entity label.  This label is used in Segment and 
        Neural entities.  This is consistent with both Matlab code as well 
        as Neuroshare DLL.
        """
        pass
    
class SegmentEntity(Entity):
    """Holds data and information for Segment or spike data found in NEV 
    files.  The timestamps of spikes as well as the position of waveforms 
    in a data file are stored in the packet_data list
    """
    def __init__(self, parser, electrode_id):
        """Initialize Segment Entity.
        
        Parameters:
            parser -- Parser class from pyns.parser, should be type NevParser
            electrode_id -- Id number for this entity 
        """
        Entity.__init__(self, parser, electrode_id)
        self.item_count = 0
        self.entity_type = EntityType.segment
        
    @property
    def label(self):
        return "elec{0:d}".format(self.electrode_id)
        
    def get_segment_info(self):
        """return the segment info struct for this entity"""
        source_count = 1
        min_sample_count = self.parser.sample_count
        max_sample_count = min_sample_count
        sample_rate = self.parser.timestamp_resolution
        # in the case of stimulation makers (with electrode ids > 5120)
        # the data is returned in V.  For all normal neural data uV is used.
        units = "uV"
        if self.electrode_id > 5120 and self.electrode_id < 10240:
            units = "V"
        return SegmentInfo(source_count, min_sample_count, 
                           max_sample_count, sample_rate, units)
        
    def get_extended_headers(self):
        """searches through the NEV file and finds all the extended headers 
        associated with this electrode_id.  This method may be slow.  If 
        that is the case, I will at the location of each extended header to 
        the SegmentEntity class
        """
        wanted_headers = ["NEUEVWAV", "NEUEVFLT", "NEUEVLBL"]
        headers = {}
        for header in self.parser.get_extended_headers():
            if header.header_type in wanted_headers:
                if header.packet_id == self.electrode_id:
                    headers[header.header_type] = header
        return headers
      
    def get_segment_data(self, index):
        """return the segment waveform corresponding to index for this entity.
        
        This function is the Neuroshare equivalent to the ns_GetSegmentData
        
        Parameters:
            index - index of wanted item
            
        Returns:
            tuple - (timestamp, waveform, unit_id)
                timestamp - time of this item in seconds from start of 
                    data taking
                waveform - waveform found in data packet.  The units may 
                   be uV (with Neural Data) or V (with stim data)
                unit_class - classification of item (0-255)
        """
        try:
            packet_index = self.packet_data[index][1]
        except:
            raise NeuroshareError(NSReturnTypes.NS_BADINDEX,
                                  "invalid entity index: {0:d}".format(index))
        packet = self.parser.get_data_packet(packet_index)
        timestamp = float(packet.timestamp) / self.parser.timestamp_resolution
        # Get all extended headers corresponding to this electrode
        headers = self.get_extended_headers()
        
        scale = 1.0
        if "NEUEVWAV" in headers.keys():
            header = headers["NEUEVWAV"]
            if header.dig_factor != 0:
                # Scale factor in header is in units of ADC per nanovolt.  Put
                # it into ADC to microvolts
                scale = float(header.dig_factor) / 1000
            elif header.stim_amp_dig_factor != 0:
                # For stim markers, the scale conversion is stored as a float 
                # and is in units of ADC to volts.  Purposly return data in
                # Volts in the case of stimulation markers.
                scale = header.stim_amp_dig_factor

        waveform = packet.waveform * scale
        # Get bitmasked unit id.  Taken from nsNEVLibrary's XfmUnitNevToNs
        if packet.unit_class == 0:
            unit_id = 0
        elif packet.unit_class == 255:
            unit_id = 1
        else:
            unit_id = 1 << packet.unit_class
        return (timestamp, waveform, unit_id)
        
    def get_seg_source_info(self):
        """Equivalent to the ns_GetSegSourceInfo from the Neuroshare API.
        Returns information found in the NEUEVFLT header, an extended
        header in .nev files.  
        
        Returns:
            SegSourceInfo instance
        """
        # putting min_val, max_val, and resolution in by hand seems consistent with the
        # way the DLL treats most files. Need to ensure that this is correct however.
        max_val = 8191.0
        min_val = -8191.0
        # To be exactly compatible with the Neuroshare DLLs, the following code needed
        # this is probably not needed or wanted with Ripple files
        # if self.electrode_id >= 129 and self.electrode_id <= 144:
            #max_val = 5000.0
            #min_val = -5000.0
        resolution = 1.0
        # Not sure where subsample_shift is stored.  Every file that I've looked returns 0 here
        subsample_shift = 0.0
        location_x = 0.0
        location_y = 0.0
        location_z = 0.0
        location_user = 0.0
        # probe_info
        # These are found in the NEUEVFLT header, if present
        high_freq_corner = 0.0
        high_freq_order = 0.0
        high_filter_type = "none"
        low_freq_corner = 0.0
        low_freq_order = 0.0
        low_filter_type = "none"
        probe_info = ""
        headers = self.get_extended_headers()
        if "NEUEVWAV" in headers.keys():
            header = headers["NEUEVWAV"]
            probe_info = "module {0:d}, pin {1:d}".format(header.phys_conn, header.conn_pin)
            if header.dig_factor != 0.0:
                # This is another strange convention to be compatible with the Neuroshare DLLs.
                # This check is likely NOT compatible with many data files include those
                # produced by Ripple devices.
                #if self.electrode_id >= 129 and self.electrode_id <= 144:                
                #     resolution = header.dig_factor*1.0e-6
                # else:
                resolution = header.dig_factor * 1.0e-3
                
        if "NEUEVFLT" in headers.keys():
            header = headers["NEUEVFLT"]
            # convert corner to Hz from mHz
            high_freq_corner = float(header.high_freq_corner) / 1000
            high_freq_order = header.high_freq_order
            if header.high_filter_type == 1:
                high_filter_type = "Butterworth"
            # convert corner to Hz from mHz
            low_freq_corner = float(header.low_freq_corner) / 1000
            low_freq_order = header.low_freq_order
            if header.low_filter_type == 1:
                low_filter_type = "Butterworth"
        return SegSourceInfo(min_val, max_val, resolution, subsample_shift, location_x,
                             location_y, location_z, location_user, high_freq_corner,
                             high_freq_order, high_filter_type, low_freq_corner, low_freq_order,
                             low_filter_type, probe_info)
    
    def get_index_by_time(self, time, flag=0):
        """Equivalent to the Neuroshare function ns_GetIndexByTime.
        Return the segment index to the segment best matched by time.
        
        Parameters:
            time - time in seconds from the start of data taking
            flag - flag to describe how to match files:
                -1 - Return the data entry occurring before
                     or inclusive of time.
                 0 - Return the data entry occurring at or
                     closest to time.
                 1 - Return the data entry occurring after or
                     inclusive of time. (default)
                     
        Returns:
            best matched index
        """
        # check that we have a valid flag
        if not flag in [-1, 0, 1]:
            raise NeuroshareError(NSReturnTypes.NS_BADINDEX,
                                  "invalid flag value {0}".format(flag))            
#            raise NeuroshareError(NSReturnTypes.NS_BADINDEX,
#                                  "{0}: invalid flag value {1}".format(self.__name__, flag))
        # put all the timestamps in a numpy array so that we may search it 
        # with a binary search algorithm from that package
        res = self.parser.timestamp_resolution
        timestamps = numpy.array([data[0] for data in self.packet_data],
                                 dtype=numpy.double)/res        
        #timestamps = numpy.array([p.timestamp for p in self.packet_data],
        #                         dtype=numpy.double)/res
        # fail if time is less than zero or if time is greater than the 
        # last bin
        if time < timestamps[0] and flag == -1:
            raise NeuroshareError(NSReturnTypes.NS_BADINDEX,
                                  "before first entry: {0}".format(time))
        if time > timestamps[-1] and flag == 1:
            raise NeuroshareError(NSReturnTypes.NS_BADINDEX,
                                  "after last entry: {0}".format(time))                         
        index = timestamps.searchsorted(time)
        # special case requested time is before the first event.  In this case
        # each case in flag does not make sense.  We will always return the 
        # the "0" index.  Note: this is slightly different than the Matlab code
        # which will return the "1" if flag == 1.
        if index == 0:
            return index
        if flag == -1:
            return index - 1
        if flag == 1:
            return index
        else:
            # TODO: clean this up!
            resleft = abs(timestamps[index - 1] - time)
            resright = abs(timestamps[index] - time)
            if resleft > resright:
                return index
            return index - 1
         
    def get_time_by_index(self, index):
        """
        Equivalent to the Neuroshare function ns_GetTimeByIndex.  Returns
        the time in seconds from the start of data taking for the time
        corresponding to index.
        
        Parameters:
            index - index of wanted item
        Returns:
            time in seconds since the start of data taking
        """
        if index < 0 or index > self.item_count:
            raise NeuroshareError(NSReturnTypes.NS_BADINDEX,
                                  "invalid index: {0}".format(index))
        res = self.parser.timestamp_resolution
        # return float(self.packet_data[index].timestamp)/res
        return float(self.packet_data[index][0])/res
    
class EventEntity(Entity):
    """Holds data and function for digital event entities found in NEV files."""
    def __init__(self, parser, mode):
        """Initialize Segment Entity.
        
        Parameters:
            parser -- Parser class from pyns.parser, should be type NevParser
            mode -- single word reason for packet insertion.
        """        
        Entity.__init__(self, parser, 0)
        self.entity_type = EntityType.event
        self.mode = mode
        
    @property
    def label(self):
        """Return label for EventEntity"""
        # The following version of the label is consistent with the Matlab code 
        # label = "elec{0:d}".format(self.electrode_id)
        # The following values for the label are consistent with the 
        # Black Rock DLL.n
        if self.mode == 129:
            label = "serial"
        else:
            label = "digin"
        return label
    
    def get_event_info(self):
        """equivalent of the Neuroshare function ns_GetEventInfo
        
        Returns:
            EventInfo instance
        """
        if self.mode == 129:
            label = "serial I/O uint16"
        elif self.mode == 1:
            label = "triggered digital uint16"
        min_data_length = 2
        max_data_length = 2
        event_type = EventType.byte
        return EventInfo(event_type, min_data_length, max_data_length,
                         self.label)
        
    def get_event_data(self, packet_index):
        """equivalent of the ns_GetEventData from the Nueroshare API"""
        time_res = self.parser.timestamp_resolution
        # packet_index = self.packet_data[packet_index].packet_index
        packet_index = self.packet_data[packet_index][1]
        packet = self.parser.get_data_packet(packet_index)

        # This commented out code seems to be more consistent with the Neuroshare
        # API, however, it doesn't make as much sense to me.  I'm just going to 
        # return all the digital data for a given entity for now.
#        reason = packet.reason
#        if reason == 0:
#            data = packet.digital_input
#        elif reason == 1:
#            data = packet.input1
#        elif reason == 2:
#            data = packet.input2
#        elif reason == 4:
#            data = packet.input3
#        elif reason == 8:
#            data = packet.input4
#        elif reason == 16:
#            data = packet.input5
#        else:                        
#            data = (packet.digital_input, packet.input1, packet.input2,
#                    packet.input3, packet.input4, packet.input5)
        data = (packet.digital_input, packet.input1, packet.input2,
        packet.input3, packet.input4, packet.input5)            
        return (float(packet.timestamp) / time_res, data)
    
    def get_index_by_time(self, time, flag=0):
        """Return the segment index to the segment best matched by time.
        
        Parameters:
            time - time in seconds from the start of data taking
            flag - flag to describe how to match files:
                -1 - Return the data entry occurring before
                     or inclusive of time.
                 0 - Return the data entry occurring at or
                     closest to time.
                 1 - Return the data entry occurring after or
                     inclusive of time. (default)
                     
        Returns:
            best matched index
        """
        # check that we have a valid flag
        if not flag in [-1, 0, 1]:
            raise NeuroshareError(NSReturnTypes.NS_BADINDEX,
                                  "invalid flag value {0}".format(flag))            
#            raise NeuroshareError(NSReturnTypes.NS_BADINDEX,
#                                  "{0}: invalid flag value {1}".format(self.__name__, flag))
        # put all the timestamps in a numpy array so that we may search it 
        # with a binary search algorithm from that package
        res = self.parser.timestamp_resolution
        timestamps = numpy.array([data[0] for data in self.packet_data],
                                 dtype=numpy.double)/res
        # timestamps = numpy.array([p.timestamp for p in self.packet_data],
        #                         dtype=numpy.double)/res
        # fail if time is less than zero or if time is greater than the 
        # last bin
        if time < timestamps[0] and flag == -1:
            raise NeuroshareError(NSReturnTypes.NS_BADINDEX,
                                  "before first entry: {0}".format(time))
        if time > timestamps[-1] and flag == 1:
            raise NeuroshareError(NSReturnTypes.NS_BADINDEX,
                                  "after last entry: {0}".format(time))                         
        index = timestamps.searchsorted(time)
        # special case requested time is before the first event.  In this case
        # each case in flag does not make sense.  We will always return the 
        # the "0" index.  Note: this is slightly different than the Matlab code
        # which will return the "1" if flag == 1.
        if index == 0:
            return index
        if flag == -1:
            return index - 1
        if flag == 1:
            return index
        else:
            # TODO: clean this up!
            resleft = abs(timestamps[index - 1] - time)
            resright = abs(timestamps[index] - time)
            if resleft > resright:
                return index
            return index - 1
         
    def get_time_by_index(self, index):
        """Retrieves time range from entity indexes
        
        Parameters:
            index - index for wanted item
            
        Returns:
            timestamp in seconds from start of data taking
        """
        if index < 0 or index > self.item_count:
            raise NeuroshareError(NSReturnTypes.NS_BADINDEX,
                                  "invalid index: {0}".format(index))            
        res = self.parser.timestamp_resolution
        # return float(self.packet_data[index].timestamp)/res
        return float(self.packet_data[index][0])/res
    
class AnalogEntity(Entity):
    """Holds data and functions needed for analog entities found in NS[0-9] 
    files.  This file stores some additional information found in NSx2.2 
    style files that include a variety of information about probes in their 
    CC headers.  Looking in constructor documentation for more information 
    about these variables. 
    """
    def __init__(self, parser, electrode_id, units, channel_index,
                   scale, electrode_label=None):
        """Initialize Analog Entity.
        
        Parameters:
            parser -- Parser class from pyns.parser, should be type 
                Nsx21Parser or Nsx22Parser
            electrode_id -- electrode id for this entity
            units -- units read from CC header
            channel_index -- Index when this entity is found in the data, 
                storing this allows jumping directly to data. 
            scale -- TODO: Add this comment
            electrode_label -- label of this electrode, read from cc 
                header in Nsx22
        """        
        Entity.__init__(self, parser, electrode_id)
        # NSx2.2 files contain the electrode label in them, however,
        # however, 2.1 files do not.  If 2.1 we store a string of length zero.
        self.electrode_label = electrode_label 
        self.item_count = self.parser.n_data_points
        self.entity_type = EntityType.analog
        # units of the data for this entity 
        self.units = units
        # channel_index is the index of this channel in the file.  This allows us
        # find where the data for this file will be found
        self.channel_index = channel_index
        self.scale = scale
        # if we have a float stream, disregard scale (it should always be one, but is sometimes mis-reported
        # in headers.
        if self.parser.is_float:
            self.scale = 1
         
    @property
    def sample_freq(self):
        """Return the sample frequency in Hz for this analog channel.  
        This number is the same for all analog channels of the same type and 
        depends on the period and the timestamp_resolution.  Generally 
        timestamp_resolution will be the same for all entities but the period 
        (or how much a waveform is decimated will change).  All of these 
        things are stored in the parser.  
        """
        return self.parser.timestamp_resolution / self.parser.period
    
    @property
    def label(self):
        """Return entity label to be consistent with Neuroshare DLL.  In the
        case of NSx2.1 these labels are included in the CC header.  In NSx2.1 we
        create this label based on the period field.
        """
        # in the case of Nsx2.2 files, the CC header contains a label for each 
        # analog entity
        if self.electrode_label:
            return self.electrode_label
        # for Nsx2.1 files, we use this form which is consistent with Neuroshare DLL
        return "{0:d} - {1:d} kS/s".format(self.electrode_id,
                                           int(self.sample_freq/1000))
        
    def get_analog_info(self):
        """equivalent to the ns_GetAnalogInfo function from the Neuroshare API"""
        # These locations are not defined in the nsx files
        location_x = 0.0
        location_y = 0.0
        location_z = 0.0
        location_user = 0.0
        # This the case of the NSx2.1 files.  These files have no extended headers
        # and some of the information in the analog info struct is not provided
        if self.parser.file_type == "NEURALSG":
            # the timestamp_resolution here is hard coded, this is consistent with
            # the DLL.  Perhaps a better method would be to get this info from 
            # the NEV header file. 
            min_val = -8191.0
            max_val = 8191.0
            probe_info = "periodic analog pin {0:d}".format(self.electrode_id)
            units = "V"
            # The following values are not defined in the NSx2.1 files as far
            # as I know.  Setting the following quantities to these values is
            # consistent with the Neuroshare DLL
            high_freq_corner = 0
            high_freq_order = 0
            high_filter_type =""
            low_freq_corner = 0
            low_freq_order = 0
            low_filter_type = ""
        # The case of a NSx2.2 file.  If this wasn't a 2.1 we know that it's
        # a 2.2 style file as this was checked in the constructor
        else:
            # read extended header from file
            header = self.parser.get_extended_header(self.channel_index)
            # for compatibilty with the DLL, this modification must be
            # made to the probe_info.  See Catalog.cpp: 1659
            probe_info = "This is channel {0:d}".format(self.electrode_id)
            # probe_info = header.electrode_label.split('\0')[0]
            # corner is stored in mHz.  This is converted to Hz here
            high_freq_corner = float(header.high_freq_corner)/1000
            high_freq_order = header.high_freq_order
            # TODO: putting in "none" as the filter type seems to be inconsistent
            # with the neuroshare DLL.  It just puts "".  However, the Matlab code
            # seems to put the string "none"
            high_filter_type = ""
            low_filter_type = ""            
            if header.high_filter_type == 0:
                high_filter_type = "none"
            elif header.high_filter_type == 1:
                high_filter_type = "Butterworth"
                
            # corner is stored in mHz.  This is converted to Hz here                            
            low_freq_corner = float(header.low_freq_corner)/1000
            low_freq_order = header.low_freq_order
            if header.low_filter_type == 0:
                low_filter_type = "none"
            elif header.low_filter_type == 1:
                low_filter_type = "Butterworth"
            #sample_rate = float(header.timestamp_resolution) / header.period
            min_val = header.min_analog_value
            max_val = header.max_analog_value
            units = header.units.split('\0')[0]

        return AnalogInfo(self.sample_freq, min_val, max_val, units, self.scale, 
                          location_x, location_y, location_z, location_user, 
                          high_freq_corner, high_freq_order, high_filter_type, 
                          low_freq_corner, low_freq_order, low_filter_type, probe_info) 
        
    def get_analog_data(self, start_index=0, index_count=None, use_scale=True):
        """equivalent of the ns_GetAnalogData.  
        returns the analog data for this entity starting from the start_index
        bin and returning the next index_count bins.  If index_count == None, 
        returns to the end of the analog waveform.
        
        Parameters:
            start_index - first bin of returned waveform (default: 0)
            index_count - number of bins in returned waveform (default: until end)
            use_scale - True: scale to physical units, False: use ADC count.  
                Default: True
                
        Returns: analog waveform for this entity.  Result will length index_count
            or start_index to the end file.
        """
        # all the heavy lifting for this function is done in the Nsx21PacketParser
        # or the Nsx22PacketParser.  Depending on what is held by this entity
        waveform = self.parser.get_analog_data(self.channel_index, start_index, index_count)
                                               
        if use_scale:
            waveform *= self.scale

        return waveform

    def get_extended_header(self):
        """read the extended header (CC) for this entity and return it.""" 
        return self.parser.get_extended_header(self.channel_index)
    
    def get_index_by_time(self, time, flag=0):
        """Return the analog index to the segment best matched by time.
        
        Parameters:
            time - time in seconds from the start of data taking
            flag - flag to describe how to match files:
                -1 - Return the data entry occurring before
                     or inclusive of time.
                 0 - Return the data entry occurring at or
                     closest to time.
                 1 - Return the data entry occurring after or
                     inclusive of time. (default)
                     
        Returns:
            best matched index
        """
        # check that we have a valid flag
        if not flag in [-1, 0, 1]:
            raise NeuroshareError(NSReturnTypes.NS_BADINDEX,
                                  "invalid flag value {0}".format(flag))            
        # put all the timestamps in a numpy array so that we may search it 
        # with a binary search algorithm from that package
        timestamps = numpy.arange(0, self.item_count, dtype=numpy.double)
        timestamps /= self.sample_freq 
        # fail if time is less than zero or if time is greater than the 
        # last bin
        if time < timestamps[0] and flag == -1:
            raise NeuroshareError(NSReturnTypes.NS_BADINDEX,
                                  "before first entry: {0}".format(time))
        if time > timestamps[-1] and flag == 1:
            raise NeuroshareError(NSReturnTypes.NS_BADINDEX,
                                  "after last entry: {0}".format(time))        
        index = timestamps.searchsorted(time)
        # special case requested time is before the first event.  In this case
        # each case in flag does not make sense.  We will always return the 
        # the "0" index.  Note: this is slightly different than the Matlab code
        # which will return the "1" if flag == 1.
        if index == 0:
            return index
        if flag == -1:
            return index - 1
        if flag == 1:
            return index
        else:
            # TODO: clean this up!
            resleft = abs(timestamps[index - 1] - time)
            resright = abs(timestamps[index] - time)
            if resleft > resright:
                return index
            return index - 1
        
    def get_time_by_index(self, index):
        """Retrieves time from entity index.
        
        Parameters:
            index - index for wanted item
            
        Returns:
            timestamp in seconds from start of data taking
        """        
        if index < 0 or index > self.item_count:
            raise NeuroshareError(NSReturnTypes.NS_BADINDEX,
                                  "invalid index: {0}".format(index))            
        return float(index)/self.sample_freq

class NeuralEntity(Entity):
    """Entity class for Neural style entities.  One of these instances will 
    be created for each class and each segment entity.  Storing this data 
    will allow a user to browser through spike waveforms organized by the 
    segment classes. 
    """
    def __init__(self, parser, electrode_id, unit_class, segment_entity):
        """Initialize Neural Entity.
        
        Parameters:
            parser -- Parser class from pyns.parser, should be type 
                NevParser
            electrode_id -- electrode id for this entity
            unit_class -- identification of the triggered waveform
            segment_entity -- reference to the corresponding segment entity 
        """
        Entity.__init__(self, parser, electrode_id)
        self.unit_class = unit_class
        self.entity_type = EntityType.neural
        # store a reference to the segment_entity
        self.segment_entity = segment_entity
        
    def get_neural_info(self):
        """equivalent of the ns_GetNeuralInfo from Neuroshare API"""
        # FIXME: source_unit_id and source_entity_id don't seem to agree with the
        # results from the Python wrapper Neuroshare DLL.  Why!? 
        source_unit_id = self.unit_class
        source_entity_id = self.electrode_id
        # The probe info string here has an extra space.  This is consistent with the 
        # results from the Neuroshare DLL
        probe_info = "module {0:d}, pin {1:d}".format(source_unit_id, source_entity_id)
        return NeuralInfo(source_entity_id, source_unit_id, probe_info)
    
    @property
    def label(self):
        """Return label for this Neural Entity by using the contained 
        segment entity
        """
        return self.segment_entity.label
    
    def get_time_by_index(self, index):
        """Retrieves time from entity index.
        
        Makes use of the held segment entity that in the neural entity
        class.  This call is wrapped to segement_entity.get_time_by_index
        
        Parameters:
            index - index for wanted item
            
        Returns:
            timestamp in seconds from start of data taking
        """
        return self.segment_entity.get_time_by_index(index)
    
    def get_index_by_time(self, time, flag=0):
        """Return index to the segment best matched by time.
        
        Makes use of the held segment entity in the neural entity class.
        This called is wrapped to segment_entity.get_index_by_time
        
        Parameters:
            time - time in seconds from the start of data taking
            flag - flag to describe how to match files:
                -1 - Return the data entry occurring before
                     or inclusive of time.
                 0 - Return the data entry occurring at or
                     closest to time.
                 1 - Return the data entry occurring after or
                     inclusive of time. (default)
                     
        Returns:
            best matched index
        """
        return self.segment_entity.get_index_by_time(time, flag)
    