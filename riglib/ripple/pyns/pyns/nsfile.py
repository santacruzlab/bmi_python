# Created on Apr 22, 2012
# @author: Elliott L. Barcikowski
"""Classes related to handling .nev and .nsx files.

The class below nsfile.NSFile is generally the entry point 
for the pyns codes. 
"""
import os
from glob import glob
from collections import namedtuple
import datetime
import sys 
from nsexceptions import NeuroshareError, NSReturnTypes
import nsparser   
from nsentity import AnalogEntity, SegmentEntity, EntityType, EventEntity, NeuralEntity
# if the psutil package is installed we will use it to check the available 
# system memory when we read in segment data.  In extremely large .nev files
# it is possible to have a large memory footprint
USE_MEM_CHECK = True
try:
    import psutil
except ImportError:
    USE_MEM_CHECK = False

# FileInfo is a namedtuple that corresponds to the ns_FILEINFO struct from the Neuroshare API
# This is returned from the File.get_file_info function found below
FileInfo = namedtuple("FileInfo", "file_type, entity_count, timestamp_resolution, time_span "\
                      "app_name time_year time_month time_day time_hour time_min "\
                      "time_sec time_millisec comment")
    
class FileData:
    """Internal data to be used by the File class that is needed to find 
    desired NEV and NSX data.
    members:
        parser -- File parser from the pyns.parser module, one of 
            NevParser, Nsx21Parser, Nsx22Parser
        time_span -- Holds the time span of data for this file in seconds
    """    
    def __init__(self, parser):
        """Initialize new FileData.
        
        Parameters:
        parser -- File parser for this style of file
        """
        # packet parser from parser module.  Will be of type
        # NevParser, Nsx21Parser, Nsx22Parser
        self.parser = parser
        # time_span is the  span of time of data found in the given file.  In the file
        # info struct, we return the largest of these values, but it is
        # useful to include the value for each file.  This is initialized
        # to zero, but will be filled as we read through data packets
        self.time_span = 0
        
    @property
    def file_type(self):
        """Return the file type, one of NEURALEV, NEURALSG, NEURALCD"""
        return self.parser.file_type
    
    @property
    def name(self):
        """return the full path to this data file.  We don't need
        to store this directly, we may just pass this from the parser
        object.
        """
        return self.parser.name
    
    @property
    def extension(self):
        """returns the file extension for this file as stored in the parser class"""
        return self.parser.fid.name.split(".")[-1]
    
class NSFile:
    """General entry point to the pyns implementation of the Neuroshare API.  
    This class loads all the NEV files associated with the file specified.  
    The files are read and all the found entities are stored.  This class 
    provides the port of the ns_GetFileInfo function from the Neuroshare API
    """
    def __init__(self, filename, proc_single=False):
        """Initialize new File instance.
        
        Parameters:
        filename -- relative path to wanted file
        proc_single -- If True only proccess the specified file, do not look
            for associated nev files (default=False)
        """
        self.name = os.path.basename(filename)[:-4]
        self.path = os.path.dirname(filename)
        self._files = []
        self.entities = []
        file_list = []
        if proc_single:
            if os.path.exists(filename):
                file_list.append(filename)            
            else:
                raise NeuroshareError(NSReturnTypes.NS_BADFILE,
                                      "input file does not exist: {0:s}".format(filename)) 
        else:
            # glob module seems to return files in reverse alphabetical order
            # We are reversing them here so that the file order with be similar
            # to that found in other Neuroshare codes.
            nsx_files = glob(filename[:-4] + '.ns[1-9]')
            nsx_files += glob(filename[:-4] + '.nf3')
#            nsx_files.reverse()
            file_list = glob(filename[:-4] + '.nev') + nsx_files
        
        if len(file_list) == 0:
            raise NeuroshareError(NSReturnTypes.NS_BADFILE,
                                  "could not find any .nev or .nsx files matching {0:s}".format(filename))
        for filename in file_list:
            parser = nsparser.ParserFactory(filename)
            file_data = FileData(parser)
            self._files.append(file_data)
            if parser.file_type == "NEURALEV":
                # All the work for the NEURALEV file is handled in the below function
                self._load_neuralev(file_data)
            elif parser.file_type == "NEURALSG":
                # units and scale are constant for all NSx2.1 files
                units = "V"
                header = parser.get_basic_header()
                # loop overall the channels found in the NSx2.1 basic header and
                # create an AnalogEntity for each
                for (channel_index, electrode_id) in enumerate(header.channel_id):
                    entity = AnalogEntity(parser, electrode_id, units, 
                                          channel_index, 1.0)
                    self.entities.append(entity)
                file_data.time_span = parser.time_span
                self._files.append(file_data)
            elif parser.file_type == "NEURALCD" or parser.file_type == "NEUCDFLT":
                header = parser.get_basic_header()
                # loop over each CC header found in the NEURALCD file.  Create an
                # AnalogEntity using the data found in those headers
                for (channel_index, header) in enumerate(parser.get_extended_headers()):
                    electrode_id = header.electrode_id
                    units = header.units.split('\0')[0]
                    label = header.electrode_label.split('\0')[0]            
                    # calculate the conversion between ADC counts and physical values.  
                    # This is will be needed when we read the analog waveforms  
                    scale = float(header.max_analog_value - header.min_analog_value)
                    scale = scale / (header.max_dig_value - header.min_dig_value)
                    entity = AnalogEntity(parser, electrode_id, units,
                                          channel_index, scale, label)
                    self.entities.append(entity)
                file_data.time_span = parser.time_span                    
            else:
                sys.stderr.write("invalid or corrupt nev file: {0:s}".format(filename))
                continue
        # shuffle all neural entities to the end of the entity list.  
        # This is not really needed put is consistent with the Neuroshare DLL 
        neural_entities = [e for e in self.entities if e.entity_type == EntityType.neural]
        self.entities = [e for e in self.entities if e.entity_type != EntityType.neural]
        self.entities += neural_entities
        # Reorder the analog entities to that decimated signals come first and the the
        # fully sampled signals come last
        # sample_freqs is a list of all the sample frequencies in order
        sample_freqs = sorted(set([ e.sample_freq for e in self.get_entities(EntityType.analog) ]))
        # find the last segment entity, insert the analog entities directly after this index
        segment_index = [index for index, e in enumerate(self.entities) if e.entity_type == EntityType.segment \
                         or e.entity_type == EntityType.event]    
        if len(segment_index) > 0:
            insert_index = segment_index[-1] + 1
        else:
            insert_index = 0
            
        analog_entities = [ e for e in self.get_entities(EntityType.analog) ]
        # filter out the non-analog entities
        self.entities = [e for e in self.entities if e.entity_type != EntityType.analog]
        # now we put them back in in the 
        for freq in sample_freqs:
            want_entities = [ e for e in analog_entities if e.sample_freq == freq ]
            for e in want_entities:
                self.entities.insert(insert_index, e)
                insert_index += 1
            
    def _load_neuralev(self, file_data):
        """A lot of work happens when NEV files are read.  This private 
        function was created to make the constructor more readable.  This 
        function facilitates the looping through NEV extended headers and 
        data packets and builds all the SegmentEntities, EventEntities, and 
        NeuralEntities.  This should only be called from the constructor.
        """
        parser = file_data.parser
        # entity search is used to organize the found entity by electrode_id
        # and count each spike event that we found
        entity_search = {}
        # When NEUEVLBL packets exist, they may come asynchronous from the NEUEVWAV
        # packets.  If the electrode found in the NEUEVLBL has not been seen yet
        # it will be stored in entity_label.  Then, we will add it to the entity
        # at the end.
        entity_labels = {}
        for header in parser.get_extended_headers():
            if header == None:
                sys.stderr.write("Warning: invalid nev header found\n")            
                continue
            # only create entities in the case of NEUEVWAV packets which
            # correspond to spike waveforms for now
            if header.header_type == "NEUEVWAV":
                entity = SegmentEntity(parser, header.packet_id)
                self.entities.append(entity)
                entity_search[entity.electrode_id] = entity
            elif header.header_type == "NEUEVLBL":
                if header.packet_id in entity_search.keys():
                    entity_search[header.packet_id].label = header.label.split("\0")[0]
                else:
                    # save the label and check on it later
                    entity_labels[header.packet_id] = header.label
            # Note: event entities often do not have DIGLABELs.  Because of
            # this we don't look at the DIGLABELS and just read through all
            # the data packets to see what digital events are found
             
        # check the number of data packets.  The segment entities will store
        # two integers for each piece of segment or event data.  It is unlikely
        # but this could grow limitless and fill up all available memory, possibly
        # causing issues with the user
        # BUG: This doesn't work...
        if USE_MEM_CHECK:
            phymem = psutil.avail_phymem() # physical memory in bytes
            neededmem = parser.n_data_packets * 8
            if neededmem > phymem:
                sys.stderr.write("warning: buffered memory may exceed available system memory\n") 
        # finish dealing with these entity_labels
        for (electrode_id, label) in entity_labels.iteritems():
            try:
                # set the label for corresponding entity, label is NULL terminated
                entity_search[electrode_id].label = label.split("\0")[0]
            except KeyError:
                # this should never happen
                sys.stderr.write("warning: Cannot find electrode: {0:d} for label {1:s}".format(electrode_id, label))
        # create a event entity dict to record event entities
        # These sometimes have event DIGLABELs and sometimes do not
        event_entities = {}
        # create a neural entity dict to record which channels 
        # saw which waveform classifications  
        neural_entities = {}
        # Look through all the NEV data packets and check for digital events and
        # see how many wave forms we have for each entity found in the headers
        # TODO: implement a fast read function that only returns the data needed
        # here.  NOT the entire packet.  This will need to include a kind of 
        # chunk read algorithm.  The needed info will be packet_id, reason or unit,
        # timestamp.
        # should look like 
        # for ipacket, ts, packet_id, unit in enumerate(parser.get_packet_header_info())  
        timestamp = 0
        for (ipacket, packet_data) in enumerate(parser.get_packet_headers()):
            timestamp = packet_data[0]
            packet_id = packet_data[1]
            # In the case of digital events, packet data refers to trigger "reason"
            unit = packet_data[2] 
            # packet_id == 0 is the case of a digital event
            if packet_id == 0:
                if not unit in event_entities.keys():
                    entity = EventEntity(parser, unit)
                    self.entities.append(entity)
                    event_entities[unit] = entity
                event_entities[unit].add_packet_data(timestamp, ipacket)
                # [2012/09/12 ELB] item_count is now incremented in add_packet_data
                # event_entities[unit].item_count += 1    
            # packet_id > 0 (corresponding to electrode_id) is a spike waveform event
            else:
                entity = entity_search[packet_id]
                entity.add_packet_data(timestamp, ipacket)
                # Note: require electrode id (packet_id) < 5120 to remove stim markers
                if packet_id > 5120:
                    continue
                # For each unit class we record the entities that have this 
                # classification. This results in the NeuralEntities and can 
                # be found with the get_neural_info function
                if not packet_id in neural_entities.keys():
                    neural_entities[packet_id] = {}                    
                # unit_class = unitpacket.unit_class
                if not unit in neural_entities[packet_id].keys():
                    neural_entities[packet_id][unit] = NeuralEntity(parser, entity.electrode_id, 
                                                                    unit, entity)
                neural_entities[packet_id][unit].item_count += 1
        for entity in self.entities:
            if entity.entity_type != EntityType.analog:
                entity.resize_packet_data()
        # If we are at the last event, record the timestamp.  These must
        # be time ordered so this most refer to the last piece of recorded data
        file_data.time_span = float(timestamp) / parser.timestamp_resolution
                            
        # Build a neural entity for each electrode and each unique unit_class found
        # in the data packets.  The neural_entity dict is looped over and the
        # neural entity is added to the _entities list
        for neural_key in sorted(neural_entities.keys()):
            entity_dict = neural_entities[neural_key]
            for entity_key in sorted(entity_dict.keys()):
                entity = entity_dict[entity_key]
                self.entities.append(entity)
        # Comment out this line to reproduce the behavior of the DLL.  Use this
        # line to reproduce the behavior of the Matlab code                    
        #self.entities = [e for e in self.entities if e.item_count > 0]

    def get_file_data(self, ext):
        """Utility function to get the FileData instance with the specified 
        file_type.  file_type should be one of nev, ns?
        
        Parameter:
        ext -- file extension
        
        Returns: Associated FileData instance or None if 
        it is not found
        """
        for file_data in self._files:
            if file_data.extension == ext:
                return file_data
        return None
    def has_file_type(self, file_type):
        """Checks to see if NEURALEV, NEURALSG, or NEURALCD type files
        were found
        """ 
        for file_data in self._files:
            if file_data.file_type == file_type:
                return True
        return False
    
    def get_entity_count(self):
        """Utilty function to return the number of entities found in 
        all the files.
        """
        return len(self.entities)
    
    def get_time(self):
        """return the datetime class corresponding to the origin time found 
        in nev files.  This corresponds to the starting time found from the
        system clock of the DAQ computer.  
        """
        info = self.get_file_info()
        time = datetime.datetime(info.time_year, info.time_month, info.time_day,
                                 info.time_hour, info.time_min, info.time_sec,
                                 info.time_millisec*1000,UTC())
        return time
    
    def get_entities(self, entity_type=None):
        """Return iterator to entity list.  If entity_type is specified return
        only entities of the desired type.  entity_type should be one of the
        members of EntityType
        
        Parameter:
        entity_type -- member from static class nsentity.EntityType, 
            default=None
            
        Returns: Iterator to entity list containing wanted entities 
        """
        for entity in self.entities:
            if entity_type != None:
                if entity.entity_type == entity_type:
                    yield entity
            else:   
                yield entity
    
    def get_entity(self, entity_index):
        """Return the entity specified by entity index"""
        try:
            return self.entities[entity_index]
        except:
            raise NeuroshareError(NSReturnTypes.NS_BADENTITY, 
                                  "invalid entity index: {0}".format(entity_index))
            
    def get_file_info(self):
        """equivalent function to the Neuroshare ns_GetFileInfo function.
        Returns: FileInfo namedtuple with ns_FILEINFO data
        """ 
        file_type = ""
        timestamp_resolution = 0.0
        time_span = 0.0
        year = 0
        month = 0
        day = 0
        hour = 0
        minute = 0
        second = 0
        millisec = 0 
        app_name = ""
        comment = ""
        # Putting a spaces at the end of this string makes 
        # this exactly equivalent the the Neuroshare DLL
        if self.has_file_type("NEURALEV"):
            file_type = "NEURALEV"
            if self.has_file_type("NEURALCD") or self.has_file_type("NEURALSG"):
                # Putting a space at the end of this string makes 
                # this exactly equivalent the the Neuroshare DLL
                file_type = "NEURALEV+ NEURAL"
        elif self.has_file_type("NEURALCD") or self.has_file_type("NEURALSG"):
            # Putting a space at the end of this string makes this exactly equivalent
            # the the Neuroshare DLL
            file_type = "NEURAL"  
        for f in self._files:
            if f.time_span > time_span: 
                time_span = f.time_span
            header = f.parser.get_basic_header()
            if header.header_type == "NEURALEV":
                year = header.time_origin.year
                month = header.time_origin.month
                day = header.time_origin.day
                hour = header.time_origin.hour
                minute = header.time_origin.minute
                second = header.time_origin.second
                millisec = header.time_origin.microsecond / 1000
                # This calculation of the timestamp_resolution creates a difference
                # when comparing to the BlackRock DLLs when looking at Ripple files.
                # The Ripple files store the timestamp_resolution as 1, however,
                # in most files (and in more recent Ripple files) this number should
                # be the same as sample_resolution and 30000
                timestamp_resolution = 1.0 / header.sample_resolution
                app_name = header.application.split("\0")[0]
                comment = header.comment.split("\0")[0]
        return FileInfo(file_type, self.get_entity_count(), timestamp_resolution,
                        time_span, app_name, year, month, day, hour, minute, second,
                        millisec, comment)

class UTC(datetime.tzinfo):
	"""UTC"""

	def utcoffset(self, dt):
		return datetime.timedelta(0) 

	def tzname(self, dt):
		return "UTC"

	def dst(self, dt):
		return datetime.timedelta(0)
