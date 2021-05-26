#!/usr/bin/env python
"""pyns_plot is a simple command line plotting tool (making use of matplotlib)
to index through plots of Neuroshare entities.  This tool is provided mostly
to give a usage example of pyns, the Python Neuroshare API.  The user may
browse through Neuroshare entities by using the keys: n - next entity,
p - previous entity, and q - to quit the program.
"""    
# Created on May 27, 2012
# @author: Elliott L. Barcikowski
import argparse
import sys
import Tkinter
import tkMessageBox
import tkFileDialog # if no file is specified, open a simple dialog
import numpy
from matplotlib import pyplot #@UnresolvedImport
from matplotlib import rc
from pyns.nsfile import NSFile
from pyns.nsentity import EntityType 
import time
import textwrap

class PlotManager:
    """class to draw entities and handle events from user.  The pyplot
    canvas help by this class is controlled with mouse clicks (to 
    advance to the next entity) or by the key-strokes (q, p, and n)
    q causes the application to quit, n advances to the next entity,
    and p goes back to the previous entity.
    """
    def __init__(self, entities, skip=0, 
                 max_segment=100, max_analog=90000):
        self.entities = entities
        self.current_entity = skip
        
        if skip >= len(self.entities):
            return
        
        self.fig = pyplot.figure()
        self.ax = self.fig.add_subplot(111)
        self.fig.canvas.mpl_connect('key_press_event', self.onpress)
        
        self.current_segment = 0
        self.current_analog = 0
        
        self.max_segment = max_segment
        self.max_analog = max_analog
        
        # draw the first entity.  If it has zero items, or is not
        # a segment or analog, go on to the next
        if not self.draw_entity():
            self.next()
        
    def onpress(self, event):
        if event.key not in ('n', 'p', 'q', 't', 'b'): return
        if event.key == 'n':
            self.next()
        if event.key == 'p':
            self.previous()
        if event.key == 't':
            self.forward()
        if event.key == 'b':
            self.back()
        if event.key == 'q':
            if tkMessageBox.askokcancel('pyns_plot', 'Quit pyns_plot?'):
                sys.exit(0)

    def previous(self):
        if self.current_entity != 0:
            self.current_entity -= 1
            # if we fail to draw an entity (basically if the 
            # entity has zero items keep iterating back
            if not self.draw_entity():
                self.previous()

    def next(self):
        self.current_analog = 0
        self.current_segment = 0
        
        if self.current_entity < len(self.entities) - 1:
            self.current_entity += 1
            # if we fail to draw an entity (basically if the 
            # entity has zero items keep iterating forward)
            if not self.draw_entity():
                self.next()
        else:
            sys.exit(0)
            
    def draw_event_entity(self):
        """Handles drawing digital events.  For now this always
        draws the digital port output.  For now we only plot the
        output for the digital input port
        """
        entity = self.entities[self.current_entity]
        if entity.entity_type != EntityType.event:
            sys.stderr.write("must specify event entity\n")
            return False
        item_count = min(entity.item_count, self.max_analog)
        title = "16 bit digital input events"
        if item_count == 0:
            return False

        self.ax.cla()
        
        self.ax.set_title(title)
        self.ax.set_xlabel('[s]')
        self.ax.set_ylabel('')
        
        timestamps = numpy.zeros(item_count, dtype=numpy.double)
        values = numpy.zeros(item_count, dtype=numpy.uint16)
        for item in xrange(0, item_count):
            data_rt = entity.get_event_data(item)
            timestamps[item] = data_rt[0]
            values[item] = 0
            for iData in range(0, 6):
                if data_rt[1][iData] != 0:
                    values[item] = data_rt[1][iData]
                    break

        self.ax.scatter(timestamps, values)

        self.fig.canvas.draw()
        return True
    
    def draw_entity(self):
        """This function is called each time an entity is called to draw.
        It looks at the entity and calls the appropriate function.
        """
        entity = self.entities[self.current_entity]
        if entity.entity_type == EntityType.segment:
            return self.draw_segment_entity()
        elif entity.entity_type == EntityType.analog:
            return self.draw_analog_entity()
        elif entity.entity_type == EntityType.event:
            return self.draw_event_entity()        
        # return false if we don't have an analog or segment entity
        return False

    def draw_segment_entity(self):
        """A utility function to overlay all the segment waveforms for a
        a specified segment entity. 
        """
        entity = self.entities[self.current_entity]
        if entity.entity_type != EntityType.segment:
            sys.stderr.write("must specify segment entity\n")
            return False
        item_count = min(entity.item_count - self.current_segment, self.max_segment)
        title = "Overlay of {0:d} segment waveforms for {1:s}".format(item_count, entity.label)
        if item_count <= 0:
            return False

        self.ax.cla()
    
        # get the time resolution though it should always be 1/30kHz
        segment_info = entity.get_segment_info()

        self.ax.set_title(title)
        
        self.ax.set_xlabel('[ms]')
        self.ax.set_ylabel('[uV]')
        for item in range(self.current_segment, self.current_segment + item_count):
            # get the segment info for the time resolution, 
            # though it should always be 30000 
            segment_info = entity.get_segment_info()
            (timestamp, waveform, a) = entity.get_segment_data(item)
            # create physical time dimensions in milliseconds                
            time = numpy.arange(0, len(waveform), dtype=numpy.double)
            time *= 1.0/segment_info.sample_rate
            time *= 1000.0
            self.ax.plot(time, waveform)

        self.fig.canvas.draw()
        return True

    def forward(self):
        
        self.current_analog = min(self.current_analog + self.max_analog,
                                  self.entities[self.current_entity].item_count)
        self.current_segment = min(self.current_segment + self.max_segment,
                                  self.entities[self.current_entity].item_count)        
        self.draw_entity()
    
    def back(self):
        self.current_analog = max(self.current_analog - self.max_analog, 0)
        self.current_segment = max(self.current_segment - self.max_segment, 0)
        
        self.draw_entity()
        
        
    def draw_analog_entity(self):
        """A utility function to draw any analog entity"""
        
        entity = self.entities[self.current_entity]
        if entity.entity_type != EntityType.analog:
            sys.stderr.write("must specify analog entity\n")
            return False
    
        self.ax.cla()        
        item_count = min(self.max_analog, entity.item_count - self.current_analog)
        if item_count <= 0:
            return False

        # The analog info will be useful to get time axis
        # resolution (i.e., ns2 or ns5 files)
        analog_info = entity.get_analog_info()
        entity_info = entity.get_entity_info()
        title = "Analog data for {0}".format(entity_info.label)
        
        if item_count < entity.item_count:
            title += "(max {0:d} bins)".format(item_count)

        self.ax.set_title(title)
        self.ax.set_xlabel('[s]')
        self.ax.set_ylabel('[uV]')
            
        waveform = entity.get_analog_data(self.current_analog, item_count)
        # create physical time dimensions in seconds                
        time = numpy.arange(self.current_analog, self.current_analog + len(waveform), 
                            dtype=numpy.double)
        time *= 1.0/analog_info.sample_rate
        #waveform *= analog_info.resolution
        self.ax.plot(time, waveform)
        
        self.fig.canvas.draw()

        return True

if __name__ == '__main__':
    description = """pyns_plot is a simple command line plotting tool 
(making use of matplotlib) to index through plots of Neuroshare entities.  
This tool is provided mostly to give a usage example of pyns, the Python 
Neuroshare API.  The user may browse through Neuroshare entities by using 
the keys: n - next entity, p - previous entity, and q - to quit the program.
"""    
    parser = argparse.ArgumentParser(description=description)
                                     
    parser.add_argument("-k", "--skip", dest="skip",
                        help="skip the first SKIP entities", default=0, type=int)
    parser.add_argument("-a", "--analog-only", dest="analog_only", default=False,
                        action="store_true", help="Only show waveforms for analog entities")
    parser.add_argument("-s", "--segment-only", dest="segment_only", default=False,
                        action="store_true", help="Only show waveforms for segment entities")
    parser.add_argument("-e", "--event-only", dest="event_only", default=False,
                        action="store_true", help="Only show digital event data")    
    parser.add_argument("-m", "--max-segments", dest="max_segments", default=100,
                        type=int, help="Only overlay MAX_SEGMENTS spikes")
    parser.add_argument("-n", "--max-analog", dest="max_analog", default=240000,
                        type=int, help="Only overlay MAX_ANALOG time bins")
    parser.add_argument('filename', nargs="?", type=str, metavar="NEV_FILE",
                        help="use NEV_FILE as input (optional)", default=None)
    
    args = parser.parse_args()
    # check options conflicts
    if args.segment_only and args.analog_only:
        parser.error("can't specify both analog-only and segment-only options")

    root = Tkinter.Tk()
    root.withdraw()    

    message = """pyns_plot shows the power and ease of using Python for quick analysis applications.
    Browse Neuroshare entities using the following keys: 
        'n' - next entities, 
        'p' - previous entity, 
        't' - forward in time,
        'b' - back in time,
        'q' - quit pyns_plot.
        """
    tkMessageBox.showinfo('pyns_plot Help', message)    
    
    # check that we have an input file and that it opens successfully  
    if not args.filename:
        # if no file is specified, we'll open a simple tk file dialog and
        # and have the user specify a file

        nev_formats = [ ("NEV", "*.nev"), ("NSx", "*.ns?"), ("NFx", "*.nf3") ]
        filename = tkFileDialog.askopenfilename(title="Choose a file", filetypes=nev_formats)
        #
        if len(filename) == 0:
            parser.error('selected invalid NEV or NSx file.')
    else:
        filename = args.filename
    nsfile = NSFile(filename)
    if nsfile == None:
        parser.error("failed to open file: {0:s}".format(args[0]))

    if args.analog_only:
        wanted_entities = [e for e in nsfile.get_entities(EntityType.analog)]
    elif args.segment_only:
        wanted_entities = [e for e in nsfile.get_entities(EntityType.segment)]
    elif args.event_only:
        wanted_entities = list(nsfile.get_entities(EntityType.event))
    else:
        wanted_entities = [e for e in nsfile.get_entities()]

    manager = PlotManager(wanted_entities, args.skip, args.max_segments, args.max_analog)

    pyplot.show()
