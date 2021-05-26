#!/usr/bin/env python
# Created on May 27, 2012
# @author: Elliott L. Barcikowski
"""Quick script to dump the info about all the electrodes found in the specified
.nev files.  This script dumps to the console the results of all the pyns  
versions of the Neuroshare API functions of the type ns_GetFileInfo, 
ns_GetEntityInfo, etc. 
"""
import argparse
import textwrap
from pyns.nsfile import NSFile
from pyns.nsentity import EntityType

import sys
import Tkinter
import tkFileDialog # if no file is specified, open a simple dialog

def dump_namedtuple(ntuple, use_full=False):
    """Utility function to dump namedtuples to stdout in an 
    easily readable form.
    """
    for key, value in ntuple._asdict().iteritems():
        if key == "header_type":
            continue
        if type(value) == str:
            if use_full:
                print "{0:<25} {1}".format(key, repr(value))
            else:            
                print "{0:<25} {1:s}".format(key, value.split('\0')[0])
        else:
            print "{0:<25} {1}".format(key, value)


if __name__ == '__main__':
    description = """pyns_dump is a simple command line tool for dumping 
the quantities in Neuroshare entities to stdout.  This program is provided 
to both provide a useful and to demonstrate the use of the pyns package.
"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-s", "--single", dest="single", default=False,
                        action="store_true", help="Only open single specified file")
    parser.add_argument("-k", "--skip", dest="skip", default=0,
                        help="Skip the first SKIP entities", type=int)
    parser.add_argument("-e", "--entity", dest="entity",help="only show entities of "\
                        "type ENTITY (analog, segment, event, neural)", default=None)
    parser.add_argument("filename", nargs="?", type=str, metavar="NEV_FILE",
                        help="use NEV_FILE as input (optional)", default=None)
    
    args = parser.parse_args()
    # check that we have an input file and that it opens successfully  
    if not args.filename:
        # if no file is specified, we'll open a simple tk file dialog and
        # and have the user specify a file
        root = Tkinter.Tk()
        root.withdraw()
        nev_formats = [ ("NEV", "*.nev"), ("NSx", "*.ns?") ]
        filename = tkFileDialog.askopenfilename(title="Choose a file",
                                                filetypes=nev_formats)
        if len(filename) == 0:
            parser.error("selected invalid NEV or NSx file.")
    else:
        filename = args.filename
    nsfile = NSFile(filename, args.single)
    if nsfile == None:
        parser.error("failed to open file: {0:s}".format(args[0])) 
    # check that entity option is okay
    want_entity = None 
    if args.entity != None:
        entity_strings = [EntityType.get_entity_string(eid) for eid in range(1, 5)]
        if not args.entity in entity_strings:
            parser.error("invalid entity: {0:s}".format(args.entity))
        else:
            want_entity = EntityType.get_entity_id(args.entity)            
    # NSFile.get_file_info() is the equivalent of the Neuroshare 
    # API function ns_GetFileInfo  
    print "====================================="
    print " File Info"
    print "====================================="
    dump_namedtuple(nsfile.get_file_info())

    skipped = 0
    for index, entity in enumerate(nsfile.entities):
        
        # check if we have the entity we want
        if want_entity:
            if entity.entity_type != want_entity:
                continue

        # allow for skipping entities
        if skipped < args.skip:
            skipped += 1
            continue

        # pyns.Entity.get_entity_info is the equivalent of the Neuroshare 
        # function ns_GetEntityInfo
        entity_info = entity.get_entity_info()
        if entity.entity_type == EntityType.segment:
            # pyns.SegmentEntity.get_segment_info is the equivalent of 
            # the Neuroshare function ns_GetSegmentInfo
            print "====================================="
            print "Entity Index: {0} Type: Segment".format(index)
            print "====================================="
            dump_namedtuple(entity_info)
            dump_namedtuple(entity.get_segment_info())
        elif entity.entity_type == EntityType.analog:
            # pyns.AnalogEntity.get_analog_info is the equivalent of 
            # the Neuroshare function ns_GetAnalogInfo            
            print "====================================="
            print "Entity Index: {0} Type: Analog".format(index)
            print "====================================="
            dump_namedtuple(entity_info)
            dump_namedtuple(entity.get_analog_info())
        elif entity.entity_type == EntityType.neural:
            # pyns.Neural.get_neural_info is the equivalent of 
            # the Neuroshare function ns_GetNeuralInfo            
            print "====================================="
            print "Entity Index: {0} Type: Neural".format(index)
            print "====================================="
            dump_namedtuple(entity_info)
            dump_namedtuple(entity.get_neural_info())

        elif entity.entity_type == EntityType.event:
            # pyns.Neural.get_neural_info is the equivalent of 
            # the Neuroshare function ns_GetNeuralInfo
            print "====================================="
            print "Entity Index: {0} Type: Event".format(index)
            print "====================================="
            dump_namedtuple(entity_info)
            dump_namedtuple(entity.get_event_info())
        else:
            sys.stderr.write("warning: invalid entity type: {0:d}\n".format(entity.entity_type))
