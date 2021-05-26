#!/usr/bin/env python 
# Created on May 17, 2012
# @author: Elliott L. Barcikowski
'''Simple example script used to print out some bad characters in 
labels found in some Ripple files. 
'''
from pyns.nsfile import NSFile
from pyns.nsentity import EntityType

BAD_FILE = "/home/elliottb/ripple/test_data/datafile0002.nev"

if __name__ == '__main__':
    f = NSFile(BAD_FILE)
    print "for file: {0:s}".format(f._files[1].parser.fid.name)
    for e in f.get_entities(EntityType.analog):
        cc = e.get_extended_header()
        print "{0:03d} ".format(e.channel_index),
        print "elecid: {0:03d}".format(e.electrode_id),
        print "labelbytestring: {0:8s}".format(repr(cc.electrode_label)),                
        print "label: {0:8s}".format(cc.electrode_label)

        
