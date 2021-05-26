#!/usr/bin/env python
# Created on May 27, 2012
# @author: Elliott L. Barcikowski
'''Quick script to dump the contents of headers and extended headers in NEV and 
and NSx files for debugging purposes.
'''
from optparse import OptionParser
from pyns.nsparser import ParserFactory

import os
from glob import glob
import Tkinter, tkFileDialog # if no file is specified, open a simple dialog

def dump_namedtuple(ntuple, use_full=False):
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
    parser = OptionParser(usage="usage: %prog [OPTIONS..] [NEV_INPUT_FILE]")
    parser.add_option("-s", "--single", dest="single", default=False,
                      action="store_true", help="Only open single specified file")
    parser.add_option("-f", "--full", dest="full", default=False,
                      action="store_true", help="For string fields, show all characters")    
    
    (options, args) = parser.parse_args()
    # check that we have an input file and that it opens successfully  
    if len(args) == 0:
        # if no file is specified, we'll open a simple tk file dialog and
        # and have the user specify a file
        root = Tkinter.Tk()
        root.withdraw()
        nev_formats = [ ("NEV", "*.nev"), ("NSX", "*.ns?") ]
        filename = tkFileDialog.askopenfilename(title="Choose a file",
                                                filetypes=nev_formats)
    else:
        filename = args[0]
    if options.single:
        file_list = [filename]
    else:
        nsx_files = glob(filename[:-4] + '.ns[1-9]')
        file_list = glob(filename[:-4] + '.nev') + nsx_files
    
    for input_file in file_list:
        parser = ParserFactory(input_file)
        file_banner = "File: {0:s} Type: {1:s}".format(os.path.basename(input_file),
                                                       parser.file_type)
        print "="*len(file_banner)
        print file_banner
        print "="*len(file_banner)
        
        basic_header = parser.get_basic_header()
        header_banner = "{0:<8s} Header".format(basic_header.header_type) 
        print '-'*len(header_banner)
        print header_banner
        print '-'*len(header_banner)
                           
        dump_namedtuple(parser.get_basic_header(), options.full)
        
        if parser.file_type == "NEURALEV" or parser.file_type == "NEURALCD":
            
            for header_index, ext_header in enumerate(parser.get_extended_headers()):
                ext_header_banner = "Extended Header #{0:3d}: {1:<8s}".format(header_index,
                                                                              ext_header.header_type)
                print '-'*len(ext_header_banner)                
                print ext_header_banner
                print '-'*len(ext_header_banner)
                
                dump_namedtuple(ext_header, options.full)
            