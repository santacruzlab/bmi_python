'''
Created on May 13, 2012

@author: Elliott L. Barcikowski

A collection of functions that dump a namedtuples resulting from the 
get_..._info functions from the Python Neuroshare API.  Placing
them in this area as they may be useful in the future. 
'''

def dump_file_info(info):
    pass

def dump_segment_info(info):
    pass

def dump_info(info):
    """An ugly, direct dump of any class.  A place-holder before a pretty
    dump routine is created for each of the info namedtuples
    """
    text = ""
    for (key, value) in info.__dict__.iteritems():
        text += "{0}: {1:6}\n".format(key[:15], value)
    return text 
    

