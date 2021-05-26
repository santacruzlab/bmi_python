"""The pyns package native Python Neuroshare implementation.

========
Overview
========
The classes and functions provided by this interface do not exactly 
replicate the functions provided by the Neuroshare API.  Instead of 
the functional Neuroshare API generally written in c, 
the pyns package was created using an object oriented programming style with 
a focus on having a form natural to the Python language.  

This project was put in place by Ripple LLC., after noticing the growing
use of the Python language in the academic and scientific communities.

=======================
Required Python Modules
=======================
The pyns package makes use of a few Python modules outside the default
modules that would be present in any Python installation.  These 
modules are standard for any scientific or computational
use of the Python language and are likely to be present on your system.

These modules are:

* numpy - A collection of array manipulation objects and functions.
When data is returned from data in the pyns package, numpy arrays 
are used to package the data for easy use with matplotlib.

* matplotlib - A collection of plotting and analysis objects and functions.  
These are not used explicitly in the pyns package, 
but they are used in all the provided examples.

* psutil - A module containing process utilities for Windows, Linux, and OS X
This module is used to check the available system memory.  Cached data 
for spike waveforms has the potential to arbitrarily large, and a check is made
to ensure enough physical memory is present on a system.  However, files 
large enough to cause problems are unlikely.

On a Windows system the Python distribution Python(x, y) makes installing
these modules and other useful analysis packages easy.  More information 
may be found at: http://www.pythonxy.com.

If using Linux platforms, the needed packages depend slightly on the 
Linux flavor, though they represent the same libraries.  On Ubuntu install 
the following packages: python-psutil, python-numpy, python-matplotlib.  
They may be installed using apt-get with the following command::

    $ sudo apt-get install python-psutil python-numpy python-matplotlib

If using a Red Hat variant, install the packages python-psutil, 
python-matplotlib, and numpy.  These may be installed using yum with the 
command::

    $ sudo yum install python-psutil numpy python-matplotlib

============
Installation
============
To install pyns system-wide on your machine either run the command line
setup.py.  This command should install successfully on any platform.  If run
in bash, this would look like::

    $ python setup.py install

If using a Windows system, a msi, produced by Python distutils, is also
available.

==============================
Neuroshare API to pyns Package
==============================
For those familiar the standard Neuroshare API, the following is a mapping 
of the traditional Neuroshare functions to their pyns equivalent.  The
arguments and return values are generally not exactly the same due to the
object oriented nature of the pyns package.  See the docstrings of the
classes and functions for more information.  Also, some examples of
general use of pyns are provided towards the end of this section.

* ns_OpenFile - pyns equivalent: :class:`pyns.NSFile`  
* ns_GetFileInfo - pyns equivalent: :meth:`pyns.NSFile.get_file_info` 
* ns_GetEntityInfo - pyns equivalent: :meth:`pyns.nsentity.Entity.get_entity_info`
* ns_GetSegmentInfo - pyns equivalent: :meth:`pyns.nsentity.SegmentEntity.get_seg_source_info`
* ns_GetSegmentSourceInfo - pyns equivalent: :meth:`pyns.nsentity.SegmentEntity.get_seg_source_info`
* ns_GetEventInfo - pyns equivalent: :meth:`pyns.nsentity.EventEntity.get_event_info`
* ns_GetAnalogInfo - pyns equivalent: :meth:`pyns.nsentity.AnalogEntity.get_analog_info`
* ns_GetNeuralInfo - pyns equivalent: :meth:`pyns.nsentity.NeuralEntity.get_neural_info`
* ns_GetEventData - pyns equivalent: :meth:`pyns.nsentity.EventEntity.get_event_info`
* ns_GetSegmentData - pyns equivalent: :meth:`pyns.nsentity.SegmentEntity.get_segment_data`
* ns_GetAnalogData - pyns equivalent: :meth:`pyns.nsentity.AnalogEntity.get_analog_data`
* ns_GetEventData - pyns equivalent: :meth:`pyns.nsentity.EventEntity.get_event_data`
* ns_GetTimeByIndex - pyns equivalent: :meth:`pyns.nsentity.Entity.get_time_by_index`
* ns_GetIndexByTime - pyns equivlanent: :meth:`pyns.nsentity.Entity.get_index_by_time`

========
Examples
========
A few examples are provided to ease new users to pyns and Python to
this package.

------------------
Simple Entity Dump
------------------
This example will open a nev file, find the associated .nsx files and
print out the result of the :meth:`pyns.nsentity.Entity.get_entity_info` 
function for each entity that was found.
::

    \"\"\"This quick example opens a .nev file named 'test_data.nev'
    and prints the ns_EntityInfo struct for each entity.
    \"\"\"
    from pyns.nsfile import NSFile

    nsfile = NSFile('test_data.nev')
    for entity in nsfile.get_entities():
        print entity.get_info()

--------------
Spike Plotting
--------------
Make use of matplotlib to plot the first electrode channel was found
to have recorded spike events.  This example makes use of the pyns functions
:meth:`pyns.nsfile.NSFile.get_entities` and 
:meth:`pyns.nsentity.SegmentEntity.get_segment_data`.
::

    \"\"\"Example of using matplotlib to plot Neuroshare data, retrieved
    with pyns.  Makes use of a fictious nev file 'test_data.nev'.
    \"\"\"
    from matplotlib import pyplot
    import numpy
    import pyns
    nsfile = pyns.NSFile('test_data.nev')
    # get segment entities
    segment_entities = [e for e in nsfile.get_entities() if e.entity_type == 3]
    segment_entity = None
    # pick the first segment entity with a non-zero item_count
    for entity in segment_entities:
        if entity.item_count > 0:
            segment_entity = entity
            break
        t = numpy.arange(0, 52, dtype=float)/30
    # overlay all the spikes for this entity
    for item in range(0, segment_entity.item_count):
        (ts, data, unit_id) = segment_entity.get_segment_data(item)
        pyplot.plot(t, data)
    pyplot.show()

---------------------------
Plotting Continous Channels
---------------------------
Make use of matplotlib to plot the first continous channel.  This example
makes use of the pyns functions :meth:`pyns.nsfile.NSFile.get_entities`, 
:meth:`pyns.nsentity.AnalogEntity.get_analog_info`, and
:meth:`pyns.nsentity.AnalogEntity.get_analog_data`.
::

    \"\"\"Example of plotting Analog entities (continuous channels) using pyns
    and matplotlib.  Makes use of a fictious nev file 'test_data.ns2'.
    \"\"\"
    from matplotlib import pyplot
    import numpy
    import pyns

    nsfile = pyns.NSFile('test_data.ns2')
    # get segment entities
    analog_entities = [e for e in nsfile.get_entities() if e.entity_type == 2]
    analog_entity = None
    # pick the first analog entity with a non-zero item_count
    for entity in analog_entities:
        if entity.item_count > 0:
            analog_entity = entity
            break
    if analog_entity == None:
        exit(0)
    analog_info = analog_entity.get_analog_info()
    # only get the first 50k data points (other wise we could be reading
    # lots and lots of data in memory at once
    data = analog_entity.get_analog_data(0, 50000)
    t = numpy.arange(0, len(data), dtype=float)/analog_info.sample_rate
    pyplot.plot(t, data)
    pyplot.xlabel('[s]')
    pyplot.ylabel('[uV]')
    pyplot.show()

================
Revision History
================
-----------
Version 0.5
-----------
* Feature: Added support for float based continuous files (.nf3)
* Feature: Added support for stimulation markers returned as segment entities

-----------
Version 0.4
-----------
* First publicly released pyns version
* Provides a complete version of Neuroshare API
"""

# The NSFile is the standard entry point for this package and 
# it will be made easily accessible by importing it from pyns.
from nsfile import NSFile
