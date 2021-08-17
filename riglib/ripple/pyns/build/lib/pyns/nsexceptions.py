# Created on May 29, 2012
# @author: Elliott L. Barcikowski
'''Collection of Error classes and return values replicating the Neuroshare 
return values.

Throughout the pyns code, instead of making use of return types, Python 
exceptions are raised when function fail.  The exceptions in pyns classes
will always be of the type NeuroshareError.  

As often as possible, exceptions are raised with an error number consistent
with the Neuroshare API.  These are defined in the static class 
NSReturnTypes
'''

class NSReturnTypes:
    """Neuroshare standard return values."""
    NS_OK = 0 # we raise exceptions instead of returning zero on success
    NS_LIBERROR = -1 # liberror will not get returned by these class 
    NS_TYPEERROR = -2
    NS_FILEERROR = -3
    NS_BADFILE = -4
    NS_BADENTITY = -5
    NS_BADSOURCE = -6
    NS_BADINDEX = -7    

class NeuroshareError(Exception):
    """Exception class to raise Neuroshare errors."""
    