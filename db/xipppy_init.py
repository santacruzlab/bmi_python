"""
XIPP communications with Trellis and Grapevine.

Before using any other library features, call open() and call close() when
done. There can only be one connection open at a time; it may sometimes be
useful to close() and re-open() the network connection.

Most functions will raise a XippPyException on error. The underlying library
does not offer useful information on the cause of an error in most cases,
so these (unfortunately) tend to lack additional information.

Property-setting functions (like stim_enable_set(), signal_set() or
filter_set()) are asynchronous- the change will not take effect immediately
since it must be sent across the XIPP connection. If it is important that
a property change be applied before the program continues, you should poll
for the desired value:

    stim_enable_set(1)
    for _ in range(1,100):
        time.sleep(.01)
        if stim_enable() == 1:
            break
    else:
        raise Exception("Timed out waiting for stim enable to take effect")
"""
import array
import ctypes
import itertools
import logging
import time as _time

import xipppy_capi as _c
from . import exception
from .fast_settle import (
    fast_settle,
    fast_settle_get_choices,
    fast_settle_get_duration,
)
from .filter import (
    SosFilterDesc,
    SosStage,
    filter_get_desc,
    filter_list_names,
    filter_list_selection,
    filter_set,
    filter_set_custom
)
from .stim import (
    StimSegment,
    StimSeq,
    stim_enable,
    stim_enable_set,
    stim_get_res,
    stim_set_res
)
from .transceiver import (
    TransceiverCmdHeader,
    TransceiverRegisterAddrs,
    TransceiverStatus,
    TransceiverCommand,
    TRANSCEIVER_STATUS_COUNTER_MAX,
    ImplantRegisterAddrs,
    POWER_STATE_R3,
    DEFAULT_SERVO_DAC_LEVEL,
    transceiver_status,
    transceiver_command,
    transceiver_enable,
    transceiver_power_servo_enable,
    transceiver_get_power_state,
    transceiver_get_implant_serial,
    transceiver_get_implant_voltage,
    transceiver_get_ir_led_level,
    transceiver_get_ir_received_light,
    transceiver_set_implant_servo_dac
)
from .mira import (
    MiraImplantCmdMode
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Helper types
class SegmentDataPacket:
    """
    Data type to mirror classic "Segments" for spike data
    """

    def __init__(self, spk=None):
        if spk is None:
            self.timestamp = 0
            self.class_id = 0
            self.wf = array.array('h', itertools.repeat(0, 52))
        else:
            self._spk = spk
            self.timestamp = spk.timeDooDas
            self.class_id = spk.class_id
            self.wf = _c.I16Array.frompointer(spk.i16)

class SegmentEventPacket:
    """
    Data type to mirror classic "Segments" for digital data
    """

    def __init__(self, event=None):
        if event is None:
            self.timestamp = 0
            self.reason = 0
            self.parallel = 0
            self.sma1 = 0
            self.sma2 = 0
            self.sma3 = 0
            self.sma4 = 0
        else:
            self.timestamp = event.timeDooDas
            self.reason = event.reason
            self.parallel = event.parallel
            self.sma1 = event.sma1
            self.sma2 = event.sma2
            self.sma3 = event.sma3
            self.sma4 = event.sma4


class TrialDescriptor:
    """
    Data type to mirror XippTrialDesc_t for ease of use
    """

    def __init__(self, val=None):

        self._oper = 150
        self._status = None
        self._file_name_base = None
        self._auto_stop_time = None
        self._auto_incr = None
        self._incr_num = None

        # here's where we'll keep the actual trial descriptor
        self._desc = _c.XippTrialDescriptor_t()
        self._desc.oper = self._oper
        self._desc.status_size = int(0)
        self._desc.file_name_size = int(0)

        if val is not None:
            self.oper = val.oper

            if val.status is not None:
                self.status = str(val.status)

            if val.file_name_base is not None:
                self.file_name_base = str(val.file_name_base)

            if val.auto_stop_time is not None:
                self.auto_stop_time = int(val.auto_stop_time)

            if val.auto_incr is not None:
                self.auto_incr = int(val.auto_incr)

            if val.incr_num is not None:
                self.incr_num = int(val.incr_num)

    @property
    def oper(self):
        if self._oper != self._desc.oper:
            self._oper = self._desc.oper
        return self._oper

    @oper.setter
    def oper(self, value):
        self._oper = int(value)
        self._desc.oper = self._oper

    @property
    def status(self):
        if self._status is None:
            return None

        s = ctypes.string_at(self._status.buffer_info()[0])
        return s.decode('utf-8')

    @status.setter
    def status(self, value):
        self._status = array.array('b', bytearray(str(value) + "\0", 'utf-8'))

        self._desc.status = self._status.buffer_info()[0]
        self._desc.status_size = len(self._status)

    @property
    def file_name_base(self):
        if self._file_name_base is None:
            return None

        s = ctypes.string_at(self._file_name_base.buffer_info()[0])
        return s.decode('utf-8')

    @file_name_base.setter
    def file_name_base(self, value):
        self._file_name_base = array.array(
            'b',
            bytearray(str(value) + "\0",
                      'utf-8')
        )
        self._desc.file_name_base = self._file_name_base.buffer_info()[0]
        self._desc.file_name_base_size = len(self._file_name_base)

    @property
    def auto_stop_time(self):
        if self._auto_stop_time is not None:
            return self._auto_stop_time.value
        else:
            return None

    @auto_stop_time.setter
    def auto_stop_time(self, value):
        self._auto_stop_time = ctypes.c_uint(value)
        self._desc.auto_stop_time = ctypes.addressof(self._auto_stop_time)

    @property
    def auto_incr(self):
        if self._auto_incr is not None:
            return self._auto_incr.value
        else:
            return None

    @auto_incr.setter
    def auto_incr(self, value):
        self._auto_incr = ctypes.c_uint(value)
        self._desc.auto_incr = ctypes.addressof(self._auto_incr)

    @property
    def incr_num(self):
        if self._incr_num is not None:
            return self._incr_num.value
        else:
            return None

    @incr_num.setter
    def incr_num(self, value):
        self._incr_num = ctypes.c_uint(value)
        self._desc.incr_num = ctypes.addressof(self._incr_num)

    def cast(self):
        return self._desc


#
# open / close
#

def _open(use_tcp=False):
    """
    The xipppy_open context manager should be used whenever possible to open
    a connection to xipppy.

    See help(xipppy.xipppy_open) for an example.

    Multiple calls to _open will have no effect. Multiple _open calls do not
    require multiple matching _close calls.
    """

    xipppy_open._user_count += 1
    if xipppy_open._user_count == 1:
        logger.debug('Opening xipppy')
        open_fn = _c.xl_open if not use_tcp else _c.xl_open_tcp
        if open_fn() != 0:
            xipppy_open._user_count = 0
            raise exception.XippPyException(
                "Library already open or failed to initialize")
    else:
        return 0


def _close():
    """
    Immediately close any open connections to the NIP. _close should not be
    called from within a xipppy_open context.
    """
    logger.debug('_close called, setting user count to 0')
    xipppy_open._user_count = 0
    return _c.xl_close()


class xipppy_open:
    """
    This is a context manager for xipppy. This is the prefered way to manage
    connections to an NIP.

    For example, to simply test the connection to the NIP:
    >>> with xipppy_open():
    ...     pass

    This context is reentrant so the following is valid (exactly one call to
    open and close is made)
    >>> def print_nip_time():
    ...     with xipppy_open():
    ...         print(time())
    >>> with xipppy_open():
    ...     print_nip_time()

    The reentrant property of this context is useful if you are building
    complex libraries on top of xipppy. You can safely wrap function calls
    that make xipppy calls in this context without
    having to detect if xipppy is already open.
    """
    _user_count = 0

    def __init__(self, use_tcp=False):
        self.use_tcp = use_tcp

    def __enter__(self):
        # Note that _open actually increments _user_count
        _open(self.use_tcp)

    def __exit__(self, type, value, traceback):
        # The following logic may seem odd, like this counter will never reach
        # zero ... but _close always sets _user_count to 0 because a closed
        # resource can't have any users.
        if xipppy_open._user_count == 0:
            logger.warning(
                'Detected a call to _close within a xipppy context. Avoid'
                'this, it can crash your program.')
        elif xipppy_open._user_count == 1:
            _close()
        elif xipppy_open._user_count > 0:
            xipppy_open._user_count -= 1

        if type:
            logger.exception('Exception occured in xipppy context',
                             exc_info=(type, value, traceback))


#
# data functions
#

def _cont_base(cfn, npoints, elecs, start_timestamp):
    # Build C arrays to store output and the input electrode array
    data_out = array.array('f', itertools.repeat(0, npoints * len(elecs)))
    (data_ptr, _) = data_out.buffer_info()

    elecs_in = array.array('I', elecs)
    (el_ptr, el_len) = elecs_in.buffer_info()

    ts_out = array.array('I', [1])
    (ts_ptr, _) = ts_out.buffer_info()

    # Call C
    npoints = cfn(ts_ptr, data_ptr, npoints, el_ptr, el_len, start_timestamp)

    # If data came back, trim to actual length gotten and return. Otherwise
    # nothing.
    del data_out[npoints * len(elecs):]
    return data_out, ts_out[0] if start_timestamp != 0 else 0


def cont_raw(npoints, elecs, start_timestamp=0):
    """
    Retrieve raw data (sampled at 30 kHz).

    Returns a tuple of (timestamp, data) where the timestamp is that of the
    first data, and data is a list of data points. If start_timestamp is not
    specified, the output timestamp is always 0.
    Args:
        npoints: number of datapoints to retrieve
        elecs:  list of electrodes to sample
        start_timestamp: NIP timestamp to start data at, or most recent if 0

    Returns:
    """
    return _cont_base(_c.xl_cont_raw, npoints, elecs, start_timestamp)


def cont_hires(npoints, elecs, start_timestamp=0):
    """
    Retrieve hires data (sampled at 2 kHz).

    Parameters and outputs are the same as the `cont_raw` function.
    Args:
        npoints: number of datapoints to retrieve
        elecs:  list of electrodes to sample
        start_timestamp: NIP timestamp to start data at, or most recent if 0

    Returns:
    """
    return _cont_base(_c.xl_cont_hires, npoints, elecs, start_timestamp)


def cont_hifreq(npoints, elecs, start_timestamp=0):
    """
    Retrieve hires data (sampled at 7.5 kHz).

    Parameters and outputs are the same as the `cont_raw` function.
    Args:
        npoints: number of datapoints to retrieve
        elecs:  list of electrodes to sample
        start_timestamp: NIP timestamp to start data at, or most recent if 0

    Returns:
    """
    return _cont_base(_c.xl_cont_hifreq, npoints, elecs, start_timestamp)


def cont_lfp(npoints, elecs, start_timestamp=0):
    """
    Retrieve lfp data (sampled at 1 kHz).

    Parameters and outputs are the same as the `cont_raw` function.

    If analog I/O is requested, the unfiltered analog data is retrieved, not
    filtered LFP data.
    Args:
        npoints: number of datapoints to retrieve
        elecs:  list of electrodes to sample
        start_timestamp: NIP timestamp to start data at, or most recent if 0

    Returns:
    """
    return _cont_base(_c.xl_cont_lfp, npoints, elecs, start_timestamp)


def cont_emg(npoints, elecs, start_timestamp=0):
    """
    Retrieve emg data

    Parameters and outputs are the same as the `cont_raw` function.

    If analog I/O is requested, the unfiltered analog data is retrieved, not
    filtered LFP data.
    Args:
        npoints: number of datapoints to retrieve
        elecs:  list of electrodes to sample
        start_timestamp: NIP timestamp to start data at, or most recent if 0

    Returns:
    """
    return _cont_base(_c.xl_cont_emg, npoints, elecs, start_timestamp)


def cont_status(npoints, elecs, start_timestamp=0):
    """
    Retrieve Mira status data (2 kHz)

    Every clock-cycle the Mira front end publishes status information concerning
    the implant connectivity and sensor information.

    | channel | name             | units |
    | ------- | ---------------- | ----- |
    | 0       | counter          |       |
    | 1       | i_status_imp     |   V   |
    | 2       | i_status_pwr     |   mA  |
    | 3       | adc_vin_v        |   V   |
    | 4       | adc_vin_a        |  mA   |
    | 5       | adc_temp_onboard |  C    |
    | 6       | adc_temp_offboard|  C    |
    | 7       | pwr_servo_state  |       |
    | 8       | impl_serial      |       |
    | 9       | impl_deviceid    |       |
    | 10      | impl_temp        |  C    |
    | 11      | impl_humidity    |  %    |
    | 12      | impl_voltage     |  V    |
    | 13      | impl_ver_hw      |       |
    | 14      | impl_ver_fw      |       |

    Args:
        npoints: number of datapoints to retrieve
        channel:  A list of channels to retrieve. See the table above. All
        channels are returned as float arrays even if they are fundamentally
        integer quantities.
        start_timestamp: NIP timestamp to start data at, or most recent if 0

    Returns:
    """
    return _cont_base(_c.xl_cont_status, npoints, elecs, start_timestamp)


# TODO: This probably doesn't work SpikeDataBuffer::GetEventPackets() might
#      be broken (xippmex uses another function)
def spk_data(elec, max_spk=1024):
    """

    Return spike data
    Args:
        elec: desired electrode, zero indexed
        max_spk: max spikes, default, 1024

    Returns:
        tuple counts, spks - count, the number of spikes
        spks - list of SegmentDataPacket classes
    """
    c_spikes = _c.SegmentDataArray(max_spk)

    event_ids = array.array('i', itertools.repeat(0, max_spk))
    (event_ptr, _) = event_ids.buffer_info()

    n = exception.check(_c.xl_spk_data(c_spikes, event_ptr, max_spk, elec))
    spikes = [SegmentDataPacket(c_spikes[i]) for i in range(min(n, max_spk))]

    # TODO: no reason to include the count (n) here.
    return n, spikes



def spk_data2(elecs, max_spk=1024):
    """

    Return spike data
    Args:
        elecs: desired electrodes, zero indexed
        max_spk: max spikes, default, 1024

    Returns:
        tuple counts, spks - count, the number of spikes
        spks - list of SegmentDataPacket classes
    """
#    func_start = _time.time()
    MAX_SEGMENTS = max_spk * len(elecs)

    c_spikes = _c.SegmentDataArray(MAX_SEGMENTS)
    c_elecs = array.array('I', elecs)
    (elecs_ptr, _) = c_elecs.buffer_info()

    c_counts = array.array('i', itertools.repeat(0, len(elecs)))
    (counts_ptr, _) = c_counts.buffer_info()

    n = exception.check(
            _c.xl_spk_data2(c_spikes, max_spk, counts_ptr, elecs_ptr,
                len(elecs)))

#    list_start = _time.time()
    spk_list = [SegmentDataPacket(c_spikes[i]) for i in range(MAX_SEGMENTS)]
#    list_stop = _time.time()

    spikes = []
#    pack_start = _time.time()
    for i in range(len(elecs)):
        _cnt = min(c_counts[i], max_spk)
        _spikes = spk_list[i*max_spk : (i+1)*max_spk]
        spikes.append([_spikes[j] for j in range(_cnt)])

#    func_stop = _time.time()
#    print('function: {:.3f} make list: {:.3f} pack list: {:.3f}'.format(
#        func_stop - func_start,
#        list_stop - list_start,
#        func_stop - pack_start))

    return spikes, list(c_counts)



def stim_data(elec, max_spk=1024):
    """
    retrieve segment data containing stim waveforms
    Args:
        elec:
        max_spk:

    Returns:
        tuple counts, events - count, the number of spikes
        events - list of SegmentDataPacket classes
    """
    c_spikes = _c.SegmentDataArray(max_spk)

    event_ids = array.array('i', itertools.repeat(0, max_spk))
    (event_ptr, _) = event_ids.buffer_info()

    n = exception.check(_c.xl_stim_data(c_spikes, event_ptr, max_spk, elec))
    spikes = [SegmentDataPacket(c_spikes[i]) for i in range(min(n, max_spk))]

    # TODO: no reason to include the count (n) here.
    return n, spikes


def digin(max_events=1024):
    """
    Retrieve digital inputs
    Args:
        max_events:

    Returns:
        tuple counts, events - count, the number of spikes
        events - list of SegmentEventPacket classes
    """
    c_events = _c.DigitalEventArray(max_events)

    event_ids = array.array('i', itertools.repeat(0, max_events))
    (event_ptr, _) = event_ids.buffer_info()

    # TODO: no check() here because the API is broken
    n = _c.xl_digin(c_events, event_ptr, max_events)
    events = [SegmentEventPacket(c_events[i]) for i in
              range(min(n, max_events))]

    # TODO: no reason to include the count (n) here.
    return n, events


def digout(outputs, values):
    """
    Produce digital outputs
    Args:
        outputs: list of integers designating desired output channels
        values: list of values, all are 0, 1 except the parallel port
            which is a 16-bit integer.

    Returns:
        None
    """
    if len(outputs) != len(values):
        raise ValueError("length of outputs and inputs must match")

    c_outputs = array.array('I', outputs)
    (outputs_ptr, _) = c_outputs.buffer_info()

    c_values = array.array('I', values)
    (values_ptr, _) = c_values.buffer_info()

    exception.check(_c.xl_digout(outputs_ptr, values_ptr, len(values)))



#
#  informational functions
#

def time():
    """Return the most recent NIP time."""
    return exception.check(_c.xl_time(), "Unable to get NIP time")


def list_elec(fe_type="", max_elecs=256):
    """
    List electrodes on a specified frontend type, returning a sequence.

    Arguments:
        fe_type: type of frontend
        max_elecs: maximum number of electrodes to return. If there are more
                   than this many, the extras will be omitted.
    """
    data_out = array.array('I', itertools.repeat(0, max_elecs))
    (data_ptr, _) = data_out.buffer_info()

    n = exception.check(_c.xl_list_elec(data_ptr, max_elecs, fe_type),
                        "unable to list electrodes")

    # Truncate array to actual size of data
    del data_out[n:]
    return data_out


def get_fe(elec):
    """Return the frontend index of the requested electrode."""
    return exception.check(_c.xl_get_fe(elec),
                           "no front end found for electrode {}".format(elec))


def get_fe_streams(elec, max_streams=32):
    """
    Return a list of stream types supported by the given electrode.
    Args:
        elec: zero indexed electrode
        max_streams: max strings to return
    """
    data = array.array('b', itertools.repeat(0, max_streams * _c.STRLEN_LABEL))
    (ptr , _) = data.buffer_info()

    n = exception.check(_c.xl_get_fe_streams(ptr, max_streams, elec),
                        "Error getting streams for fe {}".format(elec))

    ret = []
    for i in range(n):
        addr = ptr + i * _c.STRLEN_LABEL
        s = ctypes.string_at(addr)
        ret.append(s.decode('utf-8'))

    return ret


def get_nip_serial(max_size=1024):
    """
    Return string of nip serial number, eg 'R00244-0006'
    Args:
        max_size: maximum size of string
    """
    data = array.array('b', itertools.repeat(0, max_size))
    (ptr, _) = data.buffer_info()

    n = exception.check(
        _c.xl_nip_serial(ptr, max_size), "unable to get serial number")
    s = ctypes.string_at(ptr)
    return s.decode('utf-8')


def get_nipexec_version(max_size=1024):
    """
    Return string of nipexec version, eg '1.6.1.23'
    Args:
        max_size: maximum size of string
    """
    data = array.array('b', itertools.repeat(0, max_size))
    (ptr, _) = data.buffer_info()

    exception.check(
        _c.xl_nipexec_version(ptr, max_size),
        "unable to get nipexec version information"
    )
    s = ctypes.string_at(ptr)
    return s.decode('utf-8')


def get_fe_version(elec, max_size=1024):
    """
    Return R number for fe for given electrode
    Args:
        elec: zero indexed electrode
        max_size: maximum size of string
    """
    data = array.array('b', itertools.repeat(0, max_size))
    (ptr, _) = data.buffer_info()

    exception.check(_c.xl_fe_version(ptr, max_size, elec),
                    "unable to get front end version information for {}".format(
                        elec))
    s = ctypes.string_at(ptr)
    return s.decode('utf-8')


#
#  Signal functions
#

def signal(elec, stream_ty):
    """
    Return a bool indicating whether the stream of the given type on the given
    electrode is selected.
    Args:
        elec:
        stream_ty:
    """
    return bool(exception.check(_c.xl_signal(elec, stream_ty)))


def signal_raw(elec):
    """
    Return the selection status of a raw stream (like signal()).
    Args:
        elec:
    """
    return bool(exception.check(_c.xl_signal_raw(elec)))


def signal_lfp(elec):
    """
    Return the selection status of a LFP stream (like signal()).
    Args:
        elec:
    """
    return bool(exception.check(_c.xl_signal_lfp(elec)))


def signal_spk(elec):
    """
    Return the selection status of a spike stream (like signal()).
    Args:
        elec:
    """
    return bool(exception.check(_c.xl_signal_spk(elec)))


def signal_stim(elec):
    """
    Return the selection status of a stim stream (like signal()).
    Args:
        elec:
    """
    return bool(exception.check(_c.xl_signal_stim(elec)))


def signal_set(elec, stream_ty, val):
    """
    Select or deselect a signal type on a single electrode.

    Arguments:
        elec: electrode ID to operate on
        stream_ty: type of stream to set selection for (as from get_fe_streams())
        val: True to select, or False to deselect
    """
    return exception.check(_c.xl_signal_set(elec, stream_ty, int(val)))


def signal_set_raw(elec, val):
    """
    Set selection for a raw signal (like signal_set()).
    Args:
        elec:
        val:
    """
    return exception.check(_c.xl_signal_set_raw(elec, int(val)))


def signal_set_lfp(elec, val):
    """
    Set selection for a LFP signal (like signal_set()).
    Args:
        elec:
        val:
    """
    return exception.check(_c.xl_signal_set_lfp(elec, int(val)))


def signal_set_spk(elec, val):
    """
    Set selection for a spike signal (like signal_set()).
    Args:
        elec:
        val:
    """
    return exception.check(_c.xl_signal_set_spk(elec, int(val)))


def signal_set_stim(elec, val):
    """
    Set selection for a stim signal (like signal_set()).
    Args:
        elec:
        val:
    """
    return exception.check(_c.xl_signal_set_stim(elec, int(val)))


#
# trial
#

def trial(oper=150, status=None, file_name_base=None, auto_stop_time=None,
          auto_incr=None, incr_num=None):
    """
    Create a trial packet addressed to the operator with id `oper`. All other
    arguments but oper are optional, if they aren't specified they will remain
    unchanged for the operator. 'Enable remote control' must be enabled for
    input parameters to have an effect. The resulting or current state
    corresponding to each input parameter is returned for each call. To query
    parameters only, just call with no arguments.

    Args:
        oper: operator id
        status: a string which can be: recording, stopped, paused
        file_name_base: the base filename that the trial will record to
        auto_stop_time: the time in seconds after which recording will stop
        auto_incr: a boolean that indicates whether auto_increment is enabled
        incr_num: the value to set the current file increment counter to

    Returns: the resulting state of: status, file_name_base, auto_stop_time,
      auto_incr, incr_num

    """
    desc_in = TrialDescriptor()

    desc_in.oper = oper

    if status is not None:
        desc_in.status = status

    if file_name_base is not None:
        desc_in.file_name_base = file_name_base

    if auto_stop_time is not None:
        desc_in.auto_stop_time = int(auto_stop_time)

    if auto_incr is not None:
        desc_in.auto_incr = int(auto_incr)

    if incr_num is not None:
        desc_in.incr_num = int(incr_num)

    desc_out = TrialDescriptor()
    desc_out.oper = oper
    desc_out.status = 255 * "\0"
    desc_out.file_name_base = 4096 * "\0"
    desc_out.auto_stop_time = 0
    desc_out.auto_incr = 0
    desc_out.incr_num = 0

    err_code = _c.xl_trial2(desc_out.cast(), desc_in.cast())

    if err_code == -1:
        raise RuntimeError('xipplib not initialized')
    if err_code == 1:
        raise RuntimeError('No operator found')
    elif err_code == 2:
        raise RuntimeError('No response from operator')
    elif err_code == 3:
        raise RuntimeError('`status` must be: recording, stopped or paused')
    elif err_code == 4:
        raise RuntimeError('Unknown receiver code found in trial descriptor')
    elif err_code > 4:
        raise RuntimeError('xipplib returned an unknown error code')

    return (
        desc_out.status,
        desc_out.file_name_base,
        desc_out.auto_stop_time,
        bool(desc_out.auto_incr),
        desc_out.incr_num
    )


#
# impedance
#
def impedance(channels):
    """
    Take a list of channels and return measured impedances on those channels.
    """
    size = len(channels)
    chan = array.array('I', channels)
    result = array.array('f', size * [0])

    exception.check(
        _c.xl_imp(result.buffer_info()[0], chan.buffer_info()[0], size),
        'Impedance call failed.'
    )
    return result
