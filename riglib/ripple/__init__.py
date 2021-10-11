'''
Extensions of the generic riglib.source.DataSourceSystem for getting Spikes/LFP data from the Blackrock NeuroPort system over the rig's internal network (UDP)
'''
import numpy as np
import xipppy as xp
import itertools
import time
import os
import array
from collections import namedtuple, Counter
from riglib.source import DataSourceSystem

SpikeEventData = namedtuple("SpikeEventData",
                            ["chan", "unit", "ts", "arrival_ts"])
ContinuousData = namedtuple("ContinuousData", 
                            ["chan", "samples", "arrival_ts"])

class Spikes(DataSourceSystem):
    '''
    For use with a DataSource in order to acquire streaming spike data from 
    the Ripple Neural Information Processor (NIP).
    '''

    update_freq = 30000.
    dtype = np.dtype([("ts", np.float), 
                      ("chan", np.int32), 
                      ("unit", np.int32),
                      ("arrival_ts", np.float64)])

    def __init__(self, channels):
        #self.conn = xp
#        self.conn = cerelink.Connection()        
        xp._open('tcp')
        self.channels = channels
        self.chan = itertools.cycle(self.channels)
        recChans = np.array(xp.list_elec('nano'))
        self.recChans = recChans
        # make sure all channels are available with spk and lfp data
        if len(self.recChans):
            for ii in self.recChans:
                xp.signal_set(ii.item(),'spk',True)
                xp.signal_set(ii.item(),'lfp',True)
                time.sleep(0.001)

    def start(self):
        self.data = self.get_event_data()

    def stop(self):
        xp._close()

    def get(self):
        '''
        self.data = self.get_event_data()
        d = next(self.data)
        return np.array([(d.ts / self.update_freq, 
                          d.chan, 
                          d.unit, 
                          d.arrival_ts)],
                        dtype=self.dtype)
        '''
        arrival_ts = time.time()
        ch = next(self.chan)
        n, seg_data = xp.spk_data(int(ch - 1),max_spk=10)
        if len(seg_data):
            for p in seg_data:
                un = 2 ** (p.class_id + 0) # class_id's are 0,1,2,3,4, but self.unit is 2,4,8,16
                print("Chan, Unit:", ch, un)
                ts = p.timestamp
                data = np.array([(ts / self.update_freq, 
                          ch, 
                          un, 
                          arrival_ts)],
                        dtype=self.dtype)
        else:
            data = None
        return data
    
    def get_event_data(self):
        '''
        sleep_time = 0
        
            
        arrival_ts = time.time()
        # Can use xp.spike_data(elec)
        #for chan in self.channels:
        for chan in self.channels:
            print('Channel data type:', type(chan))
            n, seg_data = xp.spk_data(int(chan))
            if len(seg_data):
                for p in seg_data:
                    un = p.class_id # DEREK: do we need to do log2 here? 
                    ts = p.timestamp
                    print("Timestamp", ts)
                    yield SpikeEventData(ts=ts, chan=chan, unit=un, arrival_ts=arrival_ts)

        time.sleep(sleep_time)
        '''
        return        

"""
lfp NOT UPDATED FOR RIPPLE YET
"""

class LFP(DataSourceSystem):
    '''
    For use with a MultiChanDataSource in order to acquire streaming LFP 
    data from the Blackrock Neural Signal Processor (NSP).
    '''
    
    update_freq = 1000.
    # dtype = np.dtype([("samples", np.float)])
    dtype = np.dtype([("chan", np.int32),
                      ("samples", np.float)])

    def __init__(self, channels):     
        xp._open('tcp')
        self.channels = channels
        self.chan = itertools.cycle(self.channels)
        recChans = np.array(xp.list_elec('nano'))
        self.recChans = recChans
        # make sure all channels are available with spk and lfp data
        if len(self.recChans):
            for ii in self.recChans:
                xp.signal_set(ii.item(),'spk',True)
                xp.signal_set(ii.item(),'lfp',True)
                time.sleep(0.001)

    def start(self):
        self.data = self.get_continuous_data()

    def stop(self):
        xp._close()

    def get(self):

        n_lfp = 1000
        # chs = [int(ch-1) for ch in self.channels]
        # lfpData,lfpTimestamp=xp.cont_lfp(n_lfp,chs)
        # data = np.zeros((len(self.channels), n_lfp))
        # for i in range(len(self.channels)):
        #     data[i, :] = lfpData[i*n_lfp:(i+1)*n_lfp]

        ch = next(self.chan)
        lfpData, lfpTimestamp = xp.cont_lfp(1000, [int(ch - 1)])

        data = np.array(lfpData, dtype='float')

        # data = np.array([ch, np.array(lfpData)], dtype=self.dtype)
        # data = np.array(lfpData, dtype=self.dtype)
        return (ch, data)

    def get_continuous_data(self):
        '''
        sleep_time = 0
        n_samples = 1000
        
        while self.streaming:
            
            arrival_ts = time.time()
            data_out, ts_out = xp.cont_lfp(n_samples, self.channels)
            for i, chan in enumerate(self.channels):
                seg_data = np.array(data_out[i*n_samples:(i+1)*n_samples])
            
                yield ContinuousData(chan=chan, samples=seg_data, arrival_ts=arrival_ts)

        time.sleep(sleep_time)  
        '''
        return

