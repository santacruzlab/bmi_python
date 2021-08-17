'''
RDA (Remote Data Access) client code to receive data from the Brain Products
BrainVision Recorder.
'''

import time
import numpy as np
from struct import *
from socket import *
import array
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
#import math
# from riglib.filter import Filter
from ismore.filter import Filter

from ismore import settings, brainamp_channel_lists
from features.generator_features import Autostart
from multiprocessing import sharedctypes as shm
import ctypes

# Helper function for receiving whole message
def RecvData(socket, requestedSize):
    returnStream = ''
    while len(returnStream) < requestedSize:
        databytes = socket.recv(requestedSize - len(returnStream))
        if databytes == '':
            raise RuntimeError, "connection broken"
        returnStream += databytes
 
    return returnStream   

    
# Helper function for splitting a raw array of
# zero terminated strings (C) into an array of python strings
def SplitString(raw):
    stringlist = []
    s = ""
    for i in range(len(raw)):
        if raw[i] != '\x00':
            s = s + raw[i]
        else:
            stringlist.append(s)
            s = ""

    return stringlist
    

# Helper function for extracting eeg properties from a raw data array
# read from tcpip socket
def GetProperties(rawdata):

    # Extract numerical data
    (channelCount, samplingInterval) = unpack('<Ld', rawdata[:12])

    # Extract resolutions
    resolutions = []
    for c in range(channelCount):
        index = 12 + c * 8
        restuple = unpack('<d', rawdata[index:index+8])
        resolutions.append(restuple[0])

    # Extract channel names
    channelNames = SplitString(rawdata[12 + 8 * channelCount:])

    return (channelCount, samplingInterval, resolutions, channelNames)



class EMGData(object):
#class EMGData(Autostart):
    '''For use with a MultiChanDataSource in order to acquire streaming EMG/EEG/EOG
    data (not limited to just EEG) from the BrainProducts BrainVision Recorder.
    '''

    update_freq = 1000. #nerea
    #update_freq = 2500. # TODO -- check

    dtype = np.dtype([('data',       np.float64),
                      ('ts_arrival', np.float64)])

    RDA_MessageStart     = 1      # 
    RDA_MessageData      = 2      # message type for 16-bit data
    RDA_MessageStop      = 3      # 
    RDA_MessageData32    = 4      # message type for 32-bit data
    RDA_MessageKeepAlive = 10000  # packets of this message type can discarded


    # TODO -- added **kwargs argument to __init__ for now because MCDS is passing
    #   in source_kwargs which contains 'channels' kwarg which is not needed/expected
    #   need to fix this later
    def __init__(self, recorder_ip='192.168.137.3', nbits=16, **kwargs):
        self.recorder_ip = recorder_ip
        self.nbits = nbits
        print 'self.nbits ', self.nbits
        if self.nbits == 16:
            self.port = 51234
            self.fmt = '<h'  # little-endian byte order, signed 16-bit integer
            self.step = 2    # bytes
        elif self.nbits == 32:
            self.port = 51244
            self.fmt = '<f'  # little-endian byte order, 32-bit IEEE float
            self.step = 4    # bytes
        else:
            raise Exception('Invalid value for nbits -- must be either 16 or 32!')

        # # Create a tcpip socket to receive data and another to send commands to start and stop recording remotely
        #self.sock = socket(AF_INET, SOCK_STREAM) #original


        #self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        self.sock = socket(AF_INET, SOCK_STREAM,0) 
        self.sock.setsockopt(IPPROTO_TCP,TCP_NODELAY, 1)


        #uncomment this for remot start/stop of recorder recordings
        # self.port2 = 6700 #port to write to to start and stop recording of Recorder remotely
        # self.sock2 = socket(AF_INET, SOCK_STREAM)
        # self.sock2.connect((self.recorder_ip, self.port2))
        # #self.sock2.send("1C:\Vision\Workfiles\EMG_14Bip_channels.rwksp")
        # self.sock2.send("1C:\Vision\Workfiles\EMG_48HD_6Bip_channels.rwksp")
        # #self.sock2.send("1C:\Vision\Workfiles\EEG32channels.rwksp")
        # time.sleep(1)
        # self.sock2.send("2HD_EMG_TF")
        # time.sleep(1)
        # self.sock2.send("32")
        # time.sleep(1)
        # self.sock2.send("4")
        # time.sleep(1)
        # self.sock2.send("M")
        # time.sleep(8)


        # self.fo = open('/storage/rawdata/test_rda_data.txt','w')
        # #self.fo2 = open('/storage/rawdata/test_rda_header.txt','w')
        # self.fo3 = open('/storage/rawdata/test_rda_block.txt','w')
        # self.fo4 = open('/storage/rawdata/test_rda_before_block.txt','w')
        # self.fo5 = open('/storage/rawdata/test_rda_msgtype.txt','w')

        if 'fs' in kwargs:
            self.fs = kwargs['fs']
        else:
            # self.fs = 1000
            self.fs = 1000
        # calculate coefficients for a 4th-order Butterworth BPF from 10-450 Hz
        band  = [10, 450]  # Hz
        nyq   = 0.5 * self.fs
        low   = band[0] / nyq
        high  = band[1] / nyq
        self.bpf_coeffs = butter(4, [low, high], btype='band')
        # self.band_pass_filter = Filter(self.bpf_coeffs[0], self.bpf_coeffs[1])
        band  = [1, 48]  # Hz
        nyq   = 0.5 * self.fs
        low   = band[0] / nyq
        high  = band[1] / nyq
        self.bpf_coeffs_eeg = butter(2, [low, high], btype='band')
        # calculate coefficients for multiple 2nd-order notch filers
        self.notchf_coeffs = []
        for freq in [50, 150, 250, 350]:
            band  = [freq - 2, freq + 2]  # Hz
            nyq   = 0.5 * self.fs
            low   = band[0] / nyq
            high  = band[1] / nyq
            self.notchf_coeffs.append(butter(2, [low, high], btype='bandstop'))


        if not hasattr(self, 'brainamp_channels'):
            self.brainamp_channels = kwargs['brainamp_channels'] #channels(raw) that will be received from the BVrecorder

        if not hasattr(self, 'channels'):
            #self.channels = []
            try:
                self.channels = kwargs['channels_filt'] #channels that will be filtered and used online
            except:
                self.channels = []
             
        self.channels_all = self.brainamp_channels + self.channels #raw and filtered channels
        print ' channels all in rda.py: ', len(self.channels_all)

        if 'channels_str_2discard' in kwargs:
            self.channels_str_2discard = kwargs['channels_str_2discard'] 
        # else:
        #     channels_str_2discard = [5,11,17,23,29,35,41,47,49,51,53,55,57]
        if 'channels_str_2keep' in kwargs:
            self.channels_str_2keep = kwargs['channels_str_2keep']
        # else:
        #     channels_str_2keep = [i for i in range(59) if i not in channels_str_2discard]
        if 'channels_diag1_1' in kwargs:
            self.channels_diag1_1 = kwargs['channels_diag1_1']
        # else:
        #     channels_diag1_1 = [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46]
        if 'channels_diag1_2' in kwargs:
            self.channels_diag1_2 = kwargs['channels_diag1_2']
        # else:
        #     channels_diag1_2 = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41]
        if 'channels_diag2_1' in kwargs:
            self.channels_diag2_1 = kwargs['channels_diag2_1'] 
        # else:
        #     channels_diag2_1 = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40]
        if 'channels_diag2_2' in kwargs:
            self.channels_diag2_2 = kwargs['channels_diag2_2']
        # else:
        #     channels_diag2_2 = [7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47]

    def start(self):
        '''Start the buffering of data.'''
        print 'rda.py, start method'

        self.streaming = True
        #uncomment this for remote start/stop of recorder recordings
        #self.sock2.send("S")
        self.data = self.get_data()

    def stop(self):
        '''Stop the buffering of data.'''
        print "stop streaming"
        self.streaming = False
        # self.fo.close()
        # #self.fo2.close()
        # self.fo3.close()
        # self.fo4.close()
        # self.fo5.close()

    def disconnect(self):
        # self.sock2.send("Q")
        # print "disconnect socket"
        '''Close the connection to Recorder.'''
    
    def __del__(self):
        self.disconnect()


    # TODO -- add comment about how this will get called by the source
    def get(self):
        return self.data.next()

    # def cleanup(self, database, saveid, **kwargs):
    #     print "stop recording"
    #     self.sock2.send("Q")
    #     time.sleep(5)
    #     super(EMGData,self).cleanup(database, saveid, **kwargs)

    def get_data(self):
        '''A generator which yields packets as they are received'''
        
        self.sock.connect((self.recorder_ip, self.port))
        # self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        chan_idx = 0

        self.notch_filters = []
        for b, a in self.notchf_coeffs:
            self.notch_filters.append(Filter(b=b, a=a))
        channelCount_filt = len(self.channels)
        print ' channelCount_filt: ', channelCount_filt


        self.channel_filterbank_emg = [None]*channelCount_filt
        for k in range(channelCount_filt):
            filts = [Filter(self.bpf_coeffs[0], self.bpf_coeffs[1])]
            for b, a in self.notchf_coeffs:
                filts.append(Filter(b=b, a=a))
            self.channel_filterbank_emg[k] = filts

        self.channel_filterbank_eeg = [None]*channelCount_filt
        for k in range(channelCount_filt):
            filts = [Filter(self.bpf_coeffs_eeg[0], self.bpf_coeffs_eeg[1])]
            #filts.append(Filter(b=self.notchf_coeffs[0][0], a=self.notchf_coeffs[0][1]))
            self.channel_filterbank_eeg[k] = filts
        
        while self.streaming:


            # Get message header as raw array of chars
            rawhdr = RecvData(self.sock, 24)

            #self.fo2.write('received' + ' ' + str(time.time()) + ' \n')
            # Split array into useful information; id1 to id4 are constants

            (id1, id2, id3, id4, msgsize, msgtype) = unpack('<llllLL', rawhdr)
            #self.fo5.write(str(msgtype) + ' ' +  str(time.time()) + ' \n')

            # Get data part of message, which is of variable size
            rawdata = RecvData(self.sock, msgsize - 24) 
            
            ts_arrival = time.time()
            

            # Perform action dependend on the message type
            if msgtype == self.RDA_MessageStart:
                # Start message, extract eeg properties and display them
                (channelCount, samplingInterval, resolutions, channelNames) = GetProperties(rawdata)

                if self.fs != int(1/samplingInterval*1000000):
                    print "Error: Selected sampling frequency in rda.py file and BrainAmp Recorder do not match"
                
                #channelNames_filt = [c+'_filt' for c in channelNames]
                #if channelNames != self.channels and channelNames!= self.channels_all:
                #    if channelNames_filt != self.channels and channelNames_filt !=self.channels_all:
                #        print 'channelNames: ', channelNames
                #        print 'self.channels: ', self.channels
                #        print 'self.channels all: ', self.channels_all
                #        print 'ERROR: Selected channels do not match the streamed channel names. Double check!'
                #        raise Exception

                # else:
                #     #channelNames_filt = list()
                #     for i in range(channelCount):
                #         channelNames.append(channelNames[i] + "_filt")
                #     channelCount_all = channelCount*2 
                #     resolutions_filt = [resolutions[0]]*(len(channelNames))
                #     resolutions = resolutions + resolutions_filt

                channels = self.channels

                channelCount_all = len(self.channels_all)
                #resolutions = [resolutions[0]]*channelCount_all
                #channelCount_filt = channelCount_all - channelCount
                # channelCount_all = channelCount*2 
                # resolutions_filt = [resolutions[0]]*(len(channelNames))
                # resolutions = resolutions + resolutions_filt   
                
                # reset block counter
                lastBlock = -1
                
                
                # channels_filt_all = list()
                # for i in range(len(channelNames)):
                #     channels_filt = [channelNames[i] + "_filt" ]
                #     channels_filt_all.append(channels_filt)
                    
                
                print "Start"
                print "Overal number of channels: " + str(channelCount_all)
                print "Sampling interval: " + str(samplingInterval)
                print "Resolutions: " + str(resolutions[0])
                print "Online Channel Names: " + str(self.channels_all[0]) + "..." + str(self.channels_all[-1])
                print "Sampling Frequency: " + str(1000000/samplingInterval)

                
                # print "initializing data_all variable"
                # data_all = [] #nerea

                #channels = [int(name) for name in channelNames]
                #channels = channelNames

                

                #print type(channels)
                
                #channels = [int(name) for name in channels_filt_all]#andrea

                #print type(channels)


            elif msgtype == self.RDA_MessageStop:
                #self.send("Q")
                #self.disconnect()
                # TODO -- what to do here? set self.streaming = False?
                # call self.stop_data()? self.disconnect()?

                #save appended data in a mat file --> nerea
                # print "saving data"
                # import scipy.io
                # scipy.io.savemat('/home/tecnalia/test_with_python_mat_30', mdict={'data': data_all})
                
                pass

            elif (msgtype == self.RDA_MessageData) or (msgtype == self.RDA_MessageData32):
                #self.fo4.write(str(msgtype) + ' ' + str(time.time()) + ' \n')
                # Extract numerical data

                (block, points, markerCount) = unpack('<LLL', rawdata[:12])

                #self.fo3.write(str(block) + ' ' + str(points) + ' ' + str(time.time()) + ' \n')
                # Extract eeg/emg data
                
                # OLD, INEFFICIENT METHOD (yielding data points one at a time)
                # for i in range(points * channelCount):
                #     index = 12 + (self.step * i)
                #     AD_value = unpack(self.fmt, rawdata[index:index+self.step])[0]
                #     chan = channels[chan_idx]
                #     uV_value = AD_value * resolutions[chan_idx]
                #     yield (chan, np.array([(uV_value, ts_arrival)], dtype=self.dtype))
                #     chan_idx = (chan_idx + 1) % channelCount



                # MORE EFFICIENT -- yield all data points for a channel at once
                # data_ = array.array('h')
                # data_.fromstring(rawdata[12:])  # TODO -- make more general
                # data = np.zeros((channelCount, points), dtype=self.dtype)
                # data['data'] = np.array(data_).reshape((points, channelCount)).T
                # data['ts_arrival'] = ts_arrival
                # for chan_idx in range(channelCount):
                #     data[chan_idx, :]['data'] *= resolutions[chan_idx]
                #     chan = channels[chan_idx]
                #     yield (chan, data[chan_idx, :])


                # Filter the data as the packages arrive - andrea
                data_ = array.array('h') # 16-bits 
                #nerea testing
                # data_ = array.array('f') # 32-bits 


                data_.fromstring(rawdata[12:])  # TODO -- make more general


                #self.fo.write(str(data_[0]) + ' ' + str(time.time()) + ' \n')
                #print len(data_)
                # if np.array(data_).shape < 1200:
                #     print 'lost data'
                # print time.time()
                # print len(data_)
                
                data = np.zeros((channelCount, points), dtype=self.dtype)
                data['ts_arrival'] = ts_arrival
    
                data['data'] = np.array(data_).reshape((points, channelCount)).T
                
                data['data'] = np.array(resolutions).reshape(-1,1) * data['data']

                # data_all.append(data['data']) #nerea
                
                if channelCount_filt != 0:
                    # import time
                    # t0 = time.time()
                    datafilt = np.zeros((channelCount_all, points), dtype=self.dtype)
                    datafilt['ts_arrival'] = ts_arrival
                    filtered_data = np.zeros((channelCount_filt, points), dtype=self.dtype)
                    #filtered_data['data'] = data['data'].copy()
                

                    for k, chan_2filt in enumerate(channels):

                        if chan_2filt[:-5] in brainamp_channel_lists.emg14 + brainamp_channel_lists.emg14_bip + brainamp_channel_lists.emg8_screening : #Apply filters for bipolar emg
                            
                            
                            try:
                                if not hasattr(self, 'channels_diag1_1'):
                                #if we are not using the high-density configuration
                                # if self.brainamp_channels != brainamp_channel_lists.emg_48hd_6mono:
                                    
                                    k_raw = [num for num, chan_raw in enumerate(self.brainamp_channels) if chan_raw == chan_2filt[:-5]]
                                #print 'entering emg14 loop'
                                    filtered_data['data'][k] = data['data'][k_raw].copy()
                                    
                                    for filt in self.channel_filterbank_emg[k]:
                                        filtered_data[k]['data'] =  filt(filtered_data[k]['data'] )
                                    
                            except:
                                pass
                        #elif chan_2filt in brainamp_channel_lists.emg_48hd_6mono_filt: # bipolarize emg channels and filter them
                         #   pass #preprocessing done below instead of channel by channel
                        elif chan_2filt[:-5] in brainamp_channel_lists.eeg64 + brainamp_channel_lists.eog4 + brainamp_channel_lists.eog2 + brainamp_channel_lists.hybrid_2018_tms: #apply filters for eeg.
                            k_raw = [num for num, chan_raw in enumerate(self.brainamp_channels) if chan_raw == chan_2filt[:-5]]
                            #filtered_data['data'][k] = data['data'][k_raw].reshape(-1,points).copy()
                            filtered_data['data'][k] = data['data'][k_raw].copy()
                            # if k == 2:
                            #     print 'before', filtered_data['data'][k]
                            for filt in self.channel_filterbank_eeg[k]:
                                filtered_data[k]['data'] =  filt(filtered_data[k]['data'] ) 
                            

                    # preprocessing for high-density emg setup                      
                    #be careful some channels might have the same name as emg14_raw
                    # bipolarization along the muscle fibers and also the two diagonals
                    try:
                        if self.channels_diag1_1: # this is a more general way of checking whether we are using the high-density configuration
                        #if self.brainamp_channels == brainamp_channel_lists.emg_48hd_6mono or self.brainamp_channels == brainamp_channel_lists.emg_48hd_6mono_eog4_eeg32:
                            if self.brainamp_channels == brainamp_channel_lists.emg_48hd_6mono:
                                data_copy = data['data'].copy()

                            else:
                                data_copy = data['data'][0:len(brainamp_channel_lists.emg_48hd_6mono)].copy()#copy only the hd-emg channels in case we are also recording EEG/EOG

                            # channels_2discard = [5,11,17,23,29,35,41,47,49,51,53,55,57]
                            # channels_2keep = [i for i in range(59) if i not in channels_2discard]

                            # channels_str_2discard = [5,11,17,23,29,35,41,47,49,51,53,55,57]
                            # channels_str_2keep = [i for i in range(59) if i not in channels_str_2discard]

                            # channels_diag1_1 = [6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46]
                            # channels_diag1_2 = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41]
                            # channels_diag2_1 = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40]
                            # channels_diag2_2 = [7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47]
                            nfiltchannels_HD_montage = len(self.channels_str_2keep) + len(self.channels_diag1_1) + len(self.channels_diag2_1)
                            data_diff = np.diff(data_copy, axis = 0)
                
                            for i in range(nfiltchannels_HD_montage):
                                if i < len(self.channels_str_2keep):
                                    filtered_data[i,:] = data_diff[self.channels_str_2keep[i],:]
                                elif i < len(self.channels_str_2keep) + len(self.channels_diag1_1):
                                
                                    filtered_data[i,:] = data[self.channels_diag1_1[i-len(self.channels_str_2keep)]]['data'] - data[self.channels_diag1_2[i-len(self.channels_str_2keep)]]['data'] 
                                else:
                                    filtered_data[i,:] = data[self.channels_diag2_1[i-len(self.channels_str_2keep)-len(self.channels_diag1_1)]]['data'] - data[self.channels_diag2_2[i-len(self.channels_str_2keep)-len(self.channels_diag1_1)]]['data'] 
             
                                for filt in self.channel_filterbank_emg[i]: #channel_filterbank_emg has filters for the amount of raw channels, which is larger than the number of filtered channels. That's why this is not a problem
                                    filtered_data[i]['data'] =  filt(filtered_data[i]['data'] ) 
                    except:
                        pass
                    # elif self.brainamp_channels == brainamp_channel_lists.emg_48hd_6mono_raw:# filtered channels are less than raw channels!!!
                    #     data_copy = data['data'].copy()
                    #     channels_str_2discard = [5,11,17,23,29,35,41,47,49,51,53,55,57]
                    #     channels_str_2keep = [i for i in range(59) if i not in channels_str_2discard]
                        
                    #     data_diff = np.diff(data_copy, axis = 0)

                    #     for i in range(len(channels_2keep)):
                    #         filtered_data[i,:] = data_diff[channels_2keep[i],:]
                               
                    #         for filt in self.channel_filterbank_emg[i]: #channel_filterbank_emg has filters for the amount of raw channels, which is larger than the number of filtered channels. That's why this is not a problem
                    #             filtered_data[i]['data'] =  filt(filtered_data[i]['data'] ) 
                    
                    

                    datafilt['data'] = np.vstack([data['data'], filtered_data['data']]) 
                    # print time.time() - t0
                    #print datafilt['data']['2_filt']
                    # print 'shape', datafilt['data'].shape
                    # print datafilt['data'][34]
                        # from scipy.io import savemat
                        # import os
                        # savemat(os.path.expandvars('$HOME/code/ismore/emg_rda_filt.mat'), dict(emg_filt = filtered_data, datafilt = datafilt['data'], filterbank = self.channel_filterbank))

                    for chan_idx in range(channelCount_all):
                        #datafilt[chan_idx, :]['data'] *= resolutions[chan_idx]
                        chan = self.channels_all[chan_idx]

                        yield (chan, datafilt[chan_idx, :])
                else:
            
                    for chan_idx in range(channelCount_all):
                        # data[chan_idx, :]['data'] *= resolutions[chan_idx]
                        chan = self.channels_all[chan_idx]
                        yield (chan, data[chan_idx, :])
                
                
                # disregard marker data for now

                # # Check for overflow
                # if lastBlock != -1 and block > lastBlock + 1:
                #     print "*** Overflow with " + str(block - lastBlock) + " datablocks ***" 
                # lastBlock = block

            elif msgtype == self.RDA_MessageKeepAlive:
                pass
            else:
                raise Exception('Unrecognized RDA message type: ' + str(msgtype))

class SimEMGData(EMGData):
    def __init__(self, *args, **kwargs):
        self.dtype = np.dtype([('data',       np.float64),
                               ('ts_arrival', np.float64)])
        self.points = 50

        # if 'brainamp_channels' in kwargs:
        #     self.channels = kwargs['brainamp_channels']
        # else:
        #     self.channels = self.brainamp_channels

        if not hasattr(self, 'brainamp_channels'):
            self.brainamp_channels = kwargs['brainamp_channels'] #channels(raw) that will be received from the BVrecorder

        # if not hasattr(self, 'channels'):
        #     self.channels = kwargs['channels'] #channels that will be filtered and used online
        if not hasattr(self, 'channels'):
            try:
                self.channels = kwargs['channels_filt'] #channels that will be filtered and used online
            except:
                self.channels = []
            #self.channels = [ch + "_filt" for ch in self.brainamp_channels]
            #print "channels in rda", self.channels
        
        self.channels_all = self.brainamp_channels + self.channels #raw and filtered channels
        #count 'non_filt channels:'
        import scipy
        #non_filt_ch = [ch for ch in self.brainamp_channels if scipy.logical_or(len(ch)< 4, ch[-4:] != 'filt')]
        self.channelCount = len(self.brainamp_channels)#len(non_filt_ch)
        self.channelCount_all = len(self.channels_all)#self.channelCount + len(self.channels)
        self.channelCount_filt = len(self.channels)#self.channelCount_all - self.channelCount
        self.encoder = self.get_rand_data

    # def get(self):
    #     return self.data.next()

    def get_data(self):
        '''A generator which yields packets as they are received'''
        print 'rda.py, SIM get_data method'

        chan_idx = 0
        
        while self.streaming:
            time.sleep(.050)
            ts_arrival = time.time()
            data_  = np.random.randn(self.points*self.channelCount)
            
            data = np.ones((self.channelCount, self.points), dtype=self.dtype)
            data['ts_arrival'] = ts_arrival
            data['data'] = np.array(data_).reshape((self.points, self.channelCount)).T

            datafilt = np.ones((self.channelCount_all, self.points), dtype=self.dtype)
            datafilt['ts_arrival'] = ts_arrival
            #filtered_data = np.random.randn(channelCount_filt, self.points)
            print("Yes we go here SimEMGData")
            filtered_data = self.encoder()

            datafilt['data'] = np.vstack([data['data'], filtered_data])
            
            for chan_idx in range(self.channelCount_all):
                datafilt[chan_idx, :]['data'] *= 1 #resolutions[chan_idx]
                chan = self.channels_all[chan_idx]
                yield (chan, datafilt[chan_idx, :])

    def get_rand_data(self):
        i = np.random.randn()
        return np.zeros((self.channelCount_filt, self.points)) + i       

class ReplayEMGData(EMGData):
    def __init__(self, *args, **kwargs):
        self.dtype = np.dtype([('data',       np.float64),
                               ('ts_arrival', np.float64)])
        self.points = 50

        if 'brainamp_channels' in kwargs:
            self.channels = kwargs['brainamp_channels']
        else:
            self.channels = self.brainamp_channels

        #count 'non_filt channels:'
        import scipy
        non_filt_ch = [ch for ch in self.channels if scipy.logical_or(len(ch)< 4, ch[-4:] != 'filt')]
        self.channelCount = len(non_filt_ch)
        self.channelCount_all = len(self.channels)
        self.channelCount_filt = self.channelCount_all - self.channelCount
        self.encoder = self.get_rand_data
        self.idx = 0
        hdf_file = '/storage/rawdata/hdf/ni20151214_73_te4931.hdf'

        import tables
        self.hdf = tables.open_file(hdf_file)

        if 'fs' in kwargs:
            self.fs = kwargs['fs']
        else:
            self.fs = 1000
        # calculate coefficients for a 4th-order Butterworth BPF from 10-450 Hz
        band  = [10, 450]  # Hz
        nyq   = 0.5 * self.fs
        low   = band[0] / nyq
        high  = band[1] / nyq
        self.bpf_coeffs = butter(4, [low, high], btype='band')

        self.notchf_coeffs = []
        for freq in [50, 150, 250, 350]:
            band  = [freq - 2, freq + 2]  # Hz
            nyq   = 0.5 * self.fs
            low   = band[0] / nyq
            high  = band[1] / nyq
            self.notchf_coeffs.append(butter(2, [low, high], btype='bandstop'))

        self.channel_filterbank_emg = [None]*self.channelCount
        for k in range(self.channelCount):
            filts = [Filter(self.bpf_coeffs[0], self.bpf_coeffs[1])]
            for b, a in self.notchf_coeffs:
                filts.append(Filter(b=b, a=a))
            self.channel_filterbank_emg[k] = filts
    # def get(self):
    #     return self.data.next()

    def get_data(self):
        '''A generator which yields packets as they are received'''
        print 'rda.py, SIM get_data method'

        chan_idx = 0
        
        while self.streaming:

            ts_arrival = time.time()
            data_ = np.vstack(( [self.hdf.root.brainamp[self.idx:self.idx+self.points]['chan' + c]['data'] for c in self.channels] ))
            #data_  = np.random.randn(self.points*self.channelCount)
            self.idx += self.points
            #print 'idx', self.idx
            datafilt = np.zeros((self.channelCount_all, self.points), dtype=self.dtype)
            datafilt['ts_arrival'] = ts_arrival
            datafilt['data'] = np.array(data_)
            # print data_.shape
            # #data['data'] = np.array(data_).reshape((self.points, self.channelCount)).T
            # data['data'] = np.array(data_).T
            # datafilt = np.zeros((self.channelCount_all, self.points), dtype=self.dtype)
            # datafilt['ts_arrival'] = ts_arrival
            # for k in range(self.channelCount):
            #         if  self.channels[k] in brainamp_channel_lists.emg14_raw: #Apply filters for bipolar emg
            #             filtered_data['data'] = data['data'].copy()
            #             for filt in self.channel_filterbank_emg[k]:
            #                 filtered_data[k]['data'] =  filt(filtered_data[k]['data'] )
            # #filtered_data = np.random.randn(channelCount_filt, self.points)
            # #filtered_data = self.encoder()

            # datafilt['data'] = np.vstack([data['data'], filtered_data])

            for chan_idx in range(self.channelCount_all):
                datafilt[chan_idx, :]['data'] *= 1 #resolutions[chan_idx]
                chan = self.channels[chan_idx]
                yield (chan, datafilt[chan_idx, :])

    def get_rand_data(self):
        return np.random.randn(self.channelCount_filt, self.points)        

class SimEMGData_with_encoder(SimEMGData):
    def __init__(self, *args, **kwargs):
        self.dtype = np.dtype([('data',       np.float64),
                               ('ts_arrival', np.float64)])
        self.points = 50

        if 'brainamp_channels' in kwargs:
            self.channels = kwargs['brainamp_channels']
        else:
            self.channels = self.brainamp_channels

        #count 'non_filt channels:'
        import scipy
        non_filt_ch = [ch for ch in self.channels if scipy.logical_or(len(ch)< 4, ch[-4:] != 'filt')]
        self.channelCount = len(non_filt_ch)
        self.channelCount_all = len(self.channels)
        self.channelCount_filt = self.channelCount_all - self.channelCount
        self.encoder = EEG_Encoder(self.points, self.channelCount)
        print self.encoder

class EEG_Encoder(object):
    def __init__(self, points, filt_channel_cnt):
        t = np.linspace(0, .050, points)
        s = np.sin(10*np.pi*2*t)
        self.move_data = np.random.rand(filt_channel_cnt, points)/10.
        self.rest_data = np.tile(np.array([s]), [filt_channel_cnt, 1]) + np.random.rand(filt_channel_cnt, points)/50.
        self.state = shm.RawValue(ctypes.c_char_p, 'wait')
    
    def __call__(self):
        print self.state.value
        if self.state in ['trial', 'trial_return']:
            # 80% probability of giving zeros
            i = np.rand.random()
            if i <= 0.8:
                return self.move_data
        return self.rest_data

class LiveAmpSource(object):
    '''For use with a MultiChanDataSource in order to acquire streaming EMG/EEG/EOG
    data (not limited to just EEG) from the BrainProducts BrainVision Recorder.
    '''

    update_freq = 1000.
    
    dtype = np.dtype([('data',       np.float64),
                      ('ts_arrival', np.float64)])

    RDA_MessageStart     = 1      # 
    RDA_MessageData      = 2      # message type for 16-bit data
    RDA_MessageStop      = 3      # 
    RDA_MessageData32    = 4      # message type for 32-bit data
    RDA_MessageKeepAlive = 10000  # packets of this message type can discarded

    # TODO -- added **kwargs argument to __init__ for now because MCDS is passing
    #   in source_kwargs which contains 'channels' kwarg which is not needed/expected
    #   need to fix this later
    def __init__(self, recorder_ip='192.168.137.3', nbits=32, **kwargs):
        self.recorder_ip = recorder_ip
        self.nbits = nbits
        print 'self.nbits ', self.nbits
        if self.nbits == 16:
            self.port = 51234
            self.fmt = '<h'  # little-endian byte order, signed 16-bit integer
            self.step = 2    # bytes
        elif self.nbits == 32:
            self.port = 51244
            self.fmt = '<f'  # little-endian byte order, 32-bit IEEE float
            self.step = 4    # bytes
        else:
            raise Exception('Invalid value for nbits -- must be either 16 or 32!')

        # Create a tcpip socket to receive data and another to send commands to 
        # start and stop recording remotely

        self.sock = socket(AF_INET, SOCK_STREAM,0) 
        self.sock.setsockopt(IPPROTO_TCP,TCP_NODELAY, 1)

        if 'fs' in kwargs:
            self.fs = kwargs['fs']
        else:
            # self.fs = 1000
            self.fs = 500

        if not hasattr(self, 'brainamp_channels'):
            self.brainamp_channels = kwargs['brainamp_channels'] #channels(raw) that will be received from the BVrecorder

        if not hasattr(self, 'channels'):
            #self.channels = []
            try:
                self.channels = kwargs['channels_filt'] #channels that will be filtered and used online
            except:
                self.channels = []
             
        self.channels_all = self.brainamp_channels + self.channels #raw and filtered channels


    def start(self):
        '''Start the buffering of data.'''
        print 'rda.py, start method'

        self.streaming = True
        self.data = self.get_data()

    def stop(self):
        '''Stop the buffering of data.'''
        print "stop streaming"
        self.streaming = False

    def disconnect(self):
        '''Close the connection to Recorder.'''
        pass
    
    def __del__(self):
        self.disconnect()

    # TODO -- add comment about how this will get called by the source
    def get(self):
        return self.data.next()

    def get_data(self):
        '''A generator which yields packets as they are received'''
        
        self.sock.connect((self.recorder_ip, self.port))

        chan_idx = 0
      
        while self.streaming:
            # Get message header as raw array of chars
            rawhdr = RecvData(self.sock, 24)

            # Split array into useful information; id1 to id4 are constants
            (id1, id2, id3, id4, msgsize, msgtype) = unpack('<llllLL', rawhdr)

            # Get data part of message, which is of variable size
            rawdata = RecvData(self.sock, msgsize - 24) 
            
            ts_arrival = time.time()

            # Perform action dependend on the message type
            if msgtype == self.RDA_MessageStart:
                # Start message, extract eeg properties and display them
                (channelCount, samplingInterval, resolutions, channelNames) = GetProperties(rawdata)
                print(samplingInterval,self.fs)
                if self.fs != int(1/samplingInterval*1000000):
                    print "Error: Selected sampling frequency in rda.py file and BrainAmp Recorder do not match"

                channels = self.channels
                channelCount_all = len(self.channels_all)
                
                # reset block counter
                lastBlock = -1
                
                print "Live Amp RDA Start"
                print "Overal number of channels: " + str(channelCount_all)
                print "Sampling interval: " + str(samplingInterval)
                print "Resolutions: " + str(resolutions[0])
                print "Online Channel Names: " + str(self.channels_all[0]) + "..." + str(self.channels_all[-1])
                print "Sampling Frequency: " + str(1000000/samplingInterval)

            elif msgtype == self.RDA_MessageStop:
                print "Live Amp RDA Stop"

            elif (msgtype == self.RDA_MessageData) or (msgtype == self.RDA_MessageData32):

                # Extract numerical data
                (block, points, markerCount) = unpack('<LLL', rawdata[:12])

                # Extract data coming from LiveAmp through RDA
                
                if msgtype == self.RDA_MessageData:
                    # Create buffer array of 16-bit signed short ints
                    data_ = array.array('h')
                    # Fill buffer array with values of raw data starting after header (12) 
                    # up to specified number of points by number of channels times bytes per value (2; 2*8=16)
                    data_.fromstring(rawdata[12:12 + (points * channelCount) * 2])
                elif msgtype == self.RDA_MessageData32:
                    # Create buffer array of 32-bit floats
                    data_ = array.array('f')
                    # Fill buffer array with values of raw data starting after header (12) 
                    # up to specified number of points by number of channels times bytes per value (4; 4*8=32)
                    data_.fromstring(rawdata[12:12 + (points * channelCount) * 4])

                # Reshape data to (N_chan, N_samples) and scale to resolutions 
                data_ = np.array(data_).reshape((points, channelCount)).T * np.array(resolutions).reshape(-1,1)

                # Bipolarization
                positive_channels_idx = [i for i,c in enumerate(channelNames) if "+" in c]
                negative_channels_idx = [i for i,c in enumerate(channelNames) if "-" in c]
                hardbip_channels_idx = [i for i,c in enumerate(channelNames) if "BIP" in c]
                eeg_channel_idx = [i for i,c in enumerate(channelNames) if "EEG" in c]
                acc_channel_idx = [i for i,c in enumerate(channelNames) if "ACC" in c]
                if len(positive_channels_idx) > 0 and len(positive_channels_idx) == len(negative_channels_idx):
                   softbip_data_ = np.vstack([data_[p] - data_[n] for p,n in zip(positive_channels_idx, negative_channels_idx)])
                   eeg_data_ = data_[eeg_channel_idx]
                   acc_data_ = data_[acc_channel_idx]
                   hardbip_data_ = data_[hardbip_channels_idx]
                   data_ = np.vstack([eeg_data_, softbip_data_, hardbip_data_, acc_data_, softbip_data_, hardbip_data_, acc_data_])

                print(softbip_data_.shape,hardbip_data_.shape, eeg_data_.shape, acc_data_.shape, data_.shape)
                print("Channels Requested: {}".format(self.channels_all))
                #print(positive_channels_idx, negative_channels_idx)

                # Convert buffer array to numpy array having dtype (64 bits including timestamp of arrival time)
                data = np.zeros((data_.shape[0], points), dtype=self.dtype)
                data['ts_arrival'] = ts_arrival
                data['data'] = data_
               
                for chan_idx in range(channelCount_all):
                    chan = self.channels_all[chan_idx]
                    yield (chan, data[chan_idx, :])