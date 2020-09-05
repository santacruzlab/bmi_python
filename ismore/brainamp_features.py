'''
Features for acquiring data from the BrainAmp EMG recording system (model name?)
'''

import time

from riglib.experiment import traits
from ismore import settings
import numpy as np
from socket import *


class BrainAmpData(traits.HasTraits):
    '''Stream BrainAmp EMG/EEG/EOG data.'''

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up a MultiChanDataSource to stream data from the BrainAmp system
        '''
        from riglib import source
        from ismore.brainamp import rda

        # self.recorder_ip='192.168.137.3'
        # self.port2 = 6700 #port to write to to start and stop recording of Recorder remotely
        # self.sock2 = socket(AF_INET, SOCK_STREAM)
        # self.sock2.connect((self.recorder_ip, self.port2))
        # #self.sock2.send("1C:\Vision\Workfiles\EEG32channels.rwksp")
        # self.sock2.send("1C:\Vision\Workfiles\EMG_48HD_6Bip_channels.rwksp")
        # # self.sock2.send("1C:\Vision\Workfiles\96_mono_channels.rwksp")
        # time.sleep(1)
        # self.sock2.send("2test_HD48_12mono_channels")
        # time.sleep(1)
        # self.sock2.send("31")
        # time.sleep(1)
        # self.sock2.send("4")
        # time.sleep(1)
        # self.sock2.send("M")
        # time.sleep(8)
        # self.sock2.send("S")

        #self.brainamp_channels = settings.BRAINAMP_CHANNELS

        #Send tempfile name to 
        import tempfile
        kwargs = dict(supp_file = tempfile.NamedTemporaryFile())

        self.supp_hdf_filename = kwargs['supp_file'].name


        if not hasattr(self, 'brainamp_channels'):
            import brainamp_channel_lists
            self.brainamp_channels = brainamp_channel_lists.eeg32_raw_filt #eog2_raw_filt #eeg32_raw_filt
            
        if not hasattr(self, 'channels'):
            self.channels = []
            # import brainamp_channel_lists
            # if self.brainamp_channels == brainamp_channel_lists.emg_48hd_6mono:
            #     self.channels = []
            # else:
            #     self.channels = [ch + "_filt" for ch in self.brainamp_channels]


        try:
           kwargs['channels_str_2discard'] = self.emg_extractor_kwargs['channels_str_2discard']
           kwargs['channels_str_2keep'] = self.emg_extractor_kwargs['channels_str_2keep']
           kwargs['channels_diag1_1'] = self.emg_extractor_kwargs['channels_diag1_1']
           kwargs['channels_diag1_2'] = self.emg_extractor_kwargs['channels_diag1_2']
           kwargs['channels_diag2_1'] = self.emg_extractor_kwargs['channels_diag2_1']
           kwargs['channels_diag2_2'] = self.emg_extractor_kwargs['channels_diag2_2']
        except:
            print 'no channel selection (reduction) was made'

        # if 'channels_str_2discard' in self.extractor_kwargs.keys():
        #     kwargs['channels_str_2discard'] = self.extractor_kwargs['channels_str_2discard']
        # if 'channels_str_2keep' in self.extractor_kwargs.keys():
        #     kwargs['channels_str_2keep'] = self.extractor_kwargs['channels_str_2keep']
        # if 'channels_diag1_1' in self.extractor_kwargs.keys():
        #     kwargs['channels_diag1_1'] = self.extractor_kwargs['channels_diag1_1']
        # if 'channels_diag1_2' in self.extractor_kwargs.keys():
        #     kwargs['channels_diag1_2'] = self.extractor_kwargs['channels_diag1_2']
        # if 'channels_diag2_1' in self.extractor_kwargs.keys():
        #     kwargs['channels_diag2_1'] = self.extractor_kwargs['channels_diag2_1']
        # if 'channels_diag2_2' in self.extractor_kwargs.keys():
        #     kwargs['channels_diag2_2'] = self.extractor_kwargs['channels_diag2_2']

        self.channels_all = self.brainamp_channels + self.channels
               

        self.brainamp_source = source.MultiChanDataSource(rda.EMGData, 
            name='brainamp', channels=self.channels_all, brainamp_channels=self.brainamp_channels, 
            send_data_to_sink_manager=True, channels_filt = self.channels, **kwargs)

        from riglib import sink
        sink.sinks.register(self.brainamp_source)

        super(BrainAmpData, self).init()

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the EMG data source
        '''

        
        self.brainamp_source.start()
        self.ts_start_brainamp = time.time()

        
        try:
            super(BrainAmpData, self).run()
        finally:

            self.brainamp_source.stop()            
            # print "stop recording from brainamp features stop function"       
            # self.sock2.send("Q")
            # time.sleep(5)

    def _start_wait(self):
        while 1:
            last_ts_arrival = self.last_brainamp_data_ts_arrival()
            if not last_ts_arrival == 0:
                break
            time.sleep(0.005)

        super(BrainAmpData, self)._start_wait()
            
    def _cycle(self):
        if settings.VERIFY_BRAINAMP_DATA_ARRIVAL:
            self.verify_brainamp_data_arrival(settings.VERIFY_BRAINAMP_DATA_ARRIVAL_TIME)

        super(BrainAmpData, self)._cycle()

    def verify_brainamp_data_arrival(self, n_secs):
        time_since_brainamp_started = time.time() - self.ts_start_brainamp
        last_ts_arrival = self.last_brainamp_data_ts_arrival()
     
        if time_since_brainamp_started > n_secs:
            if last_ts_arrival == 0:
                print 'No BrainAmp data has arrived at all'
            else:
                t_elapsed = time.time() - last_ts_arrival
                if t_elapsed > n_secs:
                    print 'No BrainAmp data has arrived in the last %.1f s' % t_elapsed

    def last_brainamp_data_ts_arrival(self):
        return np.max(self.brainamp_source.get(n_pts=1, channels=self.brainamp_source.channels)['ts_arrival'])

    def cleanup(self, database, saveid, **kwargs):

        # print "stop recording from brainamp features script"
        # from socket import *
       

        # print "stop recording from brainamp features script"
        # import time

        # self.sock2.send("Q")
        # time.sleep(5)
        print "magic to make hdf file open later"
        f = self.supp_hdf_filename
        print "hdf supp filename ", f
        fl = open(f)
        
        print "f opened ", fl

        import tables
        time.sleep(1.)
        fl.close()
        # h5 = tables.openFile(f, mode='a')
        h5 = tables.open_file(f, mode='a') #nerea
      
        print "h5 ", h5
        h5.close()

        print "Cleanup Supp HDF"
        print "\tSupp HDF currently at: %s" % self.supp_hdf_filename

        dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
        print database, 'database'
        if dbname == 'default':
            database.save_data(self.supp_hdf_filename, "supp_hdf", saveid)
        else:
            database.save_data(self.supp_hdf_filename, "supp_hdf", saveid, dbname=dbname)
        super(BrainAmpData,self).cleanup(database, saveid, **kwargs)


class SimBrainAmpData(traits.HasTraits):

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up a MultiChanDataSource to stream data from the BrainAmp system
        '''

        # Use 'SimEMGData' instead of 'EMGData' as source

        self.reg()

        super(SimBrainAmpData, self).init()

    def reg(self):
        from riglib import source
        from ismore.brainamp import rda
        import tempfile
        kwargs = dict(supp_file = tempfile.NamedTemporaryFile())

        #print "brainamp channels in brainamp features", self.brainamp_channels
        if not hasattr(self, 'brainamp_channels'):
            import brainamp_channel_lists
            self.brainamp_channels = brainamp_channel_lists.eeg32_raw_filt #eog2_raw_filt #eeg32_raw_filt
            
        if not hasattr(self, 'channels'):
            self.channels = []
            # import brainamp_channel_lists
            # if self.brainamp_channels == brainamp_channel_lists.emg_48hd_6mono:
            #     self.channels = []
            # else:
            #     self.channels = [ch + "_filt" for ch in self.brainamp_channels]
            
        if 'channels_2discard' in kwargs:
            kwargs['channels_2discard'] = channels_2discard
        if 'channels_2keep' in kwargs:
            kwargs['channels_2keep'] = channels_2keep
        if 'channels_diag1_1' in kwargs:
            kwargs['channels_diag1_1'] = channels_diag1_1
        if 'channels_diag1_2' in kwargs:
            kwargs['channels_diag1_2'] = channels_diag1_2
        if 'channels_diag2_1' in kwargs:
            kwargs['channels_diag2_1'] = channels_diag2_1
        if 'channels_diag2_2' in kwargs:
            kwargs['channels_diag2_2'] = channels_diag2_2

        #print "channels in features", self.channels[-1]

        self.channels_all = self.brainamp_channels + self.channels
        #print "channels_all in features", self.channels_all[-1]
        self.supp_hdf_filename = kwargs['supp_file'].name

        self.brainamp_source = source.MultiChanDataSource(rda.SimEMGData, 
            name='brainamp', channels=self.channels_all, brainamp_channels=self.brainamp_channels, 
            send_data_to_sink_manager=True, channels_filt = self.channels,**kwargs)
        # self.brainamp_source = source.MultiChanDataSource(rda.SimEMGData, 
        #     name='brainamp',
        #     send_data_to_sink_manager=True, **kwargs)
        
        from riglib import sink
        sink.sinks.register(self.brainamp_source)
        
   
    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the EMG data source
        '''

        self.brainamp_source.start()
        self.ts_start_brainamp = time.time()

        
        try:
            super(SimBrainAmpData, self).run()
        finally:
            self.brainamp_source.stop()            
            print "stop recording from brainamp features stop function"  

    def _start_wait(self):
        while 1:
            last_ts_arrival = self.last_brainamp_data_ts_arrival()
            if not last_ts_arrival == 0:
                break
            time.sleep(0.005)

        super(SimBrainAmpData, self)._start_wait()
            
    def _cycle(self):
        if settings.VERIFY_BRAINAMP_DATA_ARRIVAL:
            self.verify_brainamp_data_arrival(settings.VERIFY_BRAINAMP_DATA_ARRIVAL_TIME)

        super(SimBrainAmpData, self)._cycle()

    def verify_brainamp_data_arrival(self, n_secs):
        time_since_brainamp_started = time.time() - self.ts_start_brainamp
        last_ts_arrival = self.last_brainamp_data_ts_arrival()
     
        if time_since_brainamp_started > n_secs:
            if last_ts_arrival == 0:
                print 'No BrainAmp data has arrived at all'
            else:
                t_elapsed = time.time() - last_ts_arrival
                if t_elapsed > n_secs:
                    print 'No BrainAmp data has arrived in the last %.1f s' % t_elapsed

    def last_brainamp_data_ts_arrival(self):
        return np.max(self.brainamp_source.get(n_pts=1, channels=self.brainamp_source.channels)['ts_arrival'])


    def cleanup(self, database, saveid, **kwargs):
        print "stop recording from brainamp features script"
        
        print "magic to make hdf file open later"
        f = self.supp_hdf_filename
        fl = open(f)
        import time
        import tables
        time.sleep(1.)
        fl.close()
        # h5 = tables.openFile(f, mode='a')
        h5 = tables.open_file(f, mode='a') #nerea
        h5.close()

        print "Cleanup Supp HDF"
        print "\tSupp HDF currently at: %s" % self.supp_hdf_filename

        dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
        print database, 'database'
        if dbname == 'default':
            database.save_data(self.supp_hdf_filename, "supp_hdf", saveid)
        else:
            database.save_data(self.supp_hdf_filename, "supp_hdf", saveid, dbname=dbname)
        super(SimBrainAmpData,self).cleanup(database, saveid, **kwargs)

class ReplayBrainAmpData(traits.HasTraits):

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up a MultiChanDataSource to stream data from the BrainAmp system
        '''

        # Use 'SimEMGData' instead of 'EMGData' as source

        self.reg()

        super(ReplayBrainAmpData, self).init()

    def reg(self):
        from riglib import source
        from ismore.brainamp import rda
        import tempfile
        kwargs = dict(supp_file = tempfile.NamedTemporaryFile())

        self.brainamp_source = source.MultiChanDataSource(rda.ReplayEMGData, 
            name='brainamp', channels=self.brainamp_channels, brainamp_channels=self.brainamp_channels, 
            send_data_to_sink_manager=True, **kwargs)
        
        from riglib import sink
        sink.sinks.register(self.brainamp_source)
        
   
    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the EMG data source
        '''

        self.brainamp_source.start()
        self.ts_start_brainamp = time.time()

        
        try:
            super(ReplayBrainAmpData, self).run()
        finally:
            self.brainamp_source.stop()            
            print "stop recording from brainamp features stop function"  

    def _start_wait(self):
        while 1:
            last_ts_arrival = self.last_brainamp_data_ts_arrival()
            if not last_ts_arrival == 0:
                break
            time.sleep(0.005)

        super(ReplayBrainAmpData, self)._start_wait()
            
    def _cycle(self):
        if settings.VERIFY_BRAINAMP_DATA_ARRIVAL:
            self.verify_brainamp_data_arrival(settings.VERIFY_BRAINAMP_DATA_ARRIVAL_TIME)

        super(ReplayBrainAmpData, self)._cycle()

    def verify_brainamp_data_arrival(self, n_secs):
        time_since_brainamp_started = time.time() - self.ts_start_brainamp
        last_ts_arrival = self.last_brainamp_data_ts_arrival()
     
        if time_since_brainamp_started > n_secs:
            if last_ts_arrival == 0:
                print 'No BrainAmp data has arrived at all'
            else:
                t_elapsed = time.time() - last_ts_arrival
                if t_elapsed > n_secs:
                    print 'No BrainAmp data has arrived in the last %.1f s' % t_elapsed

    def last_brainamp_data_ts_arrival(self):
        return np.max(self.brainamp_source.get(n_pts=1, channels=self.brainamp_source.channels)['ts_arrival'])


class SimBrainAmpData_with_encoder(SimBrainAmpData):
    
    def reg(self):
        from riglib import source
        from ismore.brainamp import rda

        self.brainamp_source = source.MultiChanDataSource(rda.SimEMGData_with_encoder, 
            name='brainamp', channels=self.brainamp_channels, brainamp_channels=self.brainamp_channels, send_data_to_sink_manager=True)
        
        from riglib import sink
        sink.sinks.register(self.brainamp_source)

    def _cycle(self):
        #from ismore.brainamp import rda
        print self.brainamp_source.source
        self.brainamp_source.source.encoder.state = self.state
        print self.state, 'simba_w_end'
        super(SimBrainAmpData_with_encoder, self)._cycle()

class LiveAmpData(BrainAmpData):
    '''Stream EMG/EEG/EOG data from the LiveAmp system via RDA.'''

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up a MultiChanDataSource to stream data from the BrainAmp system
        '''
        from riglib import source
        from riglib import sink
        from ismore.brainamp import rda

        # Send tempfile name to 
        # TODO unclear
        import tempfile
        kwargs = dict(supp_file = tempfile.NamedTemporaryFile())

        self.supp_hdf_filename = kwargs['supp_file'].name

        # TODO unclear
        if not hasattr(self, 'brainamp_channels'):
            import brainamp_channel_lists
            self.brainamp_channels = brainamp_channel_lists.eeg32_raw_filt #eog2_raw_filt #eeg32_raw_filt
            
        if not hasattr(self, 'channels'):
            self.channels = []

        try:
           kwargs['channels_str_2discard'] = self.emg_extractor_kwargs['channels_str_2discard']
           kwargs['channels_str_2keep'] = self.emg_extractor_kwargs['channels_str_2keep']
           kwargs['channels_diag1_1'] = self.emg_extractor_kwargs['channels_diag1_1']
           kwargs['channels_diag1_2'] = self.emg_extractor_kwargs['channels_diag1_2']
           kwargs['channels_diag2_1'] = self.emg_extractor_kwargs['channels_diag2_1']
           kwargs['channels_diag2_2'] = self.emg_extractor_kwargs['channels_diag2_2']
        except:
            print 'no channel selection (reduction) was made'

        self.channels_all = self.brainamp_channels + self.channels
        
        # Create new source and link it to the register
        # Sources are specified in the rda module
        self.brainamp_source = source.MultiChanDataSource(rda.LiveAmpSource, 
            name='brainamp', channels=self.channels_all, brainamp_channels=self.brainamp_channels, 
            send_data_to_sink_manager=True, channels_filt = self.channels, **kwargs)

        sink.sinks.register(self.brainamp_source)

        #FIXME What does this line do?
        super(BrainAmpData, self).init()