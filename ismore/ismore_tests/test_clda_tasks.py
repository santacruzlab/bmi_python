import ismore.invasive.bmi_ismoretasks as bmi_ismoretasks
from riglib import experiment
from features.hdf_features import SaveHDF
from ismore.brainamp_features import SimBrainAmpData
import datetime
import numpy as np
import matplotlib.pyplot as plt
from ismore.ismore_tests import test_clda
import tables
import pickle
import time
import unittest

full_session_length = 30.

class Smoothbatch(unittest.TestCase):
    def setUp(self):
        adapting_ssm = 'IsMore'
        adapt_mFR = 'No'
        Task = experiment.make(bmi_ismoretasks.SimCLDAControl, [SaveHDF])    
        targets = bmi_ismoretasks.SimBMIControl.ismore_simple(length=100)
        kwargs=dict(assist_level_time=400., assist_level=(1.,0.),session_length=full_session_length, timeout_time=60., 
            half_life=(20., 20.), half_life_time = 400., clda_update_method='Smoothbatch', clda_adapting_ssm=adapting_ssm, 
            clda_stable_neurons = '', clda_adapt_mFR = adapt_mFR, batch_time=10.)
        self.params_to_compare = ['kf_C', 'kf_Q', 'kf_C_xpose_Q_inv', 'kf_C_xpose_Q_inv_C', 'mFR', 'sdFR']

        task = Task(targets, plant_type='IsMore', **kwargs)
        task.run_sync()
        time.sleep(3)
        fnm = test_clda.save_dec_enc(task, pref='SB_test_')
        print " ******** "
        print fnm
        print " ******** "
        self.fnm = fnm
        self.task = task
        time.sleep(5)
    
    def test_spk_validity(self):
        ''' method to test that binned spikes and kin line up correctly in CLDA part of HDF'''
        hdf = tables.openFile(self.fnm[:-4]+'.hdf')
        n_updates = len(hdf.root.clda)

        ########################
        #### SPIKE MATCHING ####
        ########################

        dec_batch_size = self.task.batch_size
        tsk_batch_size = dec_batch_size*self.task.fps/(1/self.task.decoder.binlen)
        cnt = 0
        for n in range(n_updates):
            spks = hdf.root.clda[n]['spike_counts_batch']
            tsk_spks = hdf.root.task[n*tsk_batch_size:(n+1)*tsk_batch_size]['spike_counts']

            s_bat = np.squeeze(np.array(np.sum(spks, axis=1)))
            s_tsk = np.squeeze(np.array(np.sum(tsk_spks, axis=0)))

            for u in range(len(s_bat)):
                if not np.isclose(s_bat[u], s_tsk[u]):
                    cnt += 1
        print 'count: ', cnt
        self.assertTrue(cnt == 0)
           
    def test_dec_matching(self):
        ########################
        #### Dec  MATCHING ####
        ########################
        # Make sure last saved decoder matches current decoder params
        hdf = tables.openFile(self.fnm[:-4]+'.hdf')
        n_updates = len(hdf.root.clda)

        
        for parm in self.params_to_compare:
            # Last Update: 
            clda_last = hdf.root.clda[n_updates-1][parm]

            mod_parm = parm[parm.find('_')+1:]
            try:
                task_parm = getattr(self.task.decoder.filt, mod_parm)
            except:
                task_parm = getattr(self.task.decoder, mod_parm)

            if type(task_parm) is float:
                assert clda_last == task_parm
            elif type(task_parm) is np.ndarray:
                for u in range(len(task_parm)):
                    assert np.isclose(task_parm[u], clda_last[u])
            elif type(task_parm) is np.matrix:
                tp = np.squeeze(np.array(task_parm).reshape(-1))
                mp = np.squeeze(np.array(clda_last).reshape(-1))
                for u in range(len(tp)):
                    assert np.isclose(tp[u], mp[u])
            print 'passed task vs. clda_last, parameter: %s' %parm

class Baseline(Smoothbatch):
    
    def setUp(self):
        adapting_ssm = 'IsMore'
        adapt_mFR = 'No'
        Task = experiment.make(bmi_ismoretasks.SimCLDAControl, [SaveHDF])    
        targets = bmi_ismoretasks.SimBMIControl.ismore_simple(length=100)
        kwargs=dict(assist_level_time=400., assist_level=(1.,0.),session_length=full_session_length, timeout_time=60., 
            half_life=(20., 20.), half_life_time = 400., clda_update_method='Baseline', clda_adapting_ssm=adapting_ssm, 
            clda_stable_neurons = '', clda_adapt_mFR = adapt_mFR, batch_time = 3.)
        self.params_to_compare = ['mFR', 'sdFR']
        task = Task(targets, plant_type='IsMore', **kwargs)
        task.run_sync()
        fnm = test_clda.save_dec_enc(task, pref='Baseline_test_')
        print " ******** "
        print fnm
        print " ******** "
        self.fnm = fnm
        self.task = task


if __name__ == '__main__':
    unittest.main()