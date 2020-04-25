import numpy as np
import socket, struct
from ismore import settings, udp_feedback_client
import time
from ismore import common_state_lists, ismore_bmi_lib
import pandas as pd
import pickle
import os

class Patient(object):

    def __init__(self, targets_matrix_file):
        self.addrs = [settings.ARMASSIST_UDP_SERVER_ADDR, settings.REHAND_UDP_SERVER_ADDR]
        


        self.socks = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM), socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]
        self.n_dofs = [range(3), range(3, 7)]
        self.plant_types = ['ArmAssist', 'ReHand']

        self.aa_p = range(3) #common_state_lists.aa_pos_states
        self.rh_p = range(4) #common_state_lists.rh_pos_states
        self.rh_v = range(4, 8) #common_state_lists.rh_vel_states
        self.aa_v = range(3, 6)

        #self.aa = udp_feedback_client.ArmAssistData()
        #self.rh = udp_feedback_client.ReHandData()
        #self.aa.start()
        #self.last_aa_pos = pd.Series(np.zeros((3, )), dtype=self.aa_p) #self.aa.get()['data'][self.aa_p]
        #self.last_aa_pos_t = time.time()

        #self.rh.start()

        assister_kwargs = {
            'call_rate': 20,
            'xy_cutoff': 5,
        }
        self.assister = ismore_bmi_lib.ASSISTER_CLS_DICT['IsMore'](**assister_kwargs)
        self.targets_matrix = pickle.load(open(targets_matrix_file))


    def send_vel(self, vel):
        for i, (ia, sock, ndof, plant) in enumerate(zip(self.addrs, self.socks, self.n_dofs, self.plant_types)):
            self._send_command('SetSpeed %s %s\r' % (plant, self.pack_vel(vel[ndof], ndof)), ia, sock)

    def pack_vel(self, vel, n_dof):
        format_str = "%f " * len(n_dof)
        return format_str % tuple(vel)

    def _send_command(self, command, addr, sock):
        sock.sendto(command, addr)

    def _get_current_state(self):
        #aa_data = self.aa.get()['data']

        with open(os.path.expandvars('$HOME/code/bmi3d/log/armassist.txt'), 'r') as f:
            lines = f.read().splitlines()
            last_line = lines[-2]
        aa_data = np.array([float(i) for i in last_line.split(',')])

        with open(os.path.expandvars('$HOME/code/bmi3d/log/rehand.txt'), 'r') as f:
            lines = f.read().splitlines()
            last_line = lines[-2]
        rh_data = np.array([float(i) for i in last_line.split(',')])

        #daa = np.array([aa_data[0][i] - self.last_aa_pos[0][i] for i in range(3)])
        #aa_vel = daa/(time.time() - self.last_aa_pos_t)
        #self.last_aa_pos = aa_data[self.aa_p]
        #rh_data = self.rh.get()['data']

        pos = np.hstack(( aa_data[self.aa_p], rh_data[self.rh_p] ))
        vel = np.hstack(( aa_data[self.aa_v], rh_data[self.rh_v] ))
        return np.hstack((pos, vel))

    def get_to_target(self, target_pos):
        current_state = np.mat(self._get_current_state()).T
        target_state = np.mat(np.hstack((target_pos, np.zeros((7, ))))).T
        assist_kwargs = self.assister(current_state, target_state, 1., mode=None)
        self.send_vel(10*np.squeeze(np.array(assist_kwargs['Bu'][7:14])))
        return np.sum((np.array(current_state)-np.array(target_state))**2)

    def go_to_target(self, target_name, tix=0):
        if len(self.targets_matrix[target_name].shape) > 1:
            targ = self.targets_matrix[target_name][tix]
        else:
            targ = self.targets_matrix[target_name]
        d = 100
        while d > 20:
            d = self.get_to_target(targ)
            print d
