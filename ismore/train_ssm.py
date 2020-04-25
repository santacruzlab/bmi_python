#!/usr/bin/python
"""
Script to train the RW state-space model parameters from real trajectory data
"""
from db import dbfunctions as dbfn
import sys
import parse_traj
from common_state_lists import *
import numpy as np

id = sys.argv[1]
te = dbfn.TaskEntry(id)
traj_data = parse_traj.parse_trajectories(te.hdf)

if te.plant_type == "ArmAssist":
	vel_states = aa_vel_states

X1 = []
X2 = []

for trial_type in traj_data:
	vel_data = traj_data["Blue"]["traj"][vel_states].values
	X1.append(vel_data[:-1])
	X2.append(vel_data[1:])

X1 = np.mat(np.vstack(X1).T)
X2 = np.mat(np.vstack(X2).T)

A_vel = X2 * np.linalg.pinv(X1)
res = X2 - A_vel*X1
W_vel = np.cov(np.array(res), bias=1)