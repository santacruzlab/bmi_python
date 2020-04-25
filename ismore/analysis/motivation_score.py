'''
Script to read one or more hdf files and to compute a score out of them (used as a motivation score)
'''

import tables
import pickle
import numpy as np

from db import dbfunctions as dbfn
#from db.tracker import models


hdf_ids = []


db_name = "default"

hdf_names = []
for id in hdf_ids:
    te = dbfn.TaskEntry(id, dbname= db_name)
    hdf_names.append(te.hdf_filename)
    te.close_hdf()

total_nactive_DOFs = []
total_time_2target = []
total_ntrials = []
total_percent_mov = []
total_percent_max_consec_mov = []
total_n_onsets = []
total_latency = []

fs = 20

for fileidx, filename in enumerate(hdf_names):
    # load data from HDF file
    hdf = tables.openFile(filename)
    # load data from supp_HDF file    
    # store_dir_supp = '/storage/supp_hdf/'
    # index_slash = filename.encode('ascii','ignore').rfind('/')            
    # hdf_supp_name = store_dir_supp + filename.encode('ascii','ignore')[index_slash + 1:-3] + 'supp.hdf'
    # hdf_supp = tables.open_file(hdf_supp_name)
	task = hdf.root.task[:]
	task_msgs = hdf.root.task_msgs[:]

	# number of active DOFs (i.e. DOFs involved in the task)
	#active_DOFs = [] #it should be a list
	#nactive_DOFs = len(active_DOFs)
	# instead of number of active DOFs we are gonna use:
	# if only upper arm is involved --> 1/3 points
	# if only hand (+wrist) is involved --> 2/3 points
	# if whole upper limb is involved --> 1 points

	nactive_DOFs = 1/3.0

	from db.tracker import models
	te = models.TaskEntry.objects.get(id= hdf_ids[fileidx])
	sequence = str(te.sequence.generator.name) 
	# mean max velocity in the active DOFs
	# vel = task['command_vel']
	# vel_zscore = np.zeros(vel[:,active_DOFs].shape)
	# for count, dof in enumerate(active_DOFs):
	# 	vel_mean = np.mean(vel[:,dof])
	# 	vel_std = np.std(vel[:,dof])
	# 	vel_zscore[:,count] = (vel[:,dof] - vel_mean)/vel_std

	# trials realizados (= alcanzando el target)
	reached_targets = task['reached_goal_position']
	reached_subtargets = task['reached_subtarget']
	ntargets = len(np.where(reached_targets == True))
	nsubtargets = len(np.where(reached_subtargets == True))
	ntrials = ntargets + nsubtargets

	# mean time to reach target
	cues = task_msgs['msg']
	cues_events = task_msgs['time']
	#cues_trial_type = task['trial_type']
	cues_times = task['ts']

	cues_trial_start = np.sort(np.hstack([np.where(cues == 'trial')[0], np.where(cues == 'trial_return')[0]]))
	cues_trial_end = np.sort(np.hstack([np.where(cues == 'instruct_rest')[0][1:], np.where(cues == 'instruct_rest_return')[0],np.where(cues == 'wait')[0][-1]]))

	time_2target = (cues_times[cues_events[cues_trial_end]] - cues_times[cues_events[cues_trial_start]])/ntrials

	percent_mov = []
	percent_max_consec_mov = []
	n_onsets = []
	latency = []
	for n, idx_start in enumerate(cues_trial_start):
		state_decoder = task['state_decoder'][cues_events[idx_start]:cues_events[cues_trial_end[n]]]
		percent_mov = np.hstack([percent_mov, float(len(np.where(state_decoder == 1)[0]))/len(state_decoder)])
		state_diff = np.diff(state_decoder.ravel())
		onset = np.where(state_diff == 1)[0]
		end = np.where(state_diff == -1)[0]
		if len(end) < len(onset):
			end = np.hstack([end, len(state_decoder)])
		max_consec_mov = max(end - onset)
		percent_max_consec_mov = np.hstack([percent_max_consec_mov, float(max_consec_mov)/len(state_decoder)])
		n_onsets = np.hstack([n_onsets, len(onset)])
		latency = np.hstack([latency, float(onset[0]+1)/fs])

	total_nactive_DOFs = np.hstack([total_nactive_DOFs, nactive_DOFs])
	total_time_2target = np.hstack([total_time_2target, time_2target])
	total_ntrials = np.hstack([total_ntrials, ntrials])
	total_percent_mov = np.hstack([total_percent_mov, percent_mov])
	total_percent_max_consec_mov = np.hstack([total_percent_max_consec_mov, percent_max_consec_mov])
	total_n_onsets = np.hstack([total_n_onsets, n_onsets])
	total_latency = np.hstack([total_latency, latency])

# total_nactive_DOFs = np.mean(total_nactive_DOFs)
# total_time_2target = np.mean(total_time_2target)
# total_ntrials = np.mean(total_ntrials)
# total_percent_mov = np.mean(total_percent_mov)
# total_percent_max_consec_mov = np.mean(total_percent_max_consec_mov)
# total_n_onsets = np.mean(total_n_onsets)
# total_latency = np.mean(total_latency)

score = round(total_nactive_DOFs * total_ntrials * total_percent_max_consec_mov * total_percent_mov / (total_time_2target * total_n_onsets * total_latency))



