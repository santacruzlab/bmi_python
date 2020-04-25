# For a given task entry
te = 11630

channels_2train = [
    'InterFirst',
    'AbdPolLo',
    'ExtCU',
    'ExtCarp',
    'ExtDig',
    'FlexDig',
    'FlexCarp',]

hand = [0, 1]
Ext = [2, 3, 4]
Flex = [5, 6]

# plot emg features: 
tsk = dbfn.TaskEntry(te)
task = tsk.hdf.root.task
task_msgs = tsk.hdf.root.task_msgs

# Extensors and flexors
plt.plot(task[:]['grasp_emg_classifier_features_Z'][:, hand], color='green')
plt.plot(task[:]['grasp_emg_classifier_features_Z'][:, Ext], color='blue')
plt.plot(task[:]['grasp_emg_classifier_features_Z'][:, Flex], color='red')

# Index velocity: 
plt.plot(task[:]['grasp_emg_output'][:, 1]*10, color='black')

plt.legend(['EMG hand','', 'EMG Ext','','', 'EMG Flex','', 'class_output'])

# 'Go cue'
ix = np.nonzero(task_msgs[:]['msg']=='target')[0]
for i in ix:
	if task[task_msgs[i]['time']]['target_index'][0] == 0:
		plt.plot([task_msgs[i]['time'], task_msgs[i]['time']], [-5, 5], 'k--')
	else:
		plt.plot([task_msgs[i]['time'], task_msgs[i]['time']], [-5, 5], 'k-')

