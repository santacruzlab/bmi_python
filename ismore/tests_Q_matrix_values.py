import tables
import numpy as np
import matplotlib.pyplot as plt

# B1 targets- IsMore plant

# Q = np.mat(np.diag([15., 15., 15.,15., 15., 15.,15., 5, 5, 5, 5, 5, 5, 5, 0]))  # file test20151126_08_te4355 - 'magenta'

# Q[2] = np.mat(np.diag([5., 5., 5.,5., 5., 5., 5., 5, 5, 5, 5, 5, 5, 5, 0])) # file test20151126_07_te4354 - 'cyan'

# Q[0] = np.mat(np.diag([100., 100., 100.,100., 100., 100.,100., 5, 5, 5, 5, 5, 5, 5, 0])) # file test20151126_05_te4352 - 'red'

# Q[1] = np.mat(np.diag([10., 10., 10.,10., 10., 10., 10., 5, 5, 5, 5, 5, 5, 5, 0])) # file test20151126_04_te4351 - 'green'

# Q0 = np.mat(np.diag([7., 7., 7., 7., 7., 7., 7., 0, 0, 0, 0, 0, 0, 0, 0])) # file test20151126_01_te4348 - 'blue'

# R = 1e6 * np.mat(np.diag([1., 1., 1., 1., 1., 1., 1.]))

# colors: 'blue', 'green', 'red', 'cyan', 'magenta'
hdf_names = ['test20151126_01_te4348', 'test20151126_04_te4351', 'test20151126_05_te4352', 'test20151126_07_te4354','test20151126_08_te4355']

trial_type_to_plot = 'blue'

plt.figure('plant_vel_trial')
plt.figure('command_vel_trial')
plt.figure('plant_vel_trial_return')
plt.figure('command_vel_trial_return')

for name in hdf_names:
	hdf = tables.openFile('/storage/rawdata/hdf/' + name + '.hdf')
	plant_vel = hdf.root.task[:]['plant_vel']
	command_vel = hdf.root.task[:]['command_vel']

	cues = hdf.root.task_msgs[:]['msg']
	cues_trial_type = hdf.root.task[:]['trial_type']
	cues_events = hdf.root.task_msgs[:]['time']


	trial_start_events = cues_events[np.where(cues == 'trial')]
	trial_end_events = cues_events[np.where(cues == 'instruct_trial_return')]

	trial_return_start_events = cues_events[np.where(cues == 'trial_return')]
	trial_return_end_events = cues_events[np.where(cues == 'wait')][1:]

	trial_types = cues_trial_type[trial_start_events]

	# Plot
	plt.figure('command_vel_trial')
	for idx in range(len(trial_start_events)):
		if trial_types[idx] == trial_type_to_plot:
			plt.suptitle(trial_types[idx])
			plt.subplot(2,1,1)
			plt.plot(command_vel[trial_start_events[idx]:trial_end_events[idx],0])
			plt.subplot(2,1,2)
			plt.plot(command_vel[trial_start_events[idx]:trial_end_events[idx],1])
			plt.hold(True)
			plt.show()

	plt.figure('plant_vel_trial')
	for idx in range(len(trial_start_events)):
		if trial_types[idx] == trial_type_to_plot:
			plt.suptitle(trial_types[idx])
			plt.subplot(2,1,1)
			plt.plot(plant_vel[trial_start_events[idx]:trial_end_events[idx],0])
			plt.subplot(2,1,2)
			plt.plot(plant_vel[trial_start_events[idx]:trial_end_events[idx],1])
			plt.hold(True)
			plt.show()

	plt.figure('command_vel_trial_return')
	for idx in range(len(trial_return_start_events)):
		if trial_types[idx] == trial_type_to_plot:
			plt.suptitle(trial_types[idx])
			plt.subplot(2,1,1)
			plt.plot(command_vel[trial_return_start_events[idx]:trial_return_end_events[idx],0])
			plt.subplot(2,1,2)
			plt.plot(command_vel[trial_return_start_events[idx]:trial_return_end_events[idx],1])
			plt.hold(True)
			plt.show()

	plt.figure('plant_vel_trial_return')
	for idx in range(len(trial_return_start_events)):
		if trial_types[idx] == trial_type_to_plot:
			plt.suptitle(trial_types[idx])
			plt.subplot(2,1,1)
			plt.plot(plant_vel[trial_return_start_events[idx]:trial_return_end_events[idx],0])
			plt.subplot(2,1,2)
			plt.plot(plant_vel[trial_return_start_events[idx]:trial_return_end_events[idx],1])
			plt.hold(True)
			plt.show()

	hdf.close()

# B2 targets- ReHand plant
import tables
import numpy as np
# Q = np.mat(np.diag([1., 1., 1.,1., 0.5, 0.5, 0.5, 0.5, 0]))  # file test20151126_12_te4359 - 'blue'

# Q = np.mat(np.diag([15., 15., 15.,15., 5, 5, 5, 5, 0]))  # file test20151126_13_te4360 - 'green'


# R = 1e6 * np.mat(np.diag([1., 1., 1., 1.]))


# colors: 'blue', 'green', 'red', 'cyan', 'magenta'
hdf_names = ['test20151126_12_te4359', 'test20151126_13_te4360']

trial_type_to_plot = 'grasp'

plt.figure('plant_vel_trial')
plt.figure('command_vel_trial')
plt.figure('plant_vel_trial_return')
plt.figure('command_vel_trial_return')

for name in hdf_names:
	hdf = tables.openFile('/storage/rawdata/hdf/' + name + '.hdf')
	plant_vel = hdf.root.task[:]['plant_vel']
	command_vel = hdf.root.task[:]['command_vel']

	cues = hdf.root.task_msgs[:]['msg']
	cues_trial_type = hdf.root.task[:]['trial_type']
	cues_events = hdf.root.task_msgs[:]['time']


	trial_start_events = cues_events[np.where(cues == 'trial')]
	trial_end_events = cues_events[np.where(cues == 'instruct_trial_return')]

	trial_return_start_events = cues_events[np.where(cues == 'trial_return')]
	trial_return_end_events = cues_events[np.where(cues == 'wait')][1:]

	trial_types = cues_trial_type[trial_start_events]

	# Plot
	plt.figure('command_vel_trial')
	for idx in range(len(trial_start_events)):
		if trial_types[idx] == trial_type_to_plot:
			plt.suptitle(trial_types[idx])
			plt.subplot(2,2,1)
			plt.plot(command_vel[trial_start_events[idx]:trial_end_events[idx],0])
			plt.subplot(2,2,2)
			plt.plot(command_vel[trial_start_events[idx]:trial_end_events[idx],1])
			plt.subplot(2,2,3)
			plt.plot(command_vel[trial_start_events[idx]:trial_end_events[idx],2])
			plt.subplot(2,2,4)
			plt.plot(command_vel[trial_start_events[idx]:trial_end_events[idx],3])
			plt.hold(True)
			plt.show()

	plt.figure('plant_vel_trial')
	for idx in range(len(trial_start_events)):
		if trial_types[idx] == trial_type_to_plot:
			plt.suptitle(trial_types[idx])
			plt.subplot(2,2,1)
			plt.plot(plant_vel[trial_start_events[idx]:trial_end_events[idx],0])
			plt.subplot(2,2,2)
			plt.plot(plant_vel[trial_start_events[idx]:trial_end_events[idx],1])
			plt.subplot(2,2,3)
			plt.plot(plant_vel[trial_start_events[idx]:trial_end_events[idx],2])
			plt.subplot(2,2,4)
			plt.plot(plant_vel[trial_start_events[idx]:trial_end_events[idx],3])
			plt.hold(True)
			plt.show()



	plt.figure('command_vel_trial_return')
	for idx in range(len(trial_return_start_events)):
		if trial_types[idx] == trial_type_to_plot:
			plt.suptitle(trial_types[idx])
			plt.subplot(2,2,1)
			plt.plot(command_vel[trial_return_start_events[idx]:trial_return_end_events[idx],0])
			plt.subplot(2,2,2)
			plt.plot(command_vel[trial_return_start_events[idx]:trial_return_end_events[idx],1])
			plt.subplot(2,2,3)
			plt.plot(command_vel[trial_return_start_events[idx]:trial_return_end_events[idx],2])
			plt.subplot(2,2,4)
			plt.plot(command_vel[trial_return_start_events[idx]:trial_return_end_events[idx],3])
			plt.hold(True)
			plt.show()

	plt.figure('plant_vel_trial_return')
	for idx in range(len(trial_return_start_events)):
		if trial_types[idx] == trial_type_to_plot:
			plt.suptitle(trial_types[idx])
			plt.subplot(2,2,1)
			plt.plot(plant_vel[trial_return_start_events[idx]:trial_return_end_events[idx],0])
			plt.subplot(2,2,2)
			plt.plot(plant_vel[trial_return_start_events[idx]:trial_return_end_events[idx],1])
			plt.subplot(2,2,3)
			plt.plot(plant_vel[trial_return_start_events[idx]:trial_return_end_events[idx],2])
			plt.subplot(2,2,4)
			plt.plot(plant_vel[trial_return_start_events[idx]:trial_return_end_events[idx],3])
			plt.hold(True)
			plt.show()

	hdf.close()


# B1 targets-test 23/12/2015 in SS with accel limit controller. targets_matrix 4462. Accel limit aa = 0.5 and rh = 0.02
import tables
import numpy as np
# Q = np.mat(np.diag([30., 30., 30.,30., 30., 30.,30., 10, 10, 10, 10, 10, 10, 10, 0])) #blue - Accel limit aa = 0.5 and rh = 0.02

# Q = np.mat(np.diag([20., 20., 20.,20., 20., 20.,20., 20, 20, 20, 20, 20, 20, 20, 0])) # green - Accel limit aa = 0.5 and rh = 0.02

# Q = np.mat(np.diag([40., 40., 40.,40., 40., 40.,40., 20, 20, 20, 20, 20, 20, 20, 0])) # red - Accel limit aa = 0.5 and rh = 0.02

# Q = np.mat(np.diag([40., 40., 40.,40., 40., 40.,40., 10, 10, 10, 10, 10, 10, 10, 0])) # cyan - Accel limit aa = 0.2 and rh = 0.02

# colors: 'blue', 'green', 'red', 'cyan', 'magenta'
hdf_names = ['test20151223_07_te1283', 'test20151223_08_te1284', 'test20151223_09_te1285','test20151223_10_te1286']

trial_type_to_plot = 'blue'

plt.figure('plant_vel_trial')
plt.figure('command_vel_trial')
plt.figure('plant_vel_trial_return')
plt.figure('command_vel_trial_return')
plt.figure('plant_accel_trial')
plt.figure('command_accel_trial')

for name in hdf_names:
	hdf = tables.openFile('/storage/rawdata/hdf/' + name + '.hdf')
	plant_vel = hdf.root.task[:]['plant_vel']
	command_vel = hdf.root.task[:]['command_vel']

	cues = hdf.root.task_msgs[:]['msg']
	cues_trial_type = hdf.root.task[:]['trial_type']
	cues_events = hdf.root.task_msgs[:]['time']


	trial_start_events = cues_events[np.where(cues == 'trial')]
	trial_end_events = cues_events[np.where(cues == 'instruct_trial_return')]

	trial_return_start_events = cues_events[np.where(cues == 'trial_return')]
	trial_return_end_events = cues_events[np.where(cues == 'wait')][1:]

	trial_types = cues_trial_type[trial_start_events]

	# Plot
	plt.figure('command_vel_trial')
	for idx in range(len(trial_start_events)):
		if trial_types[idx] == trial_type_to_plot:
			plt.suptitle(trial_types[idx])
			plt.subplot(2,1,1)
			plt.plot(command_vel[trial_start_events[idx]:trial_end_events[idx],0])
			plt.subplot(2,1,2)
			plt.plot(command_vel[trial_start_events[idx]:trial_end_events[idx],1])
			plt.hold(True)
			plt.show()

	plt.figure('plant_vel_trial')
	for idx in range(len(trial_start_events)):
		if trial_types[idx] == trial_type_to_plot:
			plt.suptitle(trial_types[idx])
			plt.subplot(2,1,1)
			plt.plot(plant_vel[trial_start_events[idx]:trial_end_events[idx],0])
			plt.subplot(2,1,2)
			plt.plot(plant_vel[trial_start_events[idx]:trial_end_events[idx],1])
			plt.hold(True)
			plt.show()

	plt.figure('command_accel_trial')
	for idx in range(len(trial_start_events)):
		if trial_types[idx] == trial_type_to_plot:
			plt.suptitle(trial_types[idx])
			plt.subplot(2,1,1)
			plt.plot(np.diff(command_vel[trial_start_events[idx]:trial_end_events[idx],0]))
			plt.subplot(2,1,2)
			plt.plot(np.diff(command_vel[trial_start_events[idx]:trial_end_events[idx],1]))
			plt.hold(True)
			plt.show()

	plt.figure('plant_accel_trial')
	for idx in range(len(trial_start_events)):
		if trial_types[idx] == trial_type_to_plot:
			plt.suptitle(trial_types[idx])
			plt.subplot(2,1,1)
			plt.plot(np.diff(plant_vel[trial_start_events[idx]:trial_end_events[idx],0]))
			plt.subplot(2,1,2)
			plt.plot(np.diff(plant_vel[trial_start_events[idx]:trial_end_events[idx],1]))
			# plt.subplot(2,2,3)
			# plt.plot(np.diff(plant_vel[trial_start_events[idx]:trial_end_events[idx],2]))
			plt.hold(True)
			plt.show()
	#import pdb; pdb.set_trace()

	plt.figure('command_vel_trial_return')
	for idx in range(len(trial_return_start_events)):
		if trial_types[idx] == trial_type_to_plot:
			plt.suptitle(trial_types[idx])
			plt.subplot(2,1,1)
			plt.plot(command_vel[trial_return_start_events[idx]:trial_return_end_events[idx],0])
			plt.subplot(2,1,2)
			plt.plot(command_vel[trial_return_start_events[idx]:trial_return_end_events[idx],1])
			plt.hold(True)
			plt.show()

	plt.figure('plant_vel_trial_return')
	for idx in range(len(trial_return_start_events)):
		if trial_types[idx] == trial_type_to_plot:
			plt.suptitle(trial_types[idx])
			plt.subplot(2,1,1)
			plt.plot(plant_vel[trial_return_start_events[idx]:trial_return_end_events[idx],0])
			plt.subplot(2,1,2)
			plt.plot(plant_vel[trial_return_start_events[idx]:trial_return_end_events[idx],1])
			plt.hold(True)
			plt.show()

	hdf.close()
