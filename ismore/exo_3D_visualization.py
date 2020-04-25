from riglib.experiment import traits
import subprocess 
import os
import socket
import time
import settings
import time
import numpy as np
from utils.constants import rad_to_deg
from machine_settings import *


class Exo3DVisualization(object):


	def __init__(self, *args, **kwargs):

		super(Exo3DVisualization, self).__init__(*args, **kwargs)
		
		#IP address of machine to visualize the exo in Tubingen lab

		#self.visualization_machine_IP = '127.0.0.1' #'192.168.137.4'
		#self.visualization_machine_IP = '192.168.137.4'#'127.0.0.1'
		self.visualization_machine_IP = visualization_machine_IP

		self.visualization_machine_port = 9900
		self.buffer_size = 1024

		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
		#self.s = socket(AF_INET, SOCK_STREAM,0) 
        #self.s.setsockopt(IPPROTO_TCP,TCP_NODELAY, 1)
		self.s.connect((self.visualization_machine_IP, self.visualization_machine_port))
		print "exoskeleton 3D visualization started"

	def init(self):
		
		if self.plant_type == 'ArmAssist':
			self.states_3D_vis = self.pos_states
			self.plant_type_command = 'ADDBASE'
			self.handness = str()
			#initial_pos = np.hstack([settings.starting_pos[self.states_3D_vis].values, np.array([0,0,0,0])])# Instead of this, set the static exo at the target position
			#initial_pos = settings.starting_pos[self.states_3D_vis].values/10
			initial_pos = self.targets_matrix['rest'][0][self.states_3D_vis].values#/10
			self.angles = np.array([2])

		else:
			self.plant_type_command = 'ADDEXO'
			self.handness = str(self.arm_side).upper() + ' '
			# modify order to match the one specified in the 3D visualization code
			# BaseXpos,BaseYpos,BaseHeading,PronoRoll,IndexHeading,GroupHeading,ThumbHeading
			self.states_3D_vis = ['aa_px', 'aa_py', 'aa_ppsi', 'rh_pprono', 'rh_pindex', 'rh_pfing3', 'rh_pthumb']
			initial_pos = settings.starting_pos[self.states_3D_vis].values#/10
			self.angles = np.arange(2, 7)
		
		# Add, when ready, the possibility of using just the RH
		initial_pos[self.angles] *= rad_to_deg

		# Convert pos to str and add commas - add a function to do this because it is repeated three times in the code!!!
		initial_pos_str = self.convert_pos_to_str(initial_pos)

		# Create static exo (defining the target position)
		message =  self.plant_type_command + ' EXOREALTIME' + ' ' + self.handness + initial_pos_str
		self.s.send(message.encode())
		time.sleep(0.1)
		id_data = self.s.recv(self.buffer_size)
		self.exostatic_id = id_data.decode().split(":")[1]
		print 'Added static target exo'
	
		# Create real-time moving exo (defining the position of the exo moving in real-time)
		message =  self.plant_type_command + ' EXOREALTIME' + ' ' + self.handness + initial_pos_str# only if EXO this needed
		self.s.send(message.encode())
		id_data = self.s.recv(self.buffer_size)
		self.exorealtime_id = id_data.decode().split(":")[1]
		print 'Added real-time exo'

		# make the real-time exo transparent
		message = 'TOGGLETRANSPARENCY ' + self.exorealtime_id
		self.s.send(message.encode())
		time.sleep(0.1)

		# set the mat according to the arm side
		message = 'TOGGLEMAT' + ' ' + str(self.arm_side).upper() 
		self.s.send(message.encode())
		time.sleep(0.1)

		# Initialize the trial type to any entry of the target matrix
		self.current_trial_type = 'rest'#self.targets_matrix.keys()[0]
		self.current_idx = 0
		self.updated_exostatic = False

		super(Exo3DVisualization, self).init()

	def convert_pos_to_str(self, pos_array):

		pos_str = str()
		for ip, pos in enumerate(pos_array):
			if self.states_3D_vis[ip] == 'rh_pprono':
				# Compensation for new calibration method for rehand
				pos_mod = -1*pos - 85
			else:
				pos_mod = pos
			pos_str = pos_str + str(pos_mod) + ',' 
		pos_str = pos_str[:-1] 

		return pos_str

	def _while_trial(self):
		# Check if subgoal_idx has increased. If so, delete previous EXOSTATIC and create a new one at new subtarget.
		if self.goal_idx > self.current_idx:
			self.current_idx = self.goal_idx
			self.current_trial_type = self.trial_type
			self.update_exostatic()

	def _while_trial_return(self):

		# Check if subgoal_idx has increased. If so, delete previous EXOSTATIC and create a new one at new subtarget.
		if self.goal_idx > self.current_idx:
			self.current_trial_type = 'rest'
			self.current_idx = self.goal_idx
			self.update_exostatic()

	def _cycle(self):
		
		# Delete old static exo and create a new one when target changes
		if self.state == 'instruct_trial_return' and self.updated_exostatic == False:
			self.current_trial_type = 'rest'
			self.current_idx = 0
			print 'cycle_trial_return: noninv'
			self.update_exostatic()
			self.updated_exostatic = True
		elif self.state == 'instruct_trial_type' and self.updated_exostatic == False:
			self.current_trial_type = self.targets_matrix['subgoal_names'][self.trial_type][self.target_index+1][0]
			#self.current_trial_type = self.trial_type # version in Tubingen
			if self.current_trial_type == 'back':
				self.current_trial_type = 'rest'
			self.current_idx = 0
			print 'cycle_trial_type: noninv'
			self.update_exostatic()
			self.updated_exostatic = True
		else:
			self.updated_exostatic = False

		# Send current position to exorealtime
		pos_array = self.plant_pos[self.states_3D_vis].values.copy()#/10 # Check units of px and py!!!
		pos_array[self.angles] *= rad_to_deg

		# Convert pos to str and add commas - add a function to do this because it is repeated three times in the code
		realtime_pos = self.convert_pos_to_str(pos_array)
		
		if self.plant_type == 'ArmAssist':
			realtime_pos = realtime_pos[:-1] + ',0,0,0,0'
		else:
			realtime_pos = realtime_pos[:-1]
		
		# BaseXpos,BaseYpos,BaseHeading,PronoRoll,IndexHeading,GroupHeading,ThumbHeading
		message =  'DATA ' + self.exorealtime_id + ' ' + realtime_pos + '::'
		self.s.send(message.encode())

		super(Exo3DVisualization, self)._cycle()

	def update_exostatic(self):	# Function that updates the position of the EXOSTATIC (i.e. deletes the existing one and creates a new one)
			try:
				pos_array = self.targets_matrix[self.current_trial_type][self.current_idx][self.states_3D_vis].values#/10
			except:
				pos_array = self.targets_matrix[self.current_trial_type][0][self.states_3D_vis].values#/10
			
			#time.sleep(0.1)
			# delete the static exo at te previous target
			message =  'DELETE ' + self.exostatic_id 
			self.s.send(message.encode())
			time.sleep(.1)
			pos_array[self.angles] *= rad_to_deg
			target_pos = self.convert_pos_to_str(pos_array)
			
			# create new static exo at new target position
			message =  self.plant_type_command + ' EXOSTATIC' + ' ' + self.handness + target_pos
			self.s.send(message.encode())
			time.sleep(.01)
			id_data = self.s.recv(self.buffer_size)
			self.exostatic_id = id_data.decode().split(":")[1]
			
			# set the exostatic color according to the target color. Default in white
			color = ' 1,1,1' # white
			if self.current_trial_type == 'red':
				color = ' 1,0,0'
			elif self.current_trial_type == 'blue':
				color = ' 0,0,1'
			elif self.current_trial_type == 'green':
				color = ' 0,1,0'
			elif self.current_trial_type == 'brown':
				color = ' 0.55,0.25,0.15'
			elif self.current_trial_type in ['back', 'rest']:
				color = ' .8,.8,.8' # white
			
			message = 'SETCOLOR ' + self.exostatic_id + color 
			self.s.send(message.encode())
			time.sleep(0.1)

	def cleanup(self, database, saveid, **kwargs):
		super(Exo3DVisualization,self).cleanup(database, saveid, **kwargs)
		
		message =  'DELETE ' + self.exostatic_id 
		self.s.send(message.encode())
		time.sleep(0.5)

		message =  'DELETE ' + self.exorealtime_id 
		self.s.send(message.encode())
		time.sleep(0.5)

		self.s.close()

class Exo3DVisualizationInvasive(Exo3DVisualization):

	def init(self):
		super(Exo3DVisualizationInvasive, self).init()
		message = 'SETCOLOR '+self.exorealtime_id + ' 0.5,0.5,0.5'
		self.s.send(message.encode())
		time.sleep(.01)

		self.dof_ix_to_color_name = dict()
		self.dof_ix_to_color_name[0] = ['SETCOLORBASE', 'BASE']
		self.dof_ix_to_color_name[1] = ['SETCOLORBASE', 'BASE']
		self.dof_ix_to_color_name[2] = ['SETCOLORBASE', 'ARMREST']

		self.dof_ix_to_color_name[3] = ['SETCOLORHAND', 'THUMB']
		self.dof_ix_to_color_name[4] = ['SETCOLORHAND', 'INDEX']
		self.dof_ix_to_color_name[5] = ['SETCOLORHAND', 'FINGERGROUP']
		self.dof_ix_to_color_name[6] = ['SETCOLORHAND', 'SUPPRO']


	def _cycle(self):

		# Delete old static exo and create a new one when target changes
		if self.state == 'instruct_trial_type':
			self.current_trial_type = self.trial_type
			self.update_exostatic()

		# Send current position to exorealtime
		pos_array = self.plant_pos[self.states_3D_vis].values.copy()#/10 # Check units of px and py!!!
		pos_array[self.angles] *= rad_to_deg

		# Convert pos to str and add commas - add a function to do this because it is repeated three times in the code
		realtime_pos = self.convert_pos_to_str(pos_array)
		
		if self.plant_type == 'ArmAssist':
			realtime_pos = realtime_pos[:-1] + ',0,0,0,0'
		else:
			realtime_pos = realtime_pos[:-1]
		
		# BaseXpos,BaseYpos,BaseHeading,PronoRoll,IndexHeading,GroupHeading,ThumbHeading
		message =  'DATA ' + self.exorealtime_id + ' ' + realtime_pos + '::'
		self.s.send(message.encode())

		super(Exo3DVisualization, self)._cycle()

	def _start_hold(self):
		# If has entered target - make target new color
		color = ' 0.5,0.5,0.0'
		sp = ' '
		relevant_joints = [i for i in range(7) if i not in self.ignore_correctness]
		message = ''
		for j in relevant_joints:
			m = self.dof_ix_to_color_name[j][0]+sp+self.exostatic_id+sp+self.dof_ix_to_color_name[j][1]+color+'::'
			message = message+m

		self.s.send(message.encode())
		time.sleep(.01)
		super(Exo3DVisualizationInvasive, self)._start_hold()

	def update_exostatic(self):	# Function that updates the position of the EXOSTATIC (i.e. deletes the existing one and creates a new one)
		try:
			pos_array = self.targets_matrix[self.trial_type][self.target_index][self.states_3D_vis].values#/10
		except:
			pos_array = self.targets_matrix[self.trial_type][0][self.target_index].values#/10
		
		#time.sleep(0.1)
		# delete the static exo at te previous target
		#HACK
		
		#message =  'DELETE ' + self.exostatic_id 
		#self.s.send(message.encode())
		#time.sleep(.01)
		pos_array[self.angles] *= rad_to_deg
		target_pos = self.convert_pos_to_str(pos_array)
		message =  'DATA ' + self.exostatic_id + ' ' + target_pos + '::'
		# create new static exo at new target position
		#message =  self.plant_type_command + ' EXOSTATIC' + ' ' + self.handness + target_pos
		
		self.s.send(message.encode())
		#time.sleep(.01)
		#id_data = self.s.recv(self.buffer_size)
		#self.exostatic_id = id_data.decode().split(":")[1]
			
		# set the exostatic color according to the target color. Default in white
		color = ' .1,.1,.1' # dark gray

		try:
			tt = self.targets_matrix['subgoal_names'][self.trial_type][self.target_index][0]
		except:
			tt = ''

		if 'red' in tt:
			color = ' 1,0,0'
		elif 'blue' in tt:
			color = ' 0,0,1'
		elif 'green' in tt:
			color = ' 0,1,0'
		elif 'brown' in tt:
			color = ' 0.55,0.25,0.15'
		elif 'back' in tt:
			color = ' 1.,.03,.49' # furscia
		elif 'rest' in tt:
			color = ' .8,.8,.8' # white
			
		# Figure out which joints to set:
		sp = ' '
		relevant_joints = [i for i in range(7) if i not in self.ignore_correctness]
		message = ''
		for j in relevant_joints:
			m = self.dof_ix_to_color_name[j][0]+sp+self.exostatic_id+sp+self.dof_ix_to_color_name[j][1]+color+'::'
			message = message+m
		
		color_neut = ' 0.5,0.5,0.5'
		for j in self.ignore_correctness:
			m = self.dof_ix_to_color_name[j][0]+sp+self.exostatic_id+sp+self.dof_ix_to_color_name[j][1]+color_neut+'::'
			message = message+m

		self.s.send(message.encode())
		time.sleep(0.01)
		
		#self.s.close() #needed

class BMIMonitor(object):
	import socket
	def __init__(self, *args, **kwargs):
		HOST = visualization_machine_IP
		PORT = 9901
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.s.connect((HOST, PORT))
		super(BMIMonitor, self).__init__(*args, **kwargs)
	
	def _cycle(self):
		self.s.send('NONE_0')
		super(BMIMonitor, self)._cycle()
	
	# def _start_reward(self):
	# 	self.s.send('REWARD_0')
	# 	super(BMIMonitor, self)._start_reward()

	def _start_hold(self):
		self.s.send('REWARD_0')
		super(BMIMonitor, self)._start_hold()


	def _start_timeout_penalty(self):
		dist = np.sqrt(np.sum(((self.plant_pos[[0, 1]] - self.target_pos[[0, 1]])**2)))
		dist_str = 1./1000*np.round(dist*1000.)
		self.s.send('TIMEOUT_'+str(dist_str))
		super(BMIMonitor, self)._start_timeout_penalty()