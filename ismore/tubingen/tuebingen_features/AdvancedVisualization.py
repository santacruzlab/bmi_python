import ismore.settings as settings
import numpy as np
import socket

from riglib.experiment import traits
from utils.constants import rad_to_deg
from ismore.machine_settings import *

class AdvancedVisualization(object):
	def __init__(self, *args, **kwargs):

		super(AdvancedVisualization, self).__init__(*args, **kwargs)
		
		#IP address of machine to visualize the exo in Tubingen lab
		self.visualization_machine_IP = visualization_machine_IP

		self.visualization_machine_port = 9900
		self.buffer_size = 1024

		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		try:
			self.s.connect((self.visualization_machine_IP, self.visualization_machine_port))
			print "Exoskeleton 3D visualization started."
		except socket.error:
			print "Could not connect to visualization."

	def init(self):
		
		if self.plant_type == 'ArmAssist':
			self.states_3D_vis = self.pos_states
			self.plant_type_command = 'ADDBASE'
			self.handness = str()
			
			initial_pos = self.targets_matrix['rest'][0][self.states_3D_vis].values
			self.angles = np.array([2])

		else:
			self.plant_type_command = 'ADDEXO'
			self.handness = str(self.arm_side).upper() + ' '
			# Modify order to match the one specified in the 3D visualization code
			# BaseXpos,BaseYpos,BaseHeading,PronoRoll,IndexHeading,GroupHeading,ThumbHeading
			self.states_3D_vis = ['aa_px', 'aa_py', 'aa_ppsi', 'rh_pprono', 'rh_pindex', 'rh_pfing3', 'rh_pthumb']
			initial_pos = settings.starting_pos[self.states_3D_vis].values
			self.angles = np.arange(2, 7)
		
		# Convert values before sending
		initial_pos[self.angles] *= rad_to_deg
		initial_pos_str = self.convert_pos_to_str(initial_pos)

		# Create static exo (defining the target position)
		message =  self.plant_type_command + ' EXOSTATIC' + ' ' + self.handness + initial_pos_str + '::'
		self.s.send(message.encode())
		try:
			id_data = self.s.recv(self.buffer_size)
			self.exostatic_id = id_data.decode().split(":")[1]
			print 'Received: ' + id_data
			print 'Added static target exo: ' + self.exostatic_id
		except socket.timeout:
			print "Connection to Panda3D visualization timed out. Could not receive exoid."
			
		# Create real-time moving exo (defining the position of the exo moving in real-time)
		message =  self.plant_type_command + ' EXOREALTIME' + ' ' + self.handness + initial_pos_str + '::'
		self.s.send(message.encode())
		try:
			id_data = self.s.recv(self.buffer_size)
			self.exorealtime_id = id_data.decode().split(":")[1]
			print 'Received: ' + id_data
			print 'Added real-time exo :' + self.exorealtime_id
		except socket.timeout:
			print "Connection to Panda3D visualization timed out. Could not receive exoid."

		# make the real-time exo transparent
		#message = 'TOGGLETRANSPARENCY ' + self.exorealtime_id + '::'
		#self.s.send(message.encode())

		# set the mat according to the arm side
		message = 'TOGGLEMAT' + ' ' + str(self.arm_side).upper() + '::'
		self.s.send(message.encode())

		# Initialize the trial type to any entry of the target matrix
		self.current_trial_type = 'rest'
		self.current_idx = 0
		self.updated_exostatic = False

		super(AdvancedVisualization, self).init()

	def convert_pos_to_str(self, pos_array):

		pos_str = str()
		for ip, pos in enumerate(pos_array):
			if self.states_3D_vis[ip] == 'rh_pprono':
				# Compensation for new calibration method for rehand
				pos_mod = -1*pos
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
		super(AdvancedVisualization, self)._while_trial()

	def _while_trial_return(self):
		# Check if subgoal_idx has increased. If so, delete previous EXOSTATIC and create a new one at new subtarget.
		if self.goal_idx > self.current_idx:
			self.current_trial_type = 'rest'
			self.current_idx = self.goal_idx
			self.update_exostatic()

		super(AdvancedVisualization, self)._while_trial_return()

	def _cycle(self):

		# Use this code to save trial type
		#if self.state == 'instruct_trial_type':
		#	self.current_trial_type = self.trial_type
		#	self.update_exostatic()

		# Delete old static exo and create a new one when target changes
		if self.state == 'instruct_trial_return' and self.updated_exostatic == False:
			self.current_trial_type = 'rest'
			self.current_idx = 0
			print 'cycle_trial_return: noninv'
			self.update_exostatic()
			self.updated_exostatic = True
		elif self.state == 'instruct_trial_type' and self.updated_exostatic == False:
			#self.current_trial_type = self.targets_matrix['subgoal_names'][self.trial_type][self.target_index+1][0]
			self.current_trial_type = self.trial_type # version in Tubingen
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
				
		# Convert values before sending
		pos_array[self.angles] *= rad_to_deg
		realtime_pos = self.convert_pos_to_str(pos_array)
		
		if self.plant_type == 'ArmAssist':
			realtime_pos = realtime_pos[:-1] + ',0,0,0,0'
		else:
			realtime_pos = realtime_pos[:-1]
		
		# BaseXpos,BaseYpos,BaseHeading,PronoRoll,IndexHeading,GroupHeading,ThumbHeading
		message =  'DATA ' + self.exorealtime_id + ' ' + realtime_pos + '::'
		self.s.send(message.encode())

		super(AdvancedVisualization, self)._cycle()

	def update_exostatic(self):	# Function that updates the position of the EXOSTATIC (i.e. deletes the existing one and creates a new one)
			try:
				pos_array = self.targets_matrix[self.current_trial_type][self.current_idx][self.states_3D_vis].values#/10
			except:
				pos_array = self.targets_matrix[self.current_trial_type][0][self.states_3D_vis].values#/10
			
			# delete the static exo at te previous target
			message =  'DELETE ' + self.exostatic_id + '::'
			self.s.send(message.encode())

			pos_array[self.angles] *= rad_to_deg
			target_pos = self.convert_pos_to_str(pos_array)
			
			# create new static exo at new target position
			message =  self.plant_type_command + ' EXOSTATIC' + ' ' + self.handness + target_pos + '::'
			self.s.send(message.encode())
			try:
				id_data = self.s.recv(self.buffer_size)
				self.exostatic_id = id_data.decode().split(":")[1]
				print 'Added static exo :' + self.exostatic_id
			except socket.timeout:
				print "Connection to Panda3D visualization timed out. Could not receive exoid."
	
			print 'Trial type: ' + self.current_trial_type

			# set the exostatic color according to the target color. Default in white
			color = ' 1,1,1' # white
			if 'red' in self.current_trial_type:
				color = '1,0,0'
			elif 'blue' in self.current_trial_type:
				color = '0,0,1'
			elif 'green' in self.current_trial_type:
				color = '0,1,0'
			elif 'brown' in self.current_trial_type:
				color = '0.55,0.25,0.15'
			elif self.current_trial_type in ['back', 'rest']:
				color = '.36,.14,.34' # purple
						
			#Andreas self.subgoal_names[self.trial_type][self.goal_idx] <-- For F1 tasks

			message = 'SETCOLORBASE ' + self.exostatic_id + ' BASE ' + color + '::'
			message = message + 'SETCOLORBASE ' + self.exostatic_id + ' ARMREST ' + color + '::'
			message = message + 'SETCOLORHAND ' + self.exostatic_id + ' THUMB ' + color + '::'
			message = message + 'SETCOLORHAND ' + self.exostatic_id + ' INDEX ' + color + '::'
			message = message + 'SETCOLORHAND ' + self.exostatic_id + ' FINGERGROUP ' + color + '::'
			message = message + 'SETCOLORHAND ' + self.exostatic_id + ' SUPPRO ' + color + '::'
			
			self.s.send(message.encode())

	def cleanup(self, database, saveid, **kwargs):
		super(AdvancedVisualization,self).cleanup(database, saveid, **kwargs)
		
		message =  'DELETE ' + self.exostatic_id + '::'
		self.s.send(message.encode())

		message =  'DELETE ' + self.exorealtime_id + '::'
		self.s.send(message.encode())

		self.s.close()