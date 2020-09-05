from riglib.experiment import traits
import socket
import json
import db.tracker.models

#from machine_settings import *


class VideoRecording(traits.HasTraits):

	@classmethod
	def pre_init(cls, saveid, **kwargs):
		# Get saveid from db system
		VideoRecording.file_saveid = saveid

	def init(self):
		super(VideoRecording, self).init()
		print "Starting video recording..."

		# Setup connectino to Multicam-Server
		self.video_machine_IP = '172.16.161.64'
		self.video_machine_port = 9998

		# Get the name of the current subject from the database
		subject_name = db.tracker.models.TaskEntry.objects.using("default").get(id=self.file_saveid).subject.name

		# Try to set the configuration of the Multicam-Server (setting the current task entry)
		try:
			sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			sock.settimeout(0.1)
			sock.connect((self.video_machine_IP, self.video_machine_port))

			# Get the current configuration
			command = json.dumps({'Command':'REQ','Data':{'CmdType':'GETCONFIG'}})
			sock.send(command)
			current_config = sock.recv(1024)

			# Replace Sid by current subject name and task entry
			future_config = json.loads(current_config,encoding='utf-8')
			future_config["Sid"] = subject_name+'_'+str(self.file_saveid)
			# Replace "null" in Json by emtpy array (e.g. if there are no cameras or microphones active)
			for key in ["Microphones","Cameras"]:
				if future_config[key] is None:
					future_config[key] = []

			# Send configuration back to the server and terminate
			command = json.dumps({'Command':'POST','Data':{'CmdType':'SETCONFIG', 'Values': future_config}})
			sock.send(command)
			sock.close()
		except socket.timeout:
			print "Connection to Multicamserver timed out."
			print "Could not set taskentry and subject name."

		# Try to send the starting command to the Multicam-Server
		try:
			sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			sock.settimeout(0.1)
			sock.connect((self.video_machine_IP, self.video_machine_port))

			# Send command that starts the video recording to the server
			command = json.dumps({'Command':'CTL','Data':{'CmdType':'START'}})
			sock.send(command)

			# Close socket
			sock.close()
			print "Started video recording."
		except socket.timeout:
			print "Connection to Multicamserver timed out."
			print "Could not start recording."

		print "Started video recording."

	def cleanup(self, database, saveid, **kwargs):
		print "Stopping Video recording..."
		super(VideoRecording,self).cleanup(database, saveid, **kwargs)

		# Try to send the stopping command to the Multicam-Server
		sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		sock.settimeout(0.1)
		try:
			sock.connect((self.video_machine_IP, self.video_machine_port))
			# Send command that stops the video recording
			command = json.dumps({'Command':'CTL','Data':{'CmdType':'STOP'}})
			sock.send(command)
			sock.close()
			print "Stopped video recording."
		except socket.timeout:
			print "Connection to Multicamserver timed out."
			print "Could not stop recording."