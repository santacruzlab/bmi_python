
from riglib.experiment import traits
import subprocess 
import os
import socket
import time

from riglib.experiment import traits
from machine_settings import *


class StartVideo(traits.HasTraits):
	# def __init__(self,saveid):#, *args, **kwargs):
	# 	super(StartVideo, self).__init__(saveid)
	# 	print saveid

	def init(self):
		print "video recording started"
		# super(StartVideo, self).init(self,saveid)

		# self.video_machine_IP = '192.168.137.4' # Tubingen IP
		# self.video_machine_IP = '172.23.20.163' # Tecnalia IP
		# self.video_machine_IP = '172.16.160.33' # ??
		self.video_machine_IP = video_machine_IP
		self.video_machine_port = '60000'
		
		self.video_basename = 'video_%s.avi' % time.strftime('%Y_%m_%d_%H_%M_%S')

		subprocess.Popen(['./video/WebcamClient', self.video_machine_IP, self.video_machine_port, self.video_basename, 'start'], cwd = os.path.expandvars('$HOME'))

		super(StartVideo, self).init()

		# Version by Andreas (08.02.2016)
		#prmDict = {
		#	'loc': '/home/tecnalia/video/WebcamClient',
		#	'targetIp': '192.168.137.3',
		#	'targetPort': '60000',
		#		'targetFolderName': 'test',
		#		'command': 'start'
		#	}

		#	cmd = [prmDict['loc'],prmDict['targetIp'],prmDict['targetPort'],prmDict['targetFolderName'],prmDict['command']]
		#	call(cmd)

		# UDP_IP = "192.168.137.3"
		# UDP_PORT = 60000
		# MESSAGE = "test start"

		# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		# sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))

		### check db/tracker/dbp.py
		# dataname = "{subj}{time}_{num:02}_te{id}.{suff}".format(
  #           subj=entry.subject.name[:4].lower(),
  #           time=time.strftime('%Y%m%d'), num=num+1,
  #           id=entry.id, suff=suff
  #       )
  #       fullname = os.path.join(sys.path, dataname)


	def cleanup(self, database, saveid, **kwargs):
		print "video recording stopped"
		super(StartVideo,self).cleanup(database, saveid, **kwargs)
		# super(StartVideo,self).cleanup(saveid, **kwargs)
		print "save id", saveid
		self.video_basename_final = 'video_%s_te%s.avi' % (time.strftime('%Y_%m_%d_%H_%M_%S'), saveid)
		print self.video_basename_final 
	

		subprocess.Popen(['./video/WebcamClient', self.video_machine_IP, self.video_machine_port, self.video_basename_final, 'stop'], cwd = os.path.expandvars('$HOME'))

		#	prmDict = {
		#		'loc': '/home/tecnalia/video/WebcamClient',
		#		'targetIp': '192.168.137.3',
		#		'targetPort': '60000',
		#		'targetFolderName': 'test',
		#		'command': 'stop'
		#	}
		
		#	cmd = [prmDict['loc'],prmDict['targetIp'],prmDict['targetPort'],prmDict['targetFolderName'],prmDict['command']]
		#	call(cmd)




