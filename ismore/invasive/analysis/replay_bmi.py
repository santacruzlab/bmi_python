from db import dbfunctions as dbfn
import numpy as np
import matplotlib.pyplot as plt
from ismore import plants
import time
from ismore import exo_3D_visualization
import pandas as pd
from ismore import common_state_lists

def replay_bmi_commands(te_num):
	te = dbfn.TaskEntry(te_num)
	command_sent = te.hdf.root.task[:]['command_vel_sent']

	T = command_sent.shape[0]
	ismore_plant = plants.UDP_PLANT_CLS_DICT['IsMore']()
	ismore_plant.enable()

	# exo = exo_3D_visualization.Exo3DVisualizationInvasive()
	# exo.plant_type = 'IsMore'
	# exo.arm_side = 'right'
	# exo.state = 'target'
	# exo.init()
	
	for t in range(200, T):
		time.sleep(.05)
		ismore_plant.send_vel(command_sent[t, :])
		pos = ismore_plant.get_pos()
		# exo.plant_pos = pd.Series(pos, common_state_lists.ismore_pos_states)
		# exo._cycle()
		print pos


def replay_as_tho_no_assist(te_num=8495):
	