
from ismore.invasive.make_global_armassist_hull import global_hull
import pickle

def test(te_num=7672):
	te = dbfn.TaskEntry(te_num)
	safe = pickle.load(open('/storage/rawdata/hdf/test20171009_09_te7585_safetygrid_2017_10_09_AM_v3.pkl'))
	safe2 = pickle.load(open('/storage/rawdata/hdf/test20171009_09_te7585_safetygrid_2017_10_09_AM_v5.pkl'))
	global_hull = pickle.load(open('/home/tecnalia/code/ismore/invasive/armassist_hull.pkl'))
	p = te.hdf.root.task[:]['plant_pos'][:, :3]

	#f, ax = plt.subplots()
	f, ax2 = plt.subplots()
	for p0 in p:
		# if safe.is_valid_pos(p0[[0, 1]]) == False:
		# 	global_ok = global_hull.hull_xy.find_simplex(p0[[0, 1]])>= 0
		# 	if global_ok == False:
		# 		col = 'c'
		# 	else:
		# 		col = 'g'
		# else:
		# 	col = 'k'

		#ax.plot(p0[0], p0[1], '.', color=col) 

		if safe2.is_valid_pos(p0[[0, 1]]) == False:
			col = 'c'
		else:
			col = 'k'
		ax2.plot(p0[0], p0[1], '.', color=col, markersize=20) 
		

        
