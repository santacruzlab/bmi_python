import db.dbfunctions as dbfn
from scipy.spatial import Delaunay
import pickle

def make_hull():
	te = dbfn.TaskEntry(7721)
	pts = te.hdf.root.task[:]['plant_pos'][:, [0, 1, 2]]
	hull3d = Delaunay(pts)
	hull_xy = Delaunay(pts[:, [0, 1]])
	gh = global_hull(hull3d, hull_xy)
	pickle.dump(gh, open('/home/tecnalia/code/ismore/invasive/armassist_hull.pkl', 'wb'))

class global_hull(object):
	def __init__(self, hull3d, hull_xy):
		self.hull3d = hull3d
		self.hull_xy = hull_xy

