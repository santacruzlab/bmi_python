from db import dbfunctions as dbfn
from db.tracker import models
import pickle
import os

# safetygrid we've been using: 
te = dbfn.TaskEntry(10877)
sg = te.safety_grid_file
sg_pth = models.DataFile.objects.get(pk=sg)
safe = pickle.load(open(sg_pth.path))

# Parts of grid new the rest area: Add 5 degress to everything: 
g = safe._grid['min_prono'].copy()
g[~np.isnan(g)] = g[~np.isnan(g)] - (5*np.pi/180)
safe._grid['min_prono'] = g.copy()

storage = '/storage/rawdata/safety'
pkl_name = 'phaseIII_safetygrid_same_minprono_updated.pkl'
print pkl_name
safe.hdf = None

pickle.dump(safe, open(os.path.join(storage, pkl_name), 'wb'))

dfs = models.DataFile.objects.filter(path = pkl_name)

if len(dfs) == 0:
    data_sys = models.System.make_new_sys('safety')
    data_sys.name = pkl_name
    data_sys.entry = models.TaskEntry.objects.get(id=int(te.id))
    data_sys.save_to_file( safe, pkl_name, obj_name=None, entry_id=int(te.id))

elif len(dfs) >= 1:
     print "Warning: Safety grid with the same name! Choose another suffx!"
