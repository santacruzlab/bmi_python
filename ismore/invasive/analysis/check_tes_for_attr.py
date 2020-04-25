from db.tracker.models import TaskEntry
from db import dbfunctions as dbfn
import datetime

# Post Behobia
date_gte = datetime.datetime(2017, 11, 12)
te_list = TaskEntry.objects.filter(date__gte=date_gte)

x = []

for te in te_list:
	try:
		tsk = dbfn.TaskEntry(te.pk)
		if tsk.length > 120 and 'bmi' in te.task.name:
			x.append([tsk.date, tsk.params['attractor_speed'], tsk.params['attractor_speed_const']])
		tsk.hdf.close()
	except:
		print ' cant: ', te	
