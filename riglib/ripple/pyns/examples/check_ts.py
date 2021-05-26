import pyns
from matplotlib import pyplot
import numpy

input_file = "C:\Users\elliottb\Downloads\WileEDriftChoice0007.nev"

nsfile = pyns.NSFile(input_file)

event_entities = [e for e in nsfile.get_entities(1)]
entity = event_entities[0]

last_ts = 0
curr_ts = 0

diff = numpy.zeros(entity.item_count - 1)
for index in range(0, entity.item_count):
    data = entity.get_event_data(index)
    curr_ts = data[0]
    if last_ts == 0:
        last_ts = data[0]
        continue
    diff[index-1] = last_ts - curr_ts
    if curr_ts < last_ts:
        print '{0}: {1}'.format(index-1, last_ts)
        print '{0}: {1}'.format(index, curr_ts)

pyplot.hist(diff, 1000)
raw_input()

