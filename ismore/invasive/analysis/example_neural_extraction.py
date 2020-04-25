''' 
Example script to see how tuned neurons are to EMG features
during a compliant movement task. 

Execution notes: 
	1. You may run into trouble when running train._get_tmask(files, tslice, sys_name='task')
			i. I've only run into this issue with bmi files near the end of Phase II. 

	2. If you get an error like 'source registration' not recognized or 'no task source registration',
		do the following: 
			i. Go to riglib.bmi.train
			ii. Go to ~line 30
			iii. Replace the entire function sys_eq  with the one at the bottom of this script
			iv. reload riglib.bmi.train and try again
			v. If you still encounter an error, talk to preeya

'''

# Let's look compliant movement task: 9230
te_list = [9230]
te_list = [11385]

# First let's extract the neural data and bin @ 50 ms: 
cells = None # alternatively we could pass in an array if we only cared about specific cells
#e.g. cells = np.array(([ [1, 1], [2, 1], [3, 1], [5, 1], [5, 2] ]))

from invasive.analysis import tuning_analysis  # Located in ismore.invasive.analysis:

# This function opens the .nev file and .nev.hdf files to extract
#		i) cellnames -- list of all sorted and unsorted units that fired during recording
#		ii) timestamps (TS) of when each unit fired (in units of seconds since start of file recording
# 		ii) waveform (WF) that unit had when it fired

cellnames, TS, WF = tuning_analysis.get_cellnames_and_ts(te_list, cells, include_wf=True)
cellnames, TS, WF = tuning_analysis.get_cellnames_and_ts(te_list, cells, include_wf=False)
# cellnames: channel number + 0.01xunit number
# TS: array with timestamps where they fired [seconds]
# WF: Waveform Length of units


############################
####### RASTER PLOT ########
############################
# Let's try to make a raster plot from this session for task entry 9230
te_name = 9230
te_name = 11385 # te where she opened hand when woke up
f, ax = plt.subplots()

# Cellnames need to be unzipped: 
cellnames_unz = tuning_analysis.unzip_units(np.array(cellnames))

# For the sake of time let's only plot the first 10 units
for ic, c in enumerate(cellnames_unz[:10]):

	# Access each units' firing: 
	ts_unit = TS[te_name][tuple(c)]

	for t in ts_unit: 
		# Plot a vertical line for each time the unit spikes: 
		ax.plot([t, t], [ic, ic+1], 'k-')

ax.set_xlabel('Seconds of Recording')
ax.set_title('Raster Plot')

############################
####### Waveform PLOT ######
############################
# Now let's make a plot of what these units' waveforms look like: 
f2, ax2 = plt.subplots(nrows=3, ncols=3)
for ic, c in enumerate(cellnames_unz[:9]):
	
	# Select subplot:
	axi = ax2[ic / 3, ic % 3]

	# Access waveforms for this unit
	wf_unit = WF[te_name][tuple(c)]

	# wf_unit should be a T x 48 array (each waveform has 48 points spaced 33usec apart)

	# Check that ts_unit has the same size as T 
	ts_unit = TS[te_name][tuple(c)]
	assert len(ts_unit) == wf_unit.shape[0]

	# Plot waveforms: 
	axi.plot(wf_unit.T)
	axi.set_title('WF, Chan: '+str(c[0])+',Unit:'+str(c[1]), fontsize=8)

# funciton to make the plot less ugly. 
plt.tight_layout()

############################################
### Correlate Neural w/ BrainAmp Data ######
############################################
# General approach here is to bin neural data and brainamp data into 50 ms bins
# using the HDF.root.task rows as row boundaries

from db import dbfunctions as dbfn
from ismore import brainamp_channel_lists

# First let's get brainamp data: 
# Get brainamp file: 
te = dbfn.TaskEntry(te_name)
supp_hdf = tables.openFile('/storage/supp_hdf/'+te.name+'.supp.hdf')

# Time stamps from single channel: 
ba_ts = supp_hdf.root.brainamp[:]['chanExtCU']['ts_arrival']
ba_ts = np.linspace(ba_ts[0], ba_ts[-1], len(ba_ts))

# Get HDF task row timestamps (ts) 
rows_hdf_ts = te.hdf.root.task[:]['ts']

# Find the BA index that most closely matches rows_ts in terms of time
ba_ts_ix = np.array([np.argmin(np.abs(ba_ts-t)) for t in rows_hdf_ts])

# Compute Waveform Length in with 50 ms of data. Put value in wfl
wfl = []
for chan in brainamp_channel_lists.emg14_bip:
    ch = []
    for it, start_bin in enumerate(ba_ts_ix[:-1]):
    	end_bin = ba_ts_ix[it+1]
        filt = supp_hdf.root.brainamp[start_bin:end_bin]['chan'+chan]['data']
        rectified = np.abs(filt)
        if len(rectified) == 0:
        	print moose
        wfl_ = np.sum(np.abs(np.diff(rectified)))/len(rectified)
        ch.append(wfl_)
    wfl.append(np.hstack((ch)))
WFL = np.vstack((wfl))

# Now let's get neural data, binned at 50 ms: 
from riglib.bmi import train

# Now we gets the timestamps of the HDF rows 
files = dict(hdf=te.hdf_filename, blackrock=te.blackrock_filenames)

# Can define tslice to only take a portion of the neural data (e.g. first 3 min)
tslice = None 

# Function to retrieve times stamps of HDF rows (variable 'rows') -- taken from 
# arduino pulses sent to neural data file. 
_, rows = train._get_tmask(files, tslice, sys_name='task')

# Here, you should note that 'rows' is the same size as 'rows_hdf_ts' from above. 
# Possible that will be off by one or so (sometimes task sends last arduino pulse 
# before getting a chance to write last HDF.root.task row)
print 'number of HDF.root.task rows : ', len(rows_hdf_ts)
print 'number of arduino pulses recorded in neural data : ', len(rows)

# Now bin the first few units 
binned_spks = []

for ic, c in enumerate(cellnames_unz[:50]):
	ts_unit = TS[te_name][tuple(c)]

	# Remove timestamps outside task bounds: 
	ix, = np.nonzero(np.logical_and( ts_unit >= rows[0], ts_unit<= rows[-1]))

	# Count up binned spike counts:
	counts, _ = np.histogram(ts_unit[ix], rows) 

	binned_spks.append(counts)

binned_spks = np.vstack((binned_spks))

# Now we have binned spikes and binned Waveform Length
assert WFL.shape[1] == binned_spks.shape[1]
T = binned_spks.shape[1]

# New we can do things like correlate the WFL and the units: 
# Let's find a 'tuned unit': 

from decimal import Decimal

proceed = True
for ic in range(50):
	for emg in range(14):
		
		if proceed: 

			# Is this unit significantly tunied: 
			slp, intc, cc, pv, er = scipy.stats.linregress(binned_spks[ic, :], WFL[emg, :])
			if pv < 0.00001:
				
				# Plot significant units: 
				f3, ax = plt.subplots()
		
				# Add random noise to binned spike counts to assist with visualization
				ax.plot(binned_spks[ic, :]+(0.2*np.random.randn(T)), WFL[emg, :], '.')
				ch, unit = cellnames_unz[ic]
				emg_chan = brainamp_channel_lists.emg14_bip[emg]

				pv_sci = '%.2E' % Decimal(str(pv))
				r2_sci = '%.2E' % Decimal(str(cc**2))
				ax.set_title(' Channel: '+str(ch)+', Unit: '+str(unit)+' vs. EMG WFL: '+emg_chan+', pval = '+pv_sci+', R2 = '+r2_sci, fontsize=10)

				xhat = np.linspace(0, np.max(binned_spks[ic, :]), 20)
				yhat = (slp_sig*xhat) + intc
				ax.plot(xhat, yhat, 'k--')
				ax.set_xlim([-1, xhat[-1]+1])

				pr = input('continue ? Enter 1 for yes, 0 for no:  ')
				if bool(pr) is False:
					proceed = False

				plt.close(f3)



###############################################################
#### REPLACE RIGLIB.BMI.TRAIN > sys_eq WITH THE FOLLOWING #####
###############################################################
def sys_eq(sys1, sys2):
    '''
    Determine if two system strings match. A separate function is required because sometimes
    the NIDAQ card doesn't properly transmit the first character of the name of the system.

    Parameters
    ----------
    sys1: string
        Name of system from the neural file
    sys2: string
        Name of system to match

    Returns
    -------
    Boolean indicating whether sys1 and sys2 match
    '''
    if sys2 == 'task':
        if sys1 in ['TAS\x00TASK', 'btqassskh', 'btqassskkkh']:
            return True
        elif sys1[:4] in ['tqas', 'tacs','ttua', 'bttu', 'tttu']:
            return True

    return sys1 in [sys2, sys2[1:], sys2.upper()]


