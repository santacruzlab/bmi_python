from db import dbfunctions as dbfn
from riglib.bmi import train, extractor
import tables
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from riglib.blackrock import brpylib

cmap = ['maroon', 'orangered', 'darkgoldenrod', 'olivedrab', 'teal', 'steelblue', 'midnightblue', 'darkmagenta']


def unit_stats_over_te(te_list, cellnames=None, num_sec_per_bin=60., colormesh=False):
    ''' 
    For each unit, plot a timecourse of means and std dev. over time in 1min bins.
    If cellname is None, then iterate through each cell and pause after plotting, else
        cellname is [chan, unit]

    tuning_analysis.unit_stats_over_te([7135], num_sec_per_bin=10, cellnames=[[5, 10]])

    '''
    cellnames_all, TC = get_cellnames_and_ts(te_list, cellnames)

    # Plotting! 
    cellnames_unique = np.unique(np.hstack((cellnames_all)))
    cellnames_unique = unzip_units(cellnames_unique)

    mesh = []
    cells = []
    maxt = 0
    cell_cnt = 0

    for ic, (chan, unit) in enumerate(cellnames_unique):
        meshi = []

        if colormesh is False:
            f, ax = plt.subplots()
        t = 0

        for it, te in enumerate(te_list):
            try:
                ts = TC[te][chan, unit]
                skip = False
            except:
                skip = True

            if not skip:
                if len(ts) > 0:
                    max_bin = int(ts[-1]/num_sec_per_bin)

                    for im in range(max_bin):
                        ix = np.nonzero(np.logical_and(ts >= im*num_sec_per_bin, ts <(im+1)*num_sec_per_bin))[0]
                        if colormesh is False:
                            ax.plot(t, np.sum(len(ix))/float(num_sec_per_bin), '.-', color=cmap[it % 8])
                        t+= num_sec_per_bin/60.
                        meshi.append(np.sum(len(ix))/float(num_sec_per_bin))
        mesh.append(meshi)
        cell_cnt += 1
        maxt = np.max([maxt, len(meshi)])
        cells.append([chan, unit])

        if colormesh is False:
            plt.title(str(chan)+', unit: '+str(unit))
            plt.xlabel('Minutes')
            plt.show()
            cont = input('Continue?')
    if colormesh:
        M = np.zeros((cell_cnt, maxt))

        for im, meshi in enumerate(mesh):
            M[im, :len(meshi)] = meshi
        t = np.arange(M.shape[1])*num_sec_per_bin / 60. # Minutes
        plt.pcolormesh(t, np.arange(len(mesh)), M)
        plt.xlabel('Minutes')
        plt.ylabel('Cells in Decoder')
        plt.title('Change in mFR of cells in Decoder over session, sec resolution: '+str(num_sec_per_bin))
        plt.colorbar()

def plot_modulation_mean_for_each_trial(te, sec_before=1, sec_after=5, hdf_freq=20.):
    ''' 
    For each trial_type, take FR of each unit in decoder
    and plot
    '''

    ix_before = sec_before*hdf_freq
    ix_after = sec_after*hdf_freq

    task_entry = dbfn.TaskEntry(te)
    hdf = task_entry.hdf

    trial_types = np.unique(hdf.root.task[:]['trial_type'])

    # For each trial find trial onset: 
    # instruct_trial_type (#1) ... target .. hold ... targ_trans.. target
    ix = np.nonzero(hdf.root.task_msgs[:]['msg']=='instruct_trial_type')[0]
    ts = hdf.root.task_msgs[ix+1]['time'].astype(int)
    trials = hdf.root.task[ts]['trial_type']
    nunits = hdf.root.task[0]['spike_counts'].shape[0]

    PSTH = {}

    # Plotting: 
    for i_n in range(nunits):
        f, ax = plt.subplots(nrows = len(trial_types))

        for i_t, t in enumerate(trial_types):

            # Relevant trials
            i = np.nonzero(trials==t)[0]

            if len(i) > 0:
                PSTH[t] = []

                for iii, ii in enumerate(i):

                    # Timestamp of relevant trial:
                    tsii = ts[ii]

                    # Add trial to PSTH
                    if (tsii - ix_before) >= 0 and (tsii+ix_after) < len(hdf.root.task):
                        PSTH[t].append(hdf.root.task[tsii - ix_before: tsii + ix_after]['spike_counts'][:, i_n, :])

                x = np.hstack((PSTH[t]))*hdf_freq
                _, _ = plot_mean_and_sem(np.arange(-1*sec_before,sec_after, 1./hdf_freq), x, ax[i_t], array_axis=1, color=cmap[i_t], label=t)
            ax[i_t].set_title('Unit: '+str(i_n))
            ax[i_t].set_ylabel('FR (hz)')
            plt.legend()
        x=input('Continue?: ')

class taskentry(object):
    def __init__(self, length):
        self.length = length

def get_cellnames_and_ts(te_list, cellnames, skip_ts=False, include_wf=False, 
    noise_rejection=True, nev_hdf_name=None, hdf_name=None):
    TC = {}
    WF = {}
    cellnames_all = []
    print "te_list ..............",  te_list
    for ind_te, te in enumerate(te_list):
        tc = {}
        wf = {}

        # Get neural_recording file: 
        if nev_hdf_name is None: 
            try:
                task_entry = dbfn.TaskEntry(te)
                if type(task_entry.blackrock_filenames) is list:
                    nev_fname = [i for i in task_entry.blackrock_filenames if i[-8:] == '.nev.hdf']
                elif task_entry.blackrock_filenames[-8:] == '.nev.hdf':
                    nev_fname = [task_entry.blackrock_filenames]
                if len(nev_fname) == 0:
                    try:
                        nev_fname = [i+'.hdf' for i in task_entry.blackrock_filenames if i[-4:] == '.nev']
                        print 'guessing'
                    except:
                        pass
            except:
                import glob
                nev_fname = glob.glob('/storage/rawdata/blackrock/*te'+str(te)+'.nev.hdf') 
                hdf = glob.glob('/storage/rawdata/hdf/*te'+str(te)+'.hdf') 
                hdf = tables.openFile(hdf[0])

                task_entry = taskentry(len(hdf.root.task)*0.05)
        else:
            nev_fname = [nev_hdf_name[ind_te]]
            hdf = tables.openFile(hdf_name[ind_te])
            task_entry = taskentry(len(hdf.root.task)*0.05)

        # nev_fname = nev_fname[0].encode('ascii','ignore')
        # import pdb; pdb.set_trace()

        if include_wf:
            print 'processing WF'
            # raw_nev_file = [i for i in task_entry.blackrock_filenames if i[-4:] == '.nev']
            # nev_datafile = brpylib.NevFile(raw_nev_file[0])
            
            # raw_nev_data = nev_datafile.getdata()
            # nev_channel_ids = np.array(raw_nev_data['spike_events']['ChannelID'])
            # nev_datafile.close()

        if len(nev_fname) == 0:
            raise Exception('Finish making HDF files from NEV for task entry: %s'%te)

        try:
            hdf = tables.openFile(nev_fname[0])
        except:
            hdf = tables.open_file(nev_fname[0])

        if cellnames is not None:
            NR = np.zeros((len(cellnames), int((task_entry.length + 10)*1000) ))

            for ic, c in enumerate(cellnames): 
                chan = 'channel'+str(int(c[0])).zfill(5)
                try:
                    ss = getattr(hdf.root.channel, chan)
                    ix = np.nonzero(ss.spike_set[:]['Unit'] == c[1])[0]
                    ts = ss.spike_set[ix]['TimeStamp']/30000.
                    NR[ic, np.floor(ts*1000).astype(int)] = 1

                except:
                    ts = np.array([])
                
                tc[c[0], c[1]] = ts
                cellnames_all.append([c[0]+0.01*c[1]])
                
                if include_wf:
                    # Get nev_chan ix:
                    # nev_chan_ix = np.nonzero(nev_channel_ids == c[0])[0]
                    # if len(nev_chan_ix) == 1:
                    #     nev_chan_ix = nev_chan_ix[0]
                    
                    #     # Go back to the raw file
                    #     units = raw_nev_data['spike_events']['Classification'][nev_chan_ix]
                    #     waves = raw_nev_data['spike_events']['Waveforms'][nev_chan_ix]

                    try: 
                        wf[c[0],c[1]] =  ss.spike_set[ix]['Wave']
                    except:
                        pass

                    #     if c[1] != 10:
                    #         u_ix = np.nonzero(np.array(units) == str(c[1]))[0]
                    #         if len(u_ix) == 0:
                    #             u_ix = np.nonzero(np.array(units) == c[1])[0]
                    #     else:
                    #         u_ix = np.nonzero(np.array(units) == 'none')[0]
                    # else:
                    #     u_ix = []
                    #     waves = np.zeros((1, 1))
                    # # Add waves to dict!
                    # wf[c[0],c[1]] = waves[u_ix, :]

        else:
            NR = []
            for c in range(1, 97):
                chan = 'channel'+str(c).zfill(5)
                try:
                    ss = getattr(hdf.root.channel, chan)
                    proceed = True
                except:
                    proceed = False

                if proceed:
                    units = np.unique(ss.spike_set[:]['Unit'])

                    for iu in units:
                        ix = np.nonzero(ss.spike_set[:]['Unit'] == iu)[0]
                        if not skip_ts:
                            ts = ss.spike_set[ix]['TimeStamp']/30000.
                            ts = ts[ts < task_entry.length + 10]
                            tc[c, iu] = ts
                            nr = np.zeros((1, int((task_entry.length + 10)*1000) ))
                            nr[0, np.floor(ts*1000).astype(int)] = 1
                            NR.append(nr)

                        
                        if include_wf:
                            # Get nev_chan ix:
                            nev_chan_ix = np.nonzero(nev_channel_ids == c)[0]
                            assert len(nev_chan_ix) == 1
                            nev_chan_ix = nev_chan_ix[0]
                            
                            # Go back to the raw file
                            units = raw_nev_data['spike_events']['Classification'][nev_chan_ix]
                            waves = raw_nev_data['spike_events']['Waveforms'][nev_chan_ix]

                            if iu != 10:
                                u_ix = np.nonzero(np.array(units) == str(iu))[0]
                                if len(u_ix) == 0:
                                    u_ix = np.nonzero(np.array(units) == iu)[0]
                            else:
                                u_ix = np.nonzero(np.array(units) == 'none')[0]
                
                            # Add waves to dict!
                            # If there's only 1 spike we get a list. Why!
                            if type(waves) is list and len(waves) == 1:
                                waves = waves[0][np.newaxis, :]

                            wf[c, iu] = waves[u_ix, :]

                        cellnames_all.append([c+0.01*iu])
            if not skip_ts:
                NR = np.vstack((NR))
            else:
                NR = np.array([[0., 0.]])
        frac_cells = 0.05*len(cellnames_all)
        if noise_rejection:
            reject_ix = np.nonzero(np.sum(NR, axis=0) > frac_cells)[0]
            print 'Rejecting: ', len(reject_ix) / float(NR.shape[1])*100., ' percent of 1ms bins due to likely noise in te: ', te
        else: 
            reject_ix = np.array([])
            print 'Not using noise rejection! '
        
        if len(reject_ix) > 0:
            unzip_cellnames = unzip_units(np.array(cellnames_all))
            for i, (c0, c1) in enumerate(unzip_cellnames):
                ts = np.floor(tc[c0, c1]*1000)
                kp_ix = [i for i, j in enumerate(ts) if int(j) not in reject_ix]
                tc[c0, c1] = tc[c0, c1][kp_ix]

                if include_wf:
                    wf[c0, c1] = wf[c0, c1][kp_ix, :]
        
        TC[te] = tc
        TC[te, 'rej'] = reject_ix/1000.
        WF[te] = wf

    if include_wf:
        return unzip_units([c[0] for c in cellnames_all]), TC, WF
    else:
        return unzip_units([c[0] for c in cellnames_all]), TC

def cleanse_TC(TC):

    return TC
    
def unzip_units(cells):
    u = []
    for i in cells:
        u.append([int(i), int(np.round((i-int(i))*100))])
    return u

def plot_mean_and_sem(x , array, ax, color='b', array_axis=1, label='0',
    log_y=False, make_min_zero=[False,False]):
    
    mean = array.mean(axis=array_axis)
    sem_plus = mean + scipy.stats.sem(array, axis=array_axis)
    sem_minus = mean - scipy.stats.sem(array, axis=array_axis)
    
    if make_min_zero[0] is not False:
        bi, bv = get_in_range(x,make_min_zero[1])
        add = np.min(mean[bi])
    else:
        add = 0

    ax.fill_between(x, sem_plus-add, sem_minus-add, color=color, alpha=0.5)
    x = ax.plot(x,mean-add, '-',color=color,label=label)
    if log_y:
        ax.set_yscale('log')
    return x, ax

def plot_tuning_to_each_kinematic_var(te_list, cellnames=None, binlen=0.1, 
    pos_key='plant_pos', kin_source='task', kin_ix = 'all', nbins=20):
    '''
    Plot tuning to indivdual kinematics for each cell
    '''
    ## get kinematic data
    K = {}
    N = {}
    CIX = {}

    if cellnames is None:
        units, _ = get_cellnames_and_ts(te_list, None, skip_ts=True)
        units = np.unique(np.hstack((units)))
        units = unzip_units(units)
    else:
        units = cellnames

    for te in te_list:
        task_entry = dbfn.TaskEntry(te)
        files = dict(hdf=task_entry.hdf_filename, blackrock=task_entry.blackrock_filenames)

        #channel indices for this: 
        if type(task_entry.blackrock_filenames) is list:
            nev_hdf = [j for i,j in enumerate(task_entry.blackrock_filenames) if j[-8:] == '.nev.hdf'][0]
        elif task_entry.blackrock_filenames[-8:] == '.nev.hdf':
            nev_hdf = task_entry.blackrock_filenames
        else:
            raise Exception('Missing .nev.hdf file fpr %s'%te)
        hdf2 = tables.openFile(nev_hdf)

        # get unit indices for this te: 
        CIX[te] = []
        units_te = []
        for iu, u in enumerate(units):
            chan = 'channel'+str(u[0]).zfill(5)
            try:
                ss = getattr(hdf2.root.channel, chan)
                ix = np.nonzero(ss.spike_set[:]['Unit'] == u[1])[0]
                if len(ix) > 0:
                    CIX[te].append(iu)
                    units_te.append(u)
            except:
                pass

        tmask, rows = train._get_tmask(files, None, sys_name=kin_source)
        kin = train.get_plant_pos_vel(files, binlen, tmask, update_rate_hz=20., pos_key=pos_key, vel_key=None)

        ## get neural features
        extractor_cls = extractor.BinnedSpikeCountsExtractor
        extractor_kwargs = dict(units=np.vstack((units_te)))

        neural_features, _, extractor_kwargs = train.get_neural_features(files, binlen, extractor_cls.extract_from_file, 
            extractor_kwargs, tslice=None, units=np.vstack((units_te)), source=kin_source)

        K[te] = kin

        neural_features2 = np.zeros((neural_features.shape[0], len(units)))
        for i, ii in enumerate(CIX[te]):
            neural_features2[:, i] = neural_features[:, i]
        N[te] = neural_features2

    # Combine Kin and Neural Features List: 
    K_master = []
    N_master = []

    for te in te_list:
        K_master.append(K[te][1:, :])
        N_master.append(N[te][1:, :])

    K_master = np.vstack((K_master))
    N_master = np.vstack((N_master))

    # Bin kinematics into 20 bins:
    T, nkin = K_master.shape
    if kin_ix is 'all':
        kin_arr = np.arange(nkin)
    else:
        kin_arr = kin_ix

    T, nun = N_master.shape
    X = np.zeros((nun, nbins+1, len(kin_arr)))

    for n in range(nun):
        #f, ax = plt.subplots(nrows=len(kin_arr))
        #if len(kin_arr)==1:
            #ax = [ax]

        for k, ki in enumerate(kin_arr):
            ik = np.linspace(np.min(K_master[:, ki]), np.max(K_master[:, ki]), nbins)
            dig_K = np.digitize(K_master[:, ki], ik)

            for i in range(nbins+1):
                ix = np.nonzero(dig_K == i)[0]
                #if i < len(ik):
                    #ax[k].plot(ik[i], np.mean(N_master[ix, n]), 'k.')
                X[n, i, k] = np.mean(N_master[ix, n])
                #else:
                    #ax[k].plot(ik[-1]+(ik[-1]-ik[-2]), np.mean(N_master[ix, n])/binlen, 'k.') 
        #ax[0].set_title('Unit: '+str(n)+' Name: '+str(units[n][0])+', '+str(units[n][1]))
        #cont = input('Continue?')

    X[np.nonzero(np.isnan(X))]=0
    Xnorm = X / np.mean(X, axis=1)[:, np.newaxis, :]
    for k in range(len(kin_arr)):
        f, ax =plt.subplots()
        c = ax.pcolormesh(Xnorm[:, :, k])
        #plt.colorbar(cax=c)

def extract_neural_bins(hdf_fname, nev_hdf_fname, binlen, units):
    files = dict(hdf=hdf_fname, blackrock=[nev_hdf_fname])

    ## get neural timestamps data
    tmask, rows = get_tmask(hdf_fname, nev_hdf_fname)

    ## get kineamtic data 
    kin = get_plant_pos_vel(files, binlen, tmask)

    ## get neural features
    neurows = rows[tmask]
    extractor_kwargs = {}
    neural_features, units, extractor_kwargs = extract_from_file(files, neurows, binlen, units,
        extractor_kwargs, strobe_rate=20.0)

    kin = kin.T
    neural_features = neural_features.T
    return kin, neural_features
    
def get_plant_pos_vel(files, binlen, tmask, update_rate_hz=20., pos_key='plant_pos'):
    '''
    Get positions and velocity from 'task' table of HDF file

    Parameters
    ----------

    Returns
    -------
    '''
    if pos_key == 'plant_pos':  # used for ibmi tasks
        vel_key = 'plant_vel'

    hdf = tables.openFile(files['hdf'])    
    kin = hdf.root.task[:][pos_key]

    inds, = np.nonzero(tmask)
    step_fl = binlen/(1./update_rate_hz)
    if step_fl < 1: # more than one spike bin per kinematic obs
        if vel_key is not None:
            velocity = hdf.root.task[:][vel_key]
        else:
            velocity = np.diff(kin, axis=0) * update_rate_hz
            velocity = np.vstack([np.zeros(kin.shape[1]), velocity])
        kin = np.hstack([kin, velocity])

        n_repeats = int((1./update_rate_hz)/binlen)
        inds = np.sort(np.hstack([inds]*n_repeats))
        kin = kin[inds]
    else:
        step = int(binlen/(1./update_rate_hz))
        inds = inds[::step]
        kin = kin[inds]
        
        if vel_key is not None:
            velocity = hdf.root.task[inds][vel_key]
        else:
            velocity = np.diff(kin, axis=0) * 1./binlen
            velocity = np.vstack([np.zeros(kin.shape[1]), velocity])
        kin = np.hstack([kin, velocity])

    return kin

def get_tmask(hdf_fname, nev_hdf_fname):
    ''' Find the rows of the nev file '''
    from riglib.dio import parse
    nev_hdf = tables.openFile(nev_hdf_fname)
    ts = nev_hdf.root.channel.digital0001.digital_set[:]['TimeStamp']
    msgs = nev_hdf.root.channel.digital0001.digital_set[:]['Value'] + 2**16
    msgtype = np.right_shift(np.bitwise_and(msgs, parse.msgtype_mask), 8).astype(np.uint8)
    auxdata = np.right_shift(np.bitwise_and(msgs, parse.auxdata_mask), 8+3).astype(np.uint8)
    rawdata = np.bitwise_and(msgs, parse.rawdata_mask)
    data = np.vstack([ts, msgtype, auxdata, rawdata]).T

    # get system registrations
    reg = parse.registrations(data)
    syskey = None

    for key, system in reg.items():
        if sys_eq(system[0], 'task'):
            syskey = key
            break

    if syskey is None:
        raise Exception('No source registration saved in the file!')

    # get the corresponding hdf rows
    rows = parse.rowbyte(data)[syskey][:,0]
    rows = rows / 30000. # sampling rate of blackrock system 
    
    lower, upper = 0 < rows, rows < rows.max() + 1
    tmask = np.logical_and(lower, upper)

    return tmask, rows

def extract_from_file(files, neurows, binlen, units, extractor_kwargs, strobe_rate=60.0):
    '''
    Compute binned spike count features

    Parameters
    ----------
    files : dict
        Data files used to train the decoder. Should contain exactly one type of neural data file (e.g., Plexon, Blackrock, TDT)
    neurows: np.ndarray of shape (T,)
        Timestamps in the plexon time reference corresponding to bin boundaries
    binlen: float
        Length of time over which to sum spikes from the specified cells
    units: np.ndarray of shape (N, 2)
        List of units that the decoder will be trained on. The first column specifies the electrode number and the second specifies the unit on the electrode
    extractor_kwargs: dict 
        Any additional parameters to be passed to the feature extractor. This function is agnostic to the actual extractor utilized
    strobe_rate: 60.0
        The rate at which the task sends the sync pulse to the plx file

    Returns
    -------
    spike_counts : np.ndarray of shape (N, T)
        Spike counts binned over the length of the datafile.
    units : np.ndarray of shape (N, 2)
        Each row corresponds to the channel index (typically the electrode number) and 
        the unit index (an index to differentiate the possibly many units on the same electrode). These are 
        the units used in the BMI.
    extractor_kwargs : dict
        Parameters used to instantiate the feature extractor, to be stored 
        along with the trained decoder so that the exact same feature extractor can be re-created at runtime.
    '''

    nev_fname = [name for name in files['blackrock'] if '.nev' in name][0]  # only one of them
    nev_hdf_fname = [name for name in files['blackrock'] if '.nev' in name and name[-4:]=='.hdf']
    nsx_fnames = [name for name in files['blackrock'] if '.ns' in name]            
    # interpolate between the rows to 180 Hz
    if binlen < 1./strobe_rate:
        interp_rows = []
        neurows = np.hstack([neurows[0] - 1./strobe_rate, neurows])
        for r1, r2 in izip(neurows[:-1], neurows[1:]):
            interp_rows += list(np.linspace(r1, r2, 4)[1:])
        interp_rows = np.array(interp_rows)
    else:
        step = int(binlen/(1./strobe_rate)) # Downsample kinematic data according to decoder bin length (assumes non-overlapping bins)
        interp_rows = neurows[::step]

    nev_hdf = tables.openFile(nev_hdf_fname[0])

    n_bins = len(interp_rows)
    n_units = units.shape[0]
    spike_counts = np.zeros((n_bins, n_units))

    for i in range(n_units):
        chan = units[i, 0]

        # 1-based numbering (comes from web interface)
        unit = units[i, 1]  

        chan_str = str(chan).zfill(5)
        path = 'channel/channel%s/spike_set' % chan_str

        try:
            try:
                grp = nev_hdf.getNode('/'+path)
            except:
                grp = nev_hdf.get_node('/'+path)

            ts = grp[:]['TimeStamp']
            units_ts = grp[:]['Unit']
        except:
            print 'no spikes recorded on channel: ', chan_str, ': adding zeros'
            ts = []
            unit_ts = []


        # get the ts for this unit, in units of secs
        fs = 30000.
        ts = [t/fs for idx, (t, u_t) in enumerate(zip(ts, units_ts)) if u_t == unit]

        # insert value interp_rows[0]-step to beginning of interp_rows array
        interp_rows_ = np.insert(interp_rows, 0, interp_rows[0]-step)

        # use ts to fill in the spike_counts that corresponds to unit i
        spike_counts[:, i] = np.histogram(ts, interp_rows_)[0]

    return spike_counts, units, extractor_kwargs  

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


