import tables
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy.stats
from ismore import common_state_lists

def plot_training_time(hdf_fname=hdf_fname):
    # Plot training time as a bar plot
    day_dict = {}

    HDF = tables.openFile(hdf_fname)
    BM = HDF.root.block_metrics
    TM = HDF.root.trial_metrics

    for i in range(len(BM)):
        try:
            day_dict[BM[i]['te_day']].append(BM[i]['blk_len'])
        except:
            day_dict[BM[i]['te_day']] = [BM[i]['blk_len']]

    f, ax = plt.subplots()
    f2, ax2 = plt.subplots()
    
    sorted_days = np.sort(day_dict.keys())
    for i, s in enumerate(sorted_days):
        ax.bar(i, np.sum(day_dict[s])/60., width=1.)

        ix = np.nonzero(TM[:]['te_day']==s)[0]
        ax2.bar(i, len(ix), width=1.)

    for a in [ax, ax2]:
        a.set_xlim([-1, i+1])
        a.set_xticks(np.arange(0.5, i+0.5))
        a.set_xticklabels(sorted_days, rotation=45, fontsize=8)
    ax.set_ylabel('Number of Minutes Trained in BMI Per Day', fontsize=12)
    ax2.set_ylabel('Number of Trials Trained in BMI Per Day', fontsize=12)

def plot_perc_success(hdf_fname=hdf_fname):
    # Plot training time as a bar plot
    day_dict = {}

    HDF = tables.openFile(hdf_fname)
    TM = HDF.root.trial_metrics

    f, ax = plt.subplots()
    
    sorted_days = np.sort(np.unique(TM[:]['te_day']))
    for i, s in enumerate(sorted_days):
        ix = np.nonzero(TM[:]['te_day']==s)[0]
        suc = len(np.nonzero(TM[ix]['success'])[0])
        ax.bar(i, suc/float(len(ix)), width=1.)

    for a in [ax]:
        a.set_xlim([-1, i+1])
        a.set_xticks(np.arange(0.5, i+0.5))
        a.set_xticklabels(sorted_days, rotation=45, fontsize=8)
    ax.set_ylabel('Percent of Successful Trials per Day', fontsize=12)

def hybrid_tes(check=False):

    # No drift correction
    week_5 = [[8758, 8759], 
    [8762, 8763, 8764, 8766, 8767, 8768, 8769], 
    [8785, 8786, 8787, 8788, 8789],
    [8794, 8795, 8796, 8797, 8798, 8799],
    [8832, 8833, 8834, 8835, 8836],
    [8841, 8842, 8843, 8844, 8845, 8846],
    [8862, 8863, 8864, 8865, 8867, 8868]
    ]

    week_6 = [[8921, 8922, 8923, 8924, 8925, 8926],
    [8936, 8937, 8938, 8939, 8940],
    [8953, 8954, 8955, 8956, 8957, 8958],
    [8963, 8964, 8965, 8966],
    [8980, 8981, 8982, 8983, 8984, 8985],
    [9003, 9005, 9006, 9007, 9008],
    [9012, 9013, 9014, 9015, 9016],
    [9026, 9027, 9030, 9031],
    [9047, 9048, 9049, 9050, 9051, 9052, 9053]
    ]

    # no changes
    week_7 = [[9067, 9068, 9069, 9070, 9072, 9073, 9074],
    [9078, 9079, 9080, 9081, 9082],
    [9099, 9100, 9101, 9103, 9104, 9105],
    [9112, 9113, 9114, 9115, 9117, 9119],
    [9131, 9133, 9134],

    # last targets matrix change here -- 13828
    [9154, 9155, 9156, 9157, 9158, 9159],
    [9164, 9165, 9166, 9167, 9168, 9169],
    [9192, 9193, 9194, 9195, 9196, 9197],
    [9201, 9202, 9203, 9204],
    [9219, 9223, 9225, 9226, 9228]
    ]

    week_8 = [[9232, 9233, 9234, 9235, 9236],
    [9261, 9262, 9263, 9264, 9265],
    [9313, 9314, 9315, 9316, 9317, 9318, 9319],# Free movements: , 9320, 9321
    [9326, 9327, 9330, 9332],
    [9348, 9349, 9350, 9351, 9354, 9355, 9356, 9357, 9358, 9359], #free movements: 9357, 9358, 9359
    [9372, 9373]]

    # week of assessments + holidays
    week_9 = [[9380, 9381, 9382, 9383, 9384], #9384 = free, note all blocks here had no exo visualization
    [9389, 9391, 9392, 9394, 9396],
    [9441, 9443, 9444, 9445, 9446, 9447, 9448, 9449, 9450, 9451], # here some have blocked base
    [9468, 9470, 9473, 9474, 9475, 9476, ]] # here some have blocked base

    # first week of "Phase III"
    # added 'fb_gain' velocities --> scaled BMI gains 
    # added 'gotostart' part to rest phase
    week_10 = [[9496, 9497, 9498, 9499, 9500, 9501, 9502 ], # During 9502, changed RH thumb gain to 1.5 (from 1)
    [9506, 9507, 9508, 9510, 9511],
    [9524, 9525, 9526, 9527, 9528, 9530, 9531, 9532, ],
    [9536, 9537, 9538, 9539, 9540, ],
    [9551, 9552, 9553, 9554, 9556, 9557, ],
    [9560, 9561, 9564, 9565],
    [9574, 9578, 9579],
    [9584, 9585, 9589, 9592, 9593, 9594], 
    [9613, 9614, 9615, 9616, 9617],]

    week_11 = [[9626, 9627, 9628, 9629, 9630, 9631,],
    [9628, 9639, 9640, 9641, 9642],
    [9659, 9661, 9662, 9663, 9664]]

    week_12 = [[9717, 9719, 9720, 9721], # note during 9718 -- finger module broke
    [9750, 9751, 9752, 9753, 9754, ], # different hardware bc of broken fingers
    [9762, 9764, 9765, 9766, 9767, 9768],
    [9785, 9786, 9787, 9788, ],
    [9797, 9798, ],
    [9818, 9819, 9820, 9821]] 

    week_13 = [[9825, 9826, 9827, 9828, 9829, 9830, 9831], #9831 -- no feedback screen
    [9832, 9833, 9834, 9835, 9836, 9837], # comment about no feedabck screen -- check w/ Andrea
    [9856, 9858, 9860, 9861, 9862, ], # 9860, 9861, 9862 -- feedback screen off
    [9867, 9869, 9870, 9871, 9873], #feedback screen off
    [9892, 9894, 9896, 9897, 9899, 9901], # feedback screen off, bimanual ball manipulation
    [9912, ], # feedback screen ON
    [9932, 9934, 9935, 9936, 9937, 9938, 9939, 9940, 9941, 9942, ], # 9939 - 9942 inclusive --> feedback screen off 
    [9957, 9959, 9960, 9961, 9962, ] # feedback screen off for all except 9957 
    ]

    # No feedback screen here on out for BMI only: 
    week_14 = [[9974, 9975, 9977, 9978, 9979], 
    [9985, 9988, 9989, 9990, 9991],
    [10005, 10008, 10009, 10010, 10012, 10013], # 10008 onward, new safety grid (S2)
    [10015, 10016, 10018, 10019, 10020], # 10016 onward, new safety grid (S2)
    [10043, 10044, 10046, 10047, 10048],
    [10110, 10111, 10115],
    [10123, 10124, 10126,],
    [10147, 10148, 10150, 10152, 10153, 10155, ]] # 10147 -- very old safety grid

    week_15 = [[10171, 10172, 10174, 10175, 10177, 10179, ],
    [10181, 10182, 10183, 10184, 10186, 10187, 10189],
    [10225, 10226, 10228, 10229, 10230, 10231],
    [10237, 10238, 10239, 10240, 10242], 
    [10280, 10281, 10282, 10283, 10284], # 10281 -- really good
    [10285, 10286, 10288, 10289, 10291, 10292, 10293],
    [10319, 10320, 10322, 10323], 
    [10325, 10327, 10329, 10330], 
    [10350, 10351]]

    week_16 = [[10377, 10378, 10379, 10380, 10381],
    [10385, 10387, 10388, 10389, 10390],
    # Today started BMI w/ preparation period
    [10472, 10473, 10474, 10475],
    [10480, 10482, 10483, 10484, 10485, 10486], # 10482 and onwards --> no prep
    [10519, 10521, 10522, 10525, 10526], # 10519 -- no thumb, 10525&10526 --> no prep (too tired)

    # New Targ Mat for grasp: 10642, 43, 44, 45 (can barely tell though)
    [10638, 10640, 10641, 10642, 10643, 10644, 10645], # no prep in 10638, in 10640 onwards --> only think about what going to do
    ]

    # All prep task: 
    week_17 = [[10704, 10709, 10710],
    [10711, 10712, 10713, 10714, 10715], 
    [10741, 10742, 10743, 10744],
    [10746, 10747, 10748, 10749],
    [10794, 10795, 10796, 10797, 10798],
    [10835, 10836, 10838, 10839, 10851, 10852],
    [10870, 10872, 10874, 10876, 10877, 10878],]

    week_18 = [[ 
    ]]

    TE = np.hstack((week_6 + week_7 + week_8 + week_9 + week_10 + week_11 + week_12 + week_13 + week_14 + week_15 + week_16 + week_17))
    H = ['ismore_hybrid_bmi_w_emg_rest', 'ismore_hybrid_bmi_w_emg_rest_gotostart','ismore_hybrid_bmi_w_emg_rest_gotostart_prp']
    missing_te = []

    if check:
        for t in TE:
            try:
                tsk = models.TaskEntry.objects.get(pk=t)
                if tsk.task.name not in H:
                    print t, tsk.task.name
                DF = tsk.datafile_set.filter()
                for d in DF:
                    if d.system.name == 'hdf':
                        if os.path.exists(d.system.path + '/'+ d.path):
                            pass
                        else:
                            print 'missing HDF: ', t
                            missing_te.append(t)
            except:
                print ' need te: no DB: ', t
                missing_te.append(t)
        print 'missing array: ', missing_te
        if len(missing_te) > 0:
            return np.hstack((missing_te))
        else:
            return missing_te
         
    else:
        return TE

def extract_hybrid_bmi(te_list=hybrid_tes(), hdf_filename='trl_mets_hybrid_no_free_mov',
    skip_te=[], free_mov=[9320, 9321, 9357, 9358, 9359, 9384], first_n_secs=None): 
    
    # initialize hdf file:
    tf = tempfile.NamedTemporaryFile(delete=False)
    h5file = tables.openFile(tf.name, mode="w", title='Trial Metrics Hybrid')
    tab = h5file.createTable('/', 'trial_metrics', trial_metrics)
    tab2 = h5file.createTable('/', 'block_metrics', block_metrics)

    te_list_mod = np.array([i for i in te_list if i not in skip_te and i not in free_mov])

    for te in np.sort(te_list_mod):
        print ' starting TE: ', te
        # begin sorting through tes: 

        tab2row = tab2.row
        tab2row['te_num'] = te

        # get task entry wrapper
        tsk = dbfn.TaskEntry(te)
        hdf = tsk.hdf

        tab2row['blk_len'] = tsk.length
        tab2row['te_day'] = tsk.date.strftime("%y-%m-%d")
        tab2row.append()
        # get target indices ( # of times 'target' appears)
        targ_ix = np.nonzero(hdf.root.task_msgs[:]['msg']=='target')[0]
        targ_ix = np.array([i for i in targ_ix if hdf.root.task_msgs[i+1]['msg'] != 'None'])
        targ_ix_pls1 = targ_ix + 1
        
        targ_tm = hdf.root.task_msgs[targ_ix]['time']

        if first_n_secs is None:
            targ_tm_pls1 = hdf.root.task_msgs[targ_ix_pls1]['time']
        else:
            targ_tm_pls1 = targ_tm + 20*first_n_secs

        # for each target, parse: 
        for nt, (ti, ti1) in enumerate(zip(targ_tm, targ_tm_pls1)):

            # start new HDF row
            tabrow = tab.row

            # basics: 
            tabrow['te_num'] = te
            tabrow['te_name'] = tsk.task.name
            tabrow['te_day'] = tsk.date.strftime("%y-%m-%d")
            tm = tsk.date.time()
            tabrow['te_time'] = tm.strftime('%H-%M-%S')
            tabrow['ignore_dofs'] = list_to_binary(blocking_dict[hdf.root.task.attrs.ignore_correctness_jts])
            tabrow['block_dofs'] = list_to_binary(blocking_dict[hdf.root.task.attrs.blocking_opts])

            ast = np.zeros((7, ))
            ast[[0, 1, 2]] = hdf.root.task[ti]['aa_assist_level']
            ast[[3, 4, 5, 6]] = hdf.root.task[ti]['rh_assist_level']
            tabrow['assist_dofs'] = ast

            # now target specific info: 
            tt = hdf.root.task[ti]['trial_type']+'_'+str(int(float(hdf.root.task[ti]['target_index'])))
            tabrow['trial_type'] = tt

            # where the target is: 
            tabrow['target_pos'] = hdf.root.task[ti]['target_pos'][:7]

            # target metrics: 
            plant_pos = hdf.root.task[ti:ti1]['plant_pos'][:7]
            dist_to_targ = hdf.root.task[ti]['target_pos'][:7] - plant_pos
            dist_to_targ0 = np.abs(hdf.root.task[ti]['target_pos'][:7] - hdf.root.task[ti]['plant_pos'][:7])

            #How close did we get to the target: 
            closest = np.array([np.min(np.abs(dist_to_targ[:, i])) for i in range(7)])
            tabrow['dist_towards_targ_max'] = dist_to_targ0 - closest

            # Final distance along way to the target: 
            tabrow['dist_towards_targ_cum'] = np.abs(hdf.root.task[ti1]['plant_pos'][:7] - hdf.root.task[ti]['plant_pos'][:7])
            tabrow['t2t'] = (ti1 - ti ) / 20.

            # Let's look at BMI and EMG: 
            brain = np.zeros((7, ))
            emg = np.zeros((7, ))
            corr = np.zeros((7, ))
            total = 0

            for t in range(ti, ti1):
                sign_to_targ = np.sign(hdf.root.task[t]['target_pos'][:7] - hdf.root.task[t]['plant_pos'])
                brain_sign_of_vel = np.sign(hdf.root.task[t]['drive_velocity_raw_brain'])
                emg_sign_of_vel = np.sign(hdf.root.task[t]['emg_vel_raw'])

                brain += 0.5*(1+(sign_to_targ*brain_sign_of_vel))
                emg += 0.5*(1+(sign_to_targ*emg_sign_of_vel))
                corr += 0.5*(1+(emg_sign_of_vel*brain_sign_of_vel))
                total += 1

            tabrow['perc_corr_bmi'] = brain/float(total)
            tabrow['perc_corr_emg'] = emg/float(total)
            tabrow['perc_correlated_bmi_emg'] = corr/float(total)

            succ = 0
            if hdf.root.task_msgs[targ_ix_pls1[nt]]['msg'] == 'hold':
                if hdf.root.task_msgs[targ_ix_pls1[nt]+1]['msg'] == 'targ_transition':
                    succ = 1
            tabrow['success'] = succ
            tabrow.append()

        tab.flush()
        tab2.flush()
        hdf.close()

    h5file.close()

    if first_n_secs is None:
        fname = tmp_data_dir+hdf_filename+'.hdf'
    else:
        fname = tmp_data_dir+hdf_filename+'_first_' + str(first_n_secs)+'_secs.hdf'

    shutil.copyfile(tf.name, fname)
    os.remove(tf.name)
    print 'file saved: ', fname

class trial_metrics(tables.IsDescription):
    te_name = tables.StringCol(50)
    te_num = tables.Int32Col()
    te_day = tables.StringCol(8)
    te_time = tables.StringCol(8)
    trial_type = tables.StringCol(20)
    target_pos = tables.Float64Col(shape=(7, ))
    dist_towards_targ_cum = tables.Float64Col(shape=(7, ))
    dist_towards_targ_max = tables.Float64Col(shape=(7, ))
    t2t = tables.Float64Col()
    perc_corr_bmi = tables.Float64Col(shape=(7, ))
    perc_corr_emg = tables.Float64Col(shape=(7, ))
    perc_correlated_bmi_emg = tables.Float64Col(shape=(7, ))
    ignore_dofs = tables.Float64Col(shape=(7, ))
    block_dofs = tables.Float64Col(shape=(7, ))
    assist_dofs = tables.Float64Col(shape=(7, ))
    success = tables.Float64Col(shape=(1, ))

class block_metrics(tables.IsDescription):
    te_num = tables.StringCol(50)
    te_day = tables.StringCol(8)
    blk_len = tables.Float64Col()
