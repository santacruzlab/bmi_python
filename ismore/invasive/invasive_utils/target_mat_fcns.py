from ismore import plants
from db.tracker import models
from db import dbfunctions as dbfn
import pickle
from ismore.common_state_lists import *
from collections import OrderedDict
import matplotlib.pyplot as plt
import time
import pandas as pd
import os

'''
Methods to manipulate targets matrix: 
    get_many_points() -- method to grab a position from the exo w/o running a task, 
        first grabs rest positon, then sleep_target position

    add_pos_name_to_TM -- method to add a new target to current targets matrix. Used
        for sleep_target, for example 

    change_grasp_in_target_and_safety -- method to change index and fing3 grasp
        position and safety grid for extension during grasp

'''


def get_many_points():
    try_again = True
    plant = None
    targ_pos = {}
    while try_again:
        f, ax = plt.subplots()

        for t in ['rest', 'sleep_target']:
            inp = input('place exo in '+t+' position and press 1: ')
            if int(inp) == 1:
                pos, plant = grab_current_position(plant=plant)
                targ_pos[t] = pos
                ax.plot(targ_pos[t][0], targ_pos[t][1], '.')

        plt.show()
        try_ = input('Try Again? 0 or 1')
        try_again = bool(try_)

    # Cleanup plant: 
    plant.stop()
    plant.disable()

    return targ_pos

def grab_current_position(plant=None):
    if plant is None:
        plant = plants.IsMorePlantUDP()
        plant.init()
        plant.start()
        time.sleep(4)
    current_postition = plant.get_pos()
    return current_postition, plant

def add_pos_name_to_TM(targ_pos, use_old_rest_pos=True, te=11631, testing=False):
    tsk = dbfn.TaskEntry(te)
    TM = models.DataFile.objects.get(pk=tsk.targets_matrix)
    path = '/storage/target_matrices/'+TM.path

    # Add sleep task to this: 
    tm = pickle.load(open(path))
    subgoal_names_orig = tm['subgoal_names']

    # Add sleep_target, rest: 
    subgoal_names_orig['sleep_target'] = OrderedDict()
    subgoal_names_orig['sleep_target'][0] = ['go']
    subgoal_names_orig['sleep_target'][1] = ['back']

    # Add actual target: 
    tm_sleep = pd.Series(targ_pos['sleep_target'], ismore_pos_states)
    if use_old_rest_pos:
        tm_rest = tm['rest'][0]
    else:
        tm_rest = pd.Series(targ_pos['rest'], ismore_pos_states)
    comb = pd.concat([tm_sleep, tm_rest], axis = 1, ignore_index = True)

    tm['sleep_target'] = comb
    tm['subgoal_names'] = subgoal_names_orig

    ## Store a record of the data file in the database
    storage_dir = '/storage/target_matrices'
    if testing:
        pkl_name_new = 'testing_only.pkl'
    else:
        pkl_name_new = 'phaseIII_TM_w_sleep_targ.pkl'
    pickle.dump(tm, open(os.path.join(storage_dir, pkl_name_new), 'wb'))

    ## Save: 
    dfs = models.DataFile.objects.filter(path = os.path.join(storage_dir, pkl_name_new))
    db_name = 'default'

    te_id = te 
    if len(dfs) == 0:
        data_sys = models.System.make_new_sys('misc')
        data_sys.name = pkl_name_new
        data_sys.entry = models.TaskEntry.objects.using(db_name).get(id=int(te_id))
        data_sys.save_to_file(tm, pkl_name_new, obj_name=None, entry_id=int(te_id))
    else:
        raise Exception
    print 'TM saved :) '
    return os.path.join(storage_dir, pkl_name_new)

def change_grasp_in_target_and_safety(te_id, new_grasp_ext_safe, new_grasp_ext_targ, new_grasp_thumb_back_safe, new_grasp_thumb_back_targ,  suffx=''):
    tsk = dbfn.TaskEntry(te_id)
    TM = models.DataFile.objects.get(pk=tsk.targets_matrix)
    path = '/storage/target_matrices/'+TM.path
    targ = pickle.load(open(path))
    sg = models.DataFile.objects.get(pk=tsk.safety_grid_file)
    sg_path = sg.path
    try:
        safe = pickle.load(open(sg_path))
    except:
        safe = pickle.load(open('/storage/rawdata/safety/'+sg_path))
        sg_path = '/storage/rawdata/safety/'+sg_path
    if new_grasp_ext_safe < safe.rh_pindex[0] or new_grasp_ext_safe < safe.rh_pfing3[0]:
        safe.rh_pindex[0] = new_grasp_ext_safe
        safe.rh_pfing3[0] = new_grasp_ext_safe
        print 'New Safety Grid! '
        sg_path_new = sg_path[:-4]+'_more_fing_ext_'+str(new_grasp_ext_safe)+suffx+'.pkl'
        sg_path_new = '/storage/rawdata/safety/phaseV_20180524.pkl'
        print sg_path_new
        pickle.dump(safe, open(sg_path_new, 'wb'))

        dfs = models.DataFile.objects.filter(path = sg_path_new)

        if len(dfs) == 0:
            data_sys = models.System.make_new_sys('safety')
            data_sys.name = sg_path_new
            data_sys.entry = models.TaskEntry.objects.get(id=int(te_id))
            data_sys.save_to_file( safe, sg_path_new, obj_name=None, entry_id=int(te_id))

        elif len(dfs) >= 1:
             print "Warning: Safety grid with the same name! Choose another suffx!"

    if new_grasp_thumb_back_safe > safe.rh_pthumb[1]:
        safe.rh_pthumb[1] = new_grasp_thumb_back_safe
        print 'New Safety Grid! '
        sg_path_new = sg_path[:-4]+'_more_thumb_flex_'+str(new_grasp_thumb_back_safe)+suffx+'.pkl'
        sg_path_new = 'safety_HUD1_phaseIV_more_fing_ext_more_thumb_flex_grasp.pkl'
        print sg_path_new
        pickle.dump(safe, open(sg_path_new, 'wb'))

        dfs = models.DataFile.objects.filter(path = sg_path_new)

        if len(dfs) == 0:
            data_sys = models.System.make_new_sys('safety')
            data_sys.name = sg_path_new
            data_sys.entry = models.TaskEntry.objects.get(id=int(te_id))
            data_sys.save_to_file( safe, sg_path_new, obj_name=None, entry_id=int(te_id))

        elif len(dfs) >= 1:
             print "Warning: Safety grid with the same name! Choose another suffx!"


    else:
        print 'Old Safety Grid is fine!'

    safe.plot_valid_area()

    # Whenever there's a grasp, set it 
    # Plot targets:
    targ_list = ['red', 'green', 'blue', 'red to green', 'red to blue', 'green to red', 
    'green to blue', 'blue to red', 'blue to green', 'red_up', 'red_down', 'blue_up', 
    'blue_down', 'green_up', 'green_down', 'grasp', 'point', 'up', 'down']

    add_targ_list = ['red_point', 'red_grasp', 'green_point', 
        'green_grasp', 'blue_point', 'blue_grasp','red_grasp_up', 'red_point_down', 
        'green_grasp_down', 'blue_grasp_up','red_grasp_down', 'red_point_up', 
        'green_grasp_up', 'green_point_down','green_point_up','blue_grasp_down',
        'blue_point_up','blue_point_down', 'point_up', 'point_down','grasp_up','grasp_down']


    for t in targ_list + add_targ_list:
        try:
            plt.plot(targ[t][0][0], targ[t][0][1], 'k.')
            proceed=True
        except:
            print 'no target: ', t        


        if proceed:
            # Make sure pronation is good: 
            assert targ[t][0][6] < 0
            assert targ[t][1][6] < 0

            # Make sure finger / index extension is good:
            for i in [4, 5]:
                if targ[t][0][i] < -1*new_grasp_ext_targ:
                    targ[t][0][i] = new_grasp_ext_targ

            # Make sure valid xy position: 
            assert safe.is_valid_pos((targ[t][0][0], targ[t][0][1]))

        # Make thumb rest equal to -0.3
        targ[t][1]['rh_pthumb'] = -0.3

        # Make thumb point == -0.1
        if 'point' in t:
            # Changed from -0.1 --> -0.2 on 1/23/18:
            targ[t][0]['rh_pthumb'] = -0.1

            for t2 in range(targ[t].shape[1]):
                # Changed from -0.1 --> -0.2 on 1/23/18:
                if targ[t][t2]['rh_pthumb'] > -0.1:
                    targ[t][t2]['rh_pthumb'] = -0.1
                    print 'fixing: ', t, ' thumb '

        # Make 'grasp' ext bigger
        if 'grasp' in t:
            targ[t][0]['rh_pindex'] = new_grasp_ext_targ
            targ[t][0]['rh_pfing3'] = new_grasp_ext_targ
            targ[t][1]['rh_pthumb'] = new_grasp_thumb_back_targ



        #for t in add_targ_list+add_targ_list2:
        tg_base = None
        tg_fing = None
        tg_rest = targ['grasp'][1]

        for x in ['red', 'green', 'blue']:
            if x in t:
                tg_base = targ[x][0][[0, 1, 2]]

        for x in ['point', 'grasp']:
            if x in t:
                tg_fing = targ[x][0][[3, 4, 5, 6]]
                #tg_rest = targ[x][1]
                tg_pron = tg_fing[3]

        for x in ['up', 'down']:
            if x in t:
                tg_pron = targ[x][0][6]

        if tg_base is None:
            # Rest position: 
            tg_base = targ['red'][1][[0, 1, 2]]

        #tg = np.vstack((tg_base, tg_fing))
        if tg_base is not None and tg_fing is not None:
            targ[t] = targ['red'].copy()
            targ[t][0][[0, 1, 2]] = tg_base.copy()
            targ[t][0][[3, 4, 5]] = tg_fing[[0, 1, 2]].copy()
            targ[t][0][6] = tg_pron.copy()
            # targ[t][1] = tg_rest.copy() #step removed so the back rest pos is not the same for all movements, thumb closing made bigger for movs involving grasp mov only
            targ['subgoal_names'][t] = OrderedDict()
            targ['subgoal_names'][t][0] = [t]
            targ['subgoal_names'][t][1] = ['back']


    
    subject_name  = models.TaskEntry.objects.get(id=8660).subject.name
    targets_matrix_file_name = 'targets_HUD1_new_more_fing_ext'+str(new_grasp_ext_targ) + '_more_thumb_flex' + str(new_grasp_thumb_back_targ)
    targets_matrix_file_name = 'targets_HUD1_phaseIV_more_fing_ext_more_thumb_flex_grasp'
    targets_matrix_file_name = 'targets_HUD1_phaseV_20180524'
    pkl_name = targets_matrix_file_name + '.pkl'

    ## Store a record of the data file in the database
    storage_dir = '/storage/target_matrices'

    pickle.dump(targ, open(os.path.join(storage_dir, pkl_name), 'wb'))
    dfs = models.DataFile.objects.filter(path = storage_dir+pkl_name)

    if len(dfs) == 0:
        data_sys = models.System.make_new_sys('misc')
        data_sys.name = targets_matrix_file_name
        data_sys.entry = models.TaskEntry.objects.get(id=8681)
        data_sys.save_to_file( targ, pkl_name, obj_name=None, entry_id=8681)
        print 'new targ matrix!'
    else:
        print 'error! This name already exists: ', pkl_name





