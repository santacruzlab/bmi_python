''' Script to add subgoal names to be played as audio instructions, goal positions are not added. 
These new trial_types are meant to be used in active movements only, exo not involved.'''


import pickle
from collections import OrderedDict
from db.tracker import models
import os

# this is the target_matrix that we have been using in the last weeks, already modified for safety grid
pkl_name = '/storage/target_matrices/targets_HUD1_7727_7865_8164_None_HUD1_20171122_1502_fixed_thumb_point_all_targs_blue_mod_fix_cha_cha_cha_fix_B3_fix_rest.pkl'
pkl_name = '/storage/target_matrices/TM_active_movements.pkl'
pkl_name = '/storage/target_matrices/TM_active_movements_05_2018_test.pkl'
pkl_name = '/storage/target_matrices/TM_active_movements_24_05_2018.pkl'
pkl_name = '/storage/target_matrices/TM_active_movements_18_07_2018.pkl'


tm = pickle.load(open(pkl_name))

subgoal_names_orig = tm['subgoal_names']


new_trial_types_dict = dict()

# new_trial_types_dict['red_w_bottle'] 	= 'red'
# new_trial_types_dict['green_w_bottle'] 	= 'green'
# new_trial_types_dict['blue_w_bottle'] = 'blue'
# new_trial_types_dict['bottle_to_box_low'] = 'green'
# new_trial_types_dict['bottle_to_box_high'] = 'green'
# new_trial_types_dict['bottle_to_mouth'] = 'up'
# new_trial_types_dict['wring_towel_up'] = 'up'
# new_trial_types_dict['wring_towel_down'] = 'down'
# new_trial_types_dict['wrist_ext'] = 'up'
# new_trial_types_dict['thumb_ext'] = 'grasp'
# new_trial_types_dict['grasp_w_res'] = 'grasp'
# new_trial_types_dict['thumb_ext_w_res'] = 'grasp'
# new_trial_types_dict['scissors'] = 'grasp'


# new_trial_types_dict['wrist_ext_w_res'] = 'up'
# new_trial_types_dict['cylinder_bimanual'] = 'green'
# new_trial_types_dict['tray_to_front_bimanual'] = 'green'
# new_trial_types_dict['cup_to_mouth'] = 'up'
# new_trial_types_dict['open_box_bimanual'] = 'up'

# new_trial_types_dict['hold_bottle_green'] = 'green'

# from 3rd April 2018 on
# new_trial_types_dict['wrist_rotation'] = 'up'
# new_trial_types_dict['fingers_abd_add'] = 'grasp'
# new_trial_types_dict['rolling_pin_front_up'] = 'green'
# new_trial_types_dict['cup_to_box_low'] = 'green'
# new_trial_types_dict['cup_to_box_high'] = 'green'

# new_trial_types_dict['stir_w_spoon'] = 'up'
# new_trial_types_dict['fingers_abd_add_w_res'] = 'grasp'

# new_trial_types_dict['zeros'] = 'grasp'
# new_trial_types_dict['eights'] = 'grasp'

new_trial_types_dict['close_hand'] = 'down'
new_trial_types_dict['hold_arm_up'] = 'up' # shoulder flexion, elbow in extension
new_trial_types_dict['new_exercise'] = 'up'

new_trial_types_dict['hold_fingers_ext'] = 'up'
new_trial_types_dict['hold_wrist_ext'] = 'up'


new_trial_types_list = new_trial_types_dict.keys()

subgoal_names_new = subgoal_names_orig

for new_tt in new_trial_types_list:
	tt_audio = new_trial_types_dict[new_tt]
	subgoal_names_new[new_tt] = subgoal_names_orig[tt_audio]
	tm[new_tt] = tm['rest']


tm['subogal_names'] = subgoal_names_new


## Store a record of the data file in the database
storage_dir = '/storage/target_matrices'

if not os.path.exists(storage_dir):
    os.popen('mkdir -p %s' % storage_dir)

pkl_name_new = 'TM_active_movements_07_11_2018.pkl'
pickle.dump(tm, open(os.path.join(storage_dir, pkl_name_new), 'wb'))

dfs = models.DataFile.objects.filter(path = pkl_name_new)

db_name = 'default'
# dfs = models.System.objects.filter(name=targets_matrix_file_name)
#import pdb; pdb.set_trace()

te_id = 7727 #task entry -- shoul we link it to this task entry? does it matter?
if len(dfs) == 0:
    data_sys = models.System.make_new_sys('misc')
    data_sys.name = pkl_name_new
    data_sys.entry = models.TaskEntry.objects.using(db_name).get(id=int(te_id))
    data_sys.save_to_file( tm, pkl_name_new, obj_name=None, entry_id=int(te_id))

    # # df = models.Decoder()
    # data_sys.path = pkl_name
    # data_sys.name = decoder_name
    # data_sys.entry = models.TaskEntry.objects.using(db_name).get(id=id)
    # #df.entry = models.TaskEntry.objects.using(db_name).get(id=954)
    # df.save()

elif len(dfs) == 1:
    pass # no new data base record needed
elif len(dfs) > 1:
     print "Warning: More than one targets_matrix with the same name! File will be overwritten but name wont show up twice at the database"
     