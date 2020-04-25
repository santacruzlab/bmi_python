aa_xy_states  = ['aa_px', 'aa_py']
aa_pos_states = ['aa_px', 'aa_py', 'aa_ppsi']
aa_vel_states = ['aa_vx', 'aa_vy', 'aa_vpsi']
rh_pos_states = ['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono']
rh_vel_states = ['rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']
rh_pos_fingers_state = ['rh_pthumb', 'rh_pindex', 'rh_pfing3'] 
rh_vel_fingers_state = ['rh_vthumb', 'rh_vindex', 'rh_vfing3'] 
ismore_pos_states = aa_pos_states + rh_pos_states
ismore_vel_states = aa_vel_states + rh_vel_states

safety_states =  {}
safety_states['ArmAssist'] = ['aa_distXY', 'aa_ppsi']
safety_states['ReHand'] = ['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono']
safety_states['IsMore'] = safety_states['ArmAssist'] + safety_states['ReHand']

safety_states_vel =  {}
safety_states_vel['ArmAssist'] = ['aa_distXY', 'aa_vpsi']
safety_states_vel['ReHand'] = rh_vel_states
safety_states_vel['IsMore'] = safety_states_vel['ArmAssist'] + safety_states_vel['ReHand']

safety_states_min_max =  {}
safety_states_min_max['ArmAssist'] = ['min_aa_distXY', 'max_aa_distXY', 'min_aa_ppsi', 'max_aa_ppsi']
safety_states_min_max['ReHand'] = ['min_rh_pthumb', 'min_rh_pindex', 'min_rh_pfing3', 'min_rh_pprono','max_rh_pthumb','max_rh_pindex', 'max_rh_pfing3', 'max_rh_pprono']
safety_states_min_max['IsMore'] = safety_states_min_max['ArmAssist'] + safety_states_min_max['ReHand']