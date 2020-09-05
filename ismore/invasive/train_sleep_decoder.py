from riglib.bmi import train

def train(baseline_te=12020, cortical_map_2_or_3=2, 
    freq_lim=[-1, 1.2], prob_t1=0.9, prob_t2=0.1, saturate_perc=80):


    targets_matrix_file = '/storage/target_matrices/phaseIII_TM_w_sleep_targ_GGU.pkl'

    # Unit list: 
    unit_dict = pickle.load(open('/home/tecnalia/code/ismore/invasive/sleep_unit_list_mod.pkl'))
    unit_name = unit_dict['U']
    if cortical_map_2_or_3 == 2:
        e1_units = np.vstack(( unit_name[0] )) [:, :2]
        e2_units = np.vstack(( unit_name[1] )) [:, :2]
    elif cortical_map_2_or_3 == 3:
        e1_units = np.vstack(( unit_name[2] )) [:, :2]
        e2_units = np.vstack(( unit_name[3] )) [:, :2]  

    decoder, nrewards = train.test_IsmoreSleepDecoder(baseline_te, e1_units, e2_units, 
        nsteps=1, prob_t1 = prob_t1, prob_t2 = prob_t2, timeout = 15.,
        timeout_pause=1., freq_lim = freq_lim, targets_matrix=targets_matrix_file,
        skip_sim=True, saturate_perc=saturate_perc)


def link_pkl_file_from_USB_DISK_to_db(compliant_te):
    import pickle
    decoder = pickle.load(open('/media/USB_DISK/sleep_decoder_'+str(compliant_te)+'.pkl'))
    pickle.dump(decoder, open('/storage/decoders/sleep_from_te'+str(compliant_te)+'.pkl', 'wb'))
    from db.tracker import dbq
    dbq.save_bmi('sleep_from_te'+str(compliant_te), compliant_te,
        '/storage/decoders/sleep_from_te'+str(compliant_te)+'.pkl')
    