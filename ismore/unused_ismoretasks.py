

###############################################################################
############################# Tasks ToDo ######################################
###############################################################################
#in ismoretasks.py
class CalibrationMovements(IsMoreBase):
    '''TODO.'''

    sequence_generators = []

    status = {
        'move': {'stop':  None},
    }
    
    state = 'move'  # initial state

    def _cycle(self):
        self.plant_pos[:] = self.plant.get_pos()
        self.plant_vel[:] = self.plant.get_vel()

        self.update_plant_display()

        self.task_data['plant_pos']  = self.plant_pos.values
        self.task_data['plant_vel']  = self.plant_vel.values

        super(CalibrationMovements, self)._cycle()


###############################################################################
############ Tasks based on trajectory recording and playback #################
###############################################################################

#in ismoretasks.py -- to be updated if needed
class PlaybackTrajectories(NonInvasiveBase):
    '''
    Plays back trajectories stored in a file of reference trajectories.
    '''
    fps = 20
    
    status = {
        'wait': {
            'start_trial': 'go_to_start', 
            'stop': None},
        'go_to_start': {
            'at_starting_config': 'instruct_rest',
            'stop': None},              
        'instruct_rest': {
            'end_instruct': 'rest',
            'stop': None},            
        'rest': {
            'time_expired': 'instruct_trial_type',
            'stop': None},
        'instruct_trial_type': {
            'end_instruct': 'trial',
            'stop': None},
        'trial': {
            'end_trial': 'wait',
            'stop': None},
        }
    
    state = 'wait'  # initial state

    # settable parameters on web interface
    rest_interval    = traits.Tuple((2., 3.), desc='Min and max time to remain in the rest state.')
    min_advance_t    = traits.Float(0.005, desc='Minimum time to advance trajectory playback at each step of playback.') 
    search_win_t     = traits.Float(0.200, desc='Search within this time window from previous known point on trajectory.')
    aim_ahead_t      = traits.Float(0.150, desc='Aim this much time ahead of the current point on the trajectory.') 
    aim_ahead_t_psi  = traits.Float(0.200, desc='Specific to psi - aim this much time ahead of the current point on the trajectory.') 
    ref_trajectories = traits.DataFile(RefTrajectories, bmi3d_query_kwargs=dict(system__name='ref_trajectories'))
    give_feedback     = traits.Int((0,1), desc=' 0 if we do not give feedback, 1 if we give feedback')
    speed = traits.OptionsList(*speed_options, bmi3d_input_options= speed_options)
    debug = False
    feedback_times = traits.DataFile(object, desc='test', bmi3d_query_kwargs=dict(system__name='misc'))
    trial_end_states = ['trial']

    #rest and goal targets
    #targets_matrix = np.array([[400,200,0,0,0,0,1],[485,600,40,0,0,0,1]])/10. #in cm
    #targets_matrix[:,[2]] *= deg_to_rad #in rad
    # Set the "is_bmi_seed" flag so that the server knows that this is a task which can be used to create a BMI decoder
    is_bmi_seed = True
    

    def __init__(self, *args, **kwargs):
        super(PlaybackTrajectories, self).__init__(*args, **kwargs)

        self.add_dtype('trial_type',   np.str_, 40)
        self.add_dtype('ts',           'f8',    (1,))
        self.add_dtype('playback_vel', 'f8',    (len(self.vel_states),))
        self.add_dtype('playback_pos', 'f8',    (len(self.pos_states),))
        self.add_dtype('command_vel',  'f8',    (len(self.vel_states),))
        self.add_dtype('command_vel_raw',  'f8',    (len(self.vel_states),))
        self.add_dtype('command_pos',  'f8',    (len(self.pos_states),))
        self.add_dtype('aim_pos',      'f8',    (len(self.pos_states),))
        self.add_dtype('idx_aim',       int,    (1,))
        self.add_dtype('idx_aim_psi',   int,    (1,))
        self.add_dtype('idx_traj',      int,    (1,))
        #self.add_dtype('audio_feedback_start',      int,    (1,))
        self.add_dtype('subtrial_idx',      int,    (1,))        
        self.add_dtype('subtrial_idx_real',      int,    (1,)) 
        self.add_dtype('subtrial_idx_real_target',      int,    (1,))
        self.add_dtype('speed',   np.str_, 20)
        
        self.subtrial_idx = np.nan
        self.subgoal_reached = False
        self.subtrial_idx_real = np.nan
        self.subtrial_idx_real_target = np.nan
        self.subgoal_reached_real = False

    
        #if self.give_feedback:
        #    self.feedback_time = pickle.load(open(self.feedback_times, 'rb'))


        # self.give_feedback = False
        
        if self.give_feedback == 1:
            self.feedback_time = self.feedback_times
        #self.feedback_time = [0, 0, 0] #[3, 3]
        
        #area considered as acceptable as rest around the rest position (first position in trial) or the subtask end position
        self.target_rect = np.array([2., 2., np.deg2rad(20),  np.deg2rad(20), np.deg2rad(20), np.deg2rad(20), np.deg2rad(20)])# for targets during the trial time
        self.rest_rect = np.array([3., 3., np.deg2rad(10),  np.deg2rad(10), np.deg2rad(10), np.deg2rad(10), np.deg2rad(10)])# for rest position during 'go to start' time
        
        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)
        

        # for low-pass filtering command psi velocities
        self.psi_vel_buffer = RingBuffer(
            item_len=1,
            capacity=10,
        )

        self.plant.enable() 
        
        # 4th order butterworth filter for command_vel
        fs_synch = self.fps #Frequency at which the task is running
        nyq   = 0.5 * fs_synch
        cuttoff_freq  = 1.5 / nyq
        bpf_kin_coeffs = butter(4, cuttoff_freq, btype='low')

        self.command_lpfs = dict()
        for state in self.vel_states:
            self.command_lpfs[state] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1]) # low-pass filter to smooth out command velocities


    def _set_task_type(self):
        if self.trial_type in targetsB1:
            self.task_type = 'B1'
            self.n_subtasks = 2
        elif self.trial_type in targetsB2:
            self.task_type = 'B2'
            self.n_subtasks = 2
        elif self.trial_type in targetsF1_F2:
            self.task_type = 'F1'
            self.n_subtasks = 3
        
        # return task_type

    def pos_diff(self, x1, x2):
        '''
        Calculate x1 - x2, but with a specialized definition of "-"
        '''
        if self.plant_type == 'ArmAssist':
            sub_fns = [operator.sub, operator.sub, angle_subtract]
        elif self.plant_type == 'ReHand':
            sub_fns = [angle_subtract, angle_subtract, angle_subtract, angle_subtract]
        elif self.plant_type == 'IsMore':
            sub_fns = [operator.sub, operator.sub, angle_subtract, angle_subtract, angle_subtract, angle_subtract, angle_subtract]

        x1 = np.array(x1).ravel()
        x2 = np.array(x2).ravel()
        diff_ = []
        for sub_fn, i, j in izip(sub_fns, x1, x2):
            diff_.append(sub_fn(i, j))
        return np.array(diff_)

    def _set_subgoals(self):
        if self.task_type == 'B1':
            traj = self.ref_trajectories[self.trial_type]['traj']
            pos_traj = np.array(traj[self.pos_states])
            target_margin = np.array([7, 5])

            pos_traj_diff = pos_traj - pos_traj[0]
            max_xy_displ_idx = np.argmax(map(np.linalg.norm, pos_traj_diff[:,0:2]))

            # distal_goal = pos_traj[max_xy_displ_idx]
            # proximal_goal = pos_traj[len(pos_traj)-1]0

            # self.subgoals = [distal_goal, proximal_goal]
            # Find the first index in which the exo is within the rest area
            for kk in range(max_xy_displ_idx, len(pos_traj)-1):
                if np.all(np.abs(pos_traj[kk,0:2]-pos_traj[len(pos_traj)-1,0:2]) < target_margin):
                    target_goal_rest_idx = kk #+ 10 #Add 10 to compute the time when the exo is stopped in the target instead of the very first point in which the exo reaches the target area
                    break

            # Both in B1 and B2 the first target will always be reached with this algorithm
            if 'target_goal_rest_idx' in locals(): 
                self.subgoal_inds = [max_xy_displ_idx, target_goal_rest_idx]
            
            else: 
                self.subgoal_inds = [max_xy_displ_idx, len(pos_traj)-1]

            self.subgoals = [pos_traj[idx] for idx in self.subgoal_inds]
        
        elif self.task_type == 'B2':

            traj = self.ref_trajectories[self.trial_type]['traj']
            pos_traj = np.array(traj[self.pos_states])
            target_margin = np.deg2rad(10)

            if self.trial_type == 'up':
                grasp_goal_idx = np.argmin(traj['rh_pprono'].ravel())
                # Find the first index in which the exo is within the rest area
                for kk in range(grasp_goal_idx, len(pos_traj)-1):
                    if np.all(np.abs(pos_traj[kk,6]-pos_traj[len(pos_traj)-1,6]) < target_margin):
                        target_goal_rest_idx = kk #+ 10 #Add 10 to compute the time when the exo is stopped in the target instead of the very first point in which the exo reaches the target area
                        break
            elif self.trial_type == 'down':
                grasp_goal_idx = np.argmax(traj['rh_pprono'].ravel())
                # Find the first index in which the exo is within the rest area
                for kk in range(grasp_goal_idx, len(pos_traj)-1):
                    if np.all(np.abs(pos_traj[kk,6]-pos_traj[len(pos_traj)-1,6]) < target_margin):
                        target_goal_rest_idx = kk #+ 10 #Add 10 to compute the time when the exo is stopped in the target instead of the very first point in which the exo reaches the target area
                        break
            elif self.trial_type == 'point':
                grasp_goal_idx = np.argmin(traj['rh_pindex'].ravel())
                # Find the first index in which the exo is within the rest area
                for kk in range(grasp_goal_idx, len(pos_traj)-1):
                    if np.all(np.abs(pos_traj[kk,4]-pos_traj[len(pos_traj)-1,4]) < target_margin):
                        target_goal_rest_idx = kk #+ 10 #Add 10 to compute the time when the exo is stopped in the target instead of the very first point in which the exo reaches the target area
                        break
            elif self.trial_type == 'pinch':
                grasp_goal_idx = np.argmax(traj['rh_pindex'].ravel())
                # Find the first index in which the exo is within the rest area
                for kk in range(grasp_goal_idx, len(pos_traj)-1):
                    if np.all(np.abs(pos_traj[kk,4]-pos_traj[len(pos_traj)-1,4]) < target_margin):
                        target_goal_rest_idx = kk #+ 10 #Add 10 to compute the time when the exo is stopped in the target instead of the very first point in which the exo reaches the target area
                        break
            elif self.trial_type == 'grasp':
                grasp_goal_idx = np.argmin(traj['rh_pindex'].ravel())
                # Find the first index in which the exo is within the rest area
                for kk in range(grasp_goal_idx, len(pos_traj)-1):
                    if np.all(np.abs(pos_traj[kk,4]-pos_traj[len(pos_traj)-1,4]) < target_margin):
                        target_goal_rest_idx = kk #+ 10 #Add 10 to compute the time when the exo is stopped in the target instead of the very first point in which the exo reaches the target area
                        break

            # Both in B1 and B2 the first target will always be reached with this algorithm
            if 'target_goal_rest_idx' in locals(): 
                self.subgoal_inds = [grasp_goal_idx, target_goal_rest_idx]
            
            else: 
                self.subgoal_inds = [grasp_goal_idx, len(pos_traj)-1]

            #self.subgoal_inds = [grasp_goal_idx, target_goal_rest_idx]
            self.subgoals = [pos_traj[idx] for idx in self.subgoal_inds]

        elif self.task_type == 'F1':
            # fit the largest triangle possible to the trajectory
            traj = self.ref_trajectories[self.trial_type]['traj']
            pos_traj = np.array(traj[self.pos_states])

            
            # Method 1
            # pos_traj_diff = pos_traj - pos_traj[0]
            # diff = map(np.linalg.norm, pos_traj_diff[:,0:2])
            # local_minima = np.zeros(len(pos_traj_diff))
            # T = len(pos_traj_diff)
            # support = 200
            # for k in range(support, T-support):
            #     local_minima[k] = np.all(diff[k-support:k+support] <= diff[k]) 

            # local_minima[diff < 5] = 0 # exclude anything closer than 5 cm

            # local_minima_inds, = np.nonzero(local_minima)
            # self.subgoal_inds = np.hstack([local_minima_inds, len(pos_traj)-2])

            # print 'subgoal_inds'
            # print self.subgoal_inds 

            # self.subgoals = [pos_traj[idx] for idx in self.subgoal_inds]

            # self.subgoals = [distal_goal, proximal_goal]
            # self.subgoal_inds = [max_xy_displ_idx, len(pos_traj)-2]

            # Method 2: Define target area (based on x and y coordinates) for each target type
            target_goal_Red = np.array([28, 35])
            target_goal_Blue = np.array([54, 33])
            target_goal_Green = np.array([39, 45])
            target_goal_Brown = np.array([52, 46])
            target_margin = np.array([7, 5]) #np.array([2., 2., np.deg2rad(20),  np.deg2rad(10), np.deg2rad(10), np.deg2rad(10), np.deg2rad(10)])

            if self.trial_type == 'red to brown' or self.trial_type == 'blue to brown' or self.trial_type == 'green to brown':
                target_goal_pos_2 = target_goal_Brown
            if self.trial_type == 'red to green' or self.trial_type == 'blue to green' or self.trial_type == 'brown to green':
                target_goal_pos_2 = target_goal_Green
            if self.trial_type == 'red to blue' or self.trial_type == 'brown to blue' or self.trial_type == 'green to blue':
                target_goal_pos_2 = target_goal_Blue
            if self.trial_type == 'brown to red' or self.trial_type == 'blue to red' or self.trial_type == 'green to red':
                target_goal_pos_2 = target_goal_Red
            if self.trial_type == 'red to brown' or self.trial_type == 'red to blue' or self.trial_type == 'red to green':
                target_goal_pos_1 = target_goal_Red
            if self.trial_type == 'blue to brown' or self.trial_type == 'blue to red' or self.trial_type == 'blue to green':
                target_goal_pos_1 = target_goal_Blue
            if self.trial_type == 'green to brown' or self.trial_type == 'green to blue' or self.trial_type == 'green to red':
                target_goal_pos_1 = target_goal_Green
            if self.trial_type == 'brown to red' or self.trial_type == 'brown to blue' or self.trial_type == 'brown to green':
                target_goal_pos_1 = target_goal_Brown


            # Find the first index in which the exo is within the target1 area
            for kk in range(0, len(pos_traj)):
                if np.all(np.abs(pos_traj[kk,0:2]-target_goal_pos_1) < target_margin):
                    target_goal_1_idx = kk #+10 #Add 10 to compute the time when the exo is stopped in the target instead of the very first point in which the exo reaches the target area
                    break
            #np.where(np.abs(pos_traj[:,0:2]-target_goal_pos_1) <= target_margin)
            if 'target_goal_1_idx' not in locals(): 
                target_goal_1_idx = 0
                
            # Find the first index in which the exo is within the target2 area
            for kk in range (target_goal_1_idx + 30,len(pos_traj)): # Find the moment when the second target was reached imposing the condition that it should happen 30 time points after target1 at least
                if np.all(np.abs(pos_traj[kk,0:2]-target_goal_pos_2) < target_margin):
                    target_goal_2_idx = kk #+ 10 #Add 10 to compute the time when the exo is stopped in the target instead of the very first point in which the exo reaches the target area
                    break
         
            # Find the first index in which the exo is within the rest area
            if 'target_goal_2_idx' not in locals(): 
                target_goal_2_idx = target_goal_1_idx
                

            for kk in range(target_goal_2_idx, len(pos_traj)):
                if np.all(np.abs(pos_traj[kk,0:2]-pos_traj[len(pos_traj)-1,0:2]) < target_margin):
                    target_goal_rest_idx = kk #+ 10 #Add 10 to compute the time when the exo is stopped in the target instead of the very first point in which the exo reaches the target area
                    break

            if 'target_goal_rest_idx' not in locals() or target_goal_rest_idx == target_goal_2_idx: 
                target_goal_rest_idx = len(pos_traj)-1
                
        
            self.subgoal_inds = np.array([target_goal_1_idx, target_goal_2_idx, target_goal_rest_idx])
        
            self.subgoals = [pos_traj[idx] for idx in self.subgoal_inds]
            
            
            #self.subgoal_times = traj[self.subgoal_inds]['ts'].ravel() - traj[0]['ts']
        else:
            #raise ValueError("Unrecognized task type:%s " % task_type)
            pass

    def _while_trial(self):
        if self.give_feedback:
            # determine if subgoals have been accomplished
            goal_pos_state_real = self.subgoals[self.subtrial_idx_real_target]
            pos_diff_real = self.pos_diff(self.plant.get_pos(), goal_pos_state_real)
            
            if np.all(np.abs(pos_diff_real) < np.abs(self.target_rect[:len(self.pos_states)])) and self.subgoal_reached_real == False: #self.plant.get_pos() == goal_pos_state
                # if subtrial has been accomplished (close enough to goal), move on to the next subtrial
                
                self.subtrial_start_time_real = self.get_time()
                self.subtrial_idx_real += 1
                self.subtrial_idx_real_target += 1    
                self.subgoal_reached_real = True

            self.subgoal_reached_real = False
            self.subtrial_idx_real_target = min(self.subtrial_idx_real, self.n_subtasks-1)
            self.subtrial_idx_real = min(self.subtrial_idx_real, self.n_subtasks)

            goal_pos_state = self.subgoals[self.subtrial_idx]
            pos_diff = self.pos_diff(self.plant.get_pos(), goal_pos_state)

            if np.all(np.abs(pos_diff) < np.abs(self.target_rect[:len(self.pos_states)])) and self.subgoal_reached == False and self.feedback_given[self.subtrial_idx] == True: #self.idx_traj > self.subgoal_inds[self.subtrial_idx] and self.feedback_given[self.subtrial_idx] == True:
                self.subgoal_reached = True
                self.subtrial_start_time = self.get_time()
                self.subtrial_idx += 1
                            
            self.subgoal_reached = False
            self.subtrial_idx = min(self.subtrial_idx, self.n_subtasks-1)

            if self.give_feedback == 1:
                fb_time = self.feedback_time[self.trial_type][self.subtrial_idx]
                #self.task_data['audio_feedback_start'] = 0
                if (self.get_time() - self.subtrial_start_time) > fb_time and not self.feedback_given[self.subtrial_idx]:
                    self.feedback_given[self.subtrial_idx] = True
                    #self.task_data['audio_feedback_start'] = 1
                    self._play_sound(os.path.join(self.sounds_dir, 'beep.wav'))

    def init(self):
        kwargs = {
            'call_rate': self.fps,
            'xy_cutoff': 2.,
        }
        # self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT[self.plant_type](**kwargs)
        self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT[self.plant_type](speed = self.speed,**kwargs)

        #self.assister = ismore_bmi_lib.LFC_ASSISTER_CLS_DICT[self.plant_type](**kwargs)

        super(PlaybackTrajectories, self).init()

    def compute_playback_vel(self):
        '''Docstring.'''

        playback_vel = pd.Series(0.0, self.vel_states)
        aim_pos      = pd.Series(0.0, self.pos_states)
        traj = self.ref_trajectories[self.trial_type]['traj']
        self.plant_pos[:] = self.plant.get_pos()


        # number of points in the reference trajectory for the current trial type
        len_traj = traj.shape[0]

        # do this simply to avoid having to write "self." everywhere
        idx_traj = self.idx_traj
        states = self.states
        dist_fn = self.dist_fn

        # index into the current trajectory playback
        # search locally in range [start_ts, end_ts)
        # depending on the type of trial, determine where we are along the trajectory by
        # finding the idx of the point in the reference trajectory that is closest to the
        # current state of plant in either xy euclidean distance or angular l1 distance
        start_ts = traj['ts'][idx_traj] + self.min_advance_t
        end_ts   = start_ts + self.search_win_t
        search_idxs = [idx for (idx, ts) in enumerate(traj['ts']) if start_ts <= ts < end_ts]
        min_dist = np.inf
        for idx in search_idxs:
            d = dist_fn(self.plant_pos[states], traj[states].ix[idx]) 
            if idx == search_idxs[0] or d < min_dist:
                min_dist = d
                idx_traj = idx

        # find the idx of the point in the reference trajectory to aim towards
        idx_aim = idx_traj
        idx_aim_psi = idx_traj

        while idx_aim < len_traj - 1:
            if (traj['ts'][idx_aim] - traj['ts'][idx_traj]) < self.aim_ahead_t:
                idx_aim += 1
                idx_aim_psi = idx_aim 
            else:
                break

        #if self.state is not 'trial':
        #    print 'state'
        #    print self.state
        #    self.idx_aim = 0
        #    self.idx_traj = 0

        #print 'inside compute_playback_vel'
    

        if self.plant_type in ['ArmAssist', 'IsMore']:
            if idx_traj == len_traj - 1:
                playback_vel[aa_vel_states] = np.zeros(3)
                self.finished_traj_playback = True

                # Fill in the aim pos for any after-analysis
                aim_pos['aa_px'] = traj['aa_px'][idx_traj]
                aim_pos['aa_py'] = traj['aa_py'][idx_traj]
                aim_pos['aa_ppsi'] = traj['aa_ppsi'][idx_traj]
            else:
                # ArmAssist xy
                aim_pos[aa_xy_states] = traj[aa_xy_states].ix[idx_aim]
                xy_dir = norm_vec(traj[aa_xy_states].ix[idx_aim] - self.plant_pos[aa_xy_states])
                #xy_dir = norm_vec(traj[aa_xy_states].ix[idx_aim] - traj[aa_xy_states].ix[idx_traj])
                

                # since armassist does not provide velocity feedback, 
                # need to calculate the xy speed for the reference trajectory
                t1   = traj['ts'][idx_traj]
                t2   = traj['ts'][idx_aim]
                pos1 = traj[aa_xy_states].ix[idx_traj]
                pos2 = traj[aa_xy_states].ix[idx_aim]
                xy_speed = np.linalg.norm((pos2 - pos1) / (t2 - t1))  # cm/s


                # apply xy-distance-dependent min and max xy speed
                xy_dist = dist(traj[aa_xy_states].ix[idx_aim], self.plant_pos[aa_xy_states])
                max_xy_speed_1 = 15                          # cm/s
                max_xy_speed_2 = xy_dist / self.aim_ahead_t  # cm/s
                max_xy_speed   = min(max_xy_speed_1, max_xy_speed_2)
                min_xy_speed   = 0  #min(0.25 * max_xy_speed_2, max_xy_speed)
                xy_speed       = bound(xy_speed, min_xy_speed, max_xy_speed) 
                #xy_speed       =  max_xy_speed 


                # ArmAssist psi (orientation) -- handle separately from xy
                aim_pos['aa_ppsi'] = traj['aa_ppsi'][idx_aim_psi]
                psi_dir = np.sign(angle_subtract(traj['aa_ppsi'][idx_aim_psi], self.plant_pos['aa_ppsi']))
                #psi_dir = np.sign(angle_subtract(traj['aa_ppsi'][idx_aim_psi], traj['aa_ppsi'][idx_traj]))
                
                # since armassist does not provide velocity feedback, 
                # need to calculate the psi speed for the reference trajectory
                t1   = traj['ts'][idx_traj]
                t2   = traj['ts'][idx_aim_psi]
                psi1 = traj['aa_ppsi'][idx_traj]
                psi2 = traj['aa_ppsi'][idx_aim_psi]
                psi_speed = np.abs(angle_subtract(psi2, psi1) / (t2 - t1))  # rad/s

                # apply psi-distance-dependent min and max psi speed
                psi_dist = abs(angle_subtract(traj['aa_ppsi'][idx_aim_psi], self.plant_pos['aa_ppsi']))
                max_psi_speed_1 = 30*deg_to_rad                     # rad/s
                max_psi_speed_2 = psi_dist / self.aim_ahead_t_psi  # rad/s
                max_psi_speed   = min(max_psi_speed_1, max_psi_speed_2)
                min_psi_speed   = 0

             
                psi_speed       = bound(psi_speed, min_psi_speed, max_psi_speed)
                #psi_speed       = max_psi_speed

                playback_vel[['aa_vx', 'aa_vy']] = xy_speed  * xy_dir
                playback_vel['aa_vpsi']          = (psi_speed * psi_dir) #/2

                #self.x_vel_buffer.add(playback_vel['aa_vx'])
                #self.y_vel_buffer.add(playback_vel['aa_vy'])
                
                #playback_vel[['aa_vx']] = np.mean(self.x_vel_buffer.get_all(), axis=1)
                #playback_vel[['aa_vy']] = np.mean(self.y_vel_buffer.get_all(), axis=1)
                
                # Moving average filter for the output psi angular velocity
                
                self.psi_vel_buffer.add(playback_vel['aa_vpsi'])
                std_psi_vel = np.std(self.psi_vel_buffer.get_all(), axis=1)
                mean_psi_vel = np.mean(self.psi_vel_buffer.get_all(), axis=1)

                psi_vel_points = np.array(self.psi_vel_buffer.get_all())
                z1 = psi_vel_points < (mean_psi_vel + 2*std_psi_vel)
                z2 = psi_vel_points[z1] > (mean_psi_vel - 2*std_psi_vel ) 
                psi_vel_points_ok = psi_vel_points[z1]                
                psi_vel_lpf = np.mean(psi_vel_points_ok[z2])

                

                if math.isnan(psi_vel_lpf) == False:
                    playback_vel['aa_vpsi'] = psi_vel_lpf
                #else:
                #    playback_vel['aa_vpsi'] = (psi_speed * psi_dir)/2


            if (self.device_to_use == 'ArmAssist' and self.plant_type == 'IsMore') :
               playback_vel[rh_vel_states] = 0

        if self.plant_type in ['ReHand', 'IsMore']:
            if idx_traj == len_traj - 1:  # reached the end of the trajectory
                playback_vel[rh_vel_states] = np.zeros(4)
                self.finished_traj_playback = True
            else:
                aim_pos[rh_pos_states] = traj[rh_pos_states].ix[idx_aim]

                ang_dir = np.sign(angle_subtract_vec(traj[rh_pos_states].ix[idx_aim], self.plant_pos[rh_pos_states]))

                vel = traj[rh_vel_states].ix[idx_traj]
                ang_speed = np.abs(vel)
                

                # apply angular-distance-dependent min and max angular speeds
                for i, state in enumerate(rh_pos_states):
                    ang_dist = abs(angle_subtract(traj[state][idx_aim], self.plant_pos[state]))
                    max_ang_speed_1 = 40*deg_to_rad                # rad/s    
                    max_ang_speed_2 = ang_dist / self.aim_ahead_t  # rad/s
                    max_ang_speed   = min(max_ang_speed_1, max_ang_speed_2)
                    min_ang_speed   = 0
                    ang_speed[i]    = bound(ang_speed[i], min_ang_speed, max_ang_speed) 
                    #ang_speed[i]   = max_ang_speed

                    playback_vel[rh_vel_states] = ang_speed * ang_dir


            # if recorded ReHand trajectory is being used as the reference when playing back,
            # then don't move the ArmAssist at all
            if (self.device_to_use == 'ReHand' and self.plant_type == 'IsMore') :
               playback_vel[aa_vel_states] = 0
            #if recorded trajectory is B1 and plant is IsMore, do not move ReHand. need to put it again here again so that RH vel are set to 0. #nerea
            #elif (self.device_to_use == 'ArmAssist' and self.plant_type == 'IsMore') :
            #   playback_vel[rh_vel_states] = 0


        # if recorded ReHand trajectory is being used as the reference when playing back,
        # then don't move the ArmAssist at all
        if (self.device_to_use == 'ReHand' and self.plant_type == 'IsMore') :
            playback_vel[aa_vel_states] = 0
            #if recorded trajectory is B1 and plant is IsMore, do not move ReHand. need to put it again here again so that RH vel are set to ` #nerea
            #elif (self.device_to_use == 'ArmAssist' and self.plant_type == 'IsMore') :
            #   playback_vel[rh_vel_states] = 0
        
        self.idx_traj = idx_traj
        self.playback_vel = playback_vel
        
        self.task_data['idx_aim'] = idx_aim
        self.task_data['playback_vel'] = playback_vel.values
        self.task_data['aim_pos']      = aim_pos.values
        self.task_data['idx_aim_psi']  = idx_aim_psi
        self.task_data['idx_traj']     = idx_traj 


    def move_plant(self):
        command_vel  = pd.Series(0.0, self.vel_states)
        command_vel_raw  = pd.Series(0.0, self.vel_states)

        command_vel = self.playback_vel

        command_vel_raw[:] = command_vel[:]

        traj = self.ref_trajectories[self.trial_type]['traj']
        #print traj
        
        # #Apply low-pass filter to command velocities
        for state in self.vel_states:
            command_vel[state] = self.command_lpfs[state](command_vel[state])
            if np.isnan(command_vel[state]):
                command_vel[state] = 0

        
        # iterate over actual State objects, not state names
        #for state in self.ssm.states:
        #    if state.name in self.vel_states:
        #        command_vel[state.name] = bound(command_vel[state.name], state.min_val, state.max_val)

        # Command zero velocity if the task is in a non-moving state
        if self.state in ['wait','rest', 'instruct_rest', 'instruct_trial_type']: 
            command_vel[:] = 0
            self.idx_aim = 0
            self.idx_traj = 0

        #elif self.state in ['trial', 'go_to_start']: 
        elif self.state == 'go_to_start':
            current_pos = self.plant_pos[:].ravel()
            #current_vel = self.plant_vel[:].ravel()

            current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
            
            if self.state == 'go_to_start':
                #print 'traj[self.pos_states].ix[0]', traj[self.pos_states].ix[0], traj[self.pos_states].ix[1], traj[self.pos_states].ix[2]
                #print 'shape of old target_state: ', np.hstack([traj[self.pos_states].ix[0], np.zeros_like(current_pos), 1]).reshape(-1, 1).shape
                #target_state = np.array([self.targets_matrix[0,:]]).T #nerea
                #target_state = np.hstack([self.targets_matrix['rest'][0:3], np.zeros_like(current_pos),1]).reshape(-1,1)


                #print 'shape of new target state: ', target_state.shape
                target_state = np.hstack([traj[self.pos_states].ix[0], np.zeros_like(current_pos), 1]).reshape(-1, 1)
                    

            #elif self.state == 'trial':
                #target_state = np.array([self.targets_matrix[1,:]]).T
            #    target_state = np.hstack([self.targets_matrix[self.trial_type][0:3], np.zeros_like(current_pos),1 ]).reshape(-1,1)

            #target_state = np.hstack([traj[self.pos_states].ix[0], np.zeros_like(current_pos), 1]).reshape(-1, 1)
            #nerea

            #print 'target_state'
            #print target_state
            #print 'current_state'
            #print current_state

            assist_output = self.assister(current_state, target_state, 1.)
            Bu = np.array(assist_output["x_assist"]).ravel()

            #Bu = np.array(assist_output['Bu']).ravel()
            command_vel[:] = Bu[len(current_pos):len(current_pos)*2]

            # pos_diff = self.pos_diff(traj[self.pos_states].ix[0], self.plant_pos[self.pos_states])
            # signs = np.sign(pos_diff)
            # max_vel      = pd.Series(0.0, ismore_vel_states)
            # max_vel[:] = np.array([2., 2., np.deg2rad(0.5), np.deg2rad(3), np.deg2rad(3), np.deg2rad(3), np.deg2rad(3)])
            # command_vel[:] = max_vel[self.vel_states].ravel() * signs
            
            self.idx_aim = 0
            self.idx_traj = 0

        #print "vel"
        #print self.plant.get_vel()
        #print 'command_vel'
        #print command_vel.values
  
        self.plant.send_vel(command_vel.values) #send velocity command to EXO
        
        self.task_data['command_vel']  = command_vel.values
        self.task_data['command_vel_raw']  = command_vel_raw.values

    def compute_playback_pos(self):
        '''Docstring.'''
        #playback_vel = pd.Series(0.0, self.vel_states)
        playback_pos = pd.Series(0.0, self.pos_states)
        
        aim_pos      = pd.Series(0.0, self.pos_states)

        traj = self.ref_trajectories[self.trial_type]['traj']

        self.plant_pos[:] = self.plant.get_pos()


        # number of points in the reference trajectory for the current trial type
        len_traj = traj.shape[0]

        # do this simply to avoid having to write "self." everywhere
        idx_traj = self.idx_traj
        states = self.states
        dist_fn = self.dist_fn

        # index into the current trajectory playback
        # search locally in range [start_ts, end_ts)
        # depending on the type of trial, determine where we are along the trajectory by
        # finding the idx of the point in the reference trajectory that is closest to the
        # current state of plant in either xy euclidean distance or angular l1 distance
        start_ts = traj['ts'][idx_traj] + self.min_advance_t
        end_ts   = start_ts + self.search_win_t
        search_idxs = [idx for (idx, ts) in enumerate(traj['ts']) if start_ts <= ts < end_ts]
        min_dist = np.inf
        for idx in search_idxs:
            d = dist_fn(self.plant_pos[states], traj[states].ix[idx]) 
            if idx == search_idxs[0] or d < min_dist:
                min_dist = d
                idx_traj = idx

        # find the idx of the point in the reference trajectory to aim towards
        idx_aim = idx_traj
        idx_aim_psi = idx_traj

        while idx_aim < len_traj - 1:

            if (traj['ts'][idx_aim] - traj['ts'][idx_traj]) < self.aim_ahead_t:
                idx_aim += 1
                idx_aim_psi = idx_aim 
            else:
                break

        if self.plant_type in ['ArmAssist', 'IsMore']:
            if idx_traj == len_traj - 1:
                #playback_vel[aa_vel_states] = np.zeros(3)
                self.finished_traj_playback = True

                # Fill in the aim pos for any after-analysis
                aim_pos['aa_px'] = traj['aa_px'][idx_traj]
                aim_pos['aa_py'] = traj['aa_py'][idx_traj]
                aim_pos['aa_ppsi'] = traj['aa_ppsi'][idx_traj]
                playback_pos = aim_pos.copy()
                #playback_pos = aim_pos[:]
            else:
                # ArmAssist xy
                aim_pos[aa_xy_states] = traj[aa_xy_states].ix[idx_aim]
  
                # ArmAssist psi (orientation) -- handle separately from xy
                aim_pos['aa_ppsi'] = traj['aa_ppsi'][idx_aim_psi]
              
                playback_pos[aa_xy_states] = aim_pos[aa_xy_states]
                playback_pos['aa_ppsi']    = aim_pos['aa_ppsi'] 

                
        
        self.task_data['idx_aim'] = idx_aim
        self.task_data['aim_pos'] = aim_pos
        self.idx_traj = idx_traj

        self.playback_pos = playback_pos


        self.task_data['playback_pos'] = playback_pos.values
        self.task_data['aim_pos']      = aim_pos.values
        self.task_data['idx_aim']      = idx_aim
        self.task_data['idx_aim_psi']  = idx_aim_psi
        self.task_data['idx_traj']     = idx_traj 

    def move_plant_pos_control(self):
        command_pos  = pd.Series(0.0, self.pos_states)
        command_vel  = pd.Series(0.0, self.vel_states)
        
        command_pos = self.playback_pos

        traj = self.ref_trajectories[self.trial_type]['traj']

       
        # Command zero velocity if the task is in a non-moving state
        if self.state in ['wait','rest', 'instruct_rest', 'instruct_trial_type']: 
            command_vel[:] = 0
            self.plant.send_vel(command_vel.values) #send velocity command to EXO
            
        elif self.state == 'go_to_start':
            current_pos = self.plant_pos[:].ravel()
            current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
            target_state = np.hstack([traj[self.pos_states].ix[0], np.zeros_like(current_pos), 1]).reshape(-1, 1)

            assist_output = self.assister(current_state, target_state, 1)
            Bu = np.array(assist_output["x_assist"]).ravel()
            command_vel[:] = Bu[len(current_pos):len(current_pos)*2]

            self.plant.send_vel(command_vel.values) #send velocity command to EXO
            
        else:
            self.plant.send_pos(command_pos.values, 1/self.fps) #send velocity command to EXO
         
            idx_aim = 0
            idx_traj = 0

           
        self.task_data['command_pos']  = command_pos.values
        self.task_data['command_vel']  = command_vel.values
      


    def _cycle(self):
        '''Runs self.fps times per second.'''

        # get latest position/velocity information before calling move_plant()
        self.plant_pos_raw[:] = self.plant.get_pos_raw()
        self.plant_pos[:] = self.plant.get_pos() 
        
        self.plant_vel_raw[:] = self.plant.get_vel_raw()
        self.plant_vel[:] = self.plant.get_vel()

        #if self.state in ['trial','go_to_start']:
        # velocity control
        self.compute_playback_vel()
        self.move_plant()

        # position control
        # self.compute_playback_pos()
        # self.move_plant_pos_control()

        self.update_plant_display()

        # print self.subtrial_idx
        if not self.state == 'trial':
            #self.task_data['audio_feedback_start'] = 0
            self.task_data['subtrial_idx'] = -1
            self.task_data['subtrial_idx_real'] = -1
            self.task_data['subtrial_idx_real_target'] = -1
        else:
            self.task_data['subtrial_idx'] = self.subtrial_idx
            self.task_data['subtrial_idx_real'] = self.subtrial_idx_real
            self.task_data['subtrial_idx_real_target'] = self.subtrial_idx_real
        
        self.task_data['plant_pos']  = self.plant_pos_raw.values
        self.task_data['plant_pos_filt']  = self.plant_pos.values 
        self.task_data['plant_vel']  = self.plant_vel_raw.values
        self.task_data['plant_vel_filt']  = self.plant_vel.values

        self.task_data['trial_type'] = self.trial_type
        self.task_data['ts']         = time.time()
        self.task_data['speed'] = self.speed
      
        super(PlaybackTrajectories, self)._cycle()

    #### STATE AND TEST FUNCTIONS ####
    def _start_wait(self):
        # determine the random length of time to stay in the rest state
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time

        self.finished_traj_playback = False
        self.idx_traj = 0

        self.subtrial_idx = 0
        self.subtrial_idx_real = 0
        self.subtrial_idx_real_target = 0       
        self.feedback_given = [False, False, False]
        super(PlaybackTrajectories, self)._start_wait()

    def _parse_next_trial(self):
        self.trial_type = self.next_trial

        device = self.device_to_use
        if device == 'ArmAssist':
            self.dist_fn = dist
            self.states = aa_xy_states
        elif device == 'ReHand':
            self.dist_fn = l1_ang_dist
            self.states = rh_pos_states

    def _test_end_instruct(self, *args, **kwargs):
        return not pygame.mixer.music.get_busy()

    def _start_instruct_rest(self):
        self._play_sound(os.path.join(self.sounds_dir, 'rest.wav'))
        print 'rest'

    def _start_instruct_trial_type(self):
        sound_fname = os.path.join(self.sounds_dir, self.trial_type + '.wav')
        self._play_sound(sound_fname)

    def _start_trial(self):
        print self.trial_type
        #self.plant.set_pos_control() #to set it to position control during the trial state

        self._set_task_type()
        if self.give_feedback:
            self._set_subgoals()
        self.subtrial_start_time = self.get_time()
        self.subtrial_start_time_real = self.get_time()

    def _test_end_trial(self, ts):
        #end_targ = self.targets_matrix[1,:]
        #end_targ = self.targets_matrix[self.trial_type]
        #diff_to_end = np.abs(self.plant.get_pos() - end_targ[0:3])
        #return np.all(diff_to_end < self.rest_rect[:len(self.pos_states)])
        return self.finished_traj_playback

    def _test_at_starting_config(self, *args, **kwargs):
        traj = self.ref_trajectories[self.trial_type]['traj']
        #start_targ = self.targets_matrix[0,:]
        #start_targ = self.targets_matrix['rest']
        diff_to_start = np.abs(self.plant.get_pos() - traj[self.pos_states].ix[0].ravel())
        #diff_to_start = np.abs(self.plant.get_pos() - start_targ[0:3]) #nerea
        #print diff_to_start

        return np.all(diff_to_start < self.rest_rect[:len(self.pos_states)])

    def _end_trial(self):
        #self.plant.set_global_control() #change
        pass


#in exg_tasks.py
class EMGTrajectoryDecoding(PlaybackTrajectories):


    # settable parameters on web interface    
    gamma             = traits.Float(0.5,   desc='Gamma value for incorporating EMG decoded velocity.')
    emg_playback_file = traits.String('',   desc='Full path to recorded EMG data file.')
    emg_decoder       = traits.InstanceFromDB(LinearEMGDecoder, bmi3d_db_model="Decoder", bmi3d_query_kwargs=dict(name__startswith='emg_decoder'))
    use_emg_decoder   = traits.Int((0,1), desc=' 0 if we do not give feedback, 1 if we give feedback')


    def __init__(self, *args, **kwargs):
        super(EMGTrajectoryDecoding, self).__init__(*args, **kwargs)
       

        #self.channels_filt = brainamp_channel_lists.emg14_filt

        # if EMG decoder file was specified, load EMG decoder and create feature extractor 
        # if len(self.emg_decoder_file) > 3:
        #     self.emg_decoder = pickle.load(open(self.emg_decoder_file, 'rb'))

        # print settings.BRAINAMP_CHANNELS
        # channels_filt = []
        # for k in range(len(settings.BRAINAMP_CHANNELS)):
        #     channels_filt.append(settings.BRAINAMP_CHANNELS[k] + "_filt")

        emg_extractor_cls    = self.emg_decoder.extractor_cls
        emg_extractor_kwargs = self.emg_decoder.extractor_kwargs

        #print [settings.BRAINAMP_CHANNELS[chan] + "_filt" for i, chan in enumerate(settings.BRAINAMP_CHANNELS)]
        # if self.brainamp_channels != channels_filt:
        #     print 'ERROR: The selected channels in the interface do not match those defined in settings to be streamed from the amplifier.'

        # create EMG extractor object (it's 'source' will be set later in the init method)
        if self.emg_decoder.plant_type != self.plant_type:
            print 'Chosen plant_type on the interface does not match the plant type used to train the decoder. Make sure you are selecting the right one'

        

        self.brainamp_channels = emg_extractor_kwargs['brainamp_channels'] 
        
        # extractor_kwargs['channels_filt'] = list()
        # for i in range(len(extractor_kwargs['channels'])):
        #     extractor_kwargs['channels_filt'] = [extractor_kwargs['channels'][i] + "_filt"]
        #     extractor_kwargs['channels_filt'].append(extractor_kwargs['channels_filt'])

        self.emg_playback = False
        
        #self.emg_extractor = emg_extractor_cls(source=None, channels = self.brainamp_channels, **emg_extractor_kwargs)

        self.emg_extractor = emg_extractor_cls(source=None, **emg_extractor_kwargs)

        self.add_dtype('emg_features',    'f8', (self.emg_extractor.n_features,))
        self.add_dtype('emg_features_Z',  'f8', (self.emg_extractor.n_features,))
        self.add_dtype('emg_vel',         'f8', (len(self.vel_states),))
        self.add_dtype('emg_vel_lpf',     'f8', (len(self.vel_states),))

        # for calculating/updating mean and std of EMG features online
        self.features_buffer = RingBuffer(
            item_len=self.emg_extractor.n_features,
            capacity=60*self.fps,  # 60 secs
        )

        # for low-pass filtering decoded EMG velocities
        self.emg_vel_buffer = RingBuffer(
            item_len=len(self.vel_states),
            capacity=10,
        )
        
        self.plant.enable() 

        # 4th order butterworth filter for command_vel
        fs_synch = self.fps #Frequency at which the task is running
        nyq   = 0.5 * fs_synch
        cuttoff_freq  = 1.5 / nyq
        bpf_kin_coeffs = butter(2, cuttoff_freq, btype='low')

        self.command_lpfs = dict()
        for state in self.vel_states:
            self.command_lpfs[state] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1]) # low-pass filter to smooth out command velocities

    def init(self):
        super(EMGTrajectoryDecoding, self).init()
        from riglib import source
        from ismore.brainamp import rda

        self.brainamp_source = source.MultiChanDataSource(rda.EMGData, name='brainamp', channels=self.brainamp_channels, brainamp_channels=self.brainamp_channels, send_data_to_sink_manager=True)
        self.emg_extractor.source = self.brainamp_source
        #self.emg_extractor.channels = self.brainamp_channels
        

    def move_plant(self):
        '''Docstring.'''

        #playback_vel = pd.Series(0.0, self.vel_states)
        command_vel  = pd.Series(0.0, self.vel_states)
        command_vel_raw  = pd.Series(0.0, self.vel_states)
        #aim_pos      = pd.Series(0.0, self.pos_states)
        emg_vel      = pd.Series(0.0, self.vel_states) #nerea
        
        # run EMG feature extractor and decoder

        #self.emg_extractor.source = self.brainamp_source

        emg_features = self.emg_extractor() # emg_features is of type 'dict'

        #emg_features[self.emg_extractor.feature_type] = emg_features[self.emg_extractor.feature_type][len(emg_features[self.emg_extractor.feature_type])/2:]
        self.features_buffer.add(emg_features[self.emg_extractor.feature_type])
        #print 'emg_features[self.emg_extractor.feature_type]', emg_features[self.emg_extractor.feature_type]
        if 1: #self.features_buffer.num_items() > 1 * self.fps:
                # if we have more than 1 second of recent EMG data, then
                #   calculate mean and std from this data
            recent_features = self.features_buffer.get_all()
            #print 'recent_features', recent_features
            features_mean = np.mean(recent_features, axis=1)
            features_std  = np.std(recent_features, axis=1)
        else:
                # else use mean and std from the EMG data that was used to 
                #   train the decoder
            features_mean = self.emg_decoder.features_mean
            features_std  = self.emg_decoder.features_std

        features_std[features_std == 0] = 1

        # z-score the EMG features
        emg_features_Z = (emg_features[self.emg_extractor.feature_type] - features_mean) / features_std 
        emg_vel = self.emg_decoder(emg_features_Z)

        self.emg_vel_buffer.add(emg_vel[self.vel_states])

            #print 'any zeros in std vector?:', any(features_std == 0.0)
        

        emg_vel_lpf = np.mean(self.emg_vel_buffer.get_all(), axis=1)

        self.task_data['emg_features']   = emg_features[self.emg_extractor.feature_type]
        self.task_data['emg_features_Z'] = emg_features_Z
        self.task_data['emg_vel']        = emg_vel
        self.task_data['emg_vel_lpf']    = emg_vel_lpf
                  

        # combine EMG decoded velocity and playback velocity into one velocity command
        norm_playback_vel = np.linalg.norm(self.playback_vel)
        epsilon = 1e-6
        if (norm_playback_vel < epsilon):
                # if norm of the playback velocity is 0 or close to 0,
                #   then just set command velocity to 0s
            command_vel[:] = 0.0

        else:

            #feedback 1
            term1 = self.gamma * emg_vel_lpf
            term2 = (1 - self.gamma) * self.playback_vel

                #feedback 2
                # term1 = self.gamma * ((np.dot(emg_vel_lpf, playback_vel) / (norm_playback_vel**2)) * playback_vel)
                # term2 = (1 - self.gamma) * playback_vel


                #term1 = self.gamma * self.emg_decoder.gamma_coeffs * ((np.dot(emg_vel_lpf, playback_vel) / (norm_playback_vel**2)) * playback_vel)
                #term2 = (1 - self.gamma * self.emg_decoder.gamma_coeffs) * playback_vel
                

            command_vel = term1 + term2


            if (device_to_use(self.trial_type) == 'ReHand' and self.plant_type == 'IsMore') :
                command_vel[aa_vel_states] = 0.0
        


        command_vel_raw[:] = command_vel[:]

        # # # # Apply low-pass filter to command velocities
        for state in self.vel_states:
        #     print command_vel[state]
            command_vel[state] = self.command_lpfs[state](command_vel[state])
            if np.isnan(command_vel[state]):
                command_vel[state] = 0

            #command_vel[state][np.isnan(command_vel[state][:])]
              
        # iterate over actual State objects, not state names
        # for state in self.ssm.states:
        #     if state.name in self.vel_states:
        #         command_vel[state.name] = bound(command_vel[state.name], state.min_val, state.max_val)

        # Command zero velocity if the task is in a non-moving state
        if self.state in ['wait','rest', 'instruct_rest', 'instruct_trial_type']: 
            command_vel[:] = 0

        
        self.plant.send_vel(command_vel.values)
        

        
        self.task_data['command_vel']  = command_vel.values
        self.task_data['command_vel_raw']  = command_vel_raw.values



###############################################################################
####### Derivative tasks to record specific types of trajectories #############
###############################################################################

# tasks for only EMG recording (they do NOT include a "ready" period)
#in ismoretasks.py
class RecordB1_EMG(RecordTrajectoriesBase):# inherit also from RecordBrainAmpData to record B1. double check issues due to double inheritance. First inherit from RecordBrainAmpData and then the other class
    '''Task class for recording trajectories for the center-out task to four different targets.'''
    pass

      
class RecordB2_EMG(RecordTrajectoriesBase):
    '''Task class for recording trajectories for the center-out task to four different targets.'''
    pass


class RecordF1_EMG(RecordTrajectoriesBase):
    '''Task class for recording trajectories for the center-out task to four different targets.'''
    pass
    

class RecordF2_EMG(RecordTrajectoriesBase):
    '''Task class for recording trajectories for the center-out task to four different targets.'''
    pass

class RecordFreeMov_EMG(RecordTrajectoriesBase):
    '''Task class for recording free movements.'''
    pass


###############################################################################
############################ Simulated tasks ##################################
###############################################################################

#in ismoretasks.py
class SimRecordB1(SimTime, SimHDF, RecordB1):
    pass

class SimRecordBrainAmpData(Autostart):

    def __init__(self, *args, **kwargs):
        self.channel_list_name = 'eeg32_raw_filt'
        super(SimRecordBrainAmpData, self).__init__(*args, **kwargs)

        import brainamp_channel_lists
        self.brainamp_channels = brainamp_channel_lists.eeg32_raw_filt

    def init(self):

        #from riglib import source
        #from ismore.brainamp import rda
        #self.brainamp_source = source.MultiChanDataSource(rda.SimEMGData, name='brainamp', channels=self.brainamp_channels, brainamp_channels=self.brainamp_channels, send_data_to_sink_manager=True)
        super(SimRecordBrainAmpData, self).init() 



#in exg_tasks.py
class SimEMGTrajectoryDecoding(EMGTrajectoryDecoding):
    '''
    Same as above, but only for debugging purposes, so uses an old HDF file for EMG data instead of live streaming data
    '''
    emg_playback_file = traits.String('', desc='file from which to replay old EMG data. Leave blank to stream EMG data from the brainamp system')



###############################################################################
######################### Testing tasks with EXO ##############################
###############################################################################


#in ismoretasks.py -- to update if needed
class SendVelProfile(NonInvasiveBase):
    '''
    Sends a previously recorded velocity profile with no closed-loop correction.
    '''
    fps = 20
    
    # status = {
    #     'wait': {
    #         'start_trial': 'go_to_start', 
    #         'stop': None},
    #     'go_to_start': {
    #         'at_starting_config': 'instruct_rest',
    #         'stop': None},              
    #     'instruct_rest': {
    #         'end_instruct': 'rest',
    #         'stop': None},            
    #     'rest': {
    #         'time_expired': 'instruct_trial_type',
    #         'stop': None},
    #     'instruct_trial_type': {
    #         'end_instruct': 'trial',
    #         'stop': None},
    #     'trial': {
    #         'end_trial': 'wait',
    #         'stop': None},
    #     }


    status = {
        'wait': {
            'start_trial': 'trial', 
            'stop': None},
        'trial': {
            'end_trial': 'wait',
            'stop': None},
        }
    
    state = 'wait'  # initial state

    # settable parameters on web interface
    rest_interval    = traits.Tuple((2., 3.), desc='Min and max time to remain in the rest state.')
    ref_trajectories = traits.DataFile(RefTrajectories, bmi3d_query_kwargs=dict(system__name='ref_trajectories'))
    speed = traits.OptionsList(*speed_options, bmi3d_input_options= speed_options)
    debug = False
    trial_end_states = ['trial']

    is_bmi_seed = True
    

    def __init__(self, *args, **kwargs):
        super(SendVelProfile, self).__init__(*args, **kwargs)

        self.add_dtype('trial_type',   np.str_, 40)
        self.add_dtype('ts',           'f8',    (1,))
        self.add_dtype('playback_vel', 'f8',    (len(self.vel_states),))
        self.add_dtype('playback_pos', 'f8',    (len(self.pos_states),))
        self.add_dtype('command_vel',  'f8',    (len(self.vel_states),))
        self.add_dtype('command_vel_raw',  'f8',    (len(self.vel_states),))
        self.add_dtype('idx_vel',       int,    (1,))
        self.add_dtype('command_traj', int, (len(self.pos_states)*2))
        self.add_dtype('playback_traj', 'f8',    (len(self.vel_states)*2,))
        self.add_dtype('speed',   np.str_, 20)

        #area considered as acceptable as rest around the rest position (first position in trial) or the subtask end position
        self.target_rect = np.array([2., 2., np.deg2rad(20),  np.deg2rad(1), np.deg2rad(1), np.deg2rad(1), np.deg2rad(1)])# for targets during the trial time
         
        self.sounds_general_dir = os.path.expandvars('$HOME/code/ismore/sounds')
        self.sounds_dir = os.path.join(self.sounds_general_dir, self.language)
        
        # for low-pass filtering command psi velocities
        self.psi_vel_buffer = RingBuffer(
            item_len=1,
            capacity=10,
        )

        #to test SetTrajectory algorithm
        self.plant = plants.UDP_PLANT_CLS_DICT['ArmAssist']()
        print "plant selected: ", self.plant
        
        
        #self.plant.enable()
        self.plant.set_trajectory_control()


       # 4th order butterworth filter for command_vel
        fs_synch = self.fps #Frequency at which the task is running
        nyq   = 0.5 * fs_synch
        cuttoff_freq  = 1.5 / nyq
        bpf_kin_coeffs = butter(4, cuttoff_freq, btype='low')

        self.command_lpfs = dict()
        for state in self.vel_states:
            self.command_lpfs[state] = Filter(bpf_kin_coeffs[0], bpf_kin_coeffs[1]) # low-pass filter to smooth out command velocities


    def init(self):
        kwargs = {
            'call_rate': self.fps,
            'xy_cutoff': 2.,
        }
        # self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT[self.plant_type](**kwargs)
        self.assister = ismore_bmi_lib.LFC_GO_TO_START_ASSISTER_CLS_DICT[self.plant_type](speed = self.speed,**kwargs)

        #self.assister = ismore_bmi_lib.LFC_ASSISTER_CLS_DICT[self.plant_type](**kwargs)

        super(SendVelProfile, self).init()

    def compute_playback_vel(self):
        '''Docstring.'''

        playback_vel = pd.Series(0.0, self.vel_states)
        playback_traj= pd.Series(0.0, np.concatenate([self.pos_states, self.vel_states]))

        task = self.ref_trajectories[self.trial_type]['task']
        pos_profile = task['plant_pos'] #only used for sending a trajectory
        vel_profile = task['plant_vel']
        self.plant_pos[:] = self.plant.get_pos()
        idx_vel = self.idx_vel

        # number of points in the reference trajectory for the current trial type
        len_vel_profile= vel_profile.shape[0]

        if idx_vel < len_vel_profile - 1:
            idx_vel += 1
            print "idx_vel: ", idx_vel
    

        print "vel_profile idx 1", vel_profile[idx_vel,0:3]

        if self.plant_type in ['ArmAssist', 'IsMore']:
            if idx_vel == len_vel_profile - 1:
                playback_vel[aa_vel_states] = np.zeros(3)
                playback_traj[np.concatenate([aa_pos_states, aa_vel_states])] = np.zeros(6)
                self.finished_vel_playback = True

            else:
                playback_vel[aa_vel_states] = vel_profile[idx_vel,0:3]  
                playback_traj[np.concatenate([aa_pos_states, aa_vel_states])] = np.concatenate([pos_profile[idx_vel,0:3] , vel_profile[idx_vel,0:3]])
              
            if (self.device_to_use == 'ArmAssist' and self.plant_type == 'IsMore') :
               playback_vel[rh_vel_states] = 0

        if self.plant_type in ['ReHand', 'IsMore']:
            if idx_vel == len_vel_profile - 1:  # reached the end of the trajectory
                playback_vel[rh_vel_states] = np.zeros(4)
                self.finished_vel_playback = True
            else: #ToDo: generalize
                if self.plant_type == 'ReHand':
                    playback_vel[rh_vel_states] = vel_profile[idx_vel,:]
                if self.plant_type == 'IsMore':
                    playback_vel[rh_vel_states] = vel_profile[idx_vel,3:6]  

            # if recorded ReHand trajectory is being used as the reference when playing back,
            # then don't move the ArmAssist at all
            if (self.device_to_use== 'ReHand' and self.plant_type == 'IsMore') :
               playback_vel[aa_vel_states] = 0



        # if recorded ReHand trajectory is being used as the reference when playing back,
        # then don't move the ArmAssist at all
        if (self.device_to_use == 'ReHand' and self.plant_type == 'IsMore') :
            playback_vel[aa_vel_states] = 0

        #do this so that they can be accessed from other functions
        self.playback_vel = playback_vel
        self.idx_vel = idx_vel
        self.playback_traj = playback_traj

        self.task_data['idx_vel'] = idx_vel
        self.task_data['playback_vel'] = playback_vel.values
        self.task_data['playback_traj'] = playback_traj.values

    def move_plant(self):
        #command to send a velocity profile
        command_vel  = pd.Series(0.0, self.vel_states)
        command_vel = self.playback_vel
        
        command_traj = pd.Series(0.0, np.concatenate([self.pos_states, self.vel_states]))
        command_traj = self.playback_traj

        traj = self.ref_trajectories[self.trial_type]['traj']
        
        # #Apply low-pass filter to command velocities
        for state in self.vel_states:
            command_vel[state] = self.command_lpfs[state](command_vel[state])
            if np.isnan(command_vel[state]):
                command_vel[state] = 0

        # Command zero velocity if the task is in a non-moving state
        if self.state in ['wait','rest', 'instruct_rest', 'instruct_trial_type']: 
            command_vel[:] = 0
            self.idx_vel = 0

        #elif self.state in ['trial', 'go_to_start']: 
        elif self.state == 'go_to_start':
            current_pos = self.plant_pos[:].ravel()
            current_state = np.hstack([current_pos, np.zeros_like(current_pos), 1]).reshape(-1, 1)
            target_state = np.hstack([traj[self.pos_states].ix[0], np.zeros_like(current_pos), 1]).reshape(-1, 1)

            assist_output = self.assister(current_state, target_state, 1.)
            Bu = np.array(assist_output["x_assist"]).ravel()


            command_vel[:] = Bu[len(current_pos):len(current_pos)*2]

            self.idx_vel = 0

            #to test the SetTraj algorithm, let the platform send vel commands to reach the starting position and during trial states send trajectory
            self.plant.send_vel(command_vel.values) #send velocity command to EXO
            self.task_data['command_vel']  = command_vel.values
       

        #self.plant.send_vel(command_vel.values) #send velocity command to EXO
        #self.task_data['command_vel']  = command_vel.values
       

        print "command_Traj: ", command_traj.values

        self.plant.send_traj(command_traj.values)#send trajectory to AA
        self.task_data['command_traj'] = command_traj.values
      


    def _cycle(self):
        '''Runs self.fps times per second.'''

        # get latest position/velocity information before calling move_plant()
        self.plant_pos_raw[:] = self.plant.get_pos_raw()
        self.plant_pos[:] = self.plant.get_pos() 

        self.plant_vel_raw[:] = self.plant.get_vel_raw()
        self.plant_vel[:] = self.plant.get_vel()

        # velocity control
        self.compute_playback_vel()
        self.move_plant()

        self.update_plant_display()

        self.task_data['plant_pos']  = self.plant_pos_raw.values
        self.task_data['plant_pos_filt']  = self.plant_pos.values 

        self.task_data['plant_vel']  = self.plant_vel_raw.values
        self.task_data['plant_vel_filt']  = self.plant_vel.values
        
        self.task_data['trial_type'] = self.trial_type
        self.task_data['ts']         = time.time()
        self.task_data['speed'] = self.speed
      
        super(SendVelProfile, self)._cycle()

    #### STATE AND TEST FUNCTIONS ####
    def _start_wait(self):
        # determine the random length of time to stay in the rest state
        min_time, max_time = self.rest_interval
        self.rest_time = random.random() * (max_time - min_time) + min_time

        self.finished_vel_playback = False
        self.idx_vel = 0


        super(SendVelProfile, self)._start_wait()

    def _parse_next_trial(self):
        self.trial_type = self.next_trial

    def _test_end_instruct(self, *args, **kwargs):
        return not pygame.mixer.music.get_busy()

    def _start_instruct_rest(self):
        self._play_sound(os.path.join(self.sounds_dir, 'rest.wav'))
        print 'rest'

    def _start_instruct_trial_type(self):
        sound_fname = os.path.join(self.sounds_dir, self.trial_type + '.wav')
        self._play_sound(sound_fname)

    def _start_trial(self):
        print self.trial_type

    def _test_end_trial(self, ts):
        return self.finished_vel_playback

    def _test_at_starting_config(self, *args, **kwargs):
        traj = self.ref_trajectories[self.trial_type]['traj']
        diff_to_start = np.abs(self.plant.get_pos() - traj[self.pos_states].ix[0].ravel())
        print "diff_to_start: ", diff_to_start
        return np.all(diff_to_start < self.target_rect[:len(self.pos_states)])

    def _end_trial(self):
        pass

