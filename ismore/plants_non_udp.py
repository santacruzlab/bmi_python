################################################        


class BasePlantNonUDP(BasePlant):
    
    def init(self):
        pass

    def stop(self):
        pass

    def enable(self):
        pass

    def last_data_ts_arrival(self):
        # there's no delay when receiving feedback using the NonUDP classes, 
        #   since nothing is being sent over UDP and feedback data can be 
        #   requested at any time 
        return time.time()

    def disable(self):
        pass

    def enable_watchdog(self, timeout_ms):
        pass


class ArmAssistPlantNonUDP(BasePlantNonUDP):
    '''Similar methods as ArmAssistPlantUDP, but: 
        1) doesn't send/receive anything over UDP, and 
        2) uses simulated ArmAssist (can't be used with real ArmAssist).
       Use this plant to simulate having (near) instantaneous feedback.
    '''

    def __init__(self, *args, **kwargs):
        # create ArmAssist process
        aa_tstep = 0.005                  # how often the simulated ArmAssist moves itself
        aa_pic_tstep = 0.01               # how often the simulated ArmAssist PI controller acts
        KP = np.mat([[-10.,   0.,  0.],
                     [  0., -20.,  0.],
                     [  0.,   0., 20.]])  # P gain matrix
        TI = 0.1 * np.identity(3)         # I gain matrix

        self.aa = armassist.ArmAssist(aa_tstep, aa_pic_tstep, KP, TI)
        self.aa.daemon = True

    def start(self):
        '''Start the ArmAssist simulation processes.'''
        
        self.aa.start()
        self.ts_start_data = time.time()

    def send_vel(self, vel):
        vel = vel.copy()
        
        # units of vel should be: (cm/s, cm/s, rad/s)
        assert len(vel) == 3

        # don't need to convert from rad/s to deg/s
        # (aa_pic expects units of rad/s)

        vel = np.mat(vel).T
        self.aa.update_reference(vel)

    # make note -- no conversion needed

    def get_pos(self):
        return np.array(self.aa.get_state()['wf']).reshape((3,))

    def get_vel(self):
        return np.array(self.aa.get_state()['wf_dot']).reshape((3,))

    # a magic function that instantaneously moves the simulated ArmAssist to a 
    #   new position+orientation
    def set_pos(self, pos):
        '''Magically set position+orientation in units of (cm, cm, rad).'''
        wf = np.mat(pos).T
        self.aa._set_wf(wf)


class ReHandPlantNonUDP(BasePlantNonUDP):
    '''Similar methods as ReHandPlantUDP, but: 
        1) doesn't send/receive anything over UDP, and 
        2) uses simulated ReHand (can't be used with real ReHand).
       Use this plant to simulate having (near) instantaneous feedback.
    '''
    
    def __init__(self, *args, **kwargs):
        # create ReHand process
        self.rh = rehand.ReHand(tstep=0.005)
        self.rh.daemon = True

    def start(self):
        '''Start the ReHand simulation process.'''

        self.rh.start()
        self.ts_start_data = time.time()

    def send_vel(self, vel):
        vel = vel.copy()

        # units of vel should be: (rad/s, rad/s, rad/s, rad/s)
        assert len(vel) == 4
        
        # don't need to convert from rad/s to deg/s
        # (rh expects units of rad/s)

        vel = np.mat(vel).T
        self.rh.set_vel(vel)

    # no conversion needed (everything already in units of rad)

    def get_pos(self):
        return np.array(self.rh.get_state()['pos']).reshape((4,))

    def get_vel(self):
        return np.array(self.rh.get_state()['vel']).reshape((4,))

    # a magic function that instantaneously sets the simulated ReHand's angles
    def set_pos(self, pos):
        '''Magically set angles in units of (rad, rad, rad, rad).'''
        self.rh._set_pos(pos)

NONUDP_PLANT_CLS_DICT = {
    'ArmAssist': ArmAssistPlantNonUDP,
    'ReHand':    ReHandPlantNonUDP,
    'IsMore':    IsMorePlantNonUDP,
}


class IsMorePlantNonUDP(BasePlantIsMore):
    '''Similar methods as IsMorePlant, but: 
        1) doesn't send/receive anything over UDP, and 
        2) uses simulated ArmAssist+ReHand (can't be used with real devices).
       Use this plant to simulate having (near) instantaneous feedback.
    '''

    aa_plant_cls = ArmAssistPlantNonUDP
    rh_plant_cls = ReHandPlantNonUDP

    # a magic function that instantaneously moves the simulated ArmAssist to a 
    #   new position+orientation and sets the simulated ReHand's angles
    def set_pos(self, pos):
        '''Magically set ArmAssist's position+orientation in units of 
        (cm, cm, rad) and ReHand's angles in units of (rad, rad, rad, rad).
        '''
        self.aa_plant.set_pos(pos[0:3])
        self.rh_plant.set_pos(pos[3:7])
