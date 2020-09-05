
emg_biofeedback = ['ExtCarp', 'FlexCarp']

eeg32 = ['EEG'+str(i) for i in range(1, 33)]
sleep_mont = eeg32 + ['chin1', 'chin2','ecg', 'heog', 'veog']
sleep_mont_new = ['EEG'+str(i) for i in range(1, 9)] + ['EOGHR', 'EOGHL', 'EOGVTOP', 'EOGVBOT'] + ['Extra'+str(i) for i in range(0, 20)] + ['ChinR', 'ChinL', 'ExtDigP', 'FlexDigP']


# 14 EMG channels raw 
emg14 = [
    'AbdPolLo',
    'ExtCU',
    'ExtDig',
    'Flex',
    'PronTer',
    'Biceps',
    'Triceps',
    'FrontDelt',
    'MidDelt',
    'BackDelt',
    'Extra1',
    'Extra2',
    'Extra3',
    'Extra4',
]

emg14_bip = [
    'InterFirst',
    'AbdPolLo',
    'ExtCU',
    'ExtCarp',
    'ExtDig',
    'FlexDig',
    'FlexCarp',
    'PronTer',
    'Biceps',
    'Triceps',
    'FrontDelt',
    'MidDelt',
    'TeresMajor',
    'PectMajor',
]


emg14_mono = [
    'AbdPolLo_1',
    'AbdPolLo_2',
    'ExtCU_1',
    'ExtCU_2',
    'ExtDig_1',
    'ExtDig_2',
    'Flex_1',
    'Flex_2',
    'PronTer_1',
    'PronTer_2',
    'Biceps_1',
    'Biceps_2',
    'Triceps_1',
    'Triceps_2',
    'FrontDelt_1',
    'FrontDelt_2',
    'MidDelt_1',
    'MidDelt_2',
    'BackDelt_1',
    'BackDelt_2',
    'Extra1_1',
    'Extra1_2',
    'Extra2_1',
    'Extra2_2',
    'Extra3_1',
    'Extra3_2',
    'Extra4_1',
    'Extra4_2',
]

emg12_bip = [
    'AbdPolLo',
    'ExtInd',
    'ExtDig',
    'ExtCU',
    'Supinator',
    'Flex',
    'PronTer',
    'Biceps',
    'Triceps',
    'FrontDelt',
    'MidDelt',
    'BackDelt',
]
emg14_bip_filt = [chan + '_filt' for chan in emg14_bip]
emg6_bip = [ 'EMG' + str(i+1) for i in range(6)]

accXYZ = [
        'AccX',
        'AccY',
        'AccZ',
        ]

emg12_bip_filt = [i + '_filt' for i in emg12_bip]

emg14_filt = [i + '_filt' for i in emg14]

# 14 EMG channels raw and filtered
emg14_raw_filt = emg14 + emg14_filt
# single EOG channel
eog1_raw = [
    'EOG1',
]

eog1_filt = [i + '_filt' for i in eog1_raw]
# EOG chann raw + filt
eog1_raw_filt = eog1_raw + eog1_filt

# 2 EOG channels
eog2 = [
    'EOGH',
    'EOGV',
]
eog2_filt = [i + '_filt' for i in eog2]
#4 EOG channels
eog4 = [
    'EOG1',
    'EOG2',
    'EOG3',
    'EOG4',
]
# eog4 = [
#     'EOGH1',
#     'EOGH2',
#     'EOGV1',
#     'EOGV2',
# ]
eog4_filt = [i + '_filt' for i in eog4]
# EOG_old


#eog4_old_filt = [i + '_filt' for i in eog4_old]

# 2 EOG channs raw + filt
eog2_raw_filt = eog2 + eog2_filt

# 4 EOG channs raw + filt
eog4_raw_filt = eog4 + eog4_filt

# 32 EEG channels
eeg32 = [str(i+1) for i in range(32)]
eeg32_filt = [i + '_filt' for i in eeg32]
# 32 EEG channels raw + filt
eeg32_raw_filt = eeg32 + eeg32_filt

# 32 EEG + 2 EOG
eeg32_eog2 = eeg32 + eog2
eeg32_eog2_filt = eeg32_filt + eog2_filt
eeg32_eog2_raw_filt = eeg32_eog2 + eeg32_eog2_filt

# 32 EEG + 4 EOG
eeg32_eog4 = eeg32 + eog4
eeg32_eog4_filt = eeg32_filt + eog4_filt
eeg32_eog4_raw_filt = eeg32_eog4 + eeg32_eog4_filt

# EEG 16 channels
eeg16_raw = [str(i+1) for i in range(16)]
eeg16_filt = [str(i+1)+'_filt' for i in range(16)]
eeg16_raw_filt = eeg16_raw + eeg16_filt

# EEG 48 channels
eeg48 = [str(i+1) for i in range(48)]

eeg48_filt = [i + '_filt' for i in eeg48]
# 32 EEG channels raw + filt
eeg48_raw_filt = eeg48 + eeg48_filt

#EEG 64 channels
eeg64 = [str(i+1) for i in range(64)]
eeg64_filt = [i + '_filt' for i in eeg64]

# test data loss
mono_96 = [str(i+1) for i in range(96)]
mono_96_filt = [i + '_filt' for i in mono_96]
mono_96_raw_filt = mono_96 + mono_96_filt

# 48 monopolar high-density EMG channels (first 24 for extensors and last 24 for flexors)
emg_hd_48_raw = [str(i+1) + 'ext' for i in range(24)] + [str(i+25) + 'flex' for i in range(24)]
emg_hd_48_filt = [i + '_filt' for i in emg_hd_48_raw]

emg_hd_40_filt = [str(i+1) + 'ext_filt' for i in range(20)] + [str(i+21) + 'flex_filt' for i in range(20)]

emg_hd_diag1_filt = [str(i+1) + 'ext_diag1_filt' for i in range(15)] + [str(i+16) + 'flex_diag1_filt' for i in range(15)]

emg_hd_diag2_filt = [str(i+1) + 'ext_diag2_filt' for i in range(15)] + [str(i+16) + 'flex_diag2_filt' for i in range(15)]

emg_hd_48_raw_filt = emg_hd_48_raw + emg_hd_48_filt

eeg32_emg14_mono_eog4_accXYZ = eeg32 + emg14_mono + eog4 + accXYZ
eeg32_emg14_mono_eog4_accXYZ_filt = [i + '_filt' for i in eeg32_emg14_mono_eog4_accXYZ]

emg_6bip_hd= [
    'AbdPolLo',
    'Biceps',
    'Triceps',
    'FrontDelt',
    'MidDelt',
    'BackDelt'
]
emg_6mono_hd= [
    'AbdPolLo1',
    'AbdPolLo2',
    'Biceps1',
    'Biceps2',
    'Triceps1',
    'Triceps2',
    'FrontDelt1',
    'FrontDelt2',
    'MidDelt1',
    'MidDelt2',
    'BackDelt1',
    'BackDelt2'
]

emg8_screening = [
    'ExtDigP', #paretic arm
    'FlexP',
    'BicepsP',
    'TricepsP',
    'ExtDigH', #healthy arm
    'FlexH',
    'BicepsH',
    'TricepsH',

]

emg12 = [
    'AbdPolLoP', #paretic arm
    'ExtCUP',
    'ExtDigP',
    'FlexP',
    'PronTerP',
    'BicepsP',
    'TricepsP',
    'FrontDeltP',
    'MidDeltP',
    'BackDeltP',
    'ExtDigH', #healthy arm
    'BicepsH',
]

emg_upper_arm_hd= [
    'AbdPolLo',
    'Biceps',
    'Triceps',
    'FrontDelt',
    'MidDelt',
    'BackDelt'
]

# Aurelien first run to test fatigue using the exo
emg_aurelien= [
    'Biceps',
    'Triceps_lat_head',
    'Deltoid_Ant',
    'Deltoid_Medius',
    'Deltoid_Posterior',
    'PectMajor',
    'Latissimus_Dorsi',
    'Trapezius',
    'AccX',
    'AccY',
    'AccZ'
]

# Aurelien hand electrodes (hand fatigue)
emg_aurelien_hand = [
    'EXTDIG',
    'ExtCarpiUlna',
    'AbdPolLo',
    'AbdDigMin',
    'AbdPolBre',
    'FirstDorsalInterOs',
    '7',
    '8',
    'AccX',
    'AccY',
    'AccZ',
    '12',
    '13',
    '14',
    '15',
    '16'
]


emg_hybrid = [
    '1stInterOs',
    'ExtCarpRad',
    'ExtDig',
    'ExtCarpUln',
    'DeltAnt',
    'DeltMid',
    'TeresMajor',
    'Triceps',
    'PectMajor',
    'Biceps',
  
]

emg6_sleeptask = [
    'AbdPolBre',
    'FirstDorsalInter',
    'FlexCarp',
    'Biceps',
    'Triceps',
    'extensors'
]


emg_upper_arm_hd_filt = [i + '_filt' for i in emg_upper_arm_hd]

emg_6bip_hd_filt = [i + '_filt' for i in emg_6bip_hd]

emg_6mono_hd_filt = [i + '_filt' for i in emg_6mono_hd]

emg_48hd_6bip_raw = emg_hd_48_raw + emg_6bip_hd

emg_48hd_6mono = emg_hd_48_raw + emg_6mono_hd

emg_48hd_6mono_eog4_eeg32 = emg_48hd_6mono + eog4 + eeg32

emg14_eog2_eeg32 = emg14 + eog2 + eeg32
emg14_eog2_eeg32_filt = [chan + '_filt' for chan in emg14_eog2_eeg32]

emg_48hd_6mono_diags_raw = emg_48hd_6mono

emg_48hd_6mono_filt = emg_hd_40_filt + emg_6bip_hd_filt #even if it says 48hd once they are filtered ad bpolarized we get only 40

emg_48hd_6mono_diags_filt = emg_hd_40_filt + emg_hd_diag1_filt + emg_hd_diag2_filt + emg_6bip_hd_filt

# 35 channels total (EMG, EOG, EEG)
emg_eog2_eeg9_raw_filt = emg14_raw_filt + eog2_raw_filt + eeg32_raw_filt[0:19]

# 47 channels total (EEG, EMG, EOG)
eeg32_emg_eog1_raw_filt = eeg32_raw_filt + emg14_raw_filt + eog1_raw_filt

# EEG Screening channel list (EEG32_EOG2_EMG8)
eeg32_eog2_emg8 = eeg32 + eog2 + emg8_screening
eeg32_eog2_emg8_filt = [chan + '_filt' for chan in eeg32_eog2_emg8]

# ISMORE- EEG Screening channel list (EEG32_EMG8_EOG2)
eeg32_emg8_eog2 = eeg32 + emg8_screening + eog2 
eeg32_emg8_eog2_filt = [chan + '_filt' for chan in eeg32_emg8_eog2]
# EEG(monopolar) + 2EOG(bipolar)
eeg32_eog2 = eeg32 + eog2 
eeg32_eog2_filt = [chan + '_filt' for chan in eeg32_eog2]

# EEG(monopolar) 64 channels + EOG2 bipolar
eeg64_eog4 = eeg64 + eog4

# EEG + EMG test SS channel list (EEG32_EOG2_EMG12)
eeg32_eog2_emg12 = eeg32 + eog2 + emg12
eeg32_eog2_emg12_filt = [chan + '_filt' for chan in eeg32_eog2_emg12]

# 48 channels total (EMG, EOG, EEG)
eeg32_emg_eog2_filt = eeg32_filt + emg14_filt + eog2_filt
eeg32_emg_eog2 = eeg32 + emg14 + eog2 

#48 monopolar channels (for high density EMG channels) + 6 EMG channels (12 monopolar--> 6 once they are bipolarized). Computing the bipolarization of HD in one direction (along the muscle fibers).
emg_48hd_6mono_raw_filt = emg_48hd_6mono + emg_hd_40_filt + emg_6bip_hd_filt

#48 monopolar channels (for high density EMG channels) + 6 EMG channels (12 monopolar--> 6 once they are bipolarized). Computing the bipolarization of HD in three directions(along the muscle fibers + 2 diagonals).
emg_48hd_6mono_diags_raw_filt = emg_48hd_6mono + emg_hd_40_filt + emg_hd_diag1_filt + emg_hd_diag2_filt + emg_6bip_hd_filt

eeg32_emg_48hd_6mono = eeg32 + emg_48hd_6mono
#96 channels --> to test if there is any data loss with the max number of channels that we might use (EEG+hdEMG+bipEMG+EOG)
chan96_raw = [str(i+1) for i in range(96)] 
chan96_filt = [i + '_filt' for i in chan96_raw]
chan96_raw_filt = chan96_raw + chan96_filt

eeg64_eog2_emg6_accXYZ = eeg64 + eog2 + emg6_bip + accXYZ


eeg32_emg14_eog2 = eeg32 + emg14_bip + eog2
eeg32_emg14_eog2_filt = [i + '_filt' for i in eeg32_emg14_eog2]

eeg64_eog2_emg10_accXYZ = eeg64_eog2_emg6_accXYZ + [str(i+76) for i in range(5)] + [ 'EMG' + str(i+7) for i in range(4)]

# For Jasons experiment
#eeg32_emg14_eog2_accXYZ = eeg32 + emg14_bip + eog2 + [str(i+49) for i in range(7)] + accXYZ
eeg32_eog2_accXYZ = eeg32 + eog2 + [str(i+35) for i in range(6)] + accXYZ

hybrid_bmi_channels = eeg32 

eeg32_emg6_sleeptask = eeg32 + emg6_sleeptask

# EEG + EOG + EMG14_bip test SS channel list (hybrid_2018)
hybrid_2018 = eeg32 + emg14_bip + eog2 

# For LiveAmp 64 channels
liveamp64 = [str(i+1) for i in range(28)] + ['EOG_' + str(i+1) for i in range(4)] + ['EMG 1_' + str(i+1) for i in range(32)] + ['X 1'] + ['Y 1'] + ['Z 1']

hybrid_2018_tms = ['FP1','FP2','FC5','F3','F4','FC6','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P3','Pz','P4','O1','O2','CP5','CP6','AFz','ECG'] + emg14_bip + eog2

emgonly = emg14_bip + ['Biceps_HArm', 'Speaker']

emg17_bip = [
    'InterFirst',
    'AbdPolLo',
    'ExtCU',
    'ExtCarp',
    'ExtDig',
    'FlexDig',
    'FlexCarp',
    'PronTer',
    'Biceps',
    'Triceps',
    'FrontDelt',
    'MidDelt',
    'TeresMajor',
    'PectMajor',
    'EMG15',
    'EMG16',
    'EMG17'
]


# Brain Products LiveAMP channel lists (hybrid2018/2019 modified phase 2)
liveamp_eeg_emgbip = ['EEG_'+str(i+1) for i in range(32)] + emg17_bip +['Speaker'] + eog2 + ['Acc1X', 'Acc1Y', 'Acc1Z', 'Acc2X', 'Acc2Y', 'Acc2Z']
liveamp_emgbip = emg17_bip + ['Speaker'] + ['EMG19', 'EMG20'] + ['Acc1X', 'Acc1Y', 'Acc1Z']

channel_list_options = ['emgonly',
                        'eeg32_emg6_sleeptask',
                        'eeg32_eog2_accXYZ',
                        'eeg32_emg14_eog2',
                        'emg14_bip',
                        'emg_48hd_6mono_eog4_eeg32',
                        'emg14_eog2_eeg32',
                        'eeg32_emg14_mono_eog4_accXYZ',
                        'eeg32_emg_48hd_6mono',
                        'mono_96',
                        'eeg32_emg8_eog2',
                        'eeg32_eog2_emg8',
                        'eeg32_eog2_emg12',
                        'emg_48hd_6mono',
                        'eeg32',
                        'eeg32_emg_eog2',
                        'emg14',
                        'eeg16',
                        'eeg32_eog4',
                        'chan96_raw',
                        'eeg64_eog4',
                        'emg12_bip',
                        'eeg64_eog2_emg6_accXYZ',
                        'eeg64_eog2_emg10_accXYZ',
                        'sleep_mont',
                        'emg_aurelien_hand',
                        'emg_aurelien',
                        'liveamp64',
                        'hybrid_2018',
                        'hybrid_2018_tms',
                        'liveamp_eeg_emgbip',
                        'liveamp_emgbip'
                        ]