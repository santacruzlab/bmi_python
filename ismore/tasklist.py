'''
List of tasks that can be run from the web interface. It is intended that this file NOT
be version controlled so that each rig can individually pick which tasks it wants
as dependencies.
'''

from ismore import ismoretasks
from ismore.invasive import bmi_ismoretasks, sleep_task
from ismore.noninvasive import exg_tasks
from tasks import manualcontrolmultitasks, bmimultitasks, passivetasks, cursor_clda_tasks

tasks = dict(

    # old tasks

    #ismore_record_B1_EMG         = ismoretasks.RecordB1_EMG,
    #ismore_record_B2_EMG         = ismoretasks.RecordB2_EMG,
    #ismore_record_F1_EMG         = ismoretasks.RecordF1_EMG,
    #ismore_record_F2_EMG         = ismoretasks.RecordF2_EMG,
    #ismore_record_FreeMov_EMG    = ismoretasks.RecordFreeMov_EMG,    
    #ismore_record_F2             = ismoretasks.RecordF2,
    #ismore_playback_trajectories = ismoretasks.PlaybackTrajectories,    
    #ismore_playback_trajectories2 = ismoretasks.PlaybackTrajectories2,

    #ismore_visual_feedback       = bmi_ismoretasks.VisualFeedback,
    #ismore_manual_control        = bmi_ismoretasks.ManualControl,
    #ismore_bmi_control           = bmi_ismoretasks.BMIControl,
    #ismore_clda_control          = bmi_ismoretasks.CLDAControl,
    #ismore_calibration_movements = ismoretasks.CalibrationMovements,
    #ismore_record_ReHand_data    = ismoretasks.Record_ReHand_data,
    #ismore_EMG_trajectory_decoding = exg_tasks.EMGTrajectoryDecoding,
    #ismore_EMG_traj_decoding_EndPoint_control = exg_tasks.EMGTrajDecodingEndPoint,
    #ismore_SendVelProfile        = ismoretasks.SendVelProfile,
    
    # New BMI tasks: Aug 2017
    manual_control  = bmi_ismoretasks.ManualControl, 
    compliant_move = bmi_ismoretasks.CompliantMovements,
    ismore_bmi = bmi_ismoretasks.BMIControl,
    ismore_bmi_w_emg_rest = bmi_ismoretasks.BMIControl_w_Binary_EMG,
    ismore_emg_control_w_emg_rest = bmi_ismoretasks.EMGControl_w_Binary_EMG,
    ismore_hybrid_bmi_w_emg_rest = bmi_ismoretasks.Hybrid_BMIControl_w_Binary_EMG,
    ismore_clda_all = bmi_ismoretasks.CLDAControl,
    ismore_clda_hybrid = bmi_ismoretasks.Hybrid_CLDAControl_w_Binary_EMG,
    ismore_hybrid_bmi_w_emg_rest_gotostart = bmi_ismoretasks.Hybrid_BMIControl_w_Binary_EMG_BackToStart,
    ismore_hybrid_bmi_w_emg_rest_gotostart_prp = bmi_ismoretasks.Hybrid_BMIControl_w_BinEMG_Back2Start_Prep,
    ismore_hybrid_EMGhandClass_bmi_w_emg_rest_gotostart_prp = bmi_ismoretasks.Hybrid_EMGHandClassification_BMIControl_w_BinEMG_Back2Start_Prep,
    ismore_clda_hybrid_EMG_dec_class = bmi_ismoretasks.Hybrid_EMGHandClass_CLDAControl_w_BinEMG,
    ismore_replay_exo_WO_audio = bmi_ismoretasks.ReplayBMI_wo_Audio,
    ismore_replay_exo_W_audio = bmi_ismoretasks.ReplayBMI_w_Audio,
    ismore_compliant_w_prep = bmi_ismoretasks.CompliantMovements_w_prep,
    ismore_bmi_trigger_phaseV = bmi_ismoretasks.Hybrid_GraspClass_w_RestEMG_PhaseV,
    #ismore_sim_bmi = bmi_ismoretasks.SimBMIControl, 
    #ismore_sim_clda = bmi_ismoretasks.SimCLDAControl,

    # # Non-IsMore BMI tasks: Aug 2017
    # cursor_vfb = passivetasks.TargetCaptureVFB2DWindow,
    # cursor_clda = cursor_clda_tasks.CLDARMLKF_2DWindow, 
    # cursor_clda_SB = cursor_clda_tasks.CLDAKFSmoothBatch_2DWindow,
    # cursor_bmi = bmimultitasks.BMIControlMulti2DWindow,
    # joystick = manualcontrolmultitasks.JoystickMulti2DWindow,
    # cursor_obs = bmimultitasks.BMIResettingObstacles2D,
    # emg_biofeedback = bmimultitasks.BMIControlEMGBiofeedback,


    #Active movements
    active_move = exg_tasks.Active_Movements,
    sleep = sleep_task.SleepTask,
    sleep_w_reactivation = sleep_task.SleepTask_w_reactivation,    
    sleep_bmi_task = bmi_ismoretasks.BMIControl_1D_along_traj,
    mirror_therapy_mov = exg_tasks.Mirror_Therapy_Movements,
    # tasks in Tubingen
    #ismore_record_EXG             = ismoretasks.RecordEXG,
    # ismore_Record_Base_Kin          = ismoretasks.Record_Base_Kin,

    ismore_disable_system        = ismoretasks.Disable_System,
    ismore_recordGoalTargets     = ismoretasks.RecordGoalTargets,
    ismore_recordSafetyGrid      = ismoretasks.RecordSafetyGrid, 
    # ismore_GoToTarget            = ismoretasks.GoToTarget,    
    
    # ismore_record_BrainAmpData   = ismoretasks.RecordBrainAmpData,
    # ismore_RecordExGData         = exg_tasks.RecordExGData,

    # ismore_EndPointMovement      = ismoretasks.EndPointMovement,
    # ismore_CyclicEndPointMovement = ismoretasks.CyclicEndPointMovement,
    # ismore_EMGEndPointMovement   = exg_tasks.EMGEndPointMovement,
    # ismore_EXGEndPointMovement   = exg_tasks.EXGEndPointMovement,
    # ismore_EXGCyclicEndPointMovement = exg_tasks.EXGCyclicEndPointMovement,
    # ismore_SimEEGMovementDecoding = exg_tasks.SimEEGMovementDecoding,
    # ismore_SimEEGMovementDecodingNew = exg_tasks.SimEEGMovementDecodingNew,
    
    ismore_EEGScreening          = exg_tasks.EEG_Screening,
    # ismore_EEG_Movement_Decoding = exg_tasks.EEGMovementDecoding,
    # ismore_EEG_Movement_Decoding_New = exg_tasks.EEGMovementDecodingNew,
    # ismore_EEG_Cyclic_Movement_Decoding_New = exg_tasks.EEGCyclicMovementDecodingNew,
    # ismore_EEG_Movement_Decoding_New_testing = exg_tasks.EEGMovementDecodingNew_testing,
    
    
    # ismore_ExG_FM_6movs_CODA     = exg_tasks.ExG_FM_6movs_CODA, 
    # ismore_ExG_FM_3movs_CODA     = exg_tasks.ExG_FM_3movs_CODA, 
    # ismore_ExG_FM_ARAT_CODA     = exg_tasks.ExG_FM_ARAT_CODA, 

    # # Motor Learning study tasks
    # ismore_EMGDecodingMotorLearning = exg_tasks.EMGDecodingMotorLearning,
    # ismore_EMGDecodingMotorLearning_ref = exg_tasks.EMGDecodingMotorLearning_ref,
    # ismore_EMGDecodingMotorLearning_question = exg_tasks.EMGDecodingMotorLearning_question,
    
    # ismore_HybridBCI            = exg_tasks.HybridBCI,
    # ismore_EMG_SynergiesTasks   = exg_tasks.EMG_SynergiesTasks,
    # ismore_EMGClassificationEndPoint = exg_tasks.EMGClassificationEndPoint,
    
    # # testing tasks
    ismore_EndPointMovement_testing = ismoretasks.EndPointMovement_testing,
    ismore_EXGEndPointMovement_testing   = exg_tasks.EXGEndPointMovement_testing,

)
