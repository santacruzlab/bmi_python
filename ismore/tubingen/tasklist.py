'''
List of tasks that can be run from the web interface. It is intended that this file NOT
be version controlled so that each rig can individually pick which tasks it wants
as dependencies.
'''

from ismore import ismoretasks
from ismore.noninvasive import exg_tasks
from ismore.tubingen import tubingentasks
from ismore.tubingen.noninvasive_tubingen import exg_tasks_tubingen

tasks = dict(

    # tasks in Tubingen
    #ismore_record_EXG             = ismoretasks.RecordEXG,

    ismore_Record_MuscleFatigue_PassiveBase = tubingentasks.Record_MuscleFatigue_PassiveBase,
    ismore_disable_system        = ismoretasks.Disable_System,
    ismore_recordGoalTargets     = ismoretasks.RecordGoalTargets,
    ismore_recordSafetyGrid      = ismoretasks.RecordSafetyGrid, 
    ismore_GoToTarget            = ismoretasks.GoToTarget,    
    
    # ismore_record_BrainAmpData   = ismoretasks.RecordBrainAmpData,
    # ismore_RecordExGData         = exg_tasks.RecordExGData,

    # ismore_EndPointMovement      = ismoretasks.EndPointMovement,
    # ismore_CyclicEndPointMovement = ismoretasks.CyclicEndPointMovement,
    # ismore_EMGEndPointMovement   = exg_tasks.EMGEndPointMovement,
    # ismore_EXGEndPointMovement   = exg_tasks.EXGEndPointMovement,
    # ismore_EXGCyclicEndPointMovement = exg_tasks.EXGCyclicEndPointMovement,
    # ismore_SimEEGMovementDecoding = exg_tasks.SimEEGMovementDecoding,
    # ismore_SimEEGMovementDecodingNew = exg_tasks.SimEEGMovementDecodingNew,
    
    #
    ismore_EEGScreening          = exg_tasks.EEG_Screening,
    hybrid2018_EEGScreening      = tubingentasks.EEG_Screening_Tue,
    hybrid2018_Intervention      = tubingentasks.HybridBCI,
    hybrid2018_GoToTarget        = tubingentasks.hybrid_GoToTarget,
    hybrid2018_EXGEndPointMovement = tubingentasks.hybrid_EXGEndPointMovement,
    emg_only_control_2019_eval = tubingentasks.EMG_Only_Control_Evaluation,
    emg_only_control_2019_ref = tubingentasks.EMG_Only_Control_Reference,
    # ismore_EEG_Movement_Decoding = exg_tasks.EEGMovementDecoding,
    ismore_EEG_Movement_Decoding_New = exg_tasks_tubingen.EEGMovementDecodingNew,
    # ismore_EEG_Cyclic_Movement_Decoding_New = exg_tasks.EEGCyclicMovementDecodingNew,
    # ismore_EEG_Movement_Decoding_New_testing = exg_tasks.EEGMovementDecodingNew_testing,

    # # Motor Learning study tasks
    # ismore_EMGDecodingMotorLearning = exg_tasks.EMGDecodingMotorLearning,
    #ismore_EMGDecodingMotorLearning_ref = exg_tasks.EMGDecodingMotorLearning_ref,
    #ismore_EMGDecodingMotorLearning_question = exg_tasks.EMGDecodingMotorLearning_question,
    
    # ismore_HybridBCI            = exg_tasks.HybridBCI,
    # ismore_EMG_SynergiesTasks   = exg_tasks.EMG_SynergiesTasks,
    # ismore_EMGClassificationEndPoint = exg_tasks.EMGClassificationEndPoint,
    
    # # testing tasks
    ismore_EndPointMovement_testing = ismoretasks.EndPointMovement_testing,
    ismore_EXGEndPointMovement   = exg_tasks.EXGEndPointMovement,
    ismore_EXGEndPointMovement_testing   = exg_tasks.EXGEndPointMovement_testing,

    sleep_1D_EEG_BCI_exo_FES_task = tubingentasks.Sleep_1D_EEG_BCI_exo_FES_task,


)