'''
List of features which can be used to extend experiments through the web interface
'''

from features.blackrock_features import BlackrockBMI
from features.arduino_features import BlackrockSerialDIORowByte
from ismore.brainamp_features import BrainAmpData, SimBrainAmpData
from features.hdf_features import SaveHDF
from features.peripheral_device_features import ArduinoJoystick, ArduinoIMU
from ismore.start_video_features import StartVideo
from ismore.coda_features import CodaSync
from ismore.exo_3D_visualization import Exo3DVisualization, Exo3DVisualizationInvasive, BMIMonitor

features = dict(
    blackrockbmi        = BlackrockBMI,
    relay_blackrockbyte = BlackrockSerialDIORowByte,
    brainampdata        = BrainAmpData,
    simbrainampdata     = SimBrainAmpData,
    saveHDF             = SaveHDF,
    startVideo          = StartVideo,
    CodaSync            = CodaSync,
    exo_3D_visualization = Exo3DVisualization,
    # exo_3D_vis_invasive = Exo3DVisualizationInvasive,
    joystick            = ArduinoJoystick,
    imu                 = ArduinoIMU,
    bmi_monitor         = BMIMonitor,
)
