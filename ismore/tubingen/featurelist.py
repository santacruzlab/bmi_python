'''
List of features which can be used to extend experiments through the web interface
'''

from ismore.brainamp_features import BrainAmpData, SimBrainAmpData, LiveAmpData
from features.hdf_features import SaveHDF
from ismore.tubingen.tuebingen_features.VideoRecording import VideoRecording
from ismore.exo_3D_visualization import Exo3DVisualization, Exo3DVisualizationInvasive
from ismore.tubingen.tuebingen_features.AdvancedVisualization import AdvancedVisualization

features = dict(
    brainampdata        = BrainAmpData,
    liveampdata        = LiveAmpData,
    simbrainampdata 	= SimBrainAmpData,
    saveHDF             = SaveHDF,
    recordVideo 			= VideoRecording,
    AdvancedVisualization = AdvancedVisualization
)