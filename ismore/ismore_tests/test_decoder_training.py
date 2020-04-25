from riglib.bmi import train
from config import config
from riglib.bmi import extractor, state_space_models
def train(te_num):
    te = dbfn.TaskEntry(te_num)
    
    files = dict(hdf=te.hdf_filename, blackrock=[te.blackrock_filenames[0]])
    tslice = None

    #pos_key = 'cursor'
    pos_key = 'plant_pos'

    kin = kin[1:, :]
    neural_features = neural_features[:-1, :].T

    #kin = kin[1:].T
    #neural_features = neural_features[:-1].T

    ssm = state_space_models.StateSpaceEndptVel2D()

    decoder = train.train_KFDecoder_abstract(ssm, kin, neural_features, units, update_rate, tslice=tslice, zscore=zscore, **kwargs)
