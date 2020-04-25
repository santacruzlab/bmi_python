import ismore.bmi_ismoretasks as bmi_ismoretasks
from riglib import experiment
from features.hdf_features import SaveHDF
import tables
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn

cmap = ['m','g','b','k','r']

def test_LQR_mats():

    f, ax = plt.subplots(nrows=2)
    Q = dict()
    Q[0] = np.mat(np.diag([100., 100., 100., 5, 5, 5, 0]))
    Q[1] = np.mat(np.diag([10., 10., 10., 5, 5, 5, 0]))
    Q[2] = np.mat(np.diag([5., 5., 5., 5, 5, 5, 0]))
    Q[3] = np.mat(np.diag([1., 1., 1., 5, 5, 5, 0]))
    Q[4] = Q[1]
    R1 = 1e6 * np.mat(np.diag([1., 1., 1.]))
    R2 = 1e8 * np.mat(np.diag([1., 1., 1.]))
    R = dict()
    for i in range(4): R[i] = R1
    R[4] = R2

    for i in range(len(Q.keys())):
        q = Q[i]
        r = R[i]
        kwargs=dict(Q=q, R=r)

        plant_ix = 0
        plant = ['ArmAssist','ReHand','IsMore']
        targets = bmi_ismoretasks.SimBMIControl.armassist_simple(length=1)
        Task = experiment.make(bmi_ismoretasks.SimBMIControl, (SaveHDF,))
        task = Task(targets, plant_type=plant[plant_ix], assist_level=(1., 1.),**kwargs)
        task.run_sync()

        ax = plot_traj(ax, task, i)
    
    ax[0].set_ylabel('X Pos Velocity')
    ax[0].set_xlabel('Time (task iterations)')
    ax[1].set_ylabel('Y Pos Velocity')
    plt.tight_layout()
    plt.savefig('Q Agg 4')


# plant_ix = 0
# plant = ['ArmAssist','ReHand','IsMore']

# targets = bmi_ismoretasks.SimBMIControl.armassist_simple(length=1)
# Task = experiment.make(bmi_ismoretasks.SimCLDAControl, (SaveHDF,))
# task = Task(targets, plant_type=plant[plant_ix], assist_level=(1., 1.))
# task.run_sync()


def plot_traj(ax, task, ix):
    fnm = task.h5file.name
    time.sleep(5)
    hdf = tables.openFile(fnm)

    trial_bins_start = [(hdf.root.task_msgs[si-3]['time'], st[1]) for si, st in enumerate(hdf.root.task_msgs[:]) if st[0]=='reward']
    # if len(trial_bins_start)==0:
    #     trial_bins_start = [(hdf.root.task_msgs[si-1]['time'], st[1]) for si, st in enumerate(hdf.root.task_msgs[:]) if st[0]=='timeout_penalty']
    


    plant_vel = dict()
    i=0
    plant_vel[i] = hdf.root.task[trial_bins_start[i][0]:trial_bins_start[i][1]]['plant_vel']
    plant_vel[i] = movingaverage(plant_vel[i], 10)
    try:
        ax[0].plot(plant_vel[i][:,3], cmap[ix])
        ax[1].plot(plant_vel[i][:,4], cmap[ix])
    except:
        ax[0].plot(plant_vel[i][:,0], cmap[ix])
        ax[1].plot(plant_vel[i][:,1], cmap[ix])
    return ax

def movingaverage(interval, window_size):
    '''
    @summary: moving average for time x dimension interval
    @param interval: time x dimension array
    @param window_size: size in units of time
    @result: convolution, same size as interval
    '''
    ma = np.zeros_like(interval)
    window= np.ones(int(window_size))/float(window_size)
    for i in range(interval.shape[1]):
        ma[:,i] = np.convolve(interval[:,i], window, 'same')
    return ma

if __name__=="__main__":
    test_LQR_mats()

