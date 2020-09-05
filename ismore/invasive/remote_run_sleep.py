import os, subprocess
import time
import signal


def main(nmin=20, nminblk=5):
    proceed = True
    t_start = time.time()

    # make sure that the stimuli is OFF at the beginning of the recording
    path_stim_txt = os.path.expandvars('$HOME/code/ismore/invasive/stim_on_off.txt')
    stim_txt= open(path_stim_txt,"w+")
    stim_txt.write("F")
    stim_txt.close()

    path_rec_txt = os.path.expandvars('$HOME/code/ismore/invasive/rec_stop.txt')
    rec_stop_txt= open(path_rec_txt,"w+")
    rec_stop_txt.write("F")
    rec_stop_txt.close()

    while proceed:

        path_rec_txt = os.path.expandvars('$HOME/code/ismore/invasive/rec_stop.txt')
        rec_stop_txt= open(path_rec_txt,"r")
        rec_stop_flag = rec_stop_txt.read()

        if rec_stop_flag == 'T':
            "RECORDING STOPPED REMOTELY"
            break
        
        os.environ['sess'] = str(60*nminblk)
        x = subprocess.check_output('python sleep_task.py $sess', shell=True)
        if x[-13:-1] == 'cleanup done':
            proceed = True
        else:
            proceed = False

        if (time.time() - t_start) > 60*nmin:
            proceed = False
        print 'finished a block! '



    print 'finished all! '


