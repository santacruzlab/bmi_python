#!/usr/bin/python

import Tkinter as tk    # for the GUI
import ttk              # for nicer GUI widgets
import threading        # for parallel computing
import time
import subprocess
import os


# A thread that continously request the status of the MWG
class myThread (threading.Thread):
    # initialize class
    def __init__(self, ser):
        threading.Thread.__init__(self)

    # # gets called when thread is started with .start()
    def run(self):

        mHeader    = ttk.Label(root, text = "Start / Stop recordings").grid(row=0, column=1)
        mHeader    = ttk.Label(root, text = "                        ").grid(row=0, column=2)
        mHeader    = ttk.Label(root, text = "Reactivation stimulus ON / OFF ").grid(row=0, column=3)
        stop_button = ttk.Button(root, text ="STOP", command =  lambda : stop_sleep_task(False)).grid(row=3, column=1)
        stim_on_button = ttk.Button(root, text ="STIM ON", command =  lambda : write_stim_on_off('T')).grid(row=1, column=3)
        stim_off_button = ttk.Button(root, text ="STIM OFF", command =  lambda : write_stim_on_off('F')).grid(row=3, column=3)

        stim_vol_combo = ttk.Combobox(root, state='readonly')
        stim_vol_combo.grid(row=4, column=3)
        stim_vol_combo["values"] = ["0.30", "0.35", "0.40", "0.45", "0.50", "0.55", "0.60", "0.65", "0.70", "0.75", "0.80", "0.85" ]
        
        path_txt = os.path.expandvars('$HOME/code/ismore/invasive/stim_vol.txt')
        stim_vol_txt= open(path_txt,"r")
        stim_vol_value= stim_vol_txt.read()
        stim_vol_combo.set(stim_vol_value)
        stim_vol_txt.close()


        stim_vol_change_button = ttk.Button(root, text ="SET VOLUME", command =  lambda : set_stim_volume()).grid(row=4, column=4)

        stim_vol_value = stim_vol_combo.get()

        print "Initial stim volume: ",  stim_vol_value

        def set_stim_volume():
            stim_vol_value = stim_vol_combo.get()
            path_txt = os.path.expandvars('$HOME/code/ismore/invasive/stim_vol.txt')
            print 'Stim volume changed to ', stim_vol_value
            stim_vol_txt= open(path_txt,"w+")
            stim_vol_txt.write(stim_vol_value)
            stim_vol_txt.close()


def stop_sleep_task(rec_stop_flag):
    print "STOP recording"
    path_txt = os.path.expandvars('$HOME/code/ismore/invasive/rec_stop.txt')
    rec_stop_txt= open(path_txt,"w+")
    rec_stop_txt.write("T")
    rec_stop_txt.close()

def write_stim_on_off(T_F):
    if T_F == 'T':
        stim_state = 'ON'
    elif T_F == 'F':
        stim_state = 'OFF'
    print 'Reactivation stimulus is ', stim_state
    path_txt = os.path.expandvars('$HOME/code/ismore/invasive/stim_on_off.txt')
    stim_txt= open(path_txt,"w+")
    stim_txt.write(T_F)
    stim_txt.close()


root = tk.Tk()
root.geometry("600x150")
root.title("Sleep recordings")
 
# call and start update-thread
thread1 = myThread("Updating")
thread1.start()
 
# start GUI
root.mainloop()