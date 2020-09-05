#!/usr/bin/python

# import packages
import Tkinter as tk    # for the GUI
import ttk              # for nicer GUI widgets
import tkMessageBox     # for GUI testbox
import serial           # for communication with serial port
import time             # for time stuff
import threading        # for parallel computing

import traceback
import numpy as np
import struct
import datetime
import serial
import time
 
 
# A thread that continously request the status of the MWG
class myThread (threading.Thread):
    # initialize class
    def __init__(self, ser,port):
        threading.Thread.__init__(self)
        # Initialize port
        self.port = port

    # # gets called when thread is started with .start()
    def run(self):

        mHeader    = ttk.Label(root, text = "CODA synchronization").grid(row=0, column=1)
        start_button = ttk.Button(root, text ="START", command =  lambda : send_coda_msg('start')).grid(row=1, column=1)
        accept_button = ttk.Button(root, text ="ACCEPT", command = lambda: send_coda_msg('accept')).grid(row=1, column=2)
        reject_button = ttk.Button(root, text ="REJECT", command = lambda: send_coda_msg('reject')).grid(row=1, column=3)

def send_coda_msg(msg):
    # print "trial count" , trial_count
    if msg == 'accept':
        print 'accept'
        word = 0
        word_str = 'c' + struct.pack('<H', word)
        port.write(word_str)

        time.sleep(0.05)

        trial_count = 2
        word = trial_count % 8
        word_str = 'c' + struct.pack('<H', word)
        port.write(word_str)

        time.sleep(0.05)

        word = 0
        word_str = 'c' + struct.pack('<H', word)
        port.write(word_str)

    elif msg == 'reject':
        print 'reject'
        word = 0
        word_str = 'c' + struct.pack('<H', word)
        port.write(word_str)

        time.sleep(0.05)

        trial_count = 4
        word = trial_count % 8
        word_str = 'c' + struct.pack('<H', word)
        port.write(word_str)

        time.sleep(0.05)

        word = 0
        word_str = 'c' + struct.pack('<H', word)
        port.write(word_str)

    elif msg == 'start':
        print 'start'
        trial_count = 1
        word = trial_count % 8
        
        #send a 'c' for 'coda' to arduino:
        word_str = 'c' + struct.pack('<H', word)
        port.write(word_str)    

    return


port = serial.Serial('/dev/arduino_neurosync', baudrate=115200)

# set up root window
root = tk.Tk()
root.geometry("350x50")
root.title("CODA synchronization")
 
# wait
time.sleep(1)
# call and start update-thread
thread1 = myThread("Updating", port)
thread1.start()
 
# start GUI
root.mainloop()
 