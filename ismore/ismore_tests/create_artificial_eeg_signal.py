import numpy as np
import matplotlib.pyplot as plt
import math

# Creating artificial EEG signal
fsample = 1000.00 #Sample frequency in Hz
t = np.arange(0, 10 , 1/fsample)#[0 :1/fsample:10-1/fsample]; #time frames of 10seg for each state

#Rest
f = 10 # in Hz
rest_amp = 10; #rest state amplitude 10Hz
rest_noise = rest_amp*0.1*np.random.randn(len(t)) #10% of signal amplitude
rest_signal = np.zeros(len(t))
for i in np.arange(len(t)):
	rest_signal[i] = rest_amp * math.sin(f*2*math.pi*t[i]) + rest_noise[i] #rest sinusoidal signal

#Plot rest
plt.figure()
plt.plot(t, rest_signal)
# title('Rest signal');
# xlabel('Time (s)')
# ylabel('Signal (Hz)');
# ylim([-15 15]);
#import pdb; pdb.set_trace()
#Move
move_amp = 5; #mov state amplitude 
move_noise = move_amp*0.1*np.random.randn(len(t)) #10% of signal amplitude
move_signal = np.zeros(len(t))
for i in np.arange(len(t)):
	move_signal[i] = move_amp * math.sin(f*2*math.pi*t[i]) + move_noise[i]
#Plot move
plt.plot(t, move_signal, 'r')
#plt.show()
# title('Move signal');
# xlabel('Time (s)')
# ylabel('Signal (Hz)');
# ylim([-15 15]);

#Generating an entire 5min signal and its respective label
signal = [] #Signal composed by rest and  move
label = [] #1:rest, 0:move

for i in np.arange(15):
    signal = np.hstack([signal, rest_signal, move_signal])
    
    label = np.hstack([label, np.ones([len(rest_signal)]), np.zeros([len(move_signal)])])
time = np.arange(0, 5*60, 1/fsample) #time vector for 5min
    

# import pdb; pdb.set_trace()
plt.figure()
plt.plot(time,signal)
plt.show()


