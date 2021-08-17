# import comedi
import config
import time

baudrate_rew = 9600
import serial
arduino_neurosync = serial.Serial('/dev/arduino_neurosync', baudrate=self.baudrate_rew)

arduino_neurosync.write(b'l')
# comedi.comedi_dio_bitfield2(com, 0, 1, 0, 15)
print("set last data pin (D15; 16th pin) low DONE")

# com = comedi.comedi_open("/dev/arduino_neurosync")
# time.sleep(0.5)
# # strobe pin should already be low

# # set last data pin ("D15"; 16th pin) low
# comedi.comedi_dio_bitfield2(com, 0, 1, 0, 15)
# print("set last data pin (D15; 16th pin) low DONE")

# # set strobe pin high
# comedi.comedi_dio_bitfield2(com, 0, 1, 1, 16)
# print("set strobe pin high")

# # set strobe pin low
# comedi.comedi_dio_bitfield2(com, 0, 1, 0, 16)
# print("set strobe pin low")
