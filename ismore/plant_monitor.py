'''
Plot things related to the plant(s)
'''
from twisted.internet.protocol import DatagramProtocol
from twisted.internet import reactor
import numpy as np
import struct
import matplotlib.pyplot as plt

Delta = 1./60
n_dof = 7

vel_data = np.zeros([7, 100])

fig, axes = plt.subplots(n_dof, 1)
lines = [None]*n_dof
ylims = [(-10, 10)] * 7
for k in range(n_dof):
    lines[k] = axes[k].plot(vel_data[k])
    axes[k].set_ylim(ylims[k])


plt.show()

class Plot(DatagramProtocol):
    def datagramReceived(self, data, (host, port)):
        vel = struct.unpack("d"*n_dof, data)
        vel_data[:,:-1] = vel_data[:,1:]
        vel_data[:,-1] = vel

        for k in range(n_dof):
            lines[k][0].set_ydata(vel_data[k])

        plt.draw()


##class Echo(DatagramProtocol):
##    joint_angles = np.zeros(5)
##    joint_velocities = np.zeros(5)
##    joint_applied_torque = np.zeros(5)
##    n_joints = 5
##
##    def datagramReceived(self, data, (host, port)):
##        self.joint_angles += Delta * self.joint_velocities
##        if data == 's':
##            pass
##        else: # should be joint velocity
##            vel = struct.unpack('>I' + 'd'*self.n_joints, data)
##            self.joint_velocities = np.array(vel)[1:]
##
##        vec_data = tuple(np.hstack([5, self.joint_angles, 5, self.joint_velocities, 5, self.joint_applied_torque]))
##        print vec_data
##        print len(vec_data)
##        return_data = struct.pack('>IdddddIdddddIddddd', *vec_data)
##        self.transport.write(return_data, (host, 60000))

reactor.listenUDP(60003, Plot())
reactor.run()
