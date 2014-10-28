import socket
import numpy as np
from struct import *
import localization as lc
import matplotlib.pyplot as plt

BROADCAST_PORT = 58083
 
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('0.0.0.0', BROADCAST_PORT))


x=0
cycles = 5 #number of cycle readings to save in quantities of 512 samples
delay = np.zeros(cycles)


while x<cycles:
    for i in range(0,cycles):
        if i == 0:
            val = s.recvfrom(5122)
            data =  np.fromstring(val[0], dtype='H')
            times = data[0:512]*(1.0)
            mic1 = data[512:1024]*(3300.0/65535.0)
            mic2 = data[1024:1536]*(3300.0/65535.0)
            mic3 = data[1536:2048]*(3300.0/65535.0)
            mic4 = data[2048:2560]*(3300.0/65535.0)
            delay[i] = data[2560]*(1.0)
            
            #fs = 512.0 / (times[len(times)-1]*np.power(10.0,-6))
            #mic1, mic2, mic3, mic4 = lc.normalize(mic1, mic2, mic3, mic4)
            #mic1, mic2, mic3, mic4 = lc.filter(mic1, mic2, mic3, mic4, 800.0, 1200.0, fs)
        else:
            val = s.recvfrom(5122)
            data =  np.fromstring(val[0], dtype='H')
            t = data[0:512]*1.0
            m1 = data[512:1024]*(3300.0/65535.0)
            m2 = data[1024:1536]*(3300.0/65535.0)
            m3 = data[1536:2048]*(3300.0/65535.0)
            m4 = data[2048:2560]*(3300.0/65535.0)
            delay[i] = data[2560]*(1.0)
            
            elapsed_time = times[len(times)-1]
            t = delay[i] + t + elapsed_time
            
            #fs = 512.0 / (t[len(t)-1]*np.power(10.0,-6))
            #m1, m2, m3, m4 = lc.normalize(m1, m2, m3, m4)
            #m1, m2, m3, m4 = lc.filter(m1, m2, m3, m4, 800.0, 1200.0, fs)
    
    
            if delay[i] > 0:
                t = np.insert(t, 0, np.NAN)
                m1 = np.insert(m1, 0, np.NAN)
                m2 = np.insert(m2, 0, np.NAN)
                m3 = np.insert(m3, 0, np.NAN)
                m4 = np.insert(m4, 0, np.NAN)
            
                times = np.append(times, t)
                mic1 = np.append(mic1, m1)
                mic2 = np.append(mic2, m2)
                mic3 = np.append(mic3, m3)
                mic4 = np.append(mic4, m4)
            else:
                times = np.append(times, t)
                mic1 = np.append(mic1, m1)
                mic2 = np.append(mic2, m2)
                mic3 = np.append(mic3, m3)
                mic4 = np.append(mic4, m4)
    x+=1

np.savez('test.npz', times=times, mic1=mic1, mic2=mic2, mic3=mic3, mic4=mic4)
print 'saved'

fig, ax = plt.subplots()
p1 = ax.plot(times, mic1, label='Mic 1')
p2 = ax.plot(times, mic2, label='Mic 2')
p3 = ax.plot(times, mic3, label='Mic 3')
p4 = ax.plot(times, mic4, label='Mic 4')
plt.xlabel('time [microseconds]')
plt.ylabel('Voltage [mV]')
plt.title('Electrical Noise, no input to the ADC Channels')
legend = ax.legend(loc='upper right', shadow = True)
plt.show()
