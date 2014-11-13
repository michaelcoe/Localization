
# coding: utf-8

# In[2]:

import numpy as np
import localization as lc
import matplotlib.pyplot as plt
import sys
import glob

sound_files = glob.glob('/home/interam/Documents/Robot_Localization/Oct302014_data/run*.npz')
optitrak_files = glob.glob('/home/interam/Documents/Robot_Localization/Oct302014_data/run*.csv')
location = []
source = []
s_array = []

for files in sound_files:

    source[files], s_array[files], times, mic1, mic2, mic3, mic4 = lc.get_sound_data(sound_file[files], optitrak_file[files])

    sensorPosition =[[.0015, .040],[.048, .0105],[.0025,-.0165],[-.0385,.0065]]
    mics = np.matrix(sensorPosition).T
    temperature = 23


    sampleRate = len(times)/(times[len(times)-1]*np.power(10.0,-6))
    m1, m2, m3, m4 = lc.normalize(mic1, mic2, mic3,mic4)
    m1, m2, m3, m4 = lc.filter(m1, m2, m3, m4, 800, 1200, sampleRate) 
    t, m1, m2, m3, m4 = lc.interpolate(times, m1, m2, m3, m4, 10)

    testm1 = m1[600:1000]
    testm2 = m2[600:1000]
    testm3 = m3[600:1000]
    testm4 = m4[600:1000]
    m1val = testm1[2]
    m2val = testm2[2]
    m3val = testm3[2]
    m4val = testm4[2]

    cor1, cor2, cor3, cor4 = lc.correlate(testm3, testm1, testm2, testm4)
    t1, t2, t3, t4 = lc.get_taus(cor1, cor2, cor3, cor4, sampleRate)
    '''
    if m1val > m2val and m1val > m3val and m1val > m4val:
        print 1
        cor1, cor2, cor3, cor4 = lc.correlate(testm1, testm2, testm3, testm4)
        t1, t2, t3, t4 = lc.get_taus(cor1, cor2, cor3, cor4, sampleRate)
    elif m2val > m1val and m2val > m3val and m2val > m4val:
        print 2
        cor1, cor2, cor3, cor4 = lc.correlate(testm2, testm1, testm3, testm4)
        t1, t2, t3, t4 = lc.get_taus(cor1, cor2, cor3, cor4, sampleRate)
    elif m3val > m1val and m3val > m2val and m3val > m4val:
        print 3
        cor1, cor2, cor3, cor4 = lc.correlate(testm3, testm1, testm2, testm4)
        t1, t2, t3, t4 = lc.get_taus(cor1, cor2, cor3, cor4, sampleRate)
    else:
        print 4
        cor1, cor2, cor3, cor4 = lc.correlate(testm4, testm1, testm2, testm4)
        t1, t2, t3, t4 = lc.get_taus(cor1, cor2, cor3, cor4, sampleRate)
    '''
    deltat = np.array([t1, t2, t3, t4])

    c = np.argmin(deltat)
    cTime = deltat[c]

    #calculate the difference in time relative to the shortest time
    ddt = np.array([ dt - cTime for dt in deltat ])
    location = lc.tdoa(mics, ddt, temperature)



mic1Position = [trackable2[0] + mics[0,0], trackable2[1] + mics[1,0]]
mic2Position = [trackable2[0] + mics[0,1], trackable2[1] + mics[1,1]]
mic3Position = [trackable2[0] + mics[0,2], trackable2[1] + mics[1,2]]
mic4Position = [trackable2[0] + mics[0,3], trackable2[1] + mics[1,3]]
micPosition = np.array([mic1Position, mic2Position, mic3Position, mic4Position])



plt.plot(micPosition[:,0], micPosition[:,1],'rx', trackable1[0], trackable1[1],'gd', location[0], -location[1],'bd')
plt.plot([trackable1[0], location[0]], [trackable1[1], -location[1]], 'r--')
plt.xlabel('x-axis [m]')
plt.ylabel('y-axis [m]')
plt.title('Multilateration Algorithm')
plt.title('Algorithm Guess of sound location')
#plt.legend(['Microphones', 'Optitrack Sound Location', 'Algorithm Estimate'],loc='upper right')
plt.show()

