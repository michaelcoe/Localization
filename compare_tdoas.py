import numpy as np
from numpy import linalg as LA
import localization as lc
import matplotlib.pyplot as plt
import sys
import glob

sound_files = glob.glob('/home/interam/Documents/Robot_Localization/Oct302014_data/run*.npz')
optitrak_files = glob.glob('/home/interam/Documents/Robot_Localization/Oct302014_data/run*.csv')


sound_file = '/home/interam/Documents/Robot_Localization/Oct302014_data/run2.npz'
optitrak_file = '/home/interam/Documents/Robot_Localization/Oct302014_data/run2.csv'

trackable1, trackable2, times, mic1, mic2, mic3, mic4 = lc.get_sound_data(sound_file, optitrak_file)

sensorPosition =[[.0015, .040],[.048, .0105],[.0025,-.0165],[-.0385,.0065]]
mics = np.matrix(sensorPosition).T
temperature = 23


mic1Position = [trackable2[0] + mics[0,0], trackable2[1] + mics[1,0]]
mic2Position = [trackable2[0] + mics[0,1], trackable2[1] + mics[1,1]]
mic3Position = [trackable2[0] + mics[0,2], trackable2[1] + mics[1,2]]
mic4Position = [trackable2[0] + mics[0,3], trackable2[1] + mics[1,3]]
micPosition = np.array([mic1Position, mic2Position, mic3Position, mic4Position])

dist1 = LA.norm(trackable1 - mic1Position)
dist2 = LA.norm(trackable1 - mic2Position)
dist3 = LA.norm(trackable1 - mic3Position)
dist4 = LA.norm(trackable1 - mic4Position)

print dist1, dist2, dist3, dist4

v = (331.3+(0.606*temperature))
tt1 = dist1/v
tt2 = dist2/v
tt3 = dist3/v
tt4 = dist4/v

sampleRate = len(times)/(times[len(times)-1]*np.power(10.0,-6))
m1, m2, m3, m4 = lc.normalize(mic1, mic2, mic3,mic4)
m1, m2, m3, m4 = lc.run_filter(m1, m2, m3, m4, 800, 1200, sampleRate) 
t, m1, m2, m3, m4 = lc.interpolate(times, m1, m2, m3, m4, 10)

testtime = t[600:1000]
testm1 = m1[512:1024]
testm2 = m2[512:1024]
testm3 = m3[512:1024]
testm4 = m4[512:1024]
m1val = testm1[2]
m2val = testm2[2]
m3val = testm3[2]
m4val = testm4[2]

cor1, cor2, cor3, cor4 = lc.correlate(testm1, testm2, testm3, testm4)
t1, t2, t3, t4 = lc.get_taus(cor1, cor2, cor3, cor4, sampleRate)
pht1, pht2, pht3, pht4 = lc.find_phase(testm1,testm2,testm3,testm4)
pht = np.array([pht1, pht2, pht3, pht4])

deltat = np.array([t1, t2, t3, t4])

c = np.argmin(deltat)
cTime = deltat[c]

#calculate the difference in time relative to the shortest time
ddt = np.array([ dt - cTime for dt in deltat ])

cdt1 = (tt1 - tt1)
cdt2 = (tt2 - tt1)
cdt3 = (tt3 - tt1)
cdt4 = (tt4 - tt1)
cal_dt = np.array([cdt1, cdt2, cdt3, cdt4])

print pht
print ddt
print cal_dt


t = np.linspace(0, 10, 1000)
sin1 = np.sin(1000*(t))
sin2 = np.sin((1000*(t))-(np.pi/4))
plt.plot(t,sin1,t,sin2)
plt.show()
calc_timediff = (np.pi/4)/(2*np.pi*1000)


sin_phi = np.arccos(2*np.mean(sin1*sin2))

print sin_phi/(2*np.pi*1000)
print calc_timediff


