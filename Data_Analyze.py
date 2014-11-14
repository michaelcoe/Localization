import numpy as np
import localization as lc
import matplotlib.pyplot as plt
import sys
import glob

#Running in Windows
sound_files = glob.glob('H:\Coding_Projects\Localization\Data\Oct302014\Run*.npz')
optitrak_files = glob.glob('H:\Coding_Projects\Localization\Data\Oct302014\Run*.csv')

#Running in Linux-------------------------------------------------------------------
#sound_files = glob.glob('/home/interam/Documents/Robot_Localization/Oct302014_data/run*.npz')
#optitrak_files = glob.glob('/home/interam/Documents/Robot_Localization/Oct302014_data/run*.csv')
sound_files.sort()
optitrak_files.sort()


location = []
source = []
s_array = []
sourceX = []
sourceZ = []
locationX = []
locationZ = []

sensorPosition =[[.0015, .040],[.048, .0105],[.0025,-.0165],[-.0385,.0065]]
mics = np.matrix(sensorPosition).T
temperature = 23

for files in range(0,len(sound_files)):

	source_optitrak, mics_optitrak, times, mic1, mic2, mic3, mic4 = lc.get_sound_data(sound_files[files], optitrak_files[files])

	location_alg = lc.localize(times, mic1, mic2, mic3, mic4, temperature, mics)

	s_array = np.append(s_array,mics_optitrak)
	sourceX = np.append(sourceX, source_optitrak[0])
	sourceZ = np.append(sourceZ, source_optitrak[1])
	locationX = np.append(locationX, location_alg[0])
	locationZ = np.append(locationZ, location_alg[1])

mic1Position = [s_array[0] + mics[0,0], s_array[1] + mics[1,0]]
mic2Position = [s_array[0] + mics[0,1], s_array[1] + mics[1,1]]
mic3Position = [s_array[0] + mics[0,2], s_array[1] + mics[1,2]]
mic4Position = [s_array[0] + mics[0,3], s_array[1] + mics[1,3]]
micPosition = np.array([mic1Position, mic2Position, mic3Position, mic4Position])


plt.plot(micPosition[:,0], micPosition[:,1],'rx')

for y in range(0,len(sourceX)):

	plt.plot(sourceX[y], sourceZ[y],'gd', locationX[y], -locationZ[y],'bd')
	plt.plot([sourceX[y], locationX[y]], [sourceZ[y], -locationZ[y]], 'r--')
	print y

plt.xlabel('x-axis [m]')
plt.ylabel('y-axis [m]')
plt.title('Multilateration Algorithm')
plt.title('Algorithm Guess of Sound Location (2 Separate Runs)')
plt.legend(['Microphones', 'Optitrack Sound Location', 'Algorithm Estimate'],loc='upper right')
plt.show()