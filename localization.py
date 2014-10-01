import numpy as np
from numpy.linalg import *
from scipy.interpolate import interp1d
import collections

def interpolate(m1, m2, m3, m4, factor):

	x = np.linspace(0,len(m1),len(m1))

	inter_factor = np.linspace(0,len(m1),len(m1)*factor) #sets up the amount of interpolation to be done

	f1 = interp1d(x,m1)
	f2 = interp1d(x,m2)
	f3 = interp1d(x,m3)
	f4 = interp1d(x,m4)

	return f(inter_factor), f2(inter_factor), f3(inter_factor), f4(inter_factor)

def correlate(m1, m2, m3, m4):

	##----------------------------
	#Assumes that m1 is the reference microphone
	##----------------------------

	cor1 = np.correlate(m1,m2)
	cor2 = np.correlate(m1,m3)
	cor3 = np.correlate(m1,m4)

	return cor1, cor2, cor3	


def tdoa(t1, t2, t3, t4)
	#speed of sound in medium
	v = 3450
	numOfDimensions = 3
	nSensors = 5
	region = 3
	sensorRegion = 2


	#Time from emitter to each sensor
	sensorTimes = [ sqrt( dot(location-emitterLocation,location-emitterLocation) ) / v for location in sensorLocations ]

	c = argmin(sensorTimes)
	cTime = sensorTimes[c]

	#sensors delta time relative to sensor c
	t = sensorDeltaTimes = [ sensorTime - cTime for sensorTime in sensorTimes ]

	ijs = range(nSensors)
	del ijs[c]

	A = zeros([nSensors-1,numOfDimensions])
	b = zeros([nSensors-1,1])
	iRow = 0
	rankA = 0
	for i in ijs:
		for j in ijs:
			A[iRow,:] = 2*( v*(t[j])*(p[:,i]-p[:,c]).T - v*(t[i])*(p[:,j]-p[:,c]).T )
			b[iRow,0] = v*(t[i])*(v*v*(t[j])**2-p[:,j].T*p[:,j]) + \
			(v*(t[i])-v*(t[j]))*p[:,c].T*p[:,c] + \
			v*(t[j])*(p[:,i].T*p[:,i]-v*v*(t[i])**2)
			rankA = matrix_rank(A)
			if rankA >= numOfDimensions :
				break
			iRow += 1
		if rankA >= numOfDimensions:
			break

	calculatedLocation = asarray( lstsq(A,b)[0] )[:,0]

	return calculatedLocation
