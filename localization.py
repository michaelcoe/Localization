import numpy as np
from numpy.linalg import *
from scipy.interpolate import interp1d
import scipy.signal as sig
import math as m
import collections
import matplotlib.pyplot as plt
import csv
import sys

def get_sound_data(sound_file, optitrak_file):

	trackable1x = []
	trackable1z = []
	trackable2x = []
	trackable2z = []
	position_file = open(optitrak_file, 'rb')
	try:
		reader = csv.reader(position_file)
		for row in reader:
			if row[0] == 'frame':
				trackable1x.append(row[5])
				trackable1z.append(row[7])
				trackable2x.append(row[16])
				trackable2z.append(row[18])
	finally:
		position_file.close()

	sound_source = np.array([float(trackable1x[1]), -float(trackable1z[1])])
	mic_array = np.array([float(trackable2x[1]), float(trackable2z[1])])
	data = np.load(sound_file)
	times = data['times']
	mic1 = data['mic1']
	mic2 = data['mic2']
	mic3 = data['mic3']
	mic4 = data['mic4']

	return sound_source, mic_array, times, mic1, mic2, mic3, mic4

def localize(times, mic1, mic2, mic3, mic4, temperature, mics):

	sampleRate = len(times)/(times[len(times)-1]*np.power(10.0,-6))
	m1, m2, m3, m4 = normalize(mic1, mic2, mic3,mic4)
	m1, m2, m3, m4 = run_filter(m1, m2, m3, m4, 800, 1200, sampleRate)

	plt.plot(times, m1, times, m2, times, m3,times, m4)
	plt.xlabel('Time [microseconds]')
	plt.ylabel('Voltage [mV]')
	plt.title('Filtered Signal Using BandPass Filter around 1KHz')
	plt.legend(['Mic1', 'Mic2', 'Mic3','Mic4'])
	plt.show()

	t, m1, m2, m3, m4 = interpolate(times, m1, m2, m3, m4, 100)


	testm1 = m1[600:1000]
	testm2 = m2[600:1000]
	testm3 = m3[600:1000]
	testm4 = m4[600:1000]
	m1val = testm1[2]
	m2val = testm2[2]
	m3val = testm3[2]
	m4val = testm4[2]

	cor1, cor2, cor3, cor4 = correlate(testm3, testm1, testm2, testm4)
	t1, t2, t3, t4 = get_taus(cor1, cor2, cor3, cor4, sampleRate)

	deltat = np.array([t1, t2, t3, t4])

	c = np.argmin(deltat)
	cTime = deltat[c]
	

	#calculate the difference in time relative to the shortest time
	ddt = np.array([ dt - cTime for dt in deltat ])
	location = tdoa(mics, ddt, temperature)

	return location


def interpolate(time, m1, m2, m3, m4, factor):

	x = np.linspace(0,len(m1),len(m1))

	inter_factor = np.linspace(0,len(m1),len(m1)*factor) #sets up the amount of interpolation to be done

	t = interp1d(x,time)
	f1 = interp1d(x,m1)
	f2 = interp1d(x,m2)
	f3 = interp1d(x,m3)
	f4 = interp1d(x,m4)

	return t(inter_factor), f1(inter_factor), f2(inter_factor), f3(inter_factor), f4(inter_factor)

def run_filter(m1,m2,m3,m4, f1, f2, fs):

	nyq = 0.5 * fs
	low = f1 / nyq
	high = f2 / nyq
	b,a = sig.butter(6, [low, high], 'band')
	mic1 = sig.lfilter(b,a,m1)
	mic2 = sig.lfilter(b,a,m2)
	mic3 = sig.lfilter(b,a,m3)
	mic4 = sig.lfilter(b,a,m4)

	return mic1, mic2, mic3, mic4


def normalize(m1, m2, m3, m4):

	norm1 = m1 - np.nanmean(m1)
	norm2 = m2 - np.nanmean(m2)
	norm3 = m3 - np.nanmean(m3)
	norm4 = m4 - np.nanmean(m4)

	return norm1, norm2, norm3, norm4
	
def correlate(m1, m2, m3, m4):

	##----------------------------
	#Assumes that m1 is the reference microphone
	##----------------------------

	cor1 = np.correlate(m1,m1, "full")
	cor2 = np.correlate(m2,m1, "full")
	cor3 = np.correlate(m3,m1, "full")
	cor4 = np.correlate(m4,m1, "full")


	return cor1, cor2, cor3, cor4

def get_taus(cor1, cor2, cor3, cor4, sampleRate):

	c1 = np.argmax(np.absolute(cor1))
	c2 = np.argmax(np.absolute(cor2))
	c3 = np.argmax(np.absolute(cor3))
	c4 = np.argmax(np.absolute(cor4))

	tau1 = (cor1[c1]/sampleRate)*np.power(10.0,-6)
	tau2 = (cor2[c2]/sampleRate)*np.power(10.0,-6)
	tau3 = (cor3[c3]/sampleRate)*np.power(10.0,-6)
	tau4 = (cor4[c4]/sampleRate)*np.power(10.0,-6)

	return tau1, tau2, tau3, tau4

def tdoa(mics, dt, temperature):
	#speed of sound in medium
	v = (331.3+(0.606*temperature))*1000
	nSensor = 4

	t = dt
	p = mics
	c = np.argmin(t)

	ijs = range(nSensor)
	del ijs[c]

	A = np.zeros([nSensor-1,2])
	b = np.zeros([nSensor-1,1])
	iRow = 0
	rankA = 0
	for i in ijs:
		for j in ijs:
			A[iRow,:] = 2*( v*(t[j])*(p[:,i]-p[:,c]).T - v*(t[i])*(p[:,j]-p[:,c]).T )
			b[iRow,0] = v*(t[i])*(v*v*(t[j])**2-p[:,j].T*p[:,j]) + \
			(v*(t[i])-v*(t[j]))*p[:,c].T*p[:,c] + \
			v*(t[j])*(p[:,i].T*p[:,i]-v*v*(t[i])**2)
			rankA = matrix_rank(A)
			if rankA >= 2 :
				break
			iRow += 1
		if rankA >= 2:
			break

	calculatedLocation = np.asarray( lstsq(A,b)[0] )[:,0]

	return calculatedLocation

def micfunc(x):

	e2 = np.linalg.norm((m2 - x),ord=2) - dd2
	e3 = np.linalg.norm((m3 - x),ord=2) - dd3
	e4 = np.linalg.norm((m4 - x),ord=2) - dd4
 
	sq_err = (np.power(e2,2) + np.power(e3,2) + np.power(e4,2))
	return m.sqrt(sq_err)
	# minimum search

def Optimize(m1, m2, m3, m4, t1, t2, t3, t4):

	c = float((331+(0.610*25)) * 100)
	dd2 = c*t2
	dd3 = c*t3
	dd4 = c*t4

	guess = np.array([5*(np.random.random() - 1),  5*np.random.random()])
 
	results = optimize.fmin(func=micfunc,x0 = guess)

	return results