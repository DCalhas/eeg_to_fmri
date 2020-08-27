import mne

import matplotlib.pyplot as plt

from scipy import fft
from scipy.io import loadmat
import numpy as np

import os
from os import listdir
from os.path import isfile, join, isdir
from pathlib import Path

home = str(Path.home())

dataset_path = home + '/eeg_to_fmri'

##########################################################################################################################
#
#											READING UTILS
#			
##########################################################################################################################
def get_eeg_instance_01(individual, path_eeg=dataset_path+'/datasets/01/EEG/', preprocessed=True):

	individuals = sorted([f for f in listdir(path_eeg) if isdir(join(path_eeg, f))])

	individual = individuals[individual]

	if(preprocessed):
		path = path_eeg + individual + '/export/'
	else:
		path = path_eeg + individual + '/raw/'


	brainvision_files = sorted([f for f in listdir(path) if isfile(join(path, f))])

	vhdr_file = brainvision_files[1]

	complete_path = path + vhdr_file

	return mne.io.read_raw_brainvision(complete_path, preload=True, verbose=0)


def get_eeg_instance_02(individual, task=0, run=0, total_runs=3, preprocessed=True, path_eeg=dataset_path+'/datasets/02'):

    individuals = sorted([f for f in listdir(path_eeg) if isdir(join(path_eeg, f))])
    
    path_eeg = path_eeg + '/' + individuals[individual] + '/EEG'
    
    runs = sorted([f for f in listdir(path_eeg) if isdir(join(path_eeg, f))])
    
    run = runs[task*total_runs+run]

    if(preprocessed):
        path = path_eeg + '/' + run + '/EEG_noGA.mat'
    else:
        path = path_eeg + '/' + run + '/EEG_noGA.mat'

    eeg_file = loadmat(path)
    
    return eeg_file['data_noGA']

def get_eeg_dataset(number_individuals=16, path_eeg=dataset_path+'/datasets/01/EEG/', preprocessed=True):
	individuals = []

	for i in range(number_individuals):
		individuals += [get_individual(i, path_eeg=path_eeg, preprocessed=True)]

	return individuals


##########################################################################################################################
#
#											FREQUENCY UTILS
#			
##########################################################################################################################
frequency_bands = {'delta': [0.5,4], 'theta': [4,8], 'alpha': [8,13], 'beta': [13,30], 'gamma': [30, 100]}


def compute_fft(channel, fs=128):
	N = int(len(channel)/2)

	fft1 = fft(channel)

	return fft1[range(int(N/2))]

def stft(eeg, channel=0, window_size=2, fs=250, start_time=None, stop_time=None):
	signal = eeg[channel][:]
	if(type(signal) is tuple):
		signal, _ = signal
		signal = signal.reshape((signal.shape[1]))
	else:
		signal = signal.reshape((signal.shape[0]))


	if(start_time == None):
		start_time = 0
	if(stop_time == None):
		stop_time = len(signal)
	signal = signal[start_time:stop_time]

	t = []



	fs_window_size = int(window_size*fs)


	Z = []
	seconds = 0
	for time in range(start_time, stop_time, fs_window_size)[:-1]:
		fft1 = compute_fft(signal[time:time+fs_window_size], fs=fs)

		N = len(signal[time:time+fs_window_size])/2
		f = np.linspace (0, len(fft1), int(N/2))

		#average
		Z += [list(abs(fft1[1:]))]
		t += [seconds]
		seconds += window_size

	return f[1:], np.transpose(np.array(Z)), t

def mutate_stft_to_bands(Zxx, frequencies, timesteps):
	#frequency first dimension, time is second dimension
	Z_band_mutated = []

	for t in range(len(timesteps)):

		intensities = []

		for i in range(len(frequency_bands.keys())):
			intensities += [0]

		bands = dict(zip(frequency_bands.keys(), intensities))

		for f in range(len(frequencies)):
			for band in bands.keys():
				if(frequencies[f] <= frequency_bands[band][1]):
					bands[band] += Zxx[f][t]
					break

		Z_band_mutated += [list(bands.values()).copy()]

	return np.transpose(np.array(Z_band_mutated))

##########################################################################################################################
#
#											VISUALIZATION UTILS
#			
##########################################################################################################################
def plot_fft(eeg, channel=0, max_freq=30000, start_time=None, stop_time=None):
	y, _ = eeg[channel][:]
	y = y[0]
	
	fs = eeg.info['sfreq']

	if(start_time == None):
		start_time = 0
	if(stop_time == None):
		stop_time = len(y)
	
	fft1 = compute_fft(y[start_time:stop_time], fs=fs)
	
	N = int(len(y[start_time:stop_time])/2)
	f = np.linspace (0, fs, N/2)
	
	plt.figure(1)
	plt.plot (f[1:max_freq], abs (fft1)[1:max_freq])
	plt.title ('Magnitude of each frequency')
	plt.xlabel ('Frequency (Hz)')
	plt.show()


def plot_stft(eeg, channel=2, window_size=2, min_freq=None, max_freq=None):
	f, Zxx, t = stft(eeg, channel=channel, fs=eeg.info['sfreq'], window_size=window_size)
	
	if(min_freq == None):
		min_freq = 0
	if(max_freq == None):
		max_freq = len(Zxx)

	Zxx = Zxx[min_freq:max_freq]
	f = f[min_freq:max_freq]

	amplitude = np.max(Zxx)

	im = plt.pcolormesh(t, f, abs(Zxx), vmin=0, vmax=amplitude)

	plt.colorbar(im)
	plt.show()
