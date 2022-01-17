import mne

import matplotlib.pyplot as plt

from scipy.fft import fft
from scipy.io import loadmat
from scipy.signal import cwt, ricker
from scipy import signal as scipy_signal

import numpy as np

import os
from os import listdir
from os.path import isfile, join, isdir
from pathlib import Path

home = str(Path.home())

dataset_path = home + '/eeg_to_fmri'

channels_01=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz', 'ECG', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FT9', 'FT10', 'P9', 'P10', 'PO9', 'PO10', 'O9', 'O10', 'Fpz', 'CPz']
channels_02=["C3-T7","T7-LM","LM-CP5","CP5-P7","P7-PO7","PO7-PO3","PO3-O1","O1-Oz","PO3-P3","P3-CP1","Pz-CP1","CP1-C3","Cz-C3","Fp2-Fp1","Fp1-AF3","AF4-Fp2","AF3-F3","F4-AF4","F3-F7","F8-F4","FC1-F3","F4-FC2","F7-FC5","FC6-F8","FC5-T7","T8-FC6","Cz-Fz","Fz-FC1","FC2-Fz","T8-C4","RM-T8","CP6-RM","P8-CP6","PO8-P8","PO4-PO8","O2-PO4","Oz-O2","P4-PO4","CP2-P4","CP2-Pz","C4-CP2","C4-Cz","Pz-Oz"]
channels_03=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz', 'ECG', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8',  'FT9', 'FT10', 'Fpz', 'CPz']

channels_coords_10_20= {"Fpz": (0.5,0.9),"Fp1": (0.40,0.88),"Fp2": (0.6,0.88),"AFz": (0.5,0.8),"AF3": (0.43,0.79),"AF4": (0.57,0.79),"AF7": (0.29,0.83),"AF8": (0.71,0.83),"Fz": (0.5,0.7),"F1": (0.41,0.7),"F2": (0.59,0.7),"F3": (0.32,0.71),"F4": (0.68,0.71),"F5": (0.25,0.725),"F6": (0.75,0.725),"F7": (0.19,0.74),"F8": (0.81,0.74),"F9": (0.12,0.78),"F10": (0.88,0.78),"FCz": (0.5,0.6),"FC1": (0.4,0.6),"FC2": (0.6,0.6),"FC3": (0.3,0.605),"FC4": (0.7,0.605),"FC5": (0.22,0.615),"FC6": (0.78,0.615),"FT7": (0.13,0.63),"FT8": (0.87,0.63),"FT9": (0.05,0.655),"FT10": (0.95,0.655),"Cz": (0.5,0.5),"C1": (0.4,0.5),"C2": (0.6,0.5),"C3": (0.3,0.5),"C4": (0.7,0.5),"C5": (0.2,0.5),"C6": (0.8,0.5),"T7": (0.1,0.5),"T8": (0.9,0.5),"CPz": (0.5,0.4),"CP1": (0.4,0.4),"CP2": (0.6,0.4),"CP3": (0.3,0.395),"CP4": (0.7,0.395),"CP5": (0.22,0.385),"CP6": (0.78,0.385),"TP7": (0.13,0.37),"TP8": (0.87,0.37),"TP9": (0.05,0.345),"TP10": (0.95,0.345),"Pz": (0.5,0.3),"P1": (0.41,0.3),"P2": (0.59,0.3),"P3": (0.32,0.29),"P4": (0.68,0.29),"P5": (0.25,0.275),"P6": (0.75,0.275),"P7": (0.19,0.26),"P8": (0.81,0.26),"P9": (0.12,0.22),"P10": (0.88,0.22),"POz": (0.5,0.2),"PO3": (0.43,0.21),"PO4": (0.57,0.21),"PO7": (0.29,0.17),"PO8": (0.71,0.17),"PO9": (0.25,0.1),"PO10": (0.75,0.1),"Oz": (0.5,0.1),"O1": (0.40,0.12),"O2": (0.6,0.12),"O9": (0.37,0.05),"O10": (0.63,0.05)}

#frequency samples of each EEG dataset
fs_01=250
fs_02=1000
fs_03=5000

media_directory="/mnt/datasets/"
dataset_03="ds002158"

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
    
    return eeg_file['data_noGA'][:43,:]


def get_eeg_instance_03(individual, path_eeg=media_directory+dataset_03+"/", run="main_run-001", preprocessed=False):
    
    run_types=["main_run-001", "main_run-002",
              "main_run-003", "main_run-004",
              "main_run-005", "main_run-006"]
    
    assert run in run_types, dataset_03+ " contains the following recording sessions: " + str(run_types) + ", please select one."
    assert not preprocessed, "Preprocessed EEG signal is not available, only EEG events"
    
    individuals = sorted([f for f in listdir(path_eeg) if isdir(join(path_eeg, f))])[2:]

    individual = individuals[individual]

    if(preprocessed):
        path = path_eeg + "derivatives/eegprep/" + individual + "/ses-001/eeg/" + individual + "_ses-001_task-main_eeg_preproc.set"
        return mne.io.read_epochs_eeglab(path)
    else:
        path = path_eeg + individual + "/ses-001/eeg/"

        brainvision_files = sorted([f for f in listdir(path) if isfile(join(path, f))])

        vhdr_file = brainvision_files[1]

        complete_path = path + vhdr_file

        return mne.io.read_raw_brainvision(complete_path, preload=True, verbose=0)

        
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

def raw_eeg(eeg, channel=0, fs=250):
	signal = eeg[channel][:]
	if(type(signal) is tuple):
		signal, _ = signal
		signal = signal.reshape((signal.shape[1]))
	else:
		signal = signal.reshape((signal.shape[0]))

	return signal

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


def dwt(eeg, channel=0, windows=30, fs=2.0, start_time=None, stop_time=None):

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


	return cwt(signal, scipy_signal.morlet2, np.arange(1, windows))


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
	f = np.linspace (0, fs, N//2)
	
	plt.figure(1)
	plt.plot (f[1:max_freq], abs (fft1)[1:max_freq])
	plt.title ('Magnitude of each frequency')
	plt.xlabel ('Frequency (Hz)')
	plt.show()


def plot_stft(eeg, channel=2, window_size=2, min_freq=None, max_freq=None, colorbar=True):
	f, Zxx, t = stft(eeg, channel=channel, fs=eeg.info['sfreq'], window_size=window_size)
	
	if(min_freq == None):
		min_freq = 0
	if(max_freq == None):
		max_freq = len(Zxx)

	Zxx = Zxx[min_freq:max_freq]
	f = f[min_freq:max_freq]

	amplitude = np.max(Zxx)

	fig, axes = plt.subplots(1,1)

	im = axes.pcolormesh(t, f, abs(Zxx), vmin=0, vmax=amplitude)

	if(colorbar):
		fig.colorbar(im)

	fig.show()

	return fig, axes