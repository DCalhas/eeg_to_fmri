from utils import eeg_utils
from utils import fmri_utils

import numpy as np
from numpy import correlate

import mne
from nilearn.masking import apply_mask, compute_epi_mask

from sklearn.preprocessing import normalize

from scipy.signal import resample

import sys

n_partitions = 16
number_channels = 64
number_individuals = 16
n_epochs = 20


#############################################################################################################
#
#                                           LOAD DATA FUNCTION                                       
#
#############################################################################################################

def load_data(train_instances, test_instances, n_voxels=None, bold_shift=3, roi=None, roi_ica_components=None):

	mask = fmri_utils.get_population_mask()

	#Load Data
	eeg_train, bold_train = get_data(train_instances, masker=mask, 
									n_voxels=n_voxels, bold_shift=bold_shift, roi=roi, 
									roi_ica_components=roi_ica_components)
	eeg_test, bold_test = get_data(test_instances, masker=mask, 
									n_voxels=n_voxels, bold_shift=bold_shift, roi=roi, 
									roi_ica_components=roi_ica_components)

	if(train_instances):
		eeg_train = eeg_train.reshape(eeg_train.shape[0], eeg_train.shape[1], eeg_train.shape[2], eeg_train.shape[3], 1)
		bold_train = bold_train.reshape(bold_train.shape[0], bold_train.shape[1], bold_train.shape[2], 1)

	if(test_instances):
		eeg_test = eeg_test.reshape(eeg_test.shape[0], eeg_test.shape[1], eeg_test.shape[2], eeg_test.shape[3], 1)
		bold_test = bold_test.reshape(bold_test.shape[0], bold_test.shape[1], bold_test.shape[2], 1)

	return eeg_train, bold_train, eeg_test, bold_test




#16 - corresponds to a 20 second length signal with 10 time points
#32 - corresponds to a 10 second length signal with 5 time points
#individuals is a list of indexes until the maximum number of individuals
def get_data(individuals, masker=None, start_cutoff=3, bold_shift=3, n_partitions=16, n_voxels=None, roi=None, roi_ica_components=None):
	TR = 1/2.160

	X = []
	y = []


	#setting ICA
	if(roi != None and roi_ica_components != None):
		individuals_imgs = fmri_utils.get_individuals_paths()
		roi_extraction = fmri_utils.roi_time_series()
		roi_extraction._set_ICA(individuals_imgs, n_components=roi_ica_components)

	for individual in individuals:
		eeg = eeg_utils.get_eeg_instance(individual)
		x_instance = []
		#eeg
		for channel in range(len(eeg.ch_names)):
			f, Zxx, t = eeg_utils.stft(eeg, channel=channel, window_size=2) 
			Zxx_mutated = eeg_utils.mutate_stft_to_bands(Zxx, f, t)

			x_instance += [Zxx_mutated]

		x_instance = np.array(x_instance)

		#fmri
		if(roi != None and roi_ica_components != None):
			fmri_masked_instance = roi_extraction.get_ROI_time_series(individuals_imgs[individual], component=roi)
		else:
			fmri_instance = fmri_utils.get_fmri_instance_img(individual)
			fmri_masked_instance, _ = fmri_utils.get_masked_epi(fmri_instance, masker)

		fmri_resampled = []
		#build resampled BOLD signal
		if(n_voxels == None):
			n_voxels = fmri_masked_instance.shape[1]

		for voxel in range(n_voxels):
			voxel = fmri_utils.get_voxel(fmri_masked_instance, voxel=voxel)
			voxel_resampled = resample(voxel, int((len(voxel)*(1/2))/TR))
			fmri_resampled += [voxel_resampled]

		fmri_resampled = np.array(fmri_resampled)

		for partition in range(n_partitions):
			start_eeg = start_cutoff + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/n_partitions)*partition
			end_eeg = start_cutoff + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/n_partitions)*partition + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/n_partitions)

			start_bold = start_eeg
			end_bold = end_eeg 

			X += [x_instance[:,:,start_eeg:end_eeg]]

			y += list(fmri_resampled[:,start_bold+bold_shift:end_bold+bold_shift].reshape(1, fmri_resampled[:,start_bold+bold_shift:end_bold+bold_shift].shape[0], fmri_resampled[:,start_bold+bold_shift:end_bold+bold_shift].shape[1]))
		print(np.array(y).shape)

	X = np.array(X)
	y = np.array(y)

	return X, y

def create_eeg_bold_pairs(eeg, bold):
	x_eeg = []
	x_bold = []
	y = []

	#how are we going to pair these? only different individuals??
	#different timesteps of the same individual

	#redefine this variable
	instances_per_individual = 16


	#building pairs
	for individual in range(int(len(eeg)/instances_per_individual)):
		for other_individual in range(int(len(eeg)/instances_per_individual)):
			for time_partitions in range(instances_per_individual):
				if(individual == other_individual):
					x_eeg += [eeg[individual + time_partitions]]
					x_bold += [bold[other_individual + time_partitions]]
					y += [[1]]
				else:
					x_eeg += [eeg[individual + time_partitions]]
					x_bold += [bold[other_individual + time_partitions]]
					y += [[0]]

	x_eeg = np.array(x_eeg)
	x_bold = np.array(x_bold)
	y = np.array(y)

	return x_eeg, x_bold, y