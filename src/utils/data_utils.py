from utils import eeg_utils
from utils import fmri_utils

import numpy as np
from numpy import correlate

import mne
from nilearn.masking import apply_mask, compute_epi_mask
from nilearn import signal

from sklearn.preprocessing import normalize

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from scipy.signal import resample
from scipy.stats import zscore

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

def load_data(train_instances, test_instances, n_voxels=None, bold_shift=3, n_partitions=16, by_partitions=True, partition_length=None, f_resample=2, fmri_resolution_factor=4, standardize_eeg=True, roi=None, roi_ica_components=None):

	#Load Data
	eeg_train, bold_train = get_data(train_instances,
	                                n_voxels=n_voxels, bold_shift=bold_shift, n_partitions=n_partitions, 
	                                by_partitions=by_partitions, partition_length=partition_length,
	                                f_resample=f_resample, fmri_resolution_factor=fmri_resolution_factor,
	                                standardize_eeg=standardize_eeg)
	eeg_test, bold_test = get_data(test_instances,
	                                n_voxels=n_voxels, bold_shift=bold_shift, n_partitions=n_partitions, 
	                                by_partitions=by_partitions, partition_length=partition_length,
	                                f_resample=f_resample, fmri_resolution_factor=fmri_resolution_factor,
	                                standardize_eeg=standardize_eeg)

	if(train_instances):
		eeg_train = eeg_train.reshape(eeg_train.shape[0], eeg_train.shape[1], eeg_train.shape[2], eeg_train.shape[3], 1)
		bold_train = bold_train.reshape(bold_train.shape[0], bold_train.shape[1], bold_train.shape[2], 1)

	if(test_instances):
		eeg_test = eeg_test.reshape(eeg_test.shape[0], eeg_test.shape[1], eeg_test.shape[2], eeg_test.shape[3], 1)
		bold_test = bold_test.reshape(bold_test.shape[0], bold_test.shape[1], bold_test.shape[2], 1)

	return eeg_train, bold_train, eeg_test, bold_test


import sys
sys.path.append('..')
sys.path.append('../..')

from utils import eeg_utils
from utils import fmri_utils

import importlib
importlib.reload(fmri_utils)
importlib.reload(eeg_utils)

from scipy.signal import resample

from nilearn import signal, image

from scipy.stats import zscore

from sklearn.preprocessing import StandardScaler


"""
"""

def get_data_01(individuals, start_cutoff=3, bold_shift=3, n_partitions=16, by_partitions=True, partition_length=None, n_voxels=None, TR=2.160, f_resample=2, fmri_resolution_factor=5, standardize_eeg=True):
    TR = 1/TR

    X = []
    y = []
    fmri_scalers = []


    #setting mask and fMRI signals
    individuals_imgs = fmri_utils.get_individuals_paths(resolution_factor=fmri_resolution_factor, number_individuals=10)
    individuals_imgs, mask = fmri_utils.get_masked_epi(individuals_imgs)
    
    #clean fMRI signal
    for i in range(len(individuals_imgs)):
        individuals_imgs[i] = signal.clean(individuals_imgs[i], 
                                           detrend=True, 
                                           standardize=False, 
                                           low_pass=None, high_pass=0.008, t_r=1/TR)
        
        scaler = StandardScaler(copy=True)
        individuals_imgs[i] = scaler.fit_transform(individuals_imgs[i])
        fmri_scalers += [scaler]
    
    for individual in individuals:
        eeg = eeg_utils.get_eeg_instance(individual)
        x_instance = []
        #eeg
        for channel in range(len(eeg.ch_names)):
            f, Zxx, t = eeg_utils.stft(eeg, channel=channel, window_size=f_resample)
            x_instance += [Zxx]
        
        x_instance = zscore(np.array(x_instance))
        
        fmri_masked_instance = individuals_imgs[individual]

        fmri_resampled = []
        #build resampled BOLD signal
        if(n_voxels == None):
            n_voxels = fmri_masked_instance.shape[1]

        for voxel in range(n_voxels):
            voxel = fmri_utils.get_voxel(fmri_masked_instance, voxel=voxel)
            voxel_resampled = resample(voxel, int((len(voxel)*(1/f_resample))/TR))
            fmri_resampled += [voxel_resampled]

        fmri_resampled = np.array(fmri_resampled)

        if(by_partitions):

            for partition in range(n_partitions):
                start_eeg = start_cutoff + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/n_partitions)*partition
                end_eeg = start_cutoff + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/n_partitions)*partition + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/n_partitions)

                start_bold = start_eeg+bold_shift
                end_bold = end_eeg+bold_shift

                X += [x_instance[:,:,start_eeg:end_eeg]]

                y += list(fmri_resampled[:,start_bold:end_bold].reshape(1, fmri_resampled[:,start_bold:end_bold].shape[0], fmri_resampled[:,start_bold:end_bold].shape[1]))
        else:
            total_partitions = fmri_resampled.shape[1]//partition_length
            for partition in range(total_partitions):

                start_eeg = start_cutoff + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/(total_partitions))*partition
                end_eeg = start_cutoff + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/(total_partitions))*partition + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/(total_partitions))

                start_bold = start_eeg+bold_shift
                end_bold = end_eeg+bold_shift

                X += [x_instance[:,:,start_eeg:end_eeg]]

                y += list(fmri_resampled[:,start_bold:end_bold].reshape(1, fmri_resampled[:,start_bold:end_bold].shape[0], fmri_resampled[:,start_bold:end_bold].shape[1]))

    X = np.array(X)
    y = np.array(y)
    
    print(X.shape)
    print(y.shape)

    return X, y, mask, fmri_scalers




#16 - corresponds to a 20 second length signal with 10 time points
#32 - corresponds to a 10 second length signal with 5 time points
#individuals is a list of indexes until the maximum number of individuals
def get_data_roi(individuals, masker=None, start_cutoff=3, bold_shift=3, n_partitions=16, by_partitions=True, partition_length=None, n_voxels=None, f_resample=2, roi=None, roi_ica_components=None):
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
			f, Zxx, t = eeg_utils.stft(eeg, channel=channel, window_size=f_resample) 
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
			voxel_resampled = resample(voxel, int((len(voxel)*(1/f_resample))/TR))
			fmri_resampled += [voxel_resampled]

		fmri_resampled = np.array(fmri_resampled)

		if(by_partitions):

			for partition in range(n_partitions):
				start_eeg = start_cutoff + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/n_partitions)*partition
				end_eeg = start_cutoff + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/n_partitions)*partition + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/n_partitions)

				start_bold = start_eeg+bold_shift
				end_bold = end_eeg+bold_shift

				X += [x_instance[:,:,start_eeg:end_eeg]]

				y += list(fmri_resampled[:,start_bold:end_bold].reshape(1, fmri_resampled[:,start_bold:end_bold].shape[0], fmri_resampled[:,start_bold:end_bold].shape[1]))
		else:
			total_partitions = fmri_resampled.shape[1]//partition_length
			for partition in range(total_partitions):

				start_eeg = start_cutoff + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/(total_partitions))*partition
				end_eeg = start_cutoff + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/(total_partitions))*partition + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/(total_partitions))

				start_bold = start_eeg+bold_shift
				end_bold = end_eeg+bold_shift

				X += [x_instance[:,:,start_eeg:end_eeg]]

				y += list(fmri_resampled[:,start_bold:end_bold].reshape(1, fmri_resampled[:,start_bold:end_bold].shape[0], fmri_resampled[:,start_bold:end_bold].shape[1]))

		print(np.array(y).shape)

	X = np.array(X)
	y = np.array(y)

	return X, y

def create_eeg_bold_pairs(eeg, bold, instances_per_individual=16):
	x_eeg_indeces = []
	x_bold_indeces_pair = []
	x_bold_indeces_true = []
	y = []

	#how are we going to pair these? only different individuals??
	#different timesteps of the same individual

	#building pairs
	for individual in range(int(len(eeg)/instances_per_individual)):
		for other_individual in range(int(len(eeg)/instances_per_individual)):
			for time_partitions in range(instances_per_individual):
				if(individual == other_individual):
					true_pair = other_individual+time_partitions
					x_eeg_indeces += [[individual + time_partitions]]
					x_bold_indeces_pair += [[other_individual + time_partitions]]
					x_bold_indeces_true += [[true_pair]]
					y += [[1]]
				else:
					x_eeg_indeces += [[individual + time_partitions]]
					x_bold_indeces_pair += [[other_individual + time_partitions]]
					x_bold_indeces_true += [[true_pair]]
					y += [[0]]

	x_eeg_indeces = np.array(x_eeg_indeces)
	x_bold_indeces_pair = np.array(x_bold_indeces_pair)
	x_bold_indeces_true = np.array(x_bold_indeces_true)
	y = np.array(y)

	return x_eeg_indeces, x_bold_indeces_pair, y, x_bold_indeces_true


#############################################################################################################
#
#                                           STANDARDIZE DATA FUNCTION                                       
#
#############################################################################################################

def standardize(eeg, bold, eeg_scaler=None, bold_scaler=None):
    #shape = (n_samples, n_features)
    eeg_reshaped = eeg.reshape((eeg.shape[0], eeg.shape[1]*eeg.shape[2]*eeg.shape[3]*eeg.shape[4]))
    bold_reshaped = bold.reshape((bold.shape[0], bold.shape[1]*bold.shape[2]*bold.shape[3]))
    
    if(eeg_scaler == None):
        eeg_scaler = StandardScaler()
        eeg_scaler.fit(eeg_reshaped)
        
    if(bold_scaler == None):
        bold_scaler = StandardScaler()
        bold_scaler.fit(bold_reshaped)

    eeg_reshaped = eeg_scaler.transform(eeg_reshaped)
    bold_reshaped = bold_scaler.transform(bold_reshaped)

    eeg_reshaped = eeg_reshaped.reshape((eeg.shape))
    bold_reshaped = bold_reshaped.reshape((bold.shape))
    
    return eeg_reshaped, bold_reshaped, eeg_scaler, bold_scaler


"""
inverse_instance_scaler - perform inverse operation to get original fMRI signal of an instance
"""
def inverse_instance_scaler(instance, data_scaler):
    
    instance = np.swapaxes(instance, 0, 1)
    
    instance = data_scaler.inverse_transform(instance)
    
    return np.swapaxes(instance, 0, 1)

"""
inverse_set_scaler - perform inverse operation to get original fMRI signals of a dataset
"""
def inverse_set_scaler(data, data_scalers, n_partitions=25):
    unscaled_data = []
    
    for i in range(len(data)):
        
        scaler_index = i//n_partitions
        
        unscaled_data += [inverse_instance_scaler(data[i], data_scalers[scaler_index])]
        
    return np.array(unscaled_data)