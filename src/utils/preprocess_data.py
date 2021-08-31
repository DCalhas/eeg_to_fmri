import tensorflow as tf

from utils import data_utils, eeg_utils

import sys

import numpy as np

def dataset(dataset, n_individuals=8, interval_eeg=6, ind_volume_fit=True, raw_eeg=False, standardize_fmri=True, standardize_eeg=True, iqr=True, file_output=None, verbose=False):

	if(verbose):
		if(file_output == None):
			print("I: Starting to Load Data")	
		else:
			print("I: Starting to Load Data", file=file_output)

	#TR of fmri and window size of STFT
	f_resample=2.160

	eeg_train, fmri_train, scalers = data_utils.load_data(list(range(n_individuals)), raw_eeg=raw_eeg, n_voxels=None, 
															bold_shift=3, n_partitions=25, 
															mutate_bands=False,
															by_partitions=False, partition_length=14, 
															f_resample=f_resample, fmri_resolution_factor=1, 
															standardize_eeg=standardize_eeg, standardize_fmri=standardize_fmri,
															ind_volume_fit=ind_volume_fit, iqr_outlier=iqr,
															dataset=dataset)
	eeg_channels=eeg_train.shape[1]

	if(dataset=="01"):
		n_individuals_train = 6
		n_individuals_val = 2
		n_volumes = 300-3
	elif(dataset=="02"):
		n_individuals_train = 8
		n_individuals_val = 2
		n_volumes = 170-3#?

	if(raw_eeg):
		eeg_val = eeg_train[n_individuals_train*(n_volumes)*int(f_resample*getattr(eeg_utils, "fs_"+dataset)):(n_individuals_train+n_individuals_val)*n_volumes*int(f_resample*getattr(eeg_utils, "fs_"+dataset))]
		eeg_train = eeg_train[:n_individuals_train*n_volumes*int(f_resample*getattr(eeg_utils, "fs_"+dataset))]
	else:
		eeg_val = eeg_train[n_individuals_train*n_volumes:(n_individuals_train+n_individuals_val)*n_volumes]
		eeg_train = eeg_train[:n_individuals_train*n_volumes]

	fmri_val = fmri_train[n_individuals_train*n_volumes:(n_individuals_train+n_individuals_val)*n_volumes]
	fmri_train = fmri_train[:n_individuals_train*n_volumes]

	if(verbose):
		if(file_output==None):
			print("I: Finished Loading Data")
		else:
			print("I: Finished Loading Data", file=file_output)

	eeg_train, fmri_train = data_utils.create_eeg_bold_pairs(eeg_train, fmri_train, 
															raw_eeg=raw_eeg,
															fs_sample_eeg=getattr(eeg_utils, "fs_"+dataset),
															fs_sample_fmri=f_resample,
															interval_eeg=interval_eeg, 
															n_volumes=n_volumes, 
															n_individuals=n_individuals_train,
															instances_per_individual=25)
	eeg_val, fmri_val = data_utils.create_eeg_bold_pairs(eeg_val, fmri_val, 
															raw_eeg=raw_eeg,
															fs_sample_eeg=getattr(eeg_utils, "fs_"+dataset),
															fs_sample_fmri=f_resample,
															interval_eeg=interval_eeg, 
															n_volumes=n_volumes, 
															n_individuals=n_individuals_val,
															instances_per_individual=25)

	eeg_train = np.expand_dims(eeg_train, axis=-1)
	fmri_train = np.expand_dims(fmri_train, axis=-1)
	eeg_val = np.expand_dims(eeg_val, axis=-1)
	fmri_val = np.expand_dims(fmri_val, axis=-1)

	eeg_train = eeg_train.astype('float32')
	fmri_train = fmri_train.astype('float32')
	eeg_val = eeg_val.astype('float32')
	fmri_val = fmri_val.astype('float32')

	if(verbose):
		if(file_output == None):
			print("I: Pairs Created")
		else:
			print("I: Pairs Created", file=file_output)

	return (eeg_train, fmri_train), (eeg_val, fmri_val)
