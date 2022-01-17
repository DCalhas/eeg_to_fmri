import tensorflow as tf

from utils import data_utils, eeg_utils

import sys

import numpy as np

#should eeg_limit be true??
def dataset(dataset, n_individuals=8, interval_eeg=6, ind_volume_fit=True, raw_eeg=False, eeg_limit=True, eeg_f_limit=134, standardize_fmri=True, standardize_eeg=True, iqr=True, file_output=None, verbose=False):

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
															eeg_limit=eeg_limit, eeg_f_limit=eeg_f_limit,
															dataset=dataset)
	eeg_channels=eeg_train.shape[1]

	if(dataset=="01"):
		n_individuals_train = 8
		n_individuals_test = 2
		n_volumes = 300-3
	elif(dataset=="02"):
		n_individuals_train = 8
		n_individuals_test = 2
		n_volumes = 170-3#?
	elif(dataset=="03"):
		n_individuals_train = 16
		n_individuals_test = 4
		n_volumes = 373-3#?

	if(raw_eeg):
		eeg_test = eeg_train[n_individuals_train*(n_volumes)*int(f_resample*getattr(eeg_utils, "fs_"+dataset)):(n_individuals_train+n_individuals_test)*n_volumes*int(f_resample*getattr(eeg_utils, "fs_"+dataset))]
		eeg_train = eeg_train[:n_individuals_train*n_volumes*int(f_resample*getattr(eeg_utils, "fs_"+dataset))]
	else:
		eeg_test = eeg_train[n_individuals_train*n_volumes:(n_individuals_train+n_individuals_test)*n_volumes]
		eeg_train = eeg_train[:n_individuals_train*n_volumes]

	fmri_test = fmri_train[n_individuals_train*n_volumes:(n_individuals_train+n_individuals_test)*n_volumes]
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
	eeg_test, fmri_test = data_utils.create_eeg_bold_pairs(eeg_test, fmri_test, 
															raw_eeg=raw_eeg,
															fs_sample_eeg=getattr(eeg_utils, "fs_"+dataset),
															fs_sample_fmri=f_resample,
															interval_eeg=interval_eeg, 
															n_volumes=n_volumes, 
															n_individuals=n_individuals_test,
															instances_per_individual=25)

	eeg_train = np.expand_dims(eeg_train, axis=-1)
	fmri_train = np.expand_dims(fmri_train, axis=-1)
	eeg_test = np.expand_dims(eeg_test, axis=-1)
	fmri_test = np.expand_dims(fmri_test, axis=-1)

	eeg_train = eeg_train.astype('float32')
	fmri_train = fmri_train.astype('float32')
	eeg_test = eeg_test.astype('float32')
	fmri_test = fmri_test.astype('float32')

	if(verbose):
		if(file_output == None):
			print("I: Pairs Created")
		else:
			print("I: Pairs Created", file=file_output)

	return (eeg_train, fmri_train), (eeg_test, fmri_test)