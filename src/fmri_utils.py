import matplotlib.pyplot as plt
from nilearn.masking import apply_mask, compute_epi_mask
import numpy as np

import os
from os import listdir
from os.path import isfile, join, isdir


##########################################################################################################################
#
#											READING UTILS
#			
##########################################################################################################################
def get_fmri_instance(individual=0, path_fmri='/home/david/eeg_informed_fmri/datasets/01/fMRI/'):

	individuals = sorted([f for f in listdir(path_fmri) if isdir(join(path_fmri, f))])

	individual = individuals[individual]

	fmri_file = '/3_nw_mepi_rest_with_cross.nii.gz'

	complete_path = path_fmri + individual + fmri_file

	mask_img = compute_epi_mask(complete_path)

	return apply_mask(complete_path, mask_img)


##########################################################################################################################
#
#											FMRI UTILS
#			
##########################################################################################################################
# masked_data shape is (timepoints, voxels). We can plot the first 150
# timepoints from two voxels
def get_voxel(masked_fmri, voxel=0):
	return masked_fmri[:masked_fmri.shape[0], voxel]