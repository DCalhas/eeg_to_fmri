import matplotlib.pyplot as plt
from nilearn.masking import apply_mask, compute_epi_mask
from nilearn.image import smooth_img, index_img, iter_img, clean_img, math_img, mean_img
from nilearn import plotting
from nilearn import image
from nilearn.input_data import NiftiMasker


import numpy as np

import os
from os import listdir
from os.path import isfile, join, isdir

from scipy.signal import resample


##########################################################################################################################
#
#											READING UTILS
#			
##########################################################################################################################
def get_fmri_instance(individual=0, path_fmri='/home/davidcalhas/eeg_to_fmri/datasets/01/fMRI/'):

	individuals = sorted([f for f in listdir(path_fmri) if isdir(join(path_fmri, f))])

	individual = individuals[individual]

	fmri_file = '/3_nw_mepi_rest_with_cross.nii.gz'

	complete_path = path_fmri + individual + fmri_file

	mask_img = compute_epi_mask(complete_path)

	return apply_mask(complete_path, mask_img)


def get_fmri_instance_img(individual=0, path_fmri='/home/davidcalhas/eeg_to_fmri/datasets/01/fMRI/'):

	individuals = sorted([f for f in listdir(path_fmri) if isdir(join(path_fmri, f))])

	individual = individuals[individual]

	fmri_file = '/3_nw_mepi_rest_with_cross.nii.gz'

	complete_path = path_fmri + individual + fmri_file

	return image.load_img(complete_path)

def get_population_mask(path_fmri='/home/davidcalhas/eeg_to_fmri/datasets/01/fMRI/'):

    individuals = sorted([f for f in listdir(path_fmri) if isdir(join(path_fmri, f))])

    individuals_images = []
    
    target_affine = image.load_img(path_fmri + individuals[0] + '/3_nw_mepi_rest_with_cross.nii.gz').affine
    target_shape = image.load_img(path_fmri + individuals[0] + '/3_nw_mepi_rest_with_cross.nii.gz').shape
    target_shape = (target_shape[0], target_shape[1], target_shape[2])
    
    for individual in individuals:
        fmri_file = '/3_nw_mepi_rest_with_cross.nii.gz'
        individual_path = path_fmri + individual + fmri_file
        
        if(image.load_img(individual_path).affine[0][-1] != 0.0):
            
            fmri_image = image.resample_img(image.load_img(individual_path), target_affine=target_affine, target_shape=target_shape)
            
            individuals_images += [fmri_image]


    concatenated_imgs = image.concat_imgs(individuals_images)

    return NiftiMasker(mask_strategy='epi', standardize=True).fit(concatenated_imgs)



def get_individuals_ids(path_fmri='/home/davidcalhas/eeg_to_fmri/datasets/01/fMRI/'):

	individuals = sorted([f for f in listdir(path_fmri) if isdir(join(path_fmri, f))])

	return individuals


##########################################################################################################################
#
#											FMRI UTILS
#			
##########################################################################################################################
# masked_data shape is (timepoints, voxels). We can plot the first 150
# timepoints from two voxels
def get_voxel(masked_fmri, voxel=0):
	return masked_fmri[:masked_fmri.shape[0], voxel]


def get_resampled_bold(voxel, new_TR=2, TR=2.160):
	return resample(voxel, int((len(voxel)*(1/new_TR))/TR))

def get_masked_epi(fmri_instance, masker=None):
	if(masker == None):
		masker = NiftiMasker(mask_strategy='epi', standardize=True)
		masker.fit(fmri_instance)
	return masker.transform(fmri_instance), masker

def get_inverse_masked_epi(fmri_masked, masker):
	return masker.inverse_transform(fmri_masked)
