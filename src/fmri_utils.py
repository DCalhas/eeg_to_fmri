import matplotlib.pyplot as plt

from nilearn import plotting
from nilearn import image
from nilearn import _utils
from nilearn.input_data import NiftiMasker
from nilearn.decomposition import CanICA
from nilearn.masking import apply_mask, compute_epi_mask, compute_multi_epi_mask, _apply_mask_fmri
from nilearn.image import smooth_img, index_img, iter_img, clean_img, math_img, mean_img, new_img_like

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
def get_fmri_instance(individual=0, path_fmri='/home/david/eeg_informed_fmri/datasets/01/fMRI/'):

	individuals = sorted([f for f in listdir(path_fmri) if isdir(join(path_fmri, f))])

	individual = individuals[individual]

	fmri_file = '/3_nw_mepi_rest_with_cross.nii.gz'

	complete_path = path_fmri + individual + fmri_file

	mask_img = compute_epi_mask(complete_path)

	return apply_mask(complete_path, mask_img)


def get_fmri_instance_img(individual=0, path_fmri='/home/david/eeg_informed_fmri/datasets/01/fMRI/'):

	individuals = sorted([f for f in listdir(path_fmri) if isdir(join(path_fmri, f))])

	individual = individuals[individual]

	fmri_file = '/3_nw_mepi_rest_with_cross.nii.gz'

	complete_path = path_fmri + individual + fmri_file

	return image.load_img(complete_path)

def get_population_mask(path_fmri='/home/david/eeg_informed_fmri/datasets/01/fMRI/'):

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

    return NiftiMasker(compute_multi_epi_mask(individuals_images), standardize=True).fit(concatenated_imgs)



def get_individuals_ids(path_fmri='/home/david/eeg_informed_fmri/datasets/01/fMRI/'):

	individuals = sorted([f for f in listdir(path_fmri) if isdir(join(path_fmri, f))])

	return individuals


def get_individuals_paths(path_fmri='/home/david/eeg_informed_fmri/datasets/01/fMRI/'):

	individuals = sorted([f for f in listdir(path_fmri) if isdir(join(path_fmri, f))])

	target_affine = image.load_img(path_fmri + individuals[0] + '/3_nw_mepi_rest_with_cross.nii.gz').affine
	target_shape = image.load_img(path_fmri + individuals[0] + '/3_nw_mepi_rest_with_cross.nii.gz').shape
	target_shape = (target_shape[0], target_shape[1], target_shape[2])

	for i in range(len(individuals)):
		individual = individuals[i]

		fmri_file = '/3_nw_mepi_rest_with_cross.nii.gz'

		individual_path = path_fmri + individual + fmri_file

		fmri_image = image.resample_img(image.load_img(individual_path), target_affine=target_affine, target_shape=target_shape)

		individuals[i] = fmri_image

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


##########################################################################################################################
#
#											EXTRACTION OF ROI TIME SERIES
#			
##########################################################################################################################
###### Canonical ICA

#when its a population of n individuals
#imgs=[complete_path_ind_1, complete_path_ind_2, ..., complete_path_ind_n]
def _apply_mask(imgs, mask_img):
	mask_img = _utils.check_niimg_3d(mask_img)

	mask_img = _utils.check_niimg_3d(mask_img)
	mask = mask_img.get_data()
	mask = _utils.as_ndarray(mask, dtype=bool)

	mask_img = new_img_like(mask_img, mask, mask_img.affine)

	return _apply_mask_fmri(imgs, mask_img, dtype='f', smoothing_fwhm=None, ensure_finite=True)


class roi_time_series:
	def __init__(self, canica=None):
		self.canica = None

	def _set_ICA(self, imgs, n_components=20, verbose=0):
		self.canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
							memory="nilearn_cache", memory_level=2,
							threshold=3., verbose=verbose, random_state=0)
		self._fit_ICA(imgs)

	def _fit_ICA(self, imgs):
		self.canica.fit(imgs)

	def get_ROI_time_series(self, imgs, component=0, n_components=20, verbose=False):

		#smooth image
		fmri_original = image.load_img(imgs)
		fmri_img = image.smooth_img(fmri_original, fwhm=6)

		#perform ICA and get components
		if(self.canica == None):
			if(verbose):
				print("New ICA computation")
			self._set_ICA(fmri_img, n_components=n_components)

		components_img = self.canica.components_img_

		#build masker
		roi_masker = NiftiMasker(mask_img=image.index_img(components_img, component),
								standardize=True,
								memory="nilearn_cache",
								smoothing_fwhm=8)

		return _apply_mask(imgs, roi_masker.mask_img) 
