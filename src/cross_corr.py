import eeg_utils
import fmri_utils

import numpy as np
from numpy import correlate

import matplotlib.pyplot as plt

import mne
from nilearn.masking import apply_mask, compute_epi_mask

from sklearn.preprocessing import normalize

import json


def get_correlation_voxel(eeg, fmri_masked_instance, voxel=0, scaled=False, TR=2.160, start_cutoff=3, window_size=2):

	channel_correlations = []

	band_correlations = [0, 0, 0, 0, 0]

	voxel = fmri_utils.get_voxel(fmri_masked_instance, voxel=voxel)

	TR = 1/TR
	voxel_resampled = fmri_utils.get_resampled_bold(voxel)

	for channel in range(len(eeg.ch_names)):
		channel_correlations += [0]

		f, Zxx, t = eeg_utils.stft(eeg, channel=channel, window_size=window_size)

		Zxx = eeg_utils.mutate_stft_to_bands(Zxx, f, t)
		if(scaled):
			Zxx_scaled = min_max_scaler.fit_transform(Zxx)

		for band in range(len(eeg_utils.frequency_bands)):
			if(scaled):
				correlation_coeff = correlate(Zxx_scaled[band][start_cutoff:len(voxel_resampled[:3])], voxel_resampled[start_cutoff:], mode='full')
			else:
				correlation_coeff = correlate(Zxx[band][start_cutoff:len(voxel_resampled[:3])], voxel_resampled[start_cutoff:], mode='full')

			band_correlations[band] += np.max(correlation_coeff)

			channel_correlations[-1] += np.max(correlation_coeff)

	return channel_correlations, band_correlations


if __name__ == "__main__":

	individuals = fmri_utils.get_individuals_ids()

	for individual in range(16):
		
		#get first individual instances
		fmri_masked_instance = fmri_utils.get_fmri_instance(individual)
		eeg = eeg_utils.get_eeg_instance(individual)


		voxel_ids = []
		voxel_corr_coef = []
		for i in range(fmri_masked_instance.shape[1]):
			voxel_ids += [i]
			voxel_corr_coef += [0]

		rank_corr = dict(zip(voxel_ids, voxel_corr_coef))

		#perform correlation
		for voxel_id in voxel_ids:
			channels_corr, bands_corr = get_correlation_voxel(eeg, fmri_masked_instance, voxel=voxel_id, scaled=False)

			rank_corr[voxel_id] += sum(channels_corr)

			if(voxel_id % 500 == 0):
				print("Now on iteration ", voxel_id)

		#save the rank_corr to json
		with open('jsons/' + individuals[individual] + '_rank_correlations.json', 'w') as fp:
			json.dump(rank_corr, fp)

		print("Finished individual " + individuals[individual] + "\n==================================\n\n\n")