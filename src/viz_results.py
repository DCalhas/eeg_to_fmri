import sys

import iterative_naive_nas

from sklearn.model_selection import train_test_split

import tensorflow.compat.v1 as tf

import numpy as np

import custom_training

import utils.losses_utils as losses

import utils.data_utils as data_utils

import matplotlib.pyplot as plt





def get_models_and_shapes(eeg_file='../../optimized_nets/eeg/eeg_30_partitions.json', 
						bold_file='../../optimized_nets/bold/bold_30_partitions.json',
						decoder_file='../../optimized_nets/decoder/decoder_30_partitions.json'):

	json_file = open(eeg_file, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	eeg_network = tf.keras.models.model_from_json(loaded_model_json)

	json_file = open(bold_file, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	bold_network = tf.keras.models.model_from_json(loaded_model_json)

	json_file = open(decoder_file, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	decoder_network = tf.keras.models.model_from_json(loaded_model_json)

	return eeg_network, bold_network, decoder_network


def _plot_mean_std(reconstruction_loss, distance, tset="train", n_partitions=30, model="M", ax=None):

	inds_ids = []
	inds_mean = np.zeros(len(reconstruction_loss)//n_partitions)
	inds_std = np.zeros(len(reconstruction_loss)//n_partitions)

	#compute mean 
	for ind in range(inds_mean.shape[0]):
		inds_ids += ['Ind_' + str(ind+1)]
		inds_mean[ind] = np.mean(reconstruction_loss[ind:ind+n_partitions])
		inds_std[ind] = np.std(reconstruction_loss[ind:ind+n_partitions])

	print(tset + " set", "mean: ", np.mean(reconstruction_loss))
	print(tset + " set", "std: ", np.std(reconstruction_loss))


	ax.errorbar(inds_ids, inds_mean, inds_std, linestyle='None', elinewidth=0.5, ecolor='r', capsize=10.0, markersize=10.0, marker='o')
	ax.set_title(distance + " on " + tset + " set " + " (" + model + ")")
	ax.set_xlabel("Individuals")
	if("Cosine" in distance):
		ax.set_ylabel("Correlation")
	else:
		ax.set_ylabel("Distance")

def _plot_mean_std_loss(synthesized_bold, bold, distance_function, distance_name, set_name, model_name, ax=None):
	reconstruction_loss = np.zeros((synthesized_bold.shape[0], 1))

	for instance in range(len(reconstruction_loss)):
		instance_synth = synthesized_bold[instance]
		instance_bold = bold[instance]

		instance_synth = instance_synth.reshape((1, instance_synth.shape[0], instance_synth.shape[1], instance_synth.shape[2]))
		instance_bold = instance_bold.reshape((1, instance_bold.shape[0], instance_bold.shape[1], instance_bold.shape[2]))

		reconstruction_loss[instance] = distance_function(instance_synth, instance_bold).numpy()

	_plot_mean_std(reconstruction_loss, distance=distance_name, tset=set_name, model=model_name, ax=ax)



def plot_mean_std_loss(eeg_train, bold_train, 
						eeg_val, bold_val, 
						eeg_test, bold_test, 
						encoder_network, decoder_network, 
						distance_name, distance_function,
						model_name):

	plt.figure(figsize=(20,5))
	ax1 = plt.subplot(131)

	shared_eeg_train = encoder_network.predict(eeg_train)
	synthesized_bold_train = decoder_network.predict(shared_eeg_train)
	_plot_mean_std_loss(synthesized_bold_train, bold_train, distance_function, distance_name, "train", model_name, ax=ax1)

	ax2 = plt.subplot(132)

	shared_eeg_val = encoder_network.predict(eeg_val)
	synthesized_bold_val = decoder_network.predict(shared_eeg_val)
	_plot_mean_std_loss(synthesized_bold_val, bold_val, distance_function, distance_name, "validation", model_name, ax=ax2)

	ax3 = plt.subplot(133)
	shared_eeg_test = encoder_network.predict(eeg_test)
	synthesized_bold_test = decoder_network.predict(shared_eeg_test)
	_plot_mean_std_loss(synthesized_bold_test, bold_test, distance_function, distance_name, "test", model_name, ax=ax3)

	plt.show()

def plot_loss_results(eeg_train, bold_train, eeg_val, bold_val, eeg_test, bold_test, eeg_network, decoder_network, model_name):

	plot_mean_std_loss(eeg_train, bold_train, 
	eeg_val, bold_val, 
	eeg_test, bold_test, 
	eeg_network, decoder_network, 
	"Cosine", losses.get_reconstruction_loss,
	model_name)

	plot_mean_std_loss(eeg_train, bold_train, 
	eeg_val, bold_val, 
	eeg_test, bold_test, 
	eeg_network, decoder_network, 
	"Euclidean", euclidean,
	model_name)