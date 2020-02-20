import sys

import iterative_naive_nas

from sklearn.model_selection import train_test_split

import tensorflow.compat.v1 as tf

import numpy as np

import custom_training

import utils.losses_utils as losses

import utils.data_utils as data_utils

import matplotlib.pyplot as plt

from scipy import spatial

from numpy.linalg import norm



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

def _plot_mean_std_loss(synthesized_bold, bold, distance_function, distance_name, set_name, model_name, n_partitions=30, ax=None):
	reconstruction_loss = np.zeros((synthesized_bold.shape[0], 1))

	for instance in range(len(reconstruction_loss)):
		instance_synth = synthesized_bold[instance]
		instance_bold = bold[instance]

		instance_synth = instance_synth.reshape((1, instance_synth.shape[0], instance_synth.shape[1], instance_synth.shape[2]))
		instance_bold = instance_bold.reshape((1, instance_bold.shape[0], instance_bold.shape[1], instance_bold.shape[2]))

		reconstruction_loss[instance] = distance_function(instance_synth, instance_bold).numpy()

	_plot_mean_std(reconstruction_loss, distance=distance_name, tset=set_name, model=model_name, n_partitions=n_partitions, ax=ax)



def plot_mean_std_loss(eeg_train, bold_train, 
						eeg_val, bold_val, 
						eeg_test, bold_test, 
						encoder_network, decoder_network, 
						distance_name, distance_function,
						model_name, n_partitions=30):

	plt.figure(figsize=(20,5))
	ax1 = plt.subplot(131)

	shared_eeg_train = encoder_network.predict(eeg_train)
	synthesized_bold_train = decoder_network.predict(shared_eeg_train)
	_plot_mean_std_loss(synthesized_bold_train, bold_train, distance_function, distance_name, "train", model_name, n_partitions=n_partitions, ax=ax1)

	ax2 = plt.subplot(132)

	shared_eeg_val = encoder_network.predict(eeg_val)
	synthesized_bold_val = decoder_network.predict(shared_eeg_val)
	_plot_mean_std_loss(synthesized_bold_val, bold_val, distance_function, distance_name, "validation", model_name, n_partitions=n_partitions, ax=ax2)

	ax3 = plt.subplot(133)
	shared_eeg_test = encoder_network.predict(eeg_test)
	synthesized_bold_test = decoder_network.predict(shared_eeg_test)
	_plot_mean_std_loss(synthesized_bold_test, bold_test, distance_function, distance_name, "test", model_name, n_partitions=n_partitions, ax=ax3)

	plt.show()

def plot_loss_results(eeg_train, bold_train, eeg_val, bold_val, eeg_test, bold_test, eeg_network, decoder_network, model_name, n_partitions=30):

	plot_mean_std_loss(eeg_train, bold_train, 
	eeg_val, bold_val, 
	eeg_test, bold_test, 
	eeg_network, decoder_network, 
	"Log Cosine", losses.get_reconstruction_log_cosine_loss,
	model_name, n_partitions=n_partitions)

	plot_mean_std_loss(eeg_train, bold_train, 
	eeg_val, bold_val, 
	eeg_test, bold_test, 
	eeg_network, decoder_network, 
	"Log Cosine Voxels Mean", losses.get_reconstruction_log_cosine_voxel_loss,
	model_name, n_partitions=n_partitions)

	plot_mean_std_loss(eeg_train, bold_train, 
	eeg_val, bold_val, 
	eeg_test, bold_test, 
	eeg_network, decoder_network, 
	"Cosine", losses.get_reconstruction_cosine_loss,
	model_name, n_partitions=n_partitions)

	plot_mean_std_loss(eeg_train, bold_train, 
	eeg_val, bold_val, 
	eeg_test, bold_test, 
	eeg_network, decoder_network, 
	"Cosine Voxels Mean", losses.get_reconstruction_cosine_voxel_loss,
	model_name, n_partitions=n_partitions)

	plot_mean_std_loss(eeg_train, bold_train, 
	eeg_val, bold_val, 
	eeg_test, bold_test, 
	eeg_network, decoder_network, 
	"Euclidean", losses.get_euclidean_reconstruction_loss,
	model_name, n_partitions=n_partitions)


######################################################################################################################################################
#
#															PLOT VOXELS REAL AND SYNTHESIZED
#
######################################################################################################################################################


def _plot_voxel(real_signal, synth_signal, rows=1, columns=2, index=1, y_bottom=None, y_top=None):
    ax = plt.subplot(rows, columns, index)
    ax.plot(list(range(0, len(real_signal)*2, 2)), real_signal, color='b')
    ax.set_xlabel("Seconds")
    ax.set_ylabel("BOLD intensity")
    
    if(y_bottom==None and y_top==None):
        y_bottom_real = np.amin(real_signal)
        y_top_real = np.amax(real_signal)
        y_bottom_synth = np.amin(synth_signal)
        y_top_synth = np.amax(synth_signal)
        
    ax.set_ylim(y_bottom_real, y_top_real)
    
    if(index == 1):
        ax.set_title("Real BOLD Signal", y=0.99999)

        
    
    ax = plt.subplot(rows, columns, index+1)
    ax.plot(list(range(0, len(synth_signal)*2, 2)), synth_signal, color='r')
    ax.set_xlabel("Seconds")
    ax.set_ylabel("BOLD intensity")
    
    ax.set_ylim(y_bottom_synth, y_top_synth)
    
    if(index == 1):
        ax.set_title("Synthesized BOLD Signal")

def _plot_voxels(real_set, synth_set, individual=0, voxels=None, y_bottom=None, y_top=None, normalized=False):
    n_voxels=len(voxels)
    fig = plt.figure(figsize=(20,n_voxels*2))
    
    fig.suptitle('Top-' + str(len(voxels)) + ' correlated voxels', fontsize=16)
    
    if(individual != None):
        real_set = real_set[individual] 
        synth_set = synth_set[individual]
        
    index=1
    if(voxels):
        for voxel in range(n_voxels):
        	real_voxel = real_set[voxel]
        	synth_voxel = synth_set[voxel]
        	
        	if(normalized):
            	real_voxel = real_voxel/norm(real_voxel)
            	synth_voxel = synth_voxel/norm(synth_voxel)


            _plot_voxel(real_voxel, synth_voxel, 
                        rows=n_voxels, index=index, 
                        y_bottom=y_bottom, y_top=y_top)
            index += 2

    plt.show()

def rank_best_synthesized_voxels(real_signal, synth_signal, top_k=10, verbose=0):
    sort_voxels = {}
    n_voxels = real_signal.shape[0]
    
    for voxel in range(n_voxels):
    	voxel_a = real_signal[voxel].reshape((real_signal[voxel].shape[0]))
    	voxel_b = synth_signal[voxel].reshape((synth_signal[voxel].shape[0]))
    	distance_cosine = spatial.distance.cosine(voxel_a/norm(voxel_a), voxel_b/norm(voxel_b))
    	if(verbose>1):
    		print("Distance:", distance_cosine)

    	sort_voxels[voxel] = distance_cosine

    sort_voxels = dict(sorted(sort_voxels.items(), key=lambda kv: kv[1]))
    
    if(verbose>0):
    	print(list(sort_voxels.values())[0:top_k])

    return list(sort_voxels.keys())[0:top_k]