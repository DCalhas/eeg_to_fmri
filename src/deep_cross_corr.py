import eeg_utils
import fmri_utils

import numpy as np
from numpy import correlate

import matplotlib.pyplot as plt

import mne
from nilearn.masking import apply_mask, compute_epi_mask

from sklearn.preprocessing import normalize

from scipy.signal import resample

import keras

from keras.initializers import Zeros
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Conv2D, Conv3D, Reshape, Flatten, BatchNormalization, LSTM, TimeDistributed, Dense, Lambda, Input, MaxPooling2D, MaxPooling3D 
from keras.optimizers import Adam
from keras.losses import mae

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()


import keras.backend as K

import sys

n_partitions = 16
number_channels = 64
number_individuals = 16
n_epochs = 20


#16 - corresponds to a 20 second length signal with 10 time points
#32 - corresponds to a 10 second length signal with 5 time points
#individuals is a list of indexes until the maximum number of individuals
def get_data(individuals, masker=None, start_cutoff=3, n_partitions=16):
    TR = 1/2.160
    
    X = []
    y = []

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
        fmri_instance = fmri_utils.get_fmri_instance_img(individual)
        fmri_masked_instance, _ = fmri_utils.get_masked_epi(fmri_instance, masker)
        
        fmri_resampled = []
        #build resampled BOLD signal
        for voxel in range(fmri_masked_instance.shape[1]):
            voxel = fmri_utils.get_voxel(fmri_masked_instance, voxel=voxel)
            voxel_resampled = resample(voxel, int((len(voxel)*(1/2))/TR))
            fmri_resampled += [voxel_resampled]
        
        fmri_resampled = np.array(fmri_resampled)
        #print(fmri_resampled.shape)
        #fmri_resampled = fmri_resampled.reshape(fmri_resampled.shape[1], fmri_resampled.shape[0])
        #print(fmri_resampled.shape)
        for partition in range(n_partitions):
            start_eeg = start_cutoff + int(321/n_partitions)*partition
            end_eeg = start_cutoff + int(321/n_partitions)*partition + int(321/n_partitions)
            
            start_bold = start_eeg
            end_bold = end_eeg #+ 2
            
            X += [x_instance[:,:,start_eeg:end_eeg]]
            
            y += list(fmri_resampled[:,start_bold:end_bold].reshape(1, fmri_resampled[:,start_bold:end_bold].shape[0], fmri_resampled[:,start_bold:end_bold].shape[1]))
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


def eeg_network(input_shape, kernel_size, output_dim=20, activation_function='selu', regularizer=regularizers.l1(0.001)):
	model = Sequential()


	model.add(Conv3D(1, kernel_size=(2, 2, kernel_size[2]),
		activation=activation_function, strides=(2,2,1),
		input_shape=input_shape, kernel_regularizer=regularizer, 
		bias_regularizer=regularizer, activity_regularizer=regularizer))
	model.add(BatchNormalization())
	model.add(Conv3D(1, kernel_size=(2, 2, kernel_size[2]),
		activation=activation_function, strides=(2,2,1),
		input_shape=input_shape, kernel_regularizer=regularizer, 
		bias_regularizer=regularizer, activity_regularizer=regularizer))
	model.add(Reshape((16, 20, 1)))
	#model.add(BatchNormalization())
	#model.add(Conv3D(1, kernel_size=(16, 1, kernel_size[2]),
	#activation=activation_function, strides=(2,1,1),
	#input_shape=input_shape, kernel_regularizer=regularizer, 
	#bias_regularizer=regularizer, activity_regularizer=regularizer))
	#model.add(BatchNormalization())

	#model.add(TimeDistributed(Flatten()))

	#model.add(LSTM(output_dim))

	return model

def bold_network(input_shape, kernel_size, output_dim=20, activation_function='selu', regularizer=regularizers.l1(0.001)):
	model = Sequential()

	model.add(Conv2D(1, kernel_size=(100, kernel_size[1]),
		activation=activation_function, strides=(50,1),
		input_shape=input_shape, kernel_regularizer=regularizer, 
		bias_regularizer=regularizer, activity_regularizer=regularizer))
	model.add(BatchNormalization())
	model.add(Conv2D(1, kernel_size=(100, kernel_size[1]),
		activation=activation_function, strides=(12,1),
		kernel_regularizer=regularizer, 
		bias_regularizer=regularizer, activity_regularizer=regularizer))
	model.add(Reshape((16, 20, 1)))
	#model.add(BatchNormalization())
	#model.add(Conv2D(1, kernel_size=(16, kernel_size[1]),
	#activation=activation_function, strides=(1,1),
	# kernel_regularizer=regularizer, 
	#bias_regularizer=regularizer, activity_regularizer=regularizer))
	#model.add(BatchNormalization())
	#model.add(TimeDistributed(Flatten()))

	#model.add(LSTM(output_dim, input_shape=input_shape))

	return model

def contrastive_loss(y_true, y_pred):
	tf.print(y_true, output_stream=sys.stdout)
	tf.print(y_pred, output_stream=sys.stdout)
	square_pred = K.square(y_pred)
	margin_square = K.square(K.maximum(1.0 - y_pred, 0))
	return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def cross_correlation(x, y):
	#how should the normalization be done??
	x = K.l2_normalize(x, axis=1)
	y = K.l2_normalize(y, axis=1)

	a = K.batch_dot(x, y, axes=1)

	b = K.batch_dot(x, x, axes=1)
	c = K.batch_dot(y, y, axes=1)
	tf.print(a, output_stream=sys.stdout)
	tf.print(b, output_stream=sys.stdout)
	tf.print(c, output_stream=sys.stdout)


	return 1 - (a / (K.sqrt(b) * K.sqrt(c)))

def correlation(vects):
	#how should the normalization be done??
	x, y = vects
	x = K.l2_normalize(x, axis=1)
	y = K.l2_normalize(y, axis=1)

	a = K.batch_dot(x, y, axes=1)

	b = K.batch_dot(x, x, axes=1)
	c = K.batch_dot(y, y, axes=1)

	return 1 - (a / (K.sqrt(b) * K.sqrt(c)))

def cos_dist_output_shape(shapes):
	print(shapes)
	shape1, shape2 = shapes
	print(shape1, shape2)
	return (shape1[0], 1)


n_partitions = 16
output_dim = 20
activation_function = 'selu'
reg_l=0
regularizer = regularizers.l1(reg_l)

if __name__ == "__main__":

	mask = fmri_utils.get_population_mask()

	#reading data and spliting data into train and test by individuals
	X_train, y_train = get_data(list(range(14)), masker=mask)
	X_test, y_test = get_data(list(range(14, 16)), masker=mask)
	print(X_train.shape, y_train.shape)

	X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)
	X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3], 1)

	eeg_input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)

	kernel_size = (X_train.shape[1], X_train.shape[2], 1)
	print(kernel_size)
	#eeg network
	eeg_network = eeg_network(eeg_input_shape, kernel_size, output_dim=output_dim,
		activation_function=activation_function, regularizer=regularizer)
	print(eeg_network.summary())

	#BOLD network (224, 64, 5, 20) (224, 14164, 20)
	y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], 1)
	y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], 1)

	bold_input_shape = (y_train.shape[1], y_train.shape[2], 1)

	kernel_size = (y_train.shape[1], 1)
	print(kernel_size)

	bold_network = bold_network(bold_input_shape, kernel_size, output_dim=output_dim, 
		activation_function=activation_function, regularizer=regularizer)
	print(bold_network.summary())

	input_eeg = Input(shape=eeg_input_shape)
	input_bold = Input(shape=bold_input_shape)

	processed_eeg = eeg_network(input_eeg)
	processed_bold = bold_network(input_bold)

	print(input_eeg)
	print(processed_eeg)
	print(input_bold)
	print(processed_bold)
	print(cos_dist_output_shape)
	correlation = Lambda(correlation, 
		output_shape=cos_dist_output_shape)([processed_eeg, processed_bold])

	multi_modal_model = Model([input_eeg, input_bold], correlation)
	print(multi_modal_model.summary())

	multi_modal_model.compile(loss=contrastive_loss, optimizer=Adam(lr=0.001))
	eeg_network.compile(loss=cross_correlation, optimizer=Adam(lr=0.0001))


	X_train_eeg, X_train_bold, tr_y = create_eeg_bold_pairs(X_train, y_train)
	X_test_eeg, X_test_bold, te_y = create_eeg_bold_pairs(X_test, y_test)


	print(X_train_eeg.shape)

	#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
	cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
	cfg.gpu_options.allow_growth = True
	session = tf.Session(config=cfg)
	with session:
		session.run(tf.global_variables_initializer())
		history = multi_modal_model.fit([X_train_eeg, X_train_bold], 
			tr_y, epochs=n_epochs, 
			batch_size=n_partitions*100)#, validation_data=([X_test_eeg, X_test_bold], te_y))

