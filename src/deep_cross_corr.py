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

from keras.models import Sequential, Model
from keras.layers import Conv3D, Flatten, BatchNormalization, LSTM, TimeDistributed, Dense, Lambda, Input
from keras.optimizers import Adam
from keras.losses import mae

import keras.backend as K

n_partitions = 16
number_channels = 64
number_individuals = 16
n_epochs = 20


#16 - corresponds to a 20 second length signal with 10 time points
#32 - corresponds to a 10 second length signal with 5 time points
#individuals is a list of indexes until the maximum number of individuals
def get_data(individuals, TR=2.160, start_cutoff=3, n_partitions=16):
	TR = 1/TR

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
		fmri_masked_instance = fmri_utils.get_fmri_instance(individual)
		#for voxel in range(fmri_masked_instance.shape[1]):
		for voxel in range(10):#the range is just a temporary value
			voxel = fmri_utils.get_voxel(fmri_masked_instance, voxel=voxel)
			voxel_resampled = resample(voxel, int((len(voxel)*(1/2))/TR))
			for partition in range(n_partitions):
				start_eeg = start_cutoff + int(321/n_partitions)*partition
				end_eeg = start_cutoff + int(321/n_partitions)*partition + int(321/n_partitions)
				start_bold = start_eeg
				end_bold = end_eeg #+ 2
				X += [x_instance[:,:,start_eeg:end_eeg]]
				y += [voxel_resampled[start_bold:end_bold]]

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
	instances_per_individual = 10*16


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


def eeg_network(input_shape, kernel_size, output_dim=20):
	model = Sequential()

	model.add(Conv3D(1, kernel_size=kernel_size, 
		activation='selu', 
		input_shape=input_shape))

	model.add(BatchNormalization())

	model.add(TimeDistributed(Flatten()))

	model.add(LSTM(output_dim))

	return model

def bold_network(input_shape, output_dim=20):
	model = Sequential()

	model.add(LSTM(output_dim, input_shape=input_shape))

	return model

def contrastive_loss(y_true, y_pred):
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
	shape1, shape2 = shapes
	return (shape1[0], 1)







#reading data and spliting data into train and test by individuals
X_train, y_train = get_data(list(range(14)))
X_test, y_test = get_data(list(range(14, 16)))
print(X_train.shape, y_train.shape)

#reshape
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3], 1)

eeg_input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)

kernel_size = (X_train.shape[1], X_train.shape[2], 1)

#eeg network
eeg_network = eeg_network(eeg_input_shape, kernel_size)
print(eeg_network.summary())

#BOLD network
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
bold_input_shape = (y_train.shape[1], 1)

bold_network = bold_network(bold_input_shape)
print(bold_network.summary())


input_eeg = Input(shape=eeg_input_shape)
input_bold = Input(shape=bold_input_shape)

processed_eeg = eeg_network(input_eeg)
processed_bold = bold_network(input_bold)


correlation = Lambda(correlation, 
	output_shape=cos_dist_output_shape)([processed_eeg, processed_bold])

multi_modal_model = Model([input_eeg, input_bold], correlation)

multi_modal_model.compile(loss=contrastive_loss, optimizer=Adam(lr=0.0001))
eeg_network.compile(loss=cross_correlation, optimizer=Adam(lr=0.0001))


X_train_eeg, X_train_bold, tr_y = create_eeg_bold_pairs(X_train, y_train)
X_test_eeg, X_test_bold, te_y = create_eeg_bold_pairs(X_test, y_test)

history = multi_modal_model.fit([X_train_eeg, X_train_bold], 
	tr_y, epochs=n_epochs, 
	batch_size=n_partitions*10)