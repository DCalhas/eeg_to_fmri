import eeg_utils
import fmri_utils
import deep_cross_corr

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
from keras.models import model_from_json

import tensorflow.compat.v1 as tf

import keras.backend as K

import sys



n_partitions = 16
output_dim = 20
activation_function = 'selu'
reg_l=0
regularizer = regularizers.l1(reg_l)

if __name__ == "__main__":

	#mask = fmri_utils.get_population_mask()

	#reading data and spliting data into train and test by individuals
	#X_train, y_train = deep_cross_corr.get_data(list(range(14)), masker=mask)
	#X_test, y_test = deep_cross_corr.get_data(list(range(14, 16)), masker=mask)

	#X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)
	#X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3], 1)

	#eeg_input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)

	#kernel_size = (X_train.shape[1], X_train.shape[2], 1)

	#eeg network
	#eeg_network = deep_cross_corr.eeg_network(eeg_input_shape, kernel_size, output_dim=output_dim,
	#	activation_function=activation_function, regularizer=regularizer)
	#print(eeg_network.summary())

	#LOAD MODEL WEIGHTS
	json_file = open('multi_model/eeg_demo_0.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	eeg_network = model_from_json(loaded_model_json)
	# load weights into new model
	eeg_network.load_weights("multi_model/eeg_demo_0.h5")

	print(eeg_network.summary())

	history = eeg_network.fit([X_train_eeg, X_train_bold], 
		tr_y, epochs=n_epochs, 
		batch_size=n_partitions*100)#, validation_data=([X_test_eeg, X_test_bold], te_y))
