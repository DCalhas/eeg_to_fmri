import GPyOpt

import decoder
import deep_cross_corr

from sklearn.model_selection import train_test_split

import tensorflow.compat.v1 as tf

import numpy as np

def main():

	hyperparameters = [{'name': 'learning_rate', 'type': 'continuous',
	'domain': (10e-6, 10e-2)},
	{'name': 'l1_penalization_eeg', 'type': 'continuous',
	'domain': (10e-5, 10e-1)},
	{'name': 'l1_penalization_bold', 'type': 'continuous',
	'domain': (10e-5, 10e-1)},
	{'name': 'l1_penalization_decoder', 'type': 'continuous',
	'domain': (10e-5, 10e-1)},
	{'name': 'loss_coefficient', 'type': 'continuous',
	'domain': (0.0, 1.0)}]

	eeg_train, bold_train, eeg_test, bold_test = decoder.load_data(list(range(1)), list(range(1, 2)))
		
	X_train_eeg, X_train_bold, tr_y = deep_cross_corr.create_eeg_bold_pairs(eeg_train, bold_train)
	X_val_eeg, X_val_bold, tv_y = deep_cross_corr.create_eeg_bold_pairs(eeg_test, bold_test)

	#convert to tensors, for the networks to accept it as input
	X_train_eeg = tf.convert_to_tensor(X_train_eeg, dtype=np.float32)
	X_train_bold = tf.convert_to_tensor(X_train_bold, dtype=np.float32)
	tr_y = tf.convert_to_tensor(tr_y, dtype=np.float32)
	X_val_eeg = tf.convert_to_tensor(X_val_eeg, dtype=np.float32)
	X_val_bold = tf.convert_to_tensor(X_val_bold, dtype=np.float32)
	tv_y = tf.convert_to_tensor(tv_y, dtype=np.float32)

	normalization = tf.keras.layers.BatchNormalization(axis=2, input_shape=(None, X_train_bold.shape[1], X_train_bold.shape[2], X_train_bold.shape[3]))
	#normalization = tf.keras.Model((None, X_train_bold.shape[1], X_train_bold.shape[2], X_train_bold.shape[3]), normalization)

	X_train_bold = normalization(X_train_bold)
	X_val_bold = normalization(X_val_bold)

	def bayesian_optimization_function(x):
		current_learning_rate = float(x[:, 0])
		current_l1_penalization_eeg = float(x[:, 1])
		current_l1_penalization_bold = float(x[:, 2])
		current_l1_penalization_decoder = float(x[:, 3])
		current_loss_coefficient = float(x[:, 4])


		model_name = 'siamese_net_lr_' + str(current_learning_rate)


		######################################################################################################
		#
		#										DEFINING ARCHITECTURES
		#
		######################################################################################################
		#EEG network branch
		eeg_input_shape = (eeg_train.shape[1], eeg_train.shape[2], eeg_train.shape[3], 1)
		kernel_size = (eeg_train.shape[1], eeg_train.shape[2], 1)
		eeg_network = deep_cross_corr.eeg_network(eeg_input_shape, kernel_size, regularizer=tf.keras.regularizers.l1(current_l1_penalization_eeg))

		#BOLD network branch
		bold_input_shape = (bold_train.shape[1], bold_train.shape[2], 1)
		kernel_size = (bold_train.shape[1], 1)
		bold_network = deep_cross_corr.bold_network(bold_input_shape, kernel_size, regularizer=tf.keras.regularizers.l1(current_l1_penalization_bold))

		#Decoder network branch
		shared_eeg_train = eeg_network.predict(eeg_train)
		input_shape = (None, shared_eeg_train.shape[1], shared_eeg_train.shape[2], 1)
		decoder_model = decoder.decoding_network(input_shape, regularizer=tf.keras.regularizers.l1(current_l1_penalization_decoder))

		#Joining EEG and BOLD branches
		multi_modal_model = decoder.multi_modal_network(eeg_input_shape, bold_input_shape, eeg_network, bold_network)

		######################################################################################################
		#
		#										RUN TRAINING SESSION
		#
		######################################################################################################
		print("Starting training")
		tf.keras.backend.clear_session()
		validation_loss = decoder.run_training(X_train_eeg, X_train_bold, tr_y, eeg_network, decoder_model, multi_modal_model, 
			epochs=100, optimizer=tf.keras.optimizers.Adam(learning_rate=current_learning_rate), 
			linear_combination=current_loss_coefficient,
			batch_size=128,
			X_val_eeg=X_val_eeg,
			X_val_bold=X_val_bold,
			tv_y=tv_y)

		print("Model: " + model_name +
		' | Validation Loss: ' + str(validation_loss))
		tf.keras.backend.clear_session()
		return validation_loss

	optimizer = GPyOpt.methods.BayesianOptimization(
	f=bayesian_optimization_function, domain=hyperparameters, model_type="GP_MCMC", acquisition_type="EI_MCMC")

	print("Started Optimization Process")
	optimizer.run_optimization(max_iter=100)

	print("Optimized Parameters: {0}".format(optimizer.x_opt))
	print("Optimized Validation Decoder Loss: {0}".format(optimizer.fx_opt))


if __name__ == "__main__":
	main()