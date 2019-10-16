import GPyOpt

import decoder
import deep_cross_corr

from sklearn.model_selection import train_test_split

import tensorflow.compat.v1 as tf

import numpy as np

import iterative_naive_nas as nas




################################################################################################################################
#
#												BO for Neural Architecture Search
#
################################################################################################################################


def hidden_layer_NAS_BO(multi_modal_instance, eeg_domain, bold_domain, decoder_domain):

	print("Optimizing at level ", multi_modal_instance.get_level())

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


	#add element for new layer output
	output_shape = (20, 1)



	if(not (eeg_domain['domain'] and bold_domain['domain'] and decoder_domain['domain'])):
		return None, None, None, None

	eeg_train, bold_train, eeg_test, bold_test = decoder.load_data(list(range(1)), list(range(1, 2)))

	hyperparameters += [eeg_domain, bold_domain, decoder_domain]
	
	print("Finished Loading Data")

	X_train_eeg, X_train_bold, tr_y = deep_cross_corr.create_eeg_bold_pairs(eeg_train, bold_train)
	X_val_eeg, X_val_bold, tv_y = deep_cross_corr.create_eeg_bold_pairs(eeg_test, bold_test)

	print("Pairs Created")
	
	#convert to tensors, for the networks to accept it as input
	X_train_eeg = tf.convert_to_tensor(X_train_eeg, dtype=np.float32)
	X_train_bold = tf.convert_to_tensor(X_train_bold, dtype=np.float32)
	tr_y = tf.convert_to_tensor(tr_y, dtype=np.float32)
	X_val_eeg = tf.convert_to_tensor(X_val_eeg, dtype=np.float32)
	X_val_bold = tf.convert_to_tensor(X_val_bold, dtype=np.float32)
	tv_y = tf.convert_to_tensor(tv_y, dtype=np.float32)

	normalization = tf.keras.layers.BatchNormalization(axis=2, input_shape=(None, X_train_bold.shape[1], X_train_bold.shape[2], X_train_bold.shape[3]))

	X_train_bold = normalization(X_train_bold)
	X_val_bold = normalization(X_val_bold)



	def bayesian_optimization_function(x):
		current_learning_rate = float(x[:, 0])
		current_l1_penalization_eeg = float(x[:, 1])
		current_l1_penalization_bold = float(x[:, 2])
		current_l1_penalization_decoder = float(x[:, 3])
		current_loss_coefficient = float(x[:, 4])
		current_eeg_hidden_shape = int(x[:, 5])
		current_bold_hidden_shape = int(x[:, 6])
		current_decoder_hidden_shape = int(x[:, 7])


		model_name = 'siamese_net_lr_' + str(current_learning_rate)


		######################################################################################################
		#
		#										DEFINING ARCHITECTURES
		#
		######################################################################################################
		#EEG network branch
		#FIX HOW TO PUT HIDDEN LAYER SHAPE TO BUILD NET
		#EEG network branch
		eeg_input_shape = (eeg_train.shape[1], eeg_train.shape[2], eeg_train.shape[3], 1)
		current_eeg_hidden_shape = (current_eeg_hidden_shape,) + output_shape
		eeg_network = multi_modal_instance.build_eeg(eeg_input_shape, current_eeg_hidden_shape)

		#BOLD network branch
		bold_input_shape = (bold_train.shape[1], bold_train.shape[2], 1)
		current_bold_hidden_shape = (current_bold_hidden_shape,) + output_shape
		bold_network = multi_modal_instance.build_bold(bold_input_shape, current_bold_hidden_shape)

		current_decoder_hidden_shape = (current_decoder_hidden_shape,) + output_shape
		#THE ERROR IS HERE; PUT current_decoder_hidden_shape as the second argument, and the first argument is the compressed size
		decoder_network = multi_modal_instance.build_decoder(current_decoder_hidden_shape, bold_input_shape)

		if(not (eeg_network and bold_network and decoder_network)):
			return 1

		#Joining EEG and BOLD branches
		multi_modal_model = decoder.multi_modal_network(eeg_input_shape, bold_input_shape, eeg_network, bold_network)

		######################################################################################################
		#
		#										RUN TRAINING SESSION
		#
		######################################################################################################
		print("Starting training")
		tf.keras.backend.clear_session()
		validation_loss = decoder.run_training(X_train_eeg, X_train_bold, tr_y, eeg_network, decoder_network, multi_modal_model, 
			epochs=100, optimizer=tf.keras.optimizers.Adam(learning_rate=current_learning_rate), 
			linear_combination=current_loss_coefficient,
			batch_size=128,
			X_val_eeg=X_val_eeg,
			X_val_bold=X_val_bold,
			tv_y=tv_y)

		print("Model: " + model_name +
		' Train Intances: ' + str(len(X_train_bold)) + ' | Validation Instances: ' + str(len(X_val_bold)) +  ' | Validation Loss: ' + str(validation_loss))
		tf.keras.backend.clear_session()
		return validation_loss

	optimizer = GPyOpt.methods.BayesianOptimization(
	f=bayesian_optimization_function, domain=hyperparameters, model_type="GP_MCMC", acquisition_type="EI_MCMC")

	print("Started Optimization Process")
	optimizer.run_optimization(max_iter=1)

	#SAVE BEST MODELS
	#EEG network branch
	eeg_input_shape = (eeg_train[1], eeg_train[2], eeg_train[3], 1)
	eeg_optimal_shape = (int(optimizer.x_opt[-3]),) + output_shape
	eeg_network = multi_modal_instance.build_eeg(eeg_input_shape, eeg_optimal_shape)
	
	#BOLD network branch
	bold_input_shape = (bold_train[1], bold_train[2], 1)
	bold_optimal_shape = (int(optimizer.x_opt[-2]),) + output_shape
	bold_network = multi_modal_instance.build_bold(bold_input_shape, bold_optimal_shape)

	#Decoder network branch
	decoder_optimal_shape = (int(optimizer.x_opt[-1]),) + output_shape
	decoder_network = multi_modal_instance.build_decoder(decoder_optimal_shape, bold_input_shape)

	if(not (eeg_network and bold_network and decoder_network)):
		return None, None, None, None

	multi_modal_instance.save_eeg(eeg_network)
	multi_modal_instance.save_bold(bold_network)
	multi_modal_instance.save_decoder(decoder_network)

	print("Optimized Parameters: {0}".format(optimizer.x_opt))
	print("Optimized Validation Decoder Loss: {0}".format(optimizer.fx_opt))

	return optimizer.x_opt[-3], optimizer.x_opt[-2], optimizer.x_opt[-1], optimizer.fx_opt













################################################################################################################################
#
#												BO for Neural Architecture Search
#
################################################################################################################################


def NAS_BO(multi_modal_instance, output_shape_domain):

	print("Optimizing at level ", multi_modal_instance.get_level())

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


	#add element for new layer output
	output_shape = (20, 1)

	eeg_train, bold_train, eeg_test, bold_test = decoder.load_data(list(range(1)), list(range(1, 2)))

	hyperparameters += output_shape_domain


	
	print("Finished Loading Data")

	X_train_eeg, X_train_bold, tr_y = deep_cross_corr.create_eeg_bold_pairs(eeg_train, bold_train)
	X_val_eeg, X_val_bold, tv_y = deep_cross_corr.create_eeg_bold_pairs(eeg_test, bold_test)

	print("Pairs Created")
	
	#convert to tensors, for the networks to accept it as input
	X_train_eeg = tf.convert_to_tensor(X_train_eeg, dtype=np.float32)
	X_train_bold = tf.convert_to_tensor(X_train_bold, dtype=np.float32)
	tr_y = tf.convert_to_tensor(tr_y, dtype=np.float32)
	X_val_eeg = tf.convert_to_tensor(X_val_eeg, dtype=np.float32)
	X_val_bold = tf.convert_to_tensor(X_val_bold, dtype=np.float32)
	tv_y = tf.convert_to_tensor(tv_y, dtype=np.float32)

	normalization = tf.keras.layers.BatchNormalization(axis=2, input_shape=(None, X_train_bold.shape[1], X_train_bold.shape[2], X_train_bold.shape[3]))
	pooling = tf.keras.layers.MaxPooling2D((2,1))

	X_train_bold = pooling(X_train_bold)
	X_val_bold = pooling(X_val_bold)
	X_train_bold = normalization(X_train_bold)
	X_val_bold = normalization(X_val_bold)

	print(X_train_bold.shape)

	def bayesian_optimization_function(x):
		current_learning_rate = float(x[:, 0])
		current_l1_penalization_eeg = float(x[:, 1])
		current_l1_penalization_bold = float(x[:, 2])
		current_l1_penalization_decoder = float(x[:, 3])
		current_loss_coefficient = float(x[:, 4])
		current_shape = int(x[:, 5])


		model_name = 'siamese_net_lr_' + str(current_learning_rate)


		######################################################################################################
		#
		#										DEFINING ARCHITECTURES
		#
		######################################################################################################
		#EEG network branch
		eeg_input_shape = (eeg_train.shape[1], eeg_train.shape[2], eeg_train.shape[3], 1)
		current_shape = (current_shape,) + output_shape
		eeg_network = multi_modal_instance.build_eeg(eeg_input_shape, current_shape)

		#BOLD network branch
		bold_input_shape = (bold_train.shape[1], bold_train.shape[2], 1)
		bold_network = multi_modal_instance.build_bold(bold_input_shape, current_shape)

		#Decoder network branch
		decoder_network = multi_modal_instance.build_decoder(current_shape, bold_input_shape)

		if(not (eeg_network and bold_network and decoder_network)):
			return 1.0

		#Joining EEG and BOLD branches
		multi_modal_network = decoder.multi_modal_network(eeg_input_shape, bold_input_shape, eeg_network, bold_network)

		######################################################################################################
		#
		#										RUN TRAINING SESSION
		#
		######################################################################################################
		print("Starting training")

		tf.keras.backend.clear_session()
		validation_loss = decoder.run_training(X_train_eeg, X_train_bold, tr_y, eeg_network, decoder_network, multi_modal_network, 
			epochs=20, optimizer=tf.keras.optimizers.Adam(learning_rate=current_learning_rate), 
			linear_combination=current_loss_coefficient,
			batch_size=128,
			X_val_eeg=X_val_eeg,
			X_val_bold=X_val_bold,
			tv_y=tv_y)

		print("Model: " + model_name +
		' Train Intances: ' + str(len(X_train_bold)) + ' | Validation Instances: ' + str(len(X_val_bold)) +  ' | Validation Loss: ' + str(validation_loss))
		tf.keras.backend.clear_session()
		return validation_loss

	optimizer = GPyOpt.methods.BayesianOptimization(
	f=bayesian_optimization_function, domain=hyperparameters, model_type="GP_MCMC", acquisition_type="EI_MCMC")

	print("Started Optimization Process")
	optimizer.run_optimization(max_iter=1)

	#SAVE BEST MODELS
	#EEG network branch
	eeg_input_shape = (eeg_train[1], eeg_train[2], eeg_train[3], 1)
	optimal_shape = (int(optimizer.x_opt[-1]),) + output_shape
	multi_modal_instance.save_eeg(multi_modal_instance.build_eeg(eeg_input_shape, optimal_shape))

	#BOLD network branch
	bold_input_shape = (bold_train[1], bold_train[2], 1)
	multi_modal_instance.save_bold(multi_modal_instance.build_bold(bold_input_shape, optimal_shape))

	#Decoder network branch
	multi_modal_instance.save_decoder(multi_modal_instance.build_decoder(optimal_shape, bold_input_shape))

	print("Optimized Parameters: {0}".format(optimizer.x_opt))
	print("Optimized Validation Decoder Loss: {0}".format(optimizer.fx_opt))

	return optimizer.x_opt[-1], optimizer.fx_opt













def default_BO():

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

	eeg_train, bold_train, eeg_test, bold_test = decoder.load_data(list(range(14)), list(range(14, 16)))
	
	print("Finished Loading Data")

	X_train_eeg, X_train_bold, tr_y = deep_cross_corr.create_eeg_bold_pairs(eeg_train, bold_train)
	X_val_eeg, X_val_bold, tv_y = deep_cross_corr.create_eeg_bold_pairs(eeg_test, bold_test)

	print("Pairs Created")
	
	#convert to tensors, for the networks to accept it as input
	X_train_eeg = tf.convert_to_tensor(X_train_eeg, dtype=np.float32)
	X_train_bold = tf.convert_to_tensor(X_train_bold, dtype=np.float32)
	tr_y = tf.convert_to_tensor(tr_y, dtype=np.float32)
	X_val_eeg = tf.convert_to_tensor(X_val_eeg, dtype=np.float32)
	X_val_bold = tf.convert_to_tensor(X_val_bold, dtype=np.float32)
	tv_y = tf.convert_to_tensor(tv_y, dtype=np.float32)

	normalization = tf.keras.layers.BatchNormalization(axis=2, input_shape=(None, X_train_bold.shape[1], X_train_bold.shape[2], X_train_bold.shape[3]))

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
		' Train Intances: ' + str(len(X_train_bold)) + ' | Validation Instances: ' + str(len(X_val_bold)) +  ' | Validation Loss: ' + str(validation_loss))
		tf.keras.backend.clear_session()
		return validation_loss

	optimizer = GPyOpt.methods.BayesianOptimization(
	f=bayesian_optimization_function, domain=hyperparameters, model_type="GP_MCMC", acquisition_type="EI_MCMC")

	print("Started Optimization Process")
	optimizer.run_optimization(max_iter=100)

	print("Optimized Parameters: {0}".format(optimizer.x_opt))
	print("Optimized Validation Decoder Loss: {0}".format(optimizer.fx_opt))


if __name__ == "__main__":
	default_BO()