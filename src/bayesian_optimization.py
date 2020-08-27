import GPyOpt

from utils import data_utils, losses_utils

from sklearn.model_selection import train_test_split

import tensorflow.compat.v1 as tf

import numpy as np

import iterative_naive_nas as nas

import custom_training

mode=1

if (__name__ == "__main__" or mode==1):

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
	config = tf.ConfigProto(allow_soft_placement=True,
							gpu_options=gpu_options)
	config.gpu_options.allow_growth=True
	tf.enable_eager_execution(config=config)

	print("Starting to Load Data")

	n_partitions=25
	n_individuals=10
	dataset="01"

	eeg_train, bold_train, mask, scalers = data_utils.load_data(list(range(n_individuals)),n_voxels=None, 
																	bold_shift=3, n_partitions=n_partitions, 
																	mutate_bands=False,
																	by_partitions=False, partition_length=14, 
																	f_resample=1.8, fmri_resolution_factor=4, 
																	standardize_eeg=True, standardize_fmri=True,
																	dataset="01")

	frequency_resolution=eeg_train.shape[2]
	eeg_channels=eeg_train.shape[1]
	n_voxels = bold_train.shape[1]
	interval_length = bold_train.shape[2]
	n_partitions = bold_train.shape[0]//n_individuals
	
	if(dataset=="01"):
		n_individuals_train = 6
		n_individuals_val = 2

	eeg_val = eeg_train[(n_individuals_train)*n_partitions:(n_individuals_train+n_individuals_val)*n_partitions]
	bold_val = bold_train[(n_individuals_train)*n_partitions:(n_individuals_train+n_individuals_val)*n_partitions]
	eeg_train = eeg_train[:(n_individuals_train)*n_partitions]
	bold_train = bold_train[:(n_individuals_train)*n_partitions]

	print("Finished Loading Data")

	X_train_eeg, X_train_bold, tr_y, X_bold_train_target = data_utils.create_eeg_bold_pairs(eeg_train, bold_train, instances_per_individual=n_partitions)
	X_val_eeg, X_val_bold, tv_y, X_bold_val_target = data_utils.create_eeg_bold_pairs(eeg_val, bold_val, instances_per_individual=n_partitions)

	tr_y = np.array(tr_y, dtype=np.float32)
	tv_y = np.array(tv_y, dtype=np.float32)

	eeg_train = eeg_train.reshape(eeg_train.shape + (1,))
	bold_train = bold_train.reshape(bold_train.shape + (1,))
	eeg_val = eeg_val.reshape(eeg_val.shape + (1,))
	bold_val = bold_val.reshape(bold_val.shape + (1,))

	eeg_train = eeg_train.astype('float32')
	bold_train = bold_train.astype('float32')
	eeg_val = eeg_val.astype('float32')
	bold_val = bold_val.astype('float32')

	print("Pairs Created")


################################################################################################################################
#
#												BO for Neural Architecture Search
#
################################################################################################################################


def hidden_layer_NAS_BO(multi_modal_instance, eeg_domain, bold_domain, decoder_domain):

	print("Optimizing at level ", multi_modal_instance.get_level())

	hyperparameters = [{'name': 'learning_rate', 'type': 'continuous',
	'domain': (10e-15, 10e-4)},
	{'name': 'l1_penalization_eeg', 'type': 'continuous',
	'domain': (10e-5, 10e-1)},
	{'name': 'l1_penalization_bold', 'type': 'continuous',
	'domain': (10e-5, 10e-1)},
	{'name': 'l1_penalization_decoder', 'type': 'continuous',
	'domain': (10e-5, 10e-1)},
	{'name': 'loss_coefficient', 'type': 'continuous',
	'domain': (0.0, 1.0)},
	{'name': 'batch_size', 'type': 'discrete',
	'domain': (2, 4, 8, 16, 32, 64, 128)}]
	#{'name': 'dcca_output', 'type': 'discrete',
	#'domain': (10, 20, 30, 40, 50)},

	global interval_length


	#add element for new layer output
	output_shape = (int(interval_length), 1)

	if(not (eeg_domain['domain'] and bold_domain['domain'] and decoder_domain['domain'])):
		return None, None, None, None

	hyperparameters += [eeg_domain, bold_domain, decoder_domain]
		

	def bayesian_optimization_function(x):
		current_learning_rate = float(x[:, 0])
		current_l1_penalization_eeg = float(x[:, 1])
		current_l1_penalization_bold = float(x[:, 2])
		current_l1_penalization_decoder = float(x[:, 3])
		current_loss_coefficient = float(x[:, 4])
		current_batch_size = int(x[:, 5])
		#current_dcca_output = int(x[:, 5])
		current_eeg_hidden_shape = int(x[:, 6])
		current_bold_hidden_shape = int(x[:, 7])
		current_decoder_hidden_shape = int(x[:, 8])

		current_dropout = 0.5

		dcca=False


		model_name = 'bold_synthesis_net_lr_' + str(current_learning_rate)


		######################################################################################################
		#
		#										DEFINING ARCHITECTURES
		#
		######################################################################################################
		#EEG network branch
		#FIX HOW TO PUT HIDDEN LAYER SHAPE TO BUILD NET
		#EEG network branch

		global X_train_eeg, X_train_bold, X_val_bold, X_val_eeg, tv_y, tr_y, eeg_train, bold_train, eeg_val, bold_val, X_bold_train_target, X_bold_val_target
		
		eeg_input_shape = eeg_train.shape[1:]
		current_eeg_hidden_shape = (current_eeg_hidden_shape,) + output_shape
		eeg_network = multi_modal_instance.build_eeg(eeg_input_shape, 
													current_eeg_hidden_shape,
													regularization=current_l1_penalization_eeg,
													dropout=current_dropout)
		
		#BOLD network branch
		bold_input_shape = bold_train.shape[1:]
		current_bold_hidden_shape = (current_bold_hidden_shape,) + output_shape
		bold_network = multi_modal_instance.build_bold(bold_input_shape, 
													current_bold_hidden_shape,
													regularization=current_l1_penalization_bold,
													dropout=current_dropout)
		
		current_decoder_hidden_shape = (current_decoder_hidden_shape,) + output_shape
		#THE ERROR IS HERE; PUT current_decoder_hidden_shape as the second argument, and the first argument is the compressed size
		decoder_network = multi_modal_instance.build_decoder(current_decoder_hidden_shape, 
															bold_input_shape,
															regularization=current_l1_penalization_decoder,
															dropout=current_dropout)
		
		if(not (eeg_network and bold_network and decoder_network)):
			return 1

		#Joining EEG and BOLD branches
		multi_modal_model = custom_training.multi_modal_network(eeg_input_shape, bold_input_shape, eeg_network, bold_network, 
																dcca=dcca, corr_distance=True)

		#normalization of the BOLD signal, please change this
		#norm = tf.keras.Sequential()
		#norm.add(tf.keras.layers.BatchNormalization(axis=2, input_shape=(X_train_bold.shape[1], X_train_bold.shape[2], X_train_bold.shape[3])))
		#norm.build(input_shape=(X_train_bold.shape[1], X_train_bold.shape[2], X_train_bold.shape[3]))

		#X_train_bold = norm.predict(X_train_bold)
		#X_val_bold = norm.predict(X_val_bold)


		######################################################################################################
		#
		#										RUN TRAINING SESSION
		#
		######################################################################################################
		print("Starting training")		
		
		#exception can appear
		validation_loss = custom_training.linear_combination_training(X_train_eeg, X_train_bold, tr_y, eeg_network, decoder_network, multi_modal_model, 
			epochs=5, 
			encoder_optimizer=tf.keras.optimizers.Adam(learning_rate=current_learning_rate),
			decoder_optimizer=tf.keras.optimizers.Adam(learning_rate=current_learning_rate),
			loss_function=losses_utils.get_reconstruction_log_cosine_voxel_loss,
			batch_size=current_batch_size, linear_combination=current_loss_coefficient,
			X_val_eeg=X_val_eeg,
			X_val_bold=X_val_bold,
			tv_y=tv_y,
			eeg_train=eeg_train, bold_train=bold_train, eeg_val=eeg_val, bold_val=bold_val,
			X_bold_train_target=X_bold_train_target,
    		X_bold_val_target=X_bold_val_target)
		#validation_loss = custom_training.adversarial_training(X_train_eeg, X_train_bold, tr_y, eeg_network, decoder_network, multi_modal_model, 
		#	epochs=40, optimizer=tf.keras.optimizers.Adam(learning_rate=current_learning_rate),
		#	batch_size=current_batch_size, linear_combination=current_loss_coefficient,
		#	X_val_eeg=X_val_eeg,
		#	X_val_bold=X_val_bold,
		#	tv_y=tv_y)
		#	eeg_train=eeg_train, bold_train=bold_train, eeg_val=eeg_val, bold_val=bold_val, bold_network=bold_network)
		#validation_loss = custom_training.ranked_synthesis_training(X_train_eeg, X_train_bold, tr_y, eeg_network, decoder_network, multi_modal_model, 
		#	epochs=40, 
		#	encoder_optimizer=tf.keras.optimizers.Adam(learning_rate=current_learning_rate),
		#	decoder_optimizer=tf.keras.optimizers.Adam(learning_rate=current_learning_rate),
		#	batch_size=current_batch_size,
		#	X_val_eeg=X_val_eeg,
		#	X_val_bold=X_val_bold,
		#	tv_y=tv_y,
		#	eeg_train=eeg_train, bold_train=bold_train, eeg_val=eeg_val, bold_val=bold_val, bold_network=bold_network,
		#	X_bold_train_target=X_bold_train_target,
		#	X_bold_val_target=X_bold_val_target)
		#validation_loss = custom_training.dcca_training(X_train_eeg, X_train_bold, tr_y, eeg_network, decoder_network, multi_modal_model, 
		#	epochs=40, optimizer=tf.keras.optimizers.Adam(learning_rate=current_learning_rate),
		#	batch_size=current_batch_size, linear_combination=current_loss_coefficient, dcca_output=current_dcca_output,
		#	X_val_eeg=X_val_eeg,
		#	X_val_bold=X_val_bold,
		#	tv_y=tv_y)

			

		print("Model: " + model_name +
		' Train Intances: ' + str(len(X_train_bold)) + ' | Validation Instances: ' + str(len(X_val_bold)) +  ' | Validation Loss: ' + str(validation_loss))
		
		return validation_loss

	optimizer = GPyOpt.methods.BayesianOptimization(
	f=bayesian_optimization_function, domain=hyperparameters, model_type="GP_MCMC", acquisition_type="EI_MCMC")

	print("Started Optimization Process")
	optimizer.run_optimization(max_iter=1)

	#SAVE BEST MODELS
	#EEG network branch
	eeg_input_shape = (eeg_train.shape[1], eeg_train.shape[2], eeg_train.shape[3], 1)
	eeg_optimal_shape = (int(optimizer.x_opt[-3]),) + output_shape
	eeg_network = multi_modal_instance.build_eeg(eeg_input_shape, eeg_optimal_shape, 
		regularization=float(optimizer.x_opt[1]), 
		dropout=float(optimizer.x_opt[5]))
	
	#BOLD network branch
	bold_input_shape = (bold_train.shape[1], bold_train.shape[2], 1)
	bold_optimal_shape = (int(optimizer.x_opt[-2]),) + output_shape
	bold_network = multi_modal_instance.build_bold(bold_input_shape, bold_optimal_shape, 
		regularization=float(optimizer.x_opt[2]), 
		dropout=float(optimizer.x_opt[5]))

	#Decoder network branch
	decoder_optimal_shape = (int(optimizer.x_opt[-1]),) + output_shape
	decoder_network = multi_modal_instance.build_decoder(decoder_optimal_shape, bold_input_shape, 
		regularization=float(optimizer.x_opt[3]), 
		dropout=float(optimizer.x_opt[5]))

	if(not (eeg_network and bold_network and decoder_network)):
		return None, None, None, None

	multi_modal_instance.save_eeg(eeg_network)
	multi_modal_instance.save_bold(bold_network)
	multi_modal_instance.save_decoder(decoder_network)

	print("Optimized Parameters: {0}".format(optimizer.x_opt))
	print("Optimized Validation Decoder Loss: {0}".format(optimizer.fx_opt))
	print("\n\n\n\n\n\n\n\n\n\n")

	return optimizer.x_opt[-3], optimizer.x_opt[-2], optimizer.x_opt[-1], optimizer.fx_opt













################################################################################################################################
#
#												BO for Neural Architecture Search
#
################################################################################################################################


def NAS_BO(multi_modal_instance, output_shape_domain):

	print("Optimizing at level ", multi_modal_instance.get_level())

	hyperparameters = [{'name': 'learning_rate', 'type': 'continuous',
	'domain': (10e-15, 10e-4)},
	{'name': 'l1_penalization_eeg', 'type': 'continuous',
	'domain': (10e-5, 10e-1)},
	{'name': 'l1_penalization_bold', 'type': 'continuous',
	'domain': (10e-5, 10e-1)},
	{'name': 'l1_penalization_decoder', 'type': 'continuous',
	'domain': (10e-5, 10e-1)},
	{'name': 'loss_coefficient', 'type': 'continuous',
	'domain': (0.0, 1.0)},
	{'name': 'batch_size', 'type': 'discrete',
	'domain': (2, 4, 8, 16, 32, 64, 128)}]
	#{'name': 'dcca_output', 'type': 'discrete',
	#'domain': (10, 20, 30, 40, 50)},

	global interval_length

	#add element for new layer output
	output_shape = (int(interval_length), 1)

	hyperparameters += output_shape_domain

	def bayesian_optimization_function(x):
		current_learning_rate = float(x[:, 0])
		current_l1_penalization_eeg = float(x[:, 1])
		current_l1_penalization_bold = float(x[:, 2])
		current_l1_penalization_decoder = float(x[:, 3])
		current_loss_coefficient = float(x[:, 4])
		#current_dcca_output = int(x[:, 5])
		current_batch_size = int(x[:, 5])
		current_shape = int(x[:, 6])

		current_dropout = 0.5

		dcca=False


		model_name = 'bold_synthesis_net_lr_' + str(current_learning_rate)


		######################################################################################################
		#
		#										DEFINING ARCHITECTURES
		#
		######################################################################################################
		#EEG network branch

		global X_train_eeg, X_train_bold, X_val_bold, X_val_eeg, tv_y, tr_y, eeg_train, bold_train, eeg_val, bold_val, X_bold_train_target, X_bold_val_target

		print("NAS BO")
	
		eeg_input_shape = eeg_train.shape[1:]
		current_shape = (current_shape,) + output_shape
		eeg_network = multi_modal_instance.build_eeg(eeg_input_shape, 
													current_shape,
													regularization=current_l1_penalization_eeg,
													dropout=current_dropout)

		#BOLD network branch
		bold_input_shape = bold_train.shape[1:]
		bold_network = multi_modal_instance.build_bold(bold_input_shape, 
													current_shape,
													regularization=current_l1_penalization_bold,
													dropout=current_dropout)

		#Decoder network branch
		decoder_network = multi_modal_instance.build_decoder(current_shape, 
															bold_input_shape,
															regularization=current_l1_penalization_decoder,
															dropout=current_dropout)

		if(not (eeg_network and bold_network and decoder_network)):
			return 1.0

		#Joining EEG and BOLD branches
		multi_modal_model = custom_training.multi_modal_network(eeg_input_shape, bold_input_shape, eeg_network, bold_network, 
																dcca=dcca, corr_distance=True)

		#normalization of the BOLD signal, please change this
		#norm = tf.keras.Sequential()
		#norm.add(tf.keras.layers.BatchNormalization(axis=2, input_shape=(X_train_bold.shape[1], X_train_bold.shape[2], X_train_bold.shape[3])))
		#norm.build(input_shape=(X_train_bold.shape[1], X_train_bold.shape[2], X_train_bold.shape[3]))

		#X_train_bold = norm.predict(X_train_bold)
		#X_val_bold = norm.predict(X_val_bold)

		######################################################################################################
		#
		#										RUN TRAINING SESSION
		#
		######################################################################################################
		print("Starting training")

		#this try should be checked
		#validation_loss = custom_training.ranked_synthesis_training(X_train_eeg, X_train_bold, tr_y, eeg_network, decoder_network, multi_modal_model, 
		#	epochs=40, 
		#	encoder_optimizer=tf.keras.optimizers.Adam(learning_rate=current_learning_rate),
		#	decoder_optimizer=tf.keras.optimizers.Adam(learning_rate=current_learning_rate),
		#	batch_size=current_batch_size,
		#	X_val_eeg=X_val_eeg,
		#	X_val_bold=X_val_bold,
		#	tv_y=tv_y,
		#	eeg_train=eeg_train, bold_train=bold_train, eeg_val=eeg_val, bold_val=bold_val, bold_network=bold_network,
		#	X_bold_train_target=X_bold_train_target,
		#	X_bold_val_target=X_bold_val_target)
		#validation_loss = custom_training.adversarial_training(X_train_eeg, X_train_bold, tr_y, eeg_network, decoder_network, multi_modal_model, 
		#	epochs=40, 
		#	discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=current_learning_rate),
    	#	generator_optimizer=tf.keras.optimizers.Adam(learning_rate=current_learning_rate), 
		#	batch_size=current_batch_size, linear_combination=current_loss_coefficient,
		#	X_val_eeg=X_val_eeg,
		#	X_val_bold=X_val_bold,
		#	tv_y=tv_y,
		#	eeg_train=eeg_train, bold_train=bold_train, eeg_val=eeg_val, bold_val=bold_val,
		#	X_bold_train_target=X_bold_train_target,
    	#	X_bold_val_target=X_bold_val_target)
		validation_loss = custom_training.linear_combination_training(X_train_eeg, X_train_bold, tr_y, eeg_network, decoder_network, multi_modal_model, 
			epochs=5, 
			encoder_optimizer=tf.keras.optimizers.Adam(learning_rate=current_learning_rate),
			decoder_optimizer=tf.keras.optimizers.Adam(learning_rate=current_learning_rate),
			loss_function=losses_utils.get_reconstruction_log_cosine_voxel_loss,
			batch_size=current_batch_size, linear_combination=current_loss_coefficient,
			X_val_eeg=X_val_eeg,
			X_val_bold=X_val_bold,
			tv_y=tv_y,
			eeg_train=eeg_train, bold_train=bold_train, eeg_val=eeg_val, bold_val=bold_val,
			X_bold_train_target=X_bold_train_target,
    		X_bold_val_target=X_bold_val_target)
		#validation_loss = custom_training.dcca_training(X_train_eeg, X_train_bold, tr_y, eeg_network, decoder_network, multi_modal_model, 
		#	epochs=40, optimizer=tf.keras.optimizers.Adam(learning_rate=current_learning_rate),
		#	batch_size=current_batch_size, linear_combination=current_loss_coefficient, dcca_output=current_dcca_output,
		#	X_val_eeg=X_val_eeg,
		#	X_val_bold=X_val_bold,
		#	tv_y=tv_y)

		print("Model: " + model_name +
		' Train Intances: ' + str(len(X_train_bold)) + ' | Validation Instances: ' + str(len(X_val_bold)) +  ' | Validation Loss: ' + str(validation_loss))
		
		return validation_loss

	optimizer = GPyOpt.methods.BayesianOptimization(
	f=bayesian_optimization_function, domain=hyperparameters, model_type="GP_MCMC", acquisition_type="EI_MCMC")

	print("Started Optimization Process")
	optimizer.run_optimization(max_iter=1)


	#SAVE BEST MODELS
	#EEG network branch
	eeg_input_shape = (eeg_train.shape[1], eeg_train.shape[2], eeg_train.shape[3], 1)
	optimal_shape = (int(optimizer.x_opt[-1]),) + output_shape
	multi_modal_instance.save_eeg(multi_modal_instance.build_eeg(eeg_input_shape, optimal_shape,
																regularization=float(optimizer.x_opt[1]), 
																dropout=float(optimizer.x_opt[5])))

	#BOLD network branch
	bold_input_shape = (bold_train.shape[1], bold_train.shape[2], 1)
	multi_modal_instance.save_bold(multi_modal_instance.build_bold(bold_input_shape, optimal_shape,
																regularization=float(optimizer.x_opt[2]), 
																dropout=float(optimizer.x_opt[5])))

	#Decoder network branch
	multi_modal_instance.save_decoder(multi_modal_instance.build_decoder(optimal_shape, bold_input_shape,
																regularization=float(optimizer.x_opt[3]), 
																dropout=float(optimizer.x_opt[5])))

	print("Optimized Parameters: {0}".format(optimizer.x_opt))
	print("Optimized Validation Decoder Loss: {0}".format(optimizer.fx_opt))
	print("\n\n\n\n\n\n\n\n\n\n")

	return optimizer.x_opt[-1], optimizer.fx_opt