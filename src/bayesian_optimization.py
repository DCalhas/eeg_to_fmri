import GPyOpt

import decoder
import deep_cross_corr

from sklearn.model_selection import train_test_split

import tensorflow.compat.v1 as tf

import numpy as np

def main():

	hyperparameters = [{'name': 'learning_rate', 'type': 'continuous',
	'domain': (10e-6, 10e-3)},
	{'name': 'l1_penalization_eeg_1', 'type': 'continuous',
	'domain': (10e-4, 10e-1)},
	{'name': 'l1_penalization_eeg_2', 'type': 'continuous',
	'domain': (10e-4, 10e-1)},
	{'name': 'l1_penalization_eeg_3', 'type': 'continuous',
	'domain': (10e-4, 10e-1)},
	{'name': 'l1_penalization_bold_1', 'type': 'continuous',
	'domain': (10e-4, 10e-1)},
	{'name': 'l1_penalization_bold_2', 'type': 'continuous',
	'domain': (10e-4, 10e-1)},
	{'name': 'l1_penalization_decoder_1', 'type': 'continuous',
	'domain': (10e-4, 10e-1)},
	{'name': 'l1_penalization_decoder_2', 'type': 'continuous',
	'domain': (10e-4, 10e-1)},
	{'name': 'loss_coefficient', 'type': 'continuous',
	'domain': (0.0, 1.0)}]

	eeg_train, bold_train, eeg_test, bold_test = decoder.load_data(list(range(1)), list(range(1, 2)))
		
	X_train_eeg, X_train_bold, tr_y = deep_cross_corr.create_eeg_bold_pairs(eeg_train, bold_train)
	X_test_eeg, X_test_bold, te_y = deep_cross_corr.create_eeg_bold_pairs(eeg_test, bold_test)

	#convert to tensors, for the networks to accept it as input
	X_train_eeg = tf.convert_to_tensor(X_train_eeg, dtype=np.float32)
	X_train_bold = tf.convert_to_tensor(X_train_bold, dtype=np.float32)
	tr_y = tf.convert_to_tensor(tr_y, dtype=np.float32)

	def bayesian_optimization_function(x):
		current_learning_rate = float(x[:, 0])
		current_l1_penalization_eeg_1 = float(x[:, 1])
		current_l1_penalization_eeg_2 = float(x[:, 2])
		current_l1_penalization_eeg_3 = float(x[:, 3])
		current_l1_penalization_bold_1 = float(x[:, 4])
		current_l1_penalization_bold_2 = float(x[:, 5])
		current_l1_penalization_decoder_1 = float(x[:, 6])
		current_l1_penalization_decoder_2 = float(x[:, 7])
		current_loss_coefficient = float(x[:, 8])


		model_name = 'siamese_net_lr_' + str(current_learning_rate)


		#####################################################################################################3
		#
		#										DEFINING ARCHITECTURES
		#
		######################################################################################################
		#EEG network branch
		eeg_input_shape = (eeg_train.shape[1], eeg_train.shape[2], eeg_train.shape[3], 1)
		kernel_size = (eeg_train.shape[1], eeg_train.shape[2], 1)
		eeg_network = deep_cross_corr.eeg_network(eeg_input_shape, kernel_size)
		print(eeg_network.summary())

		#BOLD network branch
		bold_input_shape = (bold_train.shape[1], bold_train.shape[2], 1)
		kernel_size = (bold_train.shape[1], 1)
		bold_network = deep_cross_corr.bold_network(bold_input_shape, kernel_size)
		print(bold_network.summary())

		#Decoder network branch
		shared_eeg_train = eeg_network.predict(eeg_train)

		input_shape = (None, shared_eeg_train.shape[1], shared_eeg_train.shape[2], 1)

		decoder_model = decoder.decoding_network(input_shape)
		print(decoder_model.summary())

		#Joining EEG and BOLD branches
		multi_modal_model = decoder.multi_modal_network(eeg_input_shape, bold_input_shape, eeg_network, bold_network)

		#need to load the models and give it as parameters to the run_training function
		print("Starting training")
		tf.keras.backend.clear_session()
		decoder.run_training(X_train_eeg, X_train_bold, tr_y, eeg_network, decoder_model, multi_modal_model, 
			epochs=10, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
			batch_size=128)

		if validation_accuracy == 0:
			evaluation_accuracy = 0
		else:
			# Load the weights with best validation accuracy
			decoder.model.load_weights('models/' + model_name + '.h5')
		print("Model: " + model_name +
		' | Accuracy: ' + str(evaluation_accuracy))
		K.clear_session()
		return 1 - evaluation_accuracy

	optimizer = GPyOpt.methods.BayesianOptimization(
	f=bayesian_optimization_function, domain=hyperparameters, model_type="GP_MCMC", acquisition_type="EI_MCMC")

	print("Started Optimization Process")
	optimizer.run_optimization(max_iter=100)

	print("optimized parameters: {0}".format(optimizer.x_opt))
	print("optimized eval_accuracy: {0}".format(1 - optimizer.fx_opt))


if __name__ == "__main__":
	main()