from multiprocessing import Process

import os

def process_setup_tensorflow(memory_limit, seed=42):
	from utils import tf_config

	tf_config.set_seed(seed=seed)
	tf_config.setup_tensorflow(device="GPU", memory_limit=memory_limit)

def launch_process(function, args):
	p = Process(target=function, args=args)
	p.daemon = True
	p.start()
	p.join()


def theta_latent_fmri():
	return [{'name': 'learning_rate', 'type': 'continuous',
			'domain': (1e-10, 1e-2)},
			{'name': 'weight_decay', 'type': 'continuous',
			'domain': (1e-10, 1e-1)},
			{'name': 'batch_size', 'type': 'discrete',
			'domain': (64,)},
			{'name': 'latent', 'type': 'discrete',
			'domain': (4,5,6,7,8,9,10,15,20)},
			{'name': 'channels', 'type': 'discrete',
			'domain': (2,4)},
			{'name': 'max_pool', 'type': 'discrete',
			'domain': (0,1)},
			{'name': 'batch_norm', 'type': 'discrete',
			'domain': (0,1)},
			{'name': 'skip', 'type': 'discrete',
			'domain': (0,1)},
			{'name': 'dropout', 'type': 'discrete',
			'domain': (0,1)},
			{'name': 'out_filter', 'type': 'discrete',
			'domain': (0,1,2)}]


def load_data_latent_fmri(dataset, n_individuals, n_individuals_train, n_volumes, interval_eeg, memory_limit):
	from utils import preprocess_data
	import tensorflow as tf

	process_setup_tensorflow(memory_limit)

	with tf.device('/CPU:0'):
		train_data, test_data = preprocess_data.dataset(dataset, n_individuals=n_individuals, 
												interval_eeg=interval_eeg, 
												ind_volume_fit=False,
												standardize_fmri=True,
												iqr=False,
												verbose=True)

	return train_data[1]

def load_data_eeg_fmri(dataset, n_individuals, n_volumes, interval_eeg, memory_limit, return_test=False, setup_tf=True):
	from utils import preprocess_data
	import tensorflow as tf

	if(setup_tf):
		process_setup_tensorflow(memory_limit)

	with tf.device('/CPU:0'):
		train_data, test_data = preprocess_data.dataset(dataset, n_individuals=n_individuals, 
												interval_eeg=interval_eeg, 
												ind_volume_fit=False,
												standardize_fmri=True,
												iqr=False,
												verbose=True)
	if(return_test):
		return train_data, test_data
	return train_data


def make_dir_batches(dataset, n_individuals, n_individuals_train, n_individuals_val, n_volumes, interval_eeg, memory_limit, batch_size, batch_path):
	import tensorflow as tf
	import os
	import numpy as np
	import shutil
	from pathlib import Path

	dataset = load_data_eeg_fmri(dataset, n_individuals, n_volumes, interval_eeg, memory_limit)

	eeg, fmri = dataset

	#partition in train and validation sets
	eeg_val = eeg[n_individuals_train*n_volumes:(n_individuals_train+n_individuals_val)*n_volumes]
	eeg_train = eeg[:n_individuals_train*n_volumes]

	fmri_val = fmri[n_individuals_train*n_volumes:(n_individuals_train+n_individuals_val)*n_volumes]
	fmri_train = fmri[:n_individuals_train*n_volumes]


	dataset = tf.data.Dataset.from_tensor_slices((eeg_train, fmri_train)).batch(batch_size)

	if(Path(batch_path).exists()):
		shutil.rmtree(batch_path)

	#save
	os.mkdir(batch_path)

	batch=1
	#write train batches
	for batch_x, batch_y in dataset.repeat(1):
		np.save(batch_path+"/batch_x_"+str(batch), batch_x.numpy())
		np.save(batch_path+"/batch_y_"+str(batch), batch_y.numpy())
		batch+=1


def load_batch(tensorflow, batch_path, batch, dtype):
	import numpy as np

	batch_x, batch_y = (np.load(batch_path+"/batch_x_"+str(batch)+".npy"),
						np.load(batch_path+"/batch_y_"+str(batch)+".npy"))
	return (tensorflow.convert_to_tensor(batch_x, dtype=dtype), 
			tensorflow.convert_to_tensor(batch_y, dtype=dtype))


def batch_prediction(shared_flattened_predictions, setup, batch_path, batch, epoch, network, na_path, batch_size, learning_rate, memory_limit, best_eeg, seed):
	#imports
	import tensorflow as tf
	from utils import train, losses_utils, state_utils, tf_config
	from layers.fourier_features import RandomFourierFeatures
	from models.eeg_to_fmri import EEG_to_fMRI, call
	from models.fmri_ae import fMRI_AE
	import pickle

	tf_config.setup_tensorflow(memory_limit=memory_limit, device="GPU")
	tf.random.set_seed(seed)

	#load batch
	eeg, fmri = load_batch(tf, batch_path, batch, tf.float32)

	#unroll hyperparameters
	theta = (0.002980911194116198, 0.0004396489214334123, (9, 9, 4), (1, 1, 1), 4, (7, 7, 7), 4, True, True, True, True, 3, 1)
	learning_rate=float(theta[0])
	weight_decay = float(theta[1])
	kernel_size = theta[2]
	stride_size = theta[3]
	batch_size=int(theta[4])
	latent_dimension=theta[5]
	n_channels=int(theta[6])
	max_pool=bool(theta[7])
	batch_norm=bool(theta[8])
	skip_connections=bool(theta[9])
	dropout=bool(theta[10])
	n_stacks=int(theta[11])
	outfilter=int(theta[12])
	local=True
	
	if(setup=="fmri"):
		with open(best_eeg, "rb") as f:
			na_specification_eeg = pickle.load(f)
		with open(na_path + "/na_specification_"+str(network+1), "rb") as f:
			na_specification_fmri = pickle.load(f)
	elif(setup=="eeg"):
		with open(na_path + "/na_specification_"+str(network+1), "rb") as f:
			na_specification_eeg = pickle.load(f)
	else:
		raise NotImplementedError

	#load or build model
	with tf.device('/CPU:0'):
		loss_fn = losses_utils.mse_cosine

		if(batch == 1 and epoch == 0):
			#TODO: correct me to load right model specification
			optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
			#kernels, strides = parse_conv.cnn_to_tuple(tf.keras.models.load_model(na_path + method + "/architecture_" + str(network+1), compile=False))
			model = EEG_to_fMRI(latent_dimension, eeg.shape[1:], na_specification_eeg, 4, weight_decay=0.000, skip_connections=True,
											fourier_features=True, random_fourier=True, topographical_attention=False,
											batch_norm=True, local=True, seed=None, fmri_args = (latent_dimension, fmri.shape[1:], 
											kernel_size, stride_size, n_channels, max_pool, batch_norm, weight_decay, skip_connections,
											n_stacks, True, False, outfilter, dropout, None, False, na_specification_fmri))
			model.build(eeg.shape, fmri.shape)
			model.compile(optimizer=optimizer)
		else:
			#load model and optimizer at previous state
			model = tf.keras.models.load_model(na_path + "/architecture_" + str(network) + "_training", compile=True, 
										custom_objects={"EEG_to_fMRI": EEG_to_fMRI,
														"fMRI_AE": fMRI_AE,
														"RandomFourierFeatures": RandomFourierFeatures})
			state_utils.setup_state(tf, model.optimizer, na_path  + "/architecture_" + str(network) + "_training/opt_config", 
												na_path + "/architecture_" + str(network) + "_training/gen_config")

	loss, batch_preds = train.train_step(model, (eeg, fmri), model.optimizer, loss_fn, u_architecture=True, return_logits=True, call_fn=call)
	loss=loss.numpy()

	flattened_batch_preds=batch_preds.numpy().flatten()
	for i in range(flattened_batch_preds.shape[0]):
		shared_flattened_predictions[i] = flattened_batch_preds[i]

	print("NA", network, " at epoch", epoch+1, " and batch", batch, "with loss:", loss, end="\n")
	
	#save model
	model.save(na_path + "/architecture_" + str(network) + "_training", save_format="tf", save_traces=False)
	#save state
	state_utils.save_state(tf, model.optimizer, na_path + "/architecture_" + str(network) + "_training/opt_config", 
							na_path + "/architecture_" + str(network) + "_training/gen_config")

def continuous_training(o_predictions, batch_path, batch, learning_rate, epoch, na_path, gpu_mem, seed):
	import tensorflow as tf
	import numpy as np
	from utils import state_utils, losses_utils, tf_config, train
	from models import softmax
	
	tf_config.setup_tensorflow(memory_limit=gpu_mem, device="GPU")
	tf.random.set_seed(seed)

	loss_fn = losses_utils.mse

	if(batch==1 and epoch == 0):
		opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
		model=softmax.Softmax((o_predictions.shape[0],))
		model.build(input_shape=o_predictions.shape)
		model.compile(optimizer=opt)
	else:
		model=tf.keras.models.load_model(na_path + "/softmax_training", compile=True)
		#opt = state_utils.setup_state(tf, opt, na_path + method + "/softmax_training/opt_config", 
		state_utils.setup_state(tf, model.optimizer, na_path + "/softmax_training/opt_config", 
							na_path + "/softmax_training/gen_config")
	
	#load batch
	_, fmri = load_batch(tf, batch_path, batch, tf.float32)

	#training step to update weights with batch
	loss = train.train_step(model, (o_predictions,fmri), model.optimizer, loss_fn).numpy()
	
	print("Softmax epoch loss: ", loss, end="\n\n\n")
	print(model.trainable_variables[0].numpy())

	model.save(na_path + "/softmax_training", save_format="tf")
	#save state
	state_utils.save_state(tf, model.optimizer, na_path + "/softmax_training/opt_config", 
							na_path + "/softmax_training/gen_config")



def save_weights(epoch, na_path, save_weights_path): 
	import tensorflow as tf
	import numpy as np

	model=tf.keras.models.load_model(na_path + "/softmax_training", compile=False)

	np.save(save_weights_path+"/epoch_"+str(epoch), model.trainable_variables[0].numpy())


def cross_validation_latent_fmri(score, learning_rate, weight_decay, 
						kernel_size, stride_size,
						batch_size, latent_dimension, n_channels, 
						max_pool, batch_norm, skip_connections, dropout,
						n_stacks, outfilter, dataset, n_individuals, 
						n_individuals_train, n_volumes, 
						interval_eeg, memory_limit):

	from utils import train
	from models import fmri_ae
	from sklearn.model_selection import KFold
	import tensorflow as tf

	data = load_data_latent_fmri(dataset, n_individuals, n_individuals_train, n_volumes, interval_eeg, memory_limit)
	n_folds = 5

	for train_idx, val_idx in KFold(n_folds).split(data):
		with tf.device('/CPU:0'):
			x_train = data[train_idx]
			x_val = data[val_idx]
			
			#build model
			model = fmri_ae.fMRI_AE(latent_dimension, x_train.shape[1:], kernel_size, stride_size, n_channels,
								maxpool=max_pool, batch_norm=batch_norm, weight_decay=weight_decay, skip_connections=skip_connections,
								n_stacks=n_stacks, local=True, local_attention=False, outfilter=outfilter, dropout=dropout)
			model.build(input_shape=x_train.shape)

			#train model
			optimizer = tf.keras.optimizers.Adam(learning_rate)
			loss_fn = tf.keras.losses.MSE#replace

			train_set = tf.data.Dataset.from_tensor_slices((x_train, x_train)).batch(batch_size)
			dev_set = tf.data.Dataset.from_tensor_slices((x_val, x_val)).batch(1)
		
		train.train(train_set, model, optimizer, 
								loss_fn, epochs=10, 
								val_set=None, verbose=True)

		#evaluate
		score.value += train.evaluate(dev_set, model, loss_fn)

	score.value = (score.value-1.0)/n_folds


def cross_validation_eeg_fmri(score, fourier_features, random_fourier,
						topographical_attention, conditional_attention_style, 
						na_specification_eeg, na_specification_fmri, 
						learning_rate, weight_decay, 
						kernel_size, stride_size,
						batch_size, latent_dimension, n_channels, 
						max_pool, batch_norm, skip_connections, dropout,
						n_stacks, outfilter, dataset, n_individuals, 
						n_individuals_train, n_volumes, 
						interval_eeg, memory_limit):

	from utils import train, losses_utils
	from models import eeg_to_fmri
	from sklearn.model_selection import KFold
	import tensorflow as tf

	data = load_data_eeg_fmri(dataset, n_individuals, n_volumes, interval_eeg, memory_limit, return_test=False, setup_tf=True)
	n_folds = 5

	for train_idx, val_idx in KFold(n_folds).split(data[0]):
		with tf.device('/CPU:0'):
			x_train, y_train = (data[0][train_idx], data[1][train_idx])
			x_val, y_val = (data[0][val_idx], data[1][val_idx])
			
			model = eeg_to_fmri.EEG_to_fMRI(latent_dimension, x_train.shape[1:], na_specification_eeg, n_channels,
								weight_decay=weight_decay, skip_connections=True,
								batch_norm=True, #dropout=False,
								fourier_features=fourier_features,
								random_fourier=random_fourier,
								topographical_attention=topographical_attention,
								conditional_attention_style=conditional_attention_style,
								inverse_DFT=False, DFT=False,
								variational_iDFT=False,
								variational_coefs=(15,15,15),
								low_resolution_decoder=False,
								local=True, seed=None, 
								fmri_args = (latent_dimension, y_train.shape[1:], 
								kernel_size, stride_size, n_channels, 
								max_pool, batch_norm, weight_decay, skip_connections,
								n_stacks, True, False, outfilter, dropout, None, False, na_specification_fmri))
			
			model.build(x_train[0].shape, x_train[1].shape)
			
			optimizer = tf.keras.optimizers.Adam(learning_rate)
			loss_fn = losses_utils.mae_cosine

			train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
			dev_set = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(1)


		train.train(train_set, model, optimizer, 
								loss_fn, epochs=10, 
								val_set=None, verbose=True)

		#evaluate
		score.value += train.evaluate(dev_set, model, loss_fn)

	score.value = (score.value-1.0)/n_folds


def train_synthesis(dataset, epochs, save_path, gpu_mem, seed):
	#imports
	import tensorflow as tf

	from utils import data_utils, preprocess_data, tf_config, train, losses_utils

	from models import eeg_to_fmri

	interval_eeg=10
	tf_config.set_seed(seed=seed)#02 20
	tf_config.setup_tensorflow(device="GPU", memory_limit=gpu_mem)

	from pathlib import Path

	import numpy as np

	import pickle


	raw_eeg=False

	theta = (0.002980911194116198, 0.0004396489214334123, (9, 9, 4), (1, 1, 1), 4, (7, 7, 7), 4, True, True, True, True, 3, 1)
	#unroll hyperparameters
	learning_rate=float(theta[0])
	weight_decay = float(theta[1])
	kernel_size = theta[2]
	stride_size = theta[3]
	batch_size=int(theta[4])
	latent_dimension=theta[5]
	n_channels=int(theta[6])
	max_pool=bool(theta[7])
	batch_norm=bool(theta[8])
	skip_connections=bool(theta[9])
	dropout=bool(theta[10])
	n_stacks=int(theta[11])
	outfilter=int(theta[12])
	local=True
	with open(str(Path.home())+"/eeg_to_fmri/na_models_eeg/na_specification_2", "rb") as f:
		na_specification_eeg = pickle.load(f)
	with open(str(Path.home())+"/eeg_to_fmri/na_models_fmri/na_specification_2", "rb") as f:
		na_specification_fmri = pickle.load(f)

	with tf.device('/CPU:0'):
		
		train_data, _ = preprocess_data.dataset(dataset, 
														n_individuals=getattr(data_utils, "n_individuals_"+dataset),
														interval_eeg=interval_eeg, 
														ind_volume_fit=False,
														standardize_fmri=True,
														iqr=False,
														verbose=True)
		eeg_train, fmri_train = train_data
		
		model = eeg_to_fmri.EEG_to_fMRI(latent_dimension, eeg_train.shape[1:], na_specification_eeg, n_channels,
							weight_decay=weight_decay, skip_connections=True,
							batch_norm=True, #dropout=False,
							fourier_features=True,
							random_fourier=True,
							topographical_attention=True,
							conditional_attention_style=True,
							conditional_attention_style_prior=False,
							inverse_DFT=False, DFT=False,
							variational_iDFT=False,
							variational_coefs=(15,15,15),
							low_resolution_decoder=False,
							local=True, seed=None, 
							fmri_args = (latent_dimension, fmri_train.shape[1:], 
							kernel_size, stride_size, n_channels, 
							max_pool, batch_norm, weight_decay, skip_connections,
							n_stacks, True, False, outfilter, dropout, None, False, na_specification_fmri))
		model.build(eeg_train.shape, fmri_train.shape)
		optimizer = tf.keras.optimizers.Adam(learning_rate)
		loss_fn = losses_utils.mae_cosine
		train_set = tf.data.Dataset.from_tensor_slices((eeg_train, fmri_train)).batch(batch_size)

	print("I: Starting pretraining of synthesis network")

	loss_history = train.train(train_set, model, optimizer, 
							loss_fn, epochs=epochs, 
							u_architecture=True,
							val_set=None, verbose=True, verbose_batch=False)[0]

	print("I: Saving synthesis network at", save_path)

	model.save(save_path, save_format="tf", save_traces=False)


def create_labels(view, dataset, path):

	import numpy as np

	y_pred = np.empty((0,), dtype="float32")
	y_true = np.empty((0,), dtype="float32")

	np.save(path+view+"_y_pred.npy", y_pred, allow_pickle=True)
	np.save(path+view+"_y_true.npy", y_true, allow_pickle=True)

def append_labels(view, path, y_true, y_pred):
	import numpy as np
	np.save(path+view+"_y_pred.npy",np.append(np.load(path+view+"_y_pred.npy", allow_pickle=True), y_pred), allow_pickle=True)
	np.save(path+view+"_y_true.npy",np.append(np.load(path+view+"_y_true.npy", allow_pickle=True), y_true), allow_pickle=True)


def setup_data_loocv(view, dataset, epochs, learning_rate, batch_size, gpu_mem, seed, save_explainability, path_network, path_labels):

	from utils import preprocess_data

	from multiprocessing import Manager

	launch_process(load_data_loocv,
					(view, dataset, path_labels))

	dataset_clf_wrapper = preprocess_data.Dataset_CLF_CV(dataset, standardize_eeg=True, load=False, load_path=path_labels)

	for i in range(dataset_clf_wrapper.n_individuals):
		reg_constants = Manager().Array('d', range(2))
		#CV hyperparameter l1 and l2 reg constants
		cv_opt(reg_constants, i, view, dataset, learning_rate, batch_size, epochs, gpu_mem, seed, path_labels, path_network)
		#validate
		launch_process(loocv,
					(i, view, dataset, epochs, learning_rate, batch_size, gpu_mem, seed, save_explainability, path_network, path_labels))

def load_data_loocv(view, dataset, path_labels):
	from utils import preprocess_data
	
	raw_eeg=False
	if(view=="raw"):
		raw_eeg=True

	dataset_clf_wrapper = preprocess_data.Dataset_CLF_CV(dataset,
														eeg_limit=True, raw_eeg=raw_eeg,
														eeg_f_limit=135, standardize_eeg=True, 
														load=True, load_path=None)

	dataset_clf_wrapper.save(path_labels)

def predict(test_set, model):
	import numpy as np
	import tensorflow as tf

	hits = np.empty((0,))
	y_true = np.empty((0,))
	y_pred = np.empty((0,))

	for x,y in test_set.repeat(1):
		
		if(tf.math.reduce_all(tf.math.equal(tf.math.argmax(y, axis=-1), tf.math.argmax(model(x), axis=-1))).numpy()):
			hits = np.append(hits, 1.0)
		else:
			hits = np.append(hits, 0.0)
			
		if(y.numpy()[0,1]==1.0):
			y_true=np.append(y_true,1.0)
		else:
			y_true=np.append(y_true,0.0)
		
		y_pred=np.append(y_pred, tf.nn.softmax(model(x), axis=-1).numpy()[0,1])
	
	return hits, y_true, y_pred


def views(model, test_set, y):
	from utils import fmri_utils
	import tensorflow as tf
	import numpy as np

	dev_views = np.empty((0,)+getattr(fmri_utils, "fmri_shape_01")+(1,))
	for x, _ in test_set.repeat(1):
		dev_views = np.append(dev_views, model.view(x), axis=0)
	
	return tf.data.Dataset.from_tensor_slices((dev_views,y)).batch(1)


def cv_opt(reg_constants, fold_loocv, view, dataset, learning_rate, batch_size, epochs, gpu_mem, seed, path_labels, path_network):
	import GPyOpt

	def optimize_wrapper(theta):
		from multiprocessing import Manager

		l1_reg, l2_reg = (float(theta[:,0]), float(theta[:,1]))
		value = Manager().Array('d', range(1))

		launch_process(optimize_elastic, (value, (l1_reg, l2_reg),))

		print(value[0])
		return value[0]

	def optimize_elastic(value, theta):

		from utils import preprocess_data, tf_config, train

		from models import eeg_to_fmri, classifiers

		import tensorflow as tf

		import numpy as np

		l1_reg, l2_reg = (theta)

		tf_config.set_seed(seed=seed)
		tf_config.setup_tensorflow(device="GPU", memory_limit=gpu_mem)

		dataset_clf_wrapper = preprocess_data.Dataset_CLF_CV(dataset, standardize_eeg=True, load=False, load_path=path_labels)
		train_data, test_data = dataset_clf_wrapper.split(fold_loocv)
		dataset_clf_wrapper.X = train_data[0]
		dataset_clf_wrapper.y = train_data[1]
		dataset_clf_wrapper.set_folds(5)

		score = 0.0
		for fold in range(5):
			train_data, test_data = dataset_clf_wrapper.split(fold)
			X_train, y_train=train_data
			X_test, y_test=test_data
			with tf.device('/CPU:0'):
				optimizer = tf.keras.optimizers.Adam(learning_rate)
				loss_fn=tf.keras.losses.CategoricalCrossentropy(from_logits=True)

				train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
				test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(1)
				
				if(view=="fmri"):
					linearCLF = classifiers.view_EEG_classifier(tf.keras.models.load_model(path_network,custom_objects=eeg_to_fmri.custom_objects), 
																X_train.shape[1:],
																regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg))
				else:
					linearCLF = classifiers.LinearClassifier(regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg))
				linearCLF.build(X_train.shape)

			train.train(train_set, linearCLF, optimizer, loss_fn, epochs=epochs, val_set=None, u_architecture=False, verbose=True, verbose_batch=False)

			#evaluate
			score+=loss_fn(y_test[:,:,0], linearCLF(X_test))

			print(train_data[0].shape, test_data[0].shape)

		value[0] = score

	hyperparameters = [{'name': 'l1', 'type': 'continuous','domain': (1e-10, 1.)}, {'name': 'l2', 'type': 'continuous', 'domain': (1e-10, 1.)}]


	optimizer = GPyOpt.methods.BayesianOptimization(f=optimize_wrapper, 
													domain=hyperparameters, 
													model_type="GP_MCMC", 
													acquisition_type="EI_MCMC")
	optimizer.run_optimization(max_iter=100)

	print("Best value: ", optimizer.fx_opt)
	print("Best hyperparameters: \n", optimizer.x_opt)

	raise NotImplementedError




def loocv(fold, view, dataset, epochs, learning_rate, batch_size, gpu_mem, seed, save_explainability, path_network, path_labels):
	
	from utils import preprocess_data, tf_config, train, lrp

	from models import eeg_to_fmri, classifiers

	import tensorflow as tf

	import numpy as np

	tf_config.set_seed(seed=seed)
	tf_config.setup_tensorflow(device="GPU", memory_limit=gpu_mem)

	dataset_clf_wrapper = preprocess_data.Dataset_CLF_CV(dataset, standardize_eeg=True, load=False, load_path=path_labels)

	train_data, test_data = dataset_clf_wrapper.split(fold)
	X_train, y_train = train_data
	X_test, y_test = test_data

	with tf.device('/CPU:0'):
		optimizer = tf.keras.optimizers.Adam(learning_rate)
		loss_fn=tf.keras.losses.CategoricalCrossentropy(from_logits=True)

		train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
		test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(1)
		

		if(view=="fmri"):
			linearCLF = classifiers.view_EEG_classifier(tf.keras.models.load_model(path_network,custom_objects=eeg_to_fmri.custom_objects), 
														X_train.shape[1:])
		else:
			linearCLF = classifiers.LinearClassifier()
		linearCLF.build(X_train.shape)

	#train classifier
	train.train(train_set, linearCLF, optimizer, loss_fn, epochs=epochs, val_set=None, u_architecture=False, verbose=True, verbose_batch=False)

	#get predictionsf
	hits, y_true, y_pred = predict(test_set, linearCLF)
	#save predictions
	append_labels(view, path_labels, y_true, y_pred)

	print("Finished fold", fold)
	if(save_explainability and view=="fmri"):
		#explaing features
		#explain to fMRI view
		explainer=lrp.LRP(linearCLF.clf)
		R=lrp.explain(explainer, views(linearCLF, test_set, y_test), verbose=True)
		#explain to EEG channels
		explainer=lrp.LRP_EEG(linearCLF.view)
		attention_scores=lrp.explain(explainer, test_set, eeg=True, eeg_attention=True, fmri=False, verbose=True)
		#save explainability
		if(fold==0):
			np.save(path_labels+"R.npy", R, allow_pickle=True)
			np.save(path_labels+"attention_scores.npy", attention_scores, allow_pickle=True)
		else:
			np.save(path_labels+"R.npy", np.append(np.load(path_labels+"R.npy", allow_pickle=True), R, axis=0), allow_pickle=True)
			np.save(path_labels+"attention_scores.npy", np.append(np.load(path_labels+"attention_scores.npy", allow_pickle=True), attention_scores, axis=0), allow_pickle=True)


def compute_acc_metrics(view, path):

	import numpy as np

	y_pred = np.load(path+view+"_y_pred.npy", allow_pickle=True)
	y_true = np.load(path+view+"_y_true.npy", allow_pickle=True)

	#true positive
	tp = len(np.where(y_pred[np.where(y_true==1.0)] >= 0.5)[0])
	#true negative
	tn = len(np.where(y_pred[np.where(y_true==0.0)] < 0.5)[0])
	#false positive
	fp = len(np.where(y_pred[np.where(y_true==0.0)] >= 0.5)[0])
	#false negative
	fn = len(np.where(y_pred[np.where(y_true==1.0)] < 0.5)[0])

	print("Accuracy:", (tn+tp)/(tn+tp+fn+fp))
	print("Sensitivity:", (tp)/(tp+fn))
	print("Specificity:", (tn)/(tn+fp))
	print("F1-score:", (tp)/(tp+0.5*(fp+fn)))