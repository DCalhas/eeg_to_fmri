from multiprocessing import Process

import os

def process_setup_tensorflow(memory_limit):
	from utils import tf_config

	tf_config.set_seed(seed=42)
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

def load_data_eeg_fmri(dataset, n_individuals, n_volumes, interval_eeg, memory_limit):
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


def batch_prediction(shared_flattened_predictions, batch_path, batch, epoch, network, na_path, batch_size, learning_rate, memory_limit, seed):
	#imports
	import tensorflow as tf
	from utils import train, losses_utils
	from models import fmri_ae, eeg_to_fmri

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

	#alter this
	na_specification = ([(10,20,2),(10,20,2)], 
						[(1,1,1),(1,1,1)],
						True,
						(2,2,1),
						(1,1,1))

	#load or build model
	with tf.device('/CPU:0'):
		
		
		
		loss_fn = losses_utils.mse_cosine

		if(batch == 1 and epoch == 0):
			optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
			#kernels, strides = parse_conv.cnn_to_tuple(tf.keras.models.load_model(na_path + method + "/architecture_" + str(network+1), compile=False))
			model = eeg_to_fmri.EEG_to_fMRI(latent_dimension, eeg.shape[1:], na_specification, 4, weight_decay=0.000, skip_connections=True,
											batch_norm=True, local=True, seed=None, fmri_args = (latent_dimension, fmri.shape[1:], 
											kernel_size, stride_size, n_channels, max_pool, batch_norm, weight_decay, skip_connections,
											n_stacks, True, False, outfilter, dropout))
			model.build(eeg.shape, fmri.shape)
			model.compile(optimizer=optimizer)
		else:
			#optimizer = state_utils.setup_state(tf, optimizer, na_path + method + "/architecture_" + str(network+1) + "_training/opt_config", 
			model = tf.keras.models.load_model(na_path + "/architecture_" + str(network+1) + "_training", compile=True)
			#setup_state(tf, model.optimizer, na_path  + "/architecture_" + str(network+1) + "_training/opt_config", 
			#									na_path + "/architecture_" + str(network+1) + "_training/gen_config")

	loss, batch_preds = train.train_step(model, [eeg, fmri], model.optimizer, loss_fn, u_architecture=True, return_logits=True)
	loss=loss.numpy()

	flattened_batch_preds=batch_preds.numpy().flatten()
	for i in range(flattened_batch_preds.shape[0]):
		shared_flattened_predictions[i] = flattened_batch_preds[i]

	return
	#save model
	#model.save(na_path + "/architecture_" + str(network+1) + "_training", save_format="tf")
	#save state
	#state_utils.save_state(tf, model.optimizer, na_path + "/architecture_" + str(network+1) + "_training/opt_config", 
	#						na_path + "/architecture_" + str(network+1) + "_training/gen_config")

	raise NotImplementedError




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