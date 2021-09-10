from multiprocessing import Process


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
			'domain': (16,)},
			{'name': 'latent', 'type': 'discrete',
			'domain': (4,5,6,7,8,9,10,15,20)},
			{'name': 'channels', 'type': 'discrete',
			'domain': (2,4,8,16)},
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